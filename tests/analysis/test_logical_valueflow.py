"""Source read versions and independent kills, without storage scheduling.

The finite oracle interprets retained native byte accesses in logical order.
Native initialization follows the independent interpreter's actual stage
ordinal/primitive semantics, rather than the production read-condition field.
"""

from itertools import product

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.logical_valueflow import build_logical_valueflow
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_regions import (
    AccessOccurrence,
    BoundaryPort,
    DependencyIssueKind,
    LogicalWriteKind,
    LoopOrdinalGuard,
)
from ascendflow.dialects.physical import KernelOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Actor
from tests.analysis.test_conditional_accesses import _retarget, _structure
from tests.fixtures.nested import shared_l0c_qk_pv
from tests.models.structured_domains import expression_value, guard_active, occurrence_key
from tests.models.structured_events import _Interpret


def flow_for(source):
    return build_logical_valueflow(
        collect_native_memory(
            lower_native_structure(
                N.from_source(source) if isinstance(source, str) else source
            ).module
        )
    )


def original_read_versions(flow, parameters):
    memory = flow.memory
    root = memory.module.body.block.first_op
    assert isinstance(root, KernelOp)
    actor = Actor.AIC if root.core.data == "aic" else Actor.AIV0
    logical = _Interpret(memory, parameters, physical=False, maximum=2000)
    logical.block(root.body.block, {}, 0, 0, actor, ())
    writers, bindings, footprints = {}, set(), {}
    for command in logical.commands:
        point = command.occurrence
        occurrence = point.operation, point.actor, point.coordinates
        touched = {}
        for span in command.spans:
            for byte in range(span.start, span.end):
                touched.setdefault((span.root, span.actor, byte), set()).update(span.mode)
        for cell, mode in touched.items():
            if "R" in mode:
                bindings.add((cell, writers.get(cell), occurrence))
                footprints.setdefault(occurrence, set()).add(cell)
        for cell, mode in touched.items():
            if "W" in mode:
                writers[cell] = occurrence
    return bindings, footprints


def bound_read_versions(flow, parameters, footprints):
    memory, bindings = flow.memory, set()
    uniform = parameters | {"$group": 0, "$sid": 0}
    for read in flow.reads:
        assert read.kind is HazardKind.RAW
        domains = (
            read.domains
            + (() if isinstance(read.source, BoundaryPort) else read.source.families)
            + read.consumer.families
        )
        loops = [
            loop
            for loop in memory.control.loops.values()
            if any(domain.loop is loop.operation for domain in domains)
        ]

        def visit(index, context, *, read=read, loops=loops, domains=domains):
            if index == len(loops):
                if any(
                    not (
                        guard.evaluate(context)
                        if isinstance(guard, LoopOrdinalGuard)
                        else guard_active(guard, context, memory)
                    )
                    for guard in read.guards
                ):
                    return
                source = (
                    None
                    if isinstance(read.source, BoundaryPort)
                    else occurrence_key(read.source, context, memory)
                )
                consumer = occurrence_key(read.consumer, context, memory)
                origin = read.cell.origin.evaluate(context)
                touched = set()
                for row, byte in product(
                    range(read.cell.rows), range(read.cell.start, read.cell.stop)
                ):
                    cell = (
                        read.cell.root.owner,
                        read.cell.private,
                        origin + row * read.cell.row_stride + byte,
                    )
                    if cell in footprints.get(consumer, ()):
                        touched.add(cell)
                        bindings.add((cell, source, consumer))
                assert touched, "value binding invents a read outside its actual native footprint"
                return
            loop = loops[index]
            current = [domain for domain in domains if domain.loop is loop.operation]
            for ordinal in range(max(domain.count.evaluate(context) for domain in current)):
                child = context | {loop.ordinal.args[0]: ordinal}
                if all(
                    max(domain.first, domain.distance) <= ordinal < domain.count.evaluate(child)
                    and ordinal % domain.period == expression_value(domain.residue, child)
                    for domain in current
                ):
                    visit(index + 1, child)

        visit(0, uniform)
    return bindings


def assert_read_versions(flow, parameters):
    expected, footprints = original_read_versions(flow, parameters)
    observed = bound_read_versions(flow, parameters, footprints)
    if observed != expected:
        pytest.fail(
            f"read version mismatch: extra={len(observed - expected)}, missing={len(expected - observed)}"
        )
    unique = {}
    for cell, source, consumer in observed:
        prior = unique.setdefault((cell, consumer), source)
        assert prior == source


@pytest.fixture(scope="module")
def scratch():
    return flow_for("""
    def scratch(X:N.Tensor((n,64),N.f32),A:N.Tensor((n,64),N.f32,access="write"),
                B:N.Tensor((n,64),N.f32,access="write"),n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for g in N.Pipelined(n,stages=2):
                with N.Stage(0):
                    N.copy(X[g,:],x)
                    N.copy(x,A[g,:])
                with N.Stage(1):
                    N.fill(x,1.0)
                    N.copy(x,B[g,:])
    """)


@pytest.mark.parametrize("count", (0, 1, 3))
def test_two_stage_scratch_binds_each_read_without_source_storage_edges(scratch, count):
    scratch.require_closed()
    assert all(read.kind is HazardKind.RAW for read in scratch.reads)
    assert all(write.kind is LogicalWriteKind.KILL for write in scratch.writes)
    assert scratch.connections
    assert all(connection.left.cell == connection.right.cell for connection in scratch.connections)
    assert_read_versions(scratch, {"n": count})


@pytest.fixture(scope="module")
def shared():
    return flow_for(shared_l0c_qk_pv)


@pytest.mark.parametrize("count", (0, 1, 2))
def test_shared_qk_pv_first_clear_and_recurrence_have_exact_read_versions(shared, count):
    shared.require_closed()
    c0_reads = [read for read in shared.reads if read.cell.root.type.bank.data == "L0C"]
    assert all(isinstance(read.source, AccessOccurrence) for read in c0_reads)
    writes = [write for write in shared.writes if write.cell.root.type.bank.data == "L0C"]
    assert {write.kind for write in writes} == {LogicalWriteKind.KILL, LogicalWriteKind.UPDATE}
    updates = [write for write in writes if write.kind is LogicalWriteKind.UPDATE]
    assert updates and all(
        any(isinstance(guard, LoopOrdinalGuard) for guard in write.guards) for write in updates
    )
    recurrences = [read for read in c0_reads if read.source.operation is read.consumer.operation]
    assert recurrences and all(read.distance == 1 for read in recurrences)
    assert_read_versions(shared, {"n": count})


def test_boundary_inputs_and_all_readers_survive_closed_read_only_child():
    flow = flow_for("""
    def readers(X:N.Tensor((64,),N.f32),n:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            N.copy(X[:],x)
            for g in N.Pipelined(n,stages=1):
                N.mul(x,2.0,out=y)
            N.fill(x,1.0)
    """)
    flow.require_closed()
    assert all(isinstance(entry.source, BoundaryPort) for entry in flow.entries)
    assert all(any(read is entry for read in flow.reads) for entry in flow.entries)
    assert any(read.consumer.families for read in flow.reads)
    for count in (0, 1, 4):
        assert_read_versions(flow, {"n": count})


def test_unsupported_periodic_reset_remains_an_explicit_closure_issue():
    structure = _structure()
    _retarget(structure, 2)
    flow = build_logical_valueflow(collect_native_memory(structure.module))
    assert not flow.closed
    assert any(issue.kind is DependencyIssueKind.PARENT_ACTIVATION for issue in flow.unresolved)


def test_every_ordinal_reset_is_a_kill_without_a_fictitious_input():
    structure = _structure()
    _retarget(structure, 1)
    flow = build_logical_valueflow(collect_native_memory(structure.module))
    flow.require_closed()
    assert all(write.kind is LogicalWriteKind.KILL for write in flow.writes)
    assert all(
        write.accesses and all(access.mode.writes for access in write.accesses)
        for write in flow.writes
    )
    assert not any(read.cell.root.type.bank.data == "L0C" for read in flow.reads)
    assert_read_versions(flow, {"n": 3})


@pytest.fixture(scope="module")
def runtime_recurrence():
    return build_logical_valueflow(collect_native_memory(_structure().module))


@pytest.mark.parametrize("count", (0, 1, 2, 5))
def test_runtime_first_clear_keeps_empty_single_and_recurrent_versions(runtime_recurrence, count):
    runtime_recurrence.require_closed()
    assert not any(read.cell.root.type.bank.data == "L0C" for read in runtime_recurrence.entries)
    assert_read_versions(runtime_recurrence, {"n": count})


def test_rmw_without_reset_retains_its_boundary_value_and_update_chain():
    structure = _structure("N.matmul(aa,bb,accum=c0,initialize=False)")
    flow = build_logical_valueflow(collect_native_memory(structure.module))
    flow.require_closed()
    assert any(read.cell.root.type.bank.data == "L0C" for read in flow.entries)
    writes = [write for write in flow.writes if write.cell.root.type.bank.data == "L0C"]
    assert writes and all(write.kind is LogicalWriteKind.UPDATE for write in writes)
    for count in (0, 1, 3):
        assert_read_versions(flow, {"n": count})

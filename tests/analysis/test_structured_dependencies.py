"""Actual recursive dependency contracts, checked against small access traces."""

from itertools import product

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_frontier import QueueEndpoint
from ascendflow.analysis.access_order import build_access_order
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_regions import (
    AliasRegion,
    CountRelation,
    LoopCountGuard,
    StorageCell,
)
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import Expr
from ascendflow.target.ascend910b import Actor
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv, shared_l0c_qk_pv
from tests.models.dependency_edges import materialized_edges, trace_edges
from tests.models.structured_events import check_compiled_events


def dependencies(source):
    structure = lower_native_structure(N.from_source(source) if isinstance(source, str) else source)
    memory = collect_native_memory(structure.module)
    return build_access_order(memory)


def test_scale_real_endpoints_and_all_counts_match_access_trace():
    result = dependencies(scale)
    loop = next(iter(result.memory.control.loops.values()))
    load, compute, store = (t.operation for t in result.memory.templates)
    assert len(result.needs) == 12 and len(result.required) == 8
    assert {n.distance for n in result.needs} == {0, 2}
    assert all(n.loop is loop.operation for n in result.needs)
    assert all(n.period == 2 for n in result.needs)
    for count in range(7):
        events = []
        for ordinal in range(count):
            endpoint = lambda task, ordinal=ordinal: (
                task,
                Actor.AIV0,
                ((loop.operation, ordinal),),
            )
            events.extend(
                (
                    (endpoint(load), ((("x", ordinal % 2), False, True),)),
                    (
                        endpoint(compute),
                        ((("x", ordinal % 2), True, False), (("y", ordinal % 2), False, True)),
                    ),
                    (endpoint(store), ((("y", ordinal % 2), True, False),)),
                )
            )
        assert materialized_edges(result, {"tiles": count, "$group": 0, "$sid": 0}) == trace_edges(
            events
        )


@pytest.mark.parametrize("source", [batched_matmul, nested_copy, nested_cv, shared_l0c_qk_pv])
def test_real_cube_nested_and_cv_sources_close_without_old_completion_graph(source):
    result = dependencies(source)
    operations = {template.operation for template in result.memory.templates}
    assert any(
        n.loop is not None and n.distance is not None and n.distance > 0 for n in result.needs
    )
    assert all(
        n.source.operation in operations and n.consumer.operation in operations
        for n in result.needs
    )
    assert result.work.native_templates == result.memory.native_contract_calls
    assert result.work.cells < 100
    assert result.work.needs < 1000


def test_zero_loop_is_identity_between_actual_parent_writer_and_reader():
    result = dependencies("""
        def empty():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,0.0)
                for j in N.Pipelined(0):
                    N.fill(x,1.0)
                N.mul(x,2.0,out=y)
""")
    before, empty, after = (t.operation for t in result.memory.templates)
    assert [(n.kind, n.source.operation, n.consumer.operation) for n in result.needs] == [
        (HazardKind.RAW, before, after)
    ]
    assert all(
        n.source.operation is not empty and n.consumer.operation is not empty for n in result.needs
    )


def test_read_only_child_captures_every_reader_before_parent_overwrite():
    source = """
        def readers(n:N.i64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,0.0)
                for j in N.Pipelined(n):
                    N.mul(x,2.0,out=y)
                N.fill(x,1.0)
"""
    plan = synchronize_structure(N.from_source(source))
    release = next(n for n in plan.dependencies.needs if n.kind is HazardKind.WAR)
    assert isinstance(release.source.endpoint, QueueEndpoint)
    assert not release.source.bindings
    for count in range(5):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_rmw_same_task_next_generation_is_a_real_dependency():
    result = dependencies("""
        def rmw(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                for j in N.Pipelined(n):
                    N.mul(x,2.0,out=x)
""")
    loop_raw = next(
        n
        for n in result.needs
        if n.kind is HazardKind.RAW and n.source.operation is n.consumer.operation
    )
    assert loop_raw.distance == 1
    assert loop_raw.source != loop_raw.consumer
    assert not any(n.kind is HazardKind.WAR and n.source == n.consumer for n in result.needs)


def test_first_and_last_branches_preserve_values_without_phase_case_products():
    source = """
        def phases(n:N.i64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for j in N.Pipelined(n):
                    if j == 0:
                        N.fill(x,1.0)
                    N.mul(x,2.0,out=x)
                    if j == n-1:
                        N.copy(x,y)
"""
    plan = synchronize_structure(N.from_source(source))
    for count in range(6):
        check_compiled_events(plan, {"n": count}, invocations=2)


@pytest.mark.parametrize("width", [2, 3])
def test_compound_slot_short_and_long_children_preserve_actual_byte_values(width):
    source = f"""
        def slots(q:N.i64,k:N.i64):
            with N.Kernel():
                x,y=N.alloc_ub(({width},64)),N.alloc_ub(({width},64))
                for parent in N.serial(q):
                    for child in N.Pipelined(k):
                        slot=(parent*k+child)%{width}
                        N.fill(x[slot,:],1.0)
                        N.mul(x[slot,:],2.0,out=y[slot,:])
"""
    plan = synchronize_structure(N.from_source(source))
    assert plan.dependencies.work.cells <= 2 * width**2
    child = next(
        loop
        for loop, logical in plan.admission.memory.control.loops.items()
        if logical.count.kind == "argument" and logical.count.args[0].name == "k"
    )
    long_cells = [
        transfer
        for cell, transfer in plan.dependencies.summaries[child].items()
        if isinstance(cell, StorageCell)
        and any(
            isinstance(guard, LoopCountGuard)
            and guard.relation is CountRelation.MORE
            and guard.threshold == width - 1
            for guard in cell.conditions
        )
    ]
    if width == 2:
        assert long_cells
    # Every residue is populated under the cell's existing k >= width guard.
    # Export actual accesses rather than queue-wide ENTRY/EXIT drains.
    assert all(
        not isinstance(point.endpoint, QueueEndpoint)
        for transfer in long_cells
        for point in transfer.first_writes + transfer.tail_reads
    )
    for q, k in product(range(4), range(2 * width + 1)):
        check_compiled_events(plan, {"q": q, "k": k}, invocations=2)


def test_default_flash_attention_closes_in_compact_actual_fragment_families():
    result = dependencies(attention_target)
    assert result.work.native_templates == 53
    assert result.work.projected_accesses < 500
    assert result.work.cells < 200
    assert result.work.needs < 1500
    assert any(isinstance(n.residue, Expr) for n in result.needs)
    assert any(domain.first > 0 for n in result.needs for domain in n.domains)
    assert any(n.source.families or n.consumer.families for n in result.needs)


def test_strided_native_rows_stay_exact_compressed_rectangles():
    result = dependencies("""
        def strided(Scratch:N.Tensor((128,1024),N.f32,access="read_write")):
            with N.Kernel(kind="mixed"):
                c=N.alloc_l0c((128,128),N.f32)
                x=N.alloc_ub((8,128),N.f32)
                with N.Cube():
                    N.copy(c,Scratch[:,0:128])
                with N.Vector():
                    N.copy(Scratch[0:8,0:128],x)
    """)
    gm = {
        cell for cell in result.summaries[result.memory.module] if cell.root.type.bank.data == "GM"
    }
    assert len(gm) == 2
    assert {cell.rows for cell in gm} == {8, 120}
    assert all(cell.stop - cell.start == 512 and cell.row_stride == 4096 for cell in gm)
    raw = [n for n in result.needs if n.kind is HazardKind.RAW]
    assert len(raw) == 2
    assert {n.consumer.participant for n in raw} == {Actor.AIV0, Actor.AIV1}
    assert raw[0].consumer.operation is raw[1].consumer.operation


def test_loop_work_depends_on_templates_and_ring_slots_not_trip_count():
    def source(count):
        return f"""
            def bounded():
                with N.Kernel():
                    x,y=N.alloc_ub((2,64)),N.alloc_ub((2,64))
                    for j in N.Pipelined({count}):
                        N.fill(x[j%2,:],1.0)
                        N.mul(x[j%2,:],2.0,out=y[j%2,:])
"""

    short, huge = (dependencies(source(count)) for count in (4, 10**9))
    assert short.work == huge.work
    assert huge.work.projected_accesses == 6
    assert huge.work.cells == 4


@pytest.mark.parametrize("last", [9, 8])
def test_nonzero_serial_start_and_step_accept_both_last_and_absent_branch(last):
    result = dependencies(f"""
        def serial():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for j in N.serial(3,10,2):
                    N.fill(x,1.0)
                    if j == {last}:
                        N.copy(x,y)
    """)
    assert result.work.native_templates == 2


def test_parent_dependent_possibly_empty_child_does_not_invent_last_writer():
    result = dependencies("""
        def holes(n:N.i64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,1.0)
                for parent in N.Pipelined(n):
                    N.mul(x,2.0,out=y)
                    for child in N.Pipelined(parent%2):
                        N.fill(x,2.0)
                    N.mul(x,3.0,out=y)
""")
    from ascendflow.analysis.access_frontier import QueueEndpoint

    assert any(
        isinstance(n.source.endpoint, QueueEndpoint)
        or isinstance(n.consumer.endpoint, QueueEndpoint)
        for n in result.needs
    )


@pytest.mark.parametrize("reader", ["j+1", "0"])
def test_contracted_family_preserves_order_for_shifted_or_invariant_aliases(reader):
    result = dependencies(f"""
        def shifted():
            with N.Kernel():
                x,y=N.alloc_ub((5,64)),N.alloc_ub((64,))
                for j in N.Pipelined(4):
                    N.fill(x[j,:],1.0)
                    N.mul(x[{reader},:],2.0,out=y)
""")
    assert result.work.alias_regions == 1
    assert any(isinstance(n.cell, AliasRegion) and n.kind is HazardKind.RAW for n in result.needs)
    assert any(need.kind is HazardKind.WAR and need.distance == 1 for need in result.required)


def test_unknown_symbolic_alias_keeps_waw_without_inventing_a_complete_definition():
    result = dependencies("""
        def aliases(a:N.i64(minimum=1),b:N.i64(minimum=1)):
            with N.Kernel():
                x=N.alloc_ub((256,))
                N.fill(x[a*32:a*32+32],1.0)
                N.fill(x[b*32:b*32+32],2.0)
    """)
    assert len(result.required) == 1 and result.required[0].kind is HazardKind.WAW
    assert isinstance(result.required[0].cell, AliasRegion)
    assert not hasattr(result, "values")


def test_unknown_scalar_carry_is_not_a_loop_invariant_branch():
    result = dependencies("""
        def alternating(X:N.Tensor((64,),N.f32),n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                flag=0
                for i in N.serial(n):
                    if flag==0:
                        N.fill(x,1.0)
                    else:
                        N.copy(X[:],x)
                    flag=1-flag
    """)
    # Both paths can occur in successive iterations. The common invocation
    # ports retain both queues without assuming lifetime-long exclusivity.
    assert any(isinstance(n.source.endpoint, QueueEndpoint) for n in result.needs)

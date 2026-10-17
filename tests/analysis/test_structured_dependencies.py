"""Actual recursive dependency contracts, checked against small access traces."""

from itertools import product

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_regions import (
    AccessOccurrence,
    DependencyIssueKind,
)
from ascendflow.diagnostics import CompileError
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr
from ascendflow.target.ascend910b import Actor
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv, shared_l0c_qk_pv
from tests.models.structured_domains import expression_value, guard_active, occurrence_key


def dependencies(source):
    structure = lower_native_structure(N.from_source(source) if isinstance(source, str) else source)
    memory = collect_native_memory(structure.module)
    return build_structured_dependencies(memory)


def dependency_issues(result):
    return result.values.unresolved + result.storage.unresolved


def materialized_edges(result, values):
    """Small independent interpreter for quantified obligations, never production."""
    edges = set()
    loops = tuple(result.memory.control.loops.values())
    for need in result.needs:
        assert isinstance(need.source, AccessOccurrence)
        domains = need.domains + need.source.families + need.consumer.families
        required = {domain.loop for domain in domains}
        selected = [loop for loop in loops if loop.operation in required]

        def visit(index, bindings, selected=selected, need=need, domains=domains):
            if index == len(selected):
                if not all(guard_active(g, bindings, result.memory) for g in need.guards):
                    return
                edges.add(
                    (
                        need.kind,
                        occurrence_key(need.source, bindings, result.memory),
                        occurrence_key(need.consumer, bindings, result.memory),
                    )
                )
                return
            logical = selected[index]
            current = [domain for domain in domains if domain.loop is logical.operation]
            limit = max(domain.count.evaluate(bindings) for domain in current)
            for ordinal in range(limit):
                context = bindings | {logical.ordinal.args[0]: ordinal}
                if all(
                    ordinal >= max(domain.first, domain.distance)
                    and ordinal < domain.count.evaluate(context)
                    and ordinal % domain.period == expression_value(domain.residue, context)
                    for domain in current
                ):
                    visit(index + 1, context)

        visit(0, values)
    return edges


def trace_edges(events):
    writers, readers, edges = {}, {}, set()
    for endpoint, accesses in events:
        for cell, reads, writes in accesses:
            writer = writers.get(cell)
            if reads and writer is not None:
                edges.add((HazardKind.RAW, writer, endpoint))
            if writes:
                if writer is not None:
                    edges.add((HazardKind.WAW, writer, endpoint))
                edges.update((HazardKind.WAR, reader, endpoint) for reader in readers.get(cell, ()))
                writers[cell], readers[cell] = endpoint, []
            elif reads:
                readers.setdefault(cell, []).append(endpoint)
    return edges


def test_scale_real_endpoints_and_all_counts_match_access_trace():
    result = dependencies(scale)
    result.require_closed()
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
    result.require_closed()
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
                for j in N.Pipelined(0,stages=1):
                    N.fill(x,1.0)
                N.mul(x,2.0,out=y)
    """)
    result.require_closed()
    before, empty, after = (t.operation for t in result.memory.templates)
    assert [(n.kind, n.source.operation, n.consumer.operation) for n in result.needs] == [
        (HazardKind.RAW, before, after)
    ]
    assert all(
        n.source.operation is not empty and n.consumer.operation is not empty for n in result.needs
    )


def test_read_only_child_requires_every_reader_before_parent_overwrite():
    result = dependencies("""
        def readers(n:N.int64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,0.0)
                for j in N.Pipelined(n,stages=1):
                    N.mul(x,2.0,out=y)
                N.fill(x,1.0)
    """)
    result.require_closed()
    release = next(n for n in result.needs if n.kind is HazardKind.WAR)
    assert release.source.families
    assert isinstance(release.source.bindings[-1].ordinal, Expr)
    assert (
        release.source.families[-1].count == next(iter(result.memory.control.loops.values())).count
    )
    before, reader, after = (t.operation for t in result.memory.templates)
    loop = next(iter(result.memory.control.loops.values()))
    initial = before, Actor.AIV0, ()
    overwrite = after, Actor.AIV0, ()
    for count in range(5):
        readers = {(reader, Actor.AIV0, ((loop.operation, ordinal),)) for ordinal in range(count)}
        edges = materialized_edges(result, {"n": count, "$group": 0, "$sid": 0})
        assert {edge for edge in edges if edge[0] is HazardKind.RAW} == {
            (HazardKind.RAW, initial, endpoint) for endpoint in readers
        }
        assert {edge for edge in edges if edge[0] is HazardKind.WAR} == {
            (HazardKind.WAR, endpoint, overwrite) for endpoint in readers
        }


def test_rmw_same_task_next_generation_is_a_real_dependency():
    result = dependencies("""
        def rmw(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                for j in N.Pipelined(n,stages=1):
                    N.mul(x,2.0,out=x)
    """)
    result.require_closed()
    loop_raw = next(
        n
        for n in result.needs
        if n.kind is HazardKind.RAW and n.source.operation is n.consumer.operation
    )
    assert loop_raw.distance == 1
    assert loop_raw.source != loop_raw.consumer
    assert not any(n.kind is HazardKind.WAR and n.source == n.consumer for n in result.needs)


def test_first_and_last_phase_actual_anchors_match_small_access_trace():
    result = dependencies("""
        def phases(n:N.int64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for j in N.Pipelined(n,stages=1):
                    if j == 0:
                        N.fill(x,1.0)
                    N.mul(x,2.0,out=x)
                    if j == n-1:
                        N.copy(x,y)
    """)
    result.require_closed()
    loop = next(iter(result.memory.control.loops.values()))
    initialize, update, final = (t.operation for t in result.memory.templates)
    for count in range(6):
        events = []
        for ordinal in range(count):
            endpoint = lambda task, ordinal=ordinal: (
                task,
                Actor.AIV0,
                ((loop.operation, ordinal),),
            )
            if ordinal == 0:
                events.append((endpoint(initialize), (("x", False, True),)))
            events.append((endpoint(update), (("x", True, True),)))
            if ordinal == count - 1:
                events.append((endpoint(final), (("x", True, False), ("y", False, True))))
        assert materialized_edges(result, {"n": count, "$group": 0, "$sid": 0}) == trace_edges(
            events
        )


def test_compound_physical_slot_short_children_and_long_children_match_trace():
    result = dependencies("""
        def slots(q:N.int64,k:N.int64(minimum=1)):
            with N.Kernel():
                x,y=N.alloc_ub((2,64)),N.alloc_ub((2,64))
                for parent in N.serial(q):
                    for child in N.Pipelined(k,stages=1):
                        slot=(parent*k+child)%2
                        N.fill(x[slot,:],1.0)
                        N.mul(x[slot,:],2.0,out=y[slot,:])
    """)
    result.require_closed()
    outer, inner = tuple(result.memory.control.loops.values())
    write, read = (t.operation for t in result.memory.templates)
    assert result.work.cells <= 8
    for q, k in product(range(4), range(1, 4)):
        events = []
        for parent, child in product(range(q), range(k)):
            endpoint = lambda task, parent=parent, child=child: (
                task,
                Actor.AIV0,
                ((outer.operation, parent), (inner.operation, child)),
            )
            slot = (parent * k + child) % 2
            events.extend(
                (
                    (endpoint(write), ((("x", slot), False, True),)),
                    (endpoint(read), ((("x", slot), True, False), (("y", slot), False, True))),
                )
            )
        actual = materialized_edges(result, {"q": q, "k": k, "$group": 0, "$sid": 0})
        expected = trace_edges(events)
        assert actual == expected, (
            q,
            k,
            [
                (kind.value, [g for _, g in src[2]], [g for _, g in dst[2]])
                for kind, src, dst in expected - actual
            ],
        )


def test_default_flash_attention_closes_in_compact_actual_fragment_families():
    result = dependencies(attention_target)
    result.require_closed()
    assert result.work.native_templates == 52
    assert result.work.projected_accesses < 500
    assert result.work.cells < 200
    assert result.work.needs < 1500
    assert any(isinstance(n.residue, Expr) for n in result.needs)
    assert any(domain.first > 0 for n in result.needs for domain in n.domains)
    assert any(e.all_members for e in result.storage.exits)


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
    result.require_closed()
    gm = {cell for cell in result.values.root.cells if cell.root.type.bank.data == "GM"}
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
                    for j in N.Pipelined({count},stages=1):
                        N.fill(x[j%2,:],1.0)
                        N.mul(x[j%2,:],2.0,out=y[j%2,:])
        """

    short, huge = (dependencies(source(count)) for count in (4, 10**9))
    short.require_closed()
    huge.require_closed()
    assert short.work == huge.work
    assert huge.work.projected_accesses == 6
    assert huge.work.cells == 4


@pytest.mark.parametrize("last,closed", [(9, True), (8, False)])
def test_nonzero_serial_start_and_step_use_actual_last_induction(last, closed):
    result = dependencies(f"""
        def serial():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for j in N.serial(3,10,2):
                    N.fill(x,1.0)
                    if j == {last}:
                        N.copy(x,y)
    """)
    assert result.closed is closed
    if not closed:
        assert any(
            issue.kind is DependencyIssueKind.PARENT_ACTIVATION
            for issue in dependency_issues(result)
        )


def test_parent_dependent_possibly_empty_child_does_not_invent_last_writer():
    result = dependencies("""
        def holes(n:N.int64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,1.0)
                for parent in N.Pipelined(n,stages=1):
                    N.mul(x,2.0,out=y)
                    for child in N.Pipelined(parent%2,stages=1):
                        N.fill(x,2.0)
                    N.mul(x,3.0,out=y)
    """)
    assert not result.closed
    assert any(
        issue.kind is DependencyIssueKind.PARENT_ACTIVATION for issue in dependency_issues(result)
    )
    with pytest.raises(CompileError, match="activation"):
        result.require_closed()


@pytest.mark.parametrize("reader", ["j+1", "0"])
def test_contracted_family_cannot_erase_shifted_or_invariant_aliases(reader):
    result = dependencies(f"""
        def shifted():
            with N.Kernel():
                x,y=N.alloc_ub((5,64)),N.alloc_ub((64,))
                for j in N.Pipelined(4,stages=1):
                    N.fill(x[j,:],1.0)
                    N.mul(x[{reader},:],2.0,out=y)
    """)
    assert not result.closed
    assert any(
        issue.kind is DependencyIssueKind.NONPERIODIC_REUSE for issue in dependency_issues(result)
    )


def test_unknown_symbolic_alias_fails_closed_instead_of_disjoint_fallback():
    result = dependencies("""
        def aliases(a:N.int64(minimum=1),b:N.int64(minimum=1)):
            with N.Kernel():
                x=N.alloc_ub((256,))
                N.fill(x[a*32:a*32+32],1.0)
                N.fill(x[b*32:b*32+32],2.0)
    """)
    assert not result.closed
    assert any(
        issue.kind is DependencyIssueKind.SYMBOLIC_ALIAS for issue in dependency_issues(result)
    )
    with pytest.raises(CompileError, match="coordinate"):
        result.require_closed()


def test_unknown_scalar_carry_is_not_a_loop_invariant_branch():
    result = dependencies("""
        def alternating(X:N.Tensor((64,),N.f32),n:N.int64):
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
    # The two branches can execute on successive iterations. An unknown
    # carried scalar cannot justify their lifetime-long mutual exclusion.
    assert not result.closed
    assert any(
        issue.kind is DependencyIssueKind.PARENT_ACTIVATION for issue in dependency_issues(result)
    )

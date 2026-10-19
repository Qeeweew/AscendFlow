"""Packet lifetimes retain one actual parent clock through finite boundaries.

Small native traces independently interpret retained stage order and accesses.
These tests stop at dependency facts, before notification selection.
"""

from types import SimpleNamespace

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_regions import AccessOccurrence, LoopOrdinalGuard
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr
from ascendflow.target.ascend910b import Actor
from tests.models.dependency_edges import materialized_edges, trace_edges
from tests.models.structured_domains import actual_dispatch


def _source(offsets, reverse):
    labels = tuple("abc"[: len(offsets)])
    order = tuple(reversed(labels)) if reverse else labels
    body = "\n".join(
        f'            with N.Stage({offset},name="{label}"):\n'
        f"                N.fill(x,{float(index)})\n"
        f"                N.copy(x,O[g,{index},:])"
        for index, (label, offset) in enumerate(zip(labels, offsets, strict=True))
    )
    return f"""def scratch(O:N.Tensor((n,{len(offsets)},64),N.f32,access="write"),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,),name="scratch")
        for g in N.Pipelined(n,stages={max(offsets) + 1},issue_order={{"vector":{order!r}}}):
{body}
"""


@pytest.fixture(
    scope="module",
    params=[
        (offsets, reverse)
        for offsets in ((0, 1), (1, 0), (0, 3), (0, 1, 3))
        for reverse in (False, True)
    ],
    ids=lambda case: f"{case[0]}-{'reverse' if case[1] else 'forward'}",
)
def packet_case(request):
    offsets, reverse = request.param
    structure = lower_native_structure(
        N.from_source(_source(offsets, reverse), disjoint_external=True)
    )
    memory = collect_native_memory(structure.module)
    result = build_structured_dependencies(memory)
    partitions = [
        partition
        for partition in result.order.packet_partitions
        if partition.cell.root.owner.sym_name.data == "scratch"
    ]
    assert len(partitions) == 1
    return result, partitions[0]


def _native_trace(result, root, parameters):
    selection = SimpleNamespace(dependencies=result, dispatch=result.dispatch)
    templates = {template.operation: template for template in result.memory.templates}
    events = []
    for endpoint, context in actual_dispatch(selection, parameters, Actor.AIV0):
        touched = []
        for access in templates[endpoint[0]].accesses:
            if access.address.root is root:
                base = access.address.offset.evaluate(context)
                touched.extend(
                    (byte, access.mode.reads, access.mode.writes)
                    for byte in range(base, base + access.address.geometry.total_nbytes)
                )
        events.append((endpoint, touched))
    return events


@pytest.mark.parametrize("count", range(5))
def test_short_and_steady_packet_dependencies_match_actual_native_trace(packet_case, count):
    result, partition = packet_case
    scratch = SimpleNamespace(
        memory=result.memory,
        needs=tuple(need for need in result.needs if need.cell.root is partition.cell.root),
    )
    parameters = {"n": count, "$group": 0, "$sid": 0}
    assert materialized_edges(scratch, parameters) == trace_edges(
        _native_trace(result, partition.cell.root, parameters)
    )


def test_boundary_and_steady_storage_edges_share_canonical_parent_bindings(packet_case):
    result, partition = packet_case
    loop = partition.loop
    logical = result.memory.control.loops[loop]
    storage = [
        need
        for need in result.needs
        if need.cell == partition.cell and need.kind is not HazardKind.RAW
    ]
    assert storage and {need.kind for need in storage} == {HazardKind.WAR, HazardKind.WAW}
    singleton, steady = [], []
    for need in storage:
        assert isinstance(need.source, AccessOccurrence)
        assert need.loop is need.scope is loop
        assert type(need.distance) is int
        assert tuple(binding.loop for binding in need.source.bindings) == (loop,)
        assert tuple(binding.loop for binding in need.consumer.bindings) == (loop,)
        source, consumer = need.source.bindings[0].ordinal, need.consumer.bindings[0].ordinal
        assert isinstance(source, Expr) and consumer == logical.ordinal
        for ordinal in (0, 1, 4):
            context = {"n": 5, logical.ordinal.args[0]: ordinal}
            assert source.evaluate(context) == ordinal - need.distance
        assert len(need.domains) == 1
        domain = need.domains[0]
        assert domain.loop is loop and domain.distance == need.distance
        assert domain.period == need.period == 1
        assert domain.residue == need.residue == 0
        boundary_guard = any(
            isinstance(guard, LoopOrdinalGuard)
            and guard.loop is loop
            and guard.comparison.lhs == logical.ordinal
            and guard.comparison.predicate == "=="
            for guard in need.guards
        )
        (singleton if boundary_guard else steady).append(need)
    assert singleton and steady
    assert any(need.distance > 0 for need in storage)
    if max(packet.offset for packet in partition.packets) == 3:
        assert any(need.distance < 0 for need in storage)

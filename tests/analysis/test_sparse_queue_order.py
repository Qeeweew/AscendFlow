"""Compare sparse routing with the independent, explicit blocking relation.

The dense construction exists only in this test. Compare weighted reachability
at every real endpoint/cut, including positive reentry paths, rather than an
implementation-specific topological rank or generated instruction string.
"""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_order import build_access_order
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventCase, EventDomains
from ascendflow.analysis.queue_order import QueuePoint, _OrderConstruction
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_protocols import SynchronizationSelection
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Engine
from tests.fixtures.notification_cycles import (
    alternating_queues,
    child_to_parent,
    collective_children,
    finite_ports,
    local_child_route,
    mode2_participants,
    nested_channel_cycle,
    notification,
    port,
    prologue_stream_epilogue,
    sibling_ports,
    translated_child_route,
)
from tests.support.structured import prepared


class _RecordedOrder(_OrderConstruction):
    def __init__(self, domains):
        super().__init__(domains)
        self.graphs = {}

    def _rank(self, region, points, edges):
        self.graphs[region] = tuple(points), tuple(edges)
        return super()._rank(region, points, edges)


class _DenseOrder(_RecordedOrder):
    def _gates(self, region, points, node, edge):
        for wait in points:
            if wait.phase not in {"wait", "output"}:
                continue
            for source in points:
                if source.phase not in {"set", "input"} or not self._route(wait, source):
                    continue
                wp, sp = self._position(wait, region), self._position(source, region)
                if wp.child is not None and wp.child is sp.child:
                    edge(wait, source, 1 if region is not self.root else None)
                    continue
                forward = sp.lexical >= wp.lexical
                distance = 0 if forward else 1 if region is not self.root else None
                if region is not self.root and wait.bound is not None and source.bound is not None:
                    a, _ = self._time(wait, region)
                    b, _ = self._time(source, region)
                    if self.control.invariant_over(a, region) or self.control.invariant_over(
                        b, region
                    ):
                        al, ah = self._boundary_time(wait, region)
                        bl, bh = self._boundary_time(source, region)
                        delta = self.domains.arithmetic.exact_constant(expression("sub", b, a))
                        if (delta is not None and (delta < 0 if forward else delta <= 0)) or (
                            bh is not None and al is not None and (bh < al if forward else bh <= al)
                        ):
                            continue
                        if (delta is not None and delta > 0) or (
                            bl is not None and ah is not None and bl > ah
                        ):
                            distance = 1
                edge(wait, source, distance)


def key(point):
    return (
        point.operation,
        point.actor,
        point.engine,
        point.phase,
        point.notification,
        point.bound,
        point.case,
    )


def reachable(graph):
    points, edges = graph
    outgoing = [[] for _ in points]
    for a, b, positive in edges:
        outgoing[a].append((b, positive))
    result = set()
    for i, point in enumerate(points):
        if point.phase == "route":
            continue
        visited, pending = set(), list(outgoing[i])
        while pending:
            target, positive = pending.pop()
            if (target, positive) in visited:
                continue
            visited.add((target, positive))
            if points[target].phase != "route":
                result.add((key(point), key(points[target]), positive))
            pending.extend((b, int(positive or step)) for b, step in outgoing[target])
    return result


@pytest.mark.parametrize(
    "fixture",
    [
        nested_channel_cycle,
        alternating_queues,
        lambda: collective_children(reverse_vector=False),
        lambda: collective_children(reverse_vector=True),
        lambda: child_to_parent(scalar=True),
        lambda: child_to_parent(scalar=False),
        lambda: local_child_route(vector_publication=True),
        lambda: local_child_route(vector_publication=False),
        finite_ports,
        lambda: finite_ports(at_most_one=True),
        sibling_ports,
        lambda: sibling_ports(at_most_one=True),
        prologue_stream_epilogue,
        lambda: prologue_stream_epilogue(cyclic=True),
        lambda: translated_child_route(1)[0],
        lambda: translated_child_route(2)[0],
        lambda: mode2_participants(reverse=False),
        lambda: mode2_participants(reverse=True),
    ],
)
def test_sparse_preserves_weighted_endpoint_paths_and_child_interfaces(fixture):
    domains = prepared(fixture())
    dense, sparse = _DenseOrder(domains), _RecordedOrder(domains)
    try:
        expected = dense.run()
    except CompileError as error:
        with pytest.raises(CompileError) as actual:
            sparse.run()
        assert (actual.value.kind, actual.value.phase, actual.value.message) == (
            error.kind,
            error.phase,
            error.message,
        )
        return
    actual = sparse.run()
    assert len(expected.regions) == len(actual.regions)
    for a, b in zip(expected.regions, actual.regions, strict=True):
        assert a.operation is b.operation
        assert set(a.origins) == set(b.origins)
        assert {(key(x), key(y)) for x, y in a.transfers} == {
            (key(x), key(y)) for x, y in b.transfers
        }
        assert reachable(dense.graphs[a.operation]) == reachable(sparse.graphs[b.operation])


def sequential_domains(count):
    body = "\n".join("        N.fill(x,1.0)\n        N.load_scalar(x[:1])" for _ in range(count))
    source = N.from_source(
        "def sequential():\n    with N.Kernel():\n        x=N.alloc_ub((64,))\n" + body
    )
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = replace(build_access_order(memory), required=())
    selection = SynchronizationSelection(dependencies, (), ())
    notifications = []
    for i in range(0, len(memory.templates), 2):
        source, wait = memory.templates[i : i + 2]
        notifications.append(
            notification(
                str(i),
                port(selection, source, source=True, coordinates=(), activation=()),
                port(selection, wait, source=False, coordinates=(), activation=()),
                Engine.V,
                Engine.S,
            )
        )
    return EventDomains(replace(selection, notifications=tuple(notifications)), ResourceWork())


def test_sequential_gate_work_uses_range_indices_instead_of_pair_enumeration():
    measurements = []
    for count in (16, 64, 256):
        construction = _RecordedOrder(sequential_domains(count))
        order = construction.run()
        work = order.work
        assert work.gate_classes == 1
        assert work.gate_queries <= 2 * count
        assert work.routing_points < count
        assert work.origin_constraints == 0
        assert work.zero_relations + work.positive_relations < count * (count.bit_length() + 6)
        measurements.append(work.zero_relations + work.positive_relations)
    # Four times as many source instructions cannot produce sixteen times as
    # many gate edges. This checks structural work, independent of CPU noise.
    assert measurements[2] < 6 * measurements[1]
    assert measurements[1] < 6 * measurements[0]


def test_negative_same_actor_publication_still_rejects_clock_origin():
    construction = _RecordedOrder(sequential_domains(1))
    source, wait = construction.endpoints
    with pytest.raises(CompileError, match="clock-origin cycle"):
        construction._origins(construction.root, [(source, wait, -1)])


def test_selected_and_repeated_gate_classes_keep_exact_weights_and_range_hulls():
    # Isolate gate routing on real dispatch positions. These endpoints do
    # not claim a paired event protocol, memory coverage or ID reuse proof.
    source = N.from_source("""
def boundaries(I:N.Tensor((64,),N.f32),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for j in N.Pipelined(n,issue_order=("later","early")):
            with N.Stage(0,name="early"):
                N.fill(x,1.0)
                N.load_scalar(x[:1])
            with N.Stage(2,name="later"):
                N.copy(I,x)
                N.fill(x,2.0)
                N.load_scalar(x[:1])
""")
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = replace(build_access_order(memory), required=())
    selection = SynchronizationSelection(dependencies, (), ())
    domains = EventDomains(selection, ResourceWork())
    (loop,) = domains.control.loops.values()
    case, points = EventCase(ActionGuard(), (), (), ()), []
    for template in memory.templates:
        for phase in ("set", "wait"):
            for coordinate in (
                loop.ordinal,
                literal(0),
                literal(1),
                expression("sub", loop.count, literal(1)),
            ):
                for upper in (3, 7):
                    bound = port(
                        selection,
                        template,
                        source=phase == "set",
                        coordinates=((loop.operation, coordinate),),
                        activation=(
                            Comparison(loop.count, ">", literal(1)),
                            Comparison(loop.count, "<=", literal(upper)),
                        ),
                    )
                    engine = template.spec.engine
                    event = notification("gate", bound, bound, engine, engine)
                    points.append(
                        QueuePoint(bound.operation, bound.actor, engine, phase, event, bound, case)
                    )

    def graph(construction):
        nodes, edges = list(points), set()
        indices = {point: i for i, point in enumerate(nodes)}
        construction.boundary_groups[loop.operation] = {p: (p,) for p in points}

        def node(source):
            point = QueuePoint(source.operation, source.actor, source.engine, "route")
            indices[point] = len(nodes)
            nodes.append(point)
            return point

        def edge(a, b, distance):
            if distance is not None:
                edges.add((indices[a], indices[b], distance))

        construction._gates(loop.operation, tuple(points), node, edge)
        return nodes, edges

    assert reachable(graph(_RecordedOrder(domains))) == reachable(graph(_DenseOrder(domains)))

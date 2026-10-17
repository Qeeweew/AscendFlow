"""Whole-formula binding and conservative interface projection contracts."""

from dataclasses import replace

import pytest

from tests.analysis.test_case_progress import edge, nodes
from tests.analysis.test_progress_bound_facts import invocation
from tests.analysis.test_progress_parent_interfaces import PARENT_REUSE
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.progress_graphs import GraphTopology
from xdsl_npu.analysis.progress_regions import (
    ChannelProjection,
    InterfaceTransfer,
    RegionInterface,
)
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import Expr, expression, literal

N = Expr("argument", ("n",))


@pytest.fixture(scope="module")
def domains():
    return prepared(selected(PARENT_REUSE))


@pytest.mark.parametrize("opaque_path", (False, True))
def test_child_formula_uses_one_ancestor_shift_for_every_boolean_branch(domains, opaque_path):
    composer, parent, child, source, wait = invocation(domains)
    ordinal = domains.control.loops[parent].ordinal
    shifted_source = composer.endpoints.stream_port(source, child)
    assert shifted_source is not None
    assert dict(shifted_source.bindings) == {ordinal: expression("add", ordinal, literal(1))}
    # Both endpoints enter the same containing proof frame. Their original
    # previous/current invocation coordinates must still differ by one.
    shifted_wait = replace(wait, bindings=shifted_source.bindings)
    scope = composer.activations.region(child)

    def equals(value, constant):
        return scope.guard(ActionGuard((Comparison(value, "==", literal(constant)),)))

    following = expression("add", ordinal, literal(1))
    activation = scope.all_of(
        scope.any_of(equals(ordinal, 1), equals(ordinal, 3)),
        scope.any_of(equals(following, 2), equals(following, 5)),
    )
    if opaque_path:
        graph = scope.graph(GraphTopology(3, ((0, 1), (1, 2))), (activation, scope.true))
        activation = scope.path(graph, frozenset((1,)), 0, 2)
    provenance = (object(),)
    relation = InterfaceTransfer(shifted_source, shifted_wait, activation, False, provenance)
    interface = RegionInterface(child, (shifted_source, shifted_wait), (), (relation,))
    imported = composer.endpoints.import_interface(child, interface, ())
    (bound,) = imported.relations
    assert bound.activation is not activation
    assert not bound.exact and bound.projections is provenance
    for generation in range(6):
        context = {ordinal.args[0]: generation}
        assert activation.evaluate(context) == (generation in (1, 3) and generation + 1 in (2, 5))
        assert bound.activation.evaluate(context) == (
            generation + 1 in (1, 3) and generation + 2 in (2, 5)
        )
        source_coordinate = dict(composer.endpoints.coordinates(bound.source))[parent]
        wait_coordinate = dict(composer.endpoints.coordinates(bound.destination))[parent]
        assert wait_coordinate.evaluate(context) - source_coordinate.evaluate(context) == 1


def test_child_formula_rejects_endpoints_with_distinct_ancestor_bindings(domains):
    composer, _, child, source, wait = invocation(domains)
    scope = composer.activations.region(child)
    relation = InterfaceTransfer(source, wait, scope.true, False, ())
    interface = RegionInterface(child, (source, wait), (), (relation,))
    with pytest.raises(CompileError, match="shared actual ancestor parameter binding") as failure:
        composer.endpoints.import_interface(child, interface, (source.notification,))
    assert failure.value.phase == "structured-event-progress"


@pytest.mark.parametrize("inexact_route", (False, True))
def test_aggregate_or_keeps_exactness_only_for_a_complete_direct_witness(domains, inexact_route):
    composer, *_ = invocation(domains)
    ports = nodes(composer, 3)
    first, other = Comparison(N, "==", literal(0)), Comparison(N, "!=", literal(0))
    direct = replace(edge(composer, 0, 1, first), projections=(object(),))
    relations = (direct,)
    if inexact_route:
        relations += (
            replace(edge(composer, 0, 2, other), exact=False),
            edge(composer, 2, 1),
        )
    graph = ChannelProjection(composer.root, ports, relations, ())
    paths, witnesses = composer._paths(graph)
    paths.eliminate((2,))
    (boundary,) = composer._boundary(graph, paths, witnesses, 2)
    interface = composer._interface(graph, paths, witnesses, (), ports[:2])
    (exported,) = interface.relations
    for transfer in (boundary, exported):
        assert transfer.exact is not inexact_route
        assert transfer.projections == ((graph,) if inexact_route else direct.projections)
        assert transfer.activation.evaluate({"n": 0})
        assert transfer.activation.evaluate({"n": 1}) is inexact_route
    if inexact_route:
        assert boundary.activation.kind == "path"
    else:
        assert boundary.activation is direct.activation


def test_projecting_a_current_loop_predicate_does_not_create_an_exact_true_edge(domains):
    composer, parent, _, _, _ = invocation(domains)
    ordinal = domains.control.loops[parent].ordinal
    ports = tuple(replace(port, operation=parent) for port in nodes(composer, 2))
    direct = edge(composer, 0, 1, Comparison(ordinal, "==", literal(0)))
    graph = ChannelProjection(parent, ports, (direct,), ())
    paths, witnesses = composer._paths(graph)
    (boundary,) = composer._boundary(graph, paths, witnesses, 2)
    interface = composer._interface(graph, paths, witnesses, (), ports)
    (exported,) = interface.relations
    assert not direct.activation.evaluate({ordinal.args[0]: 1})
    for transfer in (boundary, exported):
        assert transfer.activation.kind == "true"
        assert not transfer.exact and transfer.projections == (graph,)


@pytest.mark.parametrize("stream", (False, True))
def test_real_cycle_through_an_or_path_is_never_refuted_as_an_and(domains, stream):
    composer, *_ = invocation(domains)
    ports = nodes(composer, 3)
    first, other = Comparison(N, "==", literal(0)), Comparison(N, "!=", literal(0))
    graph = ChannelProjection(
        composer.root,
        ports,
        (
            edge(composer, 0, 1, first),
            edge(composer, 0, 2, other),
            edge(composer, 2, 1),
            edge(composer, 1, 0, first),
        ),
        (),
    )
    paths, witnesses = composer._paths(graph)
    paths.eliminate((2,))
    arcs = composer._boundary(graph, paths, witnesses, 2)
    forward = next(arc for arc in arcs if arc.source == 0)
    assert forward.activation.kind == "path"
    assert all(arc.activation.evaluate({"n": 0}) for arc in arcs)
    establish = composer.ranks.stream if stream else composer.ranks.finite
    with pytest.raises(CompileError, match="rank|possible parent cycle"):
        establish(composer.root, ports[:2], arcs)


@pytest.mark.parametrize("stream", (False, True))
def test_separate_path_predicates_share_the_same_condition_context(domains, stream):
    composer, *_ = invocation(domains)
    scope = composer.activations.region(composer.root)
    first, other = Comparison(N, "==", literal(0)), Comparison(N, "!=", literal(0))
    graph = scope.graph(
        GraphTopology(4, ((0, 2), (2, 1), (1, 3), (3, 0))),
        (
            scope.guard(ActionGuard((first,))),
            scope.true,
            scope.guard(ActionGuard((other,))),
            scope.true,
        ),
    )
    hidden = frozenset((2, 3))
    arcs = (
        replace(edge(composer, 0, 1), activation=scope.path(graph, hidden, 0, 1), exact=False),
        replace(edge(composer, 1, 0), activation=scope.path(graph, hidden, 1, 0), exact=False),
    )
    for n in range(3):
        assert sum(arc.activation.evaluate({"n": n}) for arc in arcs) == 1
    if stream:
        _, rank, certificates = composer.ranks.stream(composer.root, nodes(composer, 2), arcs)
        assert not rank and len(certificates) == 2
        for n in range(3):
            assert any(certificate.condition.evaluate({"n": n}) for certificate in certificates)
    else:
        excluded, conditional = composer.ranks.finite(composer.root, nodes(composer, 2), arcs)
        assert len(excluded) == 1 and not conditional
        assert all(not excluded[0].condition.evaluate({"n": n}) for n in range(3))


def test_nested_path_stream_filter_removes_varying_predicates_at_every_graph_level(domains):
    composer, parent, _, _, _ = invocation(domains)
    scope = composer.activations.region(parent)
    ordinal = domains.control.loops[parent].ordinal
    first = scope.guard(ActionGuard((Comparison(ordinal, "==", literal(0)),)))
    other = scope.guard(ActionGuard((Comparison(ordinal, "!=", literal(0)),)))
    graph = scope.graph(
        GraphTopology(4, ((0, 2), (2, 1), (1, 3), (3, 0))),
        (first, scope.true, other, scope.true),
    )
    hidden = frozenset((2, 3))
    nested = scope.graph(
        GraphTopology(3, ((0, 1), (1, 2))),
        (scope.path(graph, hidden, 0, 1), scope.true),
    )
    arcs = (
        replace(
            edge(composer, 0, 1),
            activation=scope.path(nested, frozenset((1,)), 0, 2),
            steps=1,
            exact=False,
        ),
        replace(
            edge(composer, 1, 0),
            activation=scope.path(graph, hidden, 1, 0),
            steps=-1,
            exact=False,
        ),
    )
    # The two visits may use different generations. The same bare ordinal
    # must not exclude their cycle in a stream, even through a nested Path.
    with pytest.raises(CompileError, match="increasing blocking-interface rank"):
        composer.ranks.stream(parent, nodes(composer, 2), arcs)

"""Singleton endpoints need one guarded coordinate, independent of other boundaries."""

from types import SimpleNamespace

import pytest

from tests.test_scope_guarded_completion import command, edge, finite_covered, mark, singleton
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.completion_flow import ALL
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion

POSITIVE = CountDomain(1, 0, minimum=1)
ENDPOINTS = {
    "source-head": (singleton(2), None),
    "source-tail": (singleton(1, end=True), None),
    "target-head": (None, singleton(2)),
    "target-tail": (None, singleton(1, end=True)),
    "both-head": (singleton(1), singleton(3)),
    "both-tail": (singleton(1, end=True), singleton(2, end=True)),
    "head-tail": (singleton(1), singleton(1, end=True)),
    "tail-head": (singleton(1, end=True), singleton(2)),
}


def _endpoints(period, kind):
    graph, mapping = Graph(period, None), {}
    source, target = (command(graph, index) for index in range(2))
    for endpoint, activation in zip((source, target), ENDPOINTS[kind], strict=True):
        if activation is not None:
            mark(mapping, endpoint, activation)
    return graph, mapping, source, target


@pytest.mark.parametrize("period", (1, 3))
@pytest.mark.parametrize("distance", (-2, 0, 2))
@pytest.mark.parametrize("kind", ENDPOINTS)
def test_singleton_head_tail_and_mixed_endpoints_match_finite_bfs_after_graph_growth(
    period, distance, kind
):
    graph, mapping, source, target = _endpoints(period, kind)
    proof = GuardedCompletion(graph, mapping)
    domains = (ALL, CountDomain(4, 1, minimum=1, maximum=13))
    for linked in (False, True):
        if linked:
            edge(graph, source, target, distance)
        fresh = GuardedCompletion(graph, mapping)
        for residue in range(period):
            for domain in domains:
                activation = Activation(domain, period=period, residues=(residue,))
                actual = proof.coverage(source[0], target[0], distance, activation)
                independent = fresh.coverage(source[0], target[0], distance, activation)
                for count in range(14):
                    expected = domain.contains(count) and finite_covered(
                        graph, mapping, source, target, distance, activation, count
                    )
                    assert any(piece.contains(count) for piece in actual) == expected, (
                        period,
                        distance,
                        kind,
                        linked,
                        residue,
                        domain,
                        count,
                    )
                    assert any(piece.contains(count) for piece in independent) == expected
        assert proof.statistics()["cases"] <= proof.statistics()["queries"]


@pytest.mark.parametrize("side", ("source", "target"))
@pytest.mark.parametrize("distance", (-1, 1))
def test_native_last_gap_singletons_keep_residues_and_signed_distance(side, distance):
    graph = Graph(3, None)
    native = Command(0, "aic", SimpleNamespace(stage=0), "task", last_gap=1)
    point = "end" if side == "source" else "begin"
    nodes = tuple(graph.node(native, residue, point) for residue in range(graph.period))
    ordinary = command(graph, 1)
    source, target = (
        ((native, nodes), ordinary) if side == "source" else (ordinary, (native, nodes))
    )
    proof = GuardedCompletion(graph, {})
    for linked in (False, True):
        if linked:
            edge(graph, source, target, distance)
        for residue in range(graph.period):
            activation = Activation(POSITIVE, period=3, residues=(residue,))
            coverage = proof.coverage(source[0], target[0], distance, activation)
            for count in range(1, 10):
                assert any(piece.contains(count) for piece in coverage) == finite_covered(
                    graph, {}, source, target, distance, activation, count
                )


@pytest.mark.parametrize("kind", ("source-head", "source-tail", "target-head", "target-tail"))
def test_unrelated_large_head_and_tail_extrema_cost_only_one_real_endpoint_case(monkeypatch, kind):
    graph, mapping, source, target = _endpoints(1, kind)
    edge(graph, source, target)
    for index in range(64):
        first = command(graph, 2 + 2 * index)
        last = command(graph, 3 + 2 * index)
        mark(mapping, first, singleton(1000 * (index + 1)))
        mark(mapping, last, singleton(1000 * (index + 1), end=True))
    proof = GuardedCompletion(graph, mapping)
    potential = graph.potential()
    boundary = proof._boundary_measures(POSITIVE, potential)
    head, tail = proof._bounds(source[1][0], target[1][0], 0, boundary, *potential)
    assert head >= 64000 and tail >= 64000
    # The original enumeration would exceed even the unchanged production
    # limit. This test makes the exact single-coordinate cost explicit.
    monkeypatch.setattr("xdsl_npu.scope_guarded_completion.MAX_GUARDED_CASES", 1)
    assert proof.covered(source[0], target[0], 0, Activation(POSITIVE))
    assert proof.statistics()["cases"] == proof.statistics()["origins"] == 1


@pytest.mark.parametrize("period", (1, 3))
def test_singleton_count_holes_retain_original_count_guards(period):
    graph, mapping, source, target = _endpoints(period, "source-tail")
    mark(mapping, source, singleton(1, end=True, count=CountDomain(2, 0, minimum=0)))
    proof = GuardedCompletion(graph, mapping)
    activation = Activation(ALL)
    coverage = proof.coverage(source[0], target[0], 0, activation)
    for count in range(14):
        assert any(piece.contains(count) for piece in coverage) == finite_covered(
            graph, mapping, source, target, 0, activation, count
        )
    assert not any(piece.contains(2) for piece in coverage)
    assert any(piece.contains(3) for piece in coverage)
    edge(graph, source, target)
    assert proof.covered(source[0], target[0], 0, activation)


@pytest.mark.parametrize("inactive", ("outside-window", "disjoint-counts", "mixed-singletons"))
def test_inactive_singleton_pairs_remain_vacuous_without_completion_facts(inactive):
    graph, mapping, source, target = _endpoints(1, "source-head")
    required = CountDomain(1, 0, minimum=1, maximum=8)
    if inactive == "outside-window":
        mark(mapping, source, singleton(100))
    elif inactive == "disjoint-counts":
        mark(mapping, source, singleton(2, count=CountDomain(2, 0, minimum=0)))
        mark(mapping, target, singleton(2, count=CountDomain(2, 1, minimum=1)))
    else:
        mark(mapping, source, singleton(2))
        mark(mapping, target, singleton(2, end=True))
        required = CountDomain(2, 0, minimum=2, maximum=8)  # Equality needs N=5.
    proof = GuardedCompletion(graph, mapping)
    activation = Activation(required, Position(False, 0), Position(True, 0))
    assert proof.covered(source[0], target[0], 0, activation)
    assert proof.statistics()["origins"] == proof.statistics()["condition_facts"] == 0
    for count in range(1, 9):
        if required.contains(count):
            assert finite_covered(graph, mapping, source, target, 0, activation, count)

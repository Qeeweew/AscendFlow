"""Complete guarded proofs survive appended paths; partial proofs are reread."""

from random import Random
from types import SimpleNamespace

import pytest

from tests.test_scope_guarded_completion import command, edge, finite_covered, mark, singleton
from xdsl_npu.completion import CountDomain
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion

POSITIVE = CountDomain(1, 0, minimum=1)


@pytest.mark.parametrize("period", (1, 3))
def test_complete_answers_do_not_reregister_after_repeated_appended_protocol_edges(
    monkeypatch, period
):
    graph, mapping = Graph(period, None), {}
    source, target = (command(graph, index) for index in range(2))
    edge(graph, source, target)
    proof = GuardedCompletion(graph, mapping)
    query = source[0], target[0], 0, Activation(POSITIVE)
    assert proof.covered(*query)
    before = proof.statistics()
    # Reuse consumes no additional cases, even when the cumulative limit is full.
    monkeypatch.setattr("xdsl_npu.scope_guarded_completion.MAX_GUARDED_CASES", before["cases"])
    for index in range(2, 10):
        added = command(graph, index)
        mark(mapping, added, singleton(count=CountDomain(1, 0, exact=index)))
        edge(graph, source, added)
        assert proof.covered(*query)
        assert proof.statistics()["queries"] == before["queries"]
        assert proof.statistics()["cases"] == before["cases"]
        assert proof.arcs == len(graph.edges)
        assert len(proof.topology.families) == len(graph.nodes)
        assert proof.boundaries == {} and graph.potentials is None
    with pytest.raises(CompileError, match="case budget exceeded"):
        proof.covered(target[0], source[0], 0, Activation(POSITIVE))
    assert proof.cases >= before["cases"]  # No reset or extension of the limit.


def test_false_answer_is_reregistered_until_new_paths_prove_the_whole_domain():
    graph = Graph(1, None)
    source, middle, target = (command(graph, index) for index in range(3))
    proof = GuardedCompletion(graph, {})
    query = source[0], target[0], 0, Activation(POSITIVE)
    assert not proof.covered(*query)
    initial = proof.statistics()
    edge(graph, source, middle)
    assert not proof.covered(*query)
    assert proof.statistics()["queries"] == initial["queries"] + 1
    edge(graph, middle, target)
    assert proof.covered(*query)
    assert proof.statistics()["queries"] == initial["queries"] + 2
    assert proof.statistics()["cases"] > initial["cases"]
    complete = proof.statistics()
    unrelated = command(graph, 3)
    edge(graph, source, unrelated)
    assert proof.covered(*query)
    assert proof.statistics()["queries"] == complete["queries"]


def test_partial_count_coverage_is_discarded_and_recomputed_after_appended_paths():
    graph, mapping = Graph(1, None), {}
    source, startup, steady, target = (command(graph, index) for index in range(4))
    mark(mapping, startup, singleton(count=CountDomain(1, 0, exact=1)))
    mark(mapping, steady, Activation(CountDomain(1, 0, minimum=2)))
    edge(graph, source, startup)
    edge(graph, startup, target)
    proof = GuardedCompletion(graph, mapping)
    query = source[0], target[0], 0, Activation(POSITIVE)
    partial = proof.coverage(*query)
    assert any(piece.contains(1) for piece in partial)
    assert not any(piece.contains(2) for piece in partial)
    queries = proof.statistics()["queries"]
    edge(graph, source, steady)
    assert proof.coverage(*query) == partial
    assert proof.statistics()["queries"] == queries + 1
    edge(graph, steady, target)
    assert proof.covered(*query)
    assert proof.statistics()["queries"] == queries + 2


@pytest.mark.parametrize("alias_only", (False, True))
def test_new_endpoint_residue_cannot_reuse_an_old_vacuous_answer(alias_only):
    graph = Graph(3, None)
    source = Command(0, "aic", SimpleNamespace(stage=0), "scope_component")
    first = graph.node(source, 0, "component")
    graph.ids[source.index, 0, "begin"] = first
    second = graph.node(source, 1, "component") if alias_only else None
    target = command(graph, 1)
    graph.edge((first, 0), (target[1][0], 0))
    proof = GuardedCompletion(graph, {})
    query = source, target[0], 0, Activation(period=3, residues=(1,))
    assert proof.covered(*query)
    old_key = next(iter(proof.answers))
    before = proof.statistics()
    if second is None:
        second = graph.node(source, 1, "component")
    graph.ids[source.index, 1, "begin"] = second
    assert not proof.covered(*query)
    # The earlier proof remains valid for its original endpoint tuple only.
    assert old_key in proof.answers and len(proof.answers) == 2
    assert proof.statistics()["queries"] == before["queries"] + 1
    assert proof.statistics()["cases"] > before["cases"]


@pytest.mark.parametrize("mutation", ("change", "remove", "add"))
def test_old_activation_mutation_is_rejected_before_a_complete_cache_hit(mutation):
    graph, mapping = Graph(1, None), {}
    source, target = (command(graph, index) for index in range(2))
    edge(graph, source, target)
    if mutation != "add":
        mark(mapping, source, Activation())
    proof = GuardedCompletion(graph, mapping)
    query = source[0], target[0], 0, Activation(POSITIVE)
    assert proof.covered(*query)
    before = proof.statistics()
    if mutation == "remove":
        mapping.clear()
    else:
        mark(mapping, source, singleton(1))
    with pytest.raises(CompileError, match="activation changed"):
        proof.covered(*query)
    assert proof.statistics()["queries"] == before["queries"]
    assert proof.statistics()["cases"] == before["cases"]


@pytest.mark.parametrize("period", (1, 3))
def test_incremental_retained_answers_match_fresh_proofs_and_independent_finite_bfs(period):
    rng = Random(910 + period)
    graph, mapping = Graph(period, None), {}
    commands = [command(graph, index) for index in range(6)]
    mark(mapping, commands[1], singleton(count=CountDomain(1, 0, exact=1)))
    mark(mapping, commands[3], singleton(1, end=True))
    mark(mapping, commands[4], Activation(CountDomain(1, 0, minimum=2)))
    edge(graph, commands[0], commands[1])
    edge(graph, commands[1], commands[5])
    proof = GuardedCompletion(graph, mapping)
    queries = [
        (commands[a], commands[b], distance, activation)
        for a, b, distance in ((0, 0, 0), (0, 5, 0), (0, 5, 1), (1, 5, -1))
        for activation in (
            Activation(POSITIVE),
            singleton(count=CountDomain(2, 0, minimum=0)),
            singleton(1, end=True, count=CountDomain(2, 1, minimum=1)),
        )
    ]
    candidates = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    rng.shuffle(candidates)
    for step in range(7):
        if step:
            a, b = candidates[step - 1]
            edge(graph, commands[a], commands[b], rng.randrange(2))
        fresh = GuardedCompletion(graph, mapping)
        for source, target, distance, activation in queries:
            actual = proof.coverage(source[0], target[0], distance, activation)
            independent = fresh.coverage(source[0], target[0], distance, activation)
            for count in range(8):
                expected = activation.count.contains(count) and finite_covered(
                    graph, mapping, source, target, distance, activation, count
                )
                assert any(piece.contains(count) for piece in actual) == expected
                assert any(piece.contains(count) for piece in independent) == expected
        assert proof.topology is not fresh.topology
        assert proof.flows is not fresh.flows

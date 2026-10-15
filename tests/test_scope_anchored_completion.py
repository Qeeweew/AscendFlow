"""Anchored parent proofs compared with literal finite native-graph BFS."""

from collections import defaultdict, deque
from random import Random

import pytest

from tests.test_scope_guarded_completion import command, edge, mark, singleton
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.model import CompileError
from xdsl_npu.scope_anchored_completion import AnchoredCompletion
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion

FIRST, LAST = Position(False, 0), Position(True, 0)


def finite_covered(
    graph, mapping, source, destination, source_position, destination_position, activation, count
):
    """Instantiate only small test N; no production family or interval solver."""
    assert activation.count.contains(count)
    instances = set()
    for node, meta in enumerate(graph.nodes):
        for generation in range(meta.lower, count - meta.upper):
            if generation % graph.period != meta.residue:
                continue
            if meta.last_gap is not None and generation != count - 1 - meta.last_gap:
                continue
            active = mapping.get(node)
            if active is not None and not (
                active.count.contains(count)
                and active.first.evaluate(count) <= generation <= active.last.evaluate(count)
                and generation % active.period in active.residues
            ):
                continue
            instances.add((node, generation))
    following = defaultdict(set)
    for a, b, cycles in graph.edges:
        shift = cycles * graph.period + graph.nodes[b].residue - graph.nodes[a].residue
        for node, generation in instances:
            target = b, generation + shift
            if node == a and target in instances:
                following[node, generation].add(target)

    def generations(position):
        if position is not None:
            return (position.evaluate(count),)
        return tuple(
            g
            for g in range(activation.first.evaluate(count), activation.last.evaluate(count) + 1)
            if g % activation.period in activation.residues
        )

    for sg in generations(source_position):
        start = source[1][sg % graph.period], sg
        if start not in instances:
            continue
        seen, pending = {start}, deque([start])
        while pending:
            for node in following[pending.popleft()]:
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        for dg in generations(destination_position):
            target = destination[1][dg % graph.period], dg
            if target in instances and target not in seen:
                return False
    return True


def prefix_path(period=1, maximum=None):
    graph = Graph(period, maximum)
    source = command(graph, 0)
    prefix = command(graph, 1, kind="scope_prefix", point="queue", guard=(0, 0))
    destination = command(graph, 2)
    edge(graph, source, prefix)
    edge(graph, prefix, prefix, 1)
    edge(graph, prefix, destination)
    return graph, source, prefix, destination


@pytest.mark.parametrize("period", [1, 3])
def test_first_to_every_reader_and_every_reader_to_last_use_the_same_native_prefix(period):
    graph, source, _, destination = prefix_path(period)
    proof = AnchoredCompletion(GuardedCompletion(graph, {}))
    assert proof.covered_many(
        [
            (source[0], destination[0], FIRST, None, Activation()),
            (source[0], destination[0], None, LAST, Activation()),
            (source[0], destination[0], FIRST, LAST, Activation()),
        ]
    ) == [True, True, True]
    assert proof.statistics()["reverse_domains"] == period
    assert graph.domain_states == 0 and graph._completion is None


@pytest.mark.parametrize("period", [1, 3])
@pytest.mark.parametrize("kind", ["scope_component", "cross_set"])
def test_a_last_reader_completion_cannot_substitute_for_all_readers(period, kind):
    graph, mapping = Graph(period, None), {}
    source = command(graph, 0)
    last = command(graph, 1, kind=kind, point="end" if kind == "cross_set" else "component")
    destination = command(graph, 2)
    mark(mapping, last, singleton(end=True))
    edge(graph, source, last)
    edge(graph, last, destination)
    proof = AnchoredCompletion(GuardedCompletion(graph, mapping))
    assert proof.covered(source[0], destination[0], LAST, LAST, Activation())
    assert not proof.covered(source[0], destination[0], None, LAST, Activation())
    coverage = proof.coverage(source[0], destination[0], None, LAST, Activation())
    assert any(domain.contains(1) for domain in coverage)
    assert all(not domain.contains(2) for domain in coverage)


@pytest.mark.parametrize("period", [1, 3])
def test_fact_interval_union_covers_all_targets_without_filling_a_missing_middle(period):
    # Only the fixed first and tail nodes can supply destination facts until
    # a middle is added. Their union must not fill an uncovered middle.
    graph, mapping = Graph(period, None), {}
    source = command(graph, 0)
    prefix = command(graph, 1, kind="scope_prefix", point="queue", guard=(0, 0))
    tail, destination = command(graph, 2), command(graph, 3)
    mark(mapping, tail, singleton(end=True))
    edge(graph, source, prefix)
    edge(graph, prefix, prefix, 1)
    edge(graph, source, destination)
    edge(graph, prefix, tail)
    edge(graph, tail, destination)
    guarded = GuardedCompletion(graph, mapping)
    proof = AnchoredCompletion(guarded)
    query = source[0], destination[0], FIRST, None, Activation()
    coverage = proof.coverage(*query)
    for count in range(9):
        assert any(domain.contains(count) for domain in coverage) == (count <= 2)
    middle = command(graph, 4)
    even = CountDomain(2, 0, minimum=0)
    mark(mapping, middle, Activation(even, Position(False, 1), Position(True, 1)))
    edge(graph, prefix, middle)
    edge(graph, middle, destination)
    coverage = proof.coverage(*query)
    for count in range(9):
        assert any(domain.contains(count) for domain in coverage) == (count <= 2 or count % 2 == 0)
    assert not proof.covered(*query)
    assert proof.covered(source[0], destination[0], FIRST, None, Activation(even))


@pytest.mark.parametrize("period", [1, 3])
def test_nonzero_fixed_and_count_relative_anchors_keep_their_original_count(period):
    graph, source, _, destination = prefix_path(period)
    proof = AnchoredCompletion(GuardedCompletion(graph, {}))
    before_tail = Position(True, 1)
    queries = [
        (source[0], destination[0], Position(False, 1), LAST, Activation()),
        (source[0], destination[0], before_tail, LAST, Activation()),
        (source[0], destination[0], LAST, before_tail, Activation()),
        (source[0], destination[0], None, before_tail, Activation()),
    ]
    for query in queries:
        coverage = proof.coverage(*query)
        for count in range(9):
            expected = finite_covered(graph, {}, source, destination, *query[2:], count)
            assert any(domain.contains(count) for domain in coverage) == expected


@pytest.mark.parametrize("period", [1, 3])
def test_original_count_conditions_remain_original_in_both_time_directions(period):
    graph, mapping = Graph(period, None), {}
    source = command(graph, 0)
    conditioned = command(graph, 1)
    prefix = command(graph, 2, kind="scope_prefix", point="queue", guard=(0, 0))
    destination = command(graph, 3)
    counts = CountDomain(1, 0, minimum=3)
    mark(mapping, conditioned, Activation(counts))
    edge(graph, source, conditioned)
    edge(graph, conditioned, prefix)
    edge(graph, prefix, prefix, 1)
    edge(graph, prefix, destination)
    proof = AnchoredCompletion(GuardedCompletion(graph, mapping))
    for sp, dp in ((FIRST, None), (None, LAST), (FIRST, LAST)):
        assert proof.covered(source[0], destination[0], sp, dp, Activation(counts))
        assert not proof.covered(source[0], destination[0], sp, dp, Activation())
        coverage = proof.coverage(source[0], destination[0], sp, dp, Activation())
        assert all(not domain.contains(1) and not domain.contains(2) for domain in coverage)
        assert any(domain.contains(3) and domain.contains(1 << 50) for domain in coverage)


@pytest.mark.parametrize("period", [1, 3])
def test_random_guarded_dags_match_independent_finite_bfs_in_all_three_query_forms(period):
    random = Random(7321 + period)
    for sample in range(10):
        graph, mapping = Graph(period, None), {}
        commands = [command(graph, index) for index in range(6)]
        mark(mapping, commands[1], singleton(sample % 2, count=CountDomain(2, 0, minimum=0)))
        mark(mapping, commands[3], singleton(sample % 2, end=True))
        mark(mapping, commands[4], Activation(CountDomain(1, 0, minimum=sample % 4)))
        for a in range(6):
            for b in range(a + 1, 6):
                if random.random() < 0.5:
                    edge(graph, commands[a], commands[b], random.randrange(-1, 3))
        proof = AnchoredCompletion(GuardedCompletion(graph, mapping))
        queries, expected_batch = [], []
        for a, b in ((0, 5), (1, 5), (0, 3)):
            for sp, dp in (
                (FIRST, None),
                (LAST, None),
                (None, LAST),
                (None, FIRST),
                (FIRST, LAST),
                (LAST, FIRST),
            ):
                activation = Activation(
                    first=Position(False, sample % 2),
                    last=Position(True, sample % 2),
                    period=period,
                    residues=(sample % period,),
                )
                coverage = proof.coverage(commands[a][0], commands[b][0], sp, dp, activation)
                for count in range(9):
                    expected = finite_covered(
                        graph, mapping, commands[a], commands[b], sp, dp, activation, count
                    )
                    assert any(domain.contains(count) for domain in coverage) == expected, (
                        period,
                        sample,
                        a,
                        b,
                        sp,
                        dp,
                        count,
                        coverage,
                    )
                    exact = Activation(
                        CountDomain(1, 0, exact=count),
                        activation.first,
                        activation.last,
                        activation.period,
                        activation.residues,
                    )
                    queries.append((commands[a][0], commands[b][0], sp, dp, exact))
                    expected_batch.append(expected)
        assert (
            AnchoredCompletion(GuardedCompletion(graph, mapping)).covered_many(queries)
            == expected_batch
        )


@pytest.mark.parametrize("count", [0, 1, 3, (1 << 50) + 3])
def test_static_counts_and_large_boundaries_never_expand_parent_clock(count, monkeypatch):
    graph, source, _, destination = prefix_path(3, count)

    def forbidden(*args, **kwargs):
        pytest.fail("anchored completion cannot instantiate N or use the old path oracle")

    monkeypatch.setattr(Graph, "domain", forbidden)
    monkeypatch.setattr(Graph, "_query", forbidden)
    proof = AnchoredCompletion(GuardedCompletion(graph, {}))
    assert proof.covered(source[0], destination[0], FIRST, None, Activation())
    assert proof.covered(source[0], destination[0], None, LAST, Activation())
    assert proof.covered(source[0], destination[0], FIRST, LAST, Activation())
    assert proof.statistics()["cases"] <= 3 * graph.period**2
    assert proof.statistics()["translated_domain_states"] == 0


def test_append_replays_new_arcs_into_existing_reverse_and_forward_facts():
    graph, mapping = Graph(3, None), {}
    source = command(graph, 0)
    prefix = command(graph, 1, kind="scope_prefix", point="queue", guard=(0, 0))
    destination = command(graph, 2)
    edge(graph, prefix, prefix, 1)
    edge(graph, prefix, destination)
    proof = AnchoredCompletion(GuardedCompletion(graph, mapping))
    count = Activation(CountDomain(1, 0, minimum=2))
    queries = [
        (source[0], destination[0], FIRST, None, count),
        (source[0], destination[0], None, LAST, count),
    ]
    assert proof.covered_many(queries) == [False, False]
    reverse = dict(proof.reverse)
    middle = command(graph, 3)
    mark(mapping, middle, count)
    edge(graph, source, middle)
    edge(graph, middle, prefix)
    assert proof.covered_many(queries) == [True, True]
    assert all(proof.reverse[residue] is state for residue, state in reverse.items())
    cases = proof.statistics()["cases"]
    assert proof.covered(source[1][0], destination[1][0], FIRST, None, count)
    assert proof.statistics()["cases"] == cases


def test_unknown_endpoints_and_missing_anchors_reject_but_inactive_bounds_are_vacuous():
    graph, source, _, destination = prefix_path()
    proof = AnchoredCompletion(GuardedCompletion(graph, {}))
    with pytest.raises(CompileError, match="at least one anchored"):
        proof.covered(source[0], destination[0], None, None, Activation())
    with pytest.raises(CompileError, match="Position"):
        proof.covered(source[0], destination[0], 0, LAST, Activation())
    unknown = Command(999, source[0].actor, source[0].task, source[0].kind)
    with pytest.raises(CompileError, match="endpoint"):
        proof.covered(unknown, destination[0], FIRST, LAST, Activation())
    assert proof.covered(source[0], destination[0], Position(True, -1), None, Activation())
    assert proof.covered(source[0], destination[0], None, Position(False, -1), Activation())

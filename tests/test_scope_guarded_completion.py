"""Guarded scope facts compared with literal finite graph reachability."""

from collections import defaultdict, deque
from random import Random
from types import SimpleNamespace

import pytest

from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.completion_flow import ALL
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion


def command(graph, index, *, kind="scope_component", point="component", guard=None):
    cmd = Command(index, "aic", SimpleNamespace(stage=0), kind)
    nodes = tuple(graph.node(cmd, residue, point, guard=guard) for residue in range(graph.period))
    if point == "component":
        for residue, node in enumerate(nodes):
            graph.ids[index, residue, "begin"] = node
    return cmd, nodes


def edge(graph, source, destination, shift=0):
    for residue, node in enumerate(source[1]):
        target = residue + shift
        graph.edge((node, 0), (destination[1][target % graph.period], target // graph.period))


def mark(mapping, nodes, activation):
    mapping.update((node, activation) for node in nodes[1])


def singleton(offset=0, *, end=False, count=ALL):
    point = Position(end, offset)
    return Activation(count, point, point)


def finite_covered(graph, mapping, source, target, distance, activation, count):
    """Literal N and BFS; no production family, interval or coverage routines."""
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
            destination = b, generation + shift
            if node == a and destination in instances:
                following[node, generation].add(destination)
    for generation in range(activation.first.evaluate(count), activation.last.evaluate(count) + 1):
        if generation % activation.period not in activation.residues:
            continue
        start = source[1][generation % graph.period], generation
        destination = target[1][(generation + distance) % graph.period], generation + distance
        if start not in instances or destination not in instances:
            continue
        seen, pending = {start}, deque([start])
        while pending:
            for node in following[pending.popleft()]:
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        if destination not in seen:
            return False
    return True


@pytest.mark.parametrize("period", [1, 3])
def test_exact_count_startup_does_not_prove_other_counts_or_other_epochs(period):
    graph, mapping = Graph(period, None), {}
    source, first, target = (command(graph, index) for index in range(3))
    one = CountDomain(1, 0, exact=1)
    mark(mapping, first, singleton(count=one))
    edge(graph, source, first)
    edge(graph, first, target)
    proof = GuardedCompletion(graph, mapping)
    assert proof.covered(source[0], target[0], 0, singleton(count=one))
    assert not proof.covered(source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1)))
    assert not proof.covered(
        source[0], target[0], 0, singleton(1, count=CountDomain(1, 0, minimum=2))
    )
    coverage = proof.coverage(source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1)))
    assert any(domain.contains(1) for domain in coverage)
    assert all(not domain.contains(2) for domain in coverage)


@pytest.mark.parametrize("period", [1, 3])
def test_tail_and_signed_steady_paths_match_every_small_count(period):
    graph, mapping = Graph(period, None), {}
    source, tail, target = (command(graph, index) for index in range(3))
    mark(mapping, tail, singleton(1, end=True))
    edge(graph, source, tail)
    edge(graph, tail, target, 1)
    proof = GuardedCompletion(graph, mapping)
    assert proof.covered(
        source[0], target[0], 1, singleton(1, end=True, count=CountDomain(1, 0, minimum=2))
    )
    activation = Activation(ALL, Position(False, 0), Position(True, 1))
    covered = proof.coverage(source[0], target[0], 1, activation)
    for count in range(9):
        assert any(d.contains(count) for d in covered) == finite_covered(
            graph, mapping, source, target, 1, activation, count
        )
    assert not proof.covered(source[0], target[0], 1, activation)

    graph, mapping = Graph(period, None), {}
    source, send, wait, target = (command(graph, index) for index in range(4))
    mark(mapping, send, Activation(ALL, Position(False, 1), Position(True, 0)))
    mark(mapping, wait, Activation(ALL, Position(False, 0), Position(True, 1)))
    edge(graph, source, send)
    edge(graph, send, wait, -1)
    edge(graph, wait, target)
    proof = GuardedCompletion(graph, mapping)
    activation = Activation(CountDomain(1, 0, minimum=2), Position(False, 1), Position(True, 0))
    assert proof.covered(source[0], target[0], -1, activation)
    assert not proof.covered(target[0], source[0], 1, Activation(CountDomain(1, 0, minimum=2)))


def test_original_count_conditions_survive_minimum_extent_translation():
    graph, mapping = Graph(1, None), {}
    source, conditioned, target = (command(graph, index) for index in range(3))
    counts = CountDomain(1, 0, minimum=3)
    mark(mapping, conditioned, Activation(counts))
    edge(graph, source, conditioned)
    edge(graph, conditioned, target)
    proof = GuardedCompletion(graph, mapping)
    # The canonical local extent is 1. N>=3 describes the ORIGINAL count,
    # and must neither disappear nor be evaluated against that local extent.
    assert proof.covered(source[0], target[0], 0, Activation(counts))
    assert not proof.covered(source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1)))
    coverage = proof.coverage(source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1)))
    assert all(not domain.contains(1) and not domain.contains(2) for domain in coverage)
    assert any(domain.contains(3) and domain.contains(1 << 50) for domain in coverage)


def test_incremental_nodes_and_arcs_replay_all_domains_with_one_topology():
    graph, mapping = Graph(1, None), {}
    source, first, target = (command(graph, index) for index in range(3))
    mark(mapping, first, singleton(count=CountDomain(1, 0, exact=1)))
    edge(graph, source, first)
    edge(graph, first, target)
    proof = GuardedCompletion(graph, mapping)
    query = source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1))
    assert not proof.covered(*query)
    domains = dict(proof.flows)
    steady = command(graph, 3)
    mark(mapping, steady, Activation(CountDomain(1, 0, minimum=2)))
    edge(graph, source, steady)
    assert not proof.covered(*query)
    edge(graph, steady, target)
    assert proof.covered(*query)
    assert all(proof.flows[key] is flow for key, flow in domains.items())
    assert all(flow.topology is proof.topology for flow in proof.flows.values())
    assert graph._completion is None and graph.domain_states == 0
    queries = proof.statistics()["queries"]
    assert proof.covered(source[1][0], target[1][0], 0, query[3])
    assert proof.statistics()["queries"] == queries  # begin/component aliases share the cache.


@pytest.mark.parametrize("kind", ["set", "cross_set"])
def test_transparent_prefix_retains_the_boundary_fact_count_condition(kind):
    graph, mapping = Graph(1, None), {}
    source = command(graph, 0)
    first = command(graph, 1, kind=kind, point="end")
    prefix = command(graph, 2, kind="scope_prefix", point="queue", guard=(0, 0))
    target = command(graph, 3)
    mark(mapping, first, singleton(count=CountDomain(1, 0, exact=1)))
    edge(graph, source, first)
    edge(graph, first, prefix)
    edge(graph, prefix, prefix, 1)
    edge(graph, prefix, target)
    proof = GuardedCompletion(graph, mapping)
    assert proof.covered(source[0], target[0], 0, singleton(count=CountDomain(1, 0, exact=1)))
    assert not proof.covered(source[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1)))
    assert not proof.covered(
        source[0], target[0], 0, singleton(1, count=CountDomain(1, 0, minimum=2))
    )


@pytest.mark.parametrize("distance", [1, 2])
def test_event_wait_to_republication_uses_actual_endpoint_activation(distance):
    graph, mapping = Graph(1, None), {}
    sent = command(graph, 0, kind="set", point="end")
    waited = command(graph, 1, kind="wait", point="begin")
    mark(mapping, sent, Activation(ALL, Position(False, 0), Position(True, 1)))
    mark(mapping, waited, Activation(ALL, Position(False, 1), Position(True, 0)))
    edge(graph, sent, waited, 1)
    edge(graph, waited, sent, distance)
    proof = GuardedCompletion(graph, mapping)
    query = Activation(ALL, Position(False, 1), Position(True, 0))
    assert proof.covered(waited[0], sent[0], distance, query)
    assert not proof.covered(waited[0], sent[0], distance - 1, query)
    for count in range(8):
        exact = Activation(CountDomain(1, 0, exact=count), query.first, query.last)
        assert proof.covered(waited[0], sent[0], distance - 1, exact) == finite_covered(
            graph, mapping, waited, sent, distance - 1, exact, count
        )


@pytest.mark.parametrize("period", [1, 3])
def test_random_guarded_dags_match_independent_finite_bfs(period):
    random = Random(9527 + period)
    for sample in range(12):
        graph, mapping = Graph(period, None), {}
        commands = [command(graph, index) for index in range(6)]
        count_guard = CountDomain(2, 0, minimum=0) if sample % 3 == 0 else ALL
        mark(mapping, commands[1], singleton(sample % 2, count=count_guard))
        mark(mapping, commands[3], singleton(sample % 3, end=True))
        mark(mapping, commands[4], Activation(CountDomain(1, 0, minimum=sample % 4)))
        for a in range(6):
            for b in range(a + 1, 6):
                if random.random() < 0.45:
                    edge(graph, commands[a], commands[b], random.randrange(-1, 3))
        proof = GuardedCompletion(graph, mapping)
        exact_queries, expected_batch = [], []
        for a, b, distance in ((0, 5, -1), (0, 5, 0), (0, 5, 1), (1, 5, 1), (0, 3, 0)):
            activation = Activation(ALL, Position(False, sample % 2), Position(True, sample % 2))
            coverage = proof.coverage(commands[a][0], commands[b][0], distance, activation)
            for count in range(9):
                expected = finite_covered(
                    graph, mapping, commands[a], commands[b], distance, activation, count
                )
                assert any(domain.contains(count) for domain in coverage) == expected, (
                    period,
                    sample,
                    a,
                    b,
                    distance,
                    count,
                    coverage,
                )
                exact = Activation(
                    CountDomain(1, 0, exact=count), activation.first, activation.last
                )
                exact_queries.append((commands[a][0], commands[b][0], distance, exact))
                expected_batch.append(expected)
        assert GuardedCompletion(graph, mapping).covered_many(exact_queries) == expected_batch


@pytest.mark.parametrize("count", [0, 1, 3, (1 << 50) + 3])
def test_static_count_and_large_runtime_domains_never_expand_parent_clock(count, monkeypatch):
    graph = Graph(3, count)
    source, target = (command(graph, index) for index in range(2))
    edge(graph, source, target)

    def forbidden(*args, **kwargs):
        pytest.fail("guarded scope queries must not instantiate N or use the old path oracle")

    monkeypatch.setattr(Graph, "domain", forbidden)
    monkeypatch.setattr(Graph, "_query", forbidden)
    proof = GuardedCompletion(graph, {})
    assert proof.covered(source[0], target[0], 0, Activation())
    statistics = proof.statistics()
    assert statistics["topology_instances"] == 1
    assert statistics["topology_nodes"] == 6
    assert statistics["cases"] <= 3 and statistics["translated_domain_states"] == 0


def test_changed_activation_is_rejected_and_case_budget_is_cumulative(monkeypatch):
    graph, mapping = Graph(1, None), {}
    source, first, target = (command(graph, index) for index in range(3))
    mark(mapping, first, singleton())
    edge(graph, source, first)
    edge(graph, first, target)
    proof = GuardedCompletion(graph, mapping)
    monkeypatch.setattr("xdsl_npu.scope_guarded_completion.MAX_GUARDED_CASES", 1)
    assert proof.covered(source[0], target[0], 0, singleton())
    with pytest.raises(CompileError, match="case budget exceeded"):
        proof.covered(source[0], target[0], 0, singleton(1))
    mapping[first[1][0]] = singleton(1)
    with pytest.raises(CompileError, match="activation changed"):
        proof.covered(source[0], target[0], 0, singleton())


def test_batch_seeds_all_origins_before_propagating_shared_intervals():
    graph = Graph(1, None)
    sources, length = 16, 32
    commands = [command(graph, index) for index in range(sources + length)]
    for source in commands[:sources]:
        edge(graph, source, commands[sources])
    for a, b in zip(commands[sources:-1], commands[sources + 1 :], strict=True):
        edge(graph, a, b)
    proof = GuardedCompletion(graph, {})
    queries = [(source[0], commands[-1][0], 0, Activation()) for source in commands[:sources]]
    # Byte-level duplicates must not consume more cases, and merged facts must
    # retain the identity of every source rather than borrowing another origin.
    queries.append((commands[0][0], commands[1][0], 0, Activation()))
    assert proof.covered_many(queries * 4) == ([True] * sources + [False]) * 4
    statistics = proof.statistics()
    assert statistics["fact_transfers"] == sources + length - 1
    assert statistics["propagation_passes"] == statistics["boundary_domains"] == 1
    assert statistics["origins"] == sources
    assert statistics["queries"] == statistics["cases"] == sources + 1


def test_batch_preserves_startup_tail_and_original_count_conditions():
    graph, mapping = Graph(1, None), {}
    a, b, startup, steady, tail, target, tail_target = (command(graph, index) for index in range(7))
    one, two = CountDomain(1, 0, exact=1), CountDomain(1, 0, minimum=2)
    mark(mapping, startup, singleton(count=one))
    mark(mapping, steady, Activation(two))
    mark(mapping, tail, singleton(1, end=True, count=two))
    edge(graph, a, startup)
    edge(graph, startup, target)
    edge(graph, b, steady)
    edge(graph, steady, target)
    edge(graph, a, tail)
    edge(graph, tail, tail_target, 1)
    queries = [
        (a[0], target[0], 0, singleton(count=one)),
        (a[0], target[0], 0, Activation(CountDomain(1, 0, minimum=1))),
        (b[0], target[0], 0, Activation(two)),
        (b[0], target[0], 0, singleton(count=one)),
        (a[0], tail_target[0], 1, singleton(1, end=True, count=two)),
        (a[0], tail_target[0], 1, Activation(two, Position(False, 0), Position(True, 1))),
    ]
    proof = GuardedCompletion(graph, mapping)
    assert proof.covered_many(queries) == [True, False, True, False, True, False]
    assert proof.statistics()["propagation_passes"] == len(proof.flows)


def test_unknown_endpoint_family_is_rejected_but_missing_residue_is_inactive():
    graph = Graph(3, None)
    source = Command(0, "aic", SimpleNamespace(stage=0), "scope_component")
    first = graph.node(source, 0, "component")
    second = graph.node(source, 1, "component")
    graph.ids[source.index, 0, "begin"] = first
    target = command(graph, 1)
    graph.edge((first, 0), (target[1][0], 0))
    proof = GuardedCompletion(graph, {})
    query = source, target[0], 0, Activation(period=3, residues=(1,))
    assert proof.covered(*query)
    # Publishing another existing node alias changes the endpoint family even
    # without new hardware nodes or arcs. The old vacuous answer is invalid.
    graph.ids[source.index, 1, "begin"] = second
    assert not proof.covered(*query)
    unknown = Command(99, "aic", SimpleNamespace(stage=0), "scope_component")
    for endpoint in (unknown, 99, "source"):
        with pytest.raises(CompileError, match="endpoint"):
            proof.covered(endpoint, target[0], 0, Activation())
    with pytest.raises(CompileError, match="endpoint family"):
        proof.covered(source, target[0], 0, Activation(), source_point="unknown")

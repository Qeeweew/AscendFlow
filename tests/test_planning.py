import random
from collections import defaultdict

import pytest

from xdsl_npu import autosync
from xdsl_npu.examples import cube_pipeline, vector_pipeline
from xdsl_npu.reachability import ProofSearch
from xdsl_npu.sync import CompletionPrefixes, Graph, completion_graph, execution


@pytest.mark.parametrize("seed", range(10))
def test_path_cache_deletions_agree_with_fresh_full_closure(seed):
    rng = random.Random(seed)
    graph = Graph()
    for i in range(24):
        graph.node(str(i))
    arcs = [(a, b) for a in range(24) for b in range(a + 1, 24) if rng.random() < 0.2]
    for a, b in arcs:
        graph.edge(a, b)
    graph.close()
    reachable = [(a, b) for a in range(24) for b in range(24) if graph.precedes(a, b)]
    queries = rng.sample(reachable, min(12, len(reachable)))
    search = ProofSearch(graph, queries)
    omitted = set()
    rng.shuffle(arcs)
    for i in range(0, len(arcs), 3):
        batch = set(arcs[i : i + 3])
        exclusions = defaultdict(set)
        for a, b in omitted | batch:
            exclusions[a].add(b)
        graph.close(exclusions)
        expected = all(graph.precedes(a, b) for a, b in queries)
        # The planner's upper bound must remain the original closed graph.
        graph.close()
        assert search.remove(batch) == expected
        if expected:
            omitted |= batch
        assert all(not set(path) & omitted for path in search.paths.values())


def test_removed_counter_queries_do_not_remove_shared_physical_obligations():
    graph = Graph()
    for i in range(4):
        graph.node(str(i))
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        graph.edge(a, b)
    graph.close()
    search = ProofSearch(graph, [(0, 3), (1, 2)])
    assert search.remove([(1, 2)], drop_queries=[(1, 2)])
    assert search.queries == {(0, 3)}
    assert not search.remove([(0, 3)])


def test_completion_prefix_only_captures_prior_retirements():
    graph = Graph()
    prefixes = CompletionPrefixes(graph)
    ends, notifications = [], []
    for i in range(64):
        end = graph.node(f"task{i}.end")
        effect = graph.node(f"notify{i}.effect")
        ends.append(end)
        notifications.append(effect)
        prefixes.record("V", end)
        prefixes.capture("V", effect)
    graph.close()
    assert sum(map(len, graph.edges)) < 4 * len(ends)
    for i, effect in enumerate(notifications):
        assert all(graph.precedes(end, effect) for end in ends[: i + 1])
        assert not any(graph.precedes(effect, end) for end in ends[i + 1 :])


@pytest.mark.parametrize("factory", [vector_pipeline, cube_pipeline])
def test_native_pipeline_graph_edges_grow_with_commands(factory):
    ex = execution(autosync(factory(24)))
    graph, *_ = completion_graph(ex)
    assert sum(map(len, graph.edges)) < 10 * len(ex.commands)

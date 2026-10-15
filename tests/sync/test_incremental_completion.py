"""Incremental coverage against an independent finite per-count BFS oracle."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion import CompletionPaths, CountDomain, covers

ALL = CountDomain(1, 0, minimum=0)
EVEN = CountDomain(2, 0, minimum=0)
ODD = CountDomain(2, 1, minimum=0)


def paths():
    return CompletionPaths(lambda *_: (), lambda *_: True)


def test_missing_path_is_updated_without_rebuilding_unrelated_frontiers():
    graph = paths()
    graph.add("a", "b", ALL)
    graph.add("x", "y", ALL)
    assert not graph.coverage("a", "c", ALL)
    assert not graph.coverage("x", "z", ALL)
    expansions = graph.expansions
    graph.add("b", "c", EVEN)
    assert graph.updates == 1  # Only the source-a frontier reached b.
    assert not covers(ALL, graph.coverage("a", "c", ALL))
    assert covers(EVEN, graph.coverage("a", "c", ALL))
    graph.add("b", "c", ODD)
    assert covers(ALL, graph.coverage("a", "c", ALL))
    assert graph.expansions <= expansions + 2
    assert len(graph.frontiers) == 2
    before = graph.expansions
    assert not graph.coverage("x", "z", ALL)
    assert graph.expansions == before


def test_successful_early_query_keeps_frontier_for_another_destination():
    graph = paths()
    graph.add("a", "b", ALL)
    graph.add("b", "c", ALL)
    assert covers(ALL, graph.coverage("a", "b", ALL))
    assert covers(ALL, graph.coverage("a", "c", ALL))
    assert len(graph.frontiers) == 1
    before = graph.expansions
    assert covers(ALL, graph.coverage("a", "c", ALL))
    assert graph.expansions == before


def test_updates_do_not_compose_disjoint_guards_or_inactive_endpoints():
    graph = CompletionPaths(lambda *_: (), lambda node, domain: node != "inactive")
    graph.add("a", "b", EVEN)
    graph.add("b", "c", ODD)
    assert not graph.coverage("a", "c", ALL)
    graph.add("a", "inactive", ALL)
    graph.add("inactive", "c", ALL)
    assert not graph.coverage("a", "c", ALL)
    graph.add("b", "c", EVEN)
    coverage = graph.coverage("a", "c", ALL)
    assert covers(EVEN, coverage) and not covers(ODD, coverage)


def test_large_count_boundaries_do_not_increase_frontier_expansions():
    sizes = []
    for n in (1024, 1048576, 10**12):
        graph = paths()
        graph.add("a", "b", CountDomain(1, 0, minimum=n))
        assert not covers(ALL, graph.coverage("a", "b", ALL))
        graph.add("a", "b", CountDomain(1, 0, minimum=0, maximum=n - 1))
        assert covers(ALL, graph.coverage("a", "b", ALL))
        sizes.append((len(graph.frontiers), graph.expansions, graph.updates))
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("seed", range(6))
def test_each_insert_and_partial_query_matches_finite_graph(seed):
    rng = random.Random(seed)
    graph, edges = paths(), defaultdict(list)
    for _ in range(16):
        source, target = rng.randrange(5), rng.randrange(5)
        period = rng.choice((1, 2, 3))
        guard = CountDomain(period, rng.randrange(period), minimum=rng.randrange(4))
        graph.add(source, target, guard)
        edges[source].append((target, guard))
        for start, finish in ((0, 4), (0, 2), (1, 4)):
            coverage = graph.coverage(start, finish, ALL)
            # All thresholds are below 4, all residue classes repeat every 6.
            for count in range(10):
                seen, pending = {start}, [start]
                while pending:
                    for dst, predicate in edges[pending.pop()]:
                        if predicate.contains(count) and dst not in seen:
                            seen.add(dst)
                            pending.append(dst)
                assert any(g.contains(count) for g in coverage) == (finish in seen)

"""Incremental completion facts against an independent per-count BFS oracle."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion import CountDomain, covers
from xdsl_npu.completion_flow import ALL, CompletionFlow

EVEN = CountDomain(2, 0, minimum=0)
ODD = CountDomain(2, 1, minimum=0)


def coverage(flow, node, bit=1):
    flow.propagate()
    return tuple(g for g, bits in flow.facts[node].items() if bits & bit)


def test_missing_path_is_updated_without_rebuilding_unrelated_frontiers():
    flow = CompletionFlow()
    a, b, c, x, y, z = [flow.node() for _ in range(6)]
    flow.add_fact(a, ALL, 1)
    flow.add_fact(x, ALL, 2)
    flow.edge(a, b)
    flow.edge(x, y)
    assert not coverage(flow, c)
    assert not coverage(flow, z, 2)
    unrelated = dict(flow.facts[y])
    flow.edge(b, c, EVEN)
    assert not covers(ALL, coverage(flow, c))
    assert covers(EVEN, coverage(flow, c))
    flow.edge(b, c, ODD)
    assert covers(ALL, coverage(flow, c))
    assert flow.facts[y] == unrelated
    assert not coverage(flow, c, 2)
    assert not coverage(flow, z, 2)


def test_successful_early_query_keeps_frontier_for_another_destination():
    flow = CompletionFlow()
    a, b, c = [flow.node() for _ in range(3)]
    flow.add_fact(a, ALL, 1)
    flow.edge(a, b)
    flow.edge(b, c)
    assert covers(ALL, coverage(flow, b))
    transfers = flow.transfers
    assert covers(ALL, coverage(flow, c))
    assert flow.transfers == transfers


def test_updates_do_not_compose_disjoint_guards_or_inactive_endpoints():
    flow = CompletionFlow()
    a, b, c, inactive = [flow.node() for _ in range(4)]
    flow.add_fact(a, ALL, 1)
    flow.edge(a, b, EVEN)
    flow.edge(b, c, ODD)
    assert not coverage(flow, c)
    flow.edge(a, inactive, None)
    flow.edge(inactive, c)
    assert not coverage(flow, c)
    flow.edge(b, c, EVEN)
    result = coverage(flow, c)
    assert covers(EVEN, result) and not covers(ODD, result)


def test_large_count_boundaries_do_not_increase_frontier_expansions():
    sizes = []
    for n in (1024, 1048576, 10**12):
        flow = CompletionFlow()
        a, b = flow.node(), flow.node()
        flow.add_fact(a, ALL, 1)
        flow.edge(a, b, CountDomain(1, 0, minimum=n))
        assert not covers(ALL, coverage(flow, b))
        flow.edge(a, b, CountDomain(1, 0, minimum=0, maximum=n - 1))
        assert covers(ALL, coverage(flow, b))
        sizes.append((len(flow.edges), flow.edge_count, flow.transfers))
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("seed", range(6))
def test_each_insert_and_partial_query_matches_finite_graph(seed):
    rng = random.Random(seed)
    flow, edges = CompletionFlow(), defaultdict(list)
    for _ in range(5):
        flow.node()
    flow.add_fact(0, ALL, 1)
    flow.add_fact(1, ALL, 2)
    for _ in range(16):
        source, target = rng.randrange(5), rng.randrange(5)
        period = rng.choice((1, 2, 3))
        guard = CountDomain(period, rng.randrange(period), minimum=rng.randrange(4))
        flow.edge(source, target, guard)
        edges[source].append((target, guard))
        for start, finish in ((0, 4), (0, 2), (1, 4)):
            result = coverage(flow, finish, 1 << start)
            # Thresholds are below 4; all residues repeat every 6.
            for count in range(10):
                seen, pending = {start}, [start]
                while pending:
                    for dst, predicate in edges[pending.pop()]:
                        if predicate.contains(count) and dst not in seen:
                            seen.add(dst)
                            pending.append(dst)
                assert any(g.contains(count) for g in result) == (finish in seen)

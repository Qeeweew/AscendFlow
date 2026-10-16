"""Disjoint origins avoid containment work without sharing completion facts."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion import CountDomain
from xdsl_npu.completion_flow import ALL
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.symbolic_completion import CompletionTopology, Family, SymbolicCompletion

EVEN = CountDomain(2, 0, minimum=0)
ODD = CountDomain(2, 1, minimum=0)
SHORT = CountDomain(1, 0, minimum=1, maximum=4)


@pytest.mark.parametrize("size", (1, 8, 64))
def test_independent_origins_at_distinct_positions_skip_quadratic_scans(size):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, size - 1)))
    for position in range(size):
        flow.seed(node, (0, position), ALL, 1 << position)
    counts = flow.statistics()
    assert counts["condition_facts"] == counts["fact_attempts"] == size
    assert counts["fact_updates"] == size
    # Previously each insertion visited all earlier records twice: m*(m-1).
    assert counts["fact_comparisons"] == 0
    for position in range(size):
        assert flow.facts[node][((0, position), (0, position), ALL)] == 1 << position


def test_disjoint_origin_really_bypasses_both_record_scans():
    class NoScan(dict):
        def items(self):
            pytest.fail("a wholly disjoint origin must not scan existing records")

    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 9)))
    flow.seed(node, (0, 0), ALL, 1)
    flow.facts[node] = NoScan(flow.facts[node])
    flow.seed(node, (0, 2), ALL, 2)
    assert len(flow.facts[node]) == 2 and flow.fact_comparisons == 0


def test_mixed_bits_preserve_existing_exact_key_and_supersede_pending_facts():
    flow = SymbolicCompletion()
    source, target = [flow.node(Family(0, (0, 9))) for _ in range(2)]
    flow.edge(source, target)
    flow.seed(source, (0, 4), ALL, 1)
    flow.add_fact(source, (0, 0), (0, 8), ALL, 2)
    flow.add_fact(source, (0, 0), (0, 8), ALL, 1 | 4)
    assert flow.fact_comparisons == 4  # Overlapping insert scans both records twice.
    flow.propagate()
    expected = {((0, 0), (0, 8), ALL): 1 | 2 | 4}
    assert flow.facts[source] == flow.facts[target] == expected
    assert not flow.pending and not flow.delta


def test_partially_dominated_bits_keep_their_independent_intervals():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 9)))
    flow.add_fact(node, (0, 0), (0, 8), ALL, 1)
    flow.add_fact(node, (0, 2), (0, 3), ALL, 1 | 2)
    assert flow.facts[node] == {
        ((0, 0), (0, 8), ALL): 1,
        ((0, 2), (0, 3), ALL): 2,
    }
    before = flow.statistics(), dict(flow.delta[node])
    flow.add_fact(node, (0, 2), (0, 3), ALL, 1 | 2)
    assert flow.fact_comparisons == before[0]["fact_comparisons"] + 1
    assert flow.fact_updates == before[0]["fact_updates"]
    assert flow.delta[node] == before[1]
    assert flow.coverage(node, (0, 7), 1, ALL)
    assert not flow.coverage(node, (0, 7), 2, ALL)


def test_repeated_origin_preserves_disjoint_count_guards():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 9)))
    flow.seed(node, (0, 0), EVEN, 1)
    flow.seed(node, (0, 2), ODD, 1)
    assert flow.fact_comparisons == 2 and len(flow.facts[node]) == 2
    for count in range(8):
        for point in (0, 2):
            guards = flow.coverage(node, (0, point), 1, ALL)
            assert any(g.contains(count) for g in guards) == (count % 2 == point // 2)


def test_replay_adds_fresh_origins_without_widening_or_losing_guards():
    flow = SymbolicCompletion()
    source, middle, target = [flow.node(Family(0, (0, 9))) for _ in range(3)]
    flow.seed(source, (0, 1), ALL, 1)
    flow.seed(source, (0, 3), ALL, 2)
    flow.propagate()
    flow.edge(source, middle, 1)
    flow.propagate()
    assert flow.fact_comparisons == 0
    flow.edge(middle, target, guard=EVEN)
    flow.propagate()
    flow.edge(source, target, guard=ODD)
    flow.propagate()
    for count in range(6):
        for bit, original in ((1, 1), (2, 3)):
            for point in range(6):
                guards = flow.coverage(target, (0, point), bit, ALL)
                assert any(g.contains(count) for g in guards) == (
                    point == original + (count % 2 == 0)
                )


def test_node_masks_are_sparse_and_local_to_each_activation_domain():
    topology = CompletionTopology()
    source = topology.node(Family(0, (1, -1)))
    tail = topology.node(Family(0, (1, -1), last=(1, -1)))
    topology.node(Family(0, (1, -1)))
    topology.edge(source, tail)
    active = SymbolicCompletion(topology=topology, count=3)
    inactive = SymbolicCompletion(topology=topology, count=2, terminal=False)
    active.seed(source, (0, 2), ALL, 1)
    active.propagate()
    assert not inactive._node_bits
    inactive.seed(source, (0, 0), ALL, 1)
    inactive.propagate()
    assert set(active._node_bits) == {source, tail}
    assert set(inactive._node_bits) == {source}
    assert active._node_bits is not inactive._node_bits
    assert active.coverage(tail, (0, 2), 1, ALL)
    assert not inactive.coverage(tail, (0, 1), 1, ALL)
    assert not inactive.coverage(source, (0, 2), 1, ALL)


def test_fast_insertion_does_not_bypass_the_transfer_budget():
    flow = SymbolicCompletion()
    source, target = [flow.node(Family(0, (0, 9))) for _ in range(2)]
    flow.seed(source, (0, 0), ALL, 1)
    flow.transfers = 2_000_000
    with pytest.raises(CompileError, match="propagation budget exceeded"):
        flow.edge(source, target)
    assert not flow.facts.get(target)


def _allows(guard, count):
    if guard.exact is not None:
        return count == guard.exact
    minimum = guard.period if guard.minimum is None else guard.minimum
    return (
        count >= minimum
        and count % guard.period == guard.residue
        and (guard.maximum is None or count <= guard.maximum)
    )


def _finite_paths(families, arcs, count, origin):
    instances = {
        (node, generation)
        for node, family in enumerate(families)
        for generation in range(family.lower, family.horizon[0] * count + family.horizon[1] + 1)
        if generation % family.period == family.residue
        and (family.last is None or generation == family.last[0] * count + family.last[1])
    }
    following = defaultdict(set)
    for source, target, shift, guard in arcs:
        if _allows(guard, count):
            for node, generation in instances:
                destination = target, generation + shift
                if node == source and destination in instances:
                    following[node, generation].add(destination)
    for node, generation in instances:
        family = families[node]
        destination = node, generation + family.period
        if family.monotone and destination in instances:
            following[node, generation].add(destination)
    node, point, guard = origin
    first = node, point[0] * count + point[1]
    seen = {first} if first in instances and _allows(guard, count) else set()
    pending = list(seen)
    while pending:
        for destination in following[pending.pop()]:
            if destination not in seen:
                seen.add(destination)
                pending.append(destination)
    return instances, seen


@pytest.mark.parametrize("seed", range(6))
def test_incremental_multiple_origins_match_independent_finite_paths(seed):
    rng = random.Random(seed)
    period = rng.choice((1, 2, 3))
    families = [Family(0, (1, -1), period, i % period, monotone=i == 1) for i in range(5)]
    families[-1] = Family(0, (1, -1), period, 4 % period, last=(1, -1))
    flow = SymbolicCompletion()
    for family in families:
        flow.node(family)
    origins = (
        (0, (0, 0), ALL),
        (0, (0, period), EVEN),
        (1, (1, -1), ODD),
        (2, (0, families[2].residue), SHORT),
    )
    for index, (node, point, guard) in enumerate(origins):
        flow.seed(node, point, guard, 1 << index)
    arcs = []
    candidates = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    rng.shuffle(candidates)
    for source, target in candidates[:7]:
        shift = families[target].residue - families[source].residue + period * rng.randrange(2)
        guard = rng.choice((ALL, EVEN, ODD, SHORT, CountDomain(1, 0, exact=2)))
        arcs.append((source, target, shift, guard))
        flow.edge(source, target, shift, guard=guard)
        flow.propagate()
        for count in range(10):
            for index, origin in enumerate(origins):
                instances, expected = _finite_paths(families, arcs, count, origin)
                actual = {
                    (node, generation)
                    for node, generation in instances
                    if any(
                        _allows(domain, count)
                        for domain in flow.coverage(node, (0, generation), 1 << index, ALL)
                    )
                }
                assert actual == expected, (
                    seed,
                    count,
                    origin,
                    actual - expected,
                    expected - actual,
                )

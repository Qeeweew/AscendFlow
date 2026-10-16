"""Exact interval unions compress forward work without joining origin guards."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion import CountDomain
from xdsl_npu.completion_flow import ALL
from xdsl_npu.symbolic_completion import Family, SymbolicCompletion


def _allows(domain, count):
    if domain.exact is not None:
        return count == domain.exact
    return (
        count >= (domain.period if domain.minimum is None else domain.minimum)
        and count % domain.period == domain.residue
        and (domain.maximum is None or count <= domain.maximum)
    )


@pytest.mark.parametrize("size", (8, 64, 512))
def test_adjacent_paths_have_one_interval_and_one_downstream_transfer(size):
    flow = SymbolicCompletion()
    source, middle, target = [flow.node(Family(0, (0, size + 1))) for _ in range(3)]
    for shift in range(size):
        flow.edge(source, middle, shift)
    flow.edge(middle, target)
    flow.seed(source, (0, 0), ALL, 1)
    flow.propagate()
    counts = flow.statistics()
    assert counts["condition_facts"] == 3
    assert counts["fact_transfers"] == size + 1
    assert counts["fact_interval_joins"] == size - 1
    assert counts["pending_interval_joins"] == size - 1
    assert flow.facts[target] == {((0, 0), (0, size - 1), ALL): 1}

    # A new path extends the retained union, while only its incoming support
    # becomes pending work. A delta need not keep the retained frontier's key.
    flow.edge(source, middle, size)
    assert flow.delta[middle] == {((0, size), (0, size), ALL): 1}
    flow.propagate()
    assert flow.transfers == size + 3
    assert flow.facts[target] == {((0, 0), (0, size), ALL): 1}


@pytest.mark.parametrize("size", (64, 256, 512))
def test_repeated_sparse_origins_compare_only_their_own_records(size):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 4 * size)))
    for index in range(size):
        flow.seed(node, (0, 4 * index), ALL, 1 << index)
    for index in range(size):
        flow.seed(node, (0, 4 * index + 2), ALL, 1 << index)
    counts = flow.statistics()
    assert counts["condition_facts"] == 2 * size
    assert counts["fact_comparisons"] == 2 * size
    # Include index construction, mask tests, tree walks, and updates. This
    # cost is visible separately rather than hidden by the comparison count.
    assert 0 < counts["fact_index_steps"] <= 12 * size * size.bit_length()
    assert counts["pending_frontier_steps"] == 0


@pytest.mark.parametrize("size", (8, 32, 128))
def test_dense_masks_use_direct_scans_without_index_maintenance(size):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 3 * size)))
    bits = (1 << 256) - 1
    for index in range(size):
        flow.seed(node, (0, 3 * index), ALL, bits)
    assert flow.statistics()["fact_index_steps"] == 0
    assert flow.statistics()["condition_facts"] == size
    for index in range(size):
        assert flow.facts[node][((0, 3 * index), (0, 3 * index), ALL)] == bits


def test_shared_origin_join_does_not_widen_the_other_origins():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 6)))
    flow.seed(node, (0, 0), ALL, 1 | 2)
    flow.seed(node, (0, 1), ALL, 1)
    assert flow.facts[node] == {
        ((0, 0), (0, 0), ALL): 2,
        ((0, 0), (0, 1), ALL): 1,
    }
    assert not flow.coverage(node, (0, 1), 2, ALL)
    flow.seed(node, (0, 3), ALL, 1 | 4)
    flow.seed(node, (0, 4), ALL, 1 | 8)
    # Joining two partial masks would create three records. Keep the compact
    # two-record form and retain all three origins' exact support instead.
    assert len(flow.facts[node]) == 4
    for point in range(6):
        for bit, expected in ((1, {0, 1, 3, 4}), (2, {0}), (4, {3}), (8, {4})):
            assert bool(flow.coverage(node, (0, point), bit, ALL)) == (point in expected)


def test_joined_dominance_cancels_redundant_pending_guarded_support():
    flow = SymbolicCompletion()
    source, target = [flow.node(Family(0, (0, 6))) for _ in range(2)]
    flow.edge(source, target)
    flow.seed(source, (0, 0), ALL, 1)
    flow.propagate()
    flow.seed(source, (0, 1), ALL, 1)
    flow.seed(source, (0, 2), CountDomain(2, 0, minimum=0), 1)
    flow.seed(source, (0, 2), ALL, 1)
    # The retained [0,2] includes already-sent 0 and still-pending 1/2. The
    # separate guarded point 2 is redundant even without a pending [0,2] key.
    assert flow.facts[source] == {((0, 0), (0, 2), ALL): 1}
    assert flow.delta[source] == {
        ((0, 1), (0, 1), ALL): 1,
        ((0, 2), (0, 2), ALL): 1,
    }
    flow.propagate()
    assert flow.transfers == 3
    assert flow.facts[target] == {((0, 0), (0, 2), ALL): 1}


def test_interval_holes_and_full_count_guards_remain_distinct():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 8)))
    even, odd = CountDomain(2, 0, minimum=0), CountDomain(2, 1, minimum=0)
    flow.seed(node, (0, 0), even, 1)
    flow.seed(node, (0, 2), even, 1)
    flow.seed(node, (0, 1), odd, 1 | 2)
    for count in range(8):
        for point in range(4):
            for bit in (1, 2):
                expected = (
                    count % 2 == 0 and point in (0, 2) and bit == 1 or count % 2 == 1 and point == 1
                )
                assert (
                    any(
                        _allows(guard, count) for guard in flow.coverage(node, (0, point), bit, ALL)
                    )
                    == expected
                )


@pytest.mark.parametrize("bounded", (False, True))
def test_affine_join_requires_connection_throughout_the_guard(bounded):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (1, -1)))
    guard = CountDomain(1, 0, minimum=2, maximum=3 if bounded else None)
    flow.seed(node, (0, 0), guard, 1)
    flow.add_fact(node, (1, -2), (1, -1), guard, 1)
    assert len(flow.facts[node]) == (1 if bounded else 2)
    for count in range(2, 9):
        for point in range(count):
            expected = (not bounded or count <= 3) and point in (0, count - 2, count - 1)
            assert (
                any(_allows(domain, count) for domain in flow.coverage(node, (0, point), 1, ALL))
                == expected
            )


@pytest.mark.parametrize("seed", range(8))
def test_interval_frontier_and_incremental_arcs_match_literal_instances(seed):
    rng = random.Random(seed)
    period = rng.choice((1, 2, 3))
    families = [Family(0, (1, -1), period, i % period, monotone=i == 1) for i in range(5)]
    families[-1] = Family(0, (1, -1), period, 4 % period, last=(1, -1))
    flow = SymbolicCompletion()
    for family in families:
        flow.node(family)
    guards = (ALL, CountDomain(2, 0, minimum=0), CountDomain(2, 1, minimum=0))
    seeds = []
    for _ in range(15):
        node = rng.randrange(4)
        point = (0, families[node].residue + period * rng.randrange(3))
        guard, bits = rng.choice(guards), rng.randrange(1, 16)
        seeds.append((node, point, guard, bits))
        flow.seed(node, point, guard, bits)
    arcs = []
    for _ in range(7):
        source, target = sorted(rng.sample(range(5), 2))
        shift = families[target].residue - families[source].residue + period * rng.randrange(2)
        guard = rng.choice(guards)
        arcs.append((source, target, shift, guard))
        flow.edge(source, target, shift, guard=guard)
        flow.propagate()
        for count in range(1, 9):
            instances = {
                (node, generation)
                for node, family in enumerate(families)
                for generation in range(count)
                if generation % period == family.residue
                and (family.last is None or generation == count - 1)
            }
            following = defaultdict(set)
            for source, target, shift, guard in arcs:
                if _allows(guard, count):
                    for node, generation in instances:
                        if node == source and (target, generation + shift) in instances:
                            following[node, generation].add((target, generation + shift))
            for node, generation in instances:
                if families[node].monotone and (node, generation + period) in instances:
                    following[node, generation].add((node, generation + period))
            for bit in (1, 2, 4, 8):
                seen = {
                    (node, point[1])
                    for node, point, guard, bits in seeds
                    if bits & bit and _allows(guard, count) and (node, point[1]) in instances
                }
                work = list(seen)
                while work:
                    for target in following[work.pop()]:
                        if target not in seen:
                            seen.add(target)
                            work.append(target)
                actual = {
                    (node, generation)
                    for node, generation in instances
                    if any(
                        _allows(guard, count)
                        for guard in flow.coverage(node, (0, generation), bit, ALL)
                    )
                }
                assert actual == seen, (seed, count, bit, actual - seen, seen - actual)

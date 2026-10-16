"""Batched fact reads preserve origin guards and match literal finite paths."""

from collections import defaultdict
from random import Random

import pytest

from xdsl_npu.completion import CountDomain
from xdsl_npu.completion_flow import ALL
from xdsl_npu.model import CompileError
from xdsl_npu.symbolic_completion import CompletionTopology, Family, SymbolicCompletion

EVEN = CountDomain(2, 0, minimum=0)
ODD = CountDomain(2, 1, minimum=1)
SHORT = CountDomain(1, 0, minimum=1, maximum=4)


@pytest.mark.parametrize("size", (1, 8, 64))
def test_m_origins_read_m_target_records_once_instead_of_m_squared(size):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, size - 1)))
    origins = tuple(1 << index for index in range(size))
    for index, bit in enumerate(origins):
        flow.add_fact(node, (0, 0), (0, index), ALL, bit)
    facts, pending, delta = dict(flow.facts[node]), list(flow.pending), dict(flow.delta[node])
    before = flow.statistics()
    batched = flow.coverage_many(node, (0, 0), iter(origins), ALL)
    after = flow.statistics()
    assert batched == [[ALL] for _ in origins]
    assert after["coverage_fact_scans"] - before["coverage_fact_scans"] == size
    assert after["coverage_result_items"] - before["coverage_result_items"] == size
    assert after["coverage_batches"] - before["coverage_batches"] == 1
    assert after["coverage_requests"] - before["coverage_requests"] == size
    scalar = [flow.coverage(node, (0, 0), bit, ALL) for bit in origins]
    assert scalar == batched
    assert flow.statistics()["coverage_fact_scans"] - after["coverage_fact_scans"] == size**2
    assert {key: value for key, value in before.items() if key.startswith("fact_")} == {
        key: value for key, value in flow.statistics().items() if key.startswith("fact_")
    }
    assert flow.facts[node] == facts and flow.pending == pending and flow.delta[node] == delta


def test_even_and_odd_fact_guards_remain_paired_with_their_own_origin_bits():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 0)))
    flow.seed(node, (0, 0), EVEN, 1)
    flow.seed(node, (0, 0), ODD, 2)
    first, second = flow.coverage_many(node, (0, 0), (1, 2), ALL)
    assert first == [EVEN] and second == [ODD]
    for count in range(10):
        assert any(domain.contains(count) for domain in first) == (count % 2 == 0)
        assert any(domain.contains(count) for domain in second) == (count % 2 == 1)
    # Existing scalar masks mean ANY matching origin, including a multi-bit mask.
    assert flow.coverage(node, (0, 0), 3, ALL) == [EVEN, ODD]
    assert flow.coverage(node, (0, 0), 0, ALL) == []
    assert flow.coverage_many(node, (0, 0), (2, 1), EVEN) == [[], [EVEN]]


def test_repeated_origins_return_fresh_lists_without_sharing_across_batches_or_flows():
    topology = CompletionTopology()
    node = topology.node(Family(0, (0, 1)))
    flow, independent = [SymbolicCompletion(topology=topology) for _ in range(2)]
    flow.seed(node, (0, 0), EVEN, 1)
    first, duplicate, missing = flow.coverage_many(node, (0, 0), (1, 1, 2), ALL)
    assert first == duplicate == [EVEN] and missing == []
    first.clear()
    assert duplicate == [EVEN]
    flow.seed(node, (0, 0), ODD, 2)
    assert flow.coverage_many(node, (0, 0), (1, 2), ALL) == [[EVEN], [ODD]]
    assert flow.coverage_many(node, (0, 1), (1, 2), ALL) == [[], []]
    assert independent.coverage_many(node, (0, 0), (1, 2), ALL) == [[], []]
    before = flow.statistics()["coverage_fact_scans"]
    assert flow.coverage_many(node, (0, 0), (), ALL) == []
    assert flow.statistics()["coverage_fact_scans"] == before


@pytest.mark.parametrize("origin", (0, 3, -1, True, 1.0))
def test_batch_api_explicitly_rejects_non_single_bit_origins(origin):
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 0)))
    with pytest.raises(CompileError, match="single-bit origins"):
        flow.coverage_many(node, (0, 0), (origin,), ALL)


def _allows(domain, count):
    if domain.exact is not None:
        return count == domain.exact
    minimum = domain.period if domain.minimum is None else domain.minimum
    return (
        count >= minimum
        and count % domain.period == domain.residue
        and (domain.maximum is None or count <= domain.maximum)
    )


def _finite_paths(families, arcs, count, origin):
    """Literal generations and BFS, independent of interval/coverage helpers."""
    instances = {
        (node, generation)
        for node, family in enumerate(families)
        for generation in range(family.lower, family.horizon[0] * count + family.horizon[1] + 1)
        if generation % family.period == family.residue
        and (family.last is None or generation == family.last[0] * count + family.last[1])
    }
    following = defaultdict(set)
    for source, target, shift, source_point, target_point, guard in arcs:
        if not _allows(guard, count):
            continue
        for node, generation in instances:
            if node != source:
                continue
            if source_point is None:
                destination = target, generation + shift
            elif generation == source_point[0] * count + source_point[1]:
                destination = target, target_point[0] * count + target_point[1]
            else:
                continue
            if destination in instances:
                following[node, generation].add(destination)
    for node, generation in instances:
        family = families[node]
        following_generation = node, generation + family.period
        if family.monotone and following_generation in instances:
            following[node, generation].add(following_generation)
    node, point, guard = origin
    first = node, point[0] * count + point[1]
    reached = {first} if first in instances and _allows(guard, count) else set()
    pending = list(reached)
    while pending:
        for destination in following[pending.pop()]:
            if destination not in reached:
                reached.add(destination)
                pending.append(destination)
    return reached


@pytest.mark.parametrize("seed", range(4))
def test_batch_guards_equal_independent_finite_bfs_for_every_requested_origin(seed):
    rng = Random(seed)
    period = rng.choice((1, 2))
    families = [
        Family(0, (1, -1), period, index % period, monotone=index == 1) for index in range(4)
    ]
    families[-1] = Family(0, (1, -1), period, 3 % period, last=(1, -1))
    flow = SymbolicCompletion()
    for family in families:
        flow.node(family)
    arcs = []
    for source in range(4):
        for target in range(source + 1, 4):
            shift = families[target].residue - families[source].residue + period * rng.randrange(2)
            guard = rng.choice((ALL, EVEN, ODD, SHORT))
            flow.edge(source, target, shift, guard=guard)
            arcs.append((source, target, shift, None, None, guard))
    flow.edge(1, 2, source_point=(1, -1), target_point=(0, families[2].residue), guard=ODD)
    arcs.append((1, 2, 0, (1, -1), (0, families[2].residue), ODD))
    origins = ((0, (0, 0), EVEN), (1, (0, 1), ODD), (0, (1, -1), ALL))
    for index, (node, point, guard) in enumerate(origins):
        flow.seed(node, point, guard, 1 << index)
    flow.propagate()
    expected = {
        (count, index): _finite_paths(families, arcs, count, origin)
        for count in range(9)
        for index, origin in enumerate(origins)
    }
    for node in range(4):
        for point in ((0, 0), (0, 1), (1, -1)):
            for needed in (ALL, EVEN, ODD, SHORT, CountDomain(1, 0, exact=2)):
                batch = flow.coverage_many(node, point, (1, 2, 4, 1), needed)
                for index, result in zip((0, 1, 2, 0), batch, strict=True):
                    assert result == flow.coverage(node, point, 1 << index, needed)
                    for count in range(9):
                        target = node, point[0] * count + point[1]
                        assert any(_allows(piece, count) for piece in result) == (
                            _allows(needed, count) and target in expected[count, index]
                        ), (seed, count, node, point, index, needed)

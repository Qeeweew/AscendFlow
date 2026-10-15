"""Symbolic generation intervals versus an independent finite instance graph."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion_flow import ALL
from xdsl_npu.symbolic_completion import Family, SymbolicCompletion


def value(expression, count):
    return expression[0] * count + expression[1]


@pytest.mark.parametrize("seed", range(12))
def test_symbolic_intervals_equal_finite_native_paths_with_prefix_carries(seed):
    rng = random.Random(seed)
    period = rng.choice((1, 2, 3))
    flow = SymbolicCompletion()
    families = [
        Family(
            0,
            (1, -1),
            period,
            rng.randrange(period),
            None if i == 0 or i % 3 else (1, -1),
            i == 0 or i % 3 == 1,
        )
        for i in range(6)
    ]
    for family in families:
        flow.node(family)
    offsets = [rng.randrange(3) for _ in families]
    arcs = []
    for source in range(len(families)):
        for target in range(len(families)):
            # Every cycle passes the explicit monotone prefix 0. Arbitrary
            # unary automata without such a carry are outside this contract.
            if (source < target or target == 0) and rng.random() < 0.45:
                shift = (
                    period * (int(source >= target) + offsets[source] - offsets[target])
                    + families[target].residue
                    - families[source].residue
                )
                flow.edge(source, target, shift)
                arcs.append((source, target, shift))
    origins = [(1, (0, families[1].residue)), (4, (1, -1))]
    for bit, (node, point) in enumerate(origins):
        flow.seed(node, point, ALL, 1 << bit)
    flow.propagate()
    for count in range(14):
        instances = {
            (node, g)
            for node, family in enumerate(families)
            for g in range(family.lower, value(family.horizon, count) + 1)
            if g % family.period == family.residue
            and (family.last is None or g == value(family.last, count))
        }
        following = defaultdict(set)
        for source, target, shift in arcs:
            for node, g in instances:
                if node == source and (target, g + shift) in instances:
                    following[node, g].add((target, g + shift))
        for node, g in instances:
            family = families[node]
            if family.monotone and (node, g + period) in instances:
                following[node, g].add((node, g + period))
        for bit, (node, point) in enumerate(origins):
            origin = node, value(point, count)
            seen = {origin} if origin in instances else set()
            todo = list(seen)
            while todo:
                for target in following[todo.pop()]:
                    if target not in seen:
                        seen.add(target)
                        todo.append(target)
            actual = {
                (node, g)
                for node, g in instances
                if any(
                    bits & (1 << bit)
                    and guard.contains(count)
                    and value(lo, count) <= g <= value(hi, count)
                    for (lo, hi, guard), bits in flow.facts[node].items()
                )
            }
            assert actual == seen, (seed, count, bit, actual - seen, seen - actual)


def test_an_individual_set_effect_is_not_a_monotone_native_prefix():
    flow = SymbolicCompletion()
    effect = flow.node(Family(0, (1, -1)))
    consumer = flow.node(Family(0, (1, -1)))
    flow.edge(effect, consumer)
    flow.seed(effect, (0, 0), ALL, 1)
    flow.propagate()
    assert flow.coverage(consumer, (0, 0), 1, ALL)
    assert not flow.coverage(consumer, (0, 1), 1, ALL)


def test_fixed_source_shift_to_tail_keeps_each_affine_choice_branch_independent():
    flow = SymbolicCompletion()
    source = flow.node(Family(0, (1, -1)))
    tail = flow.node(Family(0, (1, -1), last=(1, -1)))
    flow.edge(source, tail, 1)
    flow.seed(source, (0, 0), ALL, 1)
    flow.propagate()
    coverage = flow.coverage(tail, (1, -1), 1, ALL)
    # The actual edge reaches generation 1, which is the tail only at N=2.
    # A singleton binding in the N>=2 branch must not rewrite the lower
    # endpoint used by the sibling N<=1 horizon branch.
    for count in range(8):
        assert any(domain.contains(count) for domain in coverage) == (count == 2)


def test_boundary_link_changes_clock_only_at_its_exact_source_milestone():
    flow = SymbolicCompletion()
    prefix = flow.node(Family(0, (1, -1), monotone=True))
    boundary = flow.node(Family(0, (0, 0)))
    destination = flow.node(Family(0, (0, 0)))
    flow.seed(prefix, (0, 2), ALL, 1)
    flow.edge(prefix, boundary, source_point=(1, -1), target_point=(0, 0))
    flow.edge(boundary, destination)
    flow.propagate()
    guards = flow.coverage(destination, (0, 0), 1, ALL)
    for count in range(10):
        assert any(g.contains(count) for g in guards) == (count >= 3)


def test_boundary_positions_do_not_create_head_or_tail_instruction_windows():
    from examples.gemm.reduction import reduction_gemm
    from xdsl_npu import compile
    from xdsl_npu.completion import CountDomain, Position, Requirement, Site
    from xdsl_npu.loop_sync import verify_loop
    from xdsl_npu.region_completion import RegionCompletion

    artifact = compile(reduction_gemm)
    _, analysis = verify_loop(artifact.plan, _details=True)
    loop, _, streams, graph, _ = analysis
    first = next(
        c
        for stream in streams.values()
        for c in stream
        if c.kind == "task" and not c.task.last and c.task.pipe == "MTE2"
    )
    dimensions = []
    for offset in (0, 2, 1_000_000):
        source = Site((first.actor, first.task.uid), Position(False, offset), "end", region="child")
        target = Site((first.actor, first.task.uid), Position(True, 0), "prefix", region="child")
        requirement = Requirement(
            source, target, CountDomain(1, 0, minimum=offset + 1), "retirement prefix"
        )
        proof = RegionCompletion({"child": loop}, {"child": analysis}, (requirement,), calls=(0,))
        assert proof.covered(requirement)
        dimensions.append((len(proof.flow.edges), proof.flow.edge_count))
        assert len(proof.flow.edges) < len(graph.nodes) + 20
        assert proof.flow.transfers < 2000
    assert len(set(dimensions)) == 1

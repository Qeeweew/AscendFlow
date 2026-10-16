"""The demanded fixed-child engine preserves physical point reachability."""

import random
from collections import deque

import pytest

from xdsl_npu.child_transfer_reach import LocalTransfer
from xdsl_npu.completion import CountDomain
from xdsl_npu.completion_flow import ALL
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.symbolic_completion import CompletionTopology, Family, SymbolicCompletion


def _binding(node, generation):
    return node, (0, generation)


def _literal(topology, source, target):
    """Independent finite instances, including only certified prefix carries."""
    active = set()
    for node, family in enumerate(topology.families):
        for generation in range(family.lower, family.horizon[1] + 1):
            if (
                generation % family.period == family.residue
                and (family.last is None or generation == family.last[1])
            ):
                active.add((node, generation))
    start, finish = (source[0], source[1][1]), (target[0], target[1][1])
    if start not in active or finish not in active:
        return False
    seen, pending = {start}, deque([start])
    while pending:
        node, generation = pending.popleft()
        if (node, generation) == finish:
            return True
        family = topology.families[node]
        following = []
        if family.monotone:
            following.append((node, generation + family.period))
        for target_node, shift, source_point, target_point, guard in topology.edges[node]:
            assert guard == ALL
            if source_point is None:
                following.append((target_node, generation + shift))
            elif generation == source_point[1]:
                following.append((target_node, target_point[1]))
        for point in following:
            if point in active and point not in seen:
                seen.add(point)
                pending.append(point)
    return False


def test_native_capture_does_not_make_ordinary_end_a_monotone_prefix():
    topology = CompletionTopology()
    end = topology.node(Family(0, (0, 7)))
    capture = topology.node(Family(0, (0, 7), monotone=True))
    set_effect = topology.node(Family(0, (0, 7)))
    target = topology.node(Family(0, (0, 7)))
    topology.edge(end, capture)
    topology.edge(capture, set_effect)
    topology.edge(set_effect, target)
    transfer = LocalTransfer(topology)
    assert transfer.reachable(
        _binding(end, 2), (_binding(end, 3), _binding(capture, 7), _binding(target, 7))
    ) == (False, True, True)
    assert not transfer.covered(_binding(set_effect, 2), _binding(target, 3))


def test_batch_sources_and_repeated_targets_use_one_local_flow():
    topology = CompletionTopology()
    nodes = [topology.node(Family(0, (0, 4))) for _ in range(4)]
    topology.edge(nodes[0], nodes[2])
    topology.edge(nodes[1], nodes[2])
    topology.edge(nodes[2], nodes[3])
    transfer = LocalTransfer(topology)
    first, second, last = (_binding(nodes[index], 1) for index in (0, 1, 3))
    assert transfer.predecessors((first, second, first), last) == (True, True, True)
    initial = transfer.statistics()
    assert initial["local_origins"] == 2
    assert initial["local_propagation_passes"] == 1
    assert initial["local_source_reuses"] == 1
    assert initial["local_target_reads"] == 1
    assert transfer.reachable(first, (last, last)) == (True, True)
    final = transfer.statistics()
    assert final["local_builds"] == 1
    assert final["local_propagation_passes"] == 1
    assert final["fact_transfers"] == initial["fact_transfers"]
    assert final["coverage_fact_scans"] == initial["coverage_fact_scans"]
    assert final["local_target_reuses"] == 2


def test_new_source_refreshes_only_demanded_target_masks():
    topology = CompletionTopology()
    left, right, target = [topology.node(Family(0, (0, 1))) for _ in range(3)]
    topology.edge(left, target)
    topology.edge(right, target)
    transfer = LocalTransfer(topology)
    assert transfer.covered(_binding(left, 0), _binding(target, 0))
    assert transfer.covered(_binding(right, 0), _binding(target, 0))
    assert transfer.predecessors(
        (_binding(left, 0), _binding(right, 0)), _binding(target, 0)
    ) == (True, True)
    assert transfer.statistics()["local_propagation_passes"] == 2


def test_constant_boundary_bindings_and_inactive_identity():
    topology = CompletionTopology()
    source = topology.node(Family(0, (0, 7), period=2, residue=0, monotone=True))
    last = topology.node(Family(0, (0, 7), last=(0, 5)))
    target = topology.node(Family(0, (0, 0)))
    topology.edge(source, last, source_point=(0, 6), target_point=(0, 5))
    topology.edge(last, target, source_point=(0, 5), target_point=(0, 0))
    transfer = LocalTransfer(topology)
    assert transfer.covered(_binding(source, 2), _binding(target, 0))
    assert not transfer.covered(_binding(source, 1), _binding(target, 0))
    assert transfer.covered(_binding(last, 5), _binding(last, 5))
    assert not transfer.covered(_binding(last, 4), _binding(last, 4))
    assert transfer.reachable(_binding(source, 0), ()) == ()
    assert transfer.predecessors((), _binding(target, 0)) == ()


def test_topology_builder_mutation_cannot_change_owned_transfer():
    topology = CompletionTopology()
    source, target = [topology.node(Family(0, (0, 3))) for _ in range(2)]
    transfer = LocalTransfer(topology)
    topology.edge(source, target)
    topology.node(Family(0, (0, 3)))
    assert not transfer.covered(_binding(source, 0), _binding(target, 0))
    assert transfer.statistics()["topology_nodes"] == 2
    assert transfer.statistics()["topology_edges"] == 0


@pytest.mark.parametrize("count", (1, 1000, 1_000_000_000))
def test_long_fixed_span_keeps_constant_local_analysis_size(count):
    topology = CompletionTopology()
    source = topology.node(Family(0, (0, count - 1)))
    prefix = topology.node(Family(0, (0, count - 1), monotone=True))
    target = topology.node(Family(0, (0, 0)))
    topology.edge(source, prefix)
    topology.edge(prefix, target, source_point=(0, count - 1), target_point=(0, 0))
    transfer = LocalTransfer(topology)
    assert transfer.covered(_binding(source, 0), _binding(target, 0))
    work = transfer.statistics()
    assert work["condition_facts"] == 3
    assert work["fact_transfers"] == 2
    assert work["local_origins"] == 1


@pytest.mark.parametrize("period", (1, 2, 3))
def test_random_fixed_intervals_match_independent_literal_instances(period):
    rng = random.Random(9102201 + period)
    for _ in range(8):
        topology = CompletionTopology()
        for index in range(7):
            topology.node(
                Family(
                    0,
                    (0, 6),
                    period=period,
                    residue=index % period,
                    monotone=index % 3 == 0,
                    last=(0, index % 7) if index % 4 == 2 else None,
                )
            )
        for source in range(7):
            for target in range(7):
                if rng.random() < 0.13:
                    shift = (
                        topology.families[target].residue
                        - topology.families[source].residue
                        + period * rng.randrange(-1, 2)
                    )
                    topology.edge(source, target, shift)
        transfer = LocalTransfer(topology)
        bindings = tuple(_binding(node, point) for node in range(7) for point in range(7))
        for source in rng.sample(bindings, 9):
            targets = tuple(rng.sample(bindings, 9))
            assert transfer.reachable(source, targets) == tuple(
                _literal(topology, source, target) for target in targets
            )
        target = rng.choice(bindings)
        assert transfer.predecessors(bindings, target) == tuple(
            _literal(topology, source, target) for source in bindings
        )


def test_demanded_answers_match_symbolic_native_flow_without_endpoint_pairs():
    topology = CompletionTopology()
    nodes = [topology.node(Family(0, (0, 8), monotone=index == 2)) for index in range(5)]
    topology.edge(nodes[0], nodes[1], 2)
    topology.edge(nodes[1], nodes[2], -1)
    topology.edge(nodes[2], nodes[3], 1)
    topology.edge(nodes[3], nodes[4], source_point=(0, 7), target_point=(0, 3))
    sources = (_binding(nodes[0], 0), _binding(nodes[1], 5), _binding(nodes[4], 3))
    target = _binding(nodes[4], 3)
    reference = SymbolicCompletion(topology=topology)
    for index, source in enumerate(sources):
        reference.seed(*source, ALL, 1 << index)
    reference.propagate()
    transfer = LocalTransfer(topology)
    assert transfer.predecessors(sources, target) == tuple(
        bool(reference.coverage(*target, 1 << index, ALL)) for index in range(len(sources))
    )
    assert transfer.statistics()["local_origins"] == len(sources)


@pytest.mark.parametrize("bad", ("family", "guard", "arc", "binding"))
def test_nonconstant_or_guarded_inputs_are_rejected_explicitly(bad):
    topology = CompletionTopology()
    source = topology.node(Family(0, (1, -1) if bad == "family" else (0, 3)))
    target = topology.node(Family(0, (0, 3)))
    if bad == "guard":
        topology.edge(source, target, guard=CountDomain(2, 0, minimum=0))
    elif bad == "arc":
        topology.edge(source, target, source_point=(1, -1), target_point=(0, 0))
    with pytest.raises(CompileError, match="local transfer needs"):
        transfer = LocalTransfer(topology)
        transfer.covered((source, (1, -1)), _binding(target, 0))

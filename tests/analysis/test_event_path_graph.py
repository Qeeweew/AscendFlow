"""Finite conditional endpoint graphs, without operator/resource compilation."""

from dataclasses import dataclass, replace

import pytest
from xdsl.dialects import arith
from xdsl.ir import Operation

from tests.analysis.test_guard_cover import replay
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_paths import prove_event_path_cover
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.scalar import Expr, literal
from xdsl_npu.target.ascend910b import Actor, Engine

MODE = Expr("argument", ("mode",))
EMPTY = ActionGuard()


@dataclass(frozen=True)
class _Occurrence:
    operation: Operation
    participant: Actor = Actor.AIV0


def notification(name, source=Engine.MTE2, destination=Engine.S):
    def point(placement):
        return BoundPlacement(
            _Occurrence(arith.ConstantOp.from_int_and_width(0, 64)),
            placement,
            EMPTY,
            (),
            (),
            (),
            (),
        )

    return LogicalNotification(
        name,
        (point(Placement.AFTER),),
        (point(Placement.BEFORE),),
        source,
        destination,
        NotificationKind.LOCAL_EVENT,
        (),
        None,
        literal(0),
        1,
        0,
    )


class _Domains:
    """Established endpoint protocol with explicitly supplied queue conditions."""

    def __init__(self, following):
        self.following = following
        self.work = ResourceWork()
        self.arithmetic = IntegerFacts({})

    def _following_conditions(self, current, consume, peer, source):
        self.work.lifetime_interfaces += 1
        return self.following(current, consume, peer, source)

    def _gate(self, peer):
        return peer.mechanism is NotificationKind.CROSS_EVENT or peer.destination_engine is Engine.S


def prove(domains, target, candidates):
    return prove_event_path_cover(
        domains, target, target.consumers[0], target.sources[0], (), candidates
    )


def test_future_branches_without_a_return_path_do_not_consume_graph_state_or_edges():
    target = notification("target")
    future = tuple(notification(f"future_{index}") for index in range(300))
    ranks = {peer: index for index, peer in enumerate(future)}

    def following(current, consume, peer, source):
        if peer is target:
            return ()  # The reuse target precedes every future interface.
        if current is target or ranks[current] < ranks[peer]:
            return (EMPTY,)
        return ()

    domains = _Domains(following)
    # Duplicate input instances still have one canonical endpoint/query each.
    candidates = tuple((peer, peer.sources, peer.consumers) for peer in future)
    assert prove(domains, target, candidates + candidates) is None
    assert domains.work.path_states == domains.work.path_guard_states == 1
    assert domains.work.path_edges == 0
    assert domains.work.path_edge_queries == 301


@pytest.mark.parametrize("missing_peer", (False, True))
def test_partial_direct_cover_needs_the_other_peers_conditional_path(missing_peer):
    target = notification("target")
    peer = notification("peer", Engine.V, Engine.MTE2)
    direct = ActionGuard((Comparison(MODE, "==", literal(0)),))
    indirect = ActionGuard((Comparison(MODE, "!=", literal(0)),))

    def following(current, consume, following, source):
        if current is peer:
            return (EMPTY,)
        return (direct,) if following is target else (indirect,)

    domains = _Domains(following)
    candidates = () if missing_peer else ((peer, peer.sources, peer.consumers),)
    proof = prove(domains, target, candidates)
    if missing_peer:
        assert proof is None
        return
    assert proof is not None
    for mode in (0, 1, 7):
        path = proof.paths[replay(proof.cover, {"mode": mode})]
        assert len(path.steps) == (1 if mode == 0 else 2)
        assert path.steps[0].controller
        assert path.steps[-1].notification is None
        if mode:
            assert path.steps[0].notification is peer
            assert path.steps[0].source is peer.sources[0]
            assert path.steps[-1].consume is peer.consumers[0]
        assert all(step.consume.actor is step.source.actor for step in path.steps)


def test_one_endpoint_with_distinct_guards_retains_both_conditional_instances():
    target = notification("target")
    peer = notification("peer", Engine.V, Engine.MTE2)
    candidates = tuple(
        (
            peer,
            tuple(replace(point, guard=guard) for point in peer.sources),
            tuple(replace(point, guard=guard) for point in peer.consumers),
        )
        for guard in (
            ActionGuard((Comparison(MODE, "==", literal(0)),)),
            ActionGuard((Comparison(MODE, "!=", literal(0)),)),
        )
    )

    def following(current, consume, following, source):
        return () if current is target and following is target else (EMPTY,)

    domains = _Domains(following)
    proof = prove(domains, target, candidates)
    assert proof is not None
    assert domains.work.path_states == 3
    assert domains.work.path_edges == 2
    for mode, candidate in ((0, candidates[0]), (1, candidates[1]), (7, candidates[1])):
        path = proof.paths[replay(proof.cover, {"mode": mode})]
        assert len(path.steps) == 2
        assert path.steps[0].source == candidate[1][0]
        assert path.steps[-1].consume == candidate[2][0]
        assert all(step.consume.guard.evaluate({"mode": mode}) for step in path.steps)

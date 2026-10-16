"""Conditional reuse paths through the real nested two-slot interfaces."""

from dataclasses import dataclass

import pytest

from tests.analysis.test_event_parameter_cases import nested_slot_source
from tests.analysis.test_guard_cover import replay
from tests.support.structured import selected
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_paths import EventPathCandidate, prove_event_path_cover
from xdsl_npu.analysis.hazards import HazardKind
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Engine


@dataclass(frozen=True)
class _SlotReuse:
    domains: EventDomains
    notification: LogicalNotification
    old: BoundPlacement
    goal: BoundPlacement
    assumptions: tuple[Comparison, ...]
    peers: tuple[LogicalNotification, ...]
    candidates: tuple[EventPathCandidate, ...]

    def values(self, m, p, g):
        parent, child = self.domains.control.loops.values()
        return {
            "n": 3,
            "m": m,
            "$group": 0,
            "$sid": 0,
            parent.ordinal.args[0]: p,
            child.ordinal.args[0]: g,
        }


@pytest.fixture(scope="module")
def slot_reuse():
    domains = EventDomains(selected(nested_slot_source(2)), ResourceWork())
    parent, child = domains.control.loops
    ((notification, case),) = tuple(
        (notification, case)
        for notification in domains.notifications
        if notification.mechanism is NotificationKind.LOCAL_EVENT
        and notification.source_engine is Engine.MTE2
        and notification.destination_engine is Engine.V
        and domains._clocks(notification) == (parent, child)
        for case in domains.cases[notification]
        if not case.empty
        and len(case.windows) == 2
        and all(window.period == 1 for window in case.windows)
        and case.window(child).upper is None
    )
    assert notification.requirements
    assert all(need.kind is HazardKind.RAW for need in notification.requirements)
    ((phase, delta),) = case.window(child).reuse_phases(2)
    assert delta == 2
    shifted = {
        domains.control.loops[child].ordinal: expression(
            "add", domains.control.loops[child].ordinal, literal(delta)
        )
    }
    old_sources = tuple(domains._instance(p, {}, case=case) for p in notification.sources)
    old_consumers = tuple(domains._instance(p, {}, case=case) for p in notification.consumers)
    next_sources = tuple(domains._instance(p, shifted, case=case) for p in notification.sources)
    next_consumers = tuple(domains._instance(p, shifted, case=case) for p in notification.consumers)
    (old,), (goal,) = old_consumers, next_sources
    assumptions = tuple(
        dict.fromkeys(
            case.condition.comparisons
            + phase.comparisons
            + tuple(
                test
                for placement in old_sources + old_consumers + next_sources + next_consumers
                for test in placement.guard.comparisons
            )
        )
    )
    peers = tuple(
        sorted(
            (
                peer
                for peer in domains.notifications
                if peer.mechanism is NotificationKind.LOCAL_EVENT
                and peer.source_engine is Engine.V
                and peer.destination_engine is Engine.MTE2
                and peer.clock is child
                and peer.distance == 2
                and peer.requirements
                and all(need.kind is HazardKind.WAR for need in peer.requirements)
            ),
            key=lambda peer: min(need.cell.start for need in peer.requirements),
        )
    )
    assert len(peers) == 2
    assert [min(need.cell.start for need in peer.requirements) for peer in peers] == [0, 256]
    candidates = tuple(
        (peer, sources, consumers)
        for peer in peers
        for peer_case in domains.cases[peer]
        if not peer_case.empty
        for sources, consumers in domains._instances(peer, peer_case, child, delta)
    )
    return _SlotReuse(domains, notification, old, goal, assumptions, peers, candidates)


@pytest.mark.parametrize(
    "missing_slot", (None, 0, 1), ids=("complete", "missing-slot0", "missing-slot1")
)
def test_real_slot_reuse_requires_both_conditional_war_interfaces(slot_reuse, missing_slot):
    obligation = slot_reuse
    candidates = tuple(
        candidate
        for candidate in obligation.candidates
        if missing_slot is None or candidate[0] is not obligation.peers[missing_slot]
    )
    proof = prove_event_path_cover(
        obligation.domains,
        obligation.notification,
        obligation.old,
        obligation.goal,
        obligation.assumptions,
        candidates,
    )
    if missing_slot is not None:
        values = obligation.values(4, 0, missing_slot)
        assert all(test.evaluate(values) for test in obligation.assumptions)
        assert not any(
            all(placement.guard.evaluate(values) for placement in sources + consumers)
            for _, sources, consumers in candidates
        )
        assert proof is None
        return

    assert proof is not None
    assert proof.assumptions == obligation.assumptions
    assert proof.cover.alternatives == tuple(path.condition for path in proof.paths)
    parent, child = obligation.domains.control.loops
    for m in (3, 4, 5):
        for p in range(3):
            for g in range(m - 2):
                values = obligation.values(m, p, g)
                assert all(test.evaluate(values) for test in obligation.assumptions)
                assert any(path.condition.evaluate(values) for path in proof.paths)
                path = proof.paths[replay(proof.cover, values)]
                assert path.condition.evaluate(values)
                assert len(path.steps) == 2
                assert path.steps[0].notification is obligation.peers[(p * m + g) % 2]
                assert path.steps[1].notification is None
                for step in path.steps:
                    assert step.consume.guard.evaluate(values)
                    assert step.source.guard.evaluate(values)
                source = dict(path.steps[0].source.coordinates)
                consume = dict(path.steps[1].consume.coordinates)
                assert (source[parent].evaluate(values), source[child].evaluate(values)) == (p, g)
                assert (consume[parent].evaluate(values), consume[child].evaluate(values)) == (
                    p,
                    g + 2,
                )

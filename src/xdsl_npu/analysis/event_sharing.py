"""New allocation constraints: share finite rings using generated handbacks."""

from dataclasses import dataclass

from ..target.ascend910b import Actor, Engine
from .notification_lifetimes import HandbackRule
from .structured_protocols import NotificationKind


@dataclass(frozen=True, slots=True)
class EventSharing:
    bank: tuple[Actor, Engine, Engine]
    earlier: object
    later: object
    handbacks: tuple


def _single(domains, notification):
    cases = domains.cases[notification]
    if len(cases) != 1 or cases[0].empty or cases[0].windows:
        return None
    case = cases[0]
    root = domains.memory.module.body.block.first_op
    for point in notification.sources + notification.consumers:
        if any(not domains.control.invariant_over(v, root) for _, v in point.coordinates):
            return None
        instance = domains._instance(point, {}, case=case)
        if any(
            not domains.control.invariant_over(t.lhs, root)
            or not domains.control.invariant_over(t.rhs, root)
            for t in instance.guard.comparisons
        ):
            return None
    return case


def share_finite_events(domains, rule: HandbackRule, left, right):
    if (
        left.mechanism is not NotificationKind.LOCAL_EVENT
        or right.mechanism is not NotificationKind.LOCAL_EVENT
    ):
        return None
    bank = left.sources[0].actor, left.source_engine, left.destination_engine
    if bank != (right.sources[0].actor, right.source_engine, right.destination_engine):
        return None
    a, b = _single(domains, left), _single(domains, right)
    if a is None or b is None:
        return None
    for earlier, ec, later, lc in ((left, a, right, b), (right, b, left, a)):
        old = domains._instance(earlier.consumers[0], {}, case=ec)
        target = domains._instance(later.sources[0], {}, case=lc)
        assumptions = old.guard.comparisons + target.guard.comparisons
        handbacks = rule.establish(
            earlier, old, later, target, assumptions, loop=None, delta=0, reentry=False
        )
        if handbacks is not None:
            return EventSharing(bank, earlier, later, handbacks)
    return None

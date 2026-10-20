"""New allocation constraints: real event handbacks and aligned counter FIFO."""

from dataclasses import dataclass

from ..scalar import expression, literal
from ..target.ascend910b import Actor, Engine
from .notification_lifetimes import HandbackRule
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard


@dataclass(frozen=True, slots=True)
class EventSharing:
    bank: tuple[Actor, Engine, Engine]
    earlier: object
    later: object
    handbacks: tuple


@dataclass(frozen=True, slots=True)
class CounterSharing:
    """Two aligned FIFO messages, without merging their publications/waits.

    Each publisher's same-pipe contributions retain effect FIFO. Every receiver
    consumes one contribution from each required publisher in that order;
    a faster publisher cannot substitute for another quorum member. The
    allocator separately bounds the complete credits placed on this ID.
    """

    earlier: object
    later: object
    ordered_points: tuple


def _counter_shape(notification):
    def points(values):
        return tuple(
            (
                p.actor,
                p.placement,
                p.guard,
                p.coordinates,
                p.coordinate_guards,
                p.families,
                p.selected,
            )
            for p in sorted(values, key=lambda p: p.actor.value)
        )

    return (
        notification.source_engine,
        notification.destination_engine,
        notification.clock,
        notification.ordinal,
        notification.period,
        notification.distance,
        notification.domains,
        points(notification.sources),
        points(notification.consumers),
    )


def share_counters(domains, left, right):
    """Pack aligned complete mode-2 streams whose FIFO orders agree.

    This sufficient rule keeps identical coordinates, activation and family
    selection, including the enclosing reentry. It does not identify different
    loop clocks, use cross-pipe issue as completion, or drop an AIV member.
    """
    if any(n.mechanism is not NotificationKind.CROSS_EVENT for n in (left, right)):
        return None
    if _counter_shape(left) != _counter_shape(right):
        return None
    a, b = domains.cases[left], domains.cases[right]

    def cases(values):
        return tuple(
            (c.condition, c.substitutions, c.windows, c.projections, c.empty) for c in values
        )

    if cases(a) != cases(b):
        return None

    def ordered(earlier, ec, later, lc):
        result = []
        # EventDomains already admitted the complete mode-2 quorum. The shape
        # retains its actual actors and activation. Check each publisher and
        # receiver, so their per-source FIFO streams have the same message word
        # even when the two AIVs make progress at different speeds.
        for phase in ("sources", "consumers"):
            old = sorted(getattr(earlier, phase), key=lambda p: p.actor.value)
            new = sorted(getattr(later, phase), key=lambda p: p.actor.value)
            for before, after in zip(old, new, strict=True):
                # Equal anchors would require an explicit instruction ordering
                # decision. This rule only consumes an existing strict order.
                if before.operation is after.operation:
                    return None
                for ac, bc in zip(ec, lc, strict=True):
                    if not domains.active[earlier, ac] and not domains.active[later, bc]:
                        continue
                    if domains.active[earlier, ac] != domains.active[later, bc]:
                        return None
                    x = domains._instance(before, {}, case=ac)
                    y = domains._instance(after, {}, case=bc)
                    conditions = domains._order_conditions(x, y)
                    assumptions = x.guard.comparisons + y.guard.comparisons
                    # Only a mathematical issue relation is queried. Original
                    # endpoint arithmetic was already admitted by EventDomains.
                    if not domains.arithmetic.queries.covers(conditions, assumptions).proved:
                        return None
                result.append((before, after))
        return tuple(result)

    for earlier, ec, later, lc in ((left, a, right, b), (right, b, left, a)):
        points = ordered(earlier, ec, later, lc)
        if points is not None:
            return CounterSharing(earlier, later, points)
    return None


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


def _same_iteration_streams(domains, rule, left, right):
    """One publication per member of the same exact nested iteration domain.

    Selected boundaries and translated stages need a different correspondence.
    In particular, a finite fragment never licenses an entire cell channel.
    """
    clocks = domains.clocks[left]
    if not clocks or clocks != domains.clocks[right]:
        return None
    expected = tuple((loop, domains.control.loops[loop].ordinal) for loop in clocks)
    if any(
        point.coordinates != expected or point.families
        for notification in (left, right)
        for point in notification.sources + notification.consumers
    ):
        return None
    a, b = rule._cases(left), rule._cases(right)
    if len(a) != 1 or len(b) != 1:
        return None
    a, b = a[0], b[0]
    if a.empty or b.empty or not a.windows or a.windows != b.windows:
        return None
    if a.condition != b.condition or a.substitutions != b.substitutions:
        return None
    if tuple(window.loop for window in a.windows) != clocks:
        return None
    return a, b


def _stream_handbacks(domains, rule, earlier, ec, later, lc):
    """Order both episodes, including the last/first of each nested reentry.

    Inside an iteration, consume earlier before publishing later. At every
    possible successor iteration, consume later before publishing earlier.
    Exact equal windows make these the adjacent episodes of the merged word;
    zero/one-trip windows add no fictitious successor. Nothing is emitted.
    """
    handbacks = []

    def connect(old_channel, old, target_channel, target, condition, loop, delta, reentry):
        assumptions = tuple(
            dict.fromkeys(condition.comparisons + old.guard.comparisons + target.guard.comparisons)
        )
        result = rule.establish(
            old_channel,
            old,
            target_channel,
            target,
            assumptions,
            loop=loop,
            delta=delta,
            reentry=reentry,
        )
        if result is None:
            return False
        handbacks.extend(result)
        return True

    old = domains._instance(earlier.consumers[0], {}, case=ec)
    target = domains._instance(later.sources[0], {}, case=lc)
    if not connect(earlier, old, later, target, ec.condition, ec.windows[-1].loop, 0, False):
        return None
    for index, window in enumerate(ec.windows):
        if window.upper is not None and window.upper <= 1:
            continue
        reentry = index < len(ec.windows) - 1
        for phase, delta in window.reuse_phases(1):
            if reentry:
                _, old_waits = list(domains._instances(later, lc, window.loop, 0))[-1]
                next_sources, _ = next(domains._instances(earlier, ec, window.loop, delta))
                old, target = old_waits[0], next_sources[0]
            else:
                ordinal = domains.control.loops[window.loop].ordinal
                old = domains._instance(later.consumers[0], {}, case=lc)
                target = domains._instance(
                    earlier.sources[0],
                    {ordinal: expression("add", ordinal, literal(delta))},
                    case=ec,
                )
            condition = ActionGuard(ec.condition.comparisons + phase.comparisons)
            if not connect(later, old, earlier, target, condition, window.loop, delta, reentry):
                return None
    return tuple(handbacks)


def share_events(domains, rule: HandbackRule, left, right):
    if (
        left.mechanism is not NotificationKind.LOCAL_EVENT
        or right.mechanism is not NotificationKind.LOCAL_EVENT
    ):
        return None
    bank = left.sources[0].actor, left.source_engine, left.destination_engine
    if bank != (right.sources[0].actor, right.source_engine, right.destination_engine):
        return None
    a, b = _single(domains, left), _single(domains, right)
    if a is not None and b is not None:
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
    cases = _same_iteration_streams(domains, rule, left, right)
    if cases is not None:
        a, b = cases
        for earlier, ec, later, lc in ((left, a, right, b), (right, b, left, a)):
            handbacks = _stream_handbacks(domains, rule, earlier, ec, later, lc)
            if handbacks is not None:
                return EventSharing(bank, earlier, later, handbacks)
    return None

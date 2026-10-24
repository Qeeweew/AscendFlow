"""New allocation constraints: real event handbacks and aligned counter FIFO."""

from dataclasses import dataclass

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
class CellPhaseSharing:
    """Projection of an established physical-cell lifetime onto its members.

    Splitting integer allocation never resets that word or its open uses.
    The exact original cell channel, ownership and nonduplicate members stay
    the premise; this supplies no relation to another logical credit.
    """

    credit: object
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class CounterSharing:
    """Pairwise order premises, not a certificate for a whole FIFO group.

    Packing imposes same-ID effect FIFO on each publisher's same pipe. A
    CounterGroup must additionally establish the complete word, progress and
    per-publisher backlog before allocation may use these pairwise premises.
    Publications/waits and every required quorum member remain distinct.
    """

    earlier: object
    later: object
    ordered_points: tuple


@dataclass(frozen=True, slots=True)
class CounterReuse:
    """Complete mode-2 episodes separated by every receiver's consumption.

    The proof uses only required handoffs before ID assignment. Different
    credits cannot have outstanding contributions together, so their bound
    is the maximum per-credit backlog, with every publisher kept distinct.
    """

    earlier: object
    later: object
    handbacks: tuple
    rule: str


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
    coincident = _coincident_counter_points(domains, left, right)
    if coincident is not None:
        return CounterSharing(left, right, coincident)
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


def _coincident_counter_points(domains, left, right):
    """Disjoint activations at identical real send/receive occurrences.

    Each channel has already established pairing and its complete quorum.
    Their common coordinate map and identical sites put both subsets in the
    same per-actor FIFO order. Disjoint guards ensure at most one member at
    each occurrence, so this rule needs no new action-order decision.
    """
    if (left.source_engine, left.destination_engine, domains.clocks[left]) != (
        right.source_engine,
        right.destination_engine,
        domains.clocks[right],
    ):
        return None
    points = []
    for phase in ("sources", "consumers"):
        a = sorted(getattr(left, phase), key=lambda p: p.actor.value)
        b = sorted(getattr(right, phase), key=lambda p: p.actor.value)
        if len(a) != len(b):
            return None
        for before, after in zip(a, b, strict=True):

            def site(p):
                return (
                    p.operation,
                    p.actor,
                    p.placement,
                    p.coordinates,
                    p.coordinate_guards,
                    p.families,
                    p.selected,
                )

            if site(before) != site(after):
                return None
            for ac in domains.cases[left]:
                for bc in domains.cases[right]:
                    if ac.empty or bc.empty:
                        continue
                    x = domains._instance(before, {}, case=ac)
                    y = domains._instance(after, {}, case=bc)
                    if not domains.arithmetic.queries.satisfies(
                        x.guard.comparisons + y.guard.comparisons
                    ).impossible:
                        return None
            points.append((before, after))
    return tuple(points)


def _same_iteration_streams(domains, rule, left, right):
    """Exact common invocation clocks, with optional descendant bursts.

    Complete windows keep selected boundaries and unit-translated sites.
    A missing prefix is allowed only when one window contains the other;
    every handback still retains the actual member guards.
    """
    a, b = rule._cases(left), rule._cases(right)
    if len(a) != 1 or len(b) != 1:
        return None
    a, b = a[0], b[0]
    if a.empty or b.empty or not a.windows or not b.windows:
        return None
    if a.condition != b.condition or a.substitutions != b.substitutions:
        return None
    for notification, case in ((left, a), (right, b)):
        if not rule.ordered_stream(notification, case):
            return None
    common = []
    for aw, bw in zip(a.windows, b.windows):
        if aw == bw:
            common.append(aw)
            continue
        if (aw.loop, aw.count) != (bw.loop, bw.count):
            break
        if domains.arithmetic.queries.covers((aw.membership,), bw.membership.comparisons).proved:
            common.append(aw)
        elif domains.arithmetic.queries.covers((bw.membership,), aw.membership.comparisons).proved:
            common.append(bw)
        else:
            break
    return (a, b, tuple(common)) if common else None


def _handback(domains, rule, earlier, old, later, target, assumptions, **reuse):
    # Binary sharing additionally requires forward issue on its controller.
    # Cross consumption blocks the whole controller. Its transfer still keeps
    # every publisher/receiver and the complete matched quorum.
    if (
        earlier.mechanism is NotificationKind.LOCAL_EVENT
        and not domains.arithmetic.queries.covers(
            domains._order_conditions(old, target), assumptions
        ).proved
    ):
        return None
    return rule.establish(
        earlier,
        old,
        later,
        target,
        assumptions,
        **reuse,
    )


def _stream_handbacks(domains, rule, earlier, ec, later, lc, windows, *, tails=None):
    """Order both episodes, including the last/first of each nested reentry.

    Inside an iteration, consume earlier before publishing later. At every
    possible successor iteration, consume later before publishing earlier.
    Exact common windows make these the adjacent episodes of the merged word;
    FIRST/LAST consume every descendant use inside each invocation burst.
    zero/one-trip windows add no fictitious successor. Nothing is emitted.
    """
    handbacks = []
    old_tail, old_case, new_tail, new_case = tails or (earlier, ec, later, lc)

    def connect(old_channel, olds, target_channel, targets, condition, loop, delta, reentry):
        for old in olds:
            for target in targets:
                assumptions = tuple(
                    dict.fromkeys(
                        condition.comparisons + old.guard.comparisons + target.guard.comparisons
                    )
                )
                result = _handback(
                    domains,
                    rule,
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

    loop = windows[-1].loop
    _, old_waits = list(domains._instances(old_tail, old_case, loop, 0))[-1]
    next_sources, _ = next(domains._instances(later, lc, loop, 0))
    if not connect(old_tail, old_waits, later, next_sources, ec.condition, loop, 0, False):
        return None
    for index, window in enumerate(windows):
        if window.upper is not None and window.upper <= 1:
            continue
        reentry = index < len(ec.windows) - 1 or index < len(lc.windows) - 1
        for phase, delta in window.reuse_phases(1):
            _, old_waits = list(domains._instances(new_tail, new_case, window.loop, 0))[-1]
            next_sources, _ = next(domains._instances(earlier, ec, window.loop, delta))
            condition = ActionGuard(ec.condition.comparisons + phase.comparisons)
            if not connect(
                new_tail, old_waits, earlier, next_sources, condition, window.loop, delta, reentry
            ):
                return None
            if reentry:
                # Unequal contained windows can end/start with either member
                # when a descendant is absent. Cover the opposite crossing
                # too; a selected empty child supplies no invented handback.
                _, other_waits = list(domains._instances(old_tail, old_case, window.loop, 0))[-1]
                other_sources, _ = next(domains._instances(later, lc, window.loop, delta))
                if not connect(
                    old_tail, other_waits, later, other_sources, condition, window.loop, delta, True
                ):
                    return None
    return tuple(handbacks)


def _invocation_handbacks(domains, rule, earlier, later, *, tail=None):
    """Disjoint whole-invocation episodes, including every nested use.

    Exact event windows select the last consumption of the entire earlier
    member and the first publication of the entire later member. In-order
    queues put all of their other uses on the respective sides of these
    endpoints. A real consumption handback must bridge the two; source order
    or disjoint loop predicates do not establish this relation.

    This never extends a finite fragment's permission to its cell channel:
    the allocator checks every member of both complete credits separately.
    """
    tail = earlier if tail is None else tail
    a, b = rule._cases(tail), rule._cases(later)
    if not a or not b:
        return None
    handbacks = []
    for ec in a:
        if ec.empty:
            continue
        if domains.clocks[tail] and not ec.windows:
            return None
        _, waits = list(domains._instances(tail, ec, None, 0))[-1]
        for lc in b:
            if lc.empty:
                continue
            if domains.clocks[later] and not lc.windows:
                return None
            sources, _ = next(domains._instances(later, lc, None, 0))
            for old in waits:
                for target in sources:
                    assumptions = tuple(
                        dict.fromkeys(old.guard.comparisons + target.guard.comparisons)
                    )
                    result = _handback(
                        domains,
                        rule,
                        tail,
                        old,
                        later,
                        target,
                        assumptions,
                        loop=None,
                        delta=0,
                        reentry=False,
                    )
                    if result is None:
                        return None
                    handbacks.extend(result)
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
    a, b = rule.single_case(left), rule.single_case(right)
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
        a, b, windows = cases
        for earlier, ec, later, lc in ((left, a, right, b), (right, b, left, a)):
            handbacks = _stream_handbacks(domains, rule, earlier, ec, later, lc, windows)
            if handbacks is not None:
                return EventSharing(bank, earlier, later, handbacks)
    for earlier, later in ((left, right), (right, left)):
        handbacks = _invocation_handbacks(domains, rule, earlier, later)
        if handbacks is not None:
            return EventSharing(bank, earlier, later, handbacks)
    return None


def share_event_frontiers(domains, rule, left, right):
    """Separate whole event intervals using their real extremal milestones.

    A frontier is an enclosure of equal-activation, equal-coordinate local
    messages, not a synthetic notification. Its first publisher and last
    consumer can belong to different messages. Every consumption proof uses
    the real tail channel, every new publication the real head channel. The
    caller establishes enclosure before calling this function; all forward,
    next-invocation and zero/short-window obligations remain the normal rule.
    """
    if left.head is left.tail and right.head is right.tail:
        witness = share_events(domains, rule, left.head, right.head)
        if witness is None:
            return None
        return witness.earlier is left.head, witness.handbacks

    def paired_case(head, tail, case, *, family):
        a = rule._cases(head) if family else domains.cases[head]
        b = rule._cases(tail) if family else domains.cases[tail]
        # Exact activation/case correspondence is part of the frontier key.
        # Case identities remain local to their actual notifications.
        index = a.index(case)
        return b[index]

    a, b = rule.single_case(left.head), rule.single_case(right.head)
    if a is not None and b is not None:
        for forward, earlier, ec, later, lc in (
            (True, left, a, right, b),
            (False, right, b, left, a),
        ):
            tail_case = paired_case(earlier.head, earlier.tail, ec, family=False)
            old = domains._instance(earlier.tail.consumers[0], {}, case=tail_case)
            target = domains._instance(later.head.sources[0], {}, case=lc)
            handbacks = rule.establish(
                earlier.tail,
                old,
                later.head,
                target,
                old.guard.comparisons + target.guard.comparisons,
                loop=None,
                delta=0,
                reentry=False,
            )
            if handbacks is not None:
                return forward, handbacks
        return None
    for frontier in (left, right):
        if frontier.head is frontier.tail:
            continue

        def shape(cases):
            return tuple(
                (c.condition, c.substitutions, c.windows, c.projections, c.empty) for c in cases
            )

        if shape(rule._cases(frontier.head)) != shape(rule._cases(frontier.tail)):
            # Optional family precision can differ from the paired base
            # cases. Refine the enclosure instead of identifying these cases.
            return None
    cases = _same_iteration_streams(domains, rule, left.head, right.head)
    if cases is not None:
        a, b, windows = cases
        for forward, earlier, ec, later, lc in (
            (True, left, a, right, b),
            (False, right, b, left, a),
        ):
            handbacks = _stream_handbacks(
                domains,
                rule,
                earlier.head,
                ec,
                later.head,
                lc,
                windows,
                tails=(
                    earlier.tail,
                    paired_case(earlier.head, earlier.tail, ec, family=True),
                    later.tail,
                    paired_case(later.head, later.tail, lc, family=True),
                ),
            )
            if handbacks is not None:
                return forward, handbacks
    for forward, earlier, later in ((True, left, right), (False, right, left)):
        handbacks = _invocation_handbacks(
            domains, rule, earlier.head, later.head, tail=earlier.tail
        )
        if handbacks is not None:
            return forward, handbacks
    return None


def share_counter_time(domains, rule: HandbackRule, left, right):
    """Time-share matching mode-2 routes without adding completion order.

    Each receiver must consume before *every* publisher of the next episode.
    Existing cross handoffs may establish this relation across controllers;
    same-actor issue order alone cannot. Unit clock maps and exact boundaries
    keep all uses, including empty invocations and each level of reentry.
    """
    if any(n.mechanism is not NotificationKind.CROSS_EVENT for n in (left, right)):
        return None

    def route(n):
        return (
            n.source_engine,
            n.destination_engine,
            tuple(p.actor for p in n.sources),
            tuple(p.actor for p in n.consumers),
        )

    if route(left) != route(right):
        return None
    if domains._exclusive(left, right):
        return CounterReuse(left, right, (), "uniform-exclusive")
    cases = _same_iteration_streams(domains, rule, left, right)
    if cases is not None:
        a, b, windows = cases
        for earlier, ec, later, lc in ((left, a, right, b), (right, b, left, a)):
            handbacks = _stream_handbacks(domains, rule, earlier, ec, later, lc, windows)
            if handbacks is not None:
                return CounterReuse(earlier, later, handbacks, "clock-handback")
    for notification in (left, right):
        if domains.clocks[notification]:
            if not all(
                case.empty or rule.ordered_stream(notification, case)
                for case in rule._cases(notification)
            ):
                return None
        elif rule.single_case(notification) is None:
            return None
    for earlier, later in ((left, right), (right, left)):
        handbacks = _invocation_handbacks(domains, rule, earlier, later)
        if handbacks is not None:
            return CounterReuse(earlier, later, handbacks, "invocation-handback")
    return None

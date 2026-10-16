"""Event consumption and reuse from already paired physical notifications.

All functions consume EventDomains. They neither infer memory dependencies nor
allocate IDs, generate actions, or change native dispatch. Unknown lifetime
relations remain unsupported; bounded interface search is not runtime unrolling.
"""

from collections import deque

from ..scalar import expression, literal
from .event_contracts import (
    CROSS_BACKLOG,
    LOCAL_CAPACITY,
    CaseLease,
    LeaseKind,
    LeaseStep,
    LeaseWitness,
    StructuredEventLease,
)
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard

MAX_INTERFACE_STEPS = 128


def bridge(domains, notification, old, goal, loop, delta, assumptions, candidates):
    domains.work.lease_searches += 1
    queue = deque([(notification, old, ())])
    visited = set()
    while queue:
        current, consume, steps = queue.popleft()
        key = (id(current), consume.actor, consume.coordinates)
        if key in visited:
            continue
        visited.add(key)
        if domains._follows(current, consume, notification, goal, assumptions):
            final = LeaseStep(consume, goal, None, domains._gate(current))
            chain = steps + (final,)
            domains.work.witness_steps += len(chain)
            return chain
        if len(steps) >= MAX_INTERFACE_STEPS:
            continue
        for peer, sources, consumers in candidates:
            if peer is notification:
                continue
            for source in sources:
                if domains._follows(
                    current, consume, peer, source, assumptions
                ) and domains._exists(sources + consumers, assumptions):
                    step = LeaseStep(consume, source, peer, domains._gate(current))
                    queue.extend((peer, destination, steps + (step,)) for destination in consumers)
    return None


def roundtrip(domains, notification, case, window, width):
    """Prove the nearest same-ID reuse in every member phase of this case."""
    loop = window.loop
    logical = domains.control.loops[loop]
    old_sources = tuple(domains._instance(p, {}, case=case) for p in notification.sources)
    old_consumers = tuple(domains._instance(p, {}, case=case) for p in notification.consumers)
    witnesses = []
    for phase, delta in window.reuse_phases(width):
        shifted = {logical.ordinal: expression("add", logical.ordinal, literal(delta))}
        next_sources = tuple(domains._instance(p, shifted, case=case) for p in notification.sources)
        next_consumers = tuple(
            domains._instance(p, shifted, case=case) for p in notification.consumers
        )
        condition = ActionGuard(
            tuple(dict.fromkeys(case.condition.comparisons + phase.comparisons))
        )
        assumptions = tuple(
            dict.fromkeys(
                condition.comparisons
                + tuple(
                    test
                    for p in old_sources + old_consumers + next_sources + next_consumers
                    for test in p.guard.comparisons
                )
            )
        )
        candidates = []
        for peer in domains.notifications:
            if peer.mechanism is NotificationKind.PIPE_FENCE:
                continue
            for peer_case in domains.cases[peer]:
                if peer_case.empty or not any(w.loop is loop for w in peer_case.windows):
                    continue
                # A bounded event/slot-distance search, never a loop-count scan.
                for offset in range(-min(8, delta), delta + 1):
                    for sources, consumers in domains._instances(peer, peer_case, loop, offset):
                        candidates.append((peer, sources, consumers))
        for old in old_consumers:
            for goal in next_sources:
                chain = bridge(
                    domains, notification, old, goal, loop, delta, assumptions, candidates
                )
                if chain is None:
                    return None
                witnesses.append(
                    LeaseWitness(
                        LeaseKind.CONTROLLER if not chain[:-1] else LeaseKind.QUEUE_ROUNDTRIP,
                        loop,
                        delta,
                        old,
                        goal,
                        chain,
                        condition=condition,
                    )
                )
    return tuple(witnesses)


def reentry(domains, notification, case, window, width=1):
    """Return from the last child use to the next same-ID real invocation."""
    loop = window.loop
    witnesses = []
    for phase, delta in window.reuse_phases(width):
        candidates = []
        horizon = delta + max(
            (abs(peer.distance or 0) for peer in domains.notifications), default=0
        )
        for peer in domains.notifications:
            if peer.mechanism is NotificationKind.PIPE_FENCE:
                continue
            for peer_case in domains.cases[peer]:
                if peer_case.empty:
                    continue
                for offset in range(horizon + 1):
                    for sources, consumers in domains._instances(peer, peer_case, loop, offset):
                        candidates.append((peer, sources, consumers))
        old_sources, old_consumers = list(domains._instances(notification, case, loop, 0))[-1]
        next_sources, next_consumers = next(domains._instances(notification, case, loop, delta))
        condition = ActionGuard(
            tuple(dict.fromkeys(case.condition.comparisons + phase.comparisons))
        )
        assumptions = tuple(
            dict.fromkeys(
                condition.comparisons
                + tuple(
                    test
                    for p in old_sources + old_consumers + next_sources + next_consumers
                    for test in p.guard.comparisons
                )
            )
        )
        for old in old_consumers:
            for goal in next_sources:
                chain = bridge(
                    domains, notification, old, goal, loop, delta, assumptions, candidates
                )
                if chain is None:
                    return None
                witnesses.append(
                    LeaseWitness(
                        LeaseKind.INVOCATION,
                        loop,
                        delta,
                        old,
                        goal,
                        chain,
                        condition=condition,
                    )
                )
    return tuple(witnesses)


def _establish_case(domains, notification, case):
    windows = case.windows
    capacity = (
        CROSS_BACKLOG if notification.mechanism is NotificationKind.CROSS_EVENT else LOCAL_CAPACITY
    )
    width, rings, inside, reentries = 1, [], [], []
    consume = domains._instance(notification.consumers[0], {}, case=case)
    source = domains._instance(notification.sources[0], {}, case=case)
    if not windows:
        inside.append(
            LeaseWitness(
                LeaseKind.FINITE,
                None,
                0,
                consume,
                source,
                upper_uses=1,
                condition=case.condition,
            )
        )
    for index in reversed(range(len(windows))):
        window = windows[index]
        for candidate in range(1, capacity // width + 1):
            if window.upper is not None and window.upper <= candidate:
                witnesses = (
                    LeaseWitness(
                        LeaseKind.FINITE,
                        window.loop,
                        0,
                        consume,
                        source,
                        upper_uses=window.upper,
                        condition=case.condition,
                    ),
                )
            elif index == len(windows) - 1:
                witnesses = roundtrip(domains, notification, case, window, candidate)
            else:
                witnesses = reentry(domains, notification, case, window, candidate)
            if witnesses is not None:
                break
        else:
            detail = (
                "no real consume-to-next-Set queue/controller path within the supported binary/counter capacity"
                if index == len(windows) - 1
                else "finite child uses do not establish consumption before an ID's next real invocation within the supported capacity"
            )
            domains.fail(notification, detail)
        width *= candidate
        rings.insert(0, (window, candidate))
        (inside if index == len(windows) - 1 else reentries).extend(witnesses)
    ordinal = literal(0)
    for window, ring in rings:
        local = (
            literal(0)
            if window.single or ring == 1
            else expression("rem", window.rank(), literal(ring))
        )
        ordinal = expression("add", expression("mul", ordinal, literal(ring)), local)
    cross = notification.mechanism is NotificationKind.CROSS_EVENT
    return CaseLease(
        case,
        1 if cross else width,
        ordinal,
        tuple(rings),
        tuple(inside),
        tuple(reentries),
        width if cross else None,
    )


def establish_lease(domains, notification):
    """Prove every nonempty case before reserving one shared physical ring."""
    cases = tuple(
        _establish_case(domains, notification, case)
        for case in domains.cases[notification]
        if not case.empty
    )
    return StructuredEventLease(notification, max((case.width for case in cases), default=0), cases)

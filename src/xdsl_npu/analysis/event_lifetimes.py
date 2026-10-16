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
    LeaseKind,
    LeaseStep,
    LeaseWitness,
    StructuredEventLease,
)
from .structured_protocols import NotificationKind

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


def roundtrip(domains, notification, window, width):
    loop, delta = window.loop, width * window.period
    logical = domains.control.loops[loop]
    old_sources = tuple(domains._instance(p, {}) for p in notification.sources)
    old_consumers = tuple(domains._instance(p, {}) for p in notification.consumers)
    shifted = {logical.ordinal: window.next_same_id(width)}
    next_sources = tuple(domains._instance(p, shifted) for p in notification.sources)
    next_consumers = tuple(domains._instance(p, shifted) for p in notification.consumers)
    assumptions = tuple(
        dict.fromkeys(
            window.same_id_guard(width).comparisons
            + tuple(
                test
                for p in old_sources + old_consumers + next_sources + next_consumers
                for test in p.guard.comparisons
            )
        )
    )
    candidates = []
    for peer in domains.notifications:
        if peer.mechanism is NotificationKind.PIPE_FENCE or not any(
            w.loop is loop for w in domains.windows[peer]
        ):
            continue
        # A bounded event/slot-distance search, never a loop-count scan.
        for offset in range(-min(8, delta), delta + 1):
            for sources, consumers in domains._instances(peer, loop, offset):
                candidates.append((peer, sources, consumers))
    witnesses = []
    for old in old_consumers:
        for goal in next_sources:
            chain = bridge(domains, notification, old, goal, loop, delta, assumptions, candidates)
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
                )
            )
    return tuple(witnesses)


def reentry(domains, notification, loop, delta=1):
    candidates = []
    horizon = delta + max((abs(peer.distance or 0) for peer in domains.notifications), default=0)
    for peer in domains.notifications:
        if peer.mechanism is NotificationKind.PIPE_FENCE:
            continue
        for offset in range(horizon + 1):
            for sources, consumers in domains._instances(peer, loop, offset):
                candidates.append((peer, sources, consumers))
    old_sources, old_consumers = list(domains._instances(notification, loop, 0))[-1]
    next_sources, next_consumers = next(domains._instances(notification, loop, delta))
    assumptions = tuple(
        test
        for p in old_sources + old_consumers + next_sources + next_consumers
        for test in p.guard.comparisons
    )
    witnesses = []
    for old in old_consumers:
        for goal in next_sources:
            chain = bridge(domains, notification, old, goal, loop, delta, assumptions, candidates)
            if chain is None:
                return None
            witnesses.append(LeaseWitness(LeaseKind.INVOCATION, loop, delta, old, goal, chain))
    return tuple(witnesses)


def establish_lease(domains, notification):
    windows = domains.windows[notification]
    capacity = (
        CROSS_BACKLOG if notification.mechanism is NotificationKind.CROSS_EVENT else LOCAL_CAPACITY
    )
    width, rings, inside, reentries = 1, [], [], []
    if not windows:
        inside.append(
            LeaseWitness(
                LeaseKind.FINITE,
                None,
                0,
                notification.consumers[0],
                notification.sources[0],
                upper_uses=1,
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
                        notification.consumers[0],
                        notification.sources[0],
                        upper_uses=window.upper,
                    ),
                )
            elif index == len(windows) - 1:
                witnesses = roundtrip(domains, notification, window, candidate)
            else:
                witnesses = reentry(domains, notification, window.loop, candidate * window.period)
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
    lease = StructuredEventLease(
        notification,
        1 if cross else width,
        tuple(inside),
        tuple(reentries),
        width if cross else None,
    )
    return lease, ordinal

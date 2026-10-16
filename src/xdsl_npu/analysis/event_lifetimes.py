"""Event consumption and reuse from already paired physical notifications.

All functions consume EventDomains. They neither infer memory dependencies nor
allocate IDs, generate actions, or change native dispatch. Unknown lifetime
relations remain unsupported; bounded interface search is not runtime unrolling.
"""

from ..budget import poll
from ..diagnostics import ErrorKind, require
from ..scalar import expression, literal
from .event_contracts import (
    CROSS_BACKLOG,
    LOCAL_CAPACITY,
    CaseLease,
    LeaseKind,
    LeaseWitness,
    StructuredEventLease,
)
from .event_paths import prove_event_path_cover
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard

MAX_INTERFACE_OFFSETS = 256


def _candidates(domains, loop, first, stop, *, needs_clock):
    """Project an actual dispatch-step horizon onto each peer's logical clock.

    A parent stage at offset s executes generation j-s at physical step j.
    Child-local stage offsets belong to their own clock and cannot displace
    the parent interval. This is bounded candidate discovery, not a claim
    that absence from this horizon establishes a safe event lifetime.
    """
    require(
        stop - first <= MAX_INTERFACE_OFFSETS,
        f"event interface offset horizon exceeds analysis budget {MAX_INTERFACE_OFFSETS}",
        kind=ErrorKind.ANALYSIS_LIMIT,
        phase="event-path-cover",
    )
    for peer in domains.notifications:
        if peer.mechanism is NotificationKind.PIPE_FENCE:
            continue
        shifts = []
        for bound in peer.sources + peer.consumers:
            coordinate = dict(bound.coordinates).get(loop)
            if coordinate is None:
                continue
            task = domains.dispatch.tasks[bound.operation, bound.actor]
            scope, stage = task.scope, task.stage
            while scope.operation is not loop and scope.parent is not None:
                stage, scope = scope.parent_stage, scope.parent
            if scope.operation is not loop:
                continue
            shift = domains.arithmetic.normalize(
                expression(
                    "sub",
                    expression("add", coordinate, literal(stage)),
                    domains.control.loops[loop].ordinal,
                )
            ).constant
            if shift is not None:
                shifts.append(shift)
        begin, end = first - max(shifts, default=0), stop - min(shifts, default=0)
        require(
            end - begin <= MAX_INTERFACE_OFFSETS,
            f"translated event interface horizon exceeds analysis budget {MAX_INTERFACE_OFFSETS}",
            kind=ErrorKind.ANALYSIS_LIMIT,
            phase="event-path-cover",
        )
        for peer_case in domains.cases[peer]:
            if (
                peer_case.empty
                or needs_clock
                and not any(w.loop is loop for w in peer_case.windows)
            ):
                continue
            for offset in range(begin, end):
                poll("event reuse candidates", offset=offset, horizon=end - begin)
                for sources, consumers in domains._instances(peer, peer_case, loop, offset):
                    yield peer, sources, consumers


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
        for old in old_consumers:
            for goal in next_sources:
                path_cover = prove_event_path_cover(
                    domains,
                    notification,
                    old,
                    goal,
                    assumptions,
                    _candidates(domains, loop, 0, delta + 1, needs_clock=True),
                )
                if path_cover is None:
                    return None
                # A vacuous cover has no active reuse and also satisfies this
                # universal classification; its empty proof remains explicit.
                direct = all(len(path.steps) == 1 for path in path_cover.paths)
                witnesses.append(
                    LeaseWitness(
                        LeaseKind.CONTROLLER if direct else LeaseKind.QUEUE_ROUNDTRIP,
                        loop,
                        delta,
                        old,
                        goal,
                        path_cover,
                        condition=condition,
                    )
                )
    return tuple(witnesses)


def reentry(domains, notification, case, window, width=1):
    """Return from the last child use to the next same-ID real invocation."""
    loop = window.loop
    witnesses = []
    for phase, delta in window.reuse_phases(width):
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
                path_cover = prove_event_path_cover(
                    domains,
                    notification,
                    old,
                    goal,
                    assumptions,
                    _candidates(domains, loop, 0, delta + 1, needs_clock=False),
                )
                if path_cover is None:
                    return None
                witnesses.append(
                    LeaseWitness(
                        LeaseKind.INVOCATION,
                        loop,
                        delta,
                        old,
                        goal,
                        path_cover,
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

"""Assign finite IDs to frozen virtual event resource constraints.

This module has no domain, memory-analysis, handback or solver dependency.
Allocation and rendering cannot establish new reuse facts or change sites.
"""

from itertools import combinations

from ..diagnostics import require
from ..scalar import expression, literal
from .event_coloring import (
    MAX_COLORING_NODES,
    NoEventAssignment,
    color_event_resources,
    counter_group_usage,
)
from .event_contracts import (
    CROSS_BACKLOG,
    CounterReuseGroup,
    EventAllocation,
    EventBankUsage,
    EventReservation,
    EventResourceProblem,
    PersistentEventAssignment,
    StructuredEventAssignment,
    StructuredEventPlan,
)
from .structured_protocols import NotificationKind
from .structured_schedule import SyncAction


def allocate_event_ids(problem):
    """Prefer exclusive IDs; color only frozen constraints under pressure."""
    require(
        isinstance(problem, EventResourceProblem),
        "allocation requires a generated virtual event resource problem",
    )
    nodes, failure = 0, None
    preferred = problem
    for option, candidate in enumerate((preferred, *preferred.alternatives)):
        require(
            candidate.selection is preferred.selection and candidate.order is preferred.order,
            "protocol option belongs to another generated notification program",
        )
        try:
            ids, work = color_event_resources(candidate, node_limit=MAX_COLORING_NODES - nodes)
        except NoEventAssignment as error:
            nodes += error.nodes
            failure = error
            continue
        nodes += work
        problem = candidate
        break
    else:
        raise failure
    require(all(ids[a] == ids[b] for a, b in problem.peers), "AIV peer event mappings differ")
    sharing = []
    occupied = {}
    for r in problem.reservations:
        occupied.setdefault(r.bank, {}).setdefault(r.event, set())
    for number, demand in enumerate(problem.demands):
        for b in (None,) if demand.backlog is not None else demand.banks:
            for flag in ids[number]:
                occupied.setdefault(b, {}).setdefault(flag, set()).add(number)
    seen = set()
    counter_groups, counter_reuses = [], []
    packs = tuple(frozenset(c.resources) for c in problem.counter_groups)
    for b, pool in occupied.items():
        for event, members in pool.items():
            usage = None
            if b is None and members:
                usage = counter_group_usage(problem.demands, members, problem.counter_reuses, packs)
                require(usage is not None, "mode-2 group has no complete temporal or FIFO premise")
                _, backlog = usage
                require(
                    backlog <= CROSS_BACKLOG, "mode-2 group exceeds its complete-credit backlog"
                )
                problem.work.maximum_counter_backlog = max(
                    problem.work.maximum_counter_backlog, backlog
                )
            witnesses = []
            for a, c in combinations(sorted(members), 2):
                key = (a, c)
                source = (
                    problem.counter_reuses
                    if usage is not None and usage[0] == "time"
                    else problem.relations
                )
                permission = source.get(key)
                require(permission is not None, "same ID lacks a complete-credit reuse constraint")
                witnesses.extend(permission)
                if key not in seen:
                    sharing.extend(permission)
                    seen.add(key)
            if usage is not None and len(members) > 1:
                if usage[0] == "time":
                    counter_reuses.append(
                        CounterReuseGroup(
                            event,
                            tuple(sorted(members)),
                            tuple(
                                (r, n) for r in sorted(members) for n in problem.demands[r].members
                            ),
                            usage[1],
                            tuple(witnesses),
                        )
                    )
                    continue
                certificate = next(
                    (c for c in problem.counter_groups if members <= set(c.resources)), None
                )
                require(certificate is not None, "mode-2 packing lacks a whole-word certificate")
                require(
                    certificate.order is problem.order, "mode-2 FIFO belongs to another issue order"
                )
                counter_groups.append((event, certificate, tuple(sorted(members))))
    problem.work.allocated_local_ids = sum(
        len(pool) for b, pool in occupied.items() if b is not None
    )
    problem.work.allocated_cross_ids = len(occupied.get(None, ()))
    problem.work.persistent_cells = sum(d.persistent is not None for d in problem.demands)
    problem.work.coloring_nodes = nodes
    problem.work.selected_protocol_option = option
    problem.work.peer_equalities = len(problem.peers)
    problem.work.counter_groups = len(problem.counter_groups)
    return EventAllocation(
        ids, tuple(sharing), tuple(counter_groups), option, tuple(counter_reuses)
    )


def _mapped_event(ids, phase):
    if len(ids) == 1:
        return literal(ids[0])
    if all(value == ids[0] + i for i, value in enumerate(ids)):
        return expression("add", literal(ids[0]), phase)
    # A small finite map uses existing, fully defined signed SSA arithmetic.
    # It introduces no device lookup table or effects hidden from analysis.
    result = literal(ids[0])
    for i in range(1, len(ids)):
        step = expression(
            "min",
            literal(1),
            expression("max", literal(0), expression("sub", phase, literal(i - 1))),
        )
        result = expression("add", result, expression("mul", literal(ids[i] - ids[i - 1]), step))
    return result


def allocate_structured_events(problem):
    """Map and render a generated problem, without querying any source facts."""
    allocation = allocate_event_ids(problem)
    preferred = problem
    if allocation.option:
        problem = preferred.alternatives[allocation.option - 1]
    leases = {v.notification: v for v in problem.leases}
    assignments = []
    for notification, resource in problem.uses:
        banks = (
            ()
            if notification.mechanism is NotificationKind.CROSS_EVENT
            else tuple(
                dict.fromkeys(
                    (p.actor, notification.source_engine, notification.destination_engine)
                    for p in notification.sources
                )
            )
        )
        assignments.append(
            StructuredEventAssignment(
                notification,
                () if resource is None else allocation.ids[resource],
                banks,
                leases[notification].cases,
            )
        )
    root = problem.selection.dependencies.memory.module.body.block.first_op
    persistent = tuple(
        PersistentEventAssignment(d.persistent, d.banks[0], allocation.ids[i][0])
        for i, d in enumerate(problem.demands)
        if d.persistent is not None
    )
    reservations = problem.reservations + tuple(
        EventReservation(p.bank, p.event, f"persistent ownership {p.credit.name}", root)
        for p in persistent
    )
    actions = []
    for a in problem.actions:
        if a.resource is None:
            event = None if a.fixed_id is None else literal(a.fixed_id)
        else:
            ids = allocation.ids[a.resource]
            event = _mapped_event(ids, a.phase)
        actions.append(
            SyncAction(
                a.anchor,
                a.placement,
                a.participants,
                a.kind,
                a.source,
                a.destination,
                event,
                a.guard,
                a.protocols,
            )
        )
    banks = {}
    for r in reservations:
        banks.setdefault(r.bank, set()).add(r.event)
    for a in assignments:
        for b in a.banks or (None,):
            banks.setdefault(b, set()).update(a.ids)
    return StructuredEventPlan(
        selection=problem.selection,
        actions=tuple(actions),
        assignments=tuple(assignments),
        leases=problem.leases,
        reservations=reservations,
        order=problem.order,
        work=problem.work,
        channels=problem.channels,
        banks=tuple(EventBankUsage(b, tuple(sorted(ids))) for b, ids in banks.items()),
        sharing=allocation.sharing,
        persistent=persistent,
        counter_groups=allocation.counter_groups,
        preferred_leases=preferred.leases,
        counter_reuse_groups=allocation.counter_reuse_groups,
    )

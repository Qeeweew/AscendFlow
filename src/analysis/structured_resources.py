"""Assign finite IDs to frozen virtual event resource constraints.

This module has no domain, memory-analysis or handback dependency.
Allocation and rendering cannot establish new reuse facts or change sites.
"""

from ..diagnostics import require
from ..scalar import expression, literal
from .event_coloring import (
    color_event_resources,
    counter_group_usage,
)
from .event_contracts import (
    CROSS_BACKLOG,
    CounterFIFOGroup,
    CounterReuseGroup,
    EventAllocation,
    EventBankUsage,
    EventReservation,
    EventResourceProblem,
    PersistentEventAssignment,
    StructuredEventAssignment,
    StructuredEventPlan,
)
from .event_frontier_contracts import LocalEventReuse
from .structured_protocols import NotificationKind
from .structured_schedule import SyncAction


def _check_fifo_subset(problem, certificate, members):
    """Check frozen word/materialization correspondence without new analysis."""
    expected = {
        (r, m): notification
        for r in members
        for m, notification in enumerate(problem.demands[r].members)
    }
    records = tuple((r, m, n) for r, m, n in certificate.messages if r in members)
    actual = {(r, m): n for r, m, n in records}
    require(
        len(records) == len(actual) == len(expected)
        and actual.keys() == expected.keys()
        and all(actual[key] is notification for key, notification in expected.items()),
        "FIFO word lost its complete-credit message correspondence",
    )
    first = next(iter(expected.values()))
    roles = tuple(("publish", p.actor) for p in first.sources) + tuple(
        ("consume", p.actor) for p in first.consumers
    )
    require(
        tuple((phase, actor) for phase, actor, _ in certificate.words) == roles,
        "FIFO word lost an actual publisher or receiver",
    )
    words = tuple(
        tuple(key for key in word if key[0] in members) for _, _, word in certificate.words
    )
    require(
        all(len(word) == len(expected) and set(word) == expected.keys() for word in words)
        and all(word == words[0] for word in words[1:]),
        "FIFO word lost its common complete-credit message sequence",
    )
    bound = sum(problem.demands[r].backlog for r in members)
    require(
        tuple(actor for actor, _ in certificate.contributions)
        == tuple(p.actor for p in first.sources)
        and all(bound <= maximum <= CROSS_BACKLOG for _, maximum in certificate.contributions),
        "FIFO word lost its per-publisher backlog bound",
    )


def allocate_event_ids(problem):
    """Select IDs and a frozen protocol without reentering source analysis."""
    require(
        isinstance(problem, EventResourceProblem),
        "allocation requires a generated virtual event resource problem",
    )
    preferred = problem
    candidates = (preferred, *preferred.alternatives)
    for candidate in candidates:
        require(
            candidate.selection is preferred.selection
            and candidate.order is preferred.order
            and candidate.id_pools is preferred.id_pools,
            "protocol option belongs to another generated notification program",
        )
    ids, option = color_event_resources(preferred, alternatives=preferred.alternatives)
    problem = candidates[option]
    require(all(ids[a] == ids[b] for a, b in problem.peers), "AIV peer event mappings differ")
    sharing = []
    occupied = {}
    for r in problem.reservations:
        occupied.setdefault(r.bank, {}).setdefault(r.event, set())
    for number, demand in enumerate(problem.demands):
        for b in (None,) if demand.backlog is not None else demand.banks:
            for flag in ids[number]:
                occupied.setdefault(b, {}).setdefault(flag, set()).add(number)
    counter_groups, counter_reuses, compositions = [], [], []
    packs = tuple(frozenset(c.resources) for c in problem.counter_groups)
    time_packs = tuple(c.groups for c in problem.counter_compositions)
    problem.work.local_reuse_groups = 0
    problem.work.local_reuse_rectangles = 0
    problem.work.local_reuse_row_checks = 0
    for b, pool in occupied.items():
        for event, members in pool.items():
            if b is not None:
                if len(members) > 1:
                    rectangles = problem.relations.cover_group(members)
                    require(
                        rectangles is not None, "same ID lacks a complete-credit reuse constraint"
                    )
                    sharing.append(
                        LocalEventReuse(
                            b,
                            event,
                            tuple((r, problem.demands[r].members) for r in sorted(members)),
                            rectangles,
                        )
                    )
                    problem.work.local_reuse_groups += 1
                    problem.work.local_reuse_rectangles += len(rectangles)
                    problem.work.local_reuse_row_checks += len(members)
                continue
            usage = None
            certificate = None
            composition = None
            if b is None and members:
                usage = counter_group_usage(
                    problem.demands, members, problem.counter_reuses, packs, time_packs=time_packs
                )
                require(usage is not None, "mode-2 group has no complete temporal or FIFO premise")
                _, backlog = usage
                require(
                    backlog <= CROSS_BACKLOG, "mode-2 group exceeds its complete-credit backlog"
                )
                problem.work.maximum_counter_backlog = max(
                    problem.work.maximum_counter_backlog, backlog
                )
                if usage[0] == "fifo":
                    certificate = next(
                        (c for c in problem.counter_groups if members <= set(c.resources)), None
                    )
                    require(
                        certificate is not None, "mode-2 packing lacks a whole-word certificate"
                    )
                    require(
                        certificate.order is problem.order,
                        "mode-2 FIFO belongs to another issue order",
                    )
                elif usage[0] == "time_fifo":
                    composition = next(
                        c for c in problem.counter_compositions if members <= set(c.resources)
                    )
                    require(
                        composition.order is problem.order,
                        "counter composition belongs to another issue order",
                    )
            if usage is not None and usage[0] == "time":
                if len(members) > 1:
                    rectangles = problem.counter_reuses.cover_group(members)
                    require(rectangles is not None, "counter reuse lacks a complete selected row")
                    group = CounterReuseGroup(
                        event,
                        tuple(sorted(members)),
                        tuple((r, n) for r in sorted(members) for n in problem.demands[r].members),
                        usage[1],
                        rectangles,
                    )
                    counter_reuses.append(group)
                    sharing.append(group)
                continue
            if composition is not None:
                selected = tuple(group.intersection(members) for group in composition.groups)
                rectangles = problem.counter_reuses.cover_groups(selected)
                require(
                    rectangles is not None, "counter composition lacks a selected cross-child row"
                )
                compositions.append(
                    (event, composition, tuple(sorted(members)), usage[1], rectangles)
                )
                fifo = composition.fifo
            else:
                fifo = (certificate,) if certificate is not None else ()
            # Filtering a certified word preserves its order, exclusions and
            # quorum, and can only reduce its conservative backlog bound.
            for child in fifo:
                require(child.order is problem.order, "FIFO child belongs to another issue order")
                selected = tuple(sorted(members.intersection(child.resources)))
                if len(selected) > 1:
                    _check_fifo_subset(problem, child, selected)
                    group = CounterFIFOGroup(event, child, selected)
                    sharing.append(group)
                    if composition is None:
                        counter_groups.append(group)
    problem.work.allocated_local_ids = sum(
        len(pool) for b, pool in occupied.items() if b is not None
    )
    problem.work.allocated_cross_ids = len(occupied.get(None, ()))
    problem.work.persistent_cells = sum(d.persistent is not None for d in problem.demands)
    problem.work.selected_protocol_option = option
    problem.work.peer_equalities = len(problem.peers)
    problem.work.phase_occupancies = sum(d.phase is not None for d in problem.demands)
    problem.work.counter_groups = len(problem.counter_groups)
    return EventAllocation(
        ids,
        tuple(sharing),
        tuple(counter_groups),
        option,
        tuple(counter_reuses),
        tuple(compositions),
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
        id_pools=problem.id_pools,
        occupancies=problem.demands,
        sharing=allocation.sharing,
        persistent=persistent,
        counter_groups=allocation.counter_groups,
        preferred_leases=preferred.leases,
        counter_reuse_groups=allocation.counter_reuse_groups,
        counter_compositions=allocation.counter_compositions,
    )

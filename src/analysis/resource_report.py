"""Serialize established event decisions without running further analysis.

The report separates logical credit identity, ring protocol, physical mapping
and emitted actions. Structural operation numbers come from the original
control facts, not object addresses or the notification-mutated IR walk.
"""

from functools import cache
from itertools import pairwise

from .event_contracts import LOCAL_CAPACITY, SUPPORTED_CROSS_IDS
from .structured_protocols import NotificationKind


def event_resource_report(plan):
    operations = {
        op: number for number, op in enumerate(plan.selection.dependencies.memory.control.frames)
    }
    notifications = plan.selection.notifications
    numbers = {item: number for number, item in enumerate(notifications)}
    assignments = {item.notification: item for item in plan.assignments}
    leases = {item.notification: item for item in plan.leases}
    channels = {member: number for number, c in enumerate(plan.channels) for member in c.members}

    @cache
    def text(value):
        return value.text()

    def guard(value):
        return [(text(c.lhs), c.predicate, text(c.rhs)) for c in value.comparisons]

    def bank(value):
        return [item.value for item in value]

    def point(value):
        return {
            "operation": operations[value.operation],
            "actor": value.actor.value,
            "placement": value.placement.value,
            "guard": guard(value.guard),
            "coordinates": [(operations[loop], text(e)) for loop, e in value.coordinates],
        }

    def counter_group(event, certificate, members):
        messages = {(r, m): n for r, m, n in certificate.messages}
        words, publications, consumptions = [], [], []
        for phase, actor, word in certificate.words:
            selected = tuple(key for key in word if key[0] in members)
            words.append({"phase": phase, "actor": actor.value, "messages": selected})
            points = [
                next(
                    p
                    for p in (
                        messages[key].sources if phase == "publish" else messages[key].consumers
                    )
                    if p.actor is actor
                )
                for key in selected
            ]
            target = publications if phase == "publish" else consumptions
            target.extend(
                [point(a), point(b)] for a, b in pairwise(points) if a.operation is not b.operation
            )
        # Serialize the selected complete credits, not the larger candidate
        # group's numeric bound or its unselected message transitions.
        backlog = sum(
            max(
                c.backlog
                for (r, _), n in messages.items()
                if r == resource
                for c in leases[n].cases
            )
            for resource in members
        )
        return {
            "id": event,
            "resources": list(members),
            "certified_resources": list(certificate.resources),
            "messages": [
                {"resource": r, "member": m, "notification": numbers[n]}
                for (r, m), n in messages.items()
            ],
            "words": words,
            "certified_words": [
                {"phase": phase, "actor": actor.value, "messages": word}
                for phase, actor, word in certificate.words
            ],
            "publisher_backlog": [
                {"actor": actor.value, "maximum": backlog} for actor, _ in certificate.contributions
            ],
            "certified_publisher_backlog": [
                {"actor": actor.value, "maximum": bound}
                for actor, bound in certificate.contributions
            ],
            "existing_controller_order": consumptions,
            "additional_counter_effect_order": publications,
            "clock_translations": [
                {
                    "phase": phase,
                    "actor": actor.value,
                    "shifts": [(operations[loop], text(shift)) for loop, shift in shifts],
                }
                for phase, actor, shifts in certificate.clock_shifts
            ],
            "fifo_reentry": "matching filtered words in lexicographic clock order",
            "additional_controller_orders": [],
        }

    def counter_reuse(group):
        first = group.messages[0][1]
        return {
            "id": group.event,
            "resources": list(group.resources),
            "messages": [{"resource": r, "notification": numbers[n]} for r, n in group.messages],
            "publisher_backlog": [
                {"actor": p.actor.value, "maximum": group.backlog} for p in first.sources
            ],
            "backlog_rule": "maximum of noninterfering complete credits",
            "consumption_handbacks": [
                {
                    "earlier": numbers[w.earlier],
                    "later": numbers[w.later],
                    "rule": w.rule,
                    "handbacks": [
                        {
                            "consume": point(h.consume),
                            "next_publication": point(h.next_set),
                            "condition": guard(h.condition),
                            "relays": [
                                numbers[n]
                                for n in (
                                    ()
                                    if h.relay is None
                                    else h.relay
                                    if isinstance(h.relay, tuple)
                                    else (h.relay,)
                                )
                            ],
                        }
                        for h in w.handbacks
                    ],
                }
                for w in group.witnesses
            ],
            "additional_counter_effect_order": [],
            "additional_controller_orders": [],
        }

    streams = []
    for number, item in enumerate(notifications):
        assigned, lease = assignments.get(item), leases.get(item)
        streams.append(
            {
                "number": number,
                "name": item.name,
                "mechanism": item.mechanism.value,
                "route": [item.source_engine.value, item.destination_engine.value],
                "credit": channels.get(item),
                "sources": [point(p) for p in item.sources],
                "consumers": [point(p) for p in item.consumers],
                "width": None if lease is None else lease.width,
                "cases": []
                if lease is None
                else [
                    {
                        "condition": guard(c.case.condition),
                        "width": c.width,
                        "phase": text(c.ordinal),
                        "backlog": c.backlog,
                        "policy": c.policy,
                        "rings": [(operations[w.loop], width) for w, width in c.rings],
                        "reuses": [
                            {
                                "loop": None if r.loop is None else operations[r.loop],
                                "distance": r.delta,
                                "finite_uses": r.upper_uses,
                                "handbacks": len(r.handbacks),
                            }
                            for r in (*c.inside, *c.reentries)
                        ],
                    }
                    for c in lease.cases
                ],
                "ids": [] if assigned is None else list(assigned.ids),
                "banks": [] if assigned is None else [bank(b) for b in assigned.banks],
            }
        )
    demands = {}
    credit_demands = {}
    for item in plan.assignments:
        credit = (
            ("cell", channels[item.notification])
            if item.notification in channels
            else ("stream", numbers[item.notification])
        )
        for b in item.banks or (None,):
            key = credit, b
            credit_demands[key] = max(credit_demands.get(key, 0), leases[item.notification].width)
    for (_, b), width in credit_demands.items():
        demands[b] = demands.get(b, 0) + width
    reserved = {}
    for item in plan.reservations:
        reserved.setdefault(item.bank, set()).add(item.event)
    return {
        "protocol_selection": {
            "candidates": plan.work.protocol_candidates,
            "selected_option": plan.work.selected_protocol_option,
            "changed_streams": [
                numbers[old.notification]
                for old in plan.preferred_leases
                if old != leases[old.notification]
            ],
            "preferred_changes": [
                {
                    "notification": numbers[old.notification],
                    "width": old.width,
                    "cases": [
                        {
                            "condition": guard(c.case.condition),
                            "width": c.width,
                            "phase": text(c.ordinal),
                            "backlog": c.backlog,
                            "rings": [(operations[w.loop], width) for w, width in c.rings],
                            "policy": c.policy,
                        }
                        for c in old.cases
                    ],
                }
                for old in plan.preferred_leases
                if old != leases[old.notification]
            ],
        },
        "streams": streams,
        "exclusive_demand": [
            {
                "bank": None if b is None else bank(b),
                "lanes": demand,
                "reserved": len(reserved.get(b, ())),
                "capacity": len(SUPPORTED_CROSS_IDS) if b is None else LOCAL_CAPACITY,
            }
            for b, demand in demands.items()
        ],
        "reservations": [
            {"bank": bank(r.bank), "id": r.event, "reason": r.reason} for r in plan.reservations
        ],
        "sharing": [
            {"kind": type(w).__name__, "members": [numbers[w.earlier], numbers[w.later]]}
            for w in plan.sharing
        ],
        "counter_groups": [
            counter_group(event, certificate, members)
            for event, certificate, members in plan.counter_groups
        ],
        "counter_time_reuse": [counter_reuse(group) for group in plan.counter_reuse_groups],
        "actions": [
            {
                "operation": operations[a.anchor],
                "placement": a.placement.value,
                "actors": [p.value for p in a.participants],
                "kind": a.kind.value,
                "route": [a.source.value, a.destination.value],
                "event": None if a.event is None else text(a.event),
                "guard": guard(a.guard),
                "origins": list(a.protocols),
            }
            for a in plan.actions
        ],
        "fences": sum(n.mechanism is NotificationKind.PIPE_FENCE for n in notifications),
    }

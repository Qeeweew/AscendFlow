"""Serialize established event decisions without running further analysis.

The report separates logical credit identity, ring protocol, physical mapping
and emitted actions. Structural operation numbers come from the original
control facts, not object addresses or the notification-mutated IR walk.
"""

from functools import cache
from itertools import pairwise

from .event_contracts import LOCAL_CAPACITY, SUPPORTED_CROSS_IDS
from .event_frontier_contracts import LocalEventReuse
from .event_sharing import EventSharing
from .smt import Boolean
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard


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

    frontiers, frontier_numbers = [], {}
    certificates, certificate_numbers = [], {}

    def frontier(value):
        if value not in frontier_numbers:
            number = frontier_numbers[value] = len(frontiers)
            record = {"head": numbers[value.head], "tail": numbers[value.tail]}
            frontiers.append(record)
            record["children"] = [frontier(child) for child in value.children]
            return number
        return frontier_numbers[value]

    def certificate(value):
        if value not in certificate_numbers:
            certificate_numbers[value] = len(certificates)
            certificates.append(
                {
                    "interfaces": [
                        {
                            "left": frontier(p.left),
                            "right": frontier(p.right),
                            "forward": p.forward,
                            "original": None
                            if p.original is None
                            else [type(w).__name__ for w in p.original],
                            "consumption": [
                                handback(h)
                                for h in (
                                    *p.handbacks,
                                    *(
                                        h
                                        for w in p.original or ()
                                        if isinstance(w, EventSharing)
                                        for h in w.handbacks
                                    ),
                                )
                            ],
                        }
                        for p in value.interfaces
                    ],
                }
            )
        return certificate_numbers[value]

    def resource_numbers(mask):
        result = []
        while mask:
            low = mask & -mask
            result.append(low.bit_length() - 1)
            mask ^= low
        return result

    def sharing(value):
        if isinstance(value, LocalEventReuse):
            return {
                "kind": "LocalEventReuse",
                "bank": bank(value.bank),
                "id": value.event,
                "credits": [
                    {"resource": r, "members": [numbers[n] for n in members]}
                    for r, members in value.credits
                ],
                "rectangles": [
                    {
                        "left": resource_numbers(s.left),
                        "right": resource_numbers(s.right),
                        "certificate": certificate(s.certificate),
                    }
                    for s in value.rectangles
                ],
            }
        return {
            "kind": type(value).__name__,
            "members": [numbers[value.earlier], numbers[value.later]],
        }

    @cache
    def point(value):
        return {
            "operation": operations[value.operation],
            "actor": value.actor.value,
            "placement": value.placement.value,
            "guard": guard(value.guard),
            "coordinates": [(operations[loop], text(e)) for loop, e in value.coordinates],
        }

    consumption_proofs, proof_numbers = [], {}
    consumption_predicates, predicate_numbers = [], {}

    def predicate(value):
        pending = [value]
        unfinished = []
        while pending:
            item = pending.pop()
            key = type(item), item
            if key in predicate_numbers:
                continue
            predicate_numbers[key] = len(consumption_predicates)
            record = {}
            consumption_predicates.append(record)
            if type(item) is bool:
                record["constant"] = item
            elif isinstance(item, Boolean):
                record.update(kind=item.kind, operands=[])
                pending.extend(reversed(item.operands))
                unfinished.append((record, item.operands))
            else:
                record["comparison"] = [text(item.lhs), item.predicate, text(item.rhs)]
        # Finish only the newly seen syntax; no expression path expansion.
        for record, operands in unfinished:
            record["operands"] = [predicate_numbers[type(v), v] for v in operands]
        return predicate_numbers[type(value), value]

    def consumption_proof(root):
        pending = [root]
        records = []
        while pending:
            proof = pending.pop()
            if proof in proof_numbers:
                continue
            proof_numbers[proof] = len(consumption_proofs)
            node = proof.node
            record = {
                "kind": node.kind,
                "protocol": None if node.notification is None else numbers[node.notification],
                "point": None if node.point is None else point(node.point),
                "condition": predicate(proof.condition),
            }
            if node.evidence is not None:
                fact = node.evidence
                record["established_consumption"] = {
                    "protocol": numbers[fact.origin.notification],
                    "wait": point(fact.origin.point),
                    "admission": point(fact.target),
                    "after_actual_wait": fact.after_action,
                    "premises": guard(ActionGuard(fact.premises)),
                    "answer": fact.answer.kind,
                }
            consumption_proofs.append(record)
            records.append((record, proof.inputs))
            pending.extend(reversed(proof.inputs))
        for record, inputs in records:
            record["inputs"] = [proof_numbers[child] for child in inputs]
        return proof_numbers[root]

    def handback(value):
        return {
            "origin": numbers[value.proof.origin.notification],
            "consume": point(value.consume),
            "next_publication": point(value.next_set),
            "condition": guard(value.condition),
            "proof": consumption_proof(value.proof.proof),
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
            "episodes": [
                {
                    "resources": list(e.resources),
                    "words": [
                        {
                            "phase": phase,
                            "actor": actor.value,
                            "messages": [key for key in word if key[0] in members],
                        }
                        for phase, actor, word in e.words
                    ],
                }
                for e in certificate.episodes
            ],
            "episode_order": [
                {"phase": phase, "actor": actor.value, "earlier": point(old), "later": point(new)}
                for phase, actor, old, new, _ in certificate.boundaries
            ],
            "backlog_rule": "sum of complete-credit bounds across all episodes",
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
                    "handbacks": [handback(h) for h in w.handbacks],
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
                                "proofs": [handback(h) for h in r.handbacks],
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
        "id_pools": {
            "local": plan.id_pools.local,
            "cross": plan.id_pools.cross,
            "banks": [{"bank": bank(b), "ids": ids} for b, ids in plan.id_pools.banks],
        },
        "exclusive_demand": [
            {
                "bank": None if b is None else bank(b),
                "lanes": demand,
                "reserved": len(reserved.get(b, ())),
                "capacity": len(SUPPORTED_CROSS_IDS) if b is None else LOCAL_CAPACITY,
                "assignable_ids": plan.id_pools.ids(b),
            }
            for b, demand in demands.items()
        ],
        "reservations": [
            {"bank": bank(r.bank), "id": r.event, "reason": r.reason} for r in plan.reservations
        ],
        "sharing": [sharing(w) for w in plan.sharing],
        "consumption_proofs": consumption_proofs,
        "consumption_predicates": consumption_predicates,
        "local_event_certificates": certificates,
        "local_event_frontiers": frontiers,
        "counter_groups": [
            counter_group(event, certificate, members)
            for event, certificate, members in plan.counter_groups
        ],
        "counter_time_reuse": [counter_reuse(group) for group in plan.counter_reuse_groups],
        "counter_compositions": [
            {
                "id": event,
                "resources": list(members),
                "groups": [
                    sorted(group.intersection(members))
                    for group in certificate.groups
                    if group.intersection(members)
                ],
                "maximum_backlog": backlog,
                "backlog_rule": "maximum of FIFO-child sums, after complete consumption handbacks",
                "fifo": [
                    counter_group(
                        event, leaf, tuple(sorted(set(leaf.resources).intersection(members)))
                    )
                    for leaf in certificate.fifo
                    if len(set(leaf.resources).intersection(members)) > 1
                ],
                "time_reuses": [
                    {
                        "resources": [a, b],
                        "earlier": numbers[w.earlier],
                        "later": numbers[w.later],
                        "rule": w.rule,
                        "handbacks": [handback(h) for h in w.handbacks],
                    }
                    for a, b, w in certificate.witnesses
                    if a in members and b in members
                ],
            }
            for event, certificate, members, backlog in plan.counter_compositions
        ],
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

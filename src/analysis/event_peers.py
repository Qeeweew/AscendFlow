"""Select shared AIV code from virtual actions, before assigning any ID.

Mirrored private credits keep distinct storage and hardware namespaces.
Their equal-number constraint is solely a code generation requirement.
"""

from dataclasses import replace

from ..target.ascend910b import Actor


def bind_event_peers(demands, actions):
    def signature(resource, actor):
        return tuple(
            (a.anchor, a.placement, a.kind, a.source, a.destination, a.guard, a.phase)
            for a in actions
            if a.resource == resource and a.participants == (actor,)
        )

    signatures, peers = {}, []
    for number, demand in enumerate(demands):
        if demand.backlog is not None or not demand.banks:
            continue
        actors = {bank[0] for bank in demand.banks}
        if actors not in ({Actor.AIV0}, {Actor.AIV1}):
            continue
        actor = next(iter(actors))
        key = (
            demand.width,
            tuple((s, d) for _, s, d in demand.banks),
            signature(number, actor),
        )
        if not key[-1]:
            continue
        other = Actor.AIV1 if actor is Actor.AIV0 else Actor.AIV0
        available = signatures.get((other, key), [])
        if available:
            peers.append((available.pop(0), number))
        else:
            signatures.setdefault((actor, key), []).append(number)

    representative = {b: a for a, b in peers}

    def instruction(action):
        return (
            action.kind,
            action.source,
            action.destination,
            action.guard,
            representative.get(action.resource, action.resource),
            action.phase,
            action.fixed_id,
        )

    groups = {}
    for action in actions:
        family = "cube" if action.participants == (Actor.AIC,) else "vector"
        groups.setdefault((action.anchor, action.placement, family), []).append(action)
    result = []
    for (_, _, family), group in groups.items():
        if family == "cube":
            result.extend(group)
            continue
        left = [a for a in group if Actor.AIV0 in a.participants]
        right = [a for a in group if Actor.AIV1 in a.participants]
        if len(left) != len(right) or any(
            instruction(a) != instruction(b) for a, b in zip(left, right, strict=True)
        ):
            result.extend(group)
            continue
        result.extend(
            replace(
                a,
                participants=(Actor.AIV0, Actor.AIV1),
                protocols=tuple(dict.fromkeys(a.protocols + b.protocols)),
            )
            for a, b in zip(left, right, strict=True)
        )
    return tuple(peers), tuple(result)

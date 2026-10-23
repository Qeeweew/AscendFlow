"""Whole mode-2 words and explicit compositions of resource-use episodes.

Each candidate is a finite set of complete credits, with the same actual
iteration/phase at every publisher and receiver. Deleting messages preserves
its word, quorum and conservative backlog certificate. Sharing introduces
counter-effect FIFO, not a new Wait or controller order. Its local progress
argument uses monotone source native prefixes, never Set-effect completion
inferred from issue order alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cmp_to_key
from itertools import pairwise

from ..scalar import expression
from ..target.ascend910b import Actor
from .event_contracts import CROSS_BACKLOG
from .event_sharing import CounterSharing, _counter_shape


@dataclass(frozen=True, slots=True)
class CounterGroup:
    resources: tuple[int, ...]
    messages: tuple
    words: tuple[tuple[str, Actor, tuple], ...]
    contributions: tuple[tuple[Actor, int], ...]
    fifo: tuple
    clock_shifts: tuple
    # The referenced queue order was established before this optional choice.
    order: object
    # Each child retains a same-clock word. Boundaries prove concatenation
    # at every actual publisher/receiver, never a cross-actor clock ordering.
    episodes: tuple[CounterGroup, ...] = ()
    boundaries: tuple = ()
    sharing: tuple = ()


@dataclass(frozen=True, slots=True)
class CounterTimeGroups:
    """FIFO children separated by complete-credit consumption handbacks."""

    resources: tuple[int, ...]
    groups: tuple[frozenset[int], ...]
    fifo: tuple[CounterGroup, ...]
    witnesses: tuple
    order: object


def plan_counter_time_groups(program, demands, fifo, reuses):
    """Freeze bounded combinations of existing FIFO and temporal premises.

    Different children are disjoint complete-credit sets. Every cross-child
    pair already has time noninterference, so two children cannot both have
    outstanding contributions; each child's own bound still adds its credits.
    """
    leaves = {frozenset(c.resources): c for c in fifo}
    if not leaves or not reuses:
        return ()
    neighbors = {}
    for a, b in reuses:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    choices = sorted(leaves, key=lambda values: (-len(values), tuple(sorted(values))))
    choices += [frozenset((r,)) for r, d in enumerate(demands) if d.backlog is not None]
    result = {}
    for first in leaves:
        groups, used = [first], set(first)
        eligible = set.intersection(*(neighbors.get(r, set()) for r in first))
        for choice in choices:
            if choice.intersection(used) or not choice <= eligible:
                continue
            groups.append(choice)
            used.update(choice)
            eligible.intersection_update(*(neighbors.get(r, set()) for r in choice))
        if len(groups) < 2:
            continue
        witnesses = tuple(
            (min(a, b), max(a, b), witness)
            for i, left in enumerate(groups)
            for right in groups[i + 1 :]
            for a in sorted(left)
            for b in sorted(right)
            for witness in reuses[min(a, b), max(a, b)]
        )
        resources = tuple(sorted(used))
        result[tuple(groups)] = CounterTimeGroups(
            resources,
            tuple(groups),
            tuple(leaves[g] for g in groups if g in leaves),
            witnesses,
            program.order,
        )
    return tuple(result.values())


def _certificate(program, demands, resources, relations):
    """The unchanged same-clock word, quorum and monotone-prefix rule."""
    domains = program.domains
    pairs = [
        relations.get((min(a, b), max(a, b)))
        for i, a in enumerate(resources)
        for b in resources[i + 1 :]
    ]
    if any(pair is None or any(not isinstance(w, CounterSharing) for w in pair) for pair in pairs):
        return None
    messages = [(r, i, n) for r in resources for i, n in enumerate(demands[r].members)]
    if not messages:
        return None
    first = messages[0][2]
    publishers = tuple(p.actor for p in first.sources)
    receivers = tuple(p.actor for p in first.consumers)
    if any(
        tuple(p.actor for p in n.sources) != publishers
        or tuple(p.actor for p in n.consumers) != receivers
        or (n.source_engine, n.destination_engine)
        != (first.source_engine, first.destination_engine)
        for _, _, n in messages
    ):
        return None
    backlog = sum(demands[r].backlog for r in resources)
    if backlog > CROSS_BACKLOG:
        return None
    words, fifo, clock_shifts = [], [], []
    for phase, actors in (("publish", publishers), ("consume", receivers)):
        for actor in actors:
            points, alignment = [], None
            for resource, member, notification in messages:
                point = next(
                    p
                    for p in (
                        notification.sources if phase == "publish" else notification.consumers
                    )
                    if p.actor is actor
                )
                site = domains.dispatch.site(point)
                shape = (
                    site.scope,
                    site.stage,
                    point.coordinates,
                    point.coordinate_guards,
                    point.families,
                    point.selected,
                    domains.clocks[notification],
                )
                if alignment is None:
                    alignment = shape
                elif shape != alignment:
                    # Different stage offsets can reorder A(g+1) and B(g).
                    # Pairwise order at equal g is insufficient for this rule.
                    return None
                shifts = []
                actual = dict(point.coordinates)
                for loop in domains.clocks[notification]:
                    if loop not in actual:
                        return None
                    shift = domains.arithmetic.normalize(
                        expression("sub", actual[loop], domains.control.loops[loop].ordinal)
                    )
                    if not domains.control.invariant_over(shift, loop):
                        # Equal-iteration pair order does not prove the word
                        # through a reversal or other clock remapping.
                        return None
                    shifts.append((loop, shift))
                if not points:
                    clock_shifts.append((phase, actor, tuple(shifts)))
                points.append((site.lexical, resource, member, point))
            points.sort(key=lambda item: item[:3])
            words.append((phase, actor, tuple((r, m) for _, r, m, _ in points)))
            fifo.extend(
                (a[3], b[3]) for a, b in pairwise(points) if a[3].operation is not b[3].operation
            )
    if any(word != words[0][2] for _, _, word in words[1:]):
        return None
    # New same-ID FIFO edges run along each publisher's actual issue word.
    # A later Set already captures all earlier native work on this source
    # pipe and is issued after the earlier Set's admission. The earlier
    # counter effect has no additional prerequisites once that prefix is
    # ready (backlog is bounded below capacity), so adding its effect before
    # the later effect cannot introduce a local blocking cycle. Receiver
    # words agree, including lexicographic reentry under these translations.
    # This does not assert that distinct-ID Set effects were ordered before
    # packing. Keep the AIV backlog contributions separate as well.
    return CounterGroup(
        tuple(resources),
        tuple(messages),
        tuple(words),
        tuple((actor, backlog) for actor in publishers),
        tuple(fifo),
        tuple(clock_shifts),
        program.order,
    )


def _concatenated_word(program, demands, resources, compatible, rule):
    """Compose complete-credit uses across universally ordered word blocks.

    The credit retains one allocation identity and its established backlog
    through all blocks. No loop exit clears a counter: the bound is the sum
    of complete-credit bounds, including contributions still pending across
    the boundary. FIRST/LAST only prove each actor's actual message order.
    """
    domains = program.domains
    if sum(demands[r].backlog for r in resources) > CROSS_BACKLOG:
        return None
    pieces = {}
    for r in resources:
        for member, notification in enumerate(demands[r].members):
            pieces.setdefault(_counter_shape(notification), {}).setdefault(r, []).append(
                (member, notification)
            )
    if len(pieces) < 2:
        return None
    episodes, sharing = [], []
    for uses in pieces.values():
        projected = tuple(
            replace(demand, members=tuple(n for _, n in uses.get(r, ())))
            for r, demand in enumerate(demands)
        )
        active = tuple(r for r in resources if r in uses)
        relations = {}
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                witnesses = []
                for left in projected[a].members:
                    for right in projected[b].members:
                        permission = compatible(left, right)
                        if permission is None or any(
                            not isinstance(w, CounterSharing) for w in permission
                        ):
                            return None
                        witnesses.extend(permission)
                relations[a, b] = tuple(witnesses)
                sharing.extend((a, b, w) for w in witnesses)
        certificate = _certificate(program, projected, active, relations)
        if certificate is None:
            return None
        labels = {(r, m): (r, uses[r][m][0]) for r, m, _ in certificate.messages}
        episodes.append(
            replace(
                certificate,
                messages=tuple((r, uses[r][m][0], n) for r, m, n in certificate.messages),
                words=tuple(
                    (phase, actor, tuple(labels[key] for key in word))
                    for phase, actor, word in certificate.words
                ),
            )
        )

    messages = {(r, m): n for r in resources for m, n in enumerate(demands[r].members)}
    boundaries = {}

    def before(a, b):
        key = a, b
        if key in boundaries:
            return boundaries[key]
        witnesses = []
        for (phase, actor, aw), (bp, ba, bw) in zip(
            episodes[a].words, episodes[b].words, strict=True
        ):
            if (phase, actor) != (bp, ba):
                boundaries[key] = None
                return None
            left, right = messages[aw[-1]], messages[bw[0]]
            left_cases, right_cases = rule._cases(left), rule._cases(right)
            if not left_cases or not right_cases:
                boundaries[key] = None
                return None
            for lc in left_cases:
                if lc.empty:
                    continue
                ends = tuple(domains._instances(left, lc, None, 0))[-1]
                old = next(p for p in ends[0 if phase == "publish" else 1] if p.actor is actor)
                for rc in right_cases:
                    if rc.empty:
                        continue
                    starts = next(domains._instances(right, rc, None, 0))
                    new = next(
                        p for p in starts[0 if phase == "publish" else 1] if p.actor is actor
                    )
                    order = domains._order_conditions(old, new)
                    assumptions = old.guard.comparisons + new.guard.comparisons
                    if not domains.arithmetic.queries.covers(order, assumptions).proved:
                        boundaries[key] = None
                        return None
                    witnesses.append((phase, actor, old, new, order))
        boundaries[key] = tuple(witnesses)
        return boundaries[key]

    def compare(a, b):
        if before(a, b) is not None:
            return -1
        if before(b, a) is not None:
            return 1
        return 0

    indices = sorted(range(len(episodes)), key=cmp_to_key(compare))
    links = []
    for a, b in pairwise(indices):
        if (witnesses := before(a, b)) is None:
            return None
        links.extend(witnesses)
    episodes = tuple(episodes[i] for i in indices)
    words = tuple(
        (
            phase,
            actor,
            tuple(
                key
                for e in episodes
                for p, a, word in e.words
                if (p, a) == (phase, actor)
                for key in word
            ),
        )
        for phase, actor, _ in episodes[0].words
    )
    bound = sum(demands[r].backlog for r in resources)
    return CounterGroup(
        tuple(resources),
        tuple((r, m, n) for (r, m), n in messages.items()),
        words,
        tuple((actor, bound) for actor, _ in episodes[0].contributions),
        tuple(edge for episode in episodes for edge in episode.fifo),
        tuple(shift for episode in episodes for shift in episode.clock_shifts),
        program.order,
        episodes,
        tuple(links),
        tuple(sharing),
    )


def plan_counter_groups(program, demands, relations, *, compatible, handbacks):
    """Publish overlapping, certified choices rather than one greedy partition.

    Equal activation/clock shapes retain useful whole words even when an
    earlier disjoint-residue pairing would prevent their greedy discovery.
    Shapes only select candidates; every published choice passes the same
    complete word/quorum/backlog rule. Two bounded passes retain the general
    disjoint-activation choices as well, without enumerating member subsets.
    """
    counters, shapes = [], {}
    for number, demand in enumerate(demands):
        if demand.backlog is not None:
            counters.append(number)
            shape = tuple(_counter_shape(member) for member in demand.members)
            shapes.setdefault(shape, []).append(number)

    result = {}

    def pack(numbers):
        groups = []
        for number in numbers:
            for i, (members, _) in enumerate(groups):
                candidate = _certificate(program, demands, (*members, number), relations)
                if candidate is not None:
                    groups[i] = ((*members, number), candidate)
                    break
            else:
                groups.append(((number,), None))
        for members, certificate in groups:
            if certificate is not None:
                result[members] = certificate

    pack(counters)
    for numbers in shapes.values():
        if len(numbers) < 2:
            continue
        certificate = _certificate(program, demands, tuple(numbers), relations)
        if certificate is not None:
            result[certificate.resources] = certificate
        else:
            certificate = _concatenated_word(
                program, demands, tuple(numbers), compatible, handbacks
            )
            if certificate is not None:
                result[certificate.resources] = certificate
            pack(numbers)
    return tuple(result.values())

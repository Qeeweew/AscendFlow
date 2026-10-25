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
from itertools import groupby, pairwise

from ..scalar import expression
from ..target.ascend910b import Actor
from .event_contracts import CROSS_BACKLOG
from .smt import all_of, any_of
from .structured_protocols import NotificationKind


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
    # At tied native anchors no relative action order is invented. Each
    # record proves at most one actual member activates at that occurrence.
    exclusions: tuple = ()


@dataclass(frozen=True, slots=True)
class CounterTimeGroups:
    """FIFO children separated by complete-credit consumption handbacks."""

    resources: tuple[int, ...]
    groups: tuple[frozenset[int], ...]
    fifo: tuple[CounterGroup, ...]
    rectangles: tuple
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
    full = sum(1 << r for r, demand in enumerate(demands) if demand.backlog is not None)
    choices = sorted(leaves, key=lambda values: (-len(values), tuple(sorted(values))))
    choices += [frozenset((r,)) for r, d in enumerate(demands) if d.backlog is not None]
    result = {}
    for first in leaves:
        groups, used = [first], set(first)
        occupied = sum(1 << r for r in first)
        eligible = full
        for r in first:
            eligible &= reuses.row_mask(r)
        for choice in choices:
            mask = sum(1 << r for r in choice)
            if (
                mask & occupied
                or mask & ~eligible
                or any(occupied & ~reuses.row_mask(r) for r in choice)
            ):
                continue
            groups.append(choice)
            used.update(choice)
            occupied |= mask
            for r in choice:
                eligible &= reuses.row_mask(r)
        if len(groups) < 2:
            continue
        rectangles = reuses.cover_groups(tuple(groups))
        assert rectangles is not None, "counter composition lost a selected cross-child row"
        resources = tuple(sorted(used))
        result[tuple(groups)] = CounterTimeGroups(
            resources,
            tuple(groups),
            tuple(leaves[g] for g in groups if g in leaves),
            rectangles,
            program.order,
        )
    return tuple(result.values())


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


def _certificate(program, demands, resources):
    """Establish one complete filtered word at every actual participant."""
    domains = program.domains
    messages = [(r, i, n) for r in resources for i, n in enumerate(demands[r].members)]
    if not messages:
        return None
    first = messages[0][2]
    publishers = tuple(p.actor for p in first.sources)
    receivers = tuple(p.actor for p in first.consumers)
    if any(
        n.mechanism is not NotificationKind.CROSS_EVENT
        or tuple(p.actor for p in n.sources) != publishers
        or tuple(p.actor for p in n.consumers) != receivers
        or (n.source_engine, n.destination_engine)
        != (first.source_engine, first.destination_engine)
        for _, _, n in messages
    ):
        return None
    backlog = sum(demands[r].backlog for r in resources)
    if backlog > CROSS_BACKLOG:
        return None
    words, fifo, clock_shifts, ties = [], [], [], []
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
            for _, tied in groupby(points, key=lambda item: item[0]):
                tied = tuple(tied)
                if len(tied) > 1:
                    ties.append((phase, actor, tied))
            words.append((phase, actor, tuple((r, m) for _, r, m, _ in points)))
            fifo.extend(
                (a[3], b[3]) for a, b in pairwise(points) if a[3].operation is not b[3].operation
            )
    if any(word != words[0][2] for _, _, word in words[1:]):
        return None
    exclusions = []
    for phase, actor, tied in ties:
        seen = overlap = False
        for resource, member, point in (entry[1:] for entry in tied):
            notification = demands[resource].members[member]
            active = any_of(
                *(
                    all_of(*domains.endpoints[point, case].guard.comparisons)
                    for case in domains.cases[notification]
                    if domains.active[notification, case]
                )
            )
            overlap = any_of(overlap, all_of(seen, active))
            seen = any_of(seen, active)
        answer = domains.arithmetic.queries.implies(False, (overlap,))
        if not answer.proved:
            return None
        exclusions.append((phase, actor, tuple(entry[1:3] for entry in tied), overlap, answer))
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
        exclusions=tuple(exclusions),
    )


def _concatenated_word(program, demands, resources, rule):
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
    episodes = []
    for uses in pieces.values():
        projected = tuple(
            replace(demand, members=tuple(n for _, n in uses.get(r, ())))
            for r, demand in enumerate(demands)
        )
        active = tuple(r for r in resources if r in uses)
        certificate = _certificate(program, projected, active)
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
                exclusions=tuple(
                    (phase, actor, tuple(labels[key] for key in word), condition, answer)
                    for phase, actor, word, condition, answer in certificate.exclusions
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
                ends = domains._boundary(left, lc, None, 0, last=True)
                old = next(p for p in ends[0 if phase == "publish" else 1] if p.actor is actor)
                for rc in right_cases:
                    if rc.empty:
                        continue
                    starts = domains._boundary(right, rc, None, 0, last=False)
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
    )


def plan_counter_groups(program, demands, *, handbacks):
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
                candidate = _certificate(program, demands, (*members, number))
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
        certificate = _certificate(program, demands, tuple(numbers))
        if certificate is not None:
            result[certificate.resources] = certificate
        else:
            certificate = _concatenated_word(program, demands, tuple(numbers), handbacks)
            if certificate is not None:
                result[certificate.resources] = certificate
            pack(numbers)
    return tuple(result.values())

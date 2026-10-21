"""Whole mode-2 words for the supported same-clock FIFO packing rule.

Each candidate is a finite set of complete credits, with the same actual
iteration/phase at every publisher and receiver. Deleting messages preserves
its word, quorum and conservative backlog certificate. Sharing introduces
counter-effect FIFO, not a new Wait or controller order. Its local progress
argument uses monotone source native prefixes, never Set-effect completion
inferred from issue order alone.
"""

from dataclasses import dataclass
from itertools import pairwise

from ..scalar import expression
from ..target.ascend910b import Actor
from .event_contracts import CROSS_BACKLOG
from .event_sharing import CounterSharing


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


def _certificate(program, demands, resources, relations):
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
                (a[3], b[3])
                for a, b in pairwise(points)
                if a[3].operation is not b[3].operation
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


def plan_counter_groups(program, demands, relations):
    """Build bounded deterministic candidates before trying physical IDs."""
    groups = []
    for number, demand in enumerate(demands):
        if demand.backlog is None:
            continue
        for i, (members, certificate) in enumerate(groups):
            candidate = _certificate(program, demands, (*members, number), relations)
            if candidate is not None:
                groups[i] = ((*members, number), candidate)
                break
        else:
            groups.append(((number,), None))
    return tuple(certificate for _, certificate in groups if certificate is not None)

"""Conditional boundary transfers of actual effects on one parent clock.

This is a memory transfer summary, with incoming reads left as requirements.
It grants neither initialization nor native completion. Runtime N is never
expanded: short domains depend only on the static predicate period, and long
domains compose a first word, a periodic middle transfer, and a last word.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from .budget import bounded
from .completion import CountDomain, Position, _first
from .region_summary import Access, CellSummary, ReadSpan, _scan
from .scope_guarded_access import BoundaryEffect, _Budget, _cells, _domains, _word
from .symbolic_completion import meet


@dataclass(frozen=True)
class _Transfer:
    reads_before_write: tuple[ReadSpan, ...] = ()
    first_write: Access | None = None
    last_write: Access | None = None
    pending_reads: tuple[ReadSpan, ...] = ()


IDENTITY = _Transfer()
FIRST = Position(False, 0)
LAST = Position(True, 0)


def _then(left, right):
    """Compose consecutive memory words without inventing an incoming value."""
    return _Transfer(
        left.reads_before_write + (right.reads_before_write if left.first_write is None else ()),
        left.first_write or right.first_write,
        right.last_write or left.last_write,
        right.pending_reads
        if right.last_write is not None
        else left.pending_reads + right.pending_reads,
    )


def _at(word, position, budget):
    budget.charge(len(word))
    return _Transfer(
        *_scan(
            (Access(effect.endpoint, position), "R" in effect.modes, "W" in effect.modes)
            for effect in word
        )
    )


def _split(domain, period):
    """At most one static period of short N, then a complete middle period."""
    limit = period + 1
    if domain.exact is not None:
        yield domain, domain.exact <= limit
        return
    stop = limit if domain.maximum is None else min(limit, domain.maximum)
    for count in range(_first(domain), stop + 1, domain.period):
        yield CountDomain(1, 0, exact=count), True
    long = meet(domain, CountDomain(1, 0, minimum=limit + 1))
    if long is not None:
        yield long, False


def _short(records, domain, budget):
    count = domain.exact
    result = IDENTITY
    for generation in range(count):
        kind = (
            "single"
            if count == 1
            else "first"
            if generation == 0
            else "last"
            if generation == count - 1
            else "middle"
        )
        word = _word(records, kind, generation, domain, budget)
        result = _then(result, _at(word, Position(False, generation), budget))
    return result


def _middle(words, domain, period, budget):
    # N>=P+2 guarantees that this complete cycle really occurs in the middle.
    cycle = tuple(
        (generation, effect)
        for generation in range(1, period + 1)
        for effect in words[generation % period]
    )
    budget.charge(len(cycle))
    prefix = _Transfer(
        *_scan(
            (
                Access(effect.endpoint, Position(False, generation)),
                "R" in effect.modes,
                "W" in effect.modes,
            )
            for generation, effect in cycle
        )
    )
    tail_residue = (_first(domain) - 1) % period
    if prefix.first_write is None:
        # Without a periodic writer, every middle reader remains live. These
        # are whole arithmetic spans, not only their last native milestones.
        spans = tuple(
            ReadSpan(
                effect.endpoint,
                Position(False, generation),
                Position(True, 1 + (tail_residue - 1 - generation) % period),
                period,
            )
            for generation, effect in cycle
        )
        return _Transfer(spans, None, None, spans)

    gap = next(
        distance
        for distance in range(1, period + 1)
        if any("W" in effect.modes for effect in words[(tail_residue - distance) % period])
    )
    word = words[(tail_residue - gap) % period]
    final = max(index for index, effect in enumerate(word) if "W" in effect.modes)
    last = Access(word[final].endpoint, Position(True, gap))
    pending = []
    for distance in range(gap, 0, -1):
        suffix = word[final + 1 :] if distance == gap else words[(tail_residue - distance) % period]
        budget.charge(len(suffix))
        pending.extend(
            ReadSpan.one(Access(effect.endpoint, Position(True, distance)))
            for effect in suffix
            if "R" in effect.modes
        )
    return _Transfer(prefix.reads_before_write, prefix.first_write, last, tuple(pending))


def _long(records, domain, period, budget):
    first = _at(_word(records, "first", 0, domain, budget), FIRST, budget)
    words = {
        residue: _word(records, "middle", residue, domain, budget) for residue in range(period)
    }
    middle = _middle(words, domain, period, budget)
    last = _at(_word(records, "last", (_first(domain) - 1) % period, domain, budget), LAST, budget)
    return _then(_then(first, middle), last)


@bounded
def summarize_frontiers(
    records: Iterable[BoundaryEffect], *, statistics: dict | None = None
) -> tuple[CellSummary, ...]:
    """Summarize incoming reads, first/last writes and surviving readers.

    Endpoints keep their actual actor/native-inner identity. Access positions
    and ReadSpan bounds refer only to the parent generation. Count predicates,
    sparse byte cells, logical word order and supported first/middle/last
    geometry use the existing guarded-access contracts; unknown geometry is
    rejected. N=0 and domains without effects are identity transfers, with no
    cells. Reads before the first write require an incoming value; this API has
    no initialization assumption and supplies no definition or completion fact.

    Identical effect sequences share transfer computation within this call.
    Physical cells are still emitted separately, including adjacent rows and
    cells separated by holes. ``transfer_visits`` counts only actual transfer
    computation; ``visits`` also counts indexing, signatures and output cells.
    """
    records, budget, summaries = tuple(records), _Budget(), []
    cells = short_domains = long_domains = 0
    reused_cells = transfer_visits = 0
    transfers = {}
    max_period = 1
    for root, lo, hi, work in _cells(records, budget):
        cells += 1
        # _domains reads only activation; _word additionally reads endpoint,
        # modes and logical_order. The transfer never uses root or byte bounds.
        budget.charge(len(work))
        signature = tuple(
            (effect.endpoint, effect.modes, effect.activation, effect.logical_order)
            for effect in work
        )
        if signature not in transfers:
            before = budget.visits
            frontiers, short_count, long_count, word_period = [], 0, 0, 1
            for partition, period in _domains(work, budget):
                word_period = max(word_period, period)
                for domain, short in _split(partition, period):
                    budget.charge(1)
                    if short:
                        short_count += 1
                        frontier = _short(work, domain, budget)
                    else:
                        long_count += 1
                        frontier = _long(work, domain, period, budget)
                    if frontier != IDENTITY:
                        frontiers.append((domain, frontier))
            transfers[signature] = tuple(frontiers), short_count, long_count, word_period
            transfer_visits += budget.visits - before
        else:
            reused_cells += 1
        frontiers, short_count, long_count, word_period = transfers[signature]
        short_domains += short_count
        long_domains += long_count
        max_period = max(max_period, word_period)
        for domain, frontier in frontiers:
            budget.charge(1 + len(frontier.reads_before_write) + len(frontier.pending_reads))
            summaries.append(
                CellSummary(
                    root,
                    lo,
                    hi,
                    domain,
                    frontier.reads_before_write,
                    frontier.first_write,
                    frontier.last_write,
                    frontier.pending_reads,
                )
            )
    if statistics is not None:
        statistics.update(
            visits=budget.visits,
            cells=cells,
            summaries=len(summaries),
            short_domains=short_domains,
            long_domains=long_domains,
            max_period=max_period,
            unique_transfer_words=len(transfers),
            reused_cells=reused_cells,
            transfer_visits=transfer_visits,
        )
    return tuple(summaries)

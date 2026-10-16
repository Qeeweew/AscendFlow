"""Compose access transfers of sequential regions, retaining open obligations.

This is an analysis, not a completion certificate. It emits the RAW/WAR/WAW
relations a parent must discharge using real native completion and event paths.
The final checker must build its inputs from actual IR, not accept user summaries.

Supported domains: one shared immutable runtime bound and any fixed-count
regions. Different runtime bounds, frame drift and mixed ownership are not
silently combined. Work is bounded by static pieces/domains, never runtime N.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import lcm

from xdsl.ir import SSAValue

from .budget import bounded, poll
from .diagnostics import require
from .periodic import bounded_period
from .region_patterns import HazardCoverage, HazardPattern, PhysicalRanges, materialize_hazards
from .region_summary import Access, CountDomain, ReadSpan, RegionSummary
from .scalar import Expr

MAX_SEQUENCE_VISITS = 1_000_000


@dataclass(frozen=True)
class Part:
    """One occurrence of a region, with a distinct name even on repeated calls."""

    name: str
    summary: RegionSummary


@dataclass(frozen=True)
class Endpoint:
    region: str
    access: Access | ReadSpan
    constant_count: int | None


@dataclass(frozen=True)
class Obligation:
    storage: SSAValue
    start: int
    stop: int
    domain: CountDomain
    hazard: str
    source: Endpoint
    destination: Endpoint


@dataclass(frozen=True)
class Cell:
    storage: SSAValue
    start: int
    stop: int
    domain: CountDomain
    reads_before_write: tuple[Endpoint, ...]
    first_write: Endpoint | None
    last_write: Endpoint | None
    pending_reads: tuple[Endpoint, ...]


@dataclass(frozen=True)
class _Frontier:
    reads_before_write: tuple[Endpoint, ...]
    first_write: Endpoint | None
    last_write: Endpoint | None
    pending_reads: tuple[Endpoint, ...]


@dataclass
class _Work:
    visits: int = 0

    def charge(self, amount, kind):
        self.visits += amount
        require(self.visits <= MAX_SEQUENCE_VISITS, f"sequence {kind} budget exceeded")


@dataclass(frozen=True)
class SequenceSummary:
    bound: Expr | None
    bound_value: SSAValue | None
    cells: tuple[Cell, ...]
    obligations: tuple[Obligation, ...]
    visits: int

    def for_count(self, count):
        require(type(count) is int and count >= 0, "negative/invalid sequence length")
        return tuple(cell for cell in self.cells if cell.domain.contains(count))

    def invocation_dependencies(self, *, fresh=()):
        """Exit accesses of call p against first accesses of p+1, same bound."""
        fresh = frozenset(fresh)
        result = []
        for cell in self.cells:
            if cell.storage in fresh:
                continue  # Disjoint call coordinates were checked by the caller.

            def add(hazard, source, destination, *, cell=cell):
                result.append(
                    Obligation(
                        cell.storage,
                        cell.start,
                        cell.stop,
                        cell.domain,
                        hazard,
                        source,
                        destination,
                    )
                )

            if cell.last_write is not None:
                for reader in cell.reads_before_write:
                    add("RAW", cell.last_write, reader)
                if cell.first_write is not None:
                    add("WAW", cell.last_write, cell.first_write)
            if cell.first_write is not None:
                for reader in cell.pending_reads:
                    add("WAR", reader, cell.first_write)
        return tuple(result)


@dataclass(frozen=True)
class SequenceTransfer:
    """One complete semantic transfer word over exact allocation ranges."""

    word: tuple
    ranges: tuple[PhysicalRanges, ...]


@dataclass(frozen=True)
class SequencePatternSummary:
    bound: Expr | None
    bound_value: SSAValue | None
    frontiers: tuple[SequenceTransfer, ...]
    patterns: tuple[HazardPattern, ...]
    visits: int

    def materialize(self, *, materialize_cells=True):
        """Explicit diagnostic/reference expansion; normal consumers use patterns."""
        output, work = [], _Work(self.visits)
        if materialize_cells:
            for group in self.frontiers:
                for ranges in group.ranges:
                    for lo, hi, order in ranges.pieces():
                        for index, (domain, frontier, _) in enumerate(group.word):
                            work.charge(
                                1 + len(frontier.reads_before_write) + len(frontier.pending_reads),
                                "output-piece",
                            )
                            output.append(
                                (
                                    (order, index),
                                    Cell(
                                        ranges.storage,
                                        lo,
                                        hi,
                                        domain,
                                        frontier.reads_before_write,
                                        frontier.first_write,
                                        frontier.last_write,
                                        frontier.pending_reads,
                                    ),
                                )
                            )
            output.sort(key=lambda item: item[0])
        work.charge(sum(pattern.physical_count for pattern in self.patterns), "obligation")
        return SequenceSummary(
            self.bound,
            self.bound_value,
            tuple(item[1] for item in output),
            materialize_hazards(self.patterns),
            work.visits,
        )


def _domains(entries):
    """Partition N only where a participating *dynamic* region changes transfer.

    Fixed-count children are evaluated at their own count. Even a constant
    million-trip child must not create a million-domain sequence analysis.
    """
    period, boundary = 1, 0
    for part, cells in entries:
        if part.summary.constant_count is not None:
            continue
        for cell in cells:
            d = cell.domain
            require(d.maximum is None, "bounded child transfer summaries are not yet supported")
            require(d.period > 0 and 0 <= d.residue < d.period, "invalid child count domain")
            if d.exact is not None:
                require(d.exact >= 0, "negative exact child domain")
                boundary = max(boundary, d.exact + 1)
            else:
                minimum = d.period if d.minimum is None else d.minimum
                require(minimum >= 0, "negative child domain minimum")
                boundary = max(boundary, minimum)
                period = bounded_period(lcm(period, d.period), "sequential boundary domain")
    require(boundary + period <= MAX_SEQUENCE_VISITS, "sequence count-domain budget exceeded")
    for count in range(boundary):
        yield CountDomain(1, 0, exact=count), count
    for residue in range(period):
        first = boundary + (residue - boundary) % period
        yield CountDomain(period, residue, minimum=boundary), first


def _transfers(entries, work):
    """Compose one complete semantic entry pattern, without physical bounds."""
    result = []
    for domain, count in _domains(entries):
        incoming, pending, obligations = [], [], []
        first, last = None, None

        def add(hazard, source, destination, *, obligations=obligations):
            work.charge(1, "transfer obligation")
            obligations.append((hazard, source, destination))

        for part, cells in entries:
            # Inactive domains are examined too; sharing must not hide them.
            work.charge(len(cells), "frontier")
            poll("sequential access composition", visits=work.visits)
            local_count = part.summary.constant_count
            candidates = [
                cell
                for cell in cells
                if cell.domain.contains(count if local_count is None else local_count)
            ]
            require(len(candidates) <= 1, "overlapping child byte/count domains")
            if not candidates:
                continue
            cell = candidates[0]
            work.charge(len(cell.reads_before_write) + len(cell.pending_reads) + 2, "endpoint")

            def endpoint(access, *, part=part, local_count=local_count):
                return Endpoint(part.name, access, local_count)

            readers = tuple(endpoint(reader) for reader in cell.reads_before_write)
            if last is not None:
                for reader in readers:
                    add("RAW", last, reader)
            else:
                incoming.extend(readers)
            if cell.first_write is not None:
                writer = endpoint(cell.first_write)
                if last is not None:
                    add("WAW", last, writer)
                for reader in pending:
                    add("WAR", reader, writer)
                require(cell.last_write is not None, "first writer needs an exit writer")
                first = first or writer
                last = endpoint(cell.last_write)
                pending = [endpoint(reader) for reader in cell.pending_reads]
            else:
                require(cell.last_write is None, "exit writer needs a first writer")
                pending.extend(endpoint(reader) for reader in cell.pending_reads)
        if incoming or first is not None or pending:
            work.charge(1, "transfer")
            result.append(
                (
                    domain,
                    _Frontier(tuple(incoming), first, last, tuple(pending)),
                    tuple(obligations),
                )
            )
    return tuple(result)


@bounded
def summarize_sequence_patterns(parts, *, obligation_source_regions=None, statistics=None):
    """Compose shared words without constructing physical cells or obligations.

    The sweep indexes one word occurrence per exact physical piece. It keeps
    multiplicity, original allocations and all count-domain checks. Only a new
    entry-word combination runs ``_transfers``; hazards then reference shared
    range collections. Source selection happens after the complete transfer.
    """
    sources = None if obligation_source_regions is None else frozenset(obligation_source_regions)
    parts = tuple(parts)
    require(
        len({part.name for part in parts}) == len(parts), "sequence region names must be unique"
    )
    require(all(part.name for part in parts), "sequence regions need nonempty identities")
    bounds = {part.summary.bound for part in parts if part.summary.constant_count is None}
    bound_values = {
        part.summary.bound_value for part in parts if part.summary.constant_count is None
    }
    require(
        None not in bounds
        and len(bounds) <= 1
        and len(bound_values) == len(bounds)
        and all(isinstance(value, SSAValue) for value in bound_values),
        "sequence needs one shared immutable bound with its actual SSA identity",
    )
    bound = next(iter(bounds)) if bounds else None
    bound_value = next(iter(bound_values)) if bound_values else None
    roots, root_order, words, word_values = {}, {}, {}, []
    word_cases, duplicate_pairs, work = [], {}, _Work()
    input_cells = input_pieces = 0
    for index, part in enumerate(parts):
        origins = dict(part.summary.origins)
        require(
            len(origins) == len(part.summary.origins)
            and set(origins) == set(part.summary.roots)
            and all(isinstance(root, SSAValue) for root in origins.values()),
            "sequence requires original SSA allocation identities",
        )
        shapes = {}
        for group in part.summary.groups:
            require(
                group.word and (not group.orders or len(group.orders) == len(group.word)),
                "invalid child transfer word",
            )
            word = words.get(group.word)
            if word is None:
                word = len(word_values)
                words[group.word] = word
                word_values.append(group.word)
                word_cases.append(frozenset(group.word))
                require(len(word_cases[-1]) == len(group.word), "duplicate child byte/count cell")
                work.charge(
                    sum(
                        2 + len(case.reads_before_write) + len(case.pending_reads)
                        for case in group.word
                    ),
                    "input pattern",
                )
            for ranges in group.ranges:
                require(ranges.storage in origins and ranges.spans, "invalid child byte cell")
                root = origins[ranges.storage]
                first = index, ranges.first_order
                root_order[root] = min(root_order.get(root, first), first)
                events = roots.setdefault(root, {})
                for lo, hi, _ in ranges.pieces():
                    require(
                        type(lo) is int and type(hi) is int and 0 <= lo < hi,
                        "invalid child byte cell",
                    )
                    # Exact duplicates are invalid even in inactive fixed-count
                    # domains. Compare a repeated word pair only once, without
                    # expanding its cases over every physical interval.
                    shape = shapes.setdefault((root, lo, hi), [])
                    for previous in shape:
                        pair = min(previous, word), max(previous, word)
                        duplicate = duplicate_pairs.get(pair)
                        if duplicate is None:
                            duplicate = bool(word_cases[previous] & word_cases[word])
                            duplicate_pairs[pair] = duplicate
                        require(not duplicate, "duplicate child byte/count cell")
                    shape.append(word)
                    events.setdefault(lo, []).append((True, index, input_pieces, word))
                    events.setdefault(hi, []).append((False, index, input_pieces, word))
                    input_pieces += 1
                    input_cells += len(group.word)
                    work.charge(1, "input-piece")
                    if input_pieces % 256 == 0:
                        poll("sequential access indexing", pieces=input_pieces)
    transfers = {}
    physical_intervals = reused_intervals = transfer_visits = 0
    input_pattern_visits = work.visits
    for root in sorted(roots, key=root_order.__getitem__):
        events, active = roots[root], {}
        for point, stop in pairwise(sorted(events)):
            for add, index, piece, word in events[point]:
                if add:
                    active.setdefault(index, {})[piece] = word
                else:
                    del active[index][piece]
                    if not active[index]:
                        del active[index]
            if not active:
                continue
            indices = sorted(active)
            work.charge(sum(len(active[i]) for i in indices), "entry pattern")
            signature = tuple((i, tuple(sorted(active[i].values()))) for i in indices)
            if signature not in transfers:
                before = work.visits
                entries = tuple(
                    (parts[i], tuple(case for word in word_ids for case in word_values[word]))
                    for i, word_ids in signature
                )
                transfers[signature] = _transfers(entries, work), {}
                transfer_visits += work.visits - before
            else:
                reused_intervals += 1
            transfers[signature][1].setdefault(root, []).append((point, stop, physical_intervals))
            physical_intervals += 1
            work.charge(1, "coverage")
            if physical_intervals % 256 == 0:
                poll(
                    "sequential access composition",
                    visits=work.visits,
                    intervals=physical_intervals,
                )
    frontiers, hazards = [], {}
    represented_cells = represented_obligations = omitted_obligations = 0
    for semantic, range_lists in transfers.values():
        ranges = tuple(PhysicalRanges(root, tuple(spans)) for root, spans in range_lists.items())
        pieces = sum(len(physical.spans) for physical in ranges)
        frontiers.append(SequenceTransfer(semantic, ranges))
        represented_cells += len(semantic) * pieces
        for case_index, (domain, _, obligations) in enumerate(semantic):
            for hazard_index, (hazard, source, destination) in enumerate(obligations):
                if sources is not None and source.region not in sources:
                    omitted_obligations += pieces
                    continue
                key = domain, hazard, source, destination
                hazards.setdefault(key, []).append(
                    HazardCoverage(ranges, (case_index, hazard_index))
                )
                represented_obligations += pieces
                work.charge(1, "hazard pattern")
    patterns = tuple(
        sorted(
            (HazardPattern(*key, tuple(coverage)) for key, coverage in hazards.items()),
            key=lambda pattern: pattern.first_order,
        )
    )
    if statistics is not None:
        statistics.update(
            input_cells=input_cells,
            input_physical_pieces=input_pieces,
            input_pattern_visits=input_pattern_visits,
            physical_intervals=physical_intervals,
            unique_cell_patterns=len(set().union(*word_cases)) if word_cases else 0,
            unique_transfer_words=len(words),
            unique_entry_patterns=len(transfers),
            reused_intervals=reused_intervals,
            transfer_visits=transfer_visits,
            hazard_patterns=len(patterns),
            hazard_coverages=sum(len(pattern.coverage) for pattern in patterns),
            represented_cells=represented_cells,
            represented_obligations=represented_obligations,
            materialized_cells=0,
            materialized_obligations=0,
            omitted_cells=represented_cells,
            omitted_obligations=omitted_obligations,
            materialized_visits=0,
            visits=work.visits,
        )
    return SequencePatternSummary(bound, bound_value, tuple(frontiers), patterns, work.visits)


@bounded
def summarize_sequence(
    parts, *, materialize_cells=True, obligation_source_regions=None, statistics=None
):
    """Return exact sequence frontiers and unresolved cross-region hazards.

    Empty dynamic children contribute identity; surrounding constant-count
    regions still execute. Read-only children preserve complete reader spans,
    without assuming their last issued read retires the previous ones.

    Identical entry patterns share domain selection and transfer computation
    only within this call. By default each physical interval gets its own cells
    and obligations, including holes and adjacent pieces. Output projection may
    omit cells or materialize obligations only from selected source regions.
    It preserves all transfer computation, reader histories and domain checks;
    omitted cells are not available for later invocation_dependencies(). The
    visit budget charges input signatures, unique transfers and materialized
    output records. Statistics separately count omitted output objects.
    """
    require(type(materialize_cells) is bool, "sequence cell projection must be an explicit boolean")
    sources = None if obligation_source_regions is None else frozenset(obligation_source_regions)
    parts = tuple(parts)
    require(
        len({part.name for part in parts}) == len(parts), "sequence region names must be unique"
    )
    require(all(part.name for part in parts), "sequence regions need nonempty identities")
    bounds = {part.summary.bound for part in parts if part.summary.constant_count is None}
    bound_values = {
        part.summary.bound_value for part in parts if part.summary.constant_count is None
    }
    require(
        None not in bounds
        and len(bounds) <= 1
        and len(bound_values) == len(bounds)
        and all(isinstance(value, SSAValue) for value in bound_values),
        "sequence needs one shared immutable bound with its actual SSA identity",
    )
    bound = next(iter(bounds)) if bounds else None
    bound_value = next(iter(bound_values)) if bound_values else None
    # Sweep exact physical byte pieces separately for each allocation. Names
    # only locate the original SSA allocation inside each already extracted child.
    roots, cell_patterns, work = {}, {}, _Work()
    input_cells = 0
    for index, part in enumerate(parts):
        require(
            input_cells + len(part.summary.cells) <= MAX_SEQUENCE_VISITS,
            "sequence input-piece budget exceeded",
        )
        origins = dict(part.summary.origins)
        require(
            len(origins) == len(part.summary.origins)
            and set(origins) == set(part.summary.roots)
            and all(isinstance(root, SSAValue) for root in origins.values()),
            "sequence requires original SSA allocation identities",
        )
        require(
            len(set(part.summary.cells)) == len(part.summary.cells),
            "duplicate child byte/count cell",
        )
        for cell in part.summary.cells:
            require(
                cell.storage in origins and 0 <= cell.start < cell.stop, "invalid child byte cell"
            )
            root = origins[cell.storage]
            # Coordinates and allocation identity do not affect a transfer.
            # Keep every semantic field, including complete reader families and
            # definition bindings, before interning a small per-analysis ID.
            signature = (
                cell.domain,
                cell.reads_before_write,
                cell.first_write,
                cell.last_write,
                cell.pending_reads,
                cell.definitions,
            )
            work.charge(len(cell.reads_before_write) + len(cell.pending_reads) + 2, "input pattern")
            pattern = cell_patterns.setdefault(signature, len(cell_patterns))
            events = roots.setdefault(root, {})
            events.setdefault(cell.start, []).append((True, index, cell, pattern))
            events.setdefault(cell.stop, []).append((False, index, cell, pattern))
            input_cells += 1
            require(input_cells <= MAX_SEQUENCE_VISITS, "sequence input-piece budget exceeded")
            if input_cells % 256 == 0:
                poll("sequential access indexing", pieces=input_cells)

    output, obligations, transfers = [], [], {}
    physical_intervals = reused_intervals = transfer_visits = materialized_visits = 0
    omitted_cells = omitted_obligations = 0
    input_pattern_visits = work.visits
    for root, events in roots.items():
        active = {}
        points = sorted(events)
        for point, stop in pairwise(points):
            for add, index, cell, pattern in events[point]:
                if add:
                    active.setdefault(index, {})[cell] = pattern
                else:
                    del active[index][cell]
                    if not active[index]:
                        del active[index]
            if not active:
                continue  # Never fill holes with a bounding box.
            indices = sorted(active)
            # Sorting interned IDs preserves multiplicity: two distinct cells
            # with equal semantics must still diagnose overlapping domains.
            work.charge(sum(len(active[i]) for i in indices), "entry pattern")
            signature = tuple(
                (parts[i].name, parts[i].summary.constant_count, tuple(sorted(active[i].values())))
                for i in indices
            )
            physical_intervals += 1
            if physical_intervals % 256 == 0:
                poll(
                    "sequential access composition",
                    visits=work.visits,
                    intervals=physical_intervals,
                )
            if signature not in transfers:
                before = work.visits
                entries = tuple((parts[i], tuple(active[i])) for i in indices)
                transfers[signature] = _transfers(entries, work)
                transfer_visits += work.visits - before
            else:
                reused_intervals += 1
            before = work.visits
            for domain, frontier, hazards in transfers[signature]:
                for hazard, source, destination in hazards:
                    if sources is not None and source.region not in sources:
                        omitted_obligations += 1
                        continue
                    work.charge(1, "obligation")
                    obligations.append(
                        Obligation(root, point, stop, domain, hazard, source, destination)
                    )
                if materialize_cells:
                    work.charge(
                        len(frontier.reads_before_write) + len(frontier.pending_reads) + 1,
                        "output-piece",
                    )
                    output.append(
                        Cell(
                            root,
                            point,
                            stop,
                            domain,
                            frontier.reads_before_write,
                            frontier.first_write,
                            frontier.last_write,
                            frontier.pending_reads,
                        )
                    )
                else:
                    omitted_cells += 1
            materialized_visits += work.visits - before
    if statistics is not None:
        statistics.update(
            input_cells=input_cells,
            input_pattern_visits=input_pattern_visits,
            physical_intervals=physical_intervals,
            unique_cell_patterns=len(cell_patterns),
            unique_entry_patterns=len(transfers),
            reused_intervals=reused_intervals,
            transfer_visits=transfer_visits,
            materialized_cells=len(output),
            materialized_obligations=len(obligations),
            omitted_cells=omitted_cells,
            omitted_obligations=omitted_obligations,
            materialized_visits=materialized_visits,
            visits=work.visits,
        )
    return SequenceSummary(bound, bound_value, tuple(output), tuple(obligations), work.visits)

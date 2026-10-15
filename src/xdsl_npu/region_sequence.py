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
from .model import require
from .periodic import bounded_period
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


@bounded
def summarize_sequence(parts):
    """Return exact sequence frontiers and unresolved cross-region hazards.

    Empty dynamic children contribute identity; surrounding constant-count
    regions still execute. Read-only children preserve complete reader spans,
    without assuming their last issued read retires the previous ones.
    """
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
    roots = {}
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
            events = roots.setdefault(root, {})
            events.setdefault(cell.start, []).append((True, index, cell))
            events.setdefault(cell.stop, []).append((False, index, cell))
            input_cells += 1
            require(input_cells <= MAX_SEQUENCE_VISITS, "sequence input-piece budget exceeded")
            if input_cells % 256 == 0:
                poll("sequential access indexing", pieces=input_cells)

    output, obligations, visits = [], [], 0
    for root, events in roots.items():
        active = {}
        points = sorted(events)
        for point, stop in pairwise(points):
            for add, index, cell in events[point]:
                if add:
                    active.setdefault(index, set()).add(cell)
                else:
                    active[index].remove(cell)
                    if not active[index]:
                        del active[index]
            if not active:
                continue  # Never fill holes with a bounding box.
            entries = [(parts[i], tuple(active[i])) for i in sorted(active)]
            for domain, count in _domains(entries):
                incoming, pending = [], []
                first, last = None, None

                def add(
                    hazard, source, destination, *, root=root, point=point, stop=stop, domain=domain
                ):
                    nonlocal visits
                    visits += 1
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence obligation budget exceeded")
                    obligations.append(
                        Obligation(root, point, stop, domain, hazard, source, destination)
                    )

                for part, cells in entries:
                    # Charge actual examined domains, including inactive branches.
                    visits += len(cells)
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence frontier budget exceeded")
                    poll(
                        "sequential access composition", visits=visits, obligations=len(obligations)
                    )
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
                    visits += len(cell.reads_before_write) + len(cell.pending_reads) + 2
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence endpoint budget exceeded")

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
                    visits += len(incoming) + len(pending) + 1
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence output-piece budget exceeded")
                    output.append(
                        Cell(
                            root, point, stop, domain, tuple(incoming), first, last, tuple(pending)
                        )
                    )
    return SequenceSummary(bound, bound_value, tuple(output), tuple(obligations), visits)

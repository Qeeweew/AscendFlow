"""Shared semantic words with exact physical coverage, never completion facts.

Ranges retain holes, allocation identity, and source occurrence order. A word
is created before attaching physical ranges; normal consumers never expand its
cases or hazards once per byte piece. Materialization is an explicit adapter.
"""

from dataclasses import dataclass
from functools import cached_property

from .completion import CountDomain
from .region_summary import Access, CellSummary, DefinitionRun, ReadSpan, RegionSummary


@dataclass(frozen=True)
class TransferCase:
    domain: CountDomain
    reads_before_write: tuple[ReadSpan, ...]
    first_write: Access | None
    last_write: Access | None
    pending_reads: tuple[ReadSpan, ...]
    definitions: tuple[DefinitionRun, ...] = ()

    @classmethod
    def from_cell(cls, cell):
        return cls(
            cell.domain,
            cell.reads_before_write,
            cell.first_write,
            cell.last_write,
            cell.pending_reads,
            cell.definitions,
        )


@dataclass(frozen=True)
class PhysicalRanges:
    storage: object
    # (start, stop, occurrence order); shift shares one base across query views.
    spans: tuple[tuple[int, int, int], ...]
    shift: int = 0

    def pieces(self):
        for lo, hi, order in self.spans:
            yield lo + self.shift, hi + self.shift, order

    @cached_property
    def first_order(self):
        return min(order for _, _, order in self.spans)


@dataclass(frozen=True)
class FrontierGroup:
    word: tuple[TransferCase, ...]
    ranges: tuple[PhysicalRanges, ...]
    # Optional original case positions, for interleaved rotated range views.
    orders: tuple[int, ...] = ()


def materialize_frontiers(groups):
    result = []
    for group in groups:
        for ranges in group.ranges:
            for lo, hi, order in ranges.pieces():
                for index, case in enumerate(group.word):
                    case_order = group.orders[index] if group.orders else index
                    result.append(
                        (
                            order,
                            case_order,
                            CellSummary(
                                ranges.storage,
                                lo,
                                hi,
                                case.domain,
                                case.reads_before_write,
                                case.first_write,
                                case.last_write,
                                case.pending_reads,
                                case.definitions,
                            ),
                        )
                    )
    result.sort(key=lambda item: item[:2])
    return tuple(item[2] for item in result)


def group_frontiers(cells, *, order=0):
    """Admit existing independent occupancy records without discarding history."""
    groups = {}
    for index, cell in enumerate(cells, order):
        word = (TransferCase.from_cell(cell),)
        groups.setdefault(word, {}).setdefault(cell.storage, []).append(
            (cell.start, cell.stop, index)
        )
    return tuple(
        FrontierGroup(
            word, tuple(PhysicalRanges(root, tuple(spans)) for root, spans in roots.items())
        )
        for word, roots in groups.items()
    )


@dataclass(frozen=True)
class PatternRegion:
    roots: tuple[str, ...]
    groups: tuple[FrontierGroup, ...]
    visits: int
    bound: object
    constant_count: int | None
    origins: tuple
    bound_value: object = None

    def materialize(self):
        return RegionSummary(
            self.roots,
            materialize_frontiers(self.groups),
            self.visits,
            self.bound,
            self.constant_count,
            self.origins,
            self.bound_value,
        )


@dataclass(frozen=True)
class PatternPart:
    name: str
    summary: PatternRegion

    def materialize(self):
        from .region_sequence import Part

        return Part(self.name, self.summary.materialize())


@dataclass(frozen=True)
class HazardCoverage:
    ranges: tuple[PhysicalRanges, ...]
    # The case and hazard's order inside its sequence transfer word.
    order: tuple[int, int]

    @cached_property
    def first_order(self):
        return min(ranges.first_order for ranges in self.ranges), *self.order


@dataclass(frozen=True)
class HazardPattern:
    domain: CountDomain
    hazard: str
    source: object
    destination: object
    coverage: tuple[HazardCoverage, ...]

    @cached_property
    def first_order(self):
        return min(coverage.first_order for coverage in self.coverage)

    @cached_property
    def physical_count(self):
        return sum(len(ranges.spans) for coverage in self.coverage for ranges in coverage.ranges)

    def first_range(self):
        coverage = min(self.coverage, key=lambda item: item.first_order)
        ranges = min(coverage.ranges, key=lambda item: item.first_order)
        lo, hi, _ = min(ranges.pieces(), key=lambda item: item[2])
        return ranges.storage, lo, hi


def materialize_hazards(patterns):
    from .region_sequence import Obligation

    result = []
    for pattern in patterns:
        for coverage in pattern.coverage:
            for ranges in coverage.ranges:
                for lo, hi, order in ranges.pieces():
                    result.append(
                        (
                            (order, *coverage.order),
                            Obligation(
                                ranges.storage,
                                lo,
                                hi,
                                pattern.domain,
                                pattern.hazard,
                                pattern.source,
                                pattern.destination,
                            ),
                        )
                    )
    result.sort(key=lambda item: item[0])
    return tuple(item[1] for item in result)

"""Actual byte frontiers across repeated serial queries, retaining ring phase.

One query's memory transfer comes from its physical native frontiers. A static
number of translated copies finds the first subsequent touch of every slot;
untouched bytes carry their old writer/readers through intermediate queries.
No completion edge or synchronization instruction is supplied by this module.
"""

from dataclasses import dataclass, replace

from .budget import bounded, poll
from .completion import CountDomain
from .diagnostics import require
from .loop_model import slope
from .region_patterns import (
    FrontierGroup,
    HazardPattern,
    PatternPart,
    PatternRegion,
    PhysicalRanges,
    group_frontiers,
    materialize_frontiers,
    materialize_hazards,
)
from .region_sequence import Obligation, Part, summarize_sequence, summarize_sequence_patterns
from .region_summary import CellSummary, RegionSummary
from .scope_boundary_summary import summarize_frontier_patterns, summarize_frontiers
from .scope_domains import Activation
from .scope_guarded_access import BoundaryEffect
from .symbolic_completion import meet

MAX_QUERY_SUMMARY_RECORDS = 262_144


@dataclass(frozen=True)
class QuerySummary:
    # Cell endpoints use access.endpoints to recover actual call/native/inner
    # identities. Their Access.position is the independent parent KV clock.
    cells: tuple[CellSummary, ...]
    parts: tuple[Part, ...]
    obligations: tuple[Obligation, ...]
    statistics: dict


@dataclass(frozen=True)
class QueryPatternSummary:
    """Shared semantic words and hazards with exact physical coverage."""

    frontiers: tuple[FrontierGroup, ...]
    parts: tuple[PatternPart, ...]
    patterns: tuple[HazardPattern, ...]
    statistics: dict

    def materialize(self):
        """Explicit full/reference adapter; normal planning never calls this."""
        statistics = dict(self.statistics)
        statistics["materialized_cells"] = self.statistics["summaries"]
        statistics["sequence_materialized_obligations"] = sum(
            pattern.physical_count for pattern in self.patterns
        )
        return QuerySummary(
            materialize_frontiers(self.frontiers),
            tuple(part.materialize() for part in self.parts),
            materialize_hazards(self.patterns),
            statistics,
        )


def _records(program, access, roots, reordered):
    guarded = getattr(access, "guarded_roots", frozenset())
    records = [
        effect for effect in getattr(access, "effects", ()) if effect.storage in roots - reordered
    ]
    translations = {}
    for owner, (actor, data) in enumerate(access.loop.actors.items()):
        for index, task in enumerate(data.program.tasks):
            for ref, mode in zip(task.refs, task.modes, strict=True):
                root = access.loop.root(actor, ref)
                if root not in roots:
                    continue
                require(
                    not slope(ref)
                    and not ref.count_offset
                    and not ref.count_core_stride
                    and ref.prefix_extent is None,
                    f"{root}: query reuse needs bounded periodic parent byte addresses",
                )
                translations.setdefault(root, set()).add(ref.core_terms)
                if root in guarded or root in reordered:
                    continue
                period = access.loop.storage_periods[root]
                domain = (
                    CountDomain(1, 0, minimum=0)
                    if program.count is None
                    else CountDomain(1, 0, exact=program.count)
                )
                for residue in range(period):
                    activation = Activation(domain, period=period, residues=(residue,))
                    for lo, hi in ref.segments(residue):
                        records.append(
                            BoundaryEffect(
                                (actor, task.uid),
                                root,
                                lo,
                                hi,
                                mode,
                                activation,
                                (task.logical_phase, owner, index),
                            )
                        )
                require(
                    len(records) <= MAX_QUERY_SUMMARY_RECORDS,
                    "query boundary effect budget exceeded",
                )
            poll("query boundary effects", actor=actor, task=task.uid, records=len(records))
    require(
        all(len(values) == 1 for values in translations.values()),
        "query reuse needs one common physical group translation per allocation",
    )
    return records


def _rotated(cell, contract, query):
    """Express query q relative to query zero's physical slot permutation."""
    if contract.kind != "epoch_ring":
        return (cell,)
    tile, slots = contract.frame_bytes, contract.slots
    require(
        tile is not None
        and tile > 0
        and cell.start // tile == (cell.stop - 1) // tile
        and cell.stop <= tile * slots,
        "query ring frontier crosses its proved owner/slot frame",
    )
    old = cell.start // tile
    result = []
    for residue in range(slots):
        domain = meet(cell.domain, CountDomain(slots, residue, minimum=1))
        if domain is None:
            continue
        new = (old + query * residue) % slots
        delta = (new - old) * tile
        result.append(
            replace(cell, start=cell.start + delta, stop=cell.stop + delta, domain=domain)
        )
    return tuple(result)


def _rotation_views(groups, mapping):
    """Partition actual ring slots once, sharing domains and base range tuples."""
    guarded_words, views = {}, []
    for group in groups:
        for ranges in group.ranges:
            contract = mapping.contracts[ranges.storage]
            if contract.kind != "epoch_ring":
                views.append((group.word, group.orders, ranges, None, None))
                continue
            tile, slots = contract.frame_bytes, contract.slots
            require(
                tile is not None and tile > 0,
                "query ring frontier crosses its proved owner/slot frame",
            )
            by_slot = {}
            for lo, hi, order in ranges.pieces():
                require(
                    0 <= lo < hi <= tile * slots and lo // tile == (hi - 1) // tile,
                    "query ring frontier crosses its proved owner/slot frame",
                )
                by_slot.setdefault(lo // tile, []).append((lo, hi, order))
            for old, spans in by_slot.items():
                base = tuple(spans)
                for residue in range(slots):
                    key = group.word, group.orders, slots, residue
                    guarded = guarded_words.get(key)
                    if guarded is None:
                        word, orders = [], []
                        for index, case in enumerate(group.word):
                            domain = meet(case.domain, CountDomain(slots, residue, minimum=1))
                            if domain is not None:
                                word.append(replace(case, domain=domain))
                                case_order = group.orders[index] if group.orders else index
                                orders.append(case_order * slots + residue)
                        guarded = tuple(word), tuple(orders)
                        guarded_words[key] = guarded
                    word, orders = guarded
                    if word:
                        views.append(
                            (
                                word,
                                orders,
                                PhysicalRanges(ranges.storage, base),
                                (old, residue, tile),
                                slots,
                            )
                        )
    return views


def _rotated_patterns(views, query):
    # Group by the small word identity, never hash its complete range list.
    groups = {}
    for word, orders, ranges, phase, slots in views:
        if phase is not None:
            old, residue, tile = phase
            delta = ((old + query * residue) % slots - old) * tile
            ranges = PhysicalRanges(ranges.storage, ranges.spans, delta)
        groups.setdefault((word, orders), []).append(ranges)
    return tuple(
        FrontierGroup(word, tuple(ranges), orders) for (word, orders), ranges in groups.items()
    )


@bounded
def summarize_query_patterns(program, access) -> QueryPatternSummary:
    """Derive exact query hazards before expanding byte cells or obligations.

    Shared transfer words include every reader and definition binding. Physical
    ring rotations only translate range views and intersect their count guards;
    composition uses the same complete transfer algorithm as the full API.
    """
    mapping = access.query
    require(program.invocations and mapping is not None, "query summary needs actual query access")
    require(
        all(
            access.origins[name] is contract.actual_root.ref
            for name, contract in mapping.contracts.items()
        ),
        "query summary lost actual allocation identity",
    )
    roots = {
        name
        for name, contract in mapping.contracts.items()
        if contract.kind in {"reused", "epoch_ring"}
    }
    reordered = set((access.lifetime_report or {}).get("reordered_roots", ())) & roots
    boundaries = tuple(cell for cell in access.lifetime_boundaries if cell.storage in roots)
    require(
        {cell.storage for cell in boundaries} == reordered,
        "query summary needs each reordered root's actual occupancy boundaries",
    )
    statistics = {}
    frontiers = summarize_frontier_patterns(
        _records(program, access, roots, reordered), statistics=statistics
    )
    frontiers += group_frontiers(boundaries, order=statistics["cells"])
    for group in frontiers:
        for case in group.word:
            endpoints = (
                *case.reads_before_write,
                *(() if case.first_write is None else (case.first_write,)),
                *(() if case.last_write is None else (case.last_write,)),
                *case.pending_reads,
            )
            require(
                all(endpoint.task in access.endpoints for endpoint in endpoints),
                "query frontier has no actual parent call/native endpoint",
            )
    copies = max((mapping.contracts[root].slots for root in roots), default=1) + 1
    origins = tuple((root, access.origins[root]) for root in sorted(roots))
    bound_value = next(iter(program.parents.values())).limit
    views = _rotation_views(frontiers, mapping)
    parts = tuple(
        PatternPart(
            str(query),
            PatternRegion(
                tuple(sorted(roots)),
                _rotated_patterns(views, query),
                statistics["visits"],
                program.bound,
                program.count,
                origins,
                bound_value,
            ),
        )
        for query in range(copies)
    )
    sequence_statistics = {}
    sequence = summarize_sequence_patterns(
        parts, obligation_source_regions=frozenset({"0"}), statistics=sequence_statistics
    )
    statistics.update(
        **{f"sequence_{key}": value for key, value in sequence_statistics.items()},
        query_copies=copies,
        obligations=sum(pattern.physical_count for pattern in sequence.patterns),
        hazard_patterns=len(sequence.patterns),
        reordered_cells=len(boundaries),
    )
    return QueryPatternSummary(frontiers, parts, sequence.patterns, statistics)


@bounded
def summarize_queries(program, access) -> QuerySummary:
    """Derive query-zero's hazards against its next actual overlapping uses.

    Readonly and proved disjoint query frames have no cross-query byte hazards.
    Every other root remains the original allocation. Epoch rings rotate by
    q*N modulo their actual slot count; ordinary reused roots do not rotate.
    The maximum slot count plus one bounds the copies independently of either
    runtime length. This is a memory relation, never a completion certificate.
    """
    mapping = access.query
    require(program.invocations and mapping is not None, "query summary needs actual query access")
    require(
        all(
            access.origins[name] is contract.actual_root.ref
            for name, contract in mapping.contracts.items()
        ),
        "query summary lost actual allocation identity",
    )
    roots = {
        name
        for name, contract in mapping.contracts.items()
        if contract.kind in {"reused", "epoch_ring"}
    }
    reordered = set((access.lifetime_report or {}).get("reordered_roots", ())) & roots
    boundaries = tuple(cell for cell in access.lifetime_boundaries if cell.storage in roots)
    require(
        {cell.storage for cell in boundaries} == reordered,
        "query summary needs each reordered root's actual occupancy boundaries",
    )
    statistics = {}
    cells = summarize_frontiers(_records(program, access, roots, reordered), statistics=statistics)
    cells += boundaries
    for cell in cells:
        endpoints = (
            *cell.reads_before_write,
            *(() if cell.first_write is None else (cell.first_write,)),
            *(() if cell.last_write is None else (cell.last_write,)),
            *cell.pending_reads,
        )
        require(
            all(endpoint.task in access.endpoints for endpoint in endpoints),
            "query frontier has no actual parent call/native endpoint",
        )
    copies = max((mapping.contracts[root].slots for root in roots), default=1) + 1
    origins = tuple((root, access.origins[root]) for root in sorted(roots))
    bound_value = next(iter(program.parents.values())).limit
    parts = tuple(
        Part(
            str(query),
            RegionSummary(
                tuple(sorted(roots)),
                tuple(
                    rotated
                    for cell in cells
                    for rotated in _rotated(cell, mapping.contracts[cell.storage], query)
                ),
                statistics["visits"],
                program.bound,
                program.count,
                origins,
                bound_value,
            ),
        )
        for query in range(copies)
    )
    sequence_statistics = {}
    sequence = summarize_sequence(
        parts,
        materialize_cells=False,
        obligation_source_regions=frozenset({"0"}),
        statistics=sequence_statistics,
    )
    obligations = sequence.obligations
    statistics.update(
        **{f"sequence_{key}": value for key, value in sequence_statistics.items()},
        query_copies=copies,
        obligations=len(obligations),
        reordered_cells=len(boundaries),
    )
    return QuerySummary(cells, parts, obligations, statistics)


def query_native_sites(summary, access):
    """Demand actual boundary milestones before parent child projection.

    A source retirement prefix is only a candidate witness for a ReadSpan.
    The completion checker must prove that it includes every actual reader.
    This request itself neither orders nor retires any operation.
    """
    directions = {}
    hazards = summary.patterns if isinstance(summary, QueryPatternSummary) else summary.obligations
    for obligation in hazards:
        for endpoint, points in (
            (obligation.source, ("end", "prefix", "begin", "admit_issue", "after_queue")),
            (
                obligation.destination,
                ("begin", "issue", "queue", "retire", "admit_issue", "admit_queue"),
            ),
        ):
            directions.setdefault(endpoint.access.task, set()).update(points)
    return tuple(
        sorted(
            (
                (uid, replace(site, point=point))
                for task, points in directions.items()
                for uid, site in (access.endpoints[task],)
                for point in points
            ),
            key=repr,
        )
    )

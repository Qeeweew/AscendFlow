"""Actual byte frontiers across repeated serial queries, retaining ring phase.

One query's memory transfer comes from its physical native frontiers. A static
number of translated copies finds the first subsequent touch of every slot;
untouched bytes carry their old writer/readers through intermediate queries.
No completion edge or synchronization instruction is supplied by this module.
"""

from dataclasses import dataclass, replace

from .budget import bounded, poll
from .completion import CountDomain
from .loop_model import slope
from .model import require
from .region_sequence import Obligation, Part, summarize_sequence
from .region_summary import CellSummary, RegionSummary
from .scope_boundary_summary import summarize_frontiers
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
    sequence = summarize_sequence(parts)
    obligations = tuple(
        obligation for obligation in sequence.obligations if obligation.source.region == "0"
    )
    statistics.update(
        query_copies=copies,
        sequence_visits=sequence.visits,
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
    for obligation in summary.obligations:
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

"""Select missing serial boundary relations, then allocate their event resources.

Both same-call and carried transfers use this planner. The control adapter owns
ordering and native completion facts; this module never infers completion from
lexical order, hazard labels, or the existence of another candidate.
"""

from dataclasses import dataclass, replace

from .budget import poll
from .completion import CountDomain, Point, Requirement, coalesce
from .model import require
from .region_completion import _data, _task


@dataclass(frozen=True)
class Bridge:
    source: Point
    destination: Point
    domains: tuple[CountDomain, ...]
    event_id: int
    name: str = ""
    carried: bool = False


@dataclass(frozen=True)
class Candidate:
    requirement: Requirement
    source: Point
    destination: Point


def domain_row(d):
    if d.exact is not None:
        return (1, 0, d.exact, 0)
    row = (d.period, d.residue, -1, d.period if d.minimum is None else d.minimum)
    return row if d.maximum is None else (*row, d.maximum)


def select_bridges(sequence, proof, candidates, banks, *, carried=False):
    selected = []
    prefix = "call" if carried else "region"
    for candidate in candidates:
        poll("sequence uncovered relations", selected=len(selected), carried=carried)
        missing = proof.missing(candidate.requirement)
        if not missing:
            continue
        source, destination = candidate.source, candidate.destination
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        bridge = Bridge(source, destination, missing, -1 if sp == dp else 0, carried=carried)
        selected.append(bridge)
        proof.add_bridge(bridge)

    # Coalescing equivalent predicates does not add any new completion fact.
    grouped = {}
    for bridge in selected:
        grouped.setdefault((bridge.source, bridge.destination, bridge.event_id), set()).update(
            bridge.domains
        )
    selected = [
        Bridge(
            s,
            d,
            tuple(sorted(coalesce(domains), key=domain_row)),
            event_id,
            f"{prefix}_c{i}",
            carried,
        )
        for i, ((s, d, event_id), domains) in enumerate(grouped.items())
    ]
    actor = next(iter(next(iter(sequence.regions.values())).actors))
    assigned = []
    for bridge in selected:
        if bridge.event_id >= 0:
            sp, dp = _task(sequence, bridge.source).pipe, _task(sequence, bridge.destination).pipe
            key = actor, sp, dp
            reserved = int(
                (sp, dp) == ("S", "FIX") and _data(sequence, bridge.source.region).program.config
            )
            event_id = max(banks[key], reserved)
            require(event_id < 8, f"{key}: sequence event bank exhausted; no drain fallback")
            banks[key] = event_id + 1
            bridge = replace(bridge, event_id=event_id)
        assigned.append(bridge)
    return assigned

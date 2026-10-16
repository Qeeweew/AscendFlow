"""Select missing serial boundary relations, then allocate their event resources.

Both same-call and carried transfers use this planner. The control adapter owns
ordering and native completion facts; this module never infers completion from
lexical order, hazard labels, or the existence of another candidate.
"""

from dataclasses import dataclass, replace

from .budget import poll
from .completion import CountDomain, Point, Requirement, coalesce, covers, uncovered
from .model import require
from .region_completion import _data, _task, mark

_AUTHORITY = object()


def materialization_snapshot(module, ignored):
    """Linear correspondence for an owned insertion-only materializer."""
    return tuple(
        (
            op,
            op.parent,
            tuple(op.attributes.items()),
            tuple(op.properties.items()),
            tuple(op.operands),
            tuple(result.type for result in op.results),
            tuple(
                (
                    block,
                    tuple(argument.type for argument in block.args),
                    tuple(child for child in block.ops if not isinstance(child, ignored)),
                )
                for region in op.regions
                for block in region.blocks
            ),
        )
        for op in module.walk()
        if not isinstance(op, ignored)
    )


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


@dataclass(frozen=True)
class SelectedRelation:
    requirement: Requirement
    inherited: tuple[CountDomain, ...]
    inserted: tuple[CountDomain, ...]
    source: Point
    destination: Point


@dataclass(frozen=True)
class BridgeSelection:
    """Actual coverage plus the native-capture rule for each inserted domain."""

    proof: object
    bridges: tuple[Bridge, ...]
    relations: tuple[SelectedRelation, ...]
    _authority: object

    def check(self, sequence):
        require(self._authority is _AUTHORITY, "boundary selection needs its owned result")
        for relation in self.relations:
            request = relation.requirement
            require(
                request.source.point in {"end", "prefix"}
                and request.destination.point == "begin"
                and mark(
                    sequence.regions, relation.source, request.source.point, request.source.call
                )
                == request.source
                and mark(sequence.regions, relation.destination, "begin", request.destination.call)
                == request.destination
                and covers(request.domain, (*relation.inherited, *relation.inserted)),
                "boundary construction changed its actual memory endpoints or count domain",
            )
            for domain in relation.inserted:
                require(
                    covers(
                        domain,
                        (
                            active
                            for bridge in self.bridges
                            if bridge.source == relation.source
                            and bridge.destination == relation.destination
                            and request.destination.call - request.source.call
                            == int(bridge.carried)
                            for active in bridge.domains
                        ),
                    ),
                    "boundary selection lost an inserted native completion witness",
                )
        return self


def domain_row(d):
    if d.exact is not None:
        return (1, 0, d.exact, 0)
    row = (d.period, d.residue, -1, d.period if d.minimum is None else d.minimum)
    return row if d.maximum is None else (*row, d.maximum)


def select_bridges(sequence, proof, candidates, banks, *, carried=False):
    selected, relations = [], []
    prefix = "call" if carried else "region"
    for candidate in candidates:
        poll("sequence uncovered relations", selected=len(selected), carried=carried)
        coverage = tuple(proof.coverage(candidate.requirement))
        missing = tuple(uncovered(candidate.requirement.domain, coverage))
        relations.append(
            SelectedRelation(
                candidate.requirement, coverage, missing, candidate.source, candidate.destination
            )
        )
        if not missing:
            continue
        source, destination = candidate.source, candidate.destination
        sp, dp = _task(sequence.regions, source).pipe, _task(sequence.regions, destination).pipe
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
            sp, dp = (
                _task(sequence.regions, bridge.source).pipe,
                _task(sequence.regions, bridge.destination).pipe,
            )
            key = actor, sp, dp
            reserved = int(
                (sp, dp) == ("S", "FIX")
                and _data(sequence.regions, bridge.source.region).program.config
            )
            event_id = max(banks[key], reserved)
            require(event_id < 8, f"{key}: sequence event bank exhausted; no drain fallback")
            banks[key] = event_id + 1
            bridge = replace(bridge, event_id=event_id)
        assigned.append(bridge)
    return BridgeSelection(proof, tuple(assigned), tuple(relations), _AUTHORITY).check(sequence)

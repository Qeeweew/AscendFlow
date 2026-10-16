"""Finite boundary transfers for sequential regions, without internal graphs.

An output is the union of explicitly wired input frontiers and locally
established guarantees. Requirements refer to actual consumers; they are
preconditions, never completion facts. Empty regions have identity wiring.

For P declared ports and K prefix domains, outputs contain at most P*P wires
and P*K prefix ranks. Composition substitutes only those wires. Each incoming
requirement creates at most one requirement (a set of alternative input
witnesses); it never distributes into a product of paths. A sequence with H
original requirements retains at most H clauses. compose_sequence accumulates
requirements once, avoiding repeated copying of the growing prefix ledger.

This module handles one established clock/guard partition. Binding nested
clock coordinates and establishing native capture guarantees are the owner's
responsibility; no name, shape, cache, or graph fallback supplies those facts.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.ir import Operation

from ..diagnostics import require
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine
from .frontier import Frontier, FrontierLayout, FrontierWork, KnownPrefix


@dataclass(frozen=True, slots=True)
class Port:
    actor: Actor
    engine: Engine | None
    milestone: Milestone

    def __post_init__(self):
        require(isinstance(self.actor, Actor), "region port requires a typed actor")
        require(
            isinstance(self.milestone, Milestone) and self.milestone.cumulative,
            "region port must carry a cumulative state",
        )
        require(
            self.engine is None
            if self.milestone is Milestone.ISSUE
            else supports_engine(self.actor, self.engine),
            "region port has an invalid engine",
        )


@dataclass(frozen=True, slots=True)
class BoundaryLayout:
    ports: frozenset[Port]
    frontier: FrontierLayout

    def __post_init__(self):
        ports = frozenset(self.ports)
        require(all(isinstance(port, Port) for port in ports), "invalid region boundary port")
        require(isinstance(self.frontier, FrontierLayout), "boundary needs a frontier layout")
        object.__setattr__(self, "ports", ports)


@dataclass(frozen=True, slots=True)
class Output:
    sources: frozenset[Port]
    guarantees: Frontier

    def __post_init__(self):
        object.__setattr__(self, "sources", frozenset(self.sources))
        require(
            isinstance(self.guarantees, Frontier), "region guarantee needs established prefixes"
        )


@dataclass(frozen=True, slots=True)
class Requirement:
    """At least one wired input must supply a prefix for this actual consumer.

    Alternative wires are OR; separate requirements are AND. A CV quorum is
    checked before its acquired frontier enters a transfer, not by this OR.
    """

    sources: frozenset[Port]
    prefix: KnownPrefix
    consumer: Operation

    def __post_init__(self):
        object.__setattr__(self, "sources", frozenset(self.sources))
        require(isinstance(self.prefix, KnownPrefix), "region requirement needs a typed prefix")
        require(isinstance(self.consumer, Operation), "requirement must retain its actual consumer")


@dataclass(slots=True)
class RegionWork:
    regions: int = 0
    output_rows: int = 0
    wire_visits: int = 0
    requirement_visits: int = 0
    requirement_comparisons: int = 0


@dataclass(frozen=True, init=False)
class BoundaryTransfer:
    layout: BoundaryLayout
    outputs: Mapping[Port, Output]
    requirements: tuple[Requirement, ...]

    def __init__(self, layout, outputs, requirements=()):
        require(isinstance(layout, BoundaryLayout), "region transfer needs a declared boundary")
        require(
            set(outputs) == layout.ports, "region outputs must cover exactly the boundary ports"
        )
        requirements = tuple(requirements)
        for output in outputs.values():
            require(isinstance(output, Output), "region transfer needs typed output expressions")
            require(output.sources <= layout.ports, "region wire names an undeclared input")
            require(output.guarantees.layout == layout.frontier, "region guarantee layout mismatch")
        for requirement in requirements:
            require(isinstance(requirement, Requirement), "region needs typed entry requirements")
            require(requirement.sources <= layout.ports, "requirement names an undeclared input")
            require(
                requirement.prefix.domain in layout.frontier._keys,
                "requirement names an undeclared prefix domain",
            )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "outputs", MappingProxyType(dict(outputs)))
        object.__setattr__(self, "requirements", requirements)

    @classmethod
    def identity(cls, layout):
        empty = layout.frontier.empty()
        return cls(layout, {port: Output(frozenset({port}), empty) for port in layout.ports})

    def _inputs(self, inputs):
        require(set(inputs) == self.layout.ports, "region input bindings do not match its ports")
        require(
            all(
                isinstance(value, Frontier) and value.layout == self.layout.frontier
                for value in inputs.values()
            ),
            "region input frontier layout mismatch",
        )

    def missing(self, inputs) -> tuple[Requirement, ...]:
        self._inputs(inputs)
        return tuple(
            requirement
            for requirement in self.requirements
            if not any(inputs[port].covers(requirement.prefix) for port in requirement.sources)
        )

    def apply(self, inputs, *, work: FrontierWork | None = None):
        require(not self.missing(inputs), "region entry requirements have not been established")
        return self._apply_established(inputs, work=work)

    def _apply_established(self, inputs, *, work: FrontierWork | None = None):
        """Owned callers already established this transfer's entry contract."""
        return MappingProxyType(
            {
                port: output.guarantees.acquire(
                    *(inputs[source] for source in output.sources), work=work
                )
                for port, output in self.outputs.items()
            }
        )

    @property
    def statistics(self):
        return {
            "ports": len(self.layout.ports),
            "output_wires": sum(len(output.sources) for output in self.outputs.values()),
            "output_prefixes": sum(len(output.guarantees) for output in self.outputs.values()),
            "entry_requirements": len(self.requirements),
        }


def compose_sequence(
    layout: BoundaryLayout,
    transfers: Iterable[BoundaryTransfer],
    *,
    work: RegionWork | None = None,
    frontier_work: FrontierWork | None = None,
) -> BoundaryTransfer:
    """Compose a lexical sequence once; no internal completion graph is read.

    Substitution of one output row reads at most P rows, each with P wires and
    K prefixes. Hence output work is O(L*P^2*(P+K)) for L child contracts;
    entry substitution is O(H*P*(P+1)). P/K are explicit boundary dimensions,
    not graph vertices or historical query origins. No factor contains N.
    """
    current = dict(BoundaryTransfer.identity(layout).outputs)
    clauses = {}
    for transfer in transfers:
        require(
            isinstance(transfer, BoundaryTransfer) and transfer.layout == layout,
            "sequence composition needs the same bound port and clock layout",
        )
        if work is not None:
            work.regions += 1
        for requirement in transfer.requirements:
            if work is not None:
                work.requirement_visits += 1
            sources, satisfied = set(), False
            for source in requirement.sources:
                if work is not None:
                    work.wire_visits += 1
                row = current[source]
                if row.guarantees.covers(requirement.prefix):
                    satisfied = True
                    break
                sources.update(row.sources)
            if not satisfied:
                requirement = Requirement(
                    frozenset(sources), requirement.prefix, requirement.consumer
                )
                # Equal logical needs at different instructions are distinct
                # insertion obligations. Keeping only a later consumer could
                # move its Wait past an earlier read. The boundary owner may
                # coalesce them only with a separate placement witness.
                key = requirement.sources, requirement.prefix.domain, requirement.consumer
                previous = clauses.get(key)
                if previous is not None and work is not None:
                    work.requirement_comparisons += 1
                if previous is None or requirement.prefix.rank.compare(previous.prefix.rank) > 0:
                    clauses[key] = requirement
        following = {}
        for port, output in transfer.outputs.items():
            if work is not None:
                work.output_rows += 1
            sources, guarantees = set(), [output.guarantees]
            for source in output.sources:
                if work is not None:
                    work.wire_visits += 1
                row = current[source]
                sources.update(row.sources)
                guarantees.append(row.guarantees)
            following[port] = Output(
                frozenset(sources),
                layout.frontier.empty().acquire(*guarantees, work=frontier_work),
            )
        current = following
    return BoundaryTransfer(layout, current, clauses.values())

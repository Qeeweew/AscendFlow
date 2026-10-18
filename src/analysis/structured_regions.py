"""Shared occurrence, existence-domain and region-boundary contracts.

These are memory dependency facts, not notifications or completed work.
FIRST/LAST and read-family identity retain their actual loop and participant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.dialects import scf
from xdsl.ir import Operation, SSAValue

from ..scalar import Expr
from ..target.ascend910b import Actor
from .access_frontier import AccessCut, AccessEndpoint, LoopResidueSelection
from .hazards import HazardKind
from .structured_schedule import Comparison

if TYPE_CHECKING:
    from .structured_memory import MemoryAccess


@dataclass(frozen=True, slots=True)
class LoopBinding:
    loop: Operation
    ordinal: Expr | LoopResidueSelection


@dataclass(frozen=True, slots=True)
class LoopDomain:
    """Consumer j >= max(first, distance), j % period == residue, j < count.

    A distance-d need uses producer j-d. Residues always refer to consumers.
    """

    loop: Operation
    count: Expr
    period: int = 1
    residue: int | Expr = 0
    distance: int = 0
    first: int = 0


class CountRelation(str, Enum):
    MORE = ">"
    AT_MOST = "<="


@dataclass(frozen=True, slots=True)
class LoopCountGuard:
    loop: Operation
    count: Expr
    relation: CountRelation
    threshold: int | Expr

    def evaluate(self, values: Mapping[str, int]):
        count = self.count.evaluate(values)
        threshold = (
            self.threshold if type(self.threshold) is int else self.threshold.evaluate(values)
        )
        return count > threshold if self.relation is CountRelation.MORE else count <= threshold


@dataclass(frozen=True, slots=True)
class BranchGuard:
    operation: scf.IfOp
    branch: int


@dataclass(frozen=True, slots=True)
class LoopOrdinalGuard:
    """An effect predicate in one actual loop's canonical coordinates."""

    loop: Operation
    comparison: Comparison

    def evaluate(self, values: Mapping[str, int]):
        return self.comparison.evaluate(values)


Guard = LoopCountGuard | BranchGuard | LoopOrdinalGuard


@dataclass(frozen=True, slots=True)
class AccessOccurrence:
    endpoint: AccessEndpoint
    bindings: tuple[LoopBinding, ...] = ()
    families: tuple[LoopDomain, ...] = ()

    @property
    def operation(self):
        return self.endpoint.operation

    @property
    def participant(self):
        return self.endpoint.participant

    @property
    def frame(self):
        return self.endpoint.frame


@dataclass(frozen=True, slots=True)
class StorageCell:
    """Exact byte interval relative to a retained symbolic coordinate map.

    ``origin`` retains all streaming/uniform coordinates. Distinct origins are
    never silently assumed disjoint. ``private`` is absent for shared GM.
    ``start:stop`` is the first row; rows/row_stride preserve a strided box.
    """

    root: SSAValue
    private: Actor | None
    origin: Expr
    start: int
    stop: int
    streaming: tuple[Operation, ...] = ()
    rows: int = 1
    row_stride: int = 0
    conditions: tuple[Guard, ...] = ()

    @property
    def participant(self):
        return self.private


@dataclass(frozen=True, slots=True)
class AliasRegion:
    """Possibly overlapping accesses to one actual storage instance.

    This is an ordering domain, not a byte footprint or a complete definition.
    Actual footprints stay on MemoryAccess. A write cannot KILL this region,
    and it cannot authorize physical-cell credit or private packet reordering.
    """

    root: SSAValue
    private: Actor | None
    # Shared GM is one allocation. Independent Vector peers have distinct
    # source access domains; an AIC access participates in every such domain.
    # This preserves both AIV contributions without inventing peer ordering.
    peer: Actor | None = None


@dataclass(frozen=True, slots=True)
class BoundaryPort:
    cell: StorageCell | AliasRegion


@dataclass(frozen=True, slots=True)
class StructuredNeed:
    cell: StorageCell | AliasRegion
    kind: HazardKind
    source: AccessOccurrence | BoundaryPort
    consumer: AccessOccurrence
    scope: Operation
    domains: tuple[LoopDomain, ...] = ()
    guards: tuple[Guard, ...] = ()
    loop: Operation | None = None
    distance: int | None = None
    period: int = 1
    residue: int | Expr = 0
    instance_of: StructuredNeed | None = field(default=None, repr=False, compare=False)
    _hash: int | None = field(default=None, init=False, compare=False, repr=False)

    def __hash__(self):
        value = self._hash
        if value is None:
            value = hash(
                (
                    self.cell,
                    self.kind,
                    self.source,
                    self.consumer,
                    self.scope,
                    self.domains,
                    self.guards,
                    self.loop,
                    self.distance,
                    self.period,
                    self.residue,
                )
            )
            object.__setattr__(self, "_hash", value)
        return value

    @property
    def source_cut(self):
        from ..target.ascend910b import Milestone

        return (
            AccessCut(self.source.endpoint, Milestone.NATIVE_END)
            if isinstance(self.source, AccessOccurrence)
            else None
        )

    @property
    def consumer_cut(self):
        from ..target.ascend910b import Milestone

        return AccessCut(self.consumer.endpoint, Milestone.ADMISSION)


class LogicalWriteKind(str, Enum):
    KILL = "kill"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class LogicalWrite:
    """One definition on retained native write footprints and activation.

    A kill does not read the overwritten fragment's old value. A contracted
    cell can denote a family image; ``accesses`` retains each actual operand
    footprint needed to prove per-occurrence coverage. An update retains its
    separately bound RAW input. Neither kind proves independent ownership.
    """

    cell: StorageCell
    occurrence: AccessOccurrence
    kind: LogicalWriteKind
    guards: tuple[Guard, ...]
    domains: tuple[LoopDomain, ...] = ()
    accesses: tuple[MemoryAccess, ...] = ()


@dataclass(frozen=True, slots=True)
class CellSummary:
    """Exact value frontier or source-order frontier, according to region type.

    For AliasRegion, writer fields are ordering predecessors only. They do
    not claim complete coverage or select a read's numerical definition.
    """

    cell: StorageCell | AliasRegion
    head_reads: tuple[StructuredNeed, ...] = ()
    first_write: AccessOccurrence | None = None
    last_write: AccessOccurrence | None = None
    tail_reads: tuple[StructuredNeed, ...] = ()

    @property
    def writer(self):
        return self.last_write or BoundaryPort(self.cell)

    @property
    def identity(self):
        return self.first_write is None and not self.head_reads and not self.tail_reads


@dataclass(frozen=True, slots=True)
class RegionConnection:
    """One exact source frontier splice, before choosing storage lifetimes."""

    scope: Operation
    left: CellSummary
    right: CellSummary
    guards: tuple[Guard, ...]
    loop: Operation | None = None
    distance: int | None = None
    period: int | None = None
    residue: int | Expr | None = None
    first: int = 0
    limit: Expr | None = None


@dataclass(frozen=True, slots=True)
class GuardedCellSummary:
    guards: tuple[Guard, ...]
    transfer: CellSummary


@dataclass(frozen=True, slots=True)
class ScopeSummary:
    operation: Operation
    cells: Mapping[StorageCell | AliasRegion, tuple[GuardedCellSummary, ...]]


@dataclass(frozen=True, slots=True)
class ExitObligation:
    cell: StorageCell | AliasRegion
    access: AccessOccurrence
    writer: bool
    guards: tuple[Guard, ...]

    @property
    def all_members(self):
        """A family requires every member retired; last native end is insufficient."""
        return bool(self.access.families)


class DependencyIssueKind(str, Enum):
    MULTI_RESIDUE = "multi-residue-frontier"
    PARENT_ACTIVATION = "parent-dependent-activation"
    CONTROL = "unsupported-control"
    BUDGET = "storage-proof-budget"


@dataclass(frozen=True, slots=True)
class DependencyIssue:
    kind: DependencyIssueKind
    operation: Operation
    message: str
    cell: StorageCell | AliasRegion | None = None


@dataclass(slots=True)
class DependencyWork:
    native_templates: int = 0
    projected_accesses: int = 0
    cells: int = 0
    scope_visits: int = 0
    boundary_visits: int = 0
    needs: int = 0
    loop_carries: int = 0
    alias_regions: int = 0
    disjoint_writes: int = 0

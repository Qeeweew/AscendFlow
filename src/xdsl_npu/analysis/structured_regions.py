"""Shared occurrence, existence-domain and region-boundary contracts.

These are memory dependency facts, not notifications or completed work.
FIRST/LAST and read-family identity retain their actual loop and participant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from xdsl.dialects import scf
from xdsl.ir import Operation, SSAValue

from ..scalar import Expr
from ..target.ascend910b import Actor
from .access_frontier import AccessCut, AccessEndpoint, LoopResidueSelection
from .hazards import HazardKind


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


Guard = LoopCountGuard | BranchGuard


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


class BoundaryKind(str, Enum):
    WRITER = "writer"
    READERS = "readers"


@dataclass(frozen=True, slots=True)
class BoundaryPort:
    cell: StorageCell
    kind: BoundaryKind


@dataclass(frozen=True, slots=True)
class StructuredNeed:
    cell: StorageCell
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


@dataclass(frozen=True, slots=True)
class StructuredReadCarry:
    waw: StructuredNeed
    raw: StructuredNeed
    war: StructuredNeed | None


@dataclass(frozen=True, slots=True)
class LoopCarry:
    loop: Operation
    count: Expr
    period: int
    residue: int | Expr
    need: StructuredNeed


@dataclass(frozen=True, slots=True)
class PendingReader:
    reader: AccessOccurrence
    writer: AccessOccurrence | BoundaryPort
    raw: StructuredNeed


@dataclass(frozen=True, slots=True)
class CellSummary:
    cell: StorageCell
    head_reads: tuple[PendingReader, ...] = ()
    first_write: AccessOccurrence | None = None
    last_write: AccessOccurrence | None = None
    tail_reads: tuple[PendingReader, ...] = ()
    entry_carries: tuple[StructuredReadCarry, ...] = ()

    @property
    def writer(self):
        return self.last_write or BoundaryPort(self.cell, BoundaryKind.WRITER)

    @property
    def identity(self):
        return self.first_write is None and not self.head_reads and not self.tail_reads


@dataclass(frozen=True, slots=True)
class GuardedCellSummary:
    guards: tuple[Guard, ...]
    transfer: CellSummary


@dataclass(frozen=True, slots=True)
class ScopeSummary:
    operation: Operation
    cells: Mapping[StorageCell, tuple[GuardedCellSummary, ...]]


@dataclass(frozen=True, slots=True)
class ExitObligation:
    cell: StorageCell
    access: AccessOccurrence
    writer: bool
    guards: tuple[Guard, ...]

    @property
    def all_members(self):
        """A family requires every member retired; last native end is insufficient."""
        return bool(self.access.families)


class DependencyIssueKind(str, Enum):
    SYMBOLIC_ALIAS = "symbolic-alias"
    NONPERIODIC_REUSE = "nonperiodic-reuse"
    MULTI_RESIDUE = "multi-residue-frontier"
    PARENT_ACTIVATION = "parent-dependent-activation"
    CONTROL = "unsupported-control"
    BUDGET = "storage-proof-budget"
    FRAGMENT_MAPPING = "fragment-coordinate-map"


@dataclass(frozen=True, slots=True)
class DependencyIssue:
    kind: DependencyIssueKind
    operation: Operation
    message: str
    cell: StorageCell | None = None


@dataclass(slots=True)
class DependencyWork:
    native_templates: int = 0
    projected_accesses: int = 0
    cells: int = 0
    scope_visits: int = 0
    boundary_visits: int = 0
    needs: int = 0
    loop_carries: int = 0

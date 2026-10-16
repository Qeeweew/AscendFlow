"""Typed contracts shared by event admission, lifetimes, progress and allocation.

These records describe actual endpoints and witnesses. They do not authorize
external IR or infer completion from a successful structural verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .structured_protocols import BoundPlacement, LogicalNotification, SynchronizationSelection
from .structured_schedule import ActionGuard, SyncAction

if TYPE_CHECKING:
    from .event_domains import EventCase
    from .event_paths import EventPathCover
    from .event_windows import EventWindow
    from .progress_regions import GuardedTransfer, RegionProgress
    from .structured_event_sharing import EventSharingWitness

LocalBank = tuple[Actor, Engine, Engine]
LOCAL_CAPACITY = 8
# Conservative allocator policy, not a hardware namespace capacity claim.
SUPPORTED_CROSS_IDS = tuple(range(8))
CROSS_BACKLOG = 15


class LeaseKind(str, Enum):
    FINITE = "finite-single-use-per-id"
    QUEUE_ROUNDTRIP = "queue-roundtrip"
    CONTROLLER = "controller-consumption"
    INVOCATION = "invocation-reentry"
    COUNTER = "mode2-backlog"


class _Order(str, Enum):
    PROVED = "proved"
    IMPOSSIBLE = "impossible"
    POSSIBLE = "possible"


@dataclass(frozen=True, slots=True)
class LeaseStep:
    """A real queue/controller gate followed by a selected notification."""

    consume: BoundPlacement
    source: BoundPlacement
    notification: LogicalNotification | None
    controller: bool


@dataclass(frozen=True, slots=True)
class LeaseWitness:
    kind: LeaseKind
    loop: Operation | None
    delta: int
    consume: BoundPlacement
    next_set: BoundPlacement
    path_cover: EventPathCover | None = None
    upper_uses: int | None = None
    condition: ActionGuard = field(kw_only=True)


@dataclass(frozen=True, slots=True)
class CaseLease:
    """One case's proved ring, with its own radix and reuse premises."""

    case: EventCase
    width: int
    ordinal: Expr
    rings: tuple[tuple[EventWindow, int], ...]
    inside: tuple[LeaseWitness, ...]
    reentries: tuple[LeaseWitness, ...]
    backlog: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredEventLease:
    notification: LogicalNotification
    width: int
    cases: tuple[CaseLease, ...]


@dataclass(frozen=True, slots=True)
class StructuredEventAssignment:
    notification: LogicalNotification
    ids: tuple[int, ...]
    banks: tuple[LocalBank, ...]
    cases: tuple[CaseLease, ...]


@dataclass(frozen=True, slots=True)
class EventReservation:
    bank: LocalBank
    event: int
    reason: str
    anchor: Operation


@dataclass(frozen=True, slots=True)
class ProgressPoint:
    """A proof node: a real notification action or an explicit queue cut."""

    operation: Operation
    actor: Actor
    phase: str
    notification: LogicalNotification | None = None
    component: int = 0
    projection: str = ""


@dataclass(frozen=True, slots=True)
class StructuredProgress:
    """Region composition, with local stream ranks and guarded queue interfaces.

    Finite cyclic interfaces are certified by their region exclusions. They
    do not claim one unconditional numeric rank across incompatible guards.
    Queue cuts certify blocking only, with no completion or lease semantics.
    """

    points: tuple[ProgressPoint, ...]
    arcs: tuple[GuardedTransfer, ...]
    clocks: tuple[Operation, ...]
    ranks: tuple[tuple[int, ...], ...]
    strides: tuple[int, ...] = ()
    region: RegionProgress | None = None


@dataclass(slots=True)
class ResourceWork:
    notifications: int = 0
    parameter_cases: int = 0
    parameter_splits: int = 0
    window_constructions: int = 0
    lifetime_interfaces: int = 0
    lease_searches: int = 0
    witness_steps: int = 0
    path_states: int = 0
    path_edges: int = 0
    path_edge_queries: int = 0
    path_guard_states: int = 0
    path_covers: int = 0
    cover_nodes: int = 0
    cover_queries: int = 0
    assigned_local_ids: int = 0
    assigned_cross_ids: int = 0
    progress_points: int = 0
    progress_arcs: int = 0
    progress_path_combinations: int = 0
    progress_path_alternatives: int = 0
    progress_case_branches: int = 0
    progress_condition_sets: int = 0
    progress_condition_queries: int = 0
    progress_condition_hits: int = 0


@dataclass(frozen=True)
class StructuredEventPlan:
    selection: SynchronizationSelection
    actions: tuple[SyncAction, ...]
    assignments: tuple[StructuredEventAssignment, ...]
    leases: tuple[StructuredEventLease, ...]
    reservations: tuple[EventReservation, ...]
    progress: StructuredProgress
    work: ResourceWork
    sharing: tuple[EventSharingWitness, ...] = ()

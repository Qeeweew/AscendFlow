"""Typed contracts shared by event admission, lifetimes, progress and allocation.

These records describe actual endpoints and witnesses. They do not authorize
external IR or infer completion from a successful structural verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .structured_protocols import BoundPlacement, LogicalNotification, SynchronizationSelection
from .structured_schedule import SyncAction

if TYPE_CHECKING:
    from .event_progress import GuardedTransfer, RegionProgress
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
    steps: tuple[LeaseStep, ...] = ()
    upper_uses: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredEventLease:
    notification: LogicalNotification
    width: int
    inside: tuple[LeaseWitness, ...]
    reentries: tuple[LeaseWitness, ...]
    backlog: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredEventAssignment:
    notification: LogicalNotification
    ids: tuple[int, ...]
    banks: tuple[LocalBank, ...]
    ordinal: Expr


@dataclass(frozen=True, slots=True)
class EventReservation:
    bank: LocalBank
    event: int
    reason: str
    anchor: Operation


@dataclass(frozen=True, slots=True)
class ProgressArc:
    source: int
    destination: int
    steps: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ProgressPoint:
    operation: Operation
    actor: Actor
    phase: str
    notification: LogicalNotification | None = None
    component: int = 0
    projection: str = ""


@dataclass(frozen=True, slots=True)
class StructuredProgress:
    """Region composition, with local stream ranks and guarded interfaces.

    Finite cyclic interfaces are certified by their region exclusions. They
    do not claim one unconditional numeric rank across incompatible guards.
    """

    points: tuple[ProgressPoint, ...]
    arcs: tuple[ProgressArc | GuardedTransfer, ...]
    clocks: tuple[Operation, ...]
    ranks: tuple[tuple[int, ...], ...]
    strides: tuple[int, ...] = ()
    region: RegionProgress | None = None


@dataclass(slots=True)
class ResourceWork:
    notifications: int = 0
    lifetime_interfaces: int = 0
    lease_searches: int = 0
    witness_steps: int = 0
    assigned_local_ids: int = 0
    assigned_cross_ids: int = 0
    progress_points: int = 0
    progress_arcs: int = 0


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

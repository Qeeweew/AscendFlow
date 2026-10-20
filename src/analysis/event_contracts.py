"""Generated event lifetimes and their finite hardware allocation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .structured_protocols import LogicalNotification, SynchronizationSelection
from .structured_schedule import SyncAction

if TYPE_CHECKING:
    from .cell_channels import CellChannel
    from .cell_ownership import PersistentCellCredit
    from .event_domains import EventCase
    from .event_sharing import CounterSharing, EventSharing
    from .event_windows import EventWindow
    from .notification_lifetimes import Reuse
    from .queue_order import QueueOrder

LocalBank = tuple[Actor, Engine, Engine]
LOCAL_CAPACITY = 8
# Installed 2201 ASC GetfftsConfig encodes flag_id & 0xf; all modes share
# sixteen base slots. Higher IDs alias and are never overflow capacity.
SUPPORTED_CROSS_IDS = tuple(range(16))
CROSS_BACKLOG = 15


@dataclass(frozen=True, slots=True)
class CaseLease:
    case: EventCase
    width: int
    ordinal: Expr
    rings: tuple[tuple[EventWindow, int], ...]
    inside: tuple[Reuse, ...]
    reentries: tuple[Reuse, ...]
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
class EventBankUsage:
    bank: LocalBank | None
    ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PersistentEventAssignment:
    credit: PersistentCellCredit
    bank: LocalBank
    event: int


@dataclass(slots=True)
class ResourceWork:
    notifications: int = 0
    optional_relay_declines: int = 0
    endpoint_guard_sets: int = 0
    binding_roundtrips: int = 0
    pairing_implications: int = 0
    pairing_refinements: int = 0
    parameter_cases: int = 0
    parameter_splits: int = 0
    window_constructions: int = 0
    lifetime_interfaces: int = 0
    handback_candidates: int = 0
    handback_compositions: int = 0
    ring_cases: int = 0
    cell_channels: int = 0
    cell_members: int = 0
    sharing_pairs: int = 0
    sharing_reuses: int = 0
    allocated_local_ids: int = 0
    allocated_cross_ids: int = 0
    maximum_counter_backlog: int = 0
    persistent_cells: int = 0


@dataclass(frozen=True)
class StructuredEventPlan:
    selection: SynchronizationSelection
    actions: tuple[SyncAction, ...]
    assignments: tuple[StructuredEventAssignment, ...]
    leases: tuple[StructuredEventLease, ...]
    reservations: tuple[EventReservation, ...]
    order: QueueOrder
    work: ResourceWork
    channels: tuple[CellChannel, ...]
    banks: tuple[EventBankUsage, ...]
    sharing: tuple[EventSharing | CounterSharing, ...] = ()
    persistent: tuple[PersistentEventAssignment, ...] = ()

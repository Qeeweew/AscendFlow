"""Generated event lifetimes and their finite hardware allocation outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine, SyncKind
from .structured_protocols import LogicalNotification, SynchronizationSelection
from .structured_schedule import ActionGuard, Placement, SyncAction

if TYPE_CHECKING:
    from .cell_channels import CellChannel
    from .cell_ownership import PersistentCellCredit
    from .event_counter_groups import CounterGroup, CounterTimeGroups
    from .event_domains import EventCase
    from .event_frontier_contracts import FrozenEventRelations, LocalEventReuse
    from .event_sharing import CounterReuse, CounterSharing, EventSharing
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
    policy: str = "cell-ownership"


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


@dataclass(frozen=True, slots=True)
class EventDemand:
    """One complete credit's virtual lanes, independent of hardware numbers."""

    members: tuple[LogicalNotification, ...]
    width: int
    banks: tuple[LocalBank, ...]
    backlog: int | None = None
    persistent: PersistentCellCredit | None = None


@dataclass(frozen=True, slots=True)
class VirtualEventAction:
    """A fixed real instruction site with a virtual lane or target-fixed ID."""

    anchor: Operation
    placement: Placement
    participants: tuple[Actor, ...]
    kind: SyncKind
    source: Engine
    destination: Engine
    guard: ActionGuard
    protocols: tuple[str, ...]
    resource: int | None = None
    phase: Expr | None = None
    fixed_id: int | None = None


@dataclass(frozen=True, init=False)
class EventResourceProblem:
    """Owned, frozen resource premises; no domain or solver service is exposed.

    Construct with plan_event_resources. Relations cover all members of each
    complete credit. A missing/None relation supplies no reuse permission.
    CounterSharing pairs still require a whole CounterGroup for FIFO packing.
    Work is diagnostic accounting only and never an allocation premise.
    """

    selection: SynchronizationSelection
    order: QueueOrder
    leases: tuple[StructuredEventLease, ...]
    channels: tuple[CellChannel, ...]
    demands: tuple[EventDemand, ...]
    uses: tuple[tuple[LogicalNotification, int | None], ...]
    actions: tuple[VirtualEventAction, ...]
    peers: tuple[tuple[int, int], ...]
    counter_groups: tuple[CounterGroup, ...]
    counter_compositions: tuple[CounterTimeGroups, ...]
    counter_reuses: Mapping[tuple[int, int], tuple[CounterReuse, ...]]
    reservations: tuple[EventReservation, ...]
    relations: FrozenEventRelations
    work: ResourceWork
    # Frozen smaller protocols, in cumulative structural order. They carry
    # their own phases and constraints; allocation never asks for more facts.
    alternatives: tuple[EventResourceProblem, ...]

    def __init__(self, *args, **kwargs):
        raise TypeError("construct event resources with plan_event_resources")


@dataclass(frozen=True, slots=True)
class EventAllocation:
    """A checked finite mapping of every demand's virtual lanes to real IDs."""

    ids: tuple[tuple[int, ...], ...]
    sharing: tuple[EventSharing | LocalEventReuse | CounterSharing | CounterReuse, ...]
    counter_groups: tuple = ()
    option: int = 0
    counter_reuse_groups: tuple[CounterReuseGroup, ...] = ()
    counter_compositions: tuple = ()


@dataclass(frozen=True, slots=True)
class CounterReuseGroup:
    event: int
    resources: tuple[int, ...]
    messages: tuple[tuple[int, LogicalNotification], ...]
    backlog: int
    witnesses: tuple[CounterReuse, ...]


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
    issue_order_words: int = 0
    handback_candidates: int = 0
    handback_port_candidates: int = 0
    handback_ports: int = 0
    handback_return_candidates: int = 0
    dominated_handback_returns: int = 0
    handback_cover_queries: int = 0
    handback_compositions: int = 0
    ring_cases: int = 0
    cell_channels: int = 0
    cell_members: int = 0
    sharing_pairs: int = 0
    sharing_reuses: int = 0
    frontier_enclosures: int = 0
    frontier_queries: int = 0
    frontier_blocks: int = 0
    frontier_rectangles: int = 0
    local_reuse_groups: int = 0
    local_reuse_rectangles: int = 0
    local_reuse_row_checks: int = 0
    allocated_local_ids: int = 0
    allocated_cross_ids: int = 0
    maximum_counter_backlog: int = 0
    persistent_cells: int = 0
    coloring_nodes: int = 0
    peer_equalities: int = 0
    counter_groups: int = 0
    counter_compositions: int = 0
    protocol_candidates: int = 0
    selected_protocol_option: int = 0
    counter_time_pairs: int = 0
    counter_time_cache_hits: int = 0
    boundary_instances: int = 0
    boundary_instance_reuses: int = 0
    transfer_words: int = 0
    transfer_nodes: int = 0
    transfer_edges: int = 0
    transfer_order_bands: int = 0
    transfer_range_vertices: int = 0
    completion_queries: int = 0
    completion_nodes: int = 0
    completion_cycles: int = 0
    completion_eliminated: int = 0
    transfer_components: int = 0
    transfer_may_points: int = 0
    completion_pruned: int = 0


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
    sharing: tuple[EventSharing | LocalEventReuse | CounterSharing | CounterReuse, ...] = ()
    persistent: tuple[PersistentEventAssignment, ...] = ()
    counter_groups: tuple = ()
    preferred_leases: tuple[StructuredEventLease, ...] = ()
    counter_reuse_groups: tuple[CounterReuseGroup, ...] = ()
    counter_compositions: tuple = ()

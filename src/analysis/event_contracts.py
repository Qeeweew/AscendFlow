"""Generated event lifetimes and their finite hardware allocation outputs."""

from __future__ import annotations

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
    from .event_frontier_contracts import FrozenEventRelations, LocalEventReuse, SelectedSeparation
    from .event_sharing import EventSharing
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
class EventIdPools:
    """Assignable target numbers, snapshotted before resource planning.

    A restricted pool is a subset of the real hardware namespace. Fixed
    reservations remain occupied even when outside that assignable subset.
    Private bank overrides do not merge AIV storage or numbering obligations.
    """

    local: tuple[int, ...] = tuple(range(8))
    cross: tuple[int, ...] = tuple(range(16))
    banks: tuple[tuple[LocalBank, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        def snapshot(values, limit):
            values = tuple(values)
            if any(type(n) is not int or not 0 <= n < limit for n in values):
                raise ValueError("event pool must contain actual target integer IDs")
            if len(set(values)) != len(values):
                raise ValueError("event pool repeats a target ID")
            return tuple(sorted(values))

        object.__setattr__(self, "local", snapshot(self.local, 8))
        object.__setattr__(self, "cross", snapshot(self.cross, 16))
        banks = tuple((tuple(bank), snapshot(values, 8)) for bank, values in self.banks)
        if len({bank for bank, _ in banks}) != len(banks):
            raise ValueError("event pool repeats a private bank")
        if any(
            len(bank) != 3
            or not isinstance(bank[0], Actor)
            or any(not isinstance(pipe, Engine) for pipe in bank[1:])
            for bank, _ in banks
        ):
            raise ValueError("event pool requires an actual actor/pipe bank")
        object.__setattr__(self, "banks", banks)

    def ids(self, bank):
        if bank is None:
            return self.cross
        return next((values for own, values in self.banks if own == bank), self.local)


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
    """One occupation's virtual lanes, retaining its logical credit identity.

    A whole credit is the preferred occupation. A phase may instead select
    actual notification members; their own leases and all pairing still hold.
    """

    members: tuple[LogicalNotification, ...]
    width: int
    banks: tuple[LocalBank, ...]
    backlog: int | None = None
    persistent: PersistentCellCredit | None = None
    credit: object | None = None
    phase: int | None = None


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
    selected occupation, retaining its logical credit. Missing/None gives no permission.
    Counter FIFO packing retains a whole CounterGroup, including tied-anchor exclusion.
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
    counter_reuses: FrozenEventRelations
    id_pools: EventIdPools
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
    sharing: tuple[EventSharing | LocalEventReuse | CounterFIFOGroup | CounterReuseGroup, ...]
    counter_groups: tuple[CounterFIFOGroup, ...] = ()
    option: int = 0
    counter_reuse_groups: tuple[CounterReuseGroup, ...] = ()
    counter_compositions: tuple = ()


@dataclass(frozen=True, slots=True)
class CounterFIFOGroup:
    """Selected complete credits of one established FIFO word at an actual ID."""

    event: int
    certificate: CounterGroup
    resources: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CounterReuseGroup:
    event: int
    resources: tuple[int, ...]
    messages: tuple[tuple[int, LogicalNotification], ...]
    backlog: int
    rectangles: tuple[SelectedSeparation, ...]


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
    handback_cover_queries: int = 0
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
    allocation_queries: int = 0
    allocation_variables: int = 0
    allocation_constraints: int = 0
    allocation_seconds: float = 0.0
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
    transfer_rules: int = 0
    transfer_order_bands: int = 0
    transfer_range_vertices: int = 0
    completion_queries: int = 0
    completion_forward_cuts: int = 0
    completion_countermodels: int = 0
    completion_rule_instances: int = 0
    completion_conclusions: int = 0
    completion_covered: int = 0
    transfer_capture_words: int = 0
    transfer_capture_bands: int = 0
    consumption_queries: int = 0
    consumption_forward_cuts: int = 0
    consumption_countermodels: int = 0
    consumption_rule_instances: int = 0
    consumption_conclusions: int = 0
    consumption_instances: int = 0
    consumption_regions: int = 0
    consumption_frontier_words: int = 0
    consumption_frontier_queries: int = 0
    consumption_frontier_answer_reuses: int = 0
    consumption_frontier_hits: int = 0
    consumption_fact_reuses: int = 0
    counter_structure_ids: int = 0
    consumption_acquired_boundaries: int = 0
    phase_occupancies: int = 0
    phase_candidates: int = 0


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
    id_pools: EventIdPools
    occupancies: tuple[EventDemand, ...]
    sharing: tuple[EventSharing | LocalEventReuse | CounterFIFOGroup | CounterReuseGroup, ...] = ()
    persistent: tuple[PersistentEventAssignment, ...] = ()
    counter_groups: tuple[CounterFIFOGroup, ...] = ()
    preferred_leases: tuple[StructuredEventLease, ...] = ()
    counter_reuse_groups: tuple[CounterReuseGroup, ...] = ()
    counter_compositions: tuple = ()

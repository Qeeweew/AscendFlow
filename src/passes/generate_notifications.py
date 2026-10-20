"""Construct paired, live, reusable notifications before resource allocation.

Memory/version facts are inputs. Binding establishes the real capture/acquire
positions; the physical-step rule admits their queue/controller schedule; the
queue handback rule constructs each finite/loop/reentry ring. An allocator may
rename these rings and establish sharing constraints, not rediscover progress.
"""

from dataclasses import dataclass
from time import perf_counter

from ..analysis.cell_channels import CellChannel, construct_cell_channels
from ..analysis.cell_ownership import LocalCreditOrder, PersistentCellCredit, bind_cell_credits
from ..analysis.event_contracts import ResourceWork, StructuredEventLease
from ..analysis.event_domains import EventCase, EventDomains
from ..analysis.notification_lifetimes import HandbackRule
from ..analysis.queue_order import QueueOrder, establish_queue_order
from ..analysis.structured_protocols import LogicalNotification, SynchronizationSelection
from .notification_binding import bind_notifications


@dataclass(frozen=True, init=False)
class NotificationProgram:
    selection: SynchronizationSelection
    domains: EventDomains
    order: QueueOrder
    leases: tuple[StructuredEventLease, ...]
    channels: tuple[CellChannel, ...]
    boundaries: tuple[tuple[LogicalNotification, tuple[EventCase, ...]], ...]
    timings: tuple[tuple[str, float], ...]
    credits: tuple[PersistentCellCredit, ...]
    credit_order: LocalCreditOrder

    def __init__(self, *args, **kwargs):
        raise TypeError("construct notifications with generate_notifications")


def generate_notifications(dependencies, dispatch):
    started = perf_counter()
    selection = bind_notifications(dependencies, dispatch)
    return _complete_bindings(selection, binding_s=perf_counter() - started)


def _complete_bindings(selection, *, binding_s=0.0):
    """Internal rule entry, also used by adversarial generation tests."""
    started = perf_counter()
    domains = EventDomains(selection, ResourceWork())
    credits, credit_order = bind_cell_credits(domains)
    paired = perf_counter()
    order = establish_queue_order(domains)
    ordered = perf_counter()
    channels, cell_leases, remaining = construct_cell_channels(domains)
    rule = HandbackRule(domains)
    leases = cell_leases + tuple(rule.lease(n) for n in remaining)
    completed = perf_counter()
    result = object.__new__(NotificationProgram)
    timings = (
        ("notification_binding", binding_s),
        ("endpoint_domains", paired - started),
        ("queue_order", ordered - paired),
        ("credit_lifetimes", completed - ordered),
    )
    for name, value in (
        ("selection", selection),
        ("domains", domains),
        ("order", order),
        ("leases", leases),
        ("channels", channels),
        ("boundaries", tuple(rule.boundary_cases.items())),
        ("timings", timings),
        ("credits", credits),
        ("credit_order", credit_order),
    ):
        object.__setattr__(result, name, value)
    return result

"""Construct paired, live, reusable notifications before resource allocation.

Memory/version facts are inputs. Binding establishes the real capture/acquire
positions; the physical-step rule admits their queue/controller schedule; the
queue handback rule constructs each finite/loop/reentry ring. An allocator may
map these rings to IDs; sharing constraints belong to resource planning, not
allocation or a rediscovery of progress.
"""

from dataclasses import dataclass
from time import perf_counter

from ..analysis.cell_channels import CellChannel, construct_cell_channels
from ..analysis.cell_ownership import LocalCreditOrder, PersistentCellCredit, bind_cell_credits
from ..analysis.completion_prefixes import reduce_completion_prefixes
from ..analysis.event_contracts import ResourceWork, StructuredEventLease
from ..analysis.event_domains import EventDomains
from ..analysis.notification_lifetimes import HandbackRule
from ..analysis.queue_order import QueueOrder, establish_queue_order
from ..analysis.structured_protocols import SynchronizationSelection
from .notification_binding import bind_notifications


@dataclass(frozen=True, init=False)
class NotificationProgram:
    selection: SynchronizationSelection
    domains: EventDomains
    order: QueueOrder
    leases: tuple[StructuredEventLease, ...]
    channels: tuple[CellChannel, ...]
    handbacks: HandbackRule
    timings: tuple[tuple[str, float], ...]
    credits: tuple[PersistentCellCredit, ...]
    credit_order: LocalCreditOrder

    def __init__(self, *args, **kwargs):
        raise TypeError("construct notifications with generate_notifications")


def generate_notifications(dependencies):
    started = perf_counter()
    selection = bind_notifications(dependencies)
    return _complete_bindings(selection, binding_s=perf_counter() - started)


def _complete_bindings(selection, *, binding_s=0.0):
    """Internal rule entry, also used by adversarial generation tests."""
    started = perf_counter()
    domains = EventDomains(selection, ResourceWork())
    domained = perf_counter()
    domains = reduce_completion_prefixes(domains)
    selection = domains.selection
    prefixed = perf_counter()
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
        ("endpoint_domains", domained - started),
        ("completion_prefixes", prefixed - domained),
        ("cell_credits", paired - prefixed),
        ("queue_order", ordered - paired),
        ("credit_lifetimes", completed - ordered),
    )
    for name, value in (
        ("selection", selection),
        ("domains", domains),
        ("order", order),
        ("leases", leases),
        ("channels", channels),
        ("handbacks", rule),
        ("timings", timings),
        ("credits", credits),
        ("credit_order", credit_order),
    ):
        object.__setattr__(result, name, value)
    return result

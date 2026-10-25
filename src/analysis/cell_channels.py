"""Credits owned by an adjacent physical cell lifetime.

RAW credit: Wait precedes the reader; every next writer follows all readers.
WAR credit: Wait precedes that next writer; every next reader follows its
definition. WAW credit follows the adjacent writer chain. Thus reuse is a
local induction on the already established value/storage word, not a query
through arbitrary selected event paths. Different roles retain separate
credits. Successive writers/readers may be different native operations.
"""

from dataclasses import dataclass, replace

from ..scalar import literal
from .access_frontier import QueueEndpoint
from .event_contracts import CaseLease, StructuredEventLease
from .structured_protocols import NotificationKind
from .structured_regions import StorageCell


@dataclass(frozen=True, slots=True)
class CellChannel:
    members: tuple


def owns_stable_cell(notification, control, common_clocks):
    """Whether physical lifetime induction replaces an ordinal/ring window.

    A contracted dense family may have a fixed union footprint. It is not a
    repeated physical cell *inside* its streaming clock. Outside that clock
    its complete family capture is a real, repeatable lifetime boundary.
    """
    root = control.module.body.block.first_op
    return bool(notification.anchors) and all(
        isinstance(n.cell, StorageCell)
        and not isinstance(n.source.endpoint, QueueEndpoint)
        and not isinstance(n.consumer.endpoint, QueueEndpoint)
        and control.invariant_over(n.cell.origin, root)
        and not set(n.cell.streaming).intersection(common_clocks)
        for binding in notification.anchors
        for n in (binding.requirement,)
    )


def _actual_tests(domains, point, case):
    coordinates = tuple((loop.operation, loop.ordinal) for loop in point.occurrence.frame.loops())
    actual = replace(point, coordinates=coordinates, coordinate_guards=())
    return case.condition.comparisons + tuple(
        type(t)(case.specialize(t.lhs), t.predicate, case.specialize(t.rhs))
        for t in domains._ambient(actual)
    )


def _occupations(domains, notification):
    """Actual publication/consumption sites, with their complete activation.

    Different native operations are different occurrences in the cell's
    lifetime word. At the same operation their actual domains must separate.
    In particular two last readers of one version cannot merge: their Waits
    at the common successor writer overlap. Nor can two ready consumers of
    one version merge: their Sets at the common definition overlap.
    """
    result = {}
    for phase in ("sources", "consumers"):
        for point in getattr(notification, phase):
            key = phase, point.actor, point.operation
            for case in domains.cases[notification]:
                if domains.active[notification, case]:
                    result.setdefault(key, []).append(_actual_tests(domains, point, case))
    return result


def construct_cell_channels(domains):
    leases, remaining = [], []
    groups, occupations, members = {}, [], []
    for item in domains.notifications:
        if item.mechanism is NotificationKind.PIPE_FENCE:
            continue
        if not owns_stable_cell(item, domains.control, domains.clocks[item]):
            remaining.append(item)
            continue
        # The anchor bindings already own the actual memory edges. Added
        # completion coverage never changes this physical ownership set.
        key = (
            frozenset(b.requirement.ownership_key for b in item.anchors),
            item.source_engine,
            item.destination_engine,
            item.mechanism,
        )
        sites = _occupations(domains, item)
        candidates = groups.setdefault(key, [])
        for index in candidates:
            if all(
                domains.arithmetic.queries.satisfies(previous + current).impossible
                for site, tests in sites.items()
                for previous in occupations[index].get(site, ())
                for current in tests
            ):
                break
        else:
            index = len(members)
            candidates.append(index)
            members.append([])
            occupations.append({})
        members[index].append(item)
        for site, tests in sites.items():
            occupations[index].setdefault(site, []).extend(tests)
        cases = tuple(
            CaseLease(
                c,
                1,
                literal(0),
                (),
                (),
                (),
                1 if item.mechanism is NotificationKind.CROSS_EVENT else None,
            )
            for c in domains.cases[item]
            if domains.active[item, c]
        )
        leases.append(StructuredEventLease(item, int(bool(cases)), cases))
    channels = tuple(CellChannel(tuple(group)) for group in members)
    domains.work.cell_channels = len(channels)
    domains.work.cell_members = len(leases)
    return channels, tuple(leases), tuple(remaining)

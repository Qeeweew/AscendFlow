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
    owners: tuple
    members: tuple


def owns_stable_cell(notification, control, common_clocks):
    """Whether physical lifetime induction replaces an ordinal/ring window.

    A contracted dense family may have a fixed union footprint. It is not a
    repeated physical cell *inside* its streaming clock. Outside that clock
    its complete family capture is a real, repeatable lifetime boundary.
    """
    root = control.module.body.block.first_op
    return bool(notification.origins) and all(
        isinstance(n.cell, StorageCell)
        and not isinstance(n.source.endpoint, QueueEndpoint)
        and not isinstance(n.consumer.endpoint, QueueEndpoint)
        and control.invariant_over(n.cell.origin, root)
        and not set(n.cell.streaming).intersection(common_clocks)
        for n in notification.origins
    )


def _actual_tests(domains, point, case):
    coordinates = tuple((loop.operation, loop.ordinal) for loop in point.occurrence.frame.loops())
    actual = replace(point, coordinates=coordinates, coordinate_guards=())
    return case.condition.comparisons + tuple(
        type(t)(case.specialize(t.lhs), t.predicate, case.specialize(t.rhs))
        for t in domains._ambient(actual)
    )


def _disjoint(domains, left, right):
    """No duplicate publication or consumption at one native occurrence.

    Different native operations are different occurrences in the cell's
    lifetime word. At the same operation their actual domains must separate.
    In particular two last readers of one version cannot merge: their Waits
    at the common successor writer overlap. Nor can two ready consumers of
    one version merge: their Sets at the common definition overlap.
    """
    for phase in ("sources", "consumers"):
        for a in getattr(left, phase):
            for b in getattr(right, phase):
                if a.actor is not b.actor:
                    continue
                if a.operation is not b.operation:
                    continue
                for ac in domains.cases[left]:
                    for bc in domains.cases[right]:
                        if not domains.active[left, ac] or not domains.active[right, bc]:
                            continue
                        tests = _actual_tests(domains, a, ac) + _actual_tests(domains, b, bc)
                        if not domains.arithmetic.queries.satisfies(tests).impossible:
                            return False
    return True


def construct_cell_channels(domains):
    channels, leases, remaining = [], [], []
    groups = {}
    for item in domains.notifications:
        if item.mechanism is NotificationKind.PIPE_FENCE:
            continue
        origins = item.origins
        if not owns_stable_cell(item, domains.control, domains.clocks[item]):
            remaining.append(item)
            continue
        # The origin is produced by binding, retained by cumulative captures
        # and acquire reuse, and consumed only inside this owned compile.
        # Every physical participant still has its own exact memory edge.
        key = (
            frozenset(n.ownership_key for n in origins),
            item.source_engine,
            item.destination_engine,
            item.mechanism,
        )
        candidates = groups.setdefault(key, [])
        for index in candidates:
            prior = channels[index]
            if all(_disjoint(domains, other, item) for other in prior.members):
                channels[index] = replace(
                    prior,
                    owners=tuple(dict.fromkeys(prior.owners + origins)),
                    members=prior.members + (item,),
                )
                break
        else:
            candidates.append(len(channels))
            channels.append(CellChannel(origins, (item,)))
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
    domains.work.cell_channels = len(channels)
    domains.work.cell_members = len(leases)
    return tuple(channels), tuple(leases), tuple(remaining)

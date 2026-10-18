"""Credits owned by an adjacent physical cell lifetime.

RAW credit: Wait precedes the reader; every next writer follows all readers.
WAR credit: Wait precedes that next writer; every next reader follows its
definition. WAW credit follows the adjacent writer chain. Thus reuse is a
local induction on the already established value/storage word, not a query
through arbitrary selected event paths. Different roles retain separate
credits. Successive writers/readers may be different native operations.
"""

from dataclasses import dataclass, replace

from ..indexing import arguments
from ..scalar import literal, substitute
from .access_frontier import QueueEndpoint
from .event_contracts import CaseLease, StructuredEventLease
from .guard_cover import opposite
from .structured_protocols import NotificationKind
from .structured_regions import StorageCell


@dataclass(frozen=True, slots=True)
class CellChannel:
    owners: tuple
    members: tuple


def _cell_key(cell):
    # Activation is checked on the members, not part of physical identity.
    return (cell.root, cell.private, cell.origin, cell.start, cell.stop, cell.rows, cell.row_stride)


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
                        if not _separated(domains.arithmetic, tests):
                            return False
    return True


def _separated(arithmetic, tests):
    """Mathematical coexistence, after original guard definedness admission.

    A selected LAST is often x == N-1. Substitute that established equality
    before comparing it with x < N-1; interval bounds alone intentionally do
    not bind a variable to a symbolic RHS. This never rewrites emitted guards.
    """
    bindings = {}
    for test in tests:
        if test.predicate != "==":
            continue
        a, b = substitute(test.lhs, bindings), substitute(test.rhs, bindings)
        if a.kind != "argument":
            a, b = b, a
        if a.kind == "argument" and a.args[0] not in arguments(b):
            change = {a: b}
            bindings = {v: substitute(value, change) for v, value in bindings.items()} | change
    reduced = tuple(
        type(t)(substitute(t.lhs, bindings), t.predicate, substitute(t.rhs, bindings))
        for t in tests
    )
    present = set(reduced)
    for t in reduced:
        if t.lhs == t.rhs and t.predicate in {"!=", "<", ">"}:
            return True
        if t.lhs.constant is not None and t.rhs.constant is not None and not t.evaluate({}):
            return True
        if opposite(t) in present:
            return True
    return any(
        arithmetic.prove(opposite(t), reduced[:i] + reduced[i + 1 :]) for i, t in enumerate(reduced)
    )


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
            tuple(
                (
                    _cell_key(n.cell),
                    n.kind,
                    n.source.participant,
                    n.consumer.participant,
                )
                for n in origins
            ),
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

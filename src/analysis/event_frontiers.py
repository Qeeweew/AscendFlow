"""Compositional enclosures of local event publication/consumption intervals.

Each activation class retains its first actual publisher and last actual
consumer. Enclosure is issue order on one real source/controller and one
destination queue, never native completion. Separating two enclosures must
still establish actual consumption handbacks, including every reentry.

The hierarchy factors complete-credit reuse rectangles. A failed coarse
interface is refined through the same rule to real message endpoints; it is
neither a second progress solver nor a dense pair-construction fallback.
Frozen rectangles contain data only. ID assignment cannot query domains.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..budget import poll
from .access_frontier import QueueEndpoint
from .event_frontier_contracts import CreditSeparation, EventFrontier, FrontierSeparation
from .event_sharing import share_event_frontiers
from .structured_schedule import ActionGuard


@dataclass(frozen=True, eq=False, slots=True)
class _CreditNode:
    resources: tuple[int, ...]
    frontiers: tuple[tuple[object, EventFrontier], ...]
    children: tuple[_CreditNode, ...] = ()


class LocalEventFrontiers:
    def __init__(self, domains, handbacks, demands, compatible, *, established=None):
        self.domains, self.handbacks, self.demands = domains, handbacks, demands
        self.compatible = compatible
        self.work = domains.work
        if established is not None:
            assert established.domains is domains, "event frontiers belong to another input program"
        for name in ("_keys", "_leaves", "_separations", "_joins", "_nodes"):
            setattr(self, name, {} if established is None else getattr(established, name))
        self.rows = [[] for _ in demands]

    def _key(self, notification):
        if notification in self._keys:
            return self._keys[notification]
        d = self.domains

        def point(bound):
            path = d.dispatch.position_path(bound)
            return (
                bound.actor,
                bound.placement,
                bound.guard,
                bound.coordinates,
                bound.substitutions,
                bound.coordinate_guards,
                bound.families,
                bound.selected,
                path[:-1],
                path[-1][:2],
                d.dispatch.site(bound).stage,
                bound.operation if isinstance(bound.occurrence.endpoint, QueueEndpoint) else None,
                d._input_ranges(bound),
            )

        key = (
            notification.source_engine,
            notification.destination_engine,
            notification.clock,
            notification.ordinal,
            notification.period,
            notification.distance,
            notification.domains,
            d.tests[notification],
            tuple(map(point, notification.sources)),
            tuple(map(point, notification.consumers)),
            tuple(
                (
                    case.condition,
                    case.substitutions,
                    case.windows,
                    case.projections,
                    case.empty,
                    tuple(
                        d._instance(p, {}, case=case).guard
                        for p in notification.sources + notification.consumers
                    ),
                )
                for case in d.cases[notification]
            ),
        )
        self._keys[notification] = key
        return key

    def _leaf(self, notification):
        if notification not in self._leaves:
            self._leaves[notification] = EventFrontier(
                notification, notification, frozenset((notification,))
            )
        return self._leaves[notification]

    def _join(self, left, right):
        key = left, right
        if key not in self._joins:
            self._joins[key] = self._enclose(left, right)
        return self._joins[key]

    def _enclose(self, left, right):
        d = self.domains
        sources = (left.head, right.head)
        consumers = (left.tail, right.tail)
        head = min(sources, key=lambda n: d.dispatch.site(n.sources[0]).lexical)
        tail = max(consumers, key=lambda n: d.dispatch.site(n.consumers[0]).lexical)
        # Two separate notification instructions at one native anchor have
        # no relative instruction order in these endpoint facts. Equality
        # of native positions must not turn one Wait into the other's last
        # consumer or one Set into the other's first publisher.
        if any(
            n is not head
            and d.dispatch.site(n.sources[0]).lexical == d.dispatch.site(head.sources[0]).lexical
            for n in sources
        ) or any(
            n is not tail
            and d.dispatch.site(n.consumers[0]).lexical
            == d.dispatch.site(tail.consumers[0]).lexical
            for n in consumers
        ):
            return None
        bounds = tuple((head.sources[0], n.sources[0]) for n in sources) + tuple(
            (n.consumers[0], tail.consumers[0]) for n in consumers
        )
        if any(d._order_conditions(a, b) != (ActionGuard(),) for a, b in bounds):
            return None
        self.work.frontier_enclosures += 1
        return EventFrontier(head, tail, left.members | right.members, (left, right), bounds)

    def _combine(self, entries):
        result = {}
        for key, frontier in entries:
            choices = result.setdefault(key, [])
            for i, prior in enumerate(choices):
                if (joined := self._join(prior, frontier)) is not None:
                    choices[i] = joined
                    break
            else:
                choices.append(frontier)
        return tuple((key, p) for key, values in result.items() for p in values)

    def _node(self, resources):
        key = tuple((number, self.demands[number].members) for number in resources)
        if key not in self._nodes:
            self._nodes[key] = self._construct_node(resources)
        return self._nodes[key]

    def _construct_node(self, resources):
        if len(resources) == 1:
            number = resources[0]
            entries = ((self._key(n), self._leaf(n)) for n in self.demands[number].members)
            return _CreditNode(resources, self._combine(entries))
        middle = len(resources) // 2
        left, right = self._node(resources[:middle]), self._node(resources[middle:])
        return _CreditNode(
            resources, self._combine(left.frontiers + right.frontiers), (left, right)
        )

    def _separate(self, left, right):
        key = left, right
        if key in self._separations:
            return self._separations[key]
        self.work.frontier_queries += 1
        if len(left.members) == len(right.members) == 1:
            permission = self.compatible(left.head, right.head)
            result = (
                None
                if permission is None
                else (FrontierSeparation(left, right, None, (), permission),)
            )
        else:
            if self.domains._exclusive(left.head, right.head):
                result = (FrontierSeparation(left, right, None, ()),)
            elif (
                proof := share_event_frontiers(self.domains, self.handbacks, left, right)
            ) is not None:
                forward, handbacks = proof
                result = (FrontierSeparation(left, right, forward, handbacks),)
            else:
                children, other, reverse = (
                    (left.children, right, False)
                    if left.children
                    and (not right.children or len(left.members) >= len(right.members))
                    else (right.children, left, True)
                )
                assert children, "nontrivial event enclosure lost its actual child interfaces"
                parts = []
                for child in children:
                    proof = (
                        self._separate(other, child) if reverse else self._separate(child, other)
                    )
                    if proof is None:
                        result = None
                        break
                    parts.extend(proof)
                else:
                    result = tuple(parts)
        self._separations[key] = result
        return result

    def _between(self, left, right):
        poll("local event consumption frontiers", queries=self.work.frontier_queries)
        self.work.frontier_blocks += 1
        proofs = []
        for _, a in left.frontiers:
            for _, b in right.frontiers:
                proof = self._separate(a, b)
                if proof is None:
                    break
                proofs.extend(proof)
            else:
                continue
            break
        else:
            left_mask = sum(1 << number for number in left.resources)
            right_mask = sum(1 << number for number in right.resources)
            certificate = CreditSeparation(tuple(proofs), left_mask, right_mask)
            for a, mask in ((left, right_mask), (right, left_mask)):
                for number in a.resources:
                    self.rows[number].append((mask, certificate))
            self.work.frontier_rectangles += 1
            return
        if not left.children and not right.children:
            return
        children, other, reverse = (
            (left.children, right, False)
            if left.children and (not right.children or len(left.resources) >= len(right.resources))
            else (right.children, left, True)
        )
        for child in children:
            self._between(other, child) if reverse else self._between(child, other)

    def _within(self, node):
        if node.children:
            left, right = node.children
            self._within(left)
            self._within(right)
            self._between(left, right)

    def build(self, pressured):
        banks = {}
        for number, demand in enumerate(self.demands):
            if demand.backlog is None and demand.persistent is None:
                for bank in demand.banks:
                    if bank in pressured:
                        banks.setdefault(bank, []).append(number)
        positions = {operation: i for i, operation in enumerate(self.domains.control.frames)}
        for resources in banks.values():
            resources.sort(
                key=lambda r: min(
                    positions[p.operation] for n in self.demands[r].members for p in n.sources
                )
            )
            if len(resources) > 1:
                self._within(self._node(tuple(resources)))
        return tuple(tuple(row) for row in self.rows)

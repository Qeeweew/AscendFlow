"""Establish virtual event actions and finite reuse constraints before coloring.

Only this stage can query occurrence domains or handbacks. The output has no
query callbacks; changing candidate IDs cannot trigger another domain proof.
"""

from itertools import combinations
from types import MappingProxyType

from ..budget import poll
from ..diagnostics import require
from ..dialects.physical import FixConfigOp
from ..scalar import expression, literal
from ..target.ascend910b import Actor, Engine, SyncKind
from .event_coloring import counter_fifo_partition, exclusive_pressure_banks
from .event_contracts import (
    EventDemand,
    EventIdPools,
    EventReservation,
    EventResourceProblem,
    VirtualEventAction,
)
from .event_counter_groups import plan_counter_groups, plan_counter_time_groups
from .event_frontier_contracts import FrozenEventRelations
from .event_frontiers import LocalEventFrontiers
from .event_peers import bind_event_peers
from .event_sharing import share_counter_time, share_counters, share_events
from .notification_lifetimes import HandbackRule
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard, Placement


def plan_event_resources(program, *, id_pools=None):
    from ..passes.generate_notifications import NotificationProgram

    require(
        isinstance(program, NotificationProgram),
        "resource planning requires generated notification lifetimes",
    )
    id_pools = EventIdPools() if id_pools is None else id_pools
    require(isinstance(id_pools, EventIdPools), "event pools must be frozen target numbers")
    planner = _Planning(program, id_pools=id_pools)
    preferred = planner.build()
    alternatives = []
    leases = dict(planner.leases)
    grouped = {b: a for a, b in preferred.peers}
    for number, demand in enumerate(preferred.demands):
        if number in grouped or demand.persistent is not None:
            continue
        resources = (number,) + tuple(b for a, b in preferred.peers if a == number)
        banks = (
            {None}
            if demand.backlog is not None
            else {bank for r in resources for bank in preferred.demands[r].banks}
        )
        if not banks.intersection(planner.pressured):
            continue
        proposed = dict(leases)
        changed = False
        for resource in resources:
            for member in preferred.demands[resource].members:
                old = leases[member]
                if not any(width > 1 for c in old.cases for _, width in c.rings):
                    continue
                smaller = planner.handbacks.smaller_lease(old)
                if smaller.width < old.width or (
                    demand.backlog is not None
                    and max(c.backlog for c in smaller.cases) < max(c.backlog for c in old.cases)
                ):
                    proposed[member] = smaller
                    changed = True
        if not changed:
            continue
        candidate = _Planning(
            program, id_pools=id_pools, leases=proposed, established=planner
        ).build()
        if not set(preferred.peers) <= set(candidate.peers):
            # Do not trade a mirrored protocol for new SID-specific code.
            continue
        leases = proposed
        alternatives.append(candidate)
    object.__setattr__(preferred, "alternatives", tuple(alternatives))
    preferred.work.protocol_candidates = 1 + len(alternatives)
    return preferred


class _Planning:
    def __init__(self, program, *, id_pools, leases=None, established=None):
        self.program = program
        self.id_pools = id_pools
        self.domains = program.domains
        self.leases = {v.notification: v for v in program.leases} if leases is None else leases
        self.channels = {m: c for c in program.channels for m in c.members}
        self.demands, self.uses, self.actions, self.reservations = [], [], [], []
        self.handbacks = (
            HandbackRule(self.domains, boundaries=program.boundaries)
            if established is None
            else established.handbacks
        )
        self.compatibility = {} if established is None else established.compatibility
        self.counter_compatibility = (
            {} if established is None else established.counter_compatibility
        )
        self.paired = set() if established is None else established.paired
        self.pressured = set()
        self.frontiers = None if established is None else established.frontiers

    def action(
        self,
        anchor,
        placement,
        actors,
        kind,
        source,
        destination,
        *,
        guard=None,
        protocols=(),
        resource=None,
        phase=None,
        fixed_id=None,
    ):
        guard = ActionGuard() if guard is None else guard
        self.actions.append(
            VirtualEventAction(
                anchor,
                placement,
                actors,
                kind,
                source,
                destination,
                guard,
                protocols,
                resource,
                phase,
                fixed_id,
            )
        )

    def reservations_and_credits(self):
        root = self.domains.memory.module.body.block.first_op
        for operation in self.domains.control.frames:
            engine = Engine.FIX if isinstance(operation, FixConfigOp) else None
            if engine is None:
                continue
            self.reservations.append(
                EventReservation(
                    (Actor.AIC, Engine.S, engine),
                    0,
                    "native configuration packet notify/wait",
                    operation,
                )
            )
        drained = set()
        for task in self.domains.dispatch.tasks.values():
            bank = task.actor, task.engine, Engine.S
            if task.engine is not Engine.S and bank not in drained:
                drained.add(bank)
                self.reservations.append(
                    EventReservation(bank, 7, "final native retirement drain", root)
                )
        for credit in self.program.credits:
            lifetime = credit.lifetime
            actor = lifetime.cell.private
            source, destination = lifetime.source_engine, lifetime.destination_engine
            resource = len(self.demands)
            self.demands.append(
                EventDemand((), 1, ((actor, source, destination),), persistent=credit)
            )
            for placement, kind in (
                (Placement.ENTRY, SyncKind.LOCAL_SET),
                (Placement.EXIT, SyncKind.LOCAL_WAIT),
            ):
                self.action(
                    root,
                    placement,
                    (actor,),
                    kind,
                    source,
                    destination,
                    protocols=(credit.name,),
                    resource=resource,
                    phase=literal(0),
                )
            for body in credit.rounds:
                for point, kind in (
                    (body.acquire, SyncKind.LOCAL_WAIT),
                    (body.release, SyncKind.LOCAL_SET),
                ):
                    self.action(
                        point.operation,
                        point.placement,
                        (actor,),
                        kind,
                        source,
                        destination,
                        guard=point.guard,
                        protocols=(credit.name,),
                        resource=resource,
                        phase=literal(0),
                    )

    def streams(self):
        known = {}
        for notification in self.domains.notifications:
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                for point in notification.consumers:
                    self.action(
                        point.operation,
                        point.placement,
                        (point.actor,),
                        SyncKind.FENCE,
                        notification.source_engine,
                        notification.destination_engine,
                        guard=point.guard,
                        protocols=(notification.name,),
                    )
                continue
            lease = self.leases[notification]
            if not lease.width:
                self.uses.append((notification, None))
                continue
            key = self.channels.get(notification, notification)
            if key not in known:
                members = tuple(
                    m
                    for m in (key.members if key is not notification else (notification,))
                    if self.leases[m].width
                )
                cross = notification.mechanism is NotificationKind.CROSS_EVENT
                banks = (
                    ()
                    if cross
                    else tuple(
                        dict.fromkeys(
                            (p.actor, m.source_engine, m.destination_engine)
                            for m in members
                            for p in m.sources
                        )
                    )
                )
                width = max(self.leases[m].width for m in members)
                backlog = None
                if cross:
                    require(
                        all(c.backlog is not None for m in members for c in self.leases[m].cases),
                        "counter demand needs every active case's established backlog",
                    )
                    backlog = max(c.backlog for m in members for c in self.leases[m].cases)
                known[key] = len(self.demands)
                self.demands.append(EventDemand(members, width, banks, backlog))
            resource = known[key]
            self.uses.append((notification, resource))
            for case in lease.cases:
                phase = (
                    literal(0)
                    if case.width == 1
                    else expression("rem", case.ordinal, literal(case.width))
                )
                for sources, bounds in (
                    (True, notification.sources),
                    (False, notification.consumers),
                ):
                    cross = notification.mechanism is NotificationKind.CROSS_EVENT
                    kind = (
                        (SyncKind.CROSS_SET if sources else SyncKind.CROSS_WAIT)
                        if cross
                        else (SyncKind.LOCAL_SET if sources else SyncKind.LOCAL_WAIT)
                    )
                    for bound in bounds:
                        # Equality of the virtual phase through both coordinate
                        # maps implies equality after *any* common finite map.
                        # No numeric assignment is needed for this premise.
                        key = bound, case.case, phase
                        if key not in self.paired:
                            self.domains._paired_id(lease, bound, phase, case=case.case)
                            self.paired.add(key)
                        self.action(
                            bound.operation,
                            bound.placement,
                            (bound.actor,),
                            kind,
                            notification.source_engine,
                            notification.destination_engine,
                            guard=ActionGuard(
                                case.case.condition.comparisons + bound.guard.comparisons
                            ),
                            protocols=(notification.name,),
                            resource=resource,
                            phase=bound.expression(phase),
                        )

    def compatible(self, left, right):
        pair = frozenset((left, right))
        if pair not in self.compatibility:
            if self.domains._exclusive(left, right):
                result = ()
            else:
                witness = (
                    share_counters(self.domains, left, right)
                    if left.mechanism is NotificationKind.CROSS_EVENT
                    else share_events(self.domains, self.handbacks, left, right)
                )
                result = None if witness is None else (witness,)
            self.compatibility[pair] = result
            self.domains.work.sharing_pairs += 1
        else:
            self.domains.work.sharing_reuses += 1
        return self.compatibility[pair]

    def relations(self, peers):
        result = {}
        # Facts about optional reuse must not choose a different mapping for
        # a bank that can hold every preferred credit exclusively.
        pressured = exclusive_pressure_banks(self.demands, self.reservations, peers, self.id_pools)
        self.pressured = pressured
        counters = [i for i, demand in enumerate(self.demands) if demand.backlog is not None]
        for a, b in combinations(counters, 2):
            left, right = self.demands[a], self.demands[b]
            if left.persistent is not None or right.persistent is not None:
                continue
            if None not in pressured:
                continue
            poll("virtual event reuse constraints", pairs=len(result))
            witnesses = []
            for prior in left.members:
                for member in right.members:
                    relation = self.compatible(prior, member)
                    if relation is None:
                        break
                    witnesses.extend(relation)
                else:
                    continue
                result[a, b] = None
                break
            else:
                result[a, b] = tuple(witnesses)
        self.frontiers = LocalEventFrontiers(
            self.domains, self.handbacks, self.demands, self.compatible, established=self.frontiers
        )
        return FrozenEventRelations(result, self.frontiers.build(pressured))

    def time_relations(self, counter_groups):
        result = {}
        if None not in self.pressured:
            return MappingProxyType(result)
        partition = counter_fifo_partition(
            self.demands, tuple(frozenset(group.resources) for group in counter_groups)
        )
        if partition is not None and len(partition) <= len(self.id_pools.cross):
            # A concrete certified FIFO partition already fits. Each group
            # has its own whole word, progress, quorum and backlog certificate;
            # no new consumption question is needed for candidate integers.
            self.domains.work.counter_structure_ids = len(partition)
            return MappingProxyType(result)
        counters = [i for i, d in enumerate(self.demands) if d.backlog is not None]
        for a, b in combinations(counters, 2):
            witnesses = []
            for left in self.demands[a].members:
                for right in self.demands[b].members:
                    key = frozenset((left, right))
                    if key not in self.counter_compatibility:
                        poll("mode-2 consumption handbacks", pairs=len(self.counter_compatibility))
                        self.counter_compatibility[key] = share_counter_time(
                            self.domains, self.handbacks, left, right
                        )
                        self.domains.work.counter_time_pairs += 1
                    else:
                        self.domains.work.counter_time_cache_hits += 1
                    witness = self.counter_compatibility[key]
                    if witness is None:
                        break
                    witnesses.append(witness)
                else:
                    continue
                break
            else:
                result[a, b] = tuple(witnesses)
        return MappingProxyType(result)

    def build(self):
        self.reservations_and_credits()
        self.streams()
        for reservation in self.reservations:
            if reservation.reason != "final native retirement drain":
                continue
            actor, source, destination = reservation.bank
            for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT):
                self.action(
                    reservation.anchor,
                    Placement.EXIT,
                    (actor,),
                    kind,
                    source,
                    destination,
                    protocols=(f"final_{actor.value}_{source.value}",),
                    fixed_id=reservation.event,
                )
        peers, actions = bind_event_peers(self.demands, self.actions)
        relations = self.relations(peers)
        self.domains.work.peer_equalities = len(peers)
        counter_groups = plan_counter_groups(
            self.program,
            self.demands,
            relations,
            compatible=self.compatible,
            handbacks=self.handbacks,
        )
        self.domains.work.counter_groups = len(counter_groups)
        time_relations = self.time_relations(counter_groups)
        compositions = plan_counter_time_groups(
            self.program, self.demands, counter_groups, time_relations
        )
        self.domains.work.counter_compositions = len(compositions)
        result = object.__new__(EventResourceProblem)
        for name, value in (
            ("selection", self.program.selection),
            ("order", self.program.order),
            ("leases", tuple(self.leases.values())),
            ("channels", self.program.channels),
            ("demands", tuple(self.demands)),
            ("uses", tuple(self.uses)),
            ("actions", actions),
            ("peers", peers),
            ("counter_groups", counter_groups),
            ("counter_compositions", compositions),
            ("counter_reuses", time_relations),
            ("id_pools", self.id_pools),
            ("reservations", tuple(self.reservations)),
            ("relations", relations),
            ("work", self.domains.work),
            ("alternatives", ()),
        ):
            object.__setattr__(result, name, value)
        return result

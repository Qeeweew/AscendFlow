"""Compose event-domain, lifetime, progress and bank-allocation contracts.

No stage rediscovers memory dependencies. Endpoint activation is established
once, then reused by progress, consumption and allocation. This module owns
only the finite hardware bank ledger and construction of materialization
records; symbolic domain and queue reasoning live in their named stages.
"""

from dataclasses import replace
from functools import partial

from ..diagnostics import require
from ..dialects.physical import FixConfigOp, Load3DConfigOp
from ..scalar import expression, literal
from ..target.ascend910b import Actor, Engine, SyncKind
from .event_contracts import (
    LOCAL_CAPACITY,
    SUPPORTED_CROSS_IDS,
    EventReservation,
    ResourceWork,
    StructuredEventAssignment,
    StructuredEventPlan,
)
from .event_domains import EventDomains
from .event_lifetimes import establish_lease
from .event_paths import prove_event_path_cover
from .event_progress import prove_progress
from .structured_event_sharing import EventSharingContext, prove_finite_event_sharing
from .structured_protocols import NotificationKind, SynchronizationSelection
from .structured_schedule import ActionGuard, Placement, SyncAction


def allocate_structured_events(selection: SynchronizationSelection) -> StructuredEventPlan:
    """Establish each resource obligation once and materialize its instructions."""
    work = ResourceWork(notifications=len(selection.notifications))
    domains = EventDomains(selection, work)
    progress = prove_progress(domains)
    lifetimes = {
        notification: establish_lease(domains, notification)
        for notification in domains.notifications
        if notification.mechanism is not NotificationKind.PIPE_FENCE
    }
    return _EventBanks(domains).allocate(lifetimes, progress)


class _EventBanks:
    def __init__(self, domains):
        self.domains = domains
        self.actions, self.assignments, self.reservations, self.sharing = [], [], [], []
        self.occupied, self.cross_occupied = {}, {}
        self.sharing_context = EventSharingContext(
            domains.selection,
            domains.cases,
            domains._instance,
            partial(prove_event_path_cover, domains),
        )

    def allocate(self, lifetimes, progress):
        self._reserve()
        for notification in self.domains.notifications:
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                self._place(notification, None)
                continue
            lease = lifetimes[notification]
            assignment = self._assign(notification, lease)
            self.assignments.append(assignment)
            self._place(notification, assignment)
        self._drain()
        return StructuredEventPlan(
            selection=self.domains.selection,
            actions=self._combine_actions(),
            assignments=tuple(self.assignments),
            leases=tuple(lifetimes.values()),
            reservations=tuple(self.reservations),
            progress=progress,
            work=self.domains.work,
            sharing=tuple(self.sharing),
        )

    def _reserve(self):
        root = self.domains.memory.module.body.block.first_op
        for operation in self.domains.control.frames:
            engine = (
                Engine.FIX
                if isinstance(operation, FixConfigOp)
                else Engine.MTE1
                if isinstance(operation, Load3DConfigOp)
                else None
            )
            if engine is not None:
                bank = (Actor.AIC, Engine.S, engine)
                self.occupied.setdefault(bank, {})[0] = [None]
                self.reservations.append(
                    EventReservation(bank, 0, "native configuration packet notify/wait", operation)
                )
        for task in self.domains.dispatch.tasks.values():
            if task.engine is Engine.S:
                continue
            bank = (task.actor, task.engine, Engine.S)
            if 7 not in self.occupied.setdefault(bank, {}):
                self.occupied[bank][7] = [None]
                self.reservations.append(
                    EventReservation(bank, 7, "final native retirement drain", root)
                )

    def _assign(self, notification, lease):
        width = lease.width
        cross = notification.mechanism is NotificationKind.CROSS_EVENT
        banks = (
            ()
            if cross
            else tuple(
                dict.fromkeys(
                    (bound.actor, notification.source_engine, notification.destination_engine)
                    for bound in notification.sources
                )
            )
        )
        pools = (
            (self.cross_occupied,)
            if cross
            else tuple(self.occupied.setdefault(bank, {}) for bank in banks)
        )
        capacity = len(SUPPORTED_CROSS_IDS) if cross else LOCAL_CAPACITY
        for base in range(capacity - width + 1):
            ids = tuple(range(base, base + width))
            witnesses = []
            priors = dict.fromkeys(
                prior for pool in pools for flag in ids for prior in pool.get(flag, ())
            )
            for prior in priors:
                if prior is None:
                    break  # Native initializer/final-drain reservations are never borrowed.
                if self.domains._exclusive(prior, notification):
                    continue
                witness = prove_finite_event_sharing(
                    prior, notification, context=self.sharing_context
                )
                if witness is None:
                    break
                witnesses.append(witness)
            else:
                self.sharing.extend(witnesses)
                for pool in pools:
                    for flag in ids:
                        pool.setdefault(flag, []).append(notification)
                if cross:
                    self.domains.work.assigned_cross_ids += width
                else:
                    self.domains.work.assigned_local_ids += width * len(banks)
                return StructuredEventAssignment(notification, ids, banks, lease.cases)
        if cross:
            self.domains.fail(
                notification,
                f"current supported mode-2 allocation range {SUPPORTED_CROSS_IDS} is exhausted; no counter-sharing proof was established",
            )
        self.domains.fail(
            notification,
            f"directed actor-local allocation cannot find {width} contiguous IDs in {banks} "
            "within capacity 8 including initializer/drain reservations; "
            "the supported lifetime-sharing proofs do not establish another assignment",
        )

    def _place(self, notification, assignment):
        fence = notification.mechanism is NotificationKind.PIPE_FENCE
        cross = notification.mechanism is NotificationKind.CROSS_EVENT
        if fence:
            for bound in notification.consumers:
                self.actions.append(
                    SyncAction(
                        bound.operation,
                        bound.placement,
                        (bound.actor,),
                        SyncKind.FENCE,
                        notification.source_engine,
                        notification.destination_engine,
                        guard=bound.guard,
                        protocol=notification.name,
                    )
                )
            return
        require(assignment is not None, "notification needs its allocated event")
        for lease in assignment.cases:
            case = lease.case
            event = (
                literal(assignment.ids[0])
                if lease.width == 1
                else expression(
                    "add",
                    literal(assignment.ids[0]),
                    expression("rem", lease.ordinal, literal(lease.width)),
                )
            )
            for sources, bounds in ((True, notification.sources), (False, notification.consumers)):
                kind = (
                    (SyncKind.CROSS_SET if sources else SyncKind.CROSS_WAIT)
                    if cross
                    else (SyncKind.LOCAL_SET if sources else SyncKind.LOCAL_WAIT)
                )
                for bound in bounds:
                    self.domains._paired_id(assignment, bound, event, case=case)
                    self.actions.append(
                        SyncAction(
                            bound.operation,
                            bound.placement,
                            (bound.actor,),
                            kind,
                            notification.source_engine,
                            notification.destination_engine,
                            bound.expression(event),
                            ActionGuard(case.condition.comparisons + bound.guard.comparisons),
                            notification.name,
                        )
                    )

    def _drain(self):
        for reservation in self.reservations:
            if reservation.reason != "final native retirement drain":
                continue
            actor, source, destination = reservation.bank
            for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT):
                self.actions.append(
                    SyncAction(
                        reservation.anchor,
                        Placement.EXIT,
                        (actor,),
                        kind,
                        source,
                        destination,
                        literal(reservation.event),
                        protocol=f"final_{actor.value}_{source.value}",
                    )
                )

    def _combine_actions(self):
        combined, indices = [], {}
        for action in self.actions:
            # AIV bodies are represented once. Combining equal instructions
            # retains both physical participants without cloning their body.
            family = "cube" if action.participants == (Actor.AIC,) else "vector"
            key = (
                action.anchor,
                action.placement,
                action.kind,
                action.source,
                action.destination,
                action.event,
                action.guard,
                action.protocol,
                family,
            )
            index = indices.get(key)
            if index is None:
                indices[key] = len(combined)
                combined.append(action)
            else:
                prior = combined[index]
                combined[index] = replace(
                    prior,
                    participants=tuple(dict.fromkeys(prior.participants + action.participants)),
                )
        return tuple(combined)

"""Allocate finite banks from generated credits and explicit new sharing.

No domain, memory or progress graph is rebuilt here. A cell credit's members
already share one lifetime; other ID sharing establishes a consume-to-Set
handback or aligned per-participant counter FIFO with a whole-credit budget.
Materialization receives the final IDs and unchanged real instruction sites.
"""

from dataclasses import replace

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import FixConfigOp, Load3DConfigOp
from ..scalar import expression, literal
from ..target.ascend910b import Actor, Engine, SyncKind
from .event_contracts import (
    CROSS_BACKLOG,
    LOCAL_CAPACITY,
    SUPPORTED_CROSS_IDS,
    EventBankUsage,
    EventReservation,
    PersistentEventAssignment,
    StructuredEventAssignment,
    StructuredEventPlan,
)
from .event_sharing import share_counters, share_events
from .notification_lifetimes import HandbackRule
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard, Placement, SyncAction


def allocate_structured_events(program) -> StructuredEventPlan:
    """Consume generated lifetimes; establish only bank/ID/sharing constraints."""
    from ..passes.generate_notifications import NotificationProgram

    require(
        isinstance(program, NotificationProgram),
        "allocation requires generated notification lifetimes",
    )
    return _EventBanks(
        program.domains, program.channels, program.boundaries, program.credits
    ).allocate({lease.notification: lease for lease in program.leases}, program.order)


class _EventBanks:
    def __init__(self, domains, channels, boundaries, credits):
        self.domains = domains
        self.actions, self.assignments, self.reservations, self.sharing = [], [], [], []
        self.occupied, self.cross_occupied = {}, {}
        self.handbacks = HandbackRule(domains, boundaries=boundaries)
        self.channels = {member: channel for channel in channels for member in channel.members}
        self.cell_channels = channels
        self.channel_ids = {}
        self.compatibility = {}
        self.credits = credits
        self.persistent = []

    def _compatible(self, left, right):
        # Candidate IDs do not change these members, domains or lifetimes.
        # Cache the complete relation, including a declined optional sharing.
        # A successful relation retains its real consumption handbacks.
        pair = frozenset((left, right))
        if pair in self.compatibility:
            self.domains.work.sharing_reuses += 1
            return self.compatibility[pair]
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
        return result

    def allocate(self, lifetimes, order):
        self.lifetimes = lifetimes
        self._reserve()
        self._persistent_credits()
        for notification in self.domains.notifications:
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                self._place(notification, None)
                continue
            lease = lifetimes[notification]
            assignment = self._assign(notification, lease)
            self.assignments.append(assignment)
            self._place(notification, assignment)
        self._drain()
        self.domains.work.allocated_local_ids = sum(len(pool) for pool in self.occupied.values())
        self.domains.work.allocated_cross_ids = len(self.cross_occupied)
        return StructuredEventPlan(
            selection=self.domains.selection,
            actions=_combine_actions(self.actions),
            assignments=tuple(self.assignments),
            leases=tuple(lifetimes.values()),
            reservations=tuple(self.reservations),
            order=order,
            work=self.domains.work,
            sharing=tuple(self.sharing),
            channels=self.cell_channels,
            persistent=tuple(self.persistent),
            banks=tuple(
                EventBankUsage(bank, tuple(sorted(pool))) for bank, pool in self.occupied.items()
            )
            + (EventBankUsage(None, tuple(sorted(self.cross_occupied))),),
        )

    def _persistent_credits(self):
        """One real bank slot, seed and final reclaim for each owned cell."""
        root = self.domains.memory.module.body.block.first_op
        for credit in self.credits:
            lifetime = credit.lifetime
            actor = lifetime.cell.private
            source, destination = lifetime.source_engine, lifetime.destination_engine
            bank = actor, source, destination
            pool = self.occupied.setdefault(bank, {})
            event = next((flag for flag in range(LOCAL_CAPACITY) if flag not in pool), None)
            if event is None:
                raise CompileError(
                    f"persistent ownership exhausts the directed local bank {bank} within capacity 8",
                    kind=ErrorKind.UNSUPPORTED,
                    phase="persistent-cell-resources",
                    location=str(credit.rounds[0].acquire.operation.location),
                )
            pool[event] = [None]  # Ordinary sharing cannot borrow this live token.
            self.reservations.append(
                EventReservation(bank, event, f"persistent ownership {credit.name}", root)
            )
            self.persistent.append(PersistentEventAssignment(credit, bank, event))
            for placement, kind in (
                (Placement.ENTRY, SyncKind.LOCAL_SET),
                (Placement.EXIT, SyncKind.LOCAL_WAIT),
            ):
                self.actions.append(
                    SyncAction(
                        root,
                        placement,
                        (actor,),
                        kind,
                        source,
                        destination,
                        literal(event),
                        protocols=(credit.name,),
                    )
                )
            for body in credit.rounds:
                for point, kind in (
                    (body.acquire, SyncKind.LOCAL_WAIT),
                    (body.release, SyncKind.LOCAL_SET),
                ):
                    self.actions.append(
                        SyncAction(
                            point.operation,
                            point.placement,
                            (actor,),
                            kind,
                            source,
                            destination,
                            literal(event),
                            point.guard,
                            (credit.name,),
                        )
                    )
        self.domains.work.persistent_cells = len(self.credits)

    def _counter_backlog(self, members):
        """Count each complete physical credit once, including later members."""
        credits = {}
        for member in members:
            lease = self.lifetimes[member]
            require(
                all(case.backlog is not None for case in lease.cases),
                "counter allocation requires an established backlog for every active case",
            )
            backlog = max((case.backlog for case in lease.cases), default=0)
            credit = self.channels.get(member, member)
            credits[credit] = max(credits.get(credit, 0), backlog)
        return sum(credits.values())

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
                if isinstance(operation, Load3DConfigOp) and operation.fp32 is not None:
                    precision_bank = (Actor.AIC, Engine.S, Engine.M)
                    self.occupied.setdefault(precision_bank, {})[0] = [None]
                    self.reservations.append(
                        EventReservation(precision_bank, 0, "IEEE FP32 Cube mode", operation)
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
        channel = self.channels.get(notification)
        members = (
            (notification,)
            if channel is None
            else tuple(
                member
                for member in channel.members
                if any(self.domains.active[member, c] for c in self.domains.cases[member])
            )
        )
        if channel is not None and channel in self.channel_ids:
            ids = self.channel_ids[channel]
            return StructuredEventAssignment(notification, ids, banks, lease.cases)
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
            if cross and self._counter_backlog((*priors, *members)) > CROSS_BACKLOG:
                continue
            for prior in priors:
                if prior is None:
                    break  # Native initializer/final-drain reservations are never borrowed.
                for member in members:
                    relation = self._compatible(prior, member)
                    if relation is None:
                        break
                    witnesses.extend(relation)
                else:
                    continue
                # Sharing introduces a constraint on the *whole* credit,
                # including members materialized later in notification order.
                # A handback for one finite fragment cannot license its loop.
                break
            else:
                if cross:
                    self.domains.work.maximum_counter_backlog = max(
                        self.domains.work.maximum_counter_backlog,
                        self._counter_backlog((*priors, *members)),
                    )
                self.sharing.extend(witnesses)
                for pool in pools:
                    for flag in ids:
                        pool.setdefault(flag, []).extend(members)
                if channel is not None and width:
                    self.channel_ids[channel] = ids
                return StructuredEventAssignment(notification, ids, banks, lease.cases)
        if cross:
            self.domains.fail(
                notification,
                f"current supported mode-2 allocation range {SUPPORTED_CROSS_IDS} is exhausted; no counter-sharing proof was established",
            )
        self.domains.fail(
            notification,
            f"directed actor-local allocation cannot find {width} contiguous IDs in {banks} "
            "within capacity 8 including persistent ownership, initializer and drain reservations; "
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
                        protocols=(notification.name,),
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
                            (notification.name,),
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
                        protocols=(f"final_{actor.value}_{source.value}",),
                    )
                )


def _combine_actions(actions):
    """Share equal AIV instruction sequences without changing either actor's trace.

    Placement determines the actual native/controller insertion point. Within
    one point, every actor retains its ordered sequence, including duplicate
    instructions. Equal sequences can be printed once for both participants;
    asymmetric sequences keep their existing participant guards. Protocol names
    retain provenance but cannot distinguish otherwise identical instructions.
    This changes no allocated IDs, lifetimes or quorum contributions.
    """
    groups = {}
    for action in actions:
        family = "cube" if action.participants == (Actor.AIC,) else "vector"
        groups.setdefault((action.anchor, action.placement, family), []).append(action)

    def instruction(action):
        return action.kind, action.source, action.destination, action.event, action.guard

    combined = []
    for (_, _, family), group in groups.items():
        if family == "cube":
            combined.extend(group)
            continue
        left = [action for action in group if Actor.AIV0 in action.participants]
        right = [action for action in group if Actor.AIV1 in action.participants]
        if len(left) != len(right) or any(
            instruction(a) != instruction(b) for a, b in zip(left, right, strict=True)
        ):
            combined.extend(group)
            continue
        for a, b in zip(left, right, strict=True):
            combined.append(
                replace(
                    a,
                    participants=(Actor.AIV0, Actor.AIV1),
                    protocols=tuple(dict.fromkeys(a.protocols + b.protocols)),
                )
            )
    return tuple(combined)

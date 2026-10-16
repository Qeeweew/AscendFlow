"""Historical M2 conditional binary-event leases used by the test oracle.

A lease proves old Wait consumption precedes the next Set effect using that
ID. It does not prove the selected protocols can make progress: in particular
zero-distance deadlocks require a separate scheduling check. No flag-induced
order, native-end FIFO, or Set-to-source-queue edge is used. Only a Wait whose
actual destination is Scalar may gate subsequent controller dispatch.

A roundtrip uses one selected event returning to the source queue or to a
Scalar controller gate. A direct Scalar proof uses its blocking native end
before strictly later source dispatch. Independent channels have independent
periods; projection couples only actual interfaces. Work is bounded explicitly
and never depends on the runtime trip count N. Banks are allocated exclusively;
cross-channel ID sharing is not implemented.

CANN 9.1 asc/impl/basic_api/kernel_macros.h sets QUE_MAX_EVENT=8 for arch 2201.
kernel_tpipe_impl.h::AllocEventID indexes eventPool_ by EventToIndex(HardEvent),
so this capacity belongs to each directed, actor-local event bank. The CANN
PTO EventIdCounter<SrcPipe,DstPipe> independently uses the same eight IDs.
These installed headers are evidence, not compiler build dependencies.
"""

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import gcd, lcm

from tests.legacy.analysis.protocols import EventAssignment, LocalMechanism, LocalProtocol
from xdsl_npu.analysis.native import NativeCut, NativeTask
from xdsl_npu.diagnostics import CompileError, ErrorKind, require
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone, supports_engine

LocalBank = tuple[Actor, Engine, Engine]
LOCAL_EVENT_CAPACITY = 8
DEFAULT_PROJECTION_LIMIT = 65_536


@dataclass(slots=True)
class LocalResourceWork:
    indexed_protocols: int = 0
    pair_candidates: int = 0
    width_attempts: int = 0
    pair_attempts: int = 0
    projected_residues: int = 0
    residue_candidates: int = 0
    queue_comparisons: int = 0
    controller_comparisons: int = 0
    lease_witnesses: int = 0
    assigned_ids: int = 0


@dataclass(slots=True)
class _ProjectionBudget:
    limit: int
    used: int = 0


class EventAction(str, Enum):
    SET = "set"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class EventEndpoint:
    """A selected event endpoint, anchored to an actual native operation.

    The Set anchor is a native end; the selected Set captures retirement
    there. A Wait is before its destination admission. Neither anchor alone
    establishes a completed notification. Offsets use the witness's old
    consumer logical generation g, with native stage added exactly once.
    """

    protocol: LocalProtocol
    action: EventAction
    generation_offset: int
    anchor: NativeCut


@dataclass(frozen=True, slots=True)
class QueueWitness:
    """A real Wait gates a native admission in that same hardware queue.

    This says nothing about the controller, another queue, or native ends.
    Equal ranks are accepted only for the very same operation anchor.
    """

    wait: EventEndpoint
    native: NativeCut


@dataclass(frozen=True, slots=True)
class ControllerBridge:
    """A peer's actual Scalar Wait/end gates another engine's source dispatch."""

    wait: EventEndpoint
    blocking_end: NativeCut
    dispatch: NativeCut


@dataclass(frozen=True, slots=True)
class RoundtripWitness:
    """For g=first+k*period, N>g+successor_delta, old Wait -> next Set.

    All offsets are relative to g. The exact chain is consume -> first queue
    admission -> peer source end -> peer Set -> peer Wait -> second queue
    admission or Scalar-gated source dispatch -> next source end -> next Set.
    Set/Wait pairing comes from the same protocol and its exact distance.
    """

    period: int
    residue: int
    first: int
    successor_delta: int
    consume: EventEndpoint
    peer_set: EventEndpoint
    peer_wait: EventEndpoint
    next_set: EventEndpoint
    queues: tuple[QueueWitness, QueueWitness | ControllerBridge]

    @property
    def protocol(self):
        return self.consume.protocol

    @property
    def peer(self):
        return self.peer_set.protocol


@dataclass(frozen=True, slots=True)
class ControllerWitness:
    """The actual Scalar Wait/native gates a strictly later source dispatch.

    For g=first+k*period and N>g+successor_delta, the chain is old Wait ->
    its Scalar native end -> next source dispatch -> its end -> next Set.
    A Scalar source alone does not establish this controller gate.
    """

    period: int
    residue: int
    first: int
    successor_delta: int
    consume: EventEndpoint
    blocking_end: NativeCut
    dispatch: NativeCut
    next_set: EventEndpoint

    @property
    def protocol(self):
        return self.consume.protocol


@dataclass(frozen=True, slots=True)
class EventLease:
    protocol: LocalProtocol
    width: int
    roundtrips: tuple[RoundtripWitness, ...]
    terminal: bool = False
    controllers: tuple[ControllerWitness, ...] = ()

    @property
    def witnesses(self) -> tuple[RoundtripWitness | ControllerWitness, ...]:
        return self.roundtrips + self.controllers

    def ordinal(self, consumer_generation: int) -> int:
        """Exact active notification ordinal, for materialization/testing.

        Runtime endpoint existence is still the protocol activation's guard.
        Width selection never evaluates this function or enumerates N.
        """
        activation = self.protocol.activation
        require(
            type(consumer_generation) is int
            and consumer_generation >= activation.distance
            and consumer_generation % activation.period in activation.residues,
            "event ordinal requires an active consumer generation",
        )
        if self.terminal:
            return 0
        residues = sorted(activation.residues)

        def before(generation):
            cycles, residue = divmod(generation, activation.period)
            return cycles * len(residues) + bisect_left(residues, residue)

        return before(consumer_generation) - before(activation.distance)


@dataclass(frozen=True, slots=True)
class LocalEventPlan:
    """Exclusive assignments with already-established conditional leases."""

    assignments: tuple[EventAssignment, ...]
    leases: tuple[EventLease, ...]


def _endpoint(protocol, action, offset):
    task = protocol.source if action is EventAction.SET else protocol.destination
    milestone = Milestone.NATIVE_END if action is EventAction.SET else Milestone.ADMISSION
    return EventEndpoint(protocol, action, offset, task.cut(milestone, generation_offset=offset))


def _not_before(after: NativeTask, before: NativeTask):
    """Minimum logical after-before generation difference for queue order."""
    tied_other = after.lexical == before.lexical and after.operation is not before.operation
    return before.stage - after.stage + int(after.lexical < before.lexical or tied_other)


def _strictly_after(after: NativeTask, before: NativeTask):
    return before.stage - after.stage + int(after.lexical <= before.lexical)


def _queue(wait, native, work):
    work.queue_comparisons += 1
    require(
        wait.action is EventAction.WAIT
        and wait.anchor.actor is native.actor
        and wait.anchor.engine is native.engine
        and wait.anchor.clock is native.clock
        and native.milestone is Milestone.ADMISSION,
        "event lifetime queue witness must retain its real destination queue",
    )
    comparison = native.rank.compare(wait.anchor.rank)
    require(
        comparison > 0 or (comparison == 0 and native.operation is wait.anchor.operation),
        "event lifetime queue witness reverses actual native admission order",
    )
    return QueueWitness(wait, native)


def _controller_bridge(wait, dispatch, work):
    work.controller_comparisons += 1
    require(
        wait.action is EventAction.WAIT
        and wait.anchor.engine is Engine.S
        and wait.anchor.actor is dispatch.actor
        and wait.anchor.clock is dispatch.clock
        and dispatch.milestone is Milestone.ISSUE,
        "event lifetime controller bridge requires an actual Scalar destination Wait",
    )
    blocking_end = wait.protocol.destination.cut(
        Milestone.NATIVE_END, generation_offset=wait.generation_offset
    )
    require(
        dispatch.rank.compare(blocking_end.rank) > 0,
        "event lifetime source dispatch is not after the actual Scalar gate",
    )
    return ControllerBridge(wait, blocking_end, dispatch)


def _successor_delta(protocol, residue, width):
    residues = sorted(protocol.activation.residues)
    index = bisect_left(residues, residue % protocol.activation.period)
    cycles, following = divmod(index + width, len(residues))
    return residues[following] + cycles * protocol.activation.period - residues[index]


def _roundtrips(protocol, peer, width, work, budget):
    """One return interface proves all phases, including the earliest g.

    With old consumer g and next consumer g+delta, choose peer source g+k.
    Queue or Scalar dispatch inequalities bound k. Existence adds g+k>=0 and
    k+d_peer<=delta; these establish intermediate endpoints for every N, even
    when N is exactly g+delta+1. No boundary is checked by finite simulation.
    """
    work.pair_attempts += 1
    a, b = protocol.activation, peer.activation
    if b.consumer_last or b.source_last:
        return None
    period = lcm(a.period, b.period)
    phases = len(a.residues) * (b.period // gcd(a.period, b.period))
    require(
        budget.used + phases <= budget.limit,
        f"local event pair {protocol.name}/{peer.name} exceeds projection budget "
        f"{budget.limit}; no runtime-N expansion or serialization fallback",
        kind=ErrorKind.ANALYSIS_LIMIT,
        phase="local-event-resources",
    )
    lower_queue = _not_before(peer.source, protocol.destination)
    controller_return = peer.destination.engine is not protocol.source.engine
    require(
        not controller_return or peer.destination.engine is Engine.S,
        "event lifetime return must reach the source queue or actual Scalar controller gate",
    )
    return_order = (
        _strictly_after(protocol.source, peer.destination)
        if controller_return
        else _not_before(protocol.source, peer.destination)
    )
    result = []
    for residue in sorted(a.residues):
        delta = _successor_delta(protocol, residue, width)
        upper_queue = delta - a.distance - return_order
        for phase in range(residue, period, a.period):
            budget.used += 1
            work.projected_residues += 1
            first = phase + max(0, (a.distance - phase + period - 1) // period) * period
            lower = max(lower_queue, -first)
            upper = min(upper_queue - b.distance, delta - b.distance)
            # The next active peer source at or after lower, in this phase.
            work.residue_candidates += len(b.residues)
            offset = min(lower + (r - phase - b.distance - lower) % b.period for r in b.residues)
            if offset > upper:
                return None
            consume = _endpoint(protocol, EventAction.WAIT, 0)
            peer_set = _endpoint(peer, EventAction.SET, offset)
            peer_wait = _endpoint(peer, EventAction.WAIT, offset + b.distance)
            next_set = _endpoint(protocol, EventAction.SET, delta - a.distance)
            queues = (
                _queue(
                    consume,
                    peer.source.cut(Milestone.ADMISSION, generation_offset=offset),
                    work,
                ),
                (
                    _controller_bridge(
                        peer_wait,
                        protocol.source.cut(Milestone.ISSUE, generation_offset=delta - a.distance),
                        work,
                    )
                    if controller_return
                    else _queue(
                        peer_wait,
                        protocol.source.cut(
                            Milestone.ADMISSION, generation_offset=delta - a.distance
                        ),
                        work,
                    )
                ),
            )
            result.append(
                RoundtripWitness(
                    period, phase, first, delta, consume, peer_set, peer_wait, next_set, queues
                )
            )
    return tuple(result)


def _controllers(protocol, width, work, budget):
    if protocol.destination.engine is not Engine.S:
        return None
    activation = protocol.activation
    require(
        budget.used + len(activation.residues) <= budget.limit,
        f"local Scalar event {protocol.name} exceeds projection budget {budget.limit}; "
        "no runtime-N expansion or serialization fallback",
        kind=ErrorKind.ANALYSIS_LIMIT,
        phase="local-event-resources",
    )
    result = []
    blocking_end = protocol.destination.cut(Milestone.NATIVE_END)
    for residue in sorted(activation.residues):
        budget.used += 1
        work.projected_residues += 1
        work.controller_comparisons += 1
        delta = _successor_delta(protocol, residue, width)
        source_offset = delta - activation.distance
        dispatch = protocol.source.cut(Milestone.ISSUE, generation_offset=source_offset)
        # This must follow the actual blocking Scalar native in dispatch
        # order. A same-step source with earlier lexical rank is not gated.
        if dispatch.rank.compare(blocking_end.rank) <= 0:
            return None
        first = (
            residue
            + max(0, (activation.distance - residue + activation.period - 1) // activation.period)
            * activation.period
        )
        result.append(
            ControllerWitness(
                activation.period,
                residue,
                first,
                delta,
                _endpoint(protocol, EventAction.WAIT, 0),
                blocking_end,
                dispatch,
                _endpoint(protocol, EventAction.SET, source_offset),
            )
        )
    return tuple(result)


def _lease(protocol, peers, work, budget):
    if protocol.activation.consumer_last or protocol.activation.source_last:
        return EventLease(protocol, 1, (), terminal=True)
    for width in range(1, LOCAL_EVENT_CAPACITY + 1):
        work.width_attempts += 1
        if controllers := _controllers(protocol, width, work, budget):
            work.lease_witnesses += len(controllers)
            return EventLease(protocol, width, (), controllers=controllers)
        for peer in peers:
            witnesses = _roundtrips(protocol, peer, width, work, budget)
            if witnesses is not None:
                work.lease_witnesses += len(witnesses)
                return EventLease(protocol, width, witnesses)
    raise CompileError(
        f"{protocol.name}: no consume-to-next-Set roundtrip or Scalar controller lease "
        "proved for widths 1..8; "
        "buffer slots and native-end FIFO do not establish binary-event reuse",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-event-resources",
        witness=(
            (
                f"{protocol.source.operation.uid.data}->{protocol.destination.operation.uid.data}, "
                f"distance {protocol.distance}"
            ),
            "reverse/Scalar selected interfaces: " + (", ".join(p.name for p in peers) or "none"),
        ),
    )


def allocate_local_events(
    protocols: Iterable[LocalProtocol],
    *,
    reserved: Mapping[LocalBank, Iterable[int]] | None = None,
    work: LocalResourceWork | None = None,
    projection_limit: int = DEFAULT_PROJECTION_LIMIT,
) -> LocalEventPlan:
    """Prove the smallest supported width, then assign disjoint bank IDs.

    Widths are active notification counts, not generation distances. A proof
    uses one return event interface and actual queue/Scalar controller order,
    or the actual Scalar destination's direct gate. More general paths and inter-channel
    sharing fail explicitly. The caller supplies any
    target ABI reservations (for example the configuration's S->FIX ID0).
    A successful result is conditional on materializing the selected event
    endpoints and separately establishing overall protocol progress.
    """
    protocols = tuple(protocols)
    require(
        all(isinstance(p, LocalProtocol) for p in protocols)
        and len({p.name for p in protocols}) == len(protocols),
        "local resource allocation requires distinct selected protocol identities",
    )
    require(type(projection_limit) is int and projection_limit > 0, "invalid projection budget")
    work = LocalResourceWork() if work is None else work
    budget = _ProjectionBudget(projection_limit)
    used = {}
    for bank, flags in (reserved or {}).items():
        require(
            isinstance(bank, tuple)
            and len(bank) == 3
            and isinstance(bank[0], Actor)
            and supports_engine(bank[0], bank[1])
            and supports_engine(bank[0], bank[2])
            and bank[1] is not bank[2],
            "invalid local event reservation bank",
        )
        flags = tuple(flags)
        require(
            len(set(flags)) == len(flags)
            and all(type(flag) is int and 0 <= flag < LOCAL_EVENT_CAPACITY for flag in flags),
            "local event reservations must be distinct hardware IDs 0..7",
        )
        used[bank] = set(flags)
    events = tuple(p for p in protocols if p.mechanism is LocalMechanism.EVENT)
    interfaces = {}
    for protocol in events:
        key = (
            protocol.source.actor,
            protocol.source.clock,
            protocol.source.engine,
            protocol.destination.engine,
        )
        interfaces.setdefault(key, []).append(protocol)
        work.indexed_protocols += 1
    leases = []
    for protocol in events:
        require(
            protocol.activation.consumer_last == protocol.destination.last
            and protocol.activation.source_last == protocol.source.last,
            f"{protocol.name}: event activation lost an actual terminal endpoint",
        )
        returns = (
            (Engine.S,)
            if protocol.source.engine is Engine.S
            else (protocol.source.engine, Engine.S)
        )
        peers = tuple(
            peer
            for destination in returns
            for peer in interfaces.get(
                (
                    protocol.source.actor,
                    protocol.source.clock,
                    protocol.destination.engine,
                    destination,
                ),
                (),
            )
        )
        work.pair_candidates += len(peers)
        # Prefer the least expensive actual pair; no global period is formed.
        peers = tuple(
            sorted(
                peers,
                key=lambda p: (
                    p.activation.period // gcd(protocol.activation.period, p.activation.period)
                ),
            )
        )
        leases.append(_lease(protocol, peers, work, budget))
    assignments = []
    for lease in leases:
        bank = lease.protocol.bank
        occupied = used.setdefault(bank, set())
        available = [flag for flag in range(LOCAL_EVENT_CAPACITY) if flag not in occupied]
        require(
            lease.width <= len(available),
            f"local bank {tuple(item.value for item in bank)} cannot fit the exclusive "
            f"width-{lease.width} lease for {lease.protocol.name}; capacity is 8 IDs, "
            "cross-channel lease sharing is not supported",
            kind=ErrorKind.UNSUPPORTED,
            phase="local-event-resources",
            witness=(f"already occupied/reserved IDs {sorted(occupied)}",),
        )
        ids = tuple(available[: lease.width])
        occupied.update(ids)
        assignments.append(EventAssignment(lease.protocol, ids))
        work.assigned_ids += len(ids)
    return LocalEventPlan(tuple(assignments), tuple(leases))

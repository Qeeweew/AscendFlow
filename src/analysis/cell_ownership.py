"""Closed, actor-private ownership rounds across sparse child invocations.

A round defines the complete cell before every reader, and issues all those
readers before the next overwrite of that cell. Distinct child invocations
occupy whole controller calls. Empty calls visit no round and preserve its
free token. This is not the native-cell lease rule for forwarding queue ports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind, require
from ..scalar import Expr
from ..target.ascend910b import Engine
from .access_frontier import QueueEndpoint
from .hazards import HazardKind
from .smt import all_of
from .structured_regions import AccessOccurrence, StorageCell, StructuredNeed
from .structured_schedule import Placement

if TYPE_CHECKING:
    from .structured_protocols import BoundPlacement, OrderingWitness


@dataclass(frozen=True, slots=True)
class CellRound:
    loop: Operation
    writer: AccessOccurrence
    readers: tuple[AccessOccurrence, ...]
    period: int
    residue: int | Expr
    sparse: bool


@dataclass(frozen=True, slots=True)
class CellLifetime:
    cell: StorageCell
    rounds: tuple[CellRound, ...]
    requirements: tuple[StructuredNeed, ...]
    ready: tuple[StructuredNeed, ...]
    source_engine: Engine
    destination_engine: Engine


@dataclass(frozen=True, slots=True)
class CreditRound:
    acquire: BoundPlacement
    release: BoundPlacement


@dataclass(frozen=True, slots=True)
class PersistentCellCredit:
    name: str
    lifetime: CellLifetime
    rounds: tuple[CreditRound, ...]


@dataclass(frozen=True, slots=True)
class LocalCreditOrder:
    witnesses: tuple[OrderingWitness, ...]


def infer_cell_lifetimes(builder):
    """Consume body interfaces and actual effects; never search execution paths."""
    groups = {}
    control, dispatch = builder.memory.control, builder.dispatch
    root = builder.memory.module.body.block.first_op
    for loop, cell, body, period, residue, sparse in builder.round_interfaces:
        if (
            not isinstance(cell, StorageCell)
            or cell.private is None
            or cell.streaming
            or cell.conditions
            or not control.invariant_over(cell.origin, root)
            or body.head_reads
            or len(body.first_writes) != 1
            or body.tail_writes != body.first_writes
            or not body.tail_reads
        ):
            continue
        writer = body.first_writes[0]
        points = (writer,) + body.tail_reads
        if any(
            isinstance(p.endpoint, QueueEndpoint)
            or dispatch.site(p).scope.operation is not loop
            or p.participant is not cell.private
            or p.families
            or any(b.ordinal != control.loops[b.loop].ordinal for b in p.bindings)
            for p in points
        ):
            continue

        def branches(point, loop=loop):
            frames = point.frame.ancestors()
            begin = next(i for i, frame in enumerate(frames) if frame.operation is loop) + 1
            return tuple(
                (frame.operation, frame.branch)
                for frame in frames[begin:]
                if frame.branch is not None
            )

        activation = branches(writer)
        if any(branches(p) != activation for p in body.tail_reads) or (
            not activation and (not body.ordered_write or not body.readers_follow_writes)
        ):
            continue
        if any(
            access.mode.reads
            for effect in builder.accesses.effects[writer.operation][cell]
            for access in effect.accesses
        ):
            continue
        w = dispatch.site(writer)
        readers = tuple(dispatch.site(p) for p in body.tail_reads)
        if (
            w.engine is Engine.S
            or any(r.engine is Engine.S or r.engine is w.engine for r in readers)
            or len({r.engine for r in readers}) != 1
            or any(
                not (w.stage, w.lexical) < (r.stage, r.lexical) < (w.stage + period, w.lexical)
                for r in readers
            )
        ):
            continue
        # The cell's own address period bounds the next visit. Other rings
        # never enter this check, and no runtime count is enumerated.
        groups.setdefault(cell, []).append(
            CellRound(loop, writer, body.tail_reads, period, residue, sparse or bool(activation))
        )
    lifetimes = []
    for cell, rounds in groups.items():
        if not any(r.sparse for r in rounds):
            continue
        operations = {r.writer.operation for r in rounds} | {
            p.operation for r in rounds for p in r.readers
        }
        if operations != {op for op, cells in builder.accesses.effects.items() if cell in cells}:
            continue  # A state input or escaping value is not a closed lifetime.
        routes = {
            (dispatch.site(r.readers[0]).engine, dispatch.site(r.writer).engine) for r in rounds
        }
        if len(routes) != 1:
            continue
        ready, replaced = [], []
        for need in builder.required:
            if need.cell != cell:
                continue
            current = any(
                need.kind is HazardKind.RAW
                and need.loop is r.loop
                and need.distance == 0
                and need.source == r.writer
                and need.consumer in r.readers
                for r in rounds
            )
            (ready if current else replaced).append(need)
        if not replaced or any(
            not any(n.source == r.writer and n.consumer == p for n in ready)
            for r in rounds
            for p in r.readers
        ):
            continue
        source, destination = routes.pop()
        lifetimes.append(
            CellLifetime(cell, tuple(rounds), tuple(replaced), tuple(ready), source, destination)
        )
    return tuple(lifetimes)


def bind_cell_credits(domains):
    """Bind actual rounds and establish the local forward-publication premise.

    All ordinary local publications must be issued before their acquires. A new
    private free-token edge also comes from an earlier actual issue (or entry
    seed). Such local queues can finish without a future controller issue; cross
    controller progress remains the existing queue-order rule's obligation.
    """
    from ..passes.notification_binding import _bind_notification, _IssueOrder
    from .event_domains import EventCase
    from .structured_protocols import NotificationKind, OrderingWitness
    from .structured_schedule import ActionGuard

    lifetimes = domains.selection.dependencies.order.cell_lifetimes
    if not lifetimes:
        return (), LocalCreditOrder(())
    order = _IssueOrder(domains.dispatch)
    witnesses = []
    for item in domains.notifications:
        if item.mechanism is NotificationKind.CROSS_EVENT:
            continue
        for need in item.requirements:
            source, consumer = item.sources[0], item.consumers[0]
            if order.compare(source, consumer, need) != -1:
                raise CompileError(
                    "persistent local ownership needs every local publication issued before its acquire",
                    kind=ErrorKind.UNSUPPORTED,
                    phase="persistent-cell-order",
                    location=str(consumer.operation.location),
                )
            witnesses.append(
                OrderingWitness(
                    "strict actor issue order for local credit extension",
                    source,
                    consumer,
                    item.source_engine,
                    item.destination_engine,
                )
            )
    credits = []
    case = EventCase(ActionGuard(), (), (), ())
    for lifetime in lifetimes:
        rounds = []
        for body in lifetime.rounds:
            reader = max(
                body.readers,
                key=lambda p: (domains.dispatch.site(p).stage, domains.dispatch.site(p).lexical),
            )
            need = next(
                n for n in lifetime.ready if n.source == body.writer and n.consumer == reader
            )
            paired = _bind_notification(need, domains.dispatch, order.guards(need))
            acquire = replace(paired.sources[0], placement=Placement.BEFORE)
            release = replace(paired.consumers[0], placement=Placement.AFTER)
            actual = tuple(domains._instance(p, {}, case=case) for p in (acquire, release))
            tests = tuple(p.guard.comparisons for p in actual)
            require(
                all(domains.arithmetic.ordered_definedness(t) is None for t in tests)
                and all(
                    domains.arithmetic.queries.implies(all_of(*tests[i]), tests[1 - i]).proved
                    for i in range(2)
                ),
                "persistent round activation or original guard definedness is not established",
                kind=ErrorKind.UNSUPPORTED,
                phase="persistent-cell-domains",
                location=str(acquire.operation.location),
            )
            rounds.append(CreditRound(acquire, release))
        credits.append(
            PersistentCellCredit(f"persistent_cell_{len(credits)}", lifetime, tuple(rounds))
        )
    return tuple(credits), LocalCreditOrder(tuple(witnesses))

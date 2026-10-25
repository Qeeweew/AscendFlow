"""Closed, actor-private ownership rounds across sparse child invocations.

A round defines the complete cell before every reader, and issues all those
readers before the next overwrite of that cell. Distinct child invocations
occupy whole controller calls. Empty calls visit no round and preserve its
free token. This is not the native-cell lease rule for forwarding queue ports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..diagnostics import CompileError, ErrorKind, require
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
    retirement: StructuredNeed
    readers: tuple[AccessOccurrence, ...]


@dataclass(frozen=True, slots=True)
class CellLifetime:
    cell: StorageCell
    rounds: tuple[CellRound, ...]
    requirements: tuple[StructuredNeed, ...]
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
    groups, sparse_cells, incomplete_cells = {}, set(), set()
    needs, raw_reads = {}, None
    control, dispatch = builder.memory.control, builder.dispatch
    root = builder.memory.module.body.block.first_op
    for loop, cell, body, period, _, sparse in builder.round_interfaces:
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
        last_reader = max(
            zip(body.tail_reads, readers, strict=True),
            key=lambda item: (item[1].stage, item[1].lexical),
        )[0]
        if raw_reads is None:
            raw_reads = {}
            for need in builder.required:
                needs.setdefault(need.cell, []).append(need)
                if need.kind is HazardKind.RAW and need.distance == 0:
                    raw_reads.setdefault((need.cell, need.loop, need.source, need.consumer), need)
        retirement = raw_reads.get((cell, loop, writer, last_reader))
        if retirement is None:
            incomplete_cells.add(cell)
            continue
        # The cell's own address period bounds the next visit. Other rings
        # never enter this check, and no runtime count is enumerated.
        groups.setdefault(cell, []).append(CellRound(retirement, body.tail_reads))
        if sparse or activation:
            sparse_cells.add(cell)
    if not sparse_cells:
        return ()
    # Invert the established input relations once. No cell needs to rescan
    # every effect and requirement or compare each need with every round.
    effects = {}
    for operation, cells in builder.accesses.effects.items():
        for cell in cells:
            if cell in groups:
                effects.setdefault(cell, set()).add(operation)
    lifetimes = []
    for cell, rounds in groups.items():
        if cell not in sparse_cells or cell in incomplete_cells:
            continue
        operations = {r.retirement.source.operation for r in rounds} | {
            p.operation for r in rounds for p in r.readers
        }
        if operations != effects[cell]:
            continue  # A state input or escaping value is not a closed lifetime.
        routes = {
            (dispatch.site(r.readers[0]).engine, dispatch.site(r.retirement.source).engine)
            for r in rounds
        }
        if len(routes) != 1:
            continue
        round_reads = {
            (r.retirement.loop, r.retirement.source, p) for r in rounds for p in r.readers
        }
        replaced, covered = [], set()
        for need in needs.get(cell, ()):
            edge = need.loop, need.source, need.consumer
            current = need.kind is HazardKind.RAW and need.distance == 0 and edge in round_reads
            if current:
                covered.add(edge)
            else:
                replaced.append(need)
        if not replaced or covered != round_reads:
            continue
        source, destination = routes.pop()
        lifetimes.append(CellLifetime(cell, tuple(rounds), tuple(replaced), source, destination))
    return tuple(lifetimes)


def bind_cell_credits(domains):
    """Bind actual rounds and establish the local forward-publication premise.

    All ordinary local publications must be issued before their acquires. A new
    private free-token edge also comes from an earlier actual issue (or entry
    seed). Such local queues can finish without a future controller issue; cross
    controller progress remains the existing queue-order rule's obligation.
    """
    from ..passes.notification_binding import _IssueOrder
    from .event_domains import EventCase
    from .structured_protocols import NotificationKind, OrderingWitness
    from .structured_schedule import ActionGuard

    lifetimes = domains.selection.dependencies.cell_lifetimes
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
                    witness=(
                        f"{item.name}: {need.kind.value} on {need.cell.root.owner.sym_name.data}",
                        (
                            f"{item.source_engine.value}->{item.destination_engine.value}; "
                            f"source={source.placement.value} {source.operation.location}; "
                            f"consumer={consumer.placement.value} {consumer.operation.location}"
                        ),
                        "source coordinates: "
                        + ", ".join(value.text() for _, value in source.coordinates),
                        "consumer coordinates: "
                        + ", ".join(value.text() for _, value in consumer.coordinates),
                    ),
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
    bindings = {item.requirement: item for item in domains.selection.bindings}
    case = EventCase(ActionGuard(), (), (), ())
    for lifetime in lifetimes:
        rounds = []
        for body in lifetime.rounds:
            need = body.retirement
            bound = bindings.get(need)
            require(
                bound is not None and bound.requirement is need and need.consumer in body.readers,
                "persistent retirement lost its original last-reader binding",
            )
            acquire = replace(bound.source, placement=Placement.BEFORE)
            release = replace(bound.consumer, placement=Placement.AFTER)
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

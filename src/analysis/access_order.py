"""Source access order and structured queue-prefix interfaces.

May-conflict order is independent of numerical definitions. A region exports
one interface, including on empty/untaken paths; it never exports alternatives
for a parent's activation history. Exact cells/slot residues are optional
precision. Only closed private packets may substitute dispatch storage order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from xdsl.dialects import arith, scf
from xdsl.ir import Operation

from ..diagnostics import require
from ..indexing import nonnegative, normalize_or_identity
from ..scalar import Expr, expression, literal, substitute
from .access_frontier import AccessEndpoint, LoopResidueSelection, QueueEndpoint, ResiduePosition
from .guard_cover import opposite
from .hazards import HazardKind
from .integer_facts import IntegerFacts
from .region_coordinates import RegionCoordinates
from .structured_access import PartitionedAccesses, partition_accesses
from .structured_memory import NativeMemoryFacts
from .structured_regions import (
    AccessOccurrence,
    AliasRegion,
    CountRelation,
    DependencyWork,
    LoopBinding,
    LoopCountGuard,
    LoopDomain,
    StorageCell,
    StructuredNeed,
)
from .structured_schedule import Comparison, Placement

if TYPE_CHECKING:
    from .storage_packets import PacketPartition
    from .structured_admission import StructuredAdmission


def unique(items):
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True, slots=True)
class AccessTransfer:
    """Order requirements and pending queue prefixes, never a numerical KILL.

    A write port can transmit an old prefix without writing a byte. All reader
    queues survive until an ordering successor has acquired their retirement.
    ``readers_follow_writes`` only authorizes redundant WAW elimination when
    the actual RAW/retirement chain was constructed by sequence composition.
    """

    head_reads: tuple[AccessOccurrence, ...] = ()
    first_writes: tuple[AccessOccurrence, ...] = ()
    tail_writes: tuple[AccessOccurrence, ...] = ()
    tail_reads: tuple[AccessOccurrence, ...] = ()
    readers_follow_writes: bool = True
    # An unconditional order successor, not a must-definition of bytes.
    total_write: bool = True

    @property
    def ordered_write(self):
        return bool(self.first_writes) and self.total_write

    def join(self, other):
        return AccessTransfer(
            self.head_reads if self.ordered_write else unique(self.head_reads + other.head_reads),
            self.first_writes
            if self.ordered_write
            else unique(self.first_writes + other.first_writes),
            other.tail_writes
            if other.ordered_write
            else unique(self.tail_writes + other.tail_writes),
            other.tail_reads if other.ordered_write else unique(self.tail_reads + other.tail_reads),
            other.readers_follow_writes
            if other.ordered_write
            else self.readers_follow_writes
            and other.readers_follow_writes
            and not other.first_writes,
            self.ordered_write or other.ordered_write,
        )

    def map(self, transform):
        return AccessTransfer(
            *(
                tuple(transform(p) for p in points)
                for points in (
                    self.head_reads,
                    self.first_writes,
                    self.tail_writes,
                    self.tail_reads,
                )
            ),
            self.readers_follow_writes,
            self.total_write,
        )


@dataclass(frozen=True)
class AccessOrder:
    memory: NativeMemoryFacts
    admission: StructuredAdmission | None
    accesses: PartitionedAccesses
    summaries: Mapping[Operation, Mapping[StorageCell | AliasRegion, AccessTransfer]]
    needs: tuple[StructuredNeed, ...]
    required: tuple[StructuredNeed, ...]
    packet_partitions: tuple[PacketPartition, ...]
    work: DependencyWork


def build_access_order(memory, dispatch, *, admission=None):
    require(dispatch.memory is memory, "access order needs the owned dispatch")
    if admission is not None:
        require(admission.memory is memory, "access admission belongs to another snapshot")
    return _Builder(memory, dispatch, admission).build()


class _Builder:
    def __init__(self, memory, dispatch, admission):
        self.memory, self.dispatch, self.admission = memory, dispatch, admission
        self.accesses = partition_accesses(memory, admission)
        self.work = self.accesses.work
        self.needs, self.required = [], []
        self.summaries, self.partitions = {}, []
        self.coordinates = RegionCoordinates(memory, self.accesses.patterns)
        self.facts = memory.control.environment.facts
        self.arithmetic = IntegerFacts(self.facts)
        self.iteration_writes = set()
        for cells in self.accesses.effects.values():
            for cell, effects in cells.items():
                for effect in effects:
                    frames = effect.template.frame.ancestors()
                    loops = [i for i, f in enumerate(frames) if f.loop is not None]
                    if (
                        loops
                        and not any(f.branch is not None for f in frames[loops[-1] + 1 :])
                        and any(a.mode.writes for a in effect.accesses)
                    ):
                        self.iteration_writes.add((cell, frames[loops[-1]].operation))

    def _activation(self, point):
        """Bind original branch predicates once, where an order edge is built."""
        substitutions = {}
        for binding in point.bindings:
            value = binding.ordinal
            if not isinstance(value, Expr):
                # A narrow selected port is admitted only for loop-invariant
                # predicates. Dynamic boundaries use the common envelope.
                continue
            substitutions[self.memory.control.loops[binding.loop].ordinal] = value
        tests = []
        for frame in point.frame.ancestors():
            if frame.branch is None:
                continue
            compare = frame.operation.cond.owner
            require(isinstance(compare, arith.CmpiOp), "access order needs an integer branch")
            predicates = ("==", "!=", "<", "<=", ">", ">=")
            require(compare.predicate.value.data < len(predicates), "unsupported access predicate")
            test = Comparison(
                substitute(self.memory.control.environment.expression(compare.lhs), substitutions),
                predicates[compare.predicate.value.data],
                substitute(self.memory.control.environment.expression(compare.rhs), substitutions),
            )
            tests.append(opposite(test) if frame.branch else test)
        return tuple(tests)

    def _emit(self, cell, kind, sources, consumers, scope, *, required=True, guards=(), **clock):
        guards = self.coordinates.guards(
            cell.conditions if isinstance(cell, StorageCell) else (), guards
        )
        if guards is None:
            return
        for source in sources:
            for consumer in consumers:
                if source == consumer:
                    continue
                need = self.coordinates.need(cell, kind, source, consumer, scope, guards, **clock)
                activation = unique(self._activation(source) + self._activation(consumer))
                if activation:
                    from .structured_regions import LoopOrdinalGuard

                    predicates = tuple(
                        Comparison(
                            self.memory.control.loops[d.loop].ordinal,
                            ">=",
                            literal(max(d.first, d.distance)),
                        )
                        for d in need.domains
                    ) + tuple(
                        Comparison(self.memory.control.loops[d.loop].ordinal, "<", d.count)
                        for d in need.domains
                    )
                    normalized = tuple(
                        Comparison(
                            self.arithmetic.normalize(expression("sub", t.lhs, t.rhs)),
                            t.predicate,
                            literal(0),
                        )
                        for t in predicates + activation
                    )
                    if any(
                        self.arithmetic.prove(opposite(t), normalized[:i] + normalized[i + 1 :])
                        for i, t in enumerate(normalized)
                    ):
                        continue
                    need = replace(
                        need,
                        guards=need.guards + tuple(LoopOrdinalGuard(scope, t) for t in activation),
                    )
                self.needs.append(need)
                if required:
                    self.required.append(need)
                self.work.boundary_visits += 1

    def _connect(self, cell, left, right, scope, *, retirement=True, **clock):
        self._emit(cell, HazardKind.RAW, left.tail_writes, right.head_reads, scope, **clock)
        if retirement:
            self._emit(
                cell,
                HazardKind.WAW,
                left.tail_writes,
                right.first_writes,
                scope,
                required=not left.tail_reads or not left.readers_follow_writes,
                **clock,
            )
            self._emit(cell, HazardKind.WAR, left.tail_reads, right.first_writes, scope, **clock)

    def _sequence(self, left, right, scope, *, partitions=None):
        result = dict(left)
        for cell, after in right.items():
            before = result.get(cell, AccessTransfer())
            partition = None if partitions is None else partitions.get(cell)
            retirement = partition is None or all(
                partition.key_for(a) == partition.key_for(b)
                for a in before.tail_writes + before.tail_reads
                for b in after.first_writes
            )
            self._connect(cell, before, after, scope, retirement=retirement)
            result[cell] = before.join(after)
        return result

    def _native(self, operation):
        result = {}
        for cell, effects in self.accesses.effects.get(operation, {}).items():
            transfer = AccessTransfer()
            for effect in effects:
                occurrence = AccessOccurrence(
                    AccessEndpoint.from_template(effect.template, effect.participant),
                    tuple(
                        LoopBinding(l.operation, l.ordinal) for l in effect.template.frame.loops()
                    ),
                )
                reads = any(a.mode.reads for a in effect.accesses)
                # A native that always resets has no old-state read. Other
                # conditional reads are conservative order requirements; their
                # byte definedness remains admission's separate responsibility.
                reads = reads and any(
                    a.mode.reads
                    and (
                        a.read_condition is None
                        or not (
                            a.read_condition.read.lhs.constant is not None
                            and a.read_condition.read.rhs.constant is not None
                            and not a.read_condition.read.evaluate({})
                        )
                    )
                    for a in effect.accesses
                )
                writes = any(a.mode.writes for a in effect.accesses)
                reader, writer = (occurrence,) if reads else (), (occurrence,) if writes else ()
                other = AccessTransfer(reader, writer, writer, () if writes else reader)
                self._connect(cell, transfer, other, operation)
                transfer = transfer.join(other)
            result[cell] = transfer
        return result

    def _engine(self, occurrence):
        return self.dispatch.site(occurrence).engine

    def _envelope(self, operation, cells):
        """Capture/acquire common real queue prefixes, including identity paths.

        Parent obligations are connected to these input queues. Exit Sets
        capture both their acquisitions and actual body issues. No extra
        completion or hidden body drain is inserted here.
        """
        frame = self.memory.control.frames[operation]
        if operation in self.memory.control.loops:
            frame = frame.parent
        assert frame is not None
        bindings = tuple(LoopBinding(l.operation, l.ordinal) for l in frame.loops())

        def ports(points, placement):
            return unique(
                AccessOccurrence(
                    QueueEndpoint(operation, p.participant, frame, self._engine(p), placement),
                    bindings,
                )
                for p in points
            )

        result = {}
        for cell, transfer in cells.items():
            readers = unique(transfer.head_reads + transfer.tail_reads)
            writers = unique(transfer.first_writes + transfer.tail_writes)
            result[cell] = AccessTransfer(
                ports(readers, Placement.ENTRY),
                ports(writers, Placement.ENTRY),
                ports(writers, Placement.EXIT),
                ports(readers, Placement.EXIT),
                False,
            )
            self.work.region_ports += sum(
                len(p)
                for p in (
                    result[cell].head_reads,
                    result[cell].first_writes,
                    result[cell].tail_writes,
                    result[cell].tail_reads,
                )
            )
        return result

    def _map(self, transfer, loop, ordinal, *, family=None):
        return transfer.map(lambda p: self.coordinates.bind(p, loop, ordinal, family=family))

    def _entry_read(self, cell, point, loop, residue):
        """Local native reset precision, without selecting a numerical writer."""
        if isinstance(point.endpoint, QueueEndpoint):
            return True
        bindings = {
            self.memory.control.loops[loop].ordinal: literal(residue)
            if type(residue) is int
            else residue
        }
        for binding in point.bindings:
            if binding.loop is loop:
                continue
            value = binding.ordinal
            if isinstance(value, LoopResidueSelection):
                if value.position is not ResiduePosition.FIRST:
                    return True
                value = literal(value.residue) if type(value.residue) is int else value.residue
            bindings[self.memory.control.loops[binding.loop].ordinal] = value
        effects = self.accesses.effects[point.operation][cell]
        for effect in effects:
            if effect.participant is not point.participant:
                continue
            for access in effect.accesses:
                if not access.mode.reads:
                    continue
                condition = access.read_condition
                if condition is None:
                    return True
                test = condition.read
                lhs, rhs = (
                    normalize_or_identity(substitute(v, bindings), self.facts)
                    for v in (test.lhs, test.rhs)
                )
                if (
                    lhs.constant is None
                    or rhs.constant is None
                    or type(test)(lhs, test.predicate, rhs).evaluate({})
                ):
                    return True
        return False

    def _loop(self, operation, body, partitions):
        logical = self.memory.control.loops[operation]
        result = {}
        for cell, transfer in body.items():
            period, residue = self.coordinates.pattern(cell, operation)
            streaming = isinstance(cell, StorageCell) and operation in cell.streaming
            if transfer.first_writes and not streaming and cell not in partitions:
                previous = self._map(
                    transfer, operation, expression("sub", logical.ordinal, literal(period))
                )
                self._connect(
                    cell,
                    previous,
                    transfer,
                    operation,
                    loop=operation,
                    distance=period,
                    period=period,
                    residue=residue,
                    guards=(
                        LoopCountGuard(
                            operation,
                            logical.count,
                            CountRelation.MORE,
                            expression(
                                "add",
                                literal(period),
                                literal(residue) if type(residue) is int else residue,
                            ),
                        ),
                    ),
                )
            transfer = replace(
                transfer,
                head_reads=tuple(
                    p for p in transfer.head_reads if self._entry_read(cell, p, operation, residue)
                ),
            )
            # A guaranteed populated residue permits a narrow native boundary.
            # Otherwise one actual invocation envelope forwards old state and
            # reader prefixes. It requires no activation history or child LAST.
            threshold = literal(residue) if type(residue) is int else residue
            populated = nonnegative(
                expression("sub", logical.count, expression("add", threshold, literal(1))),
                self.facts,
            )
            if not populated and isinstance(cell, StorageCell):
                # Compound-slot partitioning already separates uniform short
                # and long extents. Keep those premises when selecting a
                # boundary; forgetting them needlessly drains every queue.
                root = self.memory.module.body.block.first_op
                uniform = tuple(
                    guard
                    for guard in cell.conditions
                    if isinstance(guard, LoopCountGuard)
                    and self.memory.control.invariant_over(guard.count, root)
                    and (
                        type(guard.threshold) is int
                        or self.memory.control.invariant_over(guard.threshold, root)
                    )
                )
                populated = (
                    self.coordinates.guards(
                        uniform,
                        (
                            LoopCountGuard(
                                operation, logical.count, CountRelation.AT_MOST, threshold
                            ),
                        ),
                    )
                    is None
                )
            stable = all(
                self.memory.control.invariant_over(value, operation)
                for point in transfer.head_reads
                + transfer.tail_reads
                + transfer.first_writes
                + transfer.tail_writes
                for test in self._activation(point)
                for value in (test.lhs, test.rhs)
            )
            if (
                not populated
                or not stable
                or (transfer.first_writes and not transfer.ordered_write)
            ):
                result.update(self._envelope(operation, {cell: transfer}))
            elif not transfer.first_writes or streaming:
                family = LoopDomain(operation, logical.count, period, residue)
                result[cell] = self._map(transfer, operation, logical.ordinal, family=family)
            else:
                head, tail = (
                    self._map(
                        transfer,
                        operation,
                        LoopResidueSelection(operation, period, residue, position),
                    )
                    for position in (ResiduePosition.FIRST, ResiduePosition.LAST)
                )
                result[cell] = AccessTransfer(
                    head.head_reads,
                    head.first_writes,
                    tail.tail_writes,
                    tail.tail_reads,
                    transfer.readers_follow_writes,
                )
        return result

    def _word(self, partition):
        from .storage_word import StoragePacketPosition, build_storage_word
        from .structured_regions import LoopOrdinalGuard

        loop = partition.loop
        logical = self.memory.control.loops[loop]
        packets = {p.key: p for p in partition.packets}
        word = build_storage_word(
            tuple(StoragePacketPosition(p.key, p.offset, p.lexical) for p in packets.values()),
            logical.count,
            logical.ordinal,
        )
        for edge in word.adjacencies:
            source, consumer = packets[edge.source.key], packets[edge.consumer.key]
            delta = normalize_or_identity(
                expression("sub", edge.consumer.ordinal, edge.source.ordinal), self.facts
            ).constant
            require(delta is not None, "packet adjacency lost its generation distance")
            from .structured_schedule import Comparison

            activation = edge.guards + (
                (Comparison(logical.ordinal, "==", edge.consumer.ordinal),)
                if edge.limit is None
                else ()
            )
            # The finite storage word supplies its own actual membership.
            before = self._map(
                source.summary, loop, expression("sub", logical.ordinal, literal(delta))
            )
            after = self._map(consumer.summary, loop, logical.ordinal)
            self._connect(
                partition.cell,
                before,
                after,
                loop,
                loop=loop,
                distance=delta,
                first=edge.first,
                limit=edge.limit,
                guards=tuple(LoopOrdinalGuard(loop, test) for test in activation),
            )

    def _node(self, operation):
        self.work.scope_visits += 1
        if operation in self.accesses.effects:
            cells = self._native(operation)
        elif (
            operation in self.memory.control.loops
            and normalize_or_identity(
                self.memory.control.loops[operation].count, self.facts
            ).constant
            == 0
        ):
            cells = {}
        elif isinstance(operation, scf.IfOp):
            union = {}
            for region in operation.regions:
                branch = {}
                for block in region.blocks:
                    for child in block.ops:
                        branch = self._sequence(branch, self._node(child), operation)
                for cell, transfer in branch.items():
                    previous = union.get(cell, AccessTransfer())
                    union[cell] = AccessTransfer(
                        *(
                            unique(a + b)
                            for a, b in zip(
                                (
                                    previous.head_reads,
                                    previous.first_writes,
                                    previous.tail_writes,
                                    previous.tail_reads,
                                ),
                                (
                                    transfer.head_reads,
                                    transfer.first_writes,
                                    transfer.tail_writes,
                                    transfer.tail_reads,
                                ),
                            )
                        ),
                        False,
                        False,
                    )
            cells = union
            loops = self.memory.control.frames[operation].loops()
            if loops:
                loop = loops[-1].operation
                for cell, transfer in union.items():
                    varying = any(
                        not self.memory.control.invariant_over(value, loop)
                        for p in transfer.head_reads
                        + transfer.tail_reads
                        + transfer.first_writes
                        + transfer.tail_writes
                        for test in self._activation(p)
                        for value in (test.lhs, test.rhs)
                    )
                    if varying and (
                        transfer.head_reads
                        or transfer.tail_reads
                        or transfer.first_writes
                        and (cell, loop) not in self.iteration_writes
                    ):
                        # A conditional reader can depend on an S value
                        # defined after an earlier producer. The common
                        # branch ports forward/acquire real queue prefixes
                        # even when the read is skipped; no earlier Set is
                        # guarded by that not-yet-defined value.
                        cells.update(self._envelope(operation, {cell: transfer}))
        else:
            children = [
                child
                for region in operation.regions
                for block in region.blocks
                for child in block.ops
            ]
            pieces = [(child, self._node(child)) for child in children]
            partitions = {}
            if operation in self.memory.control.loops:
                from .storage_packets import certify_packets

                for cell in dict.fromkeys(c for _, piece in pieces for c in piece):
                    partition = certify_packets(self, operation, cell)
                    if partition is not None:
                        partitions[cell] = partition
                        self.partitions.append(partition)
            cells = {}
            for _, piece in pieces:
                cells = self._sequence(cells, piece, operation, partitions=partitions)
            if operation in self.memory.control.loops:
                for partition in partitions.values():
                    self._word(partition)
                cells = self._loop(operation, cells, partitions)
        self.summaries[operation] = MappingProxyType(cells)
        return cells

    def build(self):
        self._node(self.memory.module)
        self.work.needs = len(self.needs)
        self.work.loop_carries = sum(
            n.loop is not None and n.distance not in (None, 0) for n in self.needs
        )
        return AccessOrder(
            self.memory,
            self.admission,
            self.accesses,
            MappingProxyType(self.summaries),
            tuple(self.needs),
            tuple(self.required),
            tuple(self.partitions),
            self.work,
        )

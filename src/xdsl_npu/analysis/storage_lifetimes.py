"""Physical overwrite and retirement relations for established value lifetimes.

Source interface splices retain state/observable ordering. For a proved closed
private packet partition, its whole pipeline cell word is instead composed in
actual dispatch order. This phase never changes a read's logical definition and
never constructs source WARs merely to delete inconvenient events afterward.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.structured import PipelineLoopOp
from ..scalar import expression, literal
from .hazards import HazardKind
from .region_coordinates import RegionCoordinates, normalize_coordinate
from .storage_packets import certify_packets
from .storage_word import StoragePacketPosition, build_storage_word
from .structured_regions import (
    AccessOccurrence,
    BranchGuard,
    DependencyIssue,
    ExitObligation,
    LoopOrdinalGuard,
    RegionConnection,
    StructuredNeed,
)
from .structured_schedule import Comparison


@dataclass(frozen=True)
class StorageLifetimes:
    needs: tuple[StructuredNeed, ...]
    required: tuple[StructuredNeed, ...]
    exits: tuple[ExitObligation, ...]
    unresolved: tuple[DependencyIssue, ...]
    packet_partitions: tuple

    @property
    def closed(self):
        return not self.unresolved

    def require_closed(self):
        if self.unresolved:
            issue = self.unresolved[0]
            raise CompileError(
                issue.message,
                kind=ErrorKind.UNSUPPORTED,
                phase="storage-lifetimes",
                location=str(issue.operation.location),
                witness=(f"classification={issue.kind.value}",),
            )


def build_storage_lifetimes(values, dispatch):
    require(dispatch.memory is values.memory, "storage lifetimes need the owned dispatch")
    return _Storage(values, dispatch).build()


class _Storage:
    def __init__(self, values, dispatch):
        self.values, self.dispatch = values, dispatch
        self.needs, self.issues, self.covered = [], [], set()
        self.coordinates = RegionCoordinates(values.memory, values.accesses.patterns, self.issue)
        self.raw_ids = {id(read) for read in values.reads}
        self.raw_keys = {
            (read.cell, read.source, read.consumer, read.guards): read
            for read in values.reads
            if isinstance(read.source, AccessOccurrence)
        }

    def issue(self, kind, operation, message, cell=None):
        self.issues.append(DependencyIssue(kind, operation, message, cell))

    def _available(self, read, guards):
        # A mapped boundary read denotes an instance of the original RAW.
        # Its branch/effect conditions must still hold on this splice.
        origin = read.instance_of or read
        return (
            id(origin) in self.raw_ids
            or (read.cell, read.source, read.consumer, read.guards) in self.raw_keys
        ) and all(
            not isinstance(g, (BranchGuard, LoopOrdinalGuard)) or g in guards for g in origin.guards
        )

    def _connect(self, connection):
        left, right = connection.left, connection.right
        if right.first_write is None:
            return
        clock = {
            name: getattr(connection, name)
            for name in ("loop", "distance", "period", "residue", "first", "limit")
        }

        def emit(kind, source):
            if not isinstance(source, AccessOccurrence) or source == right.first_write:
                return None
            need = self.coordinates.need(
                left.cell,
                kind,
                source,
                right.first_write,
                connection.scope,
                connection.guards,
                **clock,
            )
            self.needs.append(need)
            return need

        waw = emit(HazardKind.WAW, left.writer)
        for read in left.tail_reads:
            war = emit(HazardKind.WAR, read.consumer)
            if (
                waw is not None
                and war is not None
                and read.source == left.writer
                and self._available(read, connection.guards)
            ):
                # Exact value edge followed by reader retirement already
                # orders this writer before this next overwrite.
                self.covered.add(id(waw))

    def _map(self, summary, loop, ordinal, guards):
        return self.coordinates.map_summary(
            summary, lambda value: self.coordinates.bind(value, loop, ordinal), loop, guards
        )

    def _word(self, partition):
        loop = partition.loop
        logical = self.values.memory.control.loops[loop]
        packets = {packet.key: packet for packet in partition.packets}
        word = build_storage_word(
            tuple(StoragePacketPosition(p.key, p.offset, p.lexical) for p in packets.values()),
            logical.count,
            logical.ordinal,
        )
        for edge in word.adjacencies:
            source, consumer = packets[edge.source.key], packets[edge.consumer.key]
            delta = normalize_coordinate(
                expression("sub", edge.consumer.ordinal, edge.source.ordinal),
                self.coordinates.facts,
            ).constant
            require(delta is not None, "packet adjacency lost its constant generation distance")
            # Boundaries and steady members use the same actual loop clock.
            # A singleton is a constrained member, not a second finite clock
            # detached from this stream's generation and event identity.
            activation = edge.guards + (
                (Comparison(logical.ordinal, "==", edge.consumer.ordinal),)
                if edge.limit is None
                else ()
            )
            guards = self.coordinates.guards(
                source.guards,
                consumer.guards,
                tuple(LoopOrdinalGuard(loop, test) for test in activation),
            )
            if guards is None:
                continue
            before = self._map(
                source.summary, loop, expression("sub", logical.ordinal, literal(delta)), guards
            )
            after = self._map(consumer.summary, loop, logical.ordinal, guards)
            self._connect(
                RegionConnection(
                    loop,
                    before,
                    after,
                    guards,
                    loop=loop,
                    distance=delta,
                    first=edge.first,
                    limit=edge.limit,
                )
            )
        return word

    def build(self):
        partitions = {}
        if self.values.closed:
            for loop in self.values.memory.control.loops:
                if not isinstance(loop, PipelineLoopOp):
                    continue
                for cell in self.values.summaries[loop].cells:
                    partition = certify_packets(
                        self.values, self.dispatch, self.coordinates, loop, cell
                    )
                    if partition is not None:
                        partitions[loop, cell] = partition
        for connection in self.values.connections:
            partition = partitions.get((connection.scope, connection.left.cell))
            if partition is not None:
                if connection.loop is partition.loop:
                    continue
                a, b = connection.left.last_write, connection.right.first_write
                if a is not None and b is not None and partition.key_for(a) != partition.key_for(b):
                    continue
            self._connect(connection)
        exits = list(self.values.exits)
        for partition in partitions.values():
            word = self._word(partition)
            if not partition.boundary_changed:
                continue
            exits = [exit for exit in exits if exit.cell != partition.cell]
            packets = {p.key: p for p in partition.packets}
            for end in word.last:
                packet = packets[end.point.key]
                guards = self.coordinates.guards(
                    packet.guards, tuple(LoopOrdinalGuard(partition.loop, g) for g in end.guards)
                )
                if guards is None:
                    continue
                summary = self._map(packet.summary, partition.loop, end.point.ordinal, guards)
                exits.append(ExitObligation(partition.cell, summary.last_write, True, guards))
                exits.extend(
                    ExitObligation(partition.cell, read.consumer, False, guards)
                    for read in summary.tail_reads
                )
        return StorageLifetimes(
            tuple(self.needs),
            tuple(n for n in self.needs if id(n) not in self.covered),
            tuple(exits),
            tuple(self.issues),
            tuple(partitions.values()),
        )

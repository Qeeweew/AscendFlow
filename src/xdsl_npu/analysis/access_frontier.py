"""Finite, unconditional per-cell access transfers and direct dependencies.

The caller supplies an exact cell and one unconditional logical access sequence
under one activation partition. Input order is source logical order, never
stage issue_order or physical dispatch order. This module neither partitions
addresses nor closes a loop/branch. Endpoints keep actual tasks, participants
and control frames; their equality does not bind distinct loop iterations.
Dependencies are outstanding native-end -> admission obligations, not ready
facts. No same-engine end FIFO or child-exit completion is assumed.

Transfers retain boundary data only. One owned collector appends internal
needs/carries as boundaries are spliced; composing a parent never copies its
children's internal needs. Reader tuples remain exact. Their processing cost
is proportional to the boundary entries actually composed, plus output needs.
There is no runtime-N scan, origin closure, or execution-history graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.dialects import scf
from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import BufferOp, TaskOp
from ..dialects.structured import PipelineLoopOp
from ..scalar import Expr
from ..target.ascend910b import Actor, Milestone
from .hazards import HazardKind
from .native import AccessMode
from .structured_control import ControlFrame

if TYPE_CHECKING:
    from .structured_memory import MemoryTemplate


@dataclass(frozen=True, slots=True)
class Cell:
    root: SSAValue
    start: int
    stop: int
    participant: Actor | None = None

    def __post_init__(self):
        require(
            isinstance(self.root, SSAValue)
            and isinstance(self.root.owner, BufferOp)
            and self.root is self.root.owner.ref,
            "access cell needs its actual allocation root SSA",
        )
        require(
            type(self.start) is int and type(self.stop) is int and 0 <= self.start < self.stop,
            "access cell needs a nonempty exact byte interval",
        )
        bank = self.root.type.bank.data
        require(
            self.participant is None if bank == "GM" else isinstance(self.participant, Actor),
            "GM cells share their root; local cells need a private participant",
        )

    @property
    def writer_port(self):
        return CellPort(self, PortKind.WRITER)

    @property
    def readers_port(self):
        return CellPort(self, PortKind.READERS)


@dataclass(frozen=True, slots=True)
class AccessEndpoint:
    """An unbound native identity, admitted once per cell in one composition.

    Reusing this identity across loop generations needs an explicit occurrence
    binding, which this finite component does not yet implement.
    """

    operation: TaskOp
    participant: Actor
    frame: ControlFrame

    def __post_init__(self):
        require(
            isinstance(self.operation, TaskOp)
            and isinstance(self.participant, Actor)
            and isinstance(self.frame, ControlFrame)
            and self.participant in self.frame.participants,
            "access endpoint needs an actual TaskOp, participant and owned control frame",
        )

    @classmethod
    def from_template(cls, template: MemoryTemplate, participant: Actor):
        return cls(template.operation, participant, template.frame)


@dataclass(frozen=True, slots=True)
class AccessCut:
    endpoint: AccessEndpoint
    milestone: Milestone


class PortKind(str, Enum):
    WRITER = "incoming-writer"
    READERS = "incoming-readers"


@dataclass(frozen=True, slots=True)
class CellPort:
    cell: Cell
    kind: PortKind


@dataclass(frozen=True, slots=True)
class MemoryNeed:
    cell: Cell
    kind: HazardKind
    source: AccessEndpoint | CellPort
    consumer: AccessEndpoint

    @property
    def source_cut(self):
        return (
            AccessCut(self.source, Milestone.NATIVE_END)
            if isinstance(self.source, AccessEndpoint)
            else None
        )

    @property
    def consumer_cut(self):
        return AccessCut(self.consumer, Milestone.ADMISSION)


@dataclass(frozen=True, slots=True)
class PendingRead:
    reader: AccessEndpoint
    writer: AccessEndpoint | CellPort
    raw: MemoryNeed | None


@dataclass(frozen=True, slots=True)
class ReadCarry:
    """A directly paired RAW and WAR cover this WAW; None WAR means RMW.

    A writer port leaves a boundary template, still requiring an input binding.
    A concrete carry is recorded while its pending reader is discharged, never
    reconstructed by joining WAWs to all historical readers afterwards.
    """

    waw: MemoryNeed
    raw: MemoryNeed
    war: MemoryNeed | None


@dataclass(frozen=True, slots=True)
class CellTransfer:
    cell: Cell
    head_reads: tuple[PendingRead, ...]
    first_write: AccessEndpoint | None
    last_write: AccessEndpoint | None
    tail_reads: tuple[PendingRead, ...]
    entry_carries: tuple[ReadCarry, ...] = ()
    _owner: AccessCollector | None = field(default=None, repr=False, compare=False)
    _native: AccessEndpoint | None = field(default=None, repr=False, compare=False)
    _token: object = field(default_factory=object, repr=False, compare=False)

    @classmethod
    def identity(cls, cell):
        return cls(cell, (), None, None, ())

    @property
    def writer(self):
        return self.cell.writer_port if self.last_write is None else self.last_write

    @property
    def forwards_writer(self):
        return self.last_write is None

    @property
    def forwards_readers(self):
        return self.first_write is None

    @property
    def entry_needs(self):
        needs = tuple(read.raw for read in self.head_reads if read.raw is not None)
        if self.first_write is None:
            return needs
        return needs + (
            MemoryNeed(self.cell, HazardKind.WAW, self.cell.writer_port, self.first_write),
            MemoryNeed(self.cell, HazardKind.WAR, self.cell.readers_port, self.first_write),
        )


@dataclass(slots=True)
class FrontierWork:
    compositions: int = 0
    reader_visits: int = 0
    needs_appended: int = 0
    carries_appended: int = 0


@dataclass(frozen=True, slots=True)
class AccessDependencies:
    needs: tuple[MemoryNeed, ...]
    read_carries: tuple[ReadCarry, ...]


class AccessCollector:
    """Append-only owned internal obligations, frozen once after composition.

    Transient admission sets reject repeated unbound native occurrences and
    reused composed children. They never enter the boundary transfer or serve
    as dependence/completion evidence.
    """

    def __init__(self, *, work: FrontierWork | None = None):
        self.work = FrontierWork() if work is None else work
        self._needs: list[MemoryNeed] = []
        self._carries: list[ReadCarry] = []
        self._frozen: AccessDependencies | None = None
        self._seen: set[tuple[Cell, AccessEndpoint]] = set()
        self._available: set[object] = set()

    def _open(self):
        require(self._frozen is None, "access collector has already been frozen")

    def _need(self, cell, kind, source, consumer):
        require(
            source != consumer,
            "repeated native endpoint needs an explicit occurrence binding",
        )
        need = MemoryNeed(cell, kind, source, consumer)
        if isinstance(source, AccessEndpoint):
            self._needs.append(need)
            self.work.needs_appended += 1
        return need

    def _admit(self, left, right):
        occurrences, children = set(), set()
        for transfer in (left, right):
            if transfer._owner is self:
                require(
                    transfer._token in self._available and transfer._token not in children,
                    "cell transfer has already been spliced into a parent",
                )
                children.add(transfer._token)
            elif transfer._native is not None:
                occurrence = (transfer.cell, transfer._native)
                require(
                    occurrence not in self._seen and occurrence not in occurrences,
                    "repeated native endpoint needs an explicit occurrence binding; "
                    "union one instruction's operands in native_cell",
                )
                occurrences.add(occurrence)
            else:
                require(
                    not transfer.head_reads
                    and transfer.first_write is None
                    and transfer.last_write is None
                    and not transfer.tail_reads
                    and not transfer.entry_carries,
                    "unowned cell transfers must come from native_cell or identity",
                )
        self._seen.update(occurrences)
        self._available.difference_update(children)

    def _carry(self, carry):
        if isinstance(carry.waw.source, AccessEndpoint):
            self._carries.append(carry)
            self.work.carries_appended += 1

    def freeze(self):
        if self._frozen is None:
            self._frozen = AccessDependencies(tuple(self._needs), tuple(self._carries))
            self._seen.clear()
            self._available.clear()
        return self._frozen


def native_cell(cell: Cell, endpoint: AccessEndpoint, mode: AccessMode) -> CellTransfer:
    """One logical occurrence's unioned operands accessing this exact cell.

    READ_WRITE models one instruction atomically, including its internal
    read-before-write. A repeated endpoint cannot represent another loop
    generation until an explicit occurrence binding is implemented.
    """
    require(isinstance(mode, AccessMode), "cell access needs a typed native access mode")
    require(
        cell.participant is None or cell.participant is endpoint.participant,
        "native access cannot use another participant's private cell",
    )
    reads = ()
    if mode.reads:
        raw = MemoryNeed(cell, HazardKind.RAW, cell.writer_port, endpoint)
        reads = (PendingRead(endpoint, cell.writer_port, raw),)
    first = endpoint if mode.writes else None
    carries = ()
    if mode.reads and mode.writes:
        waw = MemoryNeed(cell, HazardKind.WAW, cell.writer_port, endpoint)
        carries = (ReadCarry(waw, reads[0].raw, None),)
    return CellTransfer(
        cell, reads, first, first, () if mode.writes else reads, carries, _native=endpoint
    )


def _paired(waw, read, war):
    if waw is None or read.raw is None or read.writer != waw.source:
        return None
    return ReadCarry(waw, read.raw, war)


def compose_cells(
    left: CellTransfer, right: CellTransfer, collector: AccessCollector
) -> CellTransfer:
    """Splice logical left then right; internal needs stay in their owner.

    Each native occurrence and composed child may enter a parent only once.
    No endpoint equality is used as cross-generation RMW evidence.
    """
    collector._open()
    require(left.cell == right.cell, "cannot compose distinct physical cells")
    require(
        all(item._owner is None or item._owner is collector for item in (left, right)),
        "cell transfers have different internal-obligation owners",
    )
    collector._admit(left, right)
    collector.work.compositions += 1
    cell = left.cell
    bound_reads, raw_bindings = [], {}
    for read in right.head_reads:
        collector.work.reader_visits += 1
        raw = collector._need(cell, HazardKind.RAW, left.writer, read.reader)
        bound = PendingRead(read.reader, left.writer, raw)
        bound_reads.append(bound)
        raw_bindings[read.raw] = raw
    entry_carries = list(left.entry_carries)
    if right.first_write is not None:
        waw = collector._need(cell, HazardKind.WAW, left.writer, right.first_write)
        for read in left.tail_reads:
            collector.work.reader_visits += 1
            war = collector._need(cell, HazardKind.WAR, read.reader, right.first_write)
            if (carry := _paired(waw, read, war)) is not None:
                if isinstance(carry.waw.source, CellPort):
                    entry_carries.append(carry)
                else:
                    collector._carry(carry)
        for carry in right.entry_carries:
            raw = raw_bindings[carry.raw]
            if waw is not None and raw is not None:
                rebound = ReadCarry(waw, raw, carry.war)
                if isinstance(waw.source, CellPort):
                    entry_carries.append(rebound)
                else:
                    collector._carry(rebound)
    if left.first_write is None:
        head = left.head_reads + tuple(bound_reads)
    else:
        head = left.head_reads
    tail = (
        right.tail_reads if right.first_write is not None else left.tail_reads + tuple(bound_reads)
    )
    result = CellTransfer(
        cell,
        head,
        left.first_write or right.first_write,
        right.last_write or left.last_write,
        tail,
        tuple(entry_carries),
        collector,
    )
    collector._available.add(result._token)
    return result


def compose_cell_sequence(
    cell: Cell, transfers: Iterable[CellTransfer], collector: AccessCollector
) -> CellTransfer:
    """Scan admitted occurrences in logical source order, without reordering."""
    result = CellTransfer.identity(cell)
    for transfer in transfers:
        result = compose_cells(result, transfer, collector)
    return result


class ResiduePosition(str, Enum):
    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class LoopResidueSelection:
    """An actual loop's selected logical ordinal, not a closed loop summary."""

    loop: PipelineLoopOp | scf.ForOp
    period: int
    residue: int | Expr
    position: ResiduePosition

    def __post_init__(self):
        require(
            isinstance(self.loop, (PipelineLoopOp, scf.ForOp))
            and type(self.period) is int
            and self.period > 0
            and (
                type(self.residue) is int
                and 0 <= self.residue < self.period
                or isinstance(self.residue, Expr)
                and self.residue.kind == "rem"
                and self.residue.args[1].constant == self.period
            )
            and isinstance(self.position, ResiduePosition),
            "loop residue selection needs an actual loop, period/residue and FIRST/LAST",
        )

    def evaluate(self, count: int, values: dict[str, int] | None = None) -> int | None:
        """Bounded test/materialization interpreter; never enumerate N.

        The guard precedes N-1 arithmetic. Induction remains lb+ordinal*step;
        selecting a task's last occurrence proves no native retirement.
        """
        require(type(count) is int and count >= 0, "selection count must be nonnegative")
        require(
            values is not None or type(self.residue) is int,
            "symbolic residue needs coordinate values",
        )
        residue = self.residue if type(self.residue) is int else self.residue.evaluate(values)
        if count <= residue:
            return None
        if self.position is ResiduePosition.FIRST:
            return residue
        return count - 1 - ((count - 1 - residue) % self.period)

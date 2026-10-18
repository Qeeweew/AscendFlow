"""Certify closed private cell lifetimes before dispatch can reorder them.

A packet is an interval of immediate stages with one offset. Every incoming
read is defined inside it, all source RAW coordinates are retained, and its
first write covers the exact cell. This sufficient rule deliberately excludes
streaming envelopes, conditional packets and cross-generation state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..dialects.structured import PipelineLoopOp, PipelineStageOp
from ..indexing import arguments
from ..scalar import Expr
from .structured_regions import (
    AccessOccurrence,
    BranchGuard,
    CellSummary,
    Guard,
    LoopCountGuard,
    LoopOrdinalGuard,
    StorageCell,
)


@dataclass(frozen=True)
class StoragePacket:
    key: int
    stages: tuple[PipelineStageOp, ...]
    offset: int
    lexical: int
    summary: CellSummary
    guards: tuple[Guard, ...]


@dataclass(frozen=True)
class PacketPartition:
    loop: PipelineLoopOp
    cell: StorageCell
    packets: tuple[StoragePacket, ...]
    boundary_changed: bool

    def key_for(self, occurrence):
        stage = stage_in(occurrence.operation, self.loop)
        return next((p.key for p in self.packets if stage in p.stages), None)


def stage_in(operation, loop):
    while operation is not None and operation.parent_op() is not loop:
        operation = operation.parent_op()
    return operation if isinstance(operation, PipelineStageOp) else None


def _join(left, right):
    # Pure interface composition: definitions were already established by the
    # value phase. No replacement RAW or ordering requirement is emitted here.
    bound = tuple(replace(read, source=left.writer) for read in right.head_reads)
    return CellSummary(
        left.cell,
        left.head_reads + bound if left.first_write is None else left.head_reads,
        left.first_write or right.first_write,
        right.last_write or left.last_write,
        right.tail_reads if right.first_write is not None else left.tail_reads + bound,
    )


def _guard_invariant(guard, logical, memory):
    if isinstance(guard, BranchGuard):
        values = tuple(
            memory.control.environment.expression(v) for v in guard.operation.cond.owner.operands
        )
    elif isinstance(guard, LoopCountGuard):
        values = (guard.count,) + ((guard.threshold,) if isinstance(guard.threshold, Expr) else ())
    else:
        assert isinstance(guard, LoopOrdinalGuard)
        values = (guard.comparison.lhs, guard.comparison.rhs)
    return all(logical.ordinal.args[0] not in arguments(value) for value in values)


def certify_packets(values, dispatch, coordinates, loop, cell):
    if (
        not isinstance(cell, StorageCell)
        or cell.private is None
        or cell.streaming
        or coordinates.pattern(cell, loop) != (1, 0)
    ):
        return None
    stages = tuple(op for op in loop.body.block.ops if isinstance(op, PipelineStageOp))
    stages = tuple(stage for stage in stages if cell in values.summaries[stage].cells)
    if len(stages) < 2:
        return None
    # Each per-occurrence projection covers this cell. Dense family envelopes
    # are excluded above; an operand KILL is never promoted to a family kill.
    parent = {stage: stage for stage in stages}

    def find(stage):
        while parent[stage] is not stage:
            stage = parent[stage]
        return stage

    for read in values.reads:
        if read.cell != cell or not isinstance(read.source, AccessOccurrence):
            continue
        a, b = stage_in(read.source.operation, loop), stage_in(read.consumer.operation, loop)
        if a not in parent or b not in parent:
            continue
        first = next(binding.ordinal for binding in read.source.bindings if binding.loop is loop)
        second = next(binding.ordinal for binding in read.consumer.bindings if binding.loop is loop)
        if first != second:
            # A state recurrence is not an independent scratch value.
            return None
        parent[find(b)] = find(a)
    groups = {}
    for stage in stages:
        groups.setdefault(find(stage), []).append(stage)
    if len(groups) < 2:
        return None

    # Read the already established controller projection. Do not interpret
    # issue_order a second time, or flatten a nested child's internal rank.
    positions = {}
    for operation, cells in values.accesses.effects.items():
        if cell not in cells or (stage := stage_in(operation, loop)) not in parent:
            continue
        task = dispatch.tasks[operation, cell.private]
        scope, lexical = task.scope, task.lexical
        while scope.operation is not loop:
            lexical, scope = scope.parent_lexical, scope.parent
        positions[stage] = min(positions.get(stage, lexical), lexical)
    actual = tuple(sorted(stages, key=positions.__getitem__))
    logical = values.memory.control.loops[loop]
    packets = []
    for key, members in enumerate(groups.values()):
        ranks = [actual.index(stage) for stage in members]
        if ranks != list(range(min(ranks), max(ranks) + 1)):
            return None
        offsets = {stage.offset.data for stage in members}
        if len(offsets) != 1:
            return None
        summary, guards = CellSummary(cell), ()
        for stage in members:
            cases = values.summaries[stage].cells[cell]
            if len(cases) != 1:
                return None
            (case,) = cases
            guards = coordinates.guards(guards, case.guards)
            if guards is None:
                return None
            summary = _join(summary, case.transfer)
        if summary.head_reads or summary.first_write is None:
            return None
        if any(not _guard_invariant(g, logical, values.memory) for g in guards):
            return None
        packets.append(
            StoragePacket(key, tuple(members), offsets.pop(), min(ranks), summary, guards)
        )

    first = min(packets, key=lambda p: (p.offset, p.lexical))
    last = max(packets, key=lambda p: (p.offset, p.lexical))
    changed = first is not packets[0] or last is not packets[-1]
    if changed:
        # Changing an enclosing interface needs propagation. This initial
        # certificate admits it only for root-private scratch with no outside
        # access and no enclosing repeated invocation of the same allocation.
        frame = values.memory.control.frames[loop]
        if frame.parent is not None and frame.parent.loops():
            return None
        if any(
            cell in cells and stage_in(operation, loop) is None
            for operation, cells in values.accesses.effects.items()
        ):
            return None
    return PacketPartition(loop, cell, tuple(packets), changed)

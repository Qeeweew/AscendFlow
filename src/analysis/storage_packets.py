"""Local complete private packets whose physical slots may follow dispatch.

This precision rule inspects immediate stage interfaces and full native writes;
no whole-kernel value frontier or RAW object identity is an admission premise.
A state input, conditional port, streaming cell or mixed generation declines
packet reordering while retaining ordinary source order.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dialects.structured import PipelineLoopOp, PipelineStageOp
from .access_frontier import QueueEndpoint
from .structured_regions import StorageCell


@dataclass(frozen=True)
class StoragePacket:
    key: int
    stages: tuple[PipelineStageOp, ...]
    offset: int
    lexical: int
    summary: object


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


def certify_packets(builder, loop, cell):
    from .access_order import AccessTransfer

    if (
        not isinstance(loop, PipelineLoopOp)
        or not isinstance(cell, StorageCell)
        or cell.private is None
        or cell.streaming
        or cell.conditions
        or builder.coordinates.pattern(cell, loop) != (1, 0)
    ):
        return None
    stages = tuple(
        op
        for op in loop.body.block.ops
        if isinstance(op, PipelineStageOp) and cell in builder.summaries[op]
    )
    if len(stages) < 2:
        return None
    parent = {stage: stage for stage in stages}

    def find(stage):
        while parent[stage] is not stage:
            stage = parent[stage]
        return stage

    writer = None
    logical = builder.memory.control.loops[loop]
    for stage in stages:
        transfer = builder.summaries[stage][cell]
        points = (
            transfer.head_reads + transfer.first_writes + transfer.tail_writes + transfer.tail_reads
        )
        if any(isinstance(p.endpoint, QueueEndpoint) for p in points):
            return None
        if any(
            next((b.ordinal for b in p.bindings if b.loop is loop), None) != logical.ordinal
            for p in points
        ):
            return None
        if transfer.head_reads:
            if writer is None:
                return None  # Real state input, not an independent scratch value.
            parent[find(stage)] = find(writer)
        if transfer.tail_writes:
            writer = stage
    groups = {}
    for stage in stages:
        groups.setdefault(find(stage), []).append(stage)
    if len(groups) < 2:
        return None
    positions = {}
    for operation, cells in builder.accesses.effects.items():
        if cell not in cells or (stage := stage_in(operation, loop)) not in parent:
            continue
        task = builder.dispatch.tasks[operation, cell.private]
        scope, lexical = task.scope, task.lexical
        while scope.operation is not loop:
            lexical, scope = scope.parent_lexical, scope.parent
        positions[stage] = min(positions.get(stage, lexical), lexical)
    actual = tuple(sorted(stages, key=positions.__getitem__))
    packets = []
    for key, members in enumerate(groups.values()):
        ranks = [actual.index(stage) for stage in members]
        offsets = {stage.offset.data for stage in members}
        if ranks != list(range(min(ranks), max(ranks) + 1)) or len(offsets) != 1:
            return None
        summary = AccessTransfer()
        for stage in members:
            summary = summary.join(builder.summaries[stage][cell])
        if summary.head_reads or not summary.ordered_write or len(summary.first_writes) != 1:
            return None
        first_frames = summary.first_writes[0].frame.ancestors()
        index = next(i for i, frame in enumerate(first_frames) if frame.operation is loop)
        if any(frame.branch is not None for frame in first_frames[index + 1 :]):
            return None
        packets.append(StoragePacket(key, tuple(members), offsets.pop(), min(ranks), summary))
    first = min(packets, key=lambda p: (p.offset, p.lexical))
    last = max(packets, key=lambda p: (p.offset, p.lexical))
    changed = first is not packets[0] or last is not packets[-1]
    if changed:
        frame = builder.memory.control.frames[loop]
        if frame.parent is not None and frame.parent.loops():
            return None
        if any(
            cell in cells and stage_in(op, loop) is None
            for op, cells in builder.accesses.effects.items()
        ):
            return None
    return PacketPartition(loop, cell, tuple(packets), changed)

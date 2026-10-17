"""Normalize all structured loop controls without copying their bodies.

This first control pass attaches a typed PLE schedule to the actual source IR
and returns owned handles for subsequent native lowering. It does not claim
to synchronize a program or admit arbitrary loop-carried native state.

Cost is one IR traversal plus at most four phase descriptors per loop. Bounds
stay SSA or static literals. Ordinary work before/after/between nested loops
is retained in place; there is no attention, query, or fixed-child dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from xdsl.context import Context
from xdsl.dialects import scf
from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.passes import ModulePass

from ..analysis.iteration import LoopControl, LoopControls, iteration_domain
from ..diagnostics import require
from ..dialects.loops import LoopScheduleAttr
from ..dialects.source import PipelineOp, StageOp

SCHEDULE_ATTRIBUTE = "npu.loop_schedule"


def normalize_loop_controls(module: ModuleOp) -> LoopControls:
    """Attach one canonical schedule per real loop, preserving source effects.

    This is an internal transformation on a privately owned, verified module.
    Existing attributes are recomputed from actual control, never trusted as a
    proof. Unsupported step expressions are diagnosed before any mutation.
    """
    loops, index = [], {}
    visited = descriptors = 0

    def visit(operation, parent=None, stage=0):
        nonlocal visited, descriptors
        visited += 1
        child_parent, child_stage = parent, stage
        if isinstance(operation, (PipelineOp, scf.ForOp)):
            domain = iteration_domain(operation)
            horizon = operation.stages.data - 1 if isinstance(operation, PipelineOp) else 0
            schedule = LoopScheduleAttr(IntAttr(horizon))
            schedule.verify()
            control = LoopControl(
                operation,
                domain,
                schedule,
                parent,
                stage,
                1 if parent is None else parent.depth + 1,
            )
            loops.append(control)
            index[operation] = control
            descriptors += len(schedule.phases)
            child_parent, child_stage = control, 0
        elif isinstance(operation, StageOp):
            require(
                parent is not None and operation.parent_op() is parent.operation,
                "Stage must belong to its actual enclosing loop",
                phase="loop-control",
                location=str(operation.location),
            )
            child_stage = operation.offset.data
        for region in operation.regions:
            for block in region.blocks:
                for child in block.ops:
                    visit(child, child_parent, child_stage)

    visit(module)
    for loop in loops:
        loop.operation.attributes[SCHEDULE_ATTRIBUTE] = loop.schedule
    return LoopControls(module, tuple(loops), MappingProxyType(index), visited, descriptors)


@dataclass(frozen=True)
class LowerPipelinePass(ModulePass):
    """Normalize control for native lowering; synchronization is a later pass."""

    name = "npu-lower-pipeline"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        normalize_loop_controls(op)

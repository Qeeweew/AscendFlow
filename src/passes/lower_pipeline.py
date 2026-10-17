"""Normalize all structured loop controls without copying their bodies.

This control pass returns owned iteration handles for native lowering. It does not claim
to synchronize a program or admit arbitrary loop-carried native state.

Cost is one IR traversal with one domain per loop. Bounds
stay SSA or static literals. Ordinary work before/after/between nested loops
is retained in place; there is no attention, query, or fixed-child dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from xdsl.context import Context
from xdsl.dialects import scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass

from ..analysis.iteration import LoopControl, LoopControls, iteration_domain
from ..diagnostics import require
from ..dialects.source import PipelineOp, StageOp


def normalize_loop_controls(module: ModuleOp) -> LoopControls:
    """Collect one normalized domain per real loop, preserving source effects.

    This is an internal transformation on a privately owned, verified module.
    Bounds and offsets come from actual control, never a verified marker.
    """
    loops, index = [], {}
    visited = 0

    def visit(operation, parent=None, stage=0):
        nonlocal visited
        visited += 1
        child_parent, child_stage = parent, stage
        if isinstance(operation, (PipelineOp, scf.ForOp)):
            domain = iteration_domain(operation)
            horizon = (
                max(
                    (op.offset.data for op in operation.body.block.ops if isinstance(op, StageOp)),
                    default=0,
                )
                if isinstance(operation, PipelineOp)
                else 0
            )
            control = LoopControl(
                operation,
                domain,
                horizon,
                parent,
                stage,
                1 if parent is None else parent.depth + 1,
            )
            loops.append(control)
            index[operation] = control
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
    return LoopControls(module, tuple(loops), MappingProxyType(index), visited)


@dataclass(frozen=True)
class LowerPipelinePass(ModulePass):
    """Normalize control for native lowering; synchronization is a later pass."""

    name = "npu-lower-pipeline"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        normalize_loop_controls(op)

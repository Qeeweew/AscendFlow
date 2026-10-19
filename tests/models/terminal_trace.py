"""Finite final-IR native and notification trace, with no production scheduling query."""

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.physical import TaskOp
from ascendflow.target.ascend910b import Actor
from tests.models.structured_events import (
    _Interpret,
)


def trace(module, parameters, actor=Actor.AIV0):
    memory = collect_native_memory(module)
    model = _Interpret(memory, parameters, physical=True, maximum=10000)
    model.block(module.body.block.first_op.body.block, {}, 0, 0, actor)
    return tuple(
        (
            c.kind,
            c.source,
            c.destination,
            c.event,
            c.occurrence.operation.uid.data if isinstance(c.occurrence.operation, TaskOp) else None,
            tuple(g for _, g in c.occurrence.coordinates),
        )
        for c in model.commands
    )

"""Compositional literal SPR definitions; common sync handles retirement."""

from xdsl.dialects import scf

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp

_INCOMING = object()


def admit_register_definitions(memory, *, work=None):
    """Summarize each source block once, without expanding runtime loops.

    Each register has an incoming-value requirement and a finite set of
    possible outgoing literals, optionally retaining the incoming value.
    Sequence substitutes these sets; branches union paths; a repeated body
    must preserve its own incoming requirements. Zero-trip paths retain the
    old state. This proves values, never visibility or reader retirement.
    """
    templates = {t.operation: t for t in memory.templates}
    visited = 0

    def fail(condition, register, value, operation):
        require(
            condition,
            f"{register} reader lacks its required dominating configuration {value}",
            phase="register-definedness",
            location=str(operation.location),
        )

    def summarize(source):
        nonlocal visited
        visited += 1
        required, values = {}, {}

        def demand(key, value, operation):
            alternatives = values.get(key, frozenset((_INCOMING,)))
            for existing in alternatives:
                if existing is _INCOMING:
                    prior = required.get(key)
                    fail(prior is None or prior[0] == value, key[1], value, operation)
                    required[key] = value, operation
                else:
                    fail(existing == value, key[1], value, operation)

        def apply(child):
            inputs, outputs = child
            for key, (value, operation) in inputs.items():
                demand(key, value, operation)
            for key, alternatives in outputs.items():
                old = values.get(key, frozenset((_INCOMING,)))
                values[key] = frozenset(v for v in alternatives if v is not _INCOMING) | (
                    old if _INCOMING in alternatives else frozenset()
                )

        for op in source.ops:
            if isinstance(op, TaskOp):
                template = templates[op]
                for actor in template.participants:
                    for effect in template.spec.registers:
                        key = actor, effect.register
                        if effect.mode == "W":
                            values[key] = frozenset((effect.value,))
                        else:
                            demand(key, effect.value, op)
            elif isinstance(op, scf.IfOp):
                paths = [summarize(r.block) if r.blocks else ({}, {}) for r in op.regions]
                inputs, outputs = {}, {}
                for incoming, _ in paths:
                    for key, (value, operation) in incoming.items():
                        fail(key not in inputs or inputs[key][0] == value, key[1], value, operation)
                        inputs[key] = value, operation
                for key in {key for _, path in paths for key in path}:
                    outputs[key] = frozenset().union(
                        *(path.get(key, frozenset((_INCOMING,))) for _, path in paths)
                    )
                apply((inputs, outputs))
            elif isinstance(op, (scf.ForOp, PipelineLoopOp)):
                count = memory.control.loops[op].count.constant
                if count is not None and count <= 0:
                    continue
                inputs, outputs = summarize(op.body.block)
                if count != 1:
                    for key, (value, operation) in inputs.items():
                        fail(
                            all(v is _INCOMING or v == value for v in outputs.get(key, ())),
                            key[1],
                            value,
                            operation,
                        )
                if count is None:
                    outputs = {
                        key: alternatives | frozenset((_INCOMING,))
                        for key, alternatives in outputs.items()
                    }
                apply((inputs, outputs))
            elif isinstance(op, (ActorDomainOp, PipelineStageOp)):
                apply(summarize(op.body.block))
        return required, values

    required, _ = summarize(memory.module.body.block.first_op.body.block)
    for (_, register), (value, operation) in required.items():
        fail(False, register, value, operation)
    if work is not None:
        work["register_scope_summaries"] = visited
    return tuple(
        t.operation
        for t in memory.templates
        if any(effect.mode == "W" for effect in t.spec.registers)
    )

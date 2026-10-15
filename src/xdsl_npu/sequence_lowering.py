"""Lower ordinary work and retained local streams with shared allocation identity."""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr

from .ir import EndOp
from .lowering_context import ActorBindings
from .model import require
from .scalar import ARITHMETIC
from .source_ir import AllocationOp, DomainOp, PipelineOp, SizeOp, ViewOp, YieldOp


def _has_work(op):
    if isinstance(op, DomainOp):
        return any(_has_work(child) for child in op.body.block.ops)
    return not (
        isinstance(op, (AllocationOp, SizeOp, ViewOp, YieldOp, arith.ConstantOp))
        or type(op) in ARITHMETIC
    )


def needs_sequence(program):
    if any(isinstance(op, scf.ForOp) for op in program.body.block.ops):
        return False  # Existing invocation path keeps its explicit admission.
    pipelines = [op for op in program.body.block.ops if isinstance(op, PipelineOp)]
    return bool(pipelines) and (
        len(pipelines) > 1
        or any(_has_work(op) and not isinstance(op, PipelineOp) for op in program.body.block.ops)
    )


def lower_sequence(program):
    from .lowering import _Actor, _check_loop_values, _work
    from .sequence_model import extract_sequence

    require(
        program.kind.data in {"cube", "vector"}, "mixed sequence ownership is not yet implemented"
    )
    pipelines = [op for op in program.body.block.ops if isinstance(op, PipelineOp)]
    require(
        all(op.parent_op() is program for op in program.walk() if isinstance(op, PipelineOp)),
        "nested pipelines require hierarchical region synchronization",
    )
    require(
        not any(isinstance(op, scf.ForOp) for op in program.walk()),
        "serial controls inside sequential regions require hierarchy lowering",
    )
    allocations = [op for op in program.walk() if isinstance(op, AllocationOp)]
    family = program.kind.data
    chunks, pending = [], []
    for op in program.body.block.ops:
        if isinstance(op, PipelineOp):
            if pending:
                chunks.append((pipelines[0], tuple(pending)))
                pending = []
            chunks.append((op, None))
        else:
            if not isinstance(op, SizeOp):
                pending.append(op)
    if pending:
        chunks.append((pipelines[0], tuple(pending)))
    actors = []
    bindings = ActorBindings(program, family, allocations)
    for pipeline, operations in chunks:
        records = _work(program, pipeline, family, 0, operations=operations)
        if not records:
            continue
        name = f"r{len(actors)}"
        actor = _Actor(
            program,
            pipeline,
            family,
            family,
            0,
            allocations,
            symbolic=True,
            records=records,
            once=operations is not None,
            region_name=name,
            bindings=bindings,
        )
        actor.emit(finalize=False)
        actors.append(actor)
    require(actors, "sequence contains no actual source work")
    result = bindings.kernel.finish()
    kernel = bindings.kernel.op
    kernel.attributes["source_schedule"] = ArrayAttr(
        item for actor in actors for item in actor.schedule
    )
    require(isinstance(kernel.body.block.last_op, EndOp), "sequence needs a kernel exit")
    sequence = extract_sequence(result)
    for actor in actors:
        name = actor.region_name
        _check_loop_values(
            result,
            {family: actor},
            actor.pipeline,
            extracted=sequence.regions[name],
            live_in=sequence.live_in[name],
        )
    result.verify()
    return result

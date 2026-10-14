"""Lower ordinary work and retained local streams with shared allocation identity."""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr

from .ir import EndOp, FixConfigOp
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
        )
        actor.emit()
        actors.append(actor)
    require(actors, "sequence contains no actual source work")
    first = actors[0]
    result, kernel = first.kernel.module, first.kernel.op
    before = first.kernel.stream
    values = dict(first.uniform_by_expression)
    buffers = dict(first.roots)
    for actor in actors[1:]:
        for expr, value in actor.uniform_by_expression.items():
            if expr in values:
                value.replace_all_uses_with(values[expr])
            else:
                operation = value.owner
                operation.parent.detach_op(operation)
                kernel.body.block.insert_op_before(operation, before)
                values[expr] = value
        for allocation, value in actor.roots.items():
            if allocation in buffers:
                original = buffers[allocation]
                require(
                    value.owner.attributes == original.owner.attributes
                    and value.type == original.type
                    and tuple(value.owner.operands) == tuple(original.owner.operands),
                    "source allocation changed across sequential regions",
                )
                value.replace_all_uses_with(original)
            else:
                operation = value.owner
                operation.parent.detach_op(operation)
                kernel.body.block.insert_op_before(operation, before)
                buffers[allocation] = value
        require(
            sum(isinstance(op, FixConfigOp) for op in actor.kernel.op.body.block.ops) <= 1,
            "unexpected region configuration effects",
        )
        region = actor.kernel.stream
        region.parent.detach_op(region)
        kernel.body.block.insert_op_before(region, kernel.body.block.last_op)
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

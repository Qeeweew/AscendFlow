"""Lower ordinary work and retained local streams with shared allocation identity."""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr

from .ir import EndOp
from .lowering_context import ActorBindings
from .model import require
from .scalar import ARITHMETIC, from_ssa, literal
from .source_ir import AllocationOp, DomainOp, PipelineOp, SizeOp, ViewOp, YieldOp


def _has_work(op):
    if isinstance(op, DomainOp):
        return any(_has_work(child) for child in op.body.block.ops)
    return not (
        isinstance(op, (AllocationOp, SizeOp, ViewOp, YieldOp, scf.YieldOp, arith.ConstantOp))
        or type(op) in ARITHMETIC
    )


def needs_sequence(program):
    parents = [op for op in program.body.block.ops if isinstance(op, scf.ForOp)]
    if len(parents) > 1:
        return False
    scope = parents[0] if parents else program
    pipelines = [op for op in scope.body.block.ops if isinstance(op, PipelineOp)]
    return bool(pipelines) and (
        len(pipelines) > 1
        or any(_has_work(op) and not isinstance(op, PipelineOp) for op in scope.body.block.ops)
    )


def lower_sequence(program):
    from .lowering import _Actor, _check_loop_values, _work
    from .sequence_model import extract_sequence

    require(
        program.kind.data in {"cube", "vector"}, "mixed sequence ownership is not yet implemented"
    )
    parents = [op for op in program.body.block.ops if isinstance(op, scf.ForOp)]
    invocation = parents[0] if len(parents) == 1 else None
    scope = invocation or program
    if invocation is not None:
        require(
            not invocation.iter_args
            and not invocation.results
            and from_ssa(invocation.lb) == literal(0)
            and from_ssa(invocation.step) == literal(1),
            "sequential invocation requires zero-based unit-step control with no scalar carry",
        )
        require(
            all(op is invocation or not _has_work(op) for op in program.body.block.ops),
            "work outside a serial sequence requires an enclosing region contract",
        )
    pipelines = [op for op in scope.body.block.ops if isinstance(op, PipelineOp)]
    require(
        all(op.parent_op() is scope for op in program.walk() if isinstance(op, PipelineOp)),
        "nested pipelines require hierarchical region synchronization",
    )
    require(
        all(op is invocation for op in program.walk() if isinstance(op, scf.ForOp)),
        "serial controls inside sequential regions require hierarchy lowering",
    )
    allocations = [op for op in program.walk() if isinstance(op, AllocationOp)]
    family = program.kind.data
    chunks, pending = [], []
    for op in scope.body.block.ops:
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
    bindings = ActorBindings(program, family, allocations, invocation=invocation)
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
            invocation=invocation,
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

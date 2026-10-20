"""Finite native scalar/address dispatch; no asynchronous completion oracle."""

import operator

from xdsl.dialects import arith, scf

from ascendflow.dialects.physical import (
    BufferOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    RuntimeSizeOp,
    TaskOp,
)
from ascendflow.dialects.structured import ActorDomainOp, AddressOp, PipelineLoopOp, PipelineStageOp
from ascendflow.dialects.types import RefType


def interpret(block, values, arguments, trace):
    """Independent scalar evaluation; plain clock enumeration only in tests."""
    for op in block.ops:
        if isinstance(op, FixConfigOp):
            result = ()  # Register setup is outside this scalar/address oracle.
        elif isinstance(op, RuntimeSizeOp):
            result = (arguments[op.sym_name.data],)
        elif isinstance(op, CoreIdOp):
            result = (arguments.get("group", 0),)
        elif isinstance(op, BufferOp):
            result = (op,)
        elif isinstance(op, arith.ConstantOp):
            result = (op.value.value.data,)
        elif isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp)):
            a, b = (values[operand] for operand in op.operands)
            result = (
                {
                    arith.AddiOp: operator.add,
                    arith.SubiOp: operator.sub,
                    arith.MuliOp: operator.mul,
                    arith.RemUIOp: operator.mod,
                }[type(op)](a, b),
            )
        elif isinstance(op, arith.CmpiOp):
            a, b = (values[operand] for operand in op.operands)
            result = (
                {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[
                    op.predicate.value.data
                ],
            )
        elif isinstance(op, scf.ForOp):
            state = tuple(values[value] for value in op.iter_args)
            for iv in range(values[op.lb], values[op.ub], values[op.step]):
                local = values | dict(zip(op.body.block.args, (iv, *state), strict=True))
                state = interpret(op.body.block, local, arguments, trace)
            result = state
        elif isinstance(op, scf.IfOp):
            region = op.true_region if values[op.cond] else op.false_region
            result = interpret(region.block, dict(values), arguments, trace)
        elif isinstance(op, scf.YieldOp):
            return tuple(values[value] for value in op.arguments)
        elif isinstance(op, PipelineLoopOp):
            iterations = list(range(max(0, values[op.count])))
            stages = [
                stage
                for stage in op.body.block.ops
                if isinstance(stage, PipelineStageOp)
                and (stage.family is None or stage.family.data == arguments.get("actor"))
            ]
            if (
                op.issue_order is not None
                and (policy := op.issue_order.data.get(arguments.get("actor"))) is not None
            ):
                by_name = {stage.label.data: stage for stage in stages}
                stages = [by_name[label.data] for label in policy]
            horizon = max((stage.offset.data for stage in stages), default=0)
            for clock in range(len(iterations) + horizon):
                for stage in stages:
                    g = clock - stage.offset.data
                    if 0 <= g < len(iterations):
                        local = values | dict(zip(stage.body.block.args, (g,), strict=True))
                        interpret(stage.body.block, local, arguments, trace)
            result = ()
        elif isinstance(op, ActorDomainOp):
            if op.kind.data == arguments.get("actor"):
                owner = arguments.get("sid", 0) if op.kind.data == "vector" else 0
                interpret(op.body.block, values | {op.body.block.args[0]: owner}, arguments, trace)
            result = ()
        elif isinstance(op, AddressOp):
            result = ((values[op.source].sym_name.data, values[op.offset]),)
        elif isinstance(op, TaskOp):
            refs = tuple(values[value] for value in op.refs[:3] if isinstance(value.type, RefType))
            trace.append((op.native.data, op.scalar.value.data, refs))
            result = (op,)
        elif isinstance(op, EndOp):
            return ()
        else:
            raise TypeError(f"unsupported oracle instruction {op.name}")
        values.update(zip(op.results, result, strict=True))
    return ()


def dispatch(plan, **arguments):
    values, trace = {}, []
    root = plan.module.body.block.first_op
    if "actor" not in arguments and hasattr(root, "core"):
        arguments["actor"] = "cube" if root.core.data == "aic" else "vector"
    interpret(root.body.block, values, arguments, trace)
    return values, trace

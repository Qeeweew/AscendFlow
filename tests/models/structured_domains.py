"""Finite retained-control and requirement-domain interpreters for tests.

Reuses collected ControlFacts scalar expressions, counts and ordinal mappings,
admitted dispatch task membership, requirement domain/guard records, and the
production Expr and LoopResidueSelection evaluators. Loop/stage issue order and
branches are interpreted directly from retained IR. No production queue-order,
notification-selection, lifetime or progress helper supplies an interpreted
edge. Finite enumeration here is not a general symbolic proof.
"""

from xdsl.dialects import arith, scf

from ascendflow.analysis.access_frontier import LoopResidueSelection
from ascendflow.analysis.structured_regions import BranchGuard, LoopCountGuard, LoopOrdinalGuard
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp
from ascendflow.target.ascend910b import Actor


def expression_value(value, values):
    return value if type(value) is int else value.evaluate(values)


def guard_active(guard, values, memory):
    if isinstance(guard, (LoopCountGuard, LoopOrdinalGuard)):
        return guard.evaluate(values)
    assert isinstance(guard, BranchGuard)
    compare = guard.operation.cond.owner
    lhs, rhs = (memory.control.environment.expression(v).evaluate(values) for v in compare.operands)
    predicate = compare.predicate.value.data
    assert predicate in {0, 1}
    truth = lhs == rhs if predicate == 0 else lhs != rhs
    return truth == (guard.branch == 0)


def occurrence_key(occurrence, values, memory):
    bound = []
    for binding in occurrence.bindings:
        logical = memory.control.loops[binding.loop]
        ordinal = binding.ordinal
        if isinstance(ordinal, LoopResidueSelection):
            ordinal = ordinal.evaluate(logical.count.evaluate(values), values)
        else:
            ordinal = ordinal.evaluate(values)
        assert ordinal is not None
        assert 0 <= ordinal < logical.count.evaluate(values)
        bound.append((binding.loop, ordinal))
    return occurrence.operation, occurrence.participant, tuple(bound)


def actual_dispatch(selection, uniform, actor):
    """Independent lexical/physical-step interpreter; no production rank helper."""
    memory, events = selection.dependencies.memory, []
    environment = memory.control.environment
    family = "cube" if actor is Actor.AIC else "vector"
    base = uniform | {"$sid": int(actor is Actor.AIV1)}

    def scalar(value, context):
        return environment.expression(value).evaluate(context)

    def visit(block, context, coordinates):
        for operation in block.ops:
            if isinstance(operation, ActorDomainOp):
                if operation.kind.data == family:
                    visit(operation.body.block, context, coordinates)
            elif isinstance(operation, PipelineLoopOp):
                iterations = range(*(scalar(value, context) for value in operation.operands))
                stages = [op for op in operation.body.block.ops if isinstance(op, PipelineStageOp)]
                if operation.issue_order is not None and family in operation.issue_order.data:
                    by_name = {stage.label.data: stage for stage in stages}
                    stages = [by_name[label.data] for label in operation.issue_order.data[family]]
                symbol = memory.control.loops[operation].ordinal.args[0]
                for clock in range(len(iterations) + operation.schedule.horizon.data):
                    for stage in stages:
                        ordinal = clock - stage.offset.data
                        if 0 <= ordinal < len(iterations):
                            visit(
                                stage.body.block,
                                context | {symbol: ordinal},
                                coordinates + ((operation, ordinal),),
                            )
            elif isinstance(operation, scf.ForOp):
                symbol = memory.control.loops[operation].ordinal.args[0]
                for ordinal, _ in enumerate(
                    range(
                        *(
                            scalar(v, context)
                            for v in (
                                operation.lb,
                                operation.ub,
                                operation.step,
                            )
                        )
                    )
                ):
                    visit(
                        operation.body.block,
                        context | {symbol: ordinal},
                        coordinates + ((operation, ordinal),),
                    )
            elif isinstance(operation, scf.IfOp):
                comparison = operation.cond.owner
                assert isinstance(comparison, arith.CmpiOp)
                a, b = (scalar(value, context) for value in comparison.operands)
                truth = {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[
                    comparison.predicate.value.data
                ]
                region = operation.true_region if truth else operation.false_region
                if region.blocks:
                    visit(region.block, context, coordinates)
            elif isinstance(operation, TaskOp) and (operation, actor) in selection.dispatch.tasks:
                events.append(((operation, actor, coordinates), dict(context)))

    visit(memory.module.body.block.first_op.body.block, base, ())
    return events


def need_cases(need, uniform, memory):
    domains = need.domains + need.source.families + need.consumer.families
    loops = [
        loop for op, loop in memory.control.loops.items() if any(d.loop is op for d in domains)
    ]

    def visit(index, context):
        if index == len(loops):
            if all(guard_active(g, context, memory) for g in need.guards):
                yield context
            return
        logical = loops[index]
        current = [d for d in domains if d.loop is logical.operation]
        limit = max(d.count.evaluate(context) for d in current)
        for ordinal in range(limit):
            child = context | {logical.ordinal.args[0]: ordinal}
            if all(
                max(d.distance, d.first) <= ordinal < d.count.evaluate(child)
                and ordinal % d.period == expression_value(d.residue, child)
                for d in current
            ):
                yield from visit(index + 1, child)

    yield from visit(0, uniform)


def point_key(point, context, memory):
    coordinates = tuple((loop, value.evaluate(context)) for loop, value in point.coordinates)
    actual = (
        context
        | {memory.control.loops[loop].ordinal.args[0]: value for loop, value in coordinates}
        | {"$sid": int(point.actor is Actor.AIV1)}
    )
    assert point.guard.evaluate(actual)
    return point.operation, point.actor, coordinates

"""Terminal lowering of admitted pipeline domains into ordinary arith/SCF.

Native and notification operations move exactly once. Only pure SSA and control
shells are projected for the two actor families; the two AIV lanes share a body.
Source analysis keeps its immutable logical coordinates, while the resulting
module contains the actual physical clock and guarded index evaluations.
"""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.ir import Block, Region

from ..analysis.scalar_uses import ScalarUses
from ..diagnostics import require
from ..dialects.physical import EndOp, FixConfigOp, MixedOp, TaskOp
from ..dialects.structured import (
    ActorDomainOp,
    AddressOp,
    NotificationOp,
    PipelineLoopOp,
    PipelineStageOp,
)


def lower_execution(module, natives, facts, schedules):
    uses = ScalarUses(module, natives, metadata=True)
    root = module.body.block.first_op

    def remap(value, bindings):
        return bindings.get(value, value)

    def index(block, value):
        op = arith.ConstantOp.from_int_and_width(value, IndexType())
        block.add_op(op)
        return op.result

    def arithmetic(block, cls, *operands):
        op = cls(*operands)
        block.add_op(op)
        return op.results[0]

    def project(source, target, family, bindings, sid):
        for op in tuple(source.ops):
            if isinstance(op, EndOp):
                continue
            if isinstance(op, scf.YieldOp):
                target.add_op(scf.YieldOp(*(remap(v, bindings) for v in op.arguments)))
            elif isinstance(op, ActorDomainOp):
                if op.kind.data == family:
                    require(sid is not None, "actor projection lost its runtime lane")
                    local = bindings | {op.body.block.args[0]: sid}
                    project(op.body.block, target, family, local, sid)
            elif isinstance(op, PipelineLoopOp):
                if family in uses.operations[op]:
                    clock(op, target, bindings, family, sid)
            elif isinstance(op, (TaskOp, NotificationOp)):
                if family not in uses.operations[op]:
                    continue
                require(
                    len(uses.operations[op]) == 1,
                    "one native instruction belongs to two actor families",
                )
                op.parent.detach_op(op)
                op.operands = tuple(remap(v, bindings) for v in op.operands)
                if isinstance(op, TaskOp):
                    op.stage = IntAttr(0)
                target.add_op(op)
            elif isinstance(op, FixConfigOp):
                if family == "cube":
                    op.parent.detach_op(op)
                    target.add_op(op)
            elif isinstance(op, (scf.ForOp, scf.IfOp)):
                if family not in uses.operations[op]:
                    continue
                clone = op.clone_without_regions(value_mapper=bindings)
                target.add_op(clone)
                for old_region, new_region in zip(op.regions, clone.regions, strict=True):
                    for old_block in old_region.blocks:
                        block = Block(arg_types=[v.type for v in old_block.args])
                        new_region.add_block(block)
                        local = bindings | dict(zip(old_block.args, block.args, strict=True))
                        project(old_block, block, family, local, sid)
            elif isinstance(op, (AddressOp, *tuple(arith.Arith.operations))):
                if any((value, family) in uses.values for value in op.results):
                    clone = op.clone(value_mapper=bindings)
                    target.add_op(clone)
                    for old, new in zip(op.results, clone.results, strict=True):
                        if old in facts.ranges:
                            facts.bind_range(new, facts.ranges[old])
            else:
                require(False, f"unmodeled terminal execution operation {op.name}")

    def clock(op, target, bindings, family=None, sid=None):
        families = (
            (family,)
            if family
            else ("cube", "vector")
            if isinstance(root, MixedOp)
            else ("cube" if root.core.data == "aic" else "vector",)
        )
        for current in families:
            if current not in uses.operations[op]:
                continue
            destination, lane = target, sid
            if family is None and isinstance(root, MixedOp):
                destination = Block(arg_types=[IndexType()])
                domain = ActorDomainOp.build(
                    attributes={"kind": StringAttr(current)}, regions=[Region(destination)]
                )
                target.add_op(domain)
                lane = destination.args[0]
            schedule = schedules.get((op, current))
            require(schedule is not None, "terminal actor has no admitted local schedule")
            require(
                set(schedule.stages)
                == {
                    stage
                    for stage in op.body.block.ops
                    if isinstance(stage, PipelineStageOp) and current in uses.operations[stage]
                },
                "notification insertion changed a stage's participating actor families",
            )
            raw = remap(op.count, bindings)
            zero, one = index(destination, 0), index(destination, 1)
            count = arithmetic(destination, arith.MaxSIOp, raw, zero)
            horizon = schedule.horizon
            raw_range = facts.ranges.get(raw, facts.ranges.get(op.count, (0, (1 << 63) - 1)))
            if isinstance(raw.owner, arith.ConstantOp):
                value = raw.owner.value.value.data
                raw_range = value, value
            # Each actor chooses its own defined clock range. A second actor's
            # horizon cannot enlarge this loop or its shifted-ordinal arithmetic.
            shifted = raw_range[1] > (1 << 63) - 1 - horizon
            lower = index(destination, -horizon) if shifted else zero
            end = (
                count
                if shifted or horizon == 0
                else arithmetic(destination, arith.AddiOp, count, index(destination, horizon))
            )
            body = Block(arg_types=[IndexType()])
            loop = scf.ForOp(lower, end, one, [], body)
            loop.attributes["npu.pipeline_clock"] = IntAttr(-horizon if shifted else 0)
            if "npu.loop_id" in op.attributes:
                loop.attributes["npu.loop_id"] = op.attributes["npu.loop_id"]
            destination.add_op(loop)
            tick = body.args[0]
            if lane is None:
                lane = zero
            for stage in schedule.stages:
                shift = horizon - stage.offset.data if shifted else -stage.offset.data
                # Guard before evaluating a potentially overflowing shifted
                # ordinal. The tick inequalities are equivalent to 0<=g<N.
                minimum = index(body, -shift)
                limit = (
                    count
                    if shift == 0
                    else arithmetic(
                        body,
                        arith.SubiOp if shift > 0 else arith.AddiOp,
                        count,
                        index(body, abs(shift)),
                    )
                )
                first, last = arith.CmpiOp(tick, minimum, "sge"), arith.CmpiOp(tick, limit, "slt")
                body.add_ops((first, last))
                active = arithmetic(body, arith.AndIOp, first.result, last.result)
                guarded = Block()
                condition = scf.IfOp(active, [], Region(guarded), Region())
                condition.attributes["npu.stage_offset"] = IntAttr(stage.offset.data)
                body.add_op(condition)
                generation = (
                    tick
                    if shift == 0
                    else arithmetic(guarded, arith.AddiOp, tick, index(guarded, shift))
                )
                local = bindings | {stage.body.block.args[0]: generation}
                project(stage.body.block, guarded, current, local, lane)
                guarded.add_op(scf.YieldOp())
            body.add_op(scf.YieldOp())
            if family is None and isinstance(root, MixedOp):
                destination.add_op(EndOp())

    def visit(block):
        for op in tuple(block.ops):
            if isinstance(op, PipelineLoopOp):
                target = Block()
                parent, family, sid = op.parent_op(), None, None
                while parent is not None:
                    if isinstance(parent, ActorDomainOp):
                        family, sid = parent.kind.data, parent.body.block.args[0]
                        break
                    parent = parent.parent_op()
                clock(op, target, {}, family, sid)
                replacement = list(target.ops)
                for child in replacement:
                    target.detach_op(child)
                block.insert_ops_before(replacement, op)
                block.erase_op(op)
            else:
                for region in op.regions:
                    for child in region.blocks:
                        visit(child)

    visit(root.body.block)
    return module

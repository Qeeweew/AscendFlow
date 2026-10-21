"""Planned literals retain literal SSA despite equal dominating source values."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.dialects.physical import CoreIdOp
from ascendflow.dialects.structured import ActorDomainOp, NotificationOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.terminal_trace import trace

SOURCE = """
def literals(X:N.Tensor((16,16),N.f16)):
    with N.Kernel(4,kind="mixed"):
        resident=N.alloc_l1((16,16),N.f16)
        with N.Cube():
            N.copy(X,resident)
"""


def program():
    module = lower_native_structure(N.from_source(SOURCE)).module
    root = module.body.block.first_op
    core = next(op for op in root.body.block.ops if isinstance(op, CoreIdOp))
    actor = next(op for op in root.body.block.ops if isinstance(op, ActorDomainOp))
    return module, root, core, actor


def word(memory, event, guard=None):
    if guard is None:
        guard = ActionGuard()
    task = memory.templates[0].operation
    actions = tuple(
        SyncAction(
            task, Placement.AFTER, (Actor.AIC,), kind, Engine.MTE2, Engine.S,
            event, guard,
        )
        for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT)
    )
    materialize_notifications(memory, actions)
    return [op for op in memory.module.walk() if isinstance(op, NotificationOp)]


def literal_value(value):
    assert isinstance(value.owner, arith.ConstantOp)
    return value.owner.value.value.data


def test_cube_sid_does_not_replace_a_literal_after_dominating_constant_deduplication():
    module, root, _, actor = program()
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    root.body.block.insert_op_before(zero, root.body.block.first_op)
    # This valid pure-SSA transform leaves the Cube SID as the most recent
    # mathematical zero in its lexical scope. The SID is still a block arg.
    for op in tuple(module.walk()):
        if (isinstance(op, arith.ConstantOp) and op is not zero
                and isinstance(op.result.type, IndexType)
                and op.value.value.data == 0):
            op.result.replace_all_uses_with(zero.result)
            op.parent.erase_op(op)
    memory = collect_native_memory(module)
    assert memory.control.environment.expression(actor.body.block.args[0]) == literal(0)
    notifications = word(memory, literal(0))
    assert [literal_value(op.event) for op in notifications] == [0, 0]
    events = [c[3] for c in trace(module, {}, actor=Actor.AIC)
              if c[0] in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}]
    assert events == [0, 0]


@pytest.mark.parametrize("number", (0, 3))
def test_arithmetic_alias_cannot_replace_a_literal_id_or_guard_bound(number):
    module, _, core, _ = program()
    initial = collect_native_memory(module)
    task = initial.templates[0].operation
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    product = arith.MuliOp(core.value, zero.result)
    constant = arith.ConstantOp.from_int_and_width(number, IndexType())
    alias = arith.AddiOp(product.result, constant.result)
    for op in (zero, product, constant, alias):
        task.parent.insert_op_before(op, task)
    # Rebuild real control facts after the IR edit; the old analysis is unused.
    memory = collect_native_memory(module)
    assert memory.control.environment.expression(alias.result) == literal(number)
    guard = ActionGuard((Comparison(memory.control.group, ">=", literal(number)),))
    notifications = word(memory, literal(number), guard)
    assert [literal_value(op.event) for op in notifications] == [number, number]
    condition = notifications[0].parent_op().cond.owner
    assert isinstance(condition, arith.CmpiOp)
    assert literal_value(condition.rhs) == number


def test_dynamic_plan_expression_still_reuses_its_actual_dominating_ssa():
    module, _, core, _ = program()
    memory = collect_native_memory(module)
    notifications = word(memory, memory.control.group)
    assert all(op.event is core.value for op in notifications)

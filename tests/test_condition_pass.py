"""Condition rewrites checked by an independent finite physical interpreter."""

import itertools

import pytest
from xdsl.dialects import arith, scf

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.simplify_conditions import simplify_conditions
from ascendflow.passes.simplify_scalars import ScalarFacts, simplify_scalars
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import _Interpret

NESTED = """
def packets(q:N.int64,n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for query in N.serial(q):
            for kv in N.Pipelined(n,stages=3):
                with N.Stage(0,name="outer"):
                    for chunk in N.Pipelined(8,stages=4):
                        with N.Stage(0,name="work"):
                            N.fill(x,1.0)
"""


def trace(module, parameters):
    memory = collect_native_memory(module)
    model = _Interpret(memory, parameters, physical=True, maximum=10000)
    model.block(module.body.block.first_op.body.block, {}, 0, 0, Actor.AIV0)
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


def packet_program():
    structure = lower_native_structure(N.from_source(NESTED))
    memory = collect_native_memory(structure.module)
    q, kv, g = (loop.ordinal for loop in memory.control.loops.values())
    task = memory.templates[0].operation
    return structure.module, memory, task, q, kv, g


def action(
    task, event, tests, *, kind=SyncKind.LOCAL_WAIT, source=Engine.V, destination=Engine.MTE2
):
    return SyncAction(
        task,
        Placement.BEFORE,
        (Actor.AIV0,),
        kind,
        source,
        destination,
        literal(event) if isinstance(event, int) else event,
        ActionGuard(tuple(tests)),
    )


def test_nested_first_previous_and_slot_cases_become_one_guarded_instruction():
    module, memory, task, q, kv, g = packet_program()
    slot = expression("rem", g, literal(2))
    actions = []
    for r in range(2):
        actions += [
            action(
                task, 1 + r, (Comparison(g, ">=", literal(2)), Comparison(slot, "==", literal(r)))
            ),
        ]
    for prefix in (
        (Comparison(kv, ">=", literal(1)),),
        (Comparison(q, ">=", literal(1)), Comparison(kv, "==", literal(0))),
    ):
        actions += [
            action(task, 1 + r, (*prefix, Comparison(g, "==", literal(r)))) for r in range(2)
        ]
    materialize_notifications(memory, actions)
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.input_notifications == 6 and work.output_notifications == 1
    assert sum(isinstance(op, NotificationOp) for op in module.walk()) == 1
    for qcount, n in itertools.product(range(4), repeat=2):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == trace(parse(original), parameters)


def test_overlapping_identical_notifications_keep_both_contributions():
    module, memory, task, _, _, g = packet_program()
    materialize_notifications(
        memory,
        (
            action(task, 1, (Comparison(g, ">=", literal(1)),)),
            action(task, 1, (Comparison(g, ">=", literal(2)),)),
        ),
    )
    original = dump(module)
    simplify_scalars(module)
    simplify_conditions(module)
    module.verify()
    assert sum(isinstance(op, NotificationOp) for op in module.walk()) == 2
    assert trace(module, {"q": 2, "n": 3}) == trace(parse(original), {"q": 2, "n": 3})


def test_complementary_slot_relations_cover_the_whole_nested_iteration():
    module, memory, task, q, _, g = packet_program()
    left = expression("rem", q, literal(2))
    right = expression("rem", g, literal(2))
    materialize_notifications(
        memory,
        (
            action(task, 1, (Comparison(left, "==", right),)),
            action(task, 2, (Comparison(expression("add", left, right), "==", literal(1)),)),
        ),
    )
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.output_notifications == 1
    # The mutually exclusive equal/opposite slot choices cover every iteration;
    # the event selector remains, but no instruction-existence branch does.
    assert not any(isinstance(op, scf.IfOp) for op in module.walk())
    for qcount, n in itertools.product(range(5), range(4)):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == trace(parse(original), parameters)


def test_event_selector_uses_only_the_residual_conditions_inside_its_common_guard():
    module, memory, task, _, kv, g = packet_program()
    slot = expression("rem", kv, literal(4))
    materialize_notifications(
        memory,
        [
            action(
                task,
                event,
                (Comparison(g, "==", literal(7)), Comparison(slot, "==", literal(residue))),
            )
            for residue, event in enumerate((1, 3, 1, 4))
        ],
    )
    original = dump(module)
    simplify_scalars(module)
    simplify_conditions(module)
    module.verify()
    notifications = [op for op in module.walk() if isinstance(op, NotificationOp)]
    assert len(notifications) == 1
    definitions = set()

    def visit(value):
        if value in definitions:
            return
        definitions.add(value)
        if isinstance(value.owner, tuple(arith.Arith.operations)):
            for operand in value.owner.operands:
                visit(operand)

    visit(notifications[0].event)
    # The event's dependency graph no longer tests the common g==7 condition.
    assert not any(
        isinstance(value.owner, arith.CmpiOp)
        and any(
            isinstance(operand.owner, arith.ConstantOp) and operand.owner.value.value.data == 7
            for operand in value.owner.operands
        )
        for value in definitions
    )
    for qcount, n in itertools.product(range(3), range(7)):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == trace(parse(original), parameters)


def test_packet_entry_cannot_reuse_a_range_established_only_under_its_old_guard():
    module, memory, _, _, _, _ = packet_program()
    count = next(
        op for op in module.walk() if op.name == "npu.runtime_size" and op.sym_name.data == "n"
    )
    n = memory.control.environment.expression(count.value)
    event = expression("rem", expression("sub", n, literal(1)), literal(3))
    root = module.body.block.first_op
    materialize_notifications(
        memory,
        [
            SyncAction(
                root,
                Placement.ENTRY,
                (Actor.AIV0,),
                SyncKind.LOCAL_SET,
                Engine.V,
                Engine.MTE2,
                event,
                ActionGuard((Comparison(n, ">", literal(0)),)),
            )
        ],
    )
    original = dump(module)
    difference = next(
        op for op in module.walk() if isinstance(op, arith.AddiOp) and count.value in op.operands
    )
    facts = ScalarFacts(ranges={difference.result: (0, (1 << 63) - 2)})
    simplify_conditions(module, facts=facts)
    module.verify()
    for n in range(5):
        assert trace(module, {"q": 0, "n": n}) == trace(parse(original), {"q": 0, "n": n})


def test_contiguous_native_blocks_share_the_condition_without_crossing_an_effect():
    source = N.from_source("""
        def blocks(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial(n):
                    if i>=1:
                        N.fill(x,1.0)
                    if i>=1:
                        N.fill(x,2.0)
                    if i>=1:
                        if i>=2:
                            N.fill(x,3.0)
                    N.fill(x,4.0)
                    if i>=1:
                        N.fill(x,5.0)
    """)
    module = lower_native_structure(source).module
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.shared_blocks == 2
    assert sum(isinstance(op, scf.IfOp) for op in module.walk()) == 3
    for n in range(7):
        assert trace(module, {"n": n}) == trace(parse(original), {"n": n})


@pytest.mark.parametrize("period", [2, 3, 4, 8])
def test_untaken_prefix_and_total_bitmask_extension_preserve_unsigned_evaluation(period):
    source = N.from_source("""
        def protected(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
    """)
    module = lower_native_structure(source).module
    memory = collect_native_memory(module)
    task = memory.templates[0].operation
    n = next(v for v in memory.control.environment.facts if v == "n")
    from ascendflow.scalar import Expr

    number = Expr("argument", (n,))
    slot = expression("rem", expression("sub", number, literal(1)), literal(period))
    materialize_notifications(
        memory,
        (
            action(
                task, 1, (Comparison(number, ">", literal(0)), Comparison(slot, "==", literal(0)))
            ),
            action(
                task, 2, (Comparison(number, ">", literal(0)), Comparison(slot, "==", literal(1)))
            ),
        ),
    )
    original = dump(module)
    simplify_scalars(module)
    simplify_conditions(module)
    module.verify()
    if period == 3:
        assert any(
            isinstance(op, arith.RemUIOp) and isinstance(op.parent_op(), scf.IfOp)
            for op in module.walk()
        )
    else:
        assert not any(isinstance(op, arith.RemUIOp) for op in module.walk())
    for n in range(8):
        assert trace(module, {"n": n}) == trace(parse(original), {"n": n})

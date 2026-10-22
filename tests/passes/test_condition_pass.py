"""Condition rewrites checked by an independent finite physical interpreter."""

import itertools

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import UnregisteredOp

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.simplify_conditions import simplify_conditions
from ascendflow.passes.simplify_scalars import ScalarFacts, simplify_scalars
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import (
    EventModelError,
    check_structured_events,
)
from tests.models.terminal_trace import trace

NESTED = """
def packets(q:N.i64,n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for query in N.serial(q):
            for kv in N.Pipelined(n):
                with N.Stage(0,name="outer"):
                    for chunk in N.Pipelined(8):
                        with N.Stage(0,name="work"):
                            N.fill(x,1.0)
"""


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


@pytest.mark.parametrize("events", [(0, 0), (1, 3), (3, 1)])
def test_mutually_exclusive_notifications_share_for_any_physical_numbers(events):
    module, memory, task, _, _, g = packet_program()
    slot = expression("rem", g, literal(2))
    actions = tuple(
        action(task, event, (Comparison(slot, "==", literal(i)),)) for i, event in enumerate(events)
    )
    materialize_notifications(memory, actions)
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.output_notifications == 1
    for qcount, n in itertools.product(range(3), repeat=2):
        assert trace(module, {"q": qcount, "n": n}) == trace(parse(original), {"q": qcount, "n": n})


@pytest.mark.parametrize("kind", (SyncKind.CROSS_SET, SyncKind.CROSS_WAIT))
@pytest.mark.parametrize("events", ((0, 1), (11, 3), (7, 7)))
def test_binary_cross_ids_keep_constant_operands_and_the_actual_notification_word(kind, events):
    module, memory, task, _, _, g = packet_program()
    slot = expression("rem", g, literal(2))
    materialize_notifications(
        memory,
        tuple(action(task, event, (Comparison(slot, "==", literal(i)),), kind=kind)
              for i, event in enumerate(events)),
    )
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    notifications = [op for op in module.walk() if isinstance(op, NotificationOp)]
    assert work.output_notifications == len(notifications) == len(set(events))
    assert all(isinstance(op.event.owner, arith.ConstantOp) for op in notifications)
    for qcount, n in itertools.product(range(3), repeat=2):
        inputs = {"q": qcount, "n": n}
        assert trace(module, inputs) == trace(parse(original), inputs)


@pytest.mark.parametrize("kind", (SyncKind.CROSS_SET, SyncKind.CROSS_WAIT))
def test_binary_cross_selection_preserves_holes_and_multiple_cases_of_one_id(kind):
    module, memory, task, _, _, g = packet_program()
    slot = expression("rem", g, literal(4))
    materialize_notifications(
        memory,
        tuple(action(task, event, (Comparison(slot, "==", literal(i)),), kind=kind)
              for i, event in ((0, 3), (1, 11), (3, 3))),
    )
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.output_notifications == 2
    for qcount, n in itertools.product(range(3), repeat=2):
        inputs = {"q": qcount, "n": n}
        assert trace(module, inputs) == trace(parse(original), inputs)


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


def test_a_constant_id_covering_both_phases_keeps_its_word_and_guard_hole():
    module, memory, task, q, _, g = packet_program()
    actions = (
        action(task, 0, (Comparison(g, ">=", literal(4)),)),
        action(task, 1, (Comparison(g, "==", literal(3)),)),
        action(task, 2, (Comparison(q, ">=", literal(1)), Comparison(g, "==", literal(0)))),
        action(task, 3, (Comparison(q, ">=", literal(1)), Comparison(g, "==", literal(1)))),
    )
    materialize_notifications(memory, actions)
    original = dump(module)
    simplify_scalars(module)
    simplify_conditions(module)
    module.verify()
    for qcount, n in itertools.product(range(4), repeat=2):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == trace(parse(original), parameters)


@pytest.mark.parametrize(
    "steady,boundary", [((0, 1), (2, 3)), ((4, 5), (0, 1)), ((3, 4), (6, 7))]
)
def test_four_ids_with_a_shared_phase_preserve_boundaries_holes_and_every_notification(
    steady, boundary,
):
    module, memory, task, q, _, g = packet_program()
    slot = expression("rem", g, literal(2))
    actions = tuple(
        action(task, event, (Comparison(g, ">=", literal(2)), Comparison(slot, "==", literal(r))))
        for r, event in enumerate(steady)
    ) + tuple(
        action(task, event, (Comparison(q, ">=", literal(1)), Comparison(g, "==", literal(r))))
        for r, event in enumerate(boundary)
    )
    materialize_notifications(memory, actions)
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.output_notifications == 1
    for qcount, n in itertools.product(range(4), repeat=2):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == trace(parse(original), parameters)


@pytest.mark.parametrize(
    "kind", [SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT, SyncKind.CROSS_SET, SyncKind.CROSS_WAIT]
)
def test_overlapping_identical_notifications_keep_both_contributions(kind):
    module, memory, task, _, _, g = packet_program()
    materialize_notifications(
        memory,
        (
            action(task, 1, (Comparison(g, ">=", literal(1)),), kind=kind),
            action(task, 1, (Comparison(g, ">=", literal(2)),), kind=kind),
        ),
    )
    original = dump(module)
    simplify_scalars(module)
    simplify_conditions(module)
    module.verify()
    assert sum(isinstance(op, NotificationOp) for op in module.walk()) == 2
    assert trace(module, {"q": 2, "n": 3}) == trace(parse(original), {"q": 2, "n": 3})


def fence(task, tests=(), *, pipe=Engine.V, origin="", placement=Placement.BEFORE):
    return SyncAction(
        task,
        placement,
        (Actor.AIV0,),
        SyncKind.FENCE,
        pipe,
        pipe,
        guard=ActionGuard(tuple(tests)),
        protocols=(origin,),
    )


def without_repeated_fences(commands):
    """Compare concrete actor traces modulo adjacent identical fence stutters."""
    result = []
    for command in commands:
        if command[0] == SyncKind.FENCE.value and result and result[-1] == command:
            continue
        result.append(command)
    return tuple(result)


@pytest.mark.parametrize("guards", ["unconditional", "cover", "overlap"])
def test_same_prefix_fences_union_guards_and_keep_all_origins(guards):
    module, memory, task, q, _, g = packet_program()

    def ge(value, n):
        return Comparison(value, ">=", literal(n))

    def le(value, n):
        return Comparison(value, "<=", literal(n))

    cases = {
        "unconditional": ((), (ge(g, 1),), (ge(q, 1),)),
        "cover": ((ge(g, 2),), (le(g, 4),)),
        "overlap": ((ge(g, 1), le(g, 4)), (ge(g, 3), le(g, 6))),
    }[guards]
    materialize_notifications(
        memory, [fence(task, tests, origin=str(i)) for i, tests in enumerate(cases)]
    )
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    notifications = [op for op in module.walk() if isinstance(op, NotificationOp)]
    assert len(notifications) == 1
    assert set(notifications[0].protocol.data.split(", ")) == {str(i) for i in range(len(cases))}
    assert work.removed_fences == len(cases) - 1
    for qcount, n in itertools.product(range(4), repeat=2):
        parameters = {"q": qcount, "n": n}
        assert trace(module, parameters) == without_repeated_fences(
            trace(parse(original), parameters)
        )


@pytest.mark.parametrize("boundary", ["native", "loop", "opaque"])
def test_fences_do_not_cross_new_work_loops_or_unmodeled_effects(boundary):
    source = N.from_source("""
        def work(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                for i in N.serial(n):
                    N.fill(x,2.0)
                N.fill(x,3.0)
    """)
    module = lower_native_structure(source).module
    memory = collect_native_memory(module)
    first, _, last = (item.operation for item in memory.templates)
    if boundary == "native":
        actions = [fence(first), fence(last)]
    elif boundary == "loop":
        loop = next(op for op in module.walk() if isinstance(op, scf.ForOp))
        actions = [fence(loop, placement=Placement.ENTRY), fence(last)]
    else:
        actions = [fence(last), fence(last)]
    materialize_notifications(memory, actions)
    if boundary == "opaque":
        notifications = [op for op in module.walk() if isinstance(op, NotificationOp)]
        last.parent.insert_op_before(
            UnregisteredOp.with_name("test.opaque_effect").create(), notifications[-1]
        )
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert sum(isinstance(op, NotificationOp) for op in module.walk()) == 2
    assert work.removed_fences == 0


@pytest.mark.parametrize("middle", ["pipe", "event"])
def test_fence_merge_keeps_other_pipe_and_event_instruction_boundaries(middle):
    module, memory, task, _, _, _ = packet_program()
    between = fence(task, pipe=Engine.MTE2) if middle == "pipe" else action(task, 0, ())
    materialize_notifications(memory, [fence(task), between, fence(task)])
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.removed_fences == 0
    assert trace(module, {"q": 2, "n": 3}) == trace(parse(original), {"q": 2, "n": 3})


def test_fence_coalescing_preserves_finite_byte_dependencies_and_required_fences():
    source = N.from_source("""
        def work(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial(n):
                    N.fill(x,1.0)
                    N.mul(x,2.0,out=x)
                    if i>=1:
                        N.mul(x,3.0,out=x)
    """)
    module = lower_native_structure(source).module
    original = collect_native_memory(parse(dump(module)))
    memory = collect_native_memory(module)
    first, second, third = (item.operation for item in memory.templates)
    i = next(iter(memory.control.loops.values())).ordinal
    materialize_notifications(
        memory,
        [
            fence(first),
            fence(second),
            fence(second, (Comparison(i, ">=", literal(1)),)),
            fence(second, (Comparison(i, ">=", literal(2)),)),
            fence(third),
        ],
    )
    before = collect_native_memory(parse(dump(module)))
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    after = collect_native_memory(module)
    assert work.removed_fences == 2
    assert sum(isinstance(op, NotificationOp) for op in module.walk()) == 3
    for n in range(5):
        baseline = check_structured_events(before, {"n": n}, logical_reference=original)
        result = check_structured_events(after, {"n": n}, logical_reference=original)
        assert (result.bytes_read, result.bytes_written) == (
            baseline.bytes_read,
            baseline.bytes_written,
        )
        assert len(result.hazards) == len(baseline.hazards)
    required = next(
        op for op in module.walk() if isinstance(op, NotificationOp) and op.next_op is second
    )
    required.parent.erase_op(required)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(collect_native_memory(module), {"n": 1}, logical_reference=original)


def test_fence_union_preserves_each_aiv_existence_domain_in_the_shared_body():
    source = N.from_source("""
        def shared(n:N.i64):
            with N.Kernel(kind="mixed"):
                x=N.alloc_ub((64,))
                with N.Vector():
                    for i in N.serial(n):
                        N.fill(x,1.0)
    """)
    module = lower_native_structure(source).module
    memory = collect_native_memory(module)
    task = memory.templates[0].operation
    i = next(iter(memory.control.loops.values())).ordinal
    actions = [
        SyncAction(
            task,
            Placement.BEFORE,
            (actor,),
            SyncKind.FENCE,
            Engine.V,
            Engine.V,
            guard=ActionGuard(tests),
        )
        for actor, tests in (
            (Actor.AIV0, ()),
            (Actor.AIV0, (Comparison(i, ">=", literal(1)),)),
            (Actor.AIV1, (Comparison(i, ">=", literal(2)),)),
        )
    ]
    materialize_notifications(memory, actions)
    original = dump(module)
    simplify_scalars(module)
    work = simplify_conditions(module)
    module.verify()
    assert work.removed_fences == 2
    for n, actor in itertools.product(range(5), (Actor.AIV0, Actor.AIV1)):
        assert trace(module, {"n": n}, actor) == without_repeated_fences(
            trace(parse(original), {"n": n}, actor)
        )


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
        def blocks(n:N.i64):
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
        def protected(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
    """)
    module = lower_native_structure(source).module
    memory = collect_native_memory(module)
    task = memory.templates[0].operation
    n = next(v for v in memory.control.environment.facts if v.name == "n")
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

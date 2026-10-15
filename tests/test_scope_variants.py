"""Case filtering and sync planning retain one actual native child template."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr
from xdsl.ir import Block, Region

from xdsl_npu._physical_builder import Kernel
from xdsl_npu.ir import (
    BufferOp,
    CallOp,
    EndOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    ScopeGuardOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    TaskType,
    task_attrs,
)
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.loop_sync import _verify_leaf
from xdsl_npu.model import CompileError
from xdsl_npu.scope_predicates import PARENT_CASES
from xdsl_npu.scope_variants import (
    child_is_empty,
    filter_child,
    plan_child_variants,
    scope_operations,
    validate_child_scopes,
)


def task(uid, refs=()):
    return TaskOp.create(
        operands=refs, result_types=[TaskType()], attributes=task_attrs(uid, "fill", 0)
    )


def retained(*, once=False):
    parent = Block(arg_types=[IndexType()])
    body = Block(arg_types=[IndexType()])
    if once:
        stream = OnceOp.create(attributes={"uid": StringAttr("child")}, regions=[Region(body)])
    else:
        stream = StreamOp.create(attributes={"count": IntAttr(2)}, regions=[Region(body)])
    body.add_op(EndOp())
    call = CallOp.create(
        operands=[parent.args[0]],
        attributes={
            "uid": StringAttr("call"),
            "stage": IntAttr(0),
            "logical_phase": IntAttr(0),
            "issue_order": IntAttr(0),
        },
        regions=[Region(Block([stream, EndOp()]))],
    )
    parent.add_ops([call, EndOp()])
    return stream, call


def guarded(stream, call, kind, ops):
    return ScopeGuardOp.create(
        operands=[call.generation, stream.body.block.args[0]],
        attributes={"kind": StringAttr(kind)},
        regions=[Region(Block([*ops, EndOp()]))],
    )


def labels(values):
    return ArrayAttr(StringAttr(value) for value in values)


def channel(cases=("single", "last")):
    return LoopChannelOp.create(
        attributes={
            "sym_name": StringAttr("flag"),
            "kind": StringAttr("local"),
            "sources": labels(("aiv/producer",)),
            "destinations": labels(("aiv/consumer",)),
            "distance": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(3)]),
            "scope_cases": labels(cases),
        }
    )


def sync(stream, call, action, cases=("single", "last")):
    return LoopSyncOp.create(
        operands=[stream.body.block.args[0], call.generation],
        attributes={
            "action": StringAttr(action),
            "channel": StringAttr("flag"),
            "lower": IntAttr(0),
            "upper_gap": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "scope_cases": labels(cases),
        },
    )


@pytest.mark.parametrize("once", [False, True])
def test_four_cases_filter_actual_guards_without_mutating_native_template(once):
    stream, call = retained(once=once)
    update, first, last = (task(uid) for uid in ("update", "first", "last"))
    stream.body.block.insert_ops_before(
        [guarded(stream, call, "first", [first]), update, guarded(stream, call, "last", [last])],
        stream.body.block.last_op,
    )
    operations = tuple(scope_operations(stream))
    assert [(op.uid.data, guards) for op, guards in operations if isinstance(op, TaskOp)] == [
        ("first", ("first",)),
        ("update", ()),
        ("last", ("last",)),
    ]
    assert sum(isinstance(op, EndOp) for op, _ in operations) == 1
    expected = {
        "single": ("first", "update", "last"),
        "first": ("first", "update"),
        "middle": ("update",),
        "last": ("update", "last"),
    }
    before = str(call)
    for case in PARENT_CASES:
        selected = filter_child(stream, call, case)
        assert type(selected) is type(stream)
        assert (
            tuple(op.uid.data for op in selected.walk() if isinstance(op, TaskOp)) == expected[case]
        )
        assert not any(isinstance(op, ScopeGuardOp) for op in selected.walk())
    assert str(call) == before and update.parent is stream.body.block


def test_empty_case_is_explicit_and_unknown_case_is_rejected():
    stream, call = retained(once=True)
    stream.body.block.insert_op_before(
        guarded(stream, call, "first", [task("init")]), stream.body.block.last_op
    )
    assert child_is_empty(filter_child(stream, call, "middle"))
    assert not child_is_empty(filter_child(stream, call, "single"))
    with pytest.raises(CompileError, match="unknown parent child case"):
        filter_child(stream, call, "empty")


def test_actual_guard_clocks_are_validated_before_inactive_body_is_discarded():
    stream, call = retained()
    guard = guarded(stream, call, "last", [task("store")])
    stream.body.block.insert_op_before(guard, stream.body.block.last_op)
    guard.operands = (stream.body.block.args[0], stream.body.block.args[0])
    with pytest.raises(CompileError, match="actual parent and child clocks"):
        filter_child(stream, call, "first")


def test_escaping_conditional_ssa_cannot_become_undefined_after_filtering():
    stream, call = retained()
    producer = task("conditional")
    stream.body.block.insert_ops_before(
        [guarded(stream, call, "first", [producer]), task("outside", [producer.task])],
        stream.body.block.last_op,
    )
    with pytest.raises(CompileError, match="escapes its conditional body"):
        filter_child(stream, call, "last")


def test_selected_sync_metadata_is_removed_inside_unwrapped_native_guards():
    stream, call = retained()
    producer, consumer = task("producer"), task("consumer")
    send, wait = sync(stream, call, "set"), sync(stream, call, "wait")
    boundary = ScopeSyncOp.create(
        operands=[call.generation, stream.body.block.args[0]],
        attributes={
            "action": StringAttr("wait"),
            "channel": StringAttr("parent"),
            "position": IntAttr(0),
        },
    )
    stream.body.block.insert_ops_before(
        [channel(), producer, send, guarded(stream, call, "last", [wait, boundary, consumer])],
        stream.body.block.last_op,
    )
    selected = filter_child(stream, call, "last")
    for op in selected.walk():
        if isinstance(op, (LoopChannelOp, LoopSyncOp)):
            assert "scope_cases" not in op.attributes
        if isinstance(op, LoopSyncOp):
            assert op.parent_generation is None
            assert op.generation is selected.body.block.args[0]
    assert sum(isinstance(op, ScopeSyncOp) for op in selected.walk()) == 1
    middle = filter_child(stream, call, "middle")
    assert not any(isinstance(op, (LoopChannelOp, LoopSyncOp, ScopeSyncOp)) for op in middle.walk())
    assert tuple(op.uid.data for op in middle.walk() if isinstance(op, TaskOp)) == ("producer",)


@pytest.mark.parametrize("change", ["unknown", "empty", "unpaired", "clock"])
def test_unknown_or_unpaired_actual_sync_cases_are_rejected_before_filtering(change):
    stream, call = retained()
    declaration, send, wait = channel(), sync(stream, call, "set"), sync(stream, call, "wait")
    stream.body.block.insert_ops_before(
        [declaration, task("producer"), send, wait, task("consumer")], stream.body.block.last_op
    )
    if change == "unknown":
        send.attributes["scope_cases"] = labels(("bogus",))
    elif change == "empty":
        declaration.attributes["scope_cases"] = labels(())
    elif change == "unpaired":
        wait.attributes["scope_cases"] = labels(("single",))
    else:
        wait.operands = (wait.generation, wait.generation)
    with pytest.raises(CompileError, match="scope|scoped|child sync"):
        filter_child(stream, call, "middle")


def copy_source(*, two_consumers=False):
    k = Kernel("parent_cases", disjoint_external=True)
    source = k.buffer("X", "GM", 64, external=True, initialized=True)
    a, b = k.buffer("a", "UB", 64), k.buffer("b", "UB", 64)
    k.loop(1, symbolic=True)
    ar, br = k.view(a, 64), k.view(b, 64)
    k.copy("load", k.view(source, 64), ar)
    k.scale("scale", ar, br, 2.0)
    if two_consumers:
        k.scale("tail", ar, br, 3.0)
    module = k.finish()
    actual_module = module.clone()
    kernel = actual_module.body.block.first_op
    actual = next(op for op in kernel.body.block.ops if isinstance(op, StreamOp))
    kernel.body.block.detach_op(actual)
    placeholder, call = retained()
    call.body.block.insert_op_before(actual, placeholder)
    call.body.block.erase_op(placeholder)
    roots = {op.sym_name.data: op.ref for op in kernel.body.block.ops if isinstance(op, BufferOp)}
    return module, actual, call, roots


def normalized_case(module, stream, call, roots, case):
    clone = module.clone()
    kernel = clone.body.block.first_op
    mapper = {
        roots[op.sym_name.data]: op.ref for op in kernel.body.block.ops if isinstance(op, BufferOp)
    }
    child = filter_child(stream, call, case, value_mapper=mapper)
    if child_is_empty(child):
        return None
    original = next(op for op in kernel.body.block.ops if isinstance(op, StreamOp))
    kernel.body.block.insert_op_before(child, original)
    kernel.body.block.erase_op(original)
    return extract_loop(clone, region=child)


def test_planner_merges_equal_configurations_before_reserving_real_bank_ids():
    module, stream, call, roots = copy_source()
    native = tuple(op for op in stream.walk() if isinstance(op, TaskOp))
    loops = {case: normalized_case(module, stream, call, roots, case) for case in PARENT_CASES}
    actor = next(iter(loops["single"].actors))
    banks = {(actor, "MTE2", "V"): 7}
    result = plan_child_variants(stream, call, loops, banks=banks)
    assert result.cases == PARENT_CASES and not result.empty_cases
    assert result.channel_count == 1 and result.sync_count == 2
    assert banks[actor, "MTE2", "V"] == 8
    assert tuple(op for op in stream.walk() if isinstance(op, TaskOp)) == native
    declaration = next(op for op in stream.walk() if isinstance(op, LoopChannelOp))
    assert tuple(value.data for value in declaration.scope_cases) == PARENT_CASES
    assert tuple(value.data for value in declaration.event_ids) == (7,)
    validate_child_scopes(stream, call)
    for case in PARENT_CASES:
        loop = normalized_case(module, stream, call, roots, case)
        _verify_leaf(loop)


def wrap_native(stream, call, uid, kind):
    native = next(op for op in stream.walk() if isinstance(op, TaskOp) and op.uid.data == uid)
    wrapper = guarded(stream, call, kind, [])
    native.parent.insert_op_before(wrapper, native)
    native.parent.detach_op(native)
    wrapper.body.block.insert_op_before(native, wrapper.body.block.last_op)


def test_different_case_configurations_do_not_share_ids_and_budget_failure_is_atomic():
    module, stream, call, roots = copy_source(two_consumers=True)
    wrap_native(stream, call, "scale", "first")
    wrap_native(stream, call, "tail", "last")
    loops = {case: normalized_case(module, stream, call, roots, case) for case in PARENT_CASES}
    actor = next(iter(loops["single"].actors))
    banks = {(actor, "MTE2", "V"): 7}
    before = str(call)
    with pytest.raises(CompileError, match="parent-case event bank exhausted"):
        plan_child_variants(stream, call, loops, banks=banks)
    assert str(call) == before and banks == {(actor, "MTE2", "V"): 7}
    loops = {case: normalized_case(module, stream, call, roots, case) for case in PARENT_CASES}
    banks.clear()
    result = plan_child_variants(stream, call, loops, banks=banks)
    assert result.channel_count >= 2
    ids = [
        value.data
        for op in stream.walk()
        if isinstance(op, LoopChannelOp)
        for value in op.event_ids
    ]
    assert len(ids) == len(set(ids))
    validate_child_scopes(stream, call)
    for case in PARENT_CASES:
        _verify_leaf(normalized_case(module, stream, call, roots, case))


def test_empty_case_plan_is_explicit_and_cannot_hide_an_active_native():
    module, stream, call, roots = copy_source()
    for uid in ("load", "scale"):
        wrap_native(stream, call, uid, "first")
    loops = {case: normalized_case(module, stream, call, roots, case) for case in PARENT_CASES}
    result = plan_child_variants(stream, call, loops)
    assert result.cases == ("single", "first") and result.empty_cases == ("middle", "last")
    module, stream, call, roots = copy_source()
    with pytest.raises(CompileError, match="nonempty actual child case"):
        plan_child_variants(stream, call, {"single": None})

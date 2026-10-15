"""Independent finite execution of actual guarded source/native protocols."""

import numpy as np
import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from tests.fixtures.guarded import native_guarded_recurrence
from tests.models.scoped_local import execute, finite_check
from xdsl_npu import compile
from xdsl_npu.ir import (
    CallOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    ScopeChannelOp,
    ScopeGuardOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
)
from xdsl_npu.scope_lowering import lower_scopes

COUNTS = (0, 1, 2, 3, 7)
CASES = ("single", "first", "middle", "last")
TAIL = ("single", "last")


def _labels(values):
    return ArrayAttr(StringAttr(value) for value in values)


def _child(call):
    return next(op for op in call.body.block.ops if isinstance(op, (OnceOp, StreamOp)))


def _loop_sync(call, task, action, channel="", *, cases=CASES):
    child = _child(call)
    op = LoopSyncOp.create(
        operands=[child.body.block.args[0], call.generation],
        attributes={
            "action": StringAttr(action),
            "channel": StringAttr(channel),
            "lower": IntAttr(0),
            "upper_gap": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "scope_cases": _labels(cases),
        },
    )
    if action == "set":
        task.parent.insert_op_after(op, task)
    else:
        task.parent.insert_op_before(op, task)
    return op


def _ready(call, source, destination, name, *, cases=CASES):
    channel = LoopChannelOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "kind": StringAttr("local"),
            "sources": _labels(("aiv0/" + source.uid.data,)),
            "destinations": _labels(("aiv0/" + destination.uid.data,)),
            "distance": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(0), IntAttr(1)]),
            "scope_cases": _labels(cases),
        }
    )
    child = _child(call)
    child.body.block.insert_op_before(channel, child.body.block.first_op)
    _loop_sync(call, source, "set", name, cases=cases)
    _loop_sync(call, destination, "wait", name, cases=cases)


def _reuse(module, call, source, destination, position):
    """Real per-row reverse handshake across parent generations; no child drain."""
    name = f"return_x{position}"
    channel = ScopeChannelOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "kind": StringAttr("local"),
            "sources": _labels((f"aiv0/{call.uid.data}/{source.uid.data}/{position}",)),
            "destinations": _labels((f"aiv0/{call.uid.data}/{destination.uid.data}/{position}",)),
            "distance": IntAttr(1),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(position)]),
        }
    )
    kernel = module.body.block.first_op
    parent = call.parent_op()
    kernel.body.block.insert_op_before(channel, parent)
    child = _child(call)
    for task, action in ((source, "set"), (destination, "wait")):
        op = ScopeSyncOp.create(
            operands=[call.generation, child.body.block.args[0]],
            attributes={
                "action": StringAttr(action),
                "channel": StringAttr(name),
                "position": IntAttr(position),
            },
        )
        if action == "set":
            task.parent.insert_op_after(op, task)
        else:
            task.parent.insert_op_before(op, task)


@pytest.fixture(scope="module")
def manual():
    """Actual lowered source, with independently written physical synchronization."""
    module = lower_scopes(native_guarded_recurrence.to_ir().body.block.first_op)
    calls = [op for op in module.walk() if isinstance(op, CallOp)]
    for call in calls:
        for task in tuple(op for op in call.walk() if isinstance(op, TaskOp)):
            if task.native.data not in {"copy", "copy_rows"}:
                guard = task.parent_op()
                cases = ("single", guard.kind.data) if isinstance(guard, ScopeGuardOp) else CASES
                _loop_sync(call, task, "barrier", cases=cases)
    call = calls[1]
    tasks = {op.native.data: op for op in call.walk() if isinstance(op, TaskOp)}
    load, store = [op for op in call.walk() if isinstance(op, TaskOp) and op.native.data == "copy"]
    _ready(call, load, tasks["add"], "input_ready")
    _ready(call, tasks["convert_rn"], store, "tail_ready", cases=TAIL)
    for position in range(2):
        _reuse(module, call, tasks["add"], load, position)
    module.verify()
    return module


@pytest.fixture(scope="module")
def compiled():
    return compile(native_guarded_recurrence, timeout_s=40)


def _numerical(plan, count):
    rng = np.random.default_rng(901 + count)
    x = rng.normal(size=(count, 2, 8, 64)).astype(np.float32)
    sentinel = np.full((2, 8, 64), -5.5, np.float16)
    checked, values = execute(plan, count, {"X": x, "O": sentinel})
    expected = np.zeros((2, 8, 64), np.float32)
    for tile in x:
        expected = np.multiply(
            np.add(expected, tile, dtype=np.float32), np.float32(0.5), dtype=np.float32
        )
    np.testing.assert_array_equal(values["O"], expected.astype(np.float16) if count else sentinel)
    np.testing.assert_array_equal(sentinel, np.full_like(sentinel, -5.5))
    assert len(checked.issued_tasks) == (8 * count + 7 if count else 0)
    if count:
        assert sum("_fill" in name for *_, name in checked.issued_tasks) == 3
        assert sum("_cast" in name for *_, name in checked.issued_tasks) == 2
        assert {pg for pg, _, _, name in checked.issued_tasks if "_cast" in name} == {count - 1}
    else:
        assert checked.command_count == checked.bytes_read == checked.bytes_written == 0
    return checked


@pytest.mark.parametrize("count", COUNTS)
def test_manual_actual_guarded_recurrence_has_correct_values_and_reader_retirement(manual, count):
    _numerical(manual, count)


@pytest.mark.parametrize("count", COUNTS)
def test_public_compiled_artifact_has_correct_values_and_empty_output_sentinel(compiled, count):
    _numerical(compiled.plan, count)


def test_missing_actual_factor_first_initializer_is_an_undefined_read(manual):
    plan = manual.clone()
    task = next(
        op
        for op in plan.walk()
        if isinstance(op, TaskOp) and op.uid.data.endswith("pipeline_2_fill")
    )
    barrier = task.prev_op
    assert isinstance(barrier, LoopSyncOp) and barrier.action.data == "barrier"
    barrier.parent.erase_op(barrier)
    task.parent.erase_op(task)
    with pytest.raises(AssertionError, match="undefined actual read"):
        finite_check(plan, 3)


@pytest.mark.parametrize("clock", ("parent", "child", "sync_parent"))
def test_actual_clock_identity_is_checked_even_for_empty_parent(manual, clock):
    plan = manual.clone()
    call = next(op for op in plan.walk() if isinstance(op, CallOp))
    child = _child(call)
    if clock == "sync_parent":
        op = next(op for op in call.walk() if isinstance(op, LoopSyncOp))
        op.operands = (op.generation, child.body.block.args[0])
    else:
        op = next(op for op in call.walk() if isinstance(op, ScopeGuardOp))
        op.operands = (
            child.body.block.args[0] if clock == "parent" else call.generation,
            call.generation if clock == "child" else child.body.block.args[0],
        )
    with pytest.raises(AssertionError, match="different .* clock"):
        finite_check(plan, 0)


def test_missing_tail_wait_is_not_hidden_by_inactive_non_tail_parents(manual):
    plan = manual.clone()
    wait = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopSyncOp)
        and op.channel.data == "tail_ready"
        and op.action.data == "wait"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(AssertionError, match="unmatched physical Set/Wait"):
        finite_check(plan, 3)


def test_actual_channel_and_sync_scope_cases_must_agree(manual):
    plan = manual.clone()
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopChannelOp) and op.sym_name.data == "tail_ready"
    )
    channel.attributes["scope_cases"] = _labels(CASES)
    with pytest.raises(AssertionError, match="scope_cases mismatch"):
        finite_check(plan, 0)


def test_matching_wrong_scope_cases_cannot_hide_tail_memory_readiness(manual):
    plan = manual.clone()
    for op in plan.walk():
        if (isinstance(op, LoopChannelOp) and op.sym_name.data == "tail_ready") or (
            isinstance(op, LoopSyncOp) and op.channel.data == "tail_ready"
        ):
            op.attributes["scope_cases"] = _labels(("single", "first"))
    # Declaration and both actual commands agree, yet none executes in the
    # last parent. Native cast/store remain active and still need their RAW.
    with pytest.raises(AssertionError, match="unordered RAW"):
        finite_check(plan, 3)

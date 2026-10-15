"""Independent small-count execution of retained Call/Capture/Set/Wait IR.

These are CPU tests of actual device instructions, with no device launch and
no calls into production completion, summary, or event-allocation proofs.
"""

from collections import defaultdict

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr, f32
from xdsl.ir import Block, Region

from tests.fixtures.nested import nested_copy
from tests.models.scoped_local import finite_check
from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.ir import (
    BufferOp,
    CallOp,
    CaptureOp,
    EndOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    ScopeChannelOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    TaskType,
    ViewOp,
    task_attrs,
)

COUNTS = (0, 1, 2, 3, 7)


class Physical:
    """Hand-written physical IR: no planner or synchronization insertion pass."""

    def __init__(self, *, initialized=False):
        self.k = Kernel("manual_parent", disjoint_external=True)
        self.n = self.k.runtime_size("n")
        self.roots = {
            "X": self.k.buffer("X", "GM", 448, external=True, initialized=True),
            "O": self.k.buffer("O", "GM", 448, external=True),
            "a": self.k.buffer("a", "UB", 64, initialized=initialized),
            "b": self.k.buffer("b", "UB", 64),
        }
        self.k.loop(self.n, symbolic=True)
        self.k.task_block.add_op(EndOp())
        self.k.block.add_op(EndOp())
        self.plan = self.k.module

    def child(self, uid, *, count=1, stage=0, order=0, phase=0, once=False):
        return Child(self, uid, count, stage, order, phase, once)

    def channel(self, name, source, destination, *, ids=(0,), distance=0, positions=(0, 0)):
        src, src_task = source
        dst, dst_task = destination
        channel = ScopeChannelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "kind": StringAttr("local"),
                "sources": ArrayAttr(
                    [StringAttr(f"aiv0/{src.uid}/{src_task.uid.data}/{positions[0]}")]
                ),
                "destinations": ArrayAttr(
                    [StringAttr(f"aiv0/{dst.uid}/{dst_task.uid.data}/{positions[1]}")]
                ),
                "distance": IntAttr(distance),
                "period": IntAttr(1),
                "residues": ArrayAttr([IntAttr(0)]),
                "event_ids": ArrayAttr(IntAttr(event) for event in ids),
            }
        )
        self.k.block.insert_op_before(channel, self.k.stream)
        for child, task, action, position in (
            (src, src_task, "set", positions[0]),
            (dst, dst_task, "wait", positions[1]),
        ):
            sync = ScopeSyncOp.create(
                operands=[child.call.generation, child.block.args[0]],
                attributes={
                    "action": StringAttr(action),
                    "channel": StringAttr(name),
                    "position": IntAttr(position),
                },
            )
            if action == "set":
                child.block.insert_op_after(sync, task)
            else:
                child.block.insert_op_before(sync, task)
        return channel


class Child:
    def __init__(self, physical, uid, count, stage, order, phase, once):
        self.uid = uid
        self.block = Block(arg_types=[IndexType()])
        self.block.add_op(EndOp())
        self.stream = (
            OnceOp.create(attributes={"uid": StringAttr(uid)}, regions=[Region(self.block)])
            if once
            else StreamOp.create(
                attributes={"uid": StringAttr(uid), "count": IntAttr(count)},
                regions=[Region(self.block)],
            )
        )
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        width = arith.ConstantOp.from_int_and_width(64, IndexType())
        parent = physical.k.task_block.args[0]
        offset = arith.MuliOp(parent, width.result)
        captures = []
        self.captures = {}
        for name, root in physical.roots.items():
            capture = CaptureOp.create(
                operands=[root, offset.result if name in {"X", "O"} else zero.result],
                result_types=[root.type],
                attributes={"sym_name": StringAttr(uid + "_" + name), "nbytes": IntAttr(64)},
            )
            captures.append(capture)
            self.captures[name] = capture.ref
        self.call = CallOp.create(
            operands=[parent],
            attributes={
                "uid": StringAttr(uid),
                "stage": IntAttr(stage),
                "logical_phase": IntAttr(phase),
                "issue_order": IntAttr(order),
            },
            regions=[Region(Block([zero, width, offset, *captures, self.stream, EndOp()]))],
        )
        physical.k.task_block.insert_op_before(self.call, physical.k.task_block.last_op)
        self.task_count = 0

    def view(self, name, width=32, *, offset=0, stride=0, rows=1, row_stride=0):
        source = self.captures[name]
        view = ViewOp.create(
            operands=[source, self.block.args[0]],
            result_types=[source.type],
            attributes={
                "offset": IntAttr(offset),
                "stride": IntAttr(stride),
                "modulus": IntAttr(0),
                "divisor": IntAttr(1),
                "nbytes": IntAttr(width),
                "tail_nbytes": IntAttr(0),
                "rows": IntAttr(rows),
                "row_stride": IntAttr(row_stride),
            },
        )
        self.block.insert_op_before(view, self.block.last_op)
        return view.ref

    def task(self, uid, native, *refs, stage=0, order=None, phase=None):
        attrs = task_attrs(uid, native, stage)
        attrs.update(
            issue_order=IntAttr(self.task_count if order is None else order),
            logical_phase=IntAttr(self.task_count if phase is None else phase),
        )
        task = TaskOp.create(
            operands=refs,
            result_types=[TaskType(), f32] if native == "scalar_load" else [TaskType()],
            attributes=attrs,
        )
        self.block.insert_op_before(task, self.block.last_op)
        self.task_count += 1
        return task

    def ready(self, source, destination, *, ids=(0, 1)):
        name = self.uid + "_ready"
        channel = LoopChannelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "kind": StringAttr("local"),
                "sources": ArrayAttr([StringAttr("aiv0/" + source.uid.data)]),
                "destinations": ArrayAttr([StringAttr("aiv0/" + destination.uid.data)]),
                "distance": IntAttr(0),
                "period": IntAttr(1),
                "residues": ArrayAttr([IntAttr(0)]),
                "event_ids": ArrayAttr(IntAttr(event) for event in ids),
            }
        )
        self.block.insert_op_before(channel, self.block.first_op)
        for task, action in ((source, "set"), (destination, "wait")):
            sync = LoopSyncOp.create(
                operands=[self.block.args[0]],
                attributes={
                    "action": StringAttr(action),
                    "channel": StringAttr(name),
                    "lower": IntAttr(0),
                    "upper_gap": IntAttr(0),
                    "period": IntAttr(1),
                    "residues": ArrayAttr([IntAttr(0)]),
                },
            )
            if action == "set":
                self.block.insert_op_after(sync, task)
            else:
                self.block.insert_op_before(sync, task)
        return channel


@pytest.fixture(scope="module")
def copied():
    return compile(nested_copy).plan


@pytest.mark.parametrize("n", COUNTS)
def test_compiled_nested_copy_actual_two_clocks_and_every_byte(copied, n):
    result = finite_check(copied, n)
    assert len(result.issued_tasks) == 16 * n
    assert result.bytes_read == result.bytes_written == 512 * n
    assert result.command_count == len(result.issued_tasks) + len(result.events)
    assert {row[0] for row in result.issued_tasks} == set(range(n))
    assert {row[2] for row in result.issued_tasks} == ({0, 1, 2, 3} if n else set())


def test_compiled_nested_copy_uses_actual_shared_parent_pools_and_eight_ids(copied):
    calls = [op for op in copied.walk() if isinstance(op, CallOp)]
    tasks = [op for op in copied.walk() if isinstance(op, TaskOp)]
    buffers = [op for op in copied.walk() if isinstance(op, BufferOp)]
    assert len(calls) == 2 and len(tasks) == 4
    assert [op.nbytes.data for op in buffers if op.ref.type.bank.data == "UB"] == [128]
    ready, reverse = [], []
    for call in calls:
        channels = [op for op in call.walk() if isinstance(op, LoopChannelOp)]
        assert sorted(op.distance.data for op in channels) == [0, 2]
        for channel in channels:
            ids = tuple(v.data for v in channel.event_ids)
            (ready if channel.distance.data == 0 else reverse).append(ids)
    assert list(map(len, ready)) == [4, 4]
    assert list(map(len, reverse)) == [2, 2]
    parent = {
        op.sym_name.data: tuple(v.data for v in op.event_ids)
        for op in copied.walk()
        if isinstance(op, ScopeChannelOp)
    }
    assert parent.keys() == {"scope_c0", "scope_c1", "scope_c2", "scope_c3"}
    assert parent["scope_c0"] == parent["scope_c2"]
    assert parent["scope_c1"] == parent["scope_c3"]
    assert all(len(ids) == 2 for ids in parent.values())
    reverse_sets = [set(ids) for ids in (*reverse, parent["scope_c0"], parent["scope_c1"])]
    assert sum(map(len, reverse_sets)) == len(set.union(*reverse_sets)) == 8
    actual = defaultdict(set)
    for event in finite_check(copied, 7).events:
        actual[event.resource[:2]].add(event.resource[2])
    assert actual == {("MTE2", "MTE3"): set(range(8)), ("MTE3", "MTE2"): set(range(8))}


@pytest.mark.parametrize("child_index", (0, 1))
def test_child_ready_width_two_is_locally_safe_but_resets_before_old_parent_wait(
    copied, child_index
):
    plan = copied.clone()
    ready = [op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data == 0][
        child_index
    ]
    ready.attributes["event_ids"] = ArrayAttr(ready.event_ids.data[:2])
    # A single invocation passes. It is reuse across actual parent calls that
    # requires the four-ID choice; no drain is invented at child return.
    finite_check(plan, 1)
    for n in (2, 3, 7):
        with pytest.raises(AssertionError, match="binary event overwrite"):
            finite_check(plan, n)


@pytest.mark.parametrize("name", ("scope_c0", "scope_c1", "scope_c2", "scope_c3"))
def test_parent_shared_pool_wrong_rotation_pairs_a_different_notification(copied, name):
    plan = copied.clone()
    channel = next(
        op for op in plan.walk() if isinstance(op, ScopeChannelOp) and op.sym_name.data == name
    )
    channel.attributes["event_ids"] = ArrayAttr(reversed(channel.event_ids.data))
    # Both Set and Wait still use the same tuple, so their counts match. The
    # physical ID now crosses logical messages from the other shared member.
    with pytest.raises(AssertionError, match="different logical notification"):
        finite_check(plan, 7)


@pytest.mark.parametrize("wait_index", range(8))
def test_every_actual_child_or_parent_wait_is_required(copied, wait_index):
    plan = copied.clone()
    waits = [
        op
        for op in plan.walk()
        if isinstance(op, (LoopSyncOp, ScopeSyncOp)) and op.action.data == "wait"
    ]
    assert len(waits) == 8
    wait = waits[wait_index]
    wait.parent.erase_op(wait)
    with pytest.raises(AssertionError, match="unmatched physical Set/Wait"):
        finite_check(plan, 7)


def scalar_child(*, ids=(0, 1), read_first=False):
    physical = Physical()
    child = physical.child("body", count=2)
    if read_first:
        read = child.task("read", "scalar_load", child.view("a", 4, stride=32), stage=0)
        load = child.task("load", "copy", child.view("X", stride=32), child.view("a", stride=32))
    else:
        load = child.task("load", "copy", child.view("X", stride=32), child.view("a", stride=32))
        read = child.task("read", "scalar_load", child.view("a", 4, stride=32), stage=1)
    child.ready(load, read, ids=ids)
    return physical.plan


@pytest.mark.parametrize("n", COUNTS)
def test_manual_scalar_gate_preserves_event_consumption_between_parent_calls(n):
    result = finite_check(scalar_child(), n)
    assert len(result.issued_tasks) == 4 * n
    assert result.bytes_read == 72 * n and result.bytes_written == 64 * n


def test_manual_set_twice_without_consumption_is_a_binary_event_overwrite():
    with pytest.raises(AssertionError, match="binary event overwrite"):
        finite_check(scalar_child(ids=(0,)), 1)


def test_manual_scalar_wait_before_its_producer_is_a_real_command_deadlock():
    with pytest.raises(AssertionError, match="actual command deadlock"):
        finite_check(scalar_child(read_first=True), 1)


@pytest.mark.parametrize("n", COUNTS)
def test_actual_loop_pipe_barrier_has_no_event_channel(n):
    physical = Physical()
    child = physical.child("body", count=2)
    write = child.task("write", "fill", child.view("a"))
    barrier = LoopSyncOp.create(
        operands=[child.block.args[0]],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr(""),
            "lower": IntAttr(0),
            "upper_gap": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
        },
    )
    child.block.insert_op_before(barrier, write)
    result = finite_check(physical.plan, n)
    assert not result.events and result.bytes_written == 64 * n
    if n:
        barrier.parent.erase_op(barrier)
        with pytest.raises(AssertionError, match="unordered WAW"):
            finite_check(physical.plan, n)


def test_scope_barrier_period_uses_actual_destination_parent_residues():
    physical = Physical()
    child = physical.child("body")
    write = child.task("write", "fill", child.view("a"))
    channel = ScopeChannelOp.create(
        attributes={
            "sym_name": StringAttr("fence"),
            "kind": StringAttr("barrier"),
            "sources": ArrayAttr([]),
            "destinations": ArrayAttr([StringAttr("aiv0/body/write/0")]),
            "distance": IntAttr(1),
            "period": IntAttr(2),
            "residues": ArrayAttr([IntAttr(1)]),
            "event_ids": ArrayAttr([]),
        }
    )
    physical.k.block.insert_op_before(channel, physical.k.stream)
    barrier = ScopeSyncOp.create(
        operands=[child.call.generation, child.block.args[0]],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr("fence"),
            "position": IntAttr(0),
        },
    )
    child.block.insert_op_before(barrier, write)
    finite_check(physical.plan, 2)
    channel.attributes["residues"] = ArrayAttr([IntAttr(0)])
    with pytest.raises(AssertionError, match="unordered WAW"):
        finite_check(physical.plan, 2)


@pytest.mark.parametrize("native", ("mul_scalar", "unknown_native", "mad16"))
def test_unsupported_native_is_explicit_even_when_the_parent_is_empty(native):
    physical = Physical()
    child = physical.child("body")
    child.task("unknown", native, child.view("a"))
    with pytest.raises(AssertionError, match="unsupported actual native: " + native):
        finite_check(physical.plan, 0)


def hazard_pair(kind, *, synchronize):
    physical = Physical(initialized=kind == "WAR")
    first = physical.child("first", phase=0)
    second = physical.child("second", phase=1, order=1)
    if kind == "RAW":
        old = first.task("load", "copy", first.view("X"), first.view("a"))
        new = second.task("scale", "muls", second.view("a"), second.view("b"))
    elif kind == "WAR":
        old = first.task("store", "copy", first.view("a"), first.view("O"))
        new = second.task("fill", "fill", second.view("a"))
    else:
        old = first.task("load", "copy", first.view("X"), first.view("a"))
        new = second.task("fill", "fill", second.view("a"))
    if synchronize:
        physical.channel("handoff", (first, old), (second, new))
    return physical.plan


@pytest.mark.parametrize("kind", ("RAW", "WAR", "WAW"))
def test_manual_native_scope_handoff_orders_each_byte_hazard(kind):
    finite_check(hazard_pair(kind, synchronize=True), 1)
    with pytest.raises(AssertionError, match="unordered " + kind):
        finite_check(hazard_pair(kind, synchronize=False), 1)


@pytest.mark.parametrize("n", COUNTS)
def test_actual_once_children_obey_parent_fill_and_drain_and_lexical_issue_ties(n):
    physical = Physical(initialized=True)
    later = physical.child("later", once=True, stage=2, phase=1)
    later.task("read_later", "scalar_load", later.view("a", 4))
    earlier = physical.child("earlier", once=True, stage=0, phase=0)
    earlier.task("read_earlier", "scalar_load", earlier.view("a", 4, offset=32))
    result = finite_check(physical.plan, n)
    expected = [
        (step - stage, name, 0, "read_" + name)
        for step in range(n + 2)
        for name, stage in (("later", 2), ("earlier", 0))
        if 0 <= step - stage < n
    ]
    assert list(result.issued_tasks) == expected


def test_actual_pitched_rows_use_bytes_instead_of_bounding_box_overlap():
    physical = Physical()
    child = physical.child("pitched")
    child.task("even_rows", "fill", child.view("a", 4, rows=2, row_stride=16))
    odd = child.task("odd_rows", "fill", child.view("a", 4, offset=8, rows=2, row_stride=16))
    result = finite_check(physical.plan, 1)
    assert result.bytes_written == 16
    odd.refs[0].owner.attributes["offset"] = IntAttr(3)
    with pytest.raises(AssertionError, match="unordered WAW"):
        finite_check(physical.plan, 1)


def test_actual_capture_bounds_are_evaluated_in_the_parent_clock(copied):
    plan = copied.clone()
    capture = next(
        op
        for op in plan.walk()
        if isinstance(op, CaptureOp) and op.source.owner.sym_name.data == "X"
    )
    factor = capture.offset.owner.rhs.owner
    assert isinstance(factor, arith.ConstantOp) and factor.value.value.data == 128
    factor.properties["value"] = arith.ConstantOp.from_int_and_width(256, IndexType()).value
    with pytest.raises(AssertionError, match="actual capture out of bounds"):
        finite_check(plan, 3)

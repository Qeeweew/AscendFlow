"""Reconstruct proofs from corrupted final IR; never trust planner state."""

import pytest
from xdsl.dialects import scf
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from examples.state.grouped import grouped_state_sum
from tests.models.local_sequence import finite_check, physical_regions
from xdsl_npu import compile, verify_plan
from xdsl_npu.ir import BufferOp, FrameOp, RegionChannelOp, RegionSyncOp, TaskOp, ViewOp
from xdsl_npu.model import CompileError


def test_preceding_vector_barrier_cannot_retire_the_task_that_follows_it(frame_artifact):
    plan = frame_artifact.plan
    channel = next(
        op for op in plan.walk() if isinstance(op, RegionChannelOp) and op.event_id.data == -1
    )
    for op in list(plan.walk()):
        if isinstance(op, RegionSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing sequence (RAW|WAW)"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 2)


@pytest.mark.parametrize("which", [0, 1])
def test_missing_real_boundary_endpoint_is_rejected(state_artifact, which):
    plan = state_artifact.plan
    declaration = [o for o in plan.walk() if isinstance(o, RegionChannelOp)][which]
    wait = next(
        o
        for o in plan.walk()
        if isinstance(o, RegionSyncOp)
        and o.channel == declaration.sym_name
        and o.action.data == "wait"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(CompileError, match="unpaired|wrong native endpoint"):
        verify_plan(plan)


def test_removed_initialization_cannot_be_replaced_by_live_in_assumptions(state_artifact):
    plan = state_artifact.plan
    region = physical_regions(plan)[0]
    region.parent.erase_op(region)
    with pytest.raises(CompileError, match="undefined sequence entry read"):
        verify_plan(plan)


def test_final_structure_check_includes_late_regions(state_artifact):
    plan = state_artifact.plan
    task = next(o for o in physical_regions(plan)[-1].walk() if isinstance(o, TaskOp))
    task.attributes["issue_order"] = StringAttr("invalid")
    with pytest.raises(VerifyException, match="issue_order"):
        verify_plan(plan)


def test_wrong_clock_and_resource_collision_are_rejected(state_artifact):
    plan = state_artifact.plan
    wait = next(o for o in plan.walk() if isinstance(o, RegionSyncOp) and o.action.data == "wait")
    wait.operands = [physical_regions(plan)[1].body.block.args[0]]
    with pytest.raises(CompileError, match="wrong resource or coordinate"):
        verify_plan(plan)
    plan = state_artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp))
    channel.attributes["event_id"] = IntAttr(8)
    with pytest.raises(CompileError, match="overlap/exceed"):
        verify_plan(plan)


def test_zero_trip_guard_cannot_publish_a_nonexistent_final_writer(state_artifact):
    plan = state_artifact.plan
    c = next(
        o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.source_region.data == "r1"
    )
    c.attributes["domains"] = ArrayAttr([ArrayAttr(IntAttr(v) for v in (1, 0, 0, 0))])
    with pytest.raises(CompileError, match="no actual producer/consumer"):
        verify_plan(plan)


def test_bounded_predicate_is_checked_and_emitted_without_expanding_counts(state_artifact):
    from xdsl_npu import emit_asc

    plan = state_artifact.plan
    c = next(
        o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.source_region.data == "r1"
    )
    c.attributes["domains"] = ArrayAttr(
        [ArrayAttr(IntAttr(v) for v in row) for row in ((1, 0, -1, 1, 7), (1, 0, -1, 8))]
    )
    verify_plan(plan)
    assert "npu_sequence_count <= 7" in emit_asc(plan)
    for count in (0, 1, 7, 8):
        finite_check(plan, count)
    c.attributes["domains"] = ArrayAttr([c.domains.data[0]])
    with pytest.raises(CompileError, match="missing sequence RAW"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 8)


def test_missing_call_return_is_not_hidden_by_leaf_verification(grouped_artifact):
    plan = grouped_artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    for op in list(plan.walk()):
        if isinstance(op, RegionSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing sequence invocation"):
        verify_plan(plan)


def test_carried_guard_cannot_be_reinterpreted_as_forward_same_call(grouped_artifact):
    plan = grouped_artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    channel.attributes["carried"] = IntAttr(0)
    with pytest.raises(CompileError, match="lexical dispatch order"):
        verify_plan(plan)


def test_real_parent_frame_coordinate_is_required(grouped_artifact):
    plan = grouped_artifact.plan
    parent = next(o for o in plan.walk() if isinstance(o, scf.ForOp))
    frame = next(o for o in plan.walk() if isinstance(o, FrameOp))
    frame.operands = (frame.source, parent.lb, frame.extent)
    with pytest.raises(CompileError, match="actual root, invocation coordinate"):
        verify_plan(plan)


def test_whole_last_reader_not_compute_completion_protects_state_reuse():
    from dataclasses import replace

    from xdsl_npu.ir import TaskOp

    source = grouped_state_sum.source.replace(
        "(GROUPS, outputs, tiles, B)", "(GROUPS, outputs, 1, B)"
    )
    source = source.replace("T.Pipelined(tiles,", "T.Pipelined(1,")
    result = compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
    plan = result.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    add = next(o for o in plan.walk() if isinstance(o, TaskOp) and o.native.data == "add")
    region = add.parent_op()
    send = next(
        o
        for o in plan.walk()
        if isinstance(o, RegionSyncOp) and o.channel == channel.sym_name and o.action.data == "set"
    )
    send.parent.detach_op(send)
    add.parent.insert_op_after(send, add)
    send.operands = (region.body.block.args[0],)
    channel.attributes.update(
        source_region=region.uid, source_task=add.uid, source_end=IntAttr(1), event_id=IntAttr(2)
    )
    with pytest.raises(CompileError, match="missing sequence invocation WAR"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 0, invocations=2)


def test_missing_gm_publish_path_is_rejected_even_when_child_is_empty(frame_artifact):
    plan = frame_artifact.plan
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, RegionChannelOp)
        and op.source_region.data == "r0"
        and op.destination_region.data == "r2"
        and op.destination_task.data.endswith("_0_copy")
    )
    for op in list(plan.walk()):
        if isinstance(op, RegionSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing sequence (RAW|WAR)"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 0, invocations=2)


def test_missing_ub_input_retirement_does_not_get_hidden_by_a_frame_ready(frame_artifact):
    plan = frame_artifact.plan
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, RegionChannelOp)
        and op.source_region.data == "r1"
        and op.destination_region.data == "r2"
        and op.source_offset.data == 1
    )
    for op in list(plan.walk()):
        if isinstance(op, RegionSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    # The removed release also breaks the positive-count snapshot-ready path.
    with pytest.raises(CompileError, match="missing sequence (RAW.*Output|W(AW|AR).*inputs)"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 2, invocations=2)


@pytest.mark.parametrize("buffer", ["Initial", "X"])
def test_zero_group_validity_does_not_hide_a_later_group_out_of_bounds(shared_artifact, buffer):
    plan = shared_artifact.plan
    view = next(
        o
        for o in plan.walk()
        if isinstance(o, ViewOp)
        and isinstance(o.source.owner, BufferOp)
        and o.source.owner.sym_name.data == buffer
    )
    view.attributes["offset"] = IntAttr(view.offset.data + 16 * 4)
    # The source parser never sees this final-IR mutation. Group zero still fits;
    # group three accesses past the real GM allocation, independently enumerated.
    finite_check(plan, 2, invocations=1, group=0)
    with pytest.raises(CompileError, match="bound|extent|allocation"):
        verify_plan(plan)
    with pytest.raises(AssertionError, match="actual native byte out of bounds"):
        finite_check(plan, 2, invocations=1, group=3)


def test_initialized_unframed_gm_is_not_permission_to_write(shared_artifact):
    plan = shared_artifact.plan
    initial = next(
        o for o in plan.walk() if isinstance(o, BufferOp) and o.sym_name.data == "Initial"
    )
    output = next(
        o for o in plan.walk() if isinstance(o, ViewOp) and isinstance(o.source.owner, FrameOp)
    )
    assert initial.initialized.data == 1
    output.operands = (initial.ref, *output.operands[1:])
    with pytest.raises(
        CompileError, match="unframed invocation GM must be initialized and read-only"
    ):
        verify_plan(plan)


def test_unframed_input_still_requires_initialized_external_data(shared_artifact):
    plan = shared_artifact.plan
    initial = next(
        o for o in plan.walk() if isinstance(o, BufferOp) and o.sym_name.data == "Initial"
    )
    initial.attributes["initialized"] = IntAttr(0)
    with pytest.raises(
        CompileError, match="unframed invocation GM must be initialized and read-only"
    ):
        verify_plan(plan)


def test_frame_and_unframed_alias_cannot_be_given_different_roots(shared_artifact):
    plan = shared_artifact.plan
    frame = next(o for o in plan.walk() if isinstance(o, FrameOp))
    input_task = next(
        o for o in plan.walk() if isinstance(o, TaskOp) and o.refs[0].type.bank.data == "GM"
    )
    view = input_task.refs[0].owner
    view.operands = (frame.source, *view.operands[1:])
    with pytest.raises(CompileError, match="framed root must use its checked frame"):
        verify_plan(plan)

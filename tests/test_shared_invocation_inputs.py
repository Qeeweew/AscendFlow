"""Unframed GM sharing needs actual read effects and every physical group bound."""

from dataclasses import replace

import pytest
from ast_shared_state_sum import shared_state_sum
from test_sequence_sync import finite_check
from xdsl.dialects.builtin import IntAttr

from xdsl_npu import compile, verify_plan
from xdsl_npu.ir import BufferOp, FrameOp, TaskOp, ViewOp
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.fixture(scope="module")
def artifact():
    return compile(shared_state_sum.specialize(B=16, GROUPS=4))


@pytest.mark.parametrize("calls,tiles", [(0, 0), (0, 7), (1, 0), (3, 0), (3, 1), (2, 7)])
def test_actual_shared_input_and_partitioned_output_bytes_in_every_group(artifact, calls, tiles):
    plan = artifact.plan
    for group in range(4):
        finite_check(plan, tiles, invocations=calls, group=group)
    assert artifact.report["invocation_frames"] == 1
    work = runtime_work(plan, {"outputs": calls, "tiles": tiles})
    assert work["vector_requested_elements"]["add"] == 4 * calls * tiles * 16
    assert work["declared_gm_working_set_bytes"] == 4 * (1 + calls + tiles) * 16 * 4


@pytest.mark.parametrize("buffer", ["Initial", "X"])
def test_zero_group_validity_does_not_hide_a_later_group_out_of_bounds(artifact, buffer):
    plan = artifact.plan
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


def test_initialized_unframed_gm_is_not_permission_to_write(artifact):
    plan = artifact.plan
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


def test_unframed_input_still_requires_initialized_external_data(artifact):
    plan = artifact.plan
    initial = next(
        o for o in plan.walk() if isinstance(o, BufferOp) and o.sym_name.data == "Initial"
    )
    initial.attributes["initialized"] = IntAttr(0)
    with pytest.raises(
        CompileError, match="unframed invocation GM must be initialized and read-only"
    ):
        verify_plan(plan)


def test_frame_and_unframed_alias_cannot_be_given_different_roots(artifact):
    plan = artifact.plan
    frame = next(o for o in plan.walk() if isinstance(o, FrameOp))
    input_task = next(
        o for o in plan.walk() if isinstance(o, TaskOp) and o.refs[0].type.bank.data == "GM"
    )
    view = input_task.refs[0].owner
    view.operands = (frame.source, *view.operands[1:])
    with pytest.raises(CompileError, match="framed root must use its checked frame"):
        verify_plan(plan)


def test_all_groups_can_read_the_same_gm_without_inventing_exclusive_frames():
    source = (
        shared_state_sum.source.replace("(GROUPS, B), T.f32)", '(B,), T.f32, access="read_write")')
        .replace("(GROUPS, tiles, B)", "(tiles, B)")
        .replace("Initial[group, :]", "Initial[:]")
        .replace("X[group, t, :]", "X[t, :]")
    )
    result = compile(replace(shared_state_sum, source=source).specialize(B=16, GROUPS=4))
    for group in range(4):
        finite_check(result.plan, 3, invocations=2, group=group)
    assert result.report["invocation_frames"] == 1
    work = runtime_work(result.plan, {"outputs": 2, "tiles": 3})
    assert work["declared_gm_working_set_bytes"] == (1 + 3 + 4 * 2) * 16 * 4

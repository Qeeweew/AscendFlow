"""Call-local GM frame hazards remain visible across ordinary/pipeline regions."""

from dataclasses import replace

import pytest

from examples.state.frame_snapshot import frame_state_sum
from tests.models.local_sequence import finite_check
from xdsl_npu import compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import FrameOp, RegionChannelOp
from xdsl_npu.sequence_model import extract_sequence


@pytest.mark.parametrize("calls,count", [(0, 0), (0, 7), (1, 0), (3, 0), (3, 1), (2, 2), (3, 7)])
def test_actual_frame_bytes_and_events_across_groups_and_calls(frame_artifact, calls, count):
    for group in range(4):
        finite_check(frame_artifact.plan, count, invocations=calls, group=group)


def test_frame_identity_is_composed_within_call_and_fresh_across_calls(frame_artifact):
    sequence = extract_sequence(frame_artifact.plan)
    frame_cells = [c for c in sequence.summary.cells if isinstance(c.storage.owner, FrameOp)]
    assert frame_cells and all(
        c.storage.owner.source.owner.sym_name.data == "Output" for c in frame_cells
    )
    assert {
        o.hazard for o in sequence.summary.obligations if isinstance(o.storage.owner, FrameOp)
    } == {"RAW", "WAW"}
    assert all(not isinstance(o.storage.owner, FrameOp) for o in sequence.invocation_dependencies())
    assert frame_artifact.report["native_templates"] == 7
    assert frame_artifact.asc.count("for (int64_t") == 2
    assert frame_artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1


def test_a_later_frame_writer_does_not_initialize_an_earlier_frame_reader():
    source = frame_state_sum.source.replace(
        "            T.copy(state, Output[group, output, :])\n", "", 1
    )
    with pytest.raises(CompileError, match="undefined .*read on .*Output"):
        compile(replace(frame_state_sum, source=source).specialize(B=16, GROUPS=4))


def test_snapshot_ready_is_required_only_for_the_empty_child(frame_artifact):
    channel = next(
        op
        for op in frame_artifact.plan.walk()
        if isinstance(op, RegionChannelOp)
        and op.source_region.data == "r0"
        and op.destination_region.data == "r2"
        and op.destination_task.data.endswith("_0_copy")
    )
    assert [[v.data for v in row] for row in channel.domains] == [[1, 0, 0, 0]]
    # The earlier reload's ready path also protects the downstream Vector
    # overwrite. Synthesis must not create another empty-path notification.
    assert not any(
        isinstance(op, RegionChannelOp)
        and op.source_region.data == "r0"
        and op.destination_region.data == "r2"
        and op.destination_task.data.endswith("_1_add")
        for op in frame_artifact.plan.walk()
    )
    # The independent byte/event oracle checks the actual emitted commands,
    # including both last-reader parities and repeated invocations.
    for count in (0, 1, 2, 3, 7, 8):
        finite_check(frame_artifact.plan, count, invocations=3)


def test_large_constant_calls_and_child_keep_the_same_frame_proof_size():
    source = frame_state_sum.source.replace("outputs: T.int64,", "outputs: T.constexpr = 1,")
    source = source.replace("tiles: T.int64,", "tiles: T.constexpr = 1,")
    definition = replace(frame_state_sum, source=source)
    reports = [
        compile(definition.specialize(outputs=n, tiles=n, B=16, GROUPS=1)).report
        for n in (1024, 1048576)
    ]
    for key in (
        "native_templates",
        "completion_statistics",
        "proof_path_queries",
        "invocation_path_visits",
        "sequence_completion_sites",
        "sequence_completion_transfers",
        "invocation_completion_sites",
        "invocation_completion_facts",
        "invocation_completion_transfers",
    ):
        assert reports[0][key] == reports[1][key]
    assert reports[0]["completion_statistics"]["condition_facts"] > 0


def test_partial_frame_write_cannot_initialize_unwritten_bytes():
    source = frame_state_sum.source.replace(
        "T.copy(state, Output[group, output, :])",
        "T.copy(state[:B // 2], Output[group, output, :B // 2])",
        1,
    )
    with pytest.raises(CompileError, match=r"undefined sequence entry read on Output\[32:64\]"):
        compile(replace(frame_state_sum, source=source).specialize(B=16, GROUPS=1))


def test_two_real_frame_fragments_jointly_initialize_the_later_whole_reader():
    source = frame_state_sum.source.replace(
        "T.copy(state, Output[group, output, :])",
        "T.copy(state[:B // 2], Output[group, output, :B // 2])\n"
        "            T.copy(state[B // 2:], Output[group, output, B // 2:])",
        1,
    )
    result = compile(replace(frame_state_sum, source=source).specialize(B=16, GROUPS=1))
    for count in (0, 1, 2, 7):
        finite_check(result.plan, count, invocations=3)

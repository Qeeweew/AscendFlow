"""Real DSL -> fixed-child summaries -> parent events -> retained ASC loops."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.fixtures.pitched_children import cube_columns, pitched_children
from tests.models.local_sequence import finite_check
from xdsl_npu import compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import RegionChannelOp, RegionSyncOp, StreamOp, ViewOp
from xdsl_npu.sequence_model import extract_sequence
from xdsl_npu.sync import verify_plan


@pytest.fixture(scope="module")
def artifact():
    return compile(pitched_children)


def test_real_column_writers_have_one_ready_capture_for_the_whole_consumer(artifact):
    assert artifact.report["native_templates"] == 5
    assert artifact.asc.count("for (int64_t") == 2
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1  # Kernel return only.
    channels = [op for op in artifact.plan.walk() if isinstance(op, RegionChannelOp)]
    assert len(channels) == 1
    channel = channels[0]
    assert channel.source_task.data.endswith("store_2_copy")
    assert channel.source_offset.data == 3 and channel.source_end.data == 0
    sequence = extract_sequence(artifact.plan)
    assert sequence.regions["r0"].constant_count == 4
    assert sequence.regions["r0"].storage_periods["Y"] == 1
    assert {o.source.access.position.offset for o in sequence.summary.obligations} >= {0, 1, 2, 3}
    for group in (0, 1):
        finite_check(artifact.plan, 0, group=group)


def test_deleting_actual_boundary_notification_breaks_ready_proof(artifact):
    plan = artifact.plan
    for op in list(plan.walk()):
        if isinstance(op, (RegionChannelOp, RegionSyncOp)):
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match="missing.*RAW|uncovered.*RAW"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 0)


def test_partial_child_write_does_not_initialize_missing_columns():
    source = pitched_children.source.replace("N.Pipelined(4,", "N.Pipelined(3,")
    with pytest.raises(CompileError, match="undefined sequence entry read on Y"):
        compile(replace(pitched_children, source=source))


def test_parent_checks_union_of_actual_owners_across_children(artifact):
    plan = artifact.plan
    second = [op for op in plan.walk() if isinstance(op, StreamOp)][1]
    view = next(
        op
        for op in second.body.block.ops
        if isinstance(op, ViewOp) and op.source.owner.sym_name.data == "Y"
    )
    view.attributes["core_stride"] = IntAttr(0)
    view.attributes["core_terms"] = ArrayAttr([])
    with pytest.raises(CompileError, match="sequential regions disagree on GM ownership"):
        verify_plan(plan)


def test_cube_fixed_child_keeps_native_templates_and_actual_row_pitch():
    result = compile(cube_columns)
    assert result.report["native_templates"] == 5
    assert result.report["storage_periods"]["Y"] == 1
    assert result.asc.count("for (int64_t") == 1
    writes = [
        op
        for op in result.plan.walk()
        if isinstance(op, ViewOp) and op.source.owner.sym_name.data == "Y"
    ]
    assert len(writes) == 1
    assert writes[0].nbytes.data == 64
    assert writes[0].rows.data == 16
    assert writes[0].row_stride.data == 512


def test_joint_owner_budget_counts_all_groups_before_materialization(artifact, monkeypatch):
    import xdsl_npu.region_summary as summaries
    import xdsl_npu.sequence_model as sequences

    monkeypatch.setattr(summaries, "MAX_FIXED_ACCESS_SEGMENTS", 32)
    monkeypatch.setattr(
        sequences,
        "verify_translated_owners",
        lambda *args, **kwargs: pytest.fail("owner union allocated before budget check"),
    )
    with pytest.raises(CompileError, match="cross-region fixed access budget"):
        extract_sequence(artifact.plan)

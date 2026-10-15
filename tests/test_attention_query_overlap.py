"""Query-boundary overlap preserves old normalization and output state."""

import pytest

from examples.flash_attention.tiled_legacy import flash_attention_tiled
from xdsl_npu import compile, verify_plan
from xdsl_npu.builder import Kernel
from xdsl_npu.ir import ChannelOp, SyncOp
from xdsl_npu.mixed import extract_mixed
from xdsl_npu.model import CompileError, storage_layout


def example():
    return flash_attention_tiled(
        1, 1, 1, 256, 1024, 128, query_tiles_per_group=2, overlap_queries=True
    )


def test_query_overlap_has_two_denominators_and_delayed_output_initialization():
    artifact = compile(example())
    program = extract_mixed(artifact.plan).programs["aiv0"]
    tasks = {task.uid: task for task in program.tasks}
    denominator = next(storage for storage in program.storages if storage.name == "denominator")
    assert denominator.nbytes == 2 * 64 * 4
    assert tasks["init_o"].stage == tasks["update_o"].stage
    assert tasks["import_scores"].issue_order < tasks["reciprocal"].issue_order
    assert tasks["init_o"].issue_order < tasks["update_o"].issue_order
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 3


def test_reusing_previous_query_denominator_before_normalization_is_rejected(monkeypatch):
    original = Kernel.view

    def alias(self, source, *args, **kwargs):
        name = getattr(source.owner, "sym_name", None)
        if name is not None and name.data == "denominator" and kwargs.get("slots") == 2:
            kwargs["slots"] = 1
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(Kernel, "view", alias)
    with pytest.raises(CompileError, match="cycle|missing|cannot cover"):
        compile(example())


@pytest.fixture(scope="module")
def whole_update():
    return compile(
        flash_attention_tiled(
            1,
            1,
            1,
            256,
            2048,
            128,
            query_tiles_per_group=2,
            overlap_queries=True,
            vector_update="tile",
        )
    )


def test_full_next_vec1_is_dispatched_before_old_pv_gate(whole_update):
    program = extract_mixed(whole_update.plan).programs["aiv0"]
    tasks = {t.uid: t for t in program.tasks}
    soft, pv = tasks["update_m"], tasks["import_pv"]
    rows = 8
    assert pv.refs[0].total_nbytes == 64 * 128 * 4
    assert tasks["rescale_o"].params == (64, 128)
    for g in range(program.count):
        if not pv.active(g):
            continue
        next_vec1_last = (g // rows + 2) * rows - 1
        if next_vec1_last < program.count:
            assert (next_vec1_last + soft.stage, soft.issue_order) < (g + pv.stage, pv.issue_order)
    assert tasks["partial_sum"].params == (8, 1024)
    assert tasks["round_p"].native == "convert_rn"
    bridge = next(s for s in program.storages if s.name == "bridge_input")
    assert bridge.nbytes == 2 * 8 * 1024 * 4
    assert storage_layout(program.storages)[1]["UB"] <= 192 * 1024
    assert whole_update.asc.count("asc_sync_pipe(PIPE_ALL)") == 3


def test_inplace_score_slots_require_release_after_the_final_sum_read(whole_update):
    plan = whole_update.plan
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, ChannelOp)
        and op.destination.data == "import_scores"
        and op.source.data == "partial_sum"
    )
    assert channel.distance.data == 2
    kernel = channel.parent_op()
    for op in list(kernel.walk()):
        if isinstance(op, SyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing"):
        verify_plan(plan)


def test_block_max_uses_explicit_scratch_and_keeps_sum_and_rounding():
    programs = []
    for policy in ("whole", "block"):
        source = flash_attention_tiled(
            1,
            1,
            1,
            256,
            2048,
            128,
            query_tiles_per_group=2,
            overlap_queries=True,
            vector_update="tile",
            max_reduce=policy,
        )
        programs.append(extract_mixed(source).programs["aiv0"])
    old, new = ({t.uid: t for t in p.tasks} for p in programs)
    for uid in (
        "partial_sum",
        "tile_sum",
        "rescale_l",
        "update_l",
        "round_p",
        "update_o",
        "normalize",
    ):
        assert (old[uid].native, old[uid].params, old[uid].scalar) == (
            new[uid].native,
            new[uid].params,
            new[uid].scalar,
        )
    assert new["block_max"].native == new["partial_max"].native == "reduce_max8"
    assert new["block_max"].refs[1].storage.name == "max_scratch"
    assert new["partial_max"].refs[1].total_nbytes == 8 * 16 * 4
    assert storage_layout(programs[1].storages)[1]["UB"] == 191040

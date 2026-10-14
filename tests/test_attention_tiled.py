import pytest
from flash_attention_tiled import flash_attention_tiled
from xdsl.dialects.builtin import IntAttr

from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.ir import CrossSyncOp, dump, parse
from xdsl_npu.mixed import extract_mixed, global_dependencies
from xdsl_npu.model import CompileError
from xdsl_npu.performance import physical_work


@pytest.mark.parametrize(
    "bq,bk,ck,dim,dtype",
    [(64, 256, 64, 64, "f16"), (64, 512, 128, 128, "bf16"), (128, 1024, 128, 128, "f16")],
)
def test_logical_rounding_and_communication_are_separate_from_native_tiles(bq, bk, ck, dim, dtype):
    plan = autosync(
        parse(
            dump(
                flash_attention_tiled(
                    1, 1, 1, bq, bk * 2, dim, block_q=bq, block_kv=bk, cube_kv=ck, dtype=dtype
                )
            )
        )
    )
    m = extract_mixed(plan)
    report = verify_plan(plan)
    assert m.divisors == {"aic": bk // ck, "aiv0": bq // 16, "aiv1": bq // 16}
    assert report["epochs"] == 2
    p = m.programs["aic"]
    pv = next(t for t in p.tasks if t.uid == "pv")
    export = next(t for t in p.tasks if t.uid == "export_pv")
    assert pv.native == "mad_cycle" and pv.params[-1] == bk // ck
    assert sum(export.active(g) for g in range(p.count)) == 2
    for actor in ("aiv0", "aiv1"):
        tasks = {t.uid: t for t in m.programs[actor].tasks}
        assert tasks["round_p"].native == "convert_rn"
        assert tasks["partial_sum"].params == (8, bk)
        assert tasks["partial_sum"].refs[0].dtype == "f32"
    work = physical_work(plan)
    assert work["native_cube_flops"] == 4 * bq * bk * 2 * dim
    assert work["gm_instruction_requests_by_root"]["pv:W"] == 2 * bq * dim * 4
    assert emit_asc(plan).count("asc_sync_pipe(PIPE_ALL)") == 3


def test_first_segment_cannot_publish_the_whole_logical_score_block():
    plan = autosync(flash_attention_tiled(1, 1, 1, 64, 1024, 64, block_q=64))
    for op in plan.walk():
        if (
            isinstance(op, CrossSyncOp)
            and op.channel.data == "scores_ready"
            and op.action.data == "set"
        ):
            op.attributes["lower"] = IntAttr(0)
            op.attributes["upper"] = IntAttr(1)
    with pytest.raises(CompileError, match="missing mixed RAW"):
        verify_plan(plan)


def test_multiple_epochs_do_not_reorder_the_flattened_physical_frontier():
    plan = autosync(
        flash_attention_tiled(1, 1, 1, 64, 256 * 3, 64, block_q=64, block_kv=256, cube_kv=64)
    )
    m = extract_mixed(plan)
    assert verify_plan(plan)["epochs"] == 3
    for src, dst, obligation in global_dependencies(m):
        if src[0] != dst[0] and obligation.hazard == "RAW":
            assert (
                src[1].generation // m.divisors[src[0]] == dst[1].generation // m.divisors[dst[0]]
            )


def test_tiled_oracle_retains_the_declared_logical_probability_rounding():
    pytest.importorskip("numpy")
    from attention_oracle import attention_inputs

    plan = autosync(flash_attention_tiled(1, 1, 1, 64, 2048, 64, block_q=64))
    _, expected, _, report = attention_inputs(plan)
    assert report["logical_block_kv"] == 1024
    assert expected.shape == (1, 1, 64, 64)


def test_lookahead_changes_issue_and_not_dma_work_or_buffer_allocations():
    plans = [
        autosync(
            flash_attention_tiled(
                1, 1, 1, 64, 512, 64, block_q=64, block_kv=256, cube_kv=64, pipeline=policy
            )
        )
        for policy in (False, True)
    ]
    slow, fast = [physical_work(plan) for plan in plans]
    for key in ("native_cube_flops", "dma_source_bytes_by_route", "local_bank_bytes_by_actor"):
        assert slow[key] == fast[key]
    program = extract_mixed(plans[1]).programs["aic"]
    tasks = {t.uid: t for t in program.tasks}
    assert tasks["load_v"].stage + 1 == tasks["operand_v"].stage
    assert tasks["export_scores"].issue_order < tasks["load_p"].issue_order
    code = emit_asc(plans[1])
    assert code.index("// qk:") < code.index("// load_p:")
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 3


@pytest.mark.parametrize("query_tiles", [2, 4])
def test_query_boundary_retains_old_last_read_before_new_state_initialization(query_tiles):
    plan = autosync(
        flash_attention_tiled(
            1,
            2,
            1,
            64 * query_tiles,
            512,
            64,
            block_q=64,
            block_kv=256,
            cube_kv=64,
            query_tiles_per_group=query_tiles,
        )
    )
    m = extract_mixed(plan)
    assert verify_plan(parse(dump(plan)))["epochs"] == 2 * query_tiles
    actor = m.programs["aiv0"]
    tasks = {t.uid: t for t in actor.tasks}
    assert tasks["init_o"].phase_width == 4
    assert tasks["init_o"].active(0) and tasks["init_o"].active(8)
    assert not tasks["init_o"].active(4)
    assert "import_pv_tail" in tasks
    # GM ownership covers all query tiles once; query streaming saves scratch
    # allocation, not K/V DMA requests or the per-query online recurrence.
    work = physical_work(plan)
    assert work["gm_instruction_requests_by_root"]["output:W"] == 2 * 64 * query_tiles * 64 * 2
    assert work["gm_instruction_requests_by_root"]["k:R"] == 2 * query_tiles * 512 * 64 * 2
    code = emit_asc(plan)
    assert "% 8 >= 0" in code and "% 8 < 4" in code


def test_v_lookahead_cannot_silently_change_the_declared_single_slot():
    with pytest.raises(CompileError, match="two declared L1 slots"):
        flash_attention_tiled(1, 1, 1, 64, 256, 64, block_q=64, block_kv=256, l1_slots=1)

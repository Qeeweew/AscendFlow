import pytest
from xdsl.dialects.builtin import IntAttr
from xdsl.transforms.dead_code_elimination import dce

from xdsl_npu import autosync, emit_asc, flash_attention, verify_plan
from xdsl_npu.ir import (
    BufferOp,
    CrossChannelOp,
    CrossDrainOp,
    CrossSyncOp,
    TaskOp,
    ViewOp,
    dump,
    parse,
)
from xdsl_npu.mixed import ACTORS, extract_mixed, global_dependencies, mixed_execution, mixed_graph
from xdsl_npu.mixed_check import bounded_mixed_check
from xdsl_npu.model import CompileError, Instance


@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_mha_gqa_mqa_rounding_and_three_physical_rings(heads, kv_heads, dtype):
    source = flash_attention(1, heads, kv_heads, 64, 128, 64, dtype=dtype)
    plan = autosync(parse(dump(source)))
    report = verify_plan(plan)
    assert report["groups"] == heads and report["epochs"] == 1
    assert len(report["cross_counter_ids"]) == 3
    assert all(name.endswith("_ready") for name in report["cross_counter_ids"])
    assert report["scratch_bytes"] == heads * 2 * (64 * 128 * 6 + 64 * 64 * 4)
    mixed = extract_mixed(plan)
    for actor in ACTORS[1:]:
        tasks = {task.uid: task for task in mixed.programs[actor].tasks}
        assert tasks["round_p"].native == tasks["round_output"].native == "convert_rn"
        assert tasks["partial_sum"].refs[0].dtype == "f32"
        assert tasks["round_p"].refs[0].storage == tasks["partial_sum"].refs[0].storage
        assert tasks["round_p"].refs[1].dtype == dtype
        assert (
            tasks["rescale_o"].native == "mul_row_broadcast" and tasks["update_o"].native == "add"
        )
    code = emit_asc(plan)
    assert "load_cbuf_to_cb(" in code and ", true, inc);" in code
    assert "asc_copy_l12l0b_transpose" not in code
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 3
    dce(plan)
    assert verify_plan(plan)["tasks_per_group"] == report["tasks_per_group"]


@pytest.mark.parametrize(
    "epochs,gm_slots,local_slots", [(1, 1, 1), (2, 1, 1), (3, 2, 2), (5, 3, 2)]
)
def test_ring_generation_and_bidirectional_quorum(epochs, gm_slots, local_slots):
    plan = autosync(
        flash_attention(
            1,
            1,
            1,
            32,
            epochs * 64,
            64,
            block_q=32,
            block_kv=64,
            gm_slots=gm_slots,
            local_slots=local_slots,
        )
    )
    mixed, channels, commands = mixed_execution(plan)
    graph, begin, end, sends, receives, _, _ = mixed_graph(mixed, channels, commands)
    assert channels["probabilities_ready"].sources == ("aiv0", "aiv1")
    assert channels["probabilities_ready"].destinations == ("aic",)
    for actor in ("aiv0", "aiv1"):
        ids = {t.uid: i for i, t in enumerate(mixed.programs[actor].tasks)}
        load = (actor, Instance(0, ids["import_scores"]))
        soft = (actor, Instance(0, ids["exponential"]))
        assert not graph.precedes(end[soft], end[load])
        assert graph.precedes(
            sends[(actor, "probabilities_ready", 0)], receives[("aic", "probabilities_ready", 0)]
        )
    # The unchanged bidirectional ready paths cover every read-retirement
    # obligation. No redundant credit channel, Wait, Set or drain remains.
    assert not any(name.endswith("_credit") for name in channels)
    for source, destination, obligation in global_dependencies(mixed):
        if obligation.hazard == "WAR":
            assert graph.precedes(end[source], begin[destination])
    assert all(c.pipe == "S" for seq in commands.values() for c in seq if c.kind == "cross_wait")
    if epochs <= 2:
        checked = bounded_mixed_check(plan, scope="shared", max_states=200_000)
        assert checked.terminal_states == 1
        if epochs == 1:
            assert checked.exhaustive
        else:
            # The two-epoch local-event abstraction reaches its finite budget.
            # Preserve the incomplete result; do not present it as a proof.
            assert checked.exhaustive or checked.states == 200_000


@pytest.mark.parametrize(
    "channel",
    [
        "probabilities_ready",
        "pv_ready",
        "scores_ready",
    ],
)
def test_missing_release_or_visibility_is_an_actual_counterexample(channel):
    plan = autosync(
        flash_attention(1, 1, 1, 32, 64, 64, block_q=32, block_kv=64, gm_slots=1, local_slots=1)
    )
    for op in list(plan.walk()):
        if (
            isinstance(op, CrossChannelOp)
            and op.sym_name.data == channel
            or isinstance(op, (CrossSyncOp, CrossDrainOp))
            and op.channel.data == channel
        ):
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match="missing mixed RAW|missing mixed WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_mixed_check(plan, scope="shared")


@pytest.fixture(scope="module")
def causal_plan():
    return autosync(
        flash_attention(1, 2, 1, 128, 192, 64, block_q=32, block_kv=64, causal=True, query_start=17)
    )


def test_causal_prefixes_and_offset_are_real_static_protocols(causal_plan):
    from xdsl_npu.dispatch import extract_dispatch

    dispatch = extract_dispatch(parse(dump(causal_plan)))
    report = verify_plan(causal_plan)
    assert report["groups"] == 8 and report["static_cases"] == 3
    assert [mixed.epochs for _, mixed in dispatch.cases] == [1, 2, 3]
    for _, mixed in dispatch.cases:
        first, tiles, epochs, _ = (n.data for n in mixed.op.attributes["attention_domain"])
        assert epochs * 64 < 17 + first + tiles * 32 + 64
        assert (epochs - 1) * 64 <= 17 + first + tiles * 32 - 1
        for actor in ACTORS[1:]:
            assert any(t.native == "mask_causal" for t in mixed.programs[actor].tasks)
    code = emit_asc(causal_plan)
    assert code.count('extern "C" __global__') == 1
    assert "flash_attention<<<8" in code
    assert "npu_visible" in code and "asc_get_block_idx() - " in code


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [("hole", "partition"), ("binding", "identical ordered"), ("workspace", "overlap GM scores")],
)
def test_static_selector_and_cross_case_ownership_are_rechecked(causal_plan, mutation, diagnostic):
    from xdsl_npu.dispatch import extract_dispatch

    bad = causal_plan.clone()
    dispatch = extract_dispatch(bad)
    case, mixed = dispatch.cases[1]
    if mutation == "hole":
        case.attributes["lower"] = IntAttr(case.lower.data + 1)
    elif mutation == "binding":
        root = next(o for o in case.walk() if isinstance(o, BufferOp) and o.sym_name.data == "k")
        root.attributes["nbytes"] = IntAttr(root.nbytes.data + 32)
    else:
        delta = mixed.op.attributes["attention_domain"].data[-1].data * 2 * 32 * 64 * 4
        for view in case.walk():
            if (
                isinstance(view, ViewOp)
                and isinstance(view.source.owner, BufferOp)
                and view.source.owner.sym_name.data == "scores"
            ):
                view.attributes["offset"] = IntAttr(view.offset.data - delta)
    with pytest.raises(CompileError, match=diagnostic):
        verify_plan(bad)


def test_causal_numerical_oracle_uses_absolute_query_positions(causal_plan):
    np = pytest.importorskip("numpy")
    from xdsl_npu.attention_oracle import attention_inputs, decode, rounded

    raw, expected, selected, report = attention_inputs(causal_plan, data_case="uniform")
    v = decode(raw["v"], "f16", (1, 1, 192, 64))
    assert report["causal"] and report["query_start"] == 17
    for row in (0, 15, 16, 31, 32, 63, 64, 127):
        mean = rounded(v[0, 0, : 18 + row].mean(axis=0), "f16")
        assert np.allclose(expected[0, 0, row], mean, rtol=0.002, atol=0.0002)
    assert len(selected) == 128


def test_logical_phase_is_distinct_from_physical_issue_stage():
    source = flash_attention(1, 1, 1, 32, 64, 64, block_q=32, block_kv=64)
    mixed = extract_mixed(source)
    obligations = global_dependencies(mixed)
    assert any(
        src[0].startswith("aiv")
        and dst[0] == "aic"
        and o.storage == "probabilities"
        and o.hazard == "RAW"
        for src, dst, o in obligations
    )
    task = next(t for t in source.walk() if isinstance(t, TaskOp) and t.uid.data == "load_p")
    task.attributes["logical_phase"] = IntAttr(0)
    with pytest.raises(CompileError, match="undefined probabilities"):
        autosync(source)


def test_explicit_issue_that_deadlocks_a_counter_roundtrip_is_rejected():
    source = flash_attention(
        1, 1, 1, 32, 192, 64, block_q=32, block_kv=64, gm_slots=1, local_slots=2
    )
    # Delay Cube PV beyond AIV's controller Wait on that PV. The AIV cannot
    # dispatch the next score import that would permit the next score export.
    for task in source.walk():
        if (
            isinstance(task, TaskOp)
            and task.logical_phase is not None
            and task.logical_phase.data == 2
        ):
            task.attributes["stage"] = IntAttr(4)
    with pytest.raises(CompileError, match="cycle|fixed schedule"):
        autosync(source)


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ({"query_length": 63}, "complete Q/KV"),
        ({"kv_length": 127}, "complete Q/KV"),
        ({"dtype": "f32"}, "f16 or bf16"),
        ({"head_dim": 80}, "D64/128"),
        ({"query_heads": 3, "kv_heads": 2}, "Hq/Hkv"),
    ],
)
def test_unsupported_inputs_cannot_fallback(change, diagnostic):
    args = {
        "batch": 1,
        "query_heads": 2,
        "kv_heads": 1,
        "query_length": 64,
        "kv_length": 128,
        "head_dim": 64,
    }
    args.update(change)
    with pytest.raises(CompileError, match=diagnostic):
        flash_attention(**args)


def test_independent_oracle_gqa_mapping_and_b16_tie_rounding():
    np = pytest.importorskip("numpy")
    from xdsl_npu.attention_oracle import attention_inputs, decode, encode, online_attention

    ties = np.array([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8)], dtype=np.float32)
    assert np.array_equal(
        decode(encode(ties, "bf16"), "bf16"), np.array([1, 1 + 2**-6, -1], dtype=np.float32)
    )
    plan = autosync(flash_attention(1, 4, 2, 32, 128, 64, dtype="bf16", block_q=32, block_kv=64))
    raw, expected, rows, report = attention_inputs(plan, data_case="uniform")
    assert report["oracle_scope"] == "full" and len(rows) == 32
    v = decode(raw["v"], "bf16", (1, 2, 128, 64))
    # exp(0)=1 is exactly b16; output is the KV-head mean for every query row.
    for h in range(4):
        mean = decode(encode(v[0, h // 2].mean(axis=0), "bf16"), "bf16")
        assert np.array_equal(expected[0, h], np.broadcast_to(mean, (32, 64)))
    q = np.array([[0.8] * 64], dtype=np.float32)
    k = np.arange(128 * 64, dtype=np.float32).reshape(128, 64) % 13 / 16
    y = online_attention(q, k, v[0, 0], block_kv=64, dtype="bf16", scale=0.125)
    assert np.isfinite(y).all()


def test_lookahead_aliases_and_compact_reductions_have_actual_retirement_effects():
    from xdsl_npu.model import dependencies

    source = flash_attention(
        1,
        1,
        1,
        128,
        384,
        128,
        block_q=128,
        q_cache="l1",
        reuse_ub=True,
        pipeline=True,
        packed_reductions=True,
    )
    plan = autosync(parse(dump(source)))
    mixed = extract_mixed(plan)
    assert verify_plan(plan)["cross_backlog_upper_bounds"]["scores_ready"] == 2
    for actor in ACTORS[1:]:
        program = mixed.programs[actor]
        tasks = {t.uid: t for t in program.tasks}
        assert tasks["import_scores"].refs[-1].storage == tasks["import_pv"].refs[-1].storage
        assert tasks["round_output"].refs[-1].storage == tasks["export_p"].refs[0].storage
        assert tasks["exponential"].refs[0].storage != tasks["import_scores"].refs[-1].storage
        assert tasks["partial_sum"].native == "reduce_tile_sum64"
        assert tasks["tile_sum"].native == "add"
        obligations = dependencies(program, _imports=mixed.imports)
        assert any(
            o.hazard == "WAR"
            and program.tasks[o.source.task].uid == "scale_scores"
            and program.tasks[o.destination.task].uid == "import_pv"
            and o.source.generation == o.destination.generation
            for o in obligations
        )
    code = emit_asc(plan)
    assert "score_input" not in code and "output_rounded" not in code
    assert "uint8_t(64), 1, 1, 16" in code
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 3


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_packed_decode_keeps_one_absolute_position_for_all_head_rows(dtype):
    np = pytest.importorskip("numpy")
    from xdsl_npu.attention_oracle import attention_inputs, decode, rounded

    source = flash_attention(
        2,
        64,
        2,
        1,
        256,
        64,
        dtype=dtype,
        block_q=32,
        causal=True,
        query_start=17,
        packed_reductions=True,
    )
    plan = autosync(parse(dump(source)))
    mixed = extract_mixed(plan)
    assert verify_plan(plan)["groups"] == 4 and mixed.epochs == 1
    for actor in ACTORS[1:]:
        mask = next(t for t in mixed.programs[actor].tasks if t.uid == "causal_mask")
        assert mask.native == "mask_prefix" and mask.params[-1] == 18
    raw, expected, selected, report = attention_inputs(plan, scope="sampled", data_case="uniform")
    assert expected.shape == (2, 64, 1, 64) and selected.tolist() == [0]
    assert report["shape_b_hq_hkv_sq_sk_d"] == [2, 64, 2, 1, 256, 64]
    values = decode(raw["v"], dtype, (2, 2, 256, 64))
    for head in (0, 31, 32, 63):
        mean = rounded(values[1, head // 32, :18].mean(axis=0), dtype)
        assert np.allclose(expected[1, head, 0], mean, rtol=0.01, atol=0.001)


def test_packed_decode_and_lookahead_contracts_reject_unsafe_shortcuts():
    with pytest.raises(CompileError, match="padded heads unsupported"):
        flash_attention(1, 8, 1, 1, 128, 64)
    with pytest.raises(CompileError, match="requires two local slots"):
        flash_attention(1, 1, 1, 64, 128, 64, pipeline=True, local_slots=1)


def test_dense_launch_larger_than_one_case_has_exact_shared_argument_ownership():
    from xdsl_npu.dispatch import extract_dispatch

    plan = autosync(flash_attention(1, 32, 8, 4096, 128, 64))
    dispatch = extract_dispatch(parse(dump(plan)))
    assert verify_plan(plan)["groups"] == 2048
    assert [(case.lower.data, case.upper.data) for case, _ in dispatch.cases] == [
        (0, 1024),
        (1024, 2048),
    ]
    assert [m.op.attributes["attention_domain"].data[0].data for _, m in dispatch.cases] == [
        0,
        2048,
    ]
    assert emit_asc(plan).count('extern "C" __global__') == 1

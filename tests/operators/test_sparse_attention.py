"""Sparse shared-KV semantics and actual mixed-pipeline retirement."""

import numpy as np
import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from benchmarks.attention.sparse_reference import gather_oracle, oracle, selected_rows
from examples.sparse_attention.decode import sparse_mla_decode
from tests.models.structured_events import check_structured_events


@pytest.fixture
def problem():
    policy = {
        "B": 1,
        "T": 1,
        "H": 2,
        "D": 4,
        "PAGE": 4,
        "RATIO": 2,
        "WINDOW": 3,
        "TOPK": 4,
        "S": 8,
        "DTYPE": "bf16",
        "SCALE": 0.5,
    }
    data = {
        "q": np.zeros((1, 2, 4), dtype=np.float32),
        "ori_kv": np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 1, 4) / 16,
        "cmp_kv": np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 1, 4) / 4,
        "ori_block_table": np.array([[1, 0]], dtype=np.int32),
        "cmp_block_table": np.array([[1, 0]], dtype=np.int32),
        "seqused_ori_kv": np.array([5], dtype=np.int32),
        "ori_topk_length": np.array([[2]], dtype=np.int32),
        "cu_seqlens_q": np.array([0, 1], dtype=np.int32),
        "seqused_cmp_kv": np.array([2], dtype=np.int32),
        "cmp_residual_kv": np.array([1], dtype=np.int32),
        "cmp_sparse_indices": np.array([[[1, 1, 2, -1]]], dtype=np.int32),
        "sinks": np.zeros(2, dtype=np.float32),
    }
    return data, policy


def test_pages_repeated_indices_and_compression_causality(problem):
    data, policy = problem
    rows, slots = selected_rows(data, policy, 0)
    # Original positions 3,4 cross a page boundary; index 1 occurs twice,
    # index 2 is not complete at original length 5 with compression ratio 2.
    assert slots == [0, 1, 3, 4]
    assert np.array_equal(
        rows,
        np.stack(
            [
                data["ori_kv"][1, 3, 0],
                data["ori_kv"][0, 0, 0],
                data["cmp_kv"][1, 1, 0],
                data["cmp_kv"][1, 1, 0],
            ]
        ),
    )
    kv, bias = gather_oracle(data, policy)
    assert np.array_equal(kv[0, slots], rows)
    assert np.all(kv[0, [2, 5, 6, 7]] == 0)
    assert np.array_equal(
        np.isfinite(bias[0]), [True, True, False, True, True, False, False, False]
    )


def test_both_cache_planes_and_one_sink_share_a_single_denominator(problem):
    from benchmarks.attention.sparse_reference import rounded

    data, policy = problem
    rows, _ = selected_rows(data, policy, 0)
    actual = oracle(data, policy)
    # q=0, sink=0: four real rows plus exactly one zero-valued sink.
    expected = rounded((rows.sum(axis=0) / 5)[None, None, :].repeat(2, axis=1), "bf16")
    assert np.array_equal(actual, expected)
    data["cmp_sparse_indices"][0, 0, :2] = [0, 1]
    assert not np.array_equal(oracle(data, policy), actual)


def test_empty_selection_and_extreme_finite_sinks_are_zero(problem):
    data, policy = problem
    data["seqused_ori_kv"][:] = 0
    data["seqused_cmp_kv"][:] = 0
    data["cmp_residual_kv"][:] = 0
    data["cmp_sparse_indices"][:] = -1
    data["sinks"][:] = [-80, 80]
    assert np.array_equal(oracle(data, policy), np.zeros_like(data["q"]))


def test_padding_is_not_read_and_replay_window_is_respected(problem):
    data, policy = problem
    data["ori_topk_length"][:] = 1
    data["cmp_sparse_indices"][0, 0] = [0, -1, -1, -1]
    rows, slots = selected_rows(data, policy, 0)
    assert slots == [0, 3]
    assert np.array_equal(rows[0], data["ori_kv"][0, 0, 0])


def test_four_verification_tokens_have_individual_windows_indices_and_compression_edges(problem):
    data, policy = problem
    policy["T"] = 4
    data["q"] = np.zeros((4, 2, 4), dtype=np.float32)
    data["cu_seqlens_q"] = np.array([0, 4], dtype=np.int32)
    data["seqused_ori_kv"][:] = 6
    data["seqused_cmp_kv"][:] = 3
    data["cmp_residual_kv"][:] = 0
    data["ori_topk_length"] = np.zeros((4, 1), dtype=np.int32)
    data["cmp_sparse_indices"] = np.array(
        [[[2, 0, 1, -1]], [[0, 2, 1, -1]], [[1, 0, 2, -1]], [[2, 0, 1, -1]]], np.int32
    )
    for token, compressed in enumerate(([], [0], [1, 0], [2, 0, 1])):
        expected = [
            data["ori_kv"][data["ori_block_table"][0, p // 4], p % 4, 0]
            for p in range(token, token + 3)
        ] + [data["cmp_kv"][data["cmp_block_table"][0, p // 2], p % 2, 0] for p in compressed]
        rows, _ = selected_rows(data, policy, token)
        np.testing.assert_array_equal(rows, expected)
    before = oracle(data, policy)
    # Future raw and compressed cache entries exist in GM but must not affect
    # earlier queries. All tokens share the same cache and request metadata.
    data["ori_kv"][0, 1, 0] += 10  # original position 5
    data["cmp_kv"][0, 0, 0] += 10  # compressed index 2
    after = oracle(data, policy)
    np.testing.assert_array_equal(before[:3], after[:3])
    assert not np.array_equal(before[3], after[3])


@pytest.mark.parametrize(
    "features,topk,bm,vr,batch",
    [(128, 512, 32, 8, 1), (128, 2048, 32, 8, 1), (512, 512, 64, 8, 1), (512, 512, 64, 16, 4)],
)
def test_native_mixed_pipeline_has_complete_quorums_and_retirement(
    features, topk, bm, vr, batch, monkeypatch
):
    from tests.operators.test_sparse_persistent import bind_metadata, metadata_inputs

    bind_metadata(monkeypatch, metadata_inputs(batch, batch, topk))
    result = compile(
        sparse_mla_decode.specialize(
            B=batch,
            T=batch,
            GROUPS=1,
            H=bm,
            BM=bm,
            VR=vr,
            D=features,
            S=128 + topk,
            WINDOW=128,
            TOPK=topk,
            SCALE=features**-0.5,
        ),
        timeout_s=300,
    )
    trace = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=1 if topk > 512 or batch > 1 else 2,
        max_commands=40000 if topk > 512 or batch > 1 else 10000,
    )
    assert trace.bytes_read and trace.bytes_written and trace.consumptions
    # The selected finite mapping can change with the DSL's actual buffers
    # and instruction streams. Quorum, consumption and retirement are checked
    # against real accesses above, rather than requiring a particular retry.


def test_single_request_validation_cases_include_real_attention_rows():
    from benchmarks.attention.sparse_reference import inputs

    policy = {
        "B": 1,
        "T": 1,
        "H": 32,
        "D": 128,
        "PAGE": 128,
        "RATIO": 2,
        "MAX_SEQ": 4096,
        "ORI_BLOCKS": 32,
        "CMP_BLOCKS": 32,
        "WINDOW": 128,
        "TOPK": 512,
        "S": 640,
        "DTYPE": "bf16",
    }
    for case in ("window", "mixed", "padding", "duplicates", "replay", "sink"):
        data = inputs(policy, case)
        rows, _ = selected_rows(data, policy, 0)
        assert len(rows) > 0
        assert np.any(rows != 0)

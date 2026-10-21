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
        "Q": np.zeros((1, 2, 4), dtype=np.float32),
        "Ori": np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) / 16,
        "Cmp": np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4) / 4,
        "OriTable": np.array([[1, 0]], dtype=np.int32),
        "CmpTable": np.array([[1, 0]], dtype=np.int32),
        "Lengths": np.array([5], dtype=np.int32),
        "WindowLengths": np.array([2], dtype=np.int32),
        "Counts": np.array([3], dtype=np.int32),
        "Indices": np.array([[1, 1, 2, -1]], dtype=np.int32),
        "Sinks": np.zeros(2, dtype=np.float32),
    }
    return data, policy


def test_pages_repeated_indices_and_compression_causality(problem):
    data, policy = problem
    rows, slots = selected_rows(data, policy, 0)
    # Original positions 3,4 cross a page boundary; index 1 occurs twice,
    # index 2 is not complete at original length 5 with compression ratio 2.
    assert slots == [0, 1, 3, 4]
    assert np.array_equal(
        rows, np.stack([data["Ori"][1, 3], data["Ori"][0, 0], data["Cmp"][1, 1], data["Cmp"][1, 1]])
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
    data["Counts"][0] = 2
    data["Indices"][0, :2] = [0, 1]
    assert not np.array_equal(oracle(data, policy), actual)


def test_empty_selection_and_extreme_finite_sinks_are_zero(problem):
    data, policy = problem
    data["WindowLengths"][:] = 0
    data["Counts"][:] = 0
    data["Sinks"][:] = [-80, 80]
    assert np.array_equal(oracle(data, policy), np.zeros_like(data["Q"]))


def test_padding_is_not_read_and_replay_window_is_respected(problem):
    data, policy = problem
    data["WindowLengths"][:] = 1
    data["Counts"][:] = 1
    data["Indices"][0] = [0, -1, 2147483647, -2147483648]
    rows, slots = selected_rows(data, policy, 0)
    assert slots == [0, 3]
    assert np.array_equal(rows[0], data["Ori"][0, 0])


@pytest.mark.parametrize(
    "features,topk,bm,vr,batch",
    [(128, 512, 32, 8, 1), (128, 2048, 32, 8, 1), (512, 512, 64, 8, 1), (512, 512, 64, 16, 4)],
)
def test_native_mixed_pipeline_has_complete_quorums_and_retirement(
    features, topk, bm, vr, batch, monkeypatch
):
    from tests.models.structured_events import _Interpret

    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            name = root.sym_name.data
            values[operation.value] = {
                "Lengths": 4095,
                "WindowLengths": 64,
                "Counts": topk,
                "Indices": 0,
                "OriTable": (offset // 4) % 128,
                "CmpTable": (offset // 4) % 128,
            }[name]

    monkeypatch.setattr(_Interpret, "native", metadata)
    result = compile(
        sparse_mla_decode.specialize(
            B=batch,
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
    if features == 512:
        selection = result.report["event_resources"]["protocol_selection"]
        assert selection["selected_option"] > 0
        assert selection["changed_streams"]


def test_single_request_validation_cases_include_real_attention_rows():
    from benchmarks.attention.sparse_reference import inputs

    policy = {
        "B": 1,
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
    for case in ("window", "mixed", "compressed", "duplicates", "replay", "sink"):
        data = inputs(policy, case)
        rows, _ = selected_rows(data, policy, 0)
        assert len(rows) > 0
        assert np.any(rows != 0)

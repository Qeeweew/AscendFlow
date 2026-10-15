"""Independent attention arithmetic; no compiler or legacy operator factory."""

import pytest

from benchmarks.validation.attention import (
    decode,
    encode,
    inputs_and_oracle,
    online_attention,
    rounded,
)


def test_bf16_rounding_uses_ties_to_even():
    np = pytest.importorskip("numpy")
    ties = np.array([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8)], dtype=np.float32)
    assert np.array_equal(
        decode(encode(ties, "bf16"), "bf16"), np.array([1, 1 + 2**-6, -1], dtype=np.float32)
    )


def test_uniform_oracle_maps_query_heads_to_their_own_kv_head():
    np = pytest.importorskip("numpy")
    policy = {"batch": 1, "heads": 4, "kv_heads": 2, "groups": 1, "dtype": "bf16"}
    raw, expected, selected, _ = inputs_and_oracle(policy, 128, 1024, data_case="uniform")
    values = decode(raw["v"], "bf16", (1, 2, 1024, 128))
    assert selected.tolist() == list(range(128))
    for head in range(4):
        mean = rounded(values[0, head // 2].mean(axis=0), "bf16")
        assert np.array_equal(expected[0, head], np.broadcast_to(mean, (128, 128)))


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_causal_offset_uses_absolute_query_positions(dtype):
    np = pytest.importorskip("numpy")
    positions = np.array([0, 15, 16, 31, 32, 63, 64, 127])
    q = np.zeros((len(positions), 64), dtype=np.float32)
    k = np.zeros((192, 64), dtype=np.float32)
    v = rounded(
        np.arange(192, dtype=np.float32)[:, None] / 16
        + np.arange(64, dtype=np.float32)[None, :] / 256,
        dtype,
    )
    actual = online_attention(
        q,
        k,
        v,
        block_kv=64,
        dtype=dtype,
        scale=0.125,
        causal=True,
        query_start=17,
        query_positions=positions,
    )
    for index, row in enumerate(positions):
        mean = rounded(v[: 18 + row].mean(axis=0), dtype)
        assert np.allclose(actual[index], mean, rtol=0.002 if dtype == "f16" else 0.01, atol=0.001)


def test_head_rows_share_one_decode_query_position():
    np = pytest.importorskip("numpy")
    q = np.zeros((4, 64), dtype=np.float32)
    k = np.zeros((128, 64), dtype=np.float32)
    v = np.broadcast_to(np.arange(128, dtype=np.float32)[:, None] / 16, (128, 64)).copy()
    actual = online_attention(
        q,
        k,
        v,
        block_kv=64,
        dtype="f16",
        scale=0.125,
        causal=True,
        query_start=17,
        query_positions=np.zeros(4, dtype=np.int64),
    )
    assert np.array_equal(actual, np.broadcast_to(rounded(v[:18].mean(axis=0), "f16"), (4, 64)))


def test_online_oracle_rounds_probability_operands_before_pv():
    np = pytest.importorskip("numpy")
    q = np.full((1, 64), 0.8, dtype=np.float32)
    k = np.arange(128 * 64, dtype=np.float32).reshape(128, 64) % 13 / 16
    v = np.arange(128 * 64, dtype=np.float32).reshape(128, 64) % 17 / 32
    scores = np.float32((q @ k.T) * np.float32(0.125))
    probabilities = np.exp(np.float32(scores - scores.max(axis=1, keepdims=True)))
    expected = rounded(
        np.float32(
            (rounded(probabilities, "bf16") @ v)
            / probabilities.sum(axis=1, keepdims=True, dtype=np.float32)
        ),
        "bf16",
    )
    actual = online_attention(q, k, v, block_kv=128, dtype="bf16", scale=0.125)
    assert np.array_equal(actual, expected)

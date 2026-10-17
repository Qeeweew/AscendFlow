"""Independent NumPy online oracle and typed inputs, without device execution.

NumPy is a validation extra, not a compiler/runtime dependency. Large runs may
explicitly sample rows in EVERY query tile/head; that is recorded as sampled
numerical evidence, never presented as a full-output numerical comparison.
"""

from time import perf_counter

from ascendflow.diagnostics import require

from .dtypes import decode, encode


def rounded(values, dtype):
    return decode(encode(values, dtype), dtype, values.shape)


def online_attention(
    q, k, v, *, block_kv, dtype, scale, query_positions=None, causal=False, query_start=0
):
    """FP32 recurrence with P RN before PV and separate rescale/add nodes.

    This is a numerical oracle, not an execution of generated instructions.
    BLAS accumulation and native exp/reduction can have different FP32 errors.
    """
    import numpy as np

    rows, dim = q.shape
    maximum = np.full((rows,), -np.inf, dtype=np.float32)
    denominator = np.zeros((rows,), dtype=np.float32)
    output = np.zeros((rows, dim), dtype=np.float32)
    for first in range(0, k.shape[0], block_kv):
        scores = np.float32((q @ k[first : first + block_kv].T) * np.float32(scale))
        if causal:
            require(query_positions is not None, "causal oracle needs absolute query rows")
            visible = (
                np.arange(first, first + scores.shape[1])[None, :]
                <= query_start + query_positions[:, None]
            )
            scores = np.where(visible, scores, np.float32(-np.inf))
        next_maximum = np.maximum(maximum, np.max(scores, axis=1))
        # Fully masked later tiles contribute zero and retain the prior state.
        alpha = np.exp(np.float32(maximum - next_maximum))
        probabilities = np.exp(np.float32(scores - next_maximum[:, None]))
        partials = probabilities.reshape(rows, -1, 64).sum(axis=2, dtype=np.float32)
        row_sum = partials.sum(axis=1, dtype=np.float32)
        denominator = np.float32(np.float32(denominator * alpha) + row_sum)
        pv = rounded(probabilities, dtype) @ v[first : first + block_kv]
        output = np.float32(np.float32(output * alpha[:, None]) + pv)
        maximum = next_maximum
    return rounded(np.float32(output * np.float32(1 / denominator[:, None])), dtype)


def inputs_and_oracle(policy, sq, sk, *, scope="full", data_case="standard"):
    import numpy as np

    require(sq >= 0 and sq % 128 == 0 and sk > 0 and sk % 1024 == 0, "complete FA tiles required")
    require(
        scope in {"full", "sampled"} and data_case in {"standard", "uniform", "sharp"},
        "unknown numerical validation policy",
    )
    batch, heads, kh = (policy[k] for k in ("batch", "heads", "kv_heads"))
    dtype = policy["dtype"]
    rng = np.random.default_rng(9102201)
    arrays, raw = {}, {}
    for name, shape in [
        ("q", (batch, heads, sq, 128)),
        ("k", (batch, kh, sk, 128)),
        ("v", (batch, kh, sk, 128)),
    ]:
        values = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.5)
        if data_case == "uniform" and name == "k":
            values.fill(0)
        if data_case == "sharp" and name in {"q", "k"}:
            values *= np.float32(3)
        raw[name] = encode(values, dtype)
        arrays[name] = decode(raw[name], dtype, shape)
    selected = (
        np.arange(sq)
        if scope == "full"
        else np.array(
            sorted({q * 128 + r for q in range(sq // 128) for r in (0, 7, 63, 64, 71, 127)}),
            dtype=np.int64,
        )
    )
    start = perf_counter()
    expected = np.empty((batch, heads, len(selected), 128), dtype=np.float32)
    for b in range(batch):
        for h in range(heads):
            for start_row in range(0, len(selected), 128):
                rows = selected[start_row : start_row + 128]
                expected[b, h, start_row : start_row + len(rows)] = online_attention(
                    arrays["q"][b, h, rows],
                    arrays["k"][b, h // (heads // kh)],
                    arrays["v"][b, h // (heads // kh)],
                    block_kv=1024,
                    dtype=dtype,
                    scale=np.float32(1 / np.sqrt(128)),
                )
    raw["o"] = b"\xff\x7f" * (batch * heads * sq * 128)
    raw["score"] = b"\xcd" * (policy["groups"] * 2 * 128 * 1024 * 4)
    raw["p"] = b"\xcd" * (policy["groups"] * 2 * 128 * 1024 * 2)
    raw["pv"] = b"\xcd" * (policy["groups"] * 2 * 128 * 128 * 4)
    return raw, expected, selected, perf_counter() - start

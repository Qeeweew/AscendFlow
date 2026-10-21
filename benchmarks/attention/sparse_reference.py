"""Independent TND/PA_BBND semantics, including speculative verification.

The oracle selects logical rows directly from the pages and evaluates one
FP64 softmax per token/head. It does not follow the DSL's request search,
gather slots, physical pipeline or low precision probability intermediates.
"""

import numpy as np

from benchmarks.validation.dtypes import decode, encode

CASES = ("empty", "window", "mixed", "ragged", "padding", "duplicates", "replay", "sink")


def rounded(values, dtype):
    return decode(encode(values, dtype), dtype, values.shape)


def query_counts(policy, ragged=False):
    b, t = policy["B"], policy["T"]
    counts = np.full(b, t // b, dtype=np.int32)
    counts[: t % b] += 1
    if ragged and b >= 3 and t >= 3:
        size = int(counts[:3].sum())
        counts[:3] = (0, size // 3, size - size // 3)
    return counts


def query_geometry(data, policy, token):
    cu = data["cu_seqlens_q"]
    batch = int(np.searchsorted(cu, token, side="right")) - 1
    count = int(cu[batch + 1] - cu[batch])
    offset = token - int(cu[batch])
    right = max(0, int(data["seqused_ori_kv"][batch]) - count + offset + 1)
    visible = min(policy["WINDOW"], right)
    requested = int(data["ori_topk_length"][token, 0])
    if requested > 0:
        visible = min(visible, requested)
    available = int(data["seqused_cmp_kv"][batch])
    compressed_end = available * policy["RATIO"] + int(data["cmp_residual_kv"][batch])
    cutoff = max(0, compressed_end - count + offset + 1) // policy["RATIO"]
    return batch, right, visible, available, cutoff


def inputs(policy, case, query_lens=None):
    if case not in CASES:
        raise ValueError(f"unknown sparse attention case: {case}")
    b, t, h, d, page, ratio, max_seq, topk = (
        policy[key] for key in ("B", "T", "H", "D", "PAGE", "RATIO", "MAX_SEQ", "TOPK")
    )
    dtype = policy["DTYPE"]
    rng = np.random.default_rng(4102026)
    counts = (
        query_counts(policy, case == "ragged")
        if query_lens is None
        else np.array(query_lens, dtype=np.int32)
    )
    lengths = np.resize(np.array([page - 1, page + 1, 2 * page + 1, max_seq - 1]), b)
    lengths = np.minimum(max_seq, np.maximum(counts, lengths)).astype(np.int32)
    if case == "empty":
        lengths[:] = 0
    data = {
        "q": rounded(rng.normal(0, 0.5, (t, h, d)).astype(np.float32), dtype),
        "ori_kv": rounded(
            rng.normal(0, 0.5, (policy["ORI_BLOCKS"], page, 1, d)).astype(np.float32), dtype
        ),
        "cmp_kv": rounded(
            rng.normal(0, 0.5, (policy["CMP_BLOCKS"], page // ratio, 1, d)).astype(np.float32),
            dtype,
        ),
        "cu_seqlens_q": np.r_[0, counts.cumsum()].astype(np.int32),
        "seqused_ori_kv": lengths,
        "seqused_cmp_kv": lengths // ratio,
        "cmp_residual_kv": lengths % ratio,
        "ori_topk_length": np.zeros((t, 1), dtype=np.int32),
        "cmp_sparse_indices": np.full((t, 1, topk), -1, dtype=np.int32),
        "sinks": np.linspace(-2, 2, h, dtype=np.float32),
    }
    for name, blocks in (
        ("ori_block_table", policy["ORI_BLOCKS"]),
        ("cmp_block_table", policy["CMP_BLOCKS"]),
    ):
        # Read-only cache pages may be shared across requests.
        data[name] = rng.integers(0, blocks, (b, max_seq // page), dtype=np.int32)
    for token in range(t):
        _batch, _right, _visible, available, cutoff = query_geometry(data, policy, token)
        count = min(topk, available, cutoff)
        if case not in {"empty", "window"} and count > 0:
            if case == "padding":
                count = token % (count + 1)
            prefix = rng.permutation(available)[:count]
            if case == "duplicates":
                prefix[:] = 0
            elif case == "mixed":
                # Early queries must filter entries created by later tokens.
                prefix[0] = available - 1
            data["cmp_sparse_indices"][token, 0, :count] = prefix
        if case == "replay":
            data["ori_topk_length"][token, 0] = 1 + token % 7
    if case == "sink":
        data["sinks"][:] = np.resize(np.array([-80, 80, 0, 5], dtype=np.float32), h)
    return data


def selected_rows(data, policy, token):
    page, ratio = policy["PAGE"], policy["RATIO"]
    batch, right, visible, available, cutoff = query_geometry(data, policy, token)
    rows, slots = [], []
    for slot, position in enumerate(range(right - visible, right)):
        physical = data["ori_block_table"][batch, position // page]
        rows.append(data["ori_kv"][physical, position % page, 0])
        slots.append(slot)
    bound = min(policy["TOPK"], available, cutoff)
    for rank, index in enumerate(data["cmp_sparse_indices"][token, 0, :bound]):
        if index < 0:
            break
        if index >= cutoff:
            continue
        physical = data["cmp_block_table"][batch, index // (page // ratio)]
        rows.append(data["cmp_kv"][physical, index % (page // ratio), 0])
        slots.append(policy["WINDOW"] + rank)
    return np.asarray(rows, dtype=np.float32).reshape(-1, policy["D"]), slots


def oracle(data, policy):
    output = np.zeros_like(data["q"])
    for token in range(policy["T"]):
        kv, _ = selected_rows(data, policy, token)
        if not len(kv):
            continue
        scores = (data["q"][token].astype(np.float64) @ kv.astype(np.float64).T) * policy["SCALE"]
        maximum = np.maximum(scores.max(axis=1), data["sinks"])
        weights = np.exp(scores - maximum[:, None])
        denominator = weights.sum(axis=1) + np.exp(data["sinks"] - maximum)
        output[token] = (weights @ kv.astype(np.float64)) / denominator[:, None]
    return rounded(output, policy["DTYPE"])


def gather_oracle(data, policy):
    kv = np.zeros((policy["T"], policy["S"], policy["D"]), dtype=np.float32)
    bias = np.full((policy["T"], policy["S"]), -np.inf, dtype=np.float32)
    for token in range(policy["T"]):
        rows, slots = selected_rows(data, policy, token)
        kv[token, slots] = rows
        bias[token, slots] = 0
    return kv, bias

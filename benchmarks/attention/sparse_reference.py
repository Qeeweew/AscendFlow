"""Independent paged-cache semantics for the V4.1 decode example.

The oracle selects logical rows directly from the input pages, concatenates
both cache planes, and evaluates one FP64 softmax including the head sink.
It does not consume the generated gather output or follow its tile recurrence.
"""

import numpy as np

from benchmarks.validation.dtypes import decode, encode

CASES = ("empty", "window", "mixed", "compressed", "duplicates", "replay", "sink")


def rounded(values, dtype):
    return decode(encode(values, dtype), dtype, values.shape)


def inputs(policy, case):
    if case not in CASES:
        raise ValueError(f"unknown sparse attention case: {case}")
    b, h, d, page, ratio, max_seq, window, topk = (
        policy[key] for key in ("B", "H", "D", "PAGE", "RATIO", "MAX_SEQ", "WINDOW", "TOPK")
    )
    dtype = policy["DTYPE"]
    rng = np.random.default_rng(4102026)
    q = rounded(rng.normal(0, 0.5, (b, h, d)).astype(np.float32), dtype)
    ori = rounded(rng.normal(0, 0.5, (policy["ORI_BLOCKS"], page, d)).astype(np.float32), dtype)
    cmp = rounded(
        rng.normal(0, 0.5, (policy["CMP_BLOCKS"], page // ratio, d)).astype(np.float32), dtype
    )
    tables = {}
    for name, blocks in (("OriTable", policy["ORI_BLOCKS"]), ("CmpTable", policy["CMP_BLOCKS"])):
        tables[name] = rng.integers(0, blocks, (b, max_seq // page), dtype=np.int32)
    lengths = np.array(
        [min(max_seq, x) for x in (0, page - 1, page + 1, max_seq - 1)], dtype=np.int32
    )
    lengths = np.array([max_seq - 1], dtype=np.int32) if b == 1 else np.resize(lengths, b)
    visible = np.minimum(lengths, window)
    counts = np.zeros(b, dtype=np.int32)
    indices = np.full((b, topk), -1, dtype=np.int32)
    sinks = np.linspace(-2, 2, h, dtype=np.float32)
    if case == "empty":
        lengths[:] = 0
        visible[:] = 0
    elif case != "window":
        for batch in range(b):
            available = int(lengths[batch]) // ratio
            count = min(topk, available)
            counts[batch] = count
            # A permutation spans pages, need not be sorted, and ends with
            # -1 padding. Different requests deliberately share physical pages.
            if count:
                indices[batch, :count] = rng.permutation(available)[:count]
                if case == "duplicates":
                    indices[batch, :count] = indices[batch, 0]
                elif case == "mixed":
                    # Valid storage indices beyond the causal cutoff must not
                    # be read or acquire a softmax weight.
                    indices[batch, 0] = max_seq // ratio - 1
        if case == "compressed":
            visible[:] = 0
        if case == "replay":
            visible = np.minimum(visible, np.arange(b, dtype=np.int32) * 3)
        if case == "sink":
            sinks[:] = np.resize(np.array([-80, 80, 0, 5], dtype=np.float32), h)
    return {
        "Q": q,
        "Ori": ori,
        "Cmp": cmp,
        **tables,
        "Lengths": lengths,
        "WindowLengths": visible,
        "Counts": counts,
        "Indices": indices,
        "Sinks": sinks,
    }


def selected_rows(data, policy, batch):
    page, ratio = policy["PAGE"], policy["RATIO"]
    length, visible = int(data["Lengths"][batch]), int(data["WindowLengths"][batch])
    rows, slots = [], []
    for slot, position in enumerate(range(length - visible, length)):
        physical = data["OriTable"][batch, position // page]
        rows.append(data["Ori"][physical, position % page])
        slots.append(slot)
    for rank, index in enumerate(data["Indices"][batch, : data["Counts"][batch]]):
        if index < 0:
            raise ValueError("the selected prefix must contain valid nonnegative indices")
        if index >= length // ratio:
            continue
        physical = data["CmpTable"][batch, index // (page // ratio)]
        rows.append(data["Cmp"][physical, index % (page // ratio)])
        slots.append(policy["WINDOW"] + rank)
    return np.asarray(rows, dtype=np.float32).reshape(-1, policy["D"]), slots


def oracle(data, policy):
    output = np.zeros_like(data["Q"])
    for batch in range(policy["B"]):
        kv, _ = selected_rows(data, policy, batch)
        if not len(kv):
            continue
        scores = (data["Q"][batch].astype(np.float64) @ kv.astype(np.float64).T) * policy["SCALE"]
        maximum = np.maximum(scores.max(axis=1), data["Sinks"])
        weights = np.exp(scores - maximum[:, None])
        denominator = weights.sum(axis=1) + np.exp(data["Sinks"] - maximum)
        output[batch] = (weights @ kv.astype(np.float64)) / denominator[:, None]
    return rounded(output, policy["DTYPE"])


def gather_oracle(data, policy):
    kv = np.zeros((policy["B"], policy["S"], policy["D"]), dtype=np.float32)
    bias = np.full((policy["B"], policy["S"]), -np.inf, dtype=np.float32)
    for batch in range(policy["B"]):
        rows, slots = selected_rows(data, policy, batch)
        kv[batch, slots] = rows
        bias[batch, slots] = 0
    return kv, bias

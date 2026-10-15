"""Independent NumPy online oracle and typed inputs, without device execution.

NumPy is a validation extra, not a compiler/runtime dependency. Large runs may
explicitly sample rows in EVERY query tile/head; that is recorded as sampled
numerical evidence, never presented as a full-output numerical comparison.
"""

from time import perf_counter

from xdsl_npu.model import require


def encode(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    require(dtype in {"f16", "bf16"}, "attention oracle requires an explicit b16 dtype")
    if dtype == "f16":
        return values.astype("<f2").tobytes()
    bits = values.view(np.uint32)
    # RN ties-to-even. Inputs are finite; NaN payload behavior is not assumed.
    upper = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return upper.tobytes()


def decode(data, dtype, shape=None):
    import numpy as np

    if dtype == "f16":
        result = np.frombuffer(data, dtype="<f2").astype(np.float32)
    else:
        require(dtype == "bf16", "attention oracle requires an explicit b16 dtype")
        result = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    return result.reshape(shape) if shape is not None else result


def rounded(values, dtype):
    return decode(encode(values, dtype), dtype, values.shape)


def attention_description(plan):
    from xdsl_npu.ir import StaticDispatchOp
    from xdsl_npu.mixed import extract_mixed

    if isinstance(plan.body.block.first_op, StaticDispatchOp):
        from xdsl_npu.dispatch import extract_dispatch

        dispatch = extract_dispatch(plan)
        mixed = dispatch.cases[0][1]
        for _, other in dispatch.cases:
            require(
                all(
                    other.op.attributes.get(key) == mixed.op.attributes.get(key)
                    for key in (
                        "attention_shape",
                        "attention_tile",
                        "attention_scale",
                        "attention_causal",
                        "attention_start",
                        "attention_packed_decode",
                    )
                ),
                "attention cases disagree on their numerical contract",
            )
        return dispatch.roots, mixed.op.attributes, dispatch.op.sym_name.data
    mixed = extract_mixed(plan)
    return mixed.roots, mixed.op.attributes, mixed.op.sym_name.data


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
        # Rows with no visible key in this tile retain their prior state. The
        # factory omits tiles masked for the whole physical query group.
        alpha = np.exp(np.float32(maximum - next_maximum))
        probabilities = np.exp(np.float32(scores - next_maximum[:, None]))
        partials = probabilities.reshape(rows, -1, 64).sum(axis=2, dtype=np.float32)
        row_sum = partials.sum(axis=1, dtype=np.float32)
        denominator = np.float32(np.float32(denominator * alpha) + row_sum)
        pv = rounded(probabilities, dtype) @ v[first : first + block_kv]
        output = np.float32(np.float32(output * alpha[:, None]) + pv)
        maximum = next_maximum
    return rounded(np.float32(output * np.float32(1 / denominator[:, None])), dtype)


def attention_inputs(plan, *, scope="full", data_case="standard"):
    import numpy as np

    roots, attrs, name = attention_description(plan)
    require(name == "flash_attention", "expected the forward attention recipe")
    shape = tuple(n.data for n in attrs["attention_shape"])
    block_q, block_kv, _, _ = (n.data for n in attrs["attention_tile"])
    batch, heads, kv_heads, sq, sk, dim = shape
    dtype = roots["q"].ref.type.dtype.data
    scale = attrs["attention_scale"].value.data
    causal, query_start = bool(attrs["attention_causal"].data), attrs["attention_start"].data
    require(scope in {"full", "sampled"}, "attention oracle scope must be full or sampled")
    require(data_case in {"standard", "sharp", "uniform"}, "unknown attention input case")
    rng = np.random.default_rng(9102201)
    raw = {}
    arrays = {}
    for name, size in (
        ("q", (batch, heads, sq, dim)),
        ("k", (batch, kv_heads, sk, dim)),
        ("v", (batch, kv_heads, sk, dim)),
    ):
        values = rng.standard_normal(size, dtype=np.float32) * np.float32(0.5)
        if data_case == "sharp" and name in {"q", "k"}:
            values *= np.float32(3)
        if data_case == "uniform" and name == "k":
            values.fill(0)
        raw[name] = encode(values, dtype)
        arrays[name] = decode(raw[name], dtype, size)
        require(
            len(raw[name]) == roots[name].nbytes.data,
            "attention input metadata/footprint mismatch",
        )
    if scope == "full" or sq == 1:
        selected = np.arange(sq)
    else:
        # Both AIV owners and every query tile, plus sequence endpoints.
        native_tile = attrs.get("attention_native_tile")
        sample_step = native_tile.data[1].data if native_tile is not None else block_q // 2
        selected = np.array(
            sorted(
                {
                    row
                    for tile in range(sq // block_q)
                    for row in (
                        *(tile * block_q + offset for offset in range(0, block_q, sample_step)),
                        tile * block_q + block_q - 1,
                    )
                }
            )
        )
    expected = np.empty((batch, heads, len(selected), dim), dtype=np.float32)
    start = perf_counter()
    for b in range(batch):
        for h in range(heads):
            kh = h // (heads // kv_heads)
            # Bounded query blocks bound host temporary scores independently
            # of Sq; all KV blocks/rounding boundaries are still visited.
            for first in range(0, len(selected), block_q):
                indices = selected[first : first + block_q]
                expected[b, h, first : first + len(indices)] = online_attention(
                    arrays["q"][b, h, indices],
                    arrays["k"][b, kh],
                    arrays["v"][b, kh],
                    block_kv=block_kv,
                    dtype=dtype,
                    scale=scale,
                    causal=causal,
                    query_start=query_start,
                    query_positions=indices,
                )
    for name, root in roots.items():
        if name not in raw:
            raw[name] = (b"\xff\x7f" if name == "output" else b"\xcd\xcd") * (root.nbytes.data // 2)
    report = {
        "shape_b_hq_hkv_sq_sk_d": list(shape),
        "dtype": dtype,
        "block_q": block_q,
        "logical_block_kv": block_kv,
        "softmax_scale_f32": scale,
        "oracle_scope": scope,
        "query_rows_per_head": selected.tolist(),
        "numerical_elements": int(expected.size),
        "oracle_seconds": perf_counter() - start,
        "seed": 9102201,
        "data_case": data_case,
        "causal": causal,
        "query_start": query_start,
        "oracle": "NumPy FP32 online recurrence, FP32 denominator, RN operand P and output",
        "recipe_policies": [n.data for n in attrs.get("attention_policies", ())],
        "native_cube_kv_vector_rows": [n.data for n in attrs.get("attention_native_tile", ())],
    }
    return raw, expected, selected, report


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

"""Independent NumPy online oracle, typed I/O and reproducible ACL FA runs.

NumPy is a validation extra, not a compiler/runtime dependency. Large runs may
explicitly sample rows in EVERY query tile/head; that is recorded as sampled
numerical evidence, never presented as a full-output numerical comparison.
"""

import ctypes as ct
import hashlib
import json
from pathlib import Path
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


def run_attention(
    plan,
    shared_library: Path,
    output: Path,
    *,
    physical,
    repeats=30,
    warmup=5,
    scope="full",
    data_case="standard",
    backend="generated",
):
    import numpy as np

    from xdsl_npu.codegen import emit_asc
    from xdsl_npu.device import ACL, health_snapshot
    from xdsl_npu.sync import verify_plan

    verification = verify_plan(plan)
    roots, attrs, name = attention_description(plan)
    require(backend in {"generated", "aclnn"}, "attention backend must be generated or aclnn")
    require(
        backend != "aclnn" or not attrs["attention_causal"].data,
        "the ACLNN comparison currently implements the dense BNSD variant",
    )
    manifest = json.loads(shared_library.with_suffix(".compile.json").read_text())
    sha = lambda data: hashlib.sha256(data).hexdigest()
    require(
        sha(shared_library.read_bytes()) == manifest["shared_sha256"],
        "attention library artifact hash mismatch",
    )
    require(
        sha(emit_asc(plan).encode()) == manifest["source_sha256"],
        "attention source hash mismatch; recompile",
    )
    require(repeats > 0 and warmup > 0, "attention needs positive warmup and repeats")
    raw, expected, selected, report = attention_inputs(plan, scope=scope, data_case=data_case)
    # Snapshot ALL cards after host preparation, immediately before ACL use.
    health = health_snapshot(physical, output)
    runtime = ACL()
    baseline = None
    try:
        parameters = list(roots)
        selected_parameters = parameters if backend == "generated" else ["q", "k", "v", "output"]
        pointers = {root: runtime.allocate(raw[root]) for root in selected_parameters}
        if backend == "generated":
            library = ct.CDLL(str(shared_library.resolve()))
            kernel = getattr(library, f"launch_{name}")
            kernel.argtypes, kernel.restype = [ct.c_void_p] * (len(parameters) + 1), None
            args = [pointers[root] for root in parameters]

            def launch():
                kernel(runtime.stream, *args)
        else:
            from attention_reference import FlashAttentionReference

            baseline = FlashAttentionReference(
                runtime,
                *(pointers[root] for root in selected_parameters),
                report["shape_b_hq_hkv_sq_sk_d"],
                report["dtype"],
                report["softmax_scale_f32"],
            )
            launch = baseline
        for _ in range(warmup):
            launch()
        runtime.synchronize()
        actual_bytes = runtime.read(pointers["output"], len(raw["output"]))
        (output / "device-output.bin").write_bytes(actual_bytes)
        b, h, _, sq, _, d = report["shape_b_hq_hkv_sq_sk_d"]
        actual = decode(actual_bytes, report["dtype"], (b, h, sq, d))
        require(np.isfinite(actual).all(), "attention produced nonfinite/unwritten output")
        checked = actual[:, :, selected]
        # Includes native exp/FP32 reduction error and final b16 rounding.
        rtol, atol = (0.002, 0.0002) if report["dtype"] == "f16" else (0.01, 0.001)
        error = float(np.max(np.abs(checked - expected)))
        if not np.allclose(checked, expected, rtol=rtol, atol=atol):
            failed = np.argwhere(np.abs(checked - expected) > atol + rtol * np.abs(expected))[0]
            index = tuple(failed)
            raise ValueError(
                f"attention numerical mismatch at {index}: actual={checked[index]}, expected={expected[index]}, max_abs_error={error}"
            )
        start, stop = runtime.event(), runtime.event()
        runtime.record(start)
        for _ in range(repeats):
            launch()
        runtime.record(stop)
        runtime.synchronize()
        elapsed = ct.c_float()
        runtime.call(
            "aclrtEventElapsedTime",
            [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            ct.byref(elapsed),
            start,
            stop,
        )
        report.update(
            kernel=name,
            backend=backend,
            health=health,
            verification=verification,
            warmup=warmup,
            repeats=repeats,
            max_abs_error=error,
            rtol=rtol,
            atol=atol,
            finite_output_elements=int(actual.size),
            stream_interval_us_per_launch=elapsed.value * 1000 / repeats,
            timing_scope="ACL event stream interval; includes submission gaps; not kernel-only",
            compiled_source_sha256=manifest["source_sha256"],
            shared_sha256=manifest["shared_sha256"],
            argument_sha256={root: sha(raw[root]) for root in parameters},
            library=str(shared_library.resolve()),
        )
        if baseline is not None:
            import attention_reference

            from xdsl_npu.toolchain import CANN

            report.update(
                reference_interface=baseline.interface,
                reference_workspace_bytes=baseline.workspace_size.value,
                reference_auxiliary_bytes=baseline.auxiliary_bytes,
                reference_adapter_sha256=sha(Path(attention_reference.__file__).read_bytes()),
                reference_opapi_sha256=sha((CANN / "lib64/libopapi.so").read_bytes()),
                reference_numerical_contract="CANN native tiling/rounding may differ from the specified logical BK; comparison uses stated tolerances",
            )
        (output / "device.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if baseline is not None:
            baseline.close()
        runtime.close()

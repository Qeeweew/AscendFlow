"""Device execution of legacy FA; mathematical validation lives separately."""

import ctypes as ct
import hashlib
import json
from pathlib import Path

from benchmarks.validation.attention import attention_description, attention_inputs, decode
from xdsl_npu.model import require


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
            from benchmarks.references.attention_aclnn import FlashAttentionReference

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
            import benchmarks.references.attention_aclnn as attention_reference
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

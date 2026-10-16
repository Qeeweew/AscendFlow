"""Fixed-policy normalization/softmax: generated ASC vs AscendC and optional ACLNN.

Profile one selected variant with --rows 256 --warmup 5 --repeats 30. With
GROUPS=32 and D=4096 this is an 8192x4096 f32 input. Softmax --rows 1024
uses 32768x1024 f32; --rows always counts rows per group, not native tiles.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from benchmarks.artifacts import read_artifact
from benchmarks.references import ascendc_source
from benchmarks.work import structure_inventory, work_record
from examples.normalization.rms_norm import rms_normalize
from examples.normalization.softmax import softmax
from xdsl_npu.backend.cann import CANN
from xdsl_npu.build import build_module
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.diagnostics import require
from xdsl_npu.performance import hardware_snapshot
from xdsl_npu.target.scalars import round_f32


def oracle(values, columns, epsilon):
    import numpy as np

    output = np.empty_like(values)
    # Bound host temporary memory independently of the number of rows.
    for begin in range(0, len(values), 128):
        block = values[begin : begin + 128]
        squared = np.multiply(block, block, dtype=np.float32)
        partial = (
            squared.reshape(-1, columns // 64, 64).sum(axis=2, dtype=np.float64).astype(np.float32)
        )
        total = partial.sum(axis=1, dtype=np.float64).astype(np.float32)
        mean = np.multiply(total, np.float32(1 / columns), dtype=np.float32)
        adjusted = np.add(mean, np.float32(epsilon), dtype=np.float32)
        root = np.sqrt(adjusted, dtype=np.float32)
        inverse = np.divide(np.float32(1), root, dtype=np.float32)
        output[begin : begin + len(block)] = np.multiply(block, inverse[:, None], dtype=np.float32)
    return output


def softmax_oracle(values):
    import numpy as np

    output = np.empty_like(values)
    for begin in range(0, len(values), 128):
        block = values[begin : begin + 128].astype(np.float64)
        exponential = np.exp(block - block.max(axis=1, keepdims=True))
        output[begin : begin + len(block)] = exponential / exponential.sum(axis=1, keepdims=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", choices=["rms", "softmax"], default="rms")
    parser.add_argument(
        "--variant", choices=["generated", "ascendc", "aclnn", "both", "all"], default="both"
    )
    parser.add_argument("--columns", type=int)
    parser.add_argument("--tile-rows", type=int, default=8)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path, help="unique per-run device evidence JSON")
    parser.add_argument("--rows", type=int, nargs="+", help="rows per execution group")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    is_softmax = args.operator == "softmax"
    args.columns = args.columns or (1024 if is_softmax else 4096)
    args.rows = (
        args.rows
        if args.rows is not None
        else ([0, 8, 16, 24, 1024] if is_softmax else [0, 1, 2, 4, 256])
    )
    require(
        args.columns in {64, 128, 256, 512, 1024, 2048, 4096}
        and 1 <= args.groups <= 40
        and all(row >= 0 for row in args.rows)
        and args.warmup >= 0
        and args.repeats > 0,
        "invalid fixed test policy",
    )
    require(
        is_softmax or args.variant not in {"aclnn", "all"}, "ACLNN comparison here requires softmax"
    )
    require(
        not is_softmax
        or (
            args.tile_rows % 8 == 0
            and args.tile_rows > 0
            and args.columns <= 1024
            and all(row % args.tile_rows == 0 for row in args.rows)
        ),
        "softmax needs complete row tiles, multiple-of-eight tile rows and columns<=1024",
    )
    require(
        args.variant not in {"aclnn", "all"} or all(row > 0 for row in args.rows),
        "ACLNN comparison uses nonempty shapes; test zero tiles with generated/ascendc",
    )
    definition = softmax if is_softmax else rms_normalize
    parameters = {"D": args.columns, "GROUPS": args.groups}
    if is_softmax:
        parameters["R"] = args.tile_rows
    variants = ["generated", "ascendc"] if args.variant in {"both", "all"} else [args.variant]
    if args.variant == "all":
        variants.append("aclnn")
    build_generated = args.build and "generated" in variants
    generated = args.output / "generated"
    generated_check = generated if build_generated else generated / "source-check"
    build_module(
        lambda: definition.specialize(**parameters),
        generated_check,
        shared=build_generated,
        timeout_s=args.timeout,
    )
    result = read_artifact(generated_check)
    inventory = structure_inventory(result.plan)
    sources = {"generated": (result.name, result.asc)}

    def reference_source():
        return (
            ascendc_source("ascendc_softmax.asc.in" if is_softmax else "ascendc_normalize.asc.in")
            .replace("@D@", str(args.columns))
            .replace("@GROUPS@", str(args.groups))
            .replace("@EPS@", repr(round_f32(1e-6)))
            .replace("@R@", str(args.tile_rows))
        )

    manifests = {}
    for variant in variants:
        if variant == "aclnn":
            manifests[variant] = {
                "interface": "aclnnSoftmax",
                "cann": str(CANN),
                "libraries": {
                    name: hashlib.sha256((CANN / "lib64" / name).read_bytes()).hexdigest()
                    for name in ("libopapi.so", "libnnopbase.so")
                },
            }
            continue
        directory = (args.output / variant).resolve()
        if variant == "ascendc":
            check = directory if args.build else directory / "source-check"
            build_module(
                reference_source,
                check,
                name="softmax_reference" if is_softmax else "normalize_reference",
                shared=args.build,
                timeout_s=args.timeout,
            )
            reference = read_artifact(check)
            sources[variant] = reference.name, reference.asc
        name, source = sources[variant]
        if args.physical is None:
            continue
        manifest = json.loads((directory / f"{name}.compile.json").read_text())
        require(
            hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"],
            "source differs from CANN build",
        )
        require(
            hashlib.sha256((directory / f"{name}.so").read_bytes()).hexdigest()
            == manifest["shared_sha256"],
            "library differs from manifest",
        )
        manifests[variant] = manifest
    if args.physical is None:
        print(result.timings)
        return
    record_path = args.record or args.output / f"device-{args.variant}.json"
    health = health_snapshot(args.physical, record_path.parent)
    hardware = hardware_snapshot(args.physical)
    import numpy as np

    runtime = ACL()
    records = []
    try:
        for variant in variants:
            name = "aclnnSoftmax" if variant == "aclnn" else sources[variant][0]
            launch = None
            if variant != "aclnn":
                library = ct.CDLL(str((args.output / variant / f"{name}.so").resolve()))
                launch = getattr(library, "launch_" + name)
                launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int64]
                launch.restype = ct.c_int
            for rows in args.rows:
                values = (
                    ((np.arange(args.groups * rows * args.columns, dtype=np.int32) % 127 - 63) / 16)
                    .astype(np.float32)
                    .reshape(-1, args.columns)
                )
                expected = (
                    softmax_oracle(values) if is_softmax else oracle(values, args.columns, 1e-6)
                )
                guard = b"\x5a" * 128
                raw = values.tobytes() + guard
                x, y = runtime.allocate(raw), runtime.allocate(b"\xcd" * values.nbytes + guard)
                reference = None
                count = rows // args.tile_rows if is_softmax else rows
                if variant == "aclnn":
                    from benchmarks.references.aclnn import SoftmaxReference

                    reference = SoftmaxReference(runtime, x, y, len(values), args.columns)

                def invoke(reference=reference, launch=launch, x=x, y=y, count=count):
                    if reference is not None:
                        reference()
                    else:
                        require(
                            launch(runtime.stream, x, y, count) == 0, "launch contract rejected"
                        )

                for _ in range(args.warmup):
                    invoke()
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    invoke()
                runtime.record(stop)
                runtime.synchronize()
                actual = runtime.read(y, values.nbytes + len(guard))
                numeric = np.frombuffer(actual[: values.nbytes], dtype=np.float32).reshape(
                    expected.shape
                )
                require(
                    np.allclose(numeric, expected, rtol=5e-5, atol=5e-6),
                    "normalization differs from independent oracle",
                )
                require(
                    actual[values.nbytes :] == guard and runtime.read(x, len(raw)) == raw,
                    "input/output guards changed",
                )
                elapsed = ct.c_float()
                runtime.call(
                    "aclrtEventElapsedTime",
                    [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
                    ct.byref(elapsed),
                    start,
                    stop,
                )
                record = {
                    "variant": variant,
                    "kernel": name,
                    "shape": list(values.shape),
                    "dtype": "f32",
                    "operator": args.operator,
                    "tile_rows": args.tile_rows if is_softmax else 1,
                    "rows_per_group": rows,
                    "max_abs_error": float(np.max(np.abs(numeric - expected)))
                    if numeric.size
                    else 0,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed.value * 1000 / args.repeats,
                    "timing_scope": "ACL stream interval including submission gaps, not kernel-only",
                    "cache_policy": "same input/output allocations reused; no flush",
                    "input_sha256": hashlib.sha256(raw).hexdigest(),
                    "output_sha256": hashlib.sha256(actual).hexdigest(),
                    "work": work_record(
                        inventory,
                        algorithm_counts={
                            "add_mul_operations": len(values)
                            * (2 * args.columns - 1 if is_softmax else 3 * args.columns + 1),
                            "max_comparisons": len(values) * (args.columns - 1)
                            if is_softmax
                            else 0,
                            "exp_elements": values.size if is_softmax else 0,
                            "sqrt_elements": 0 if is_softmax else len(values),
                            "div_elements": values.size if is_softmax else len(values),
                            "logical_input_bytes": values.nbytes,
                            "logical_output_bytes": expected.nbytes,
                        },
                        algorithm_scope="scalar normalization/softmax problem convention; not Vector recipe work or memory requests",
                    ),
                }
                if is_softmax:
                    record["max_row_sum_error"] = (
                        float(np.max(np.abs(numeric.sum(axis=1, dtype=np.float64) - 1)))
                        if numeric.size
                        else 0
                    )
                    require(np.isfinite(numeric).all(), "softmax has nonfinite output")
                if reference is not None:
                    reference.close()
                records.append(record)
                print(
                    json.dumps({key: value for key, value in record.items() if key != "work"}),
                    flush=True,
                )
    finally:
        runtime.close()
    report = {
        "health": health,
        "hardware": hardware,
        "variants": manifests,
        "cases": records,
        "comparison_scope": (
            "stable last-axis FP32 softmax; AscendC fixed schedule, ACLNN independently chooses policy/order"
            if is_softmax
            else "same normalization primitive and fixed schedule; excludes gamma/bias; not an ACLNN RMSNorm replacement"
        ),
        "numerical_contract": (
            "max, subtract, exp, sum, div; generated/AscendC R64 partial + packed merge; full float64 oracle rtol=5e-5 atol=5e-6"
            if is_softmax
            else "R64 partial + native merge, FP32 mean/epsilon/sqrt/div/scale; native floating reduction checked with stated tolerance"
        ),
        "dsl_source_sha256": hashlib.sha256(definition.source.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compilation": result.timings,
    }
    record_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

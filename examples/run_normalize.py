"""Fixed-policy normalization: generated ASC vs independent AscendC source.

Profile one selected variant with --rows 256 --warmup 5 --repeats 30. With
GROUPS=32 and D=4096 this is an 8192x4096 f32 input, not a tiny timing case.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ast_normalize import rms_normalize

from xdsl_npu import compile
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require, round_f32
from xdsl_npu.performance import hardware_snapshot, runtime_work
from xdsl_npu.toolchain import offline_compile


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path, help="unique per-run device evidence JSON")
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 4, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(
        args.columns in {64, 128, 256, 512, 1024, 2048, 4096}
        and 1 <= args.groups <= 40
        and all(row >= 0 for row in args.rows)
        and args.warmup >= 0
        and args.repeats > 0,
        "invalid fixed test policy",
    )
    result = compile(rms_normalize.specialize(D=args.columns, GROUPS=args.groups))
    if args.build or args.physical is None:
        result.write(args.output / "generated")
    manual = Path(__file__).with_name("ascendc_normalize.asc.in").read_text()
    manual = (
        manual.replace("@D@", str(args.columns))
        .replace("@GROUPS@", str(args.groups))
        .replace("@EPS@", repr(round_f32(1e-6)))
    )
    sources = {"generated": (result.name, result.asc), "ascendc": ("normalize_reference", manual)}
    variants = list(sources) if args.variant == "both" else [args.variant]
    manifests = {}
    for variant in variants:
        name, source = sources[variant]
        directory = (args.output / variant).resolve()
        if args.build:
            offline_compile(source, directory, name, shared=True)
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
            name, _ = sources[variant]
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
                expected = oracle(values, args.columns, 1e-6)
                guard = b"\x5a" * 128
                raw = values.tobytes() + guard
                x, y = runtime.allocate(raw), runtime.allocate(b"\xcd" * values.nbytes + guard)
                for _ in range(args.warmup):
                    require(launch(runtime.stream, x, y, rows) == 0, "launch contract rejected")
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    require(launch(runtime.stream, x, y, rows) == 0, "launch contract rejected")
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
                    "requested_work": runtime_work(result.plan, {"rows": rows}),
                    "work_basis": "generated physical IR; reference has the same data/numerical contract, but sync counts are not inferred from C++",
                }
                records.append(record)
                print(
                    json.dumps(
                        {key: value for key, value in record.items() if key != "requested_work"}
                    ),
                    flush=True,
                )
    finally:
        runtime.close()
    report = {
        "health": health,
        "hardware": hardware,
        "variants": manifests,
        "cases": records,
        "comparison_scope": "same normalization primitive and fixed schedule; excludes gamma/bias; not an ACLNN RMSNorm replacement",
        "numerical_contract": "R64 partial + native merge, FP32 mean/epsilon/sqrt/div/scale; native floating reduction checked with stated tolerance",
        "compilation": result.timings,
    }
    record_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

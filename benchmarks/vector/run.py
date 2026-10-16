"""Build once, run an AST dynamic-length three-stage Vector pipeline.

Set ASCEND_RT_VISIBLE_DEVICES=<physical> and ASCEND_DEVICE_ID=0 before Python.
The length counts complete 64-element tiles per group; native tile sizes are
static. Device timings are diagnostic stream intervals, not tuned throughput.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from benchmarks.artifacts import read_artifact
from benchmarks.work import structure_inventory, work_record
from xdsl_npu.build import build_file
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.diagnostics import require


def run(result, library, output, physical, counts, repeats, *, grouped=False, warmup=5):
    import numpy as np

    require(
        all(count >= 0 for count in counts) and repeats > 0 and warmup > 0,
        "nonnegative lengths and positive repeats required",
    )
    manifest = json.loads(library.with_suffix(".compile.json").read_text())
    require(
        hashlib.sha256(result.asc.encode()).hexdigest() == manifest["source_sha256"],
        "compiled library source does not match this compilation",
    )
    require(
        hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "compiled library does not match its manifest",
    )
    health = health_snapshot(physical, output)
    runtime = ACL()
    records = []
    inventory = structure_inventory(result.plan)
    groups = inventory["groups"]
    require(len(inventory["runtime_parameters"]) == 1, "Vector runner needs one runtime parameter")
    if grouped:
        require(
            result.name == "grouped_scale" and groups == 20,
            "runner requires the declared grouped example policy",
        )
    try:
        module = ct.CDLL(str(library.resolve()))
        launch = getattr(module, f"launch_{result.name}")
        launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int64]
        launch.restype = ct.c_int
        guard = b"\x5a" * 128
        for count in counts:
            shape = (2, 3, count, 64) if grouped else (groups, count, 64)
            elements = 6 * count * 64 if grouped else groups * count * 64
            values = ((np.arange(elements, dtype=np.int32) % 127 - 63) / 16).astype(np.float32)
            raw_input = values.tobytes() + guard
            raw_output = b"\xcd" * values.nbytes + guard
            x, y = runtime.allocate(raw_input), runtime.allocate(raw_output)
            require(launch(runtime.stream, x, y, count) == 0, "runtime length contract rejected")
            runtime.synchronize()
            raw = runtime.read(y, len(raw_output))
            actual = np.frombuffer(raw[: values.nbytes], dtype=np.float32)
            require(
                np.array_equal(actual, values * np.float32(2)),
                "device result differs from FP32 scale oracle",
            )
            require(
                raw[values.nbytes :] == guard, "output guard changed, including empty-loop case"
            )
            require(runtime.read(x, len(raw_input)) == raw_input, "read-only input changed")
            for _ in range(warmup - 1):
                require(
                    launch(runtime.stream, x, y, count) == 0, "runtime length contract rejected"
                )
            runtime.synchronize()
            start, stop = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(repeats):
                require(
                    launch(runtime.stream, x, y, count) == 0, "runtime length contract rejected"
                )
            runtime.record(stop)
            runtime.synchronize()
            raw = runtime.read(y, len(raw_output))
            require(
                np.array_equal(
                    np.frombuffer(raw[: values.nbytes], dtype=np.float32), values * np.float32(2)
                )
                and raw[values.nbytes :] == guard,
                "repeated launches changed numerical output or bounds guard",
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
                "sequence_length" if grouped else "tiles_per_group": count,
                "shape": list(shape),
                "dtype": "f32",
                "elements": values.size,
                "max_abs_error": 0.0,
                "empty_loop_and_bounds_guards": "checked",
                "warmup": warmup,
                "repeats": repeats,
                "stream_interval_us_per_launch": elapsed.value * 1000 / repeats,
                "timing_scope": "ACL event stream interval, including submission gaps; not kernel-only",
                "input_sha256": hashlib.sha256(raw_input).hexdigest(),
                "work": work_record(
                    inventory,
                    algorithm_counts={
                        "fp32_multiply_operations": values.size,
                        "logical_input_bytes": values.nbytes,
                        "logical_output_bytes": values.nbytes,
                    },
                    algorithm_scope="elementwise FP32 scale problem; no native or traffic inference",
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        require(
            launch(runtime.stream, x, y, -1) == -1, "negative length must be rejected before launch"
        )
    finally:
        runtime.close()
    report = {
        "health": health,
        "kernel": result.name,
        "library": str(library.resolve()),
        "source_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "verification": result.report,
        "compilation": result.timings,
        "same_binary_for_all_lengths": True,
        "cases": records,
    }
    (output / "device.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("scale", "grouped"), default="scale")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument(
        "--library", type=Path, help="reuse an exactly matching compiled shared library"
    )
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    grouped = args.example == "grouped"
    counts = args.counts or ([16, 48, 64, 4096, 131072] if grouped else [0, 1, 2, 4, 4096, 65536])
    # A source check for --library must not invalidate that library's existing
    # manifest. Every build directory still belongs to one operator build.
    build_output = args.output if args.library is None else args.output / "source-check"
    build_file(
        Path(__file__).resolve().parents[2] / f"examples/vector/{args.example}.py",
        build_output,
        entry="grouped_scale" if grouped else "scale",
        compile_asc=args.library is None and args.offline,
        shared=args.library is None and args.physical is not None,
        timeout_s=args.timeout,
    )
    # Device execution only consumes published ASC/metadata; it does not need
    # to compile the same source again or manufacture a verified mutable IR.
    result = read_artifact(build_output)
    library = args.library
    if args.physical is not None and library is None:
        library = args.output / f"{result.name}.so"
    if args.physical is not None:
        run(result, library, args.output, args.physical, counts, args.repeats, grouped=grouped)
    else:
        print(json.dumps({"verification": result.report, "compilation": result.timings}, indent=2))


if __name__ == "__main__":
    main()

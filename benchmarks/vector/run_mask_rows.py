"""Build or reuse the row-mask example and check it with an independent oracle.

Build and publish an offline shared library (one hard 300 s build deadline)::

    python -m benchmarks.vector.run_mask_rows --build --output build/mask_rows

Re-read a published artifact without invoking the compiler, then run it on a
healthy card explicitly mapped to logical device 0::

    ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 \
    python -m benchmarks.vector.run_mask_rows --physical 2 \
        --artifact build/mask_rows/artifact \
        --library build/mask_rows/artifact/mask_rows.so \
        --output reports/mask-rows-device

With ``--build`` the artifact defaults to ``--output/artifact`` and evidence
goes to ``--output``; with ``--library`` the artifact defaults to the library's
directory. Only ``--build`` invokes the compiler or CANN; a reused library is
checked against its published ASC, source snapshot and manifest. ``--rows`` and
``--columns`` must match the dimensions recorded in that source snapshot.

The oracle is independent Python-int lane logic (no compiler predicate
interpreter, no NumPy int64 wrap): ``cols < hi``, ``cols != lo`` and
``rows >= cols - q`` with exact arithmetic for nonnegative int64 values. The
default cases exercise prefix/trailing, complement inside/outside, empty/full
and negative-unit row prefixes. For the large 1x16448 shape use e.g. ``--rows 1
--columns 16448 --cases 3:100:16447 16382:16400:16447 0:16448:0``; the first
selects 256 whole 64-lane blocks plus a 63-lane tail.

This records numerical and identity evidence only: no kernel-time or
performance comparison.
"""

import argparse
import ctypes as ct
import hashlib
import json
import re
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from benchmarks.artifacts import read_artifact
from benchmarks.vector.run_masks import published_manifest, resolve_shape

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/vector/mask_rows.py"
ENTRY = "mask_rows"
INT64_MAX = (1 << 63) - 1
BUFFERS = ("X", "Y0", "Y1", "Y2")
DEFAULT_CASES = (
    "3:100:127",  # prefix over most columns, point complement inside
    "129:65:65",  # prefix past the default width, complement outside
    "0:0:0",  # empty prefix, complement excluding column 0
    "300:1000000:1000000",  # full prefix and complement, row origin beyond width
)
EXPECTED_LAUNCH = (
    "void* stream",
    "void* arg_X",
    "void* arg_Y0",
    "void* arg_Y1",
    "void* arg_Y2",
    "int64_t npu_size_q",
    "int64_t npu_size_lo",
    "int64_t npu_size_hi",
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_cases(values):
    cases = []
    for value in values:
        parts = value.split(":")
        require(len(parts) == 3, f"case {value!r} must be q:lo:hi")
        case = tuple(int(part) for part in parts)
        require(
            all(0 <= item <= INT64_MAX for item in case),
            f"case {value!r} must use nonnegative signed int64 runtime values",
        )
        cases.append(case)
    return tuple(cases)


def field_tensor(rows, columns):
    """Deterministic f32 input whose lanes never equal the three fill constants."""
    import numpy as np

    base = np.arange(rows * columns, dtype=np.int64)
    values = ((base * 13 + 7) % 199).astype(np.float32) / np.float32(4) - np.float32(124.75)
    values = values.reshape(rows, columns)
    require(values.shape == (rows, columns), "input does not match the published shape")
    require(not np.isin(values, np.float32([-7.0, -9.0, -11.0])).any(), "input hits a fill value")
    return values


def oracle(x, q, lo, hi):
    """Independent Python-int lane conditions; untouched lanes keep the input."""
    import numpy as np

    rows, columns = x.shape
    require(rows > 0 and columns > 0, "positive shape required")
    prefix = np.array([c < hi for c in range(columns)], dtype=bool)
    complement = np.array([c != lo for c in range(columns)], dtype=bool)
    # Python ints keep ``cols - q`` exact at the signed int64 extremes.
    row_prefix = np.array([[c <= r + q for c in range(columns)] for r in range(rows)], dtype=bool)
    prefix = np.broadcast_to(prefix, (rows, columns))
    complement = np.broadcast_to(complement, (rows, columns))
    y0 = np.where(prefix, np.float32(-7), x).astype(np.float32)
    y1 = np.where(complement, np.float32(-9), x).astype(np.float32)
    y2 = np.where(row_prefix, np.float32(-11), x).astype(np.float32)
    return y0, y1, y2


def validate_snapshot_shapes(directory, shape):
    """All four external snapshot allocations must be the f32 shape we parse."""
    path = Path(directory) / "source.xdsl"
    require(path.is_file(), f"missing published source snapshot: {path}")
    text = path.read_text()
    for name in BUFFERS:
        lines = [
            line
            for line in text.splitlines()
            if '"npu_source.alloc"' in line and f'sym_name = "{name}"' in line
        ]
        require(len(lines) == 1, f"published snapshot must declare exactly one {name} allocation")
        tensor = re.search(r"!npu_source\.tensor<\[([^\]]*)\]", lines[0])
        require(tensor is not None, f"published {name} allocation has no tensor type")
        dimensions = tuple(
            int(value) for value in re.findall(r"#builtin\.int<(-?\d+)>", tensor.group(1))
        )
        require(dimensions == shape, f"published {name} dimensions {dimensions} != {shape}")
        require('"f32"' in lines[0] and '"GM"' in lines[0], f"published {name} is not f32 GM")


def validate_plan_buffers(module, shape):
    """Cross-check the snapshot shape against compiled flat external buffers."""
    from ascendflow.dialects.physical import BufferOp

    rows, columns = shape
    buffers = {
        operation.sym_name.data: operation.nbytes.data
        for operation in module.walk()
        if isinstance(operation, BufferOp)
    }
    require(set(BUFFERS) <= set(buffers), "published plan lacks the expected external buffers")
    require(
        all(buffers[name] == rows * columns * 4 for name in BUFFERS),
        "source-snapshot shape does not match the published flat buffer sizes",
    )


def launch_signature(asc):
    """The emitted ABI must be stream, X, Y0, Y1, Y2, q, lo, hi in that order."""
    found = re.search(r"int launch_mask_rows\(([^)]*)\)", asc)
    require(found is not None, "published ASC has no launch_mask_rows signature")
    return tuple(part.strip() for part in found.group(1).split(","))


def provenance_record(directory, result, library, manifest):
    snapshot = Path(directory) / "source.xdsl"
    plan = Path(directory) / "after.xdsl"
    require(plan.is_file(), f"missing published plan IR: {plan}")
    return {
        "artifact_directory": str(Path(directory).resolve()),
        "library": str(Path(library).resolve()),
        "manifest": str(Path(library).with_suffix(".compile.json").resolve()),
        "kernel": result.name,
        "target": manifest.get("target"),
        "asc_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "source_snapshot_sha256": _digest(snapshot.read_bytes()),
        "plan_sha256": _digest(plan.read_bytes()),
        "example_path": str(SOURCE.relative_to(ROOT)),
        "example_sha256": _digest(SOURCE.read_bytes()),
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": _digest(Path(__file__).read_bytes()),
        "launch_signature": list(launch_signature(result.asc)),
        "compile_options": {
            key: manifest.get(key)
            for key in (
                "target",
                "cann",
                "commands",
                "compiler_version",
                "binary_sha256",
                "seconds",
            )
        },
        "oracle": "independent Python-int lane conditions; bit-exact f32",
    }


def run(
    result, library, output, physical, cases, repeats, *, artifact, shape, provenance, manifest
):
    import numpy as np

    rows, columns = shape
    require(repeats > 0, "positive repeats are required")
    validate_snapshot_shapes(artifact, shape)
    validate_plan_buffers(result.plan, shape)
    signature = launch_signature(result.asc)
    require(signature == EXPECTED_LAUNCH, f"unexpected launch ABI: {signature}")
    health = health_snapshot(physical, Path(output))
    runtime = ACL()
    records = []
    try:
        module = ct.CDLL(str(Path(library).resolve()))
        launch = getattr(module, f"launch_{result.name}")
        launch.argtypes = [ct.c_void_p] * 5 + [ct.c_int64] * 3
        launch.restype = ct.c_int
        guard = b"\x5a" * 128
        x = field_tensor(rows, columns)
        raw_x = x.tobytes() + guard
        nbytes = rows * columns * 4
        require(len(raw_x) == nbytes + len(guard), "input byte count mismatch")
        raw_x_digest = _digest(raw_x)

        def launch_case(devices, q, lo, hi):
            require(
                launch(runtime.stream, *devices, q, lo, hi) == 0,
                f"row-mask launch rejected its runtime contract: case={(q, lo, hi)}",
            )

        def read_outputs(devices):
            raw = [runtime.read(device, nbytes + len(guard)) for device in devices[1:]]
            actual = [np.frombuffer(item[:nbytes], dtype=np.float32).reshape(shape) for item in raw]
            return raw, actual

        for q, lo, hi in cases:
            expected = oracle(x, q, lo, hi)
            buffers = [b"\xcd" * nbytes + guard for _ in range(3)]
            devices = [runtime.allocate(raw_x), *(runtime.allocate(raw) for raw in buffers)]
            launch_case(devices, q, lo, hi)
            runtime.synchronize()
            raw, actual = read_outputs(devices)
            errors = [
                float(np.max(np.abs(values - want)))
                for values, want in zip(actual, expected, strict=True)
            ]
            require(
                all(
                    np.array_equal(values, want)
                    for values, want in zip(actual, expected, strict=True)
                ),
                f"device row-mask output differs from the independent oracle: "
                f"case={(q, lo, hi)}, shape={shape}, errors={errors}",
            )
            require(all(item[nbytes:] == guard for item in raw), "output guard changed")
            require(runtime.read(devices[0], len(raw_x)) == raw_x, "read-only input changed")
            for _ in range(repeats - 1):
                launch_case(devices, q, lo, hi)
            runtime.synchronize()
            repeated_raw, repeated = read_outputs(devices)
            require(
                all(
                    np.array_equal(values, want)
                    for values, want in zip(repeated, expected, strict=True)
                ),
                "repeated row-mask launches changed numerical output",
            )
            require(
                all(item[nbytes:] == guard for item in repeated_raw),
                "repeated launches changed an output bounds guard",
            )
            require(
                runtime.read(devices[0], len(raw_x)) == raw_x,
                "repeated launches changed the read-only input",
            )
            record = {
                "q": q,
                "lo": lo,
                "hi": hi,
                "shape": list(shape),
                "dtype": "f32",
                "mask_lanes": {
                    "y0_cols_lt_hi": int(np.count_nonzero(expected[0] != x)),
                    "y1_cols_ne_lo": int(np.count_nonzero(expected[1] != x)),
                    "y2_rows_ge_cols_minus_q": int(np.count_nonzero(expected[2] != x)),
                },
                "per_output": [
                    {"name": name, "max_abs_error": error, "bit_exact": True}
                    for name, error in zip(("Y0", "Y1", "Y2"), errors, strict=True)
                ],
                "bit_exact": True,
                "input_and_output_guards": "checked",
                "input_sha256": raw_x_digest,
                "repeats": repeats,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        runtime.close()
    report = {
        "health": health,
        "kernel": result.name,
        "shape": list(shape),
        "library": str(Path(library).resolve()),
        "source_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "provenance": provenance,
        "verification": result.report,
        "compilation": result.timings,
        "oracle": "independent Python-int lane conditions; bit-exact f32",
        "timing": "not measured; numerical evidence only",
        "cases": records,
    }
    (Path(output) / "device.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="evidence/report directory")
    parser.add_argument("--build", action="store_true", help="offline shared CANN build")
    parser.add_argument("--artifact", type=Path, help="published build directory")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    require(args.repeats > 0, "positive repeats are required")
    require(0 < args.timeout <= 300, "the operator build deadline is at most 300 s")
    cases = parse_cases(args.cases)
    if args.rows is not None:
        require(args.rows > 0, "--rows must be positive")
    if args.columns is not None:
        require(args.columns > 0, "--columns must be positive")
    require(
        not (args.build and args.library is not None), "--build cannot reuse a published --library"
    )
    output = args.output.resolve()
    if args.build:
        artifact = (args.artifact or output / "artifact").resolve()
        require(artifact != output, "build artifacts and evidence must use separate directories")
        build_file(
            SOURCE,
            artifact,
            entry=ENTRY,
            parameters={
                "ROWS": 4 if args.rows is None else args.rows,
                "COLUMNS": 256 if args.columns is None else args.columns,
            },
            compile_asc=True,
            shared=True,
            timeout_s=args.timeout,
        )
    else:
        if args.artifact is not None:
            artifact = args.artifact.resolve()
        elif args.library is not None:
            artifact = args.library.parent.resolve()
        else:
            artifact = output
    result = read_artifact(artifact)
    shape = resolve_shape(artifact, args.rows, args.columns)
    library = args.library if args.library is not None else artifact / f"{result.name}.so"
    require(
        library.is_file(),
        f"published shared library missing: {library}; pass --build, --library, or a built artifact",
    )
    manifest = published_manifest(artifact, result, library)
    provenance = provenance_record(artifact, result, library, manifest)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "kernel": result.name,
        "shape": list(shape),
        "artifact_directory": str(artifact),
        "evidence_directory": str(output),
        "library": str(library.resolve()),
        "provenance": provenance,
        "verification": result.report,
        "compilation": result.timings,
        "device_report": str(output / "device.json") if args.physical is not None else None,
    }
    (output / "runner.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.physical is None:
        print(json.dumps(summary, indent=2))
        return 0
    run(
        result,
        library,
        output,
        args.physical,
        cases,
        args.repeats,
        artifact=artifact,
        shape=shape,
        provenance=provenance,
        manifest=manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

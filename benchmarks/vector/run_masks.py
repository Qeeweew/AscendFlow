"""Build or reuse the generic mask example and check it against a logical oracle.

Build and publish an offline shared library (one hard 300 s build deadline)::

    python -m benchmarks.vector.run_masks --build --output build/masks

Re-read a published artifact without invoking the compiler, then run it on a
healthy card explicitly mapped to logical device 0::

    ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 \
    python -m benchmarks.vector.run_masks --physical 2 \
        --artifact build/masks --library build/masks/masks.so \
        --output reports/masks-device

``--artifact`` defaults to the library's directory (or to ``--output`` without
``--library``). An artifact directory holds the published ``verification.json``,
ASC, source snapshot and ``.compile.json`` manifest. Reuse checks those bytes and
the manifest; only ``--build`` invokes the compiler or CANN. ``--rows`` and
``--columns`` must match the dimensions recorded in the published source
snapshot; omit them to use those verified dimensions.

The oracle is independent NumPy/Python lane logic, not the compiler predicate
interpreter. Coordinate conditions are evaluated with Python integers so that
nonnegative runtime origins up to ``INT64_MAX`` stay exact.
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
from benchmarks.work import structure_inventory, work_record

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/vector/masks.py"
ENTRY = "masks"
INT64_MAX = (1 << 63) - 1
BLOCK = 64
DEFAULT_CASES = (
    "0:0:0",  # empty interval, origin q=0
    "7:0:0",  # same empty interval, q moves the origin boundary
    "0:32:96",  # partial interval, q=0
    "7:32:96",  # same partial interval, q=7
    "0:0:1000000000",  # full interval on the covered shapes
    f"{INT64_MAX}:0:0",  # exact maximum signed runtime origin
    "12345:64:64",  # empty interval with large nonnegative origins
    "3:0:1000000000",  # full interval with a nonzero origin
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def source_snapshot_shape(directory):
    """Read the only artifact file that retains logical (not flat) dimensions."""
    path = Path(directory) / "source.xdsl"
    require(path.is_file(), f"missing published source snapshot: {path}")
    lines = [
        line
        for line in path.read_text().splitlines()
        if '"npu_source.alloc"' in line and 'sym_name = "X"' in line
    ]
    require(len(lines) == 1, "published source snapshot must declare exactly one X allocation")
    line = lines[0]
    tensor = re.search(r"!npu_source\.tensor<\[([^\]]*)\]", line)
    require(tensor is not None, "published X allocation has no tensor type")
    dimensions = [int(value) for value in re.findall(r"#builtin\.int<(-?\d+)>", tensor.group(1))]
    require(
        len(dimensions) == 2 and all(value > 0 for value in dimensions),
        f"published X dimensions must be two positive extents: {dimensions}",
    )
    require('"f32"' in line and '"GM"' in line, "published X must be a f32 GM tensor")
    require(
        dimensions[1] % BLOCK == 0,
        "published X columns must be whole packed 64-lane blocks",
    )
    return tuple(dimensions)


def resolve_shape(directory, rows, columns):
    """Use verified artifact dimensions and reject mismatched CLI shapes."""
    shape = source_snapshot_shape(directory)
    if rows is not None:
        require(rows == shape[0], f"--rows {rows} does not match published rows {shape[0]}")
    if columns is not None:
        require(
            columns == shape[1],
            f"--columns {columns} does not match published columns {shape[1]}",
        )
    return shape


def coordinates(rows, columns, q, lo, hi):
    """Coordinate predicates with Python ints; no NumPy int64 wrapping."""
    require(rows > 0 and columns > 0, "positive shape required")
    interval = [[lo <= c < hi for c in range(columns)] for _ in range(rows)]
    periodic = [[((c // BLOCK) % 2) == 0 for c in range(columns)] for _ in range(rows)]
    origin = [[(c + q) > r for c in range(columns)] for r in range(rows)]
    different = [
        [a != b for a, b in zip(row_i, row_p, strict=True)]
        for row_i, row_p in zip(interval, periodic, strict=True)
    ]
    fill = [
        [a and b for a, b in zip(row_d, row_o, strict=True)]
        for row_d, row_o in zip(different, origin, strict=True)
    ]
    return {
        "interval": interval,
        "periodic": periodic,
        "origin": origin,
        "different": different,
        "fill": fill,
    }


def field_tensors(rows, columns):
    """Deterministic f32 inputs whose 64-lane compare chunks genuinely differ."""
    import numpy as np

    base = np.arange(rows * columns, dtype=np.int64)
    chunk = base // BLOCK
    values = ((base % 97) - 48).astype(np.float32) / np.float32(8)
    values = values + (chunk % 2).astype(np.float32) * np.float32(2)
    # z spans the post-scale data range so that p&~q and ~p&q both occur;
    # otherwise the materialized AND would coincide with p on every true-p lane.
    other = (((base * 7) % 89) - 44).astype(np.float32) / np.float32(2)
    return values.reshape(rows, columns), other.reshape(rows, columns)


def oracle(x, z, q, lo, hi):
    """Derive all three outputs from Python/NumPy logical conditions only."""
    import numpy as np

    rows, columns = x.shape
    mask = coordinates(rows, columns, q, lo, hi)
    interval = np.array(mask["interval"], dtype=bool)
    fill = np.array(mask["fill"], dtype=bool)
    update = np.where(interval, x * np.float32(2), x) + np.float32(1)
    y0 = np.where(fill, np.float32(-3), update)
    p = x > np.float32(0.25)
    data = x * np.float32(5)
    q_pred = data > z
    not_p = ~p
    both = p & q_pred
    either = both | not_p
    y1 = np.where(either, data, z)
    y2 = np.where(both, np.float32(-9), y1)
    return y0.astype(np.float32), y1.astype(np.float32), y2.astype(np.float32)


def _parse_cases(values):
    cases = []
    for value in values:
        parts = value.split(":")
        require(len(parts) == 3, f"case {value!r} must be q:lo:hi")
        case = tuple(int(part) for part in parts)
        # The source exposes nonnegative signed int64 runtime sizes.
        require(
            all(0 <= item <= INT64_MAX for item in case),
            f"case {value!r} must use nonnegative signed int64 runtime sizes",
        )
        cases.append(case)
    return cases


def _chunk_bitmaps(predicate):
    columns = predicate.shape[1]
    require(columns % BLOCK == 0, "chunked compare requires whole 64-lane blocks")
    return [predicate[:, start : start + BLOCK].tobytes() for start in range(0, columns, BLOCK)]


def compare_discrimination(x, z):
    """Check that adjacent 64-lane data compare bitmaps differ."""
    import numpy as np

    p = x > np.float32(0.25)
    data = x * np.float32(5)
    q_pred = data > z
    p_chunks = _chunk_bitmaps(p)
    q_chunks = _chunk_bitmaps(q_pred)
    adjacent = [
        p_chunks[index] != p_chunks[index + 1] and q_chunks[index] != q_chunks[index + 1]
        for index in range(len(p_chunks) - 1)
    ]
    return {
        "chunks": len(p_chunks),
        "p_true_lanes": int(np.count_nonzero(p)),
        "p_false_lanes": int(np.count_nonzero(~p)),
        "q_true_lanes": int(np.count_nonzero(q_pred)),
        "q_false_lanes": int(np.count_nonzero(~q_pred)),
        "p_true_q_false_lanes": int(np.count_nonzero(p & ~q_pred)),
        "p_false_q_true_lanes": int(np.count_nonzero(~p & q_pred)),
        "adjacent_chunks_distinct": bool(adjacent) and all(adjacent),
    }


def lane_coverage(mask, fill):
    """Preserved updated, preserved untouched and filled lane counts."""
    import numpy as np

    interval = np.array(mask["interval"], dtype=bool)
    filled = np.array(fill, dtype=bool)
    coverage = "empty" if not interval.any() else "full" if interval.all() else "partial"
    return {
        "interval": coverage,
        "filled_lanes": int(np.count_nonzero(filled)),
        "interval_true_preserved_lanes": int(np.count_nonzero(interval & ~filled)),
        "interval_false_preserved_lanes": int(np.count_nonzero(~interval & ~filled)),
    }


def published_manifest(directory, result, library):
    """Check ASC/library/source-snapshot identity without invoking a compiler."""
    directory, library = Path(directory), Path(library)
    require(directory.is_dir(), f"missing published artifact directory: {directory}")
    require(library.is_file(), f"missing published shared library: {library}")
    manifest_path = library.with_suffix(".compile.json")
    require(manifest_path.is_file(), f"missing published build manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(
        isinstance(manifest.get("commands"), list) and manifest["commands"],
        "published manifest has no CANN command provenance",
    )
    require(
        isinstance(manifest.get("source_sha256"), str)
        and isinstance(manifest.get("shared_sha256"), str),
        "published manifest lacks ASC/library identities",
    )
    asc = directory / f"{result.name}.asc"
    require(asc.is_file(), f"missing published ASC: {asc}")
    require(
        _digest(result.asc.encode()) == manifest["source_sha256"],
        "artifact ASC does not match the published manifest",
    )
    require(
        _digest(asc.read_bytes()) == manifest["source_sha256"],
        "published ASC file does not match the published manifest",
    )
    require(
        _digest(library.read_bytes()) == manifest["shared_sha256"],
        "shared library does not match the published manifest",
    )
    command = manifest["commands"][0]
    require("-o" in command, "published manifest's first command has no output argument")
    source_index = command.index("-o") - 1
    require(0 <= source_index < len(command), "published manifest names no ASC source")
    require(
        Path(command[source_index]).resolve() == asc.resolve(),
        "published library was built from a different ASC snapshot",
    )
    return manifest


def provenance_record(directory, result, library, manifest):
    snapshot = Path(directory) / "source.xdsl"
    return {
        "artifact_directory": str(Path(directory).resolve()),
        "library": str(Path(library).resolve()),
        "manifest": str(Path(library).with_suffix(".compile.json").resolve()),
        "kernel": result.name,
        "target": manifest.get("target"),
        "asc_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "source_snapshot_sha256": _digest(snapshot.read_bytes()),
        "example_path": str(SOURCE.relative_to(ROOT)),
        "example_sha256": _digest(SOURCE.read_bytes()),
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": _digest(Path(__file__).read_bytes()),
        "oracle": "independent NumPy/Python logical lane conditions; bit-exact f32",
    }


def validate_plan_buffers(module, rows, columns):
    """Cross-check the source-snapshot shape against the compiled flat buffers."""
    from ascendflow.dialects.physical import BufferOp

    buffers = {
        operation.sym_name.data: operation.nbytes.data
        for operation in module.walk()
        if isinstance(operation, BufferOp)
    }
    require(
        {"X", "Z", "Y0", "Y1", "Y2"} <= set(buffers),
        "published plan lacks the expected external buffers",
    )
    expected = rows * columns * 4
    require(
        all(buffers[name] == expected for name in ("X", "Z", "Y0", "Y1", "Y2")),
        "source-snapshot shape does not match the published flat buffer sizes",
    )


def run(
    result,
    library,
    output,
    physical,
    cases,
    repeats,
    warmup,
    *,
    rows,
    columns,
    provenance,
    manifest,
):
    import numpy as np

    require(repeats > 0 and warmup > 0, "positive repeats and warmup are required")
    require(rows > 0 and columns > 0 and columns % BLOCK == 0, "invalid published shape")
    validate_plan_buffers(result.plan, rows, columns)
    health = health_snapshot(physical, output)
    runtime = ACL()
    records = []
    inventory = structure_inventory(result.plan)
    try:
        module = ct.CDLL(str(Path(library).resolve()))
        launch = getattr(module, f"launch_{result.name}")
        launch.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int64,
            ct.c_int64,
            ct.c_int64,
        ]
        launch.restype = ct.c_int
        guard = b"\x5a" * 128
        x, z = field_tensors(rows, columns)
        require(x.shape == (rows, columns), "runner input does not match the published shape")
        raw_x = x.tobytes() + guard
        raw_z = z.tobytes() + guard
        require(len(raw_x) - len(guard) == rows * columns * 4, "input byte count mismatch")
        discrimination = compare_discrimination(x, z)
        require(discrimination["p_true_lanes"] > 0, "data predicate p has no true lane")
        require(discrimination["p_false_lanes"] > 0, "data predicate p has no false lane")
        require(discrimination["q_true_lanes"] > 0, "data predicate q has no true lane")
        require(discrimination["q_false_lanes"] > 0, "data predicate q has no false lane")
        require(
            discrimination["p_true_q_false_lanes"] > 0
            and discrimination["p_false_q_true_lanes"] > 0,
            "data does not discriminate the materialized AND from p or q alone",
        )
        if columns >= 2 * BLOCK:
            require(
                discrimination["adjacent_chunks_distinct"],
                "adjacent 64-lane data compare bitmaps do not differ",
            )
        for q, lo, hi in cases:
            expected = oracle(x, z, q, lo, hi)
            mask = coordinates(rows, columns, q, lo, hi)
            coverage = lane_coverage(mask, mask["fill"])
            require(coverage["filled_lanes"] > 0, "masked fill has no selected lane")
            interval = np.array(mask["interval"], dtype=bool)
            if interval.any():
                require(
                    coverage["interval_true_preserved_lanes"] > 0,
                    "no preserved lane from the updated interval set is observable",
                )
            if (~interval).any():
                require(
                    coverage["interval_false_preserved_lanes"] > 0,
                    "no preserved untouched lane outside the interval is observable",
                )
            buffers = [b"\xcd" * x.nbytes + guard for _ in range(3)]
            devices = [
                runtime.allocate(raw_x),
                runtime.allocate(raw_z),
                *(runtime.allocate(raw) for raw in buffers),
            ]
            require(
                launch(runtime.stream, *devices, q, lo, hi) == 0,
                "mask example launch rejected its runtime contract",
            )
            runtime.synchronize()
            outputs = [
                runtime.read(device, len(raw))
                for device, raw in zip(devices[2:], buffers, strict=True)
            ]
            actual = [
                np.frombuffer(raw[: x.nbytes], dtype=np.float32).reshape(x.shape) for raw in outputs
            ]
            errors = [
                float(np.max(np.abs(values - expected)))
                for values, expected in zip(actual, expected, strict=True)
            ]
            require(
                all(
                    np.array_equal(values, expected)
                    for values, expected in zip(actual, expected, strict=True)
                ),
                f"device mask output differs from the independent logical oracle: "
                f"case={q, lo, hi}, errors={errors}",
            )
            require(all(raw[x.nbytes :] == guard for raw in outputs), "output guard changed")
            require(
                runtime.read(devices[0], len(raw_x)) == raw_x
                and runtime.read(devices[1], len(raw_z)) == raw_z,
                "read-only input changed",
            )
            for _ in range(warmup - 1):
                require(
                    launch(runtime.stream, *devices, q, lo, hi) == 0,
                    "mask example launch rejected its runtime contract",
                )
            runtime.synchronize()
            start, stop = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(repeats):
                require(
                    launch(runtime.stream, *devices, q, lo, hi) == 0,
                    "mask example launch rejected its runtime contract",
                )
            runtime.record(stop)
            runtime.synchronize()
            outputs = [
                runtime.read(device, len(raw))
                for device, raw in zip(devices[2:], buffers, strict=True)
            ]
            repeated = [
                np.frombuffer(raw[: x.nbytes], dtype=np.float32).reshape(x.shape) for raw in outputs
            ]
            require(
                all(
                    np.array_equal(values, expected)
                    for values, expected in zip(repeated, expected, strict=True)
                )
                and all(raw[x.nbytes :] == guard for raw in outputs),
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
                "q": q,
                "lo": lo,
                "hi": hi,
                "shape": [rows, columns],
                "dtype": "f32",
                "mask_coverage": {
                    "interval": coverage["interval"],
                    "interval_true_preserved_lanes": coverage["interval_true_preserved_lanes"],
                    "interval_false_preserved_lanes": coverage["interval_false_preserved_lanes"],
                    "filled_lanes": coverage["filled_lanes"],
                    "periodic": (
                        "adjacent_64_lane_chunks_differ"
                        if columns >= 2 * BLOCK
                        else "single_64_lane_chunk"
                    ),
                    "coordinate_boolean": "interval_xor_periodic_and_origin",
                    "origin_condition": "cols_plus_q_gt_rows",
                    "data_predicates": "compare_where_not_or_and_masked_fill",
                    "earlier_compare_after_overwrite": "checked",
                    "count_op_between_masked_ops": "checked",
                    "data_compare": discrimination,
                },
                "max_abs_error": max(errors),
                "bit_exact": True,
                "input_and_output_guards": "checked",
                "warmup": warmup,
                "repeats": repeats,
                "stream_interval_us_per_launch": elapsed.value * 1000 / repeats,
                "timing_scope": "ACL event stream interval, including submission gaps; not kernel-only",
                "input_sha256": _digest(raw_x),
                "work": work_record(
                    inventory,
                    algorithm_counts={
                        "logical_lanes": x.size,
                        "masked_updates": 2,
                        "data_comparisons": 2,
                        "predicate_boolean_ops": 3,
                        "selects": 1,
                        "data_masked_fills": 1,
                    },
                    algorithm_scope="logical generic-mask problem; no native or traffic inference",
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        runtime.close()
    report = {
        "health": health,
        "kernel": result.name,
        "shape": [rows, columns],
        "library": str(Path(library).resolve()),
        "source_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "provenance": provenance,
        "verification": result.report,
        "compilation": result.timings,
        "oracle": "independent NumPy/Python logical lane conditions; bit-exact f32",
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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    require(args.repeats > 0 and args.warmup > 0, "positive repeats and warmup are required")
    require(0 < args.timeout <= 300, "the operator build deadline is at most 300 s")
    cases = _parse_cases(args.cases)
    if args.rows is not None:
        require(args.rows > 0, "--rows must be positive")
    if args.columns is not None:
        require(
            args.columns > 0 and args.columns % BLOCK == 0, "--columns must be whole 64-lane blocks"
        )
    require(
        not (args.build and args.library is not None), "--build cannot reuse a published --library"
    )
    output = args.output.resolve()
    artifact = (
        args.artifact
        if args.artifact is not None
        else args.library.parent
        if args.library is not None
        else args.output
    ).resolve()
    if args.build:
        build_file(
            SOURCE,
            artifact,
            entry=ENTRY,
            parameters={
                "ROWS": 8 if args.rows is None else args.rows,
                "COLUMNS": 128 if args.columns is None else args.columns,
            },
            compile_asc=True,
            shared=True,
            timeout_s=args.timeout,
        )
    result = read_artifact(artifact)
    rows, columns = resolve_shape(artifact, args.rows, args.columns)
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
        "shape": [rows, columns],
        "artifact_directory": str(artifact),
        "library": str(library.resolve()),
        "provenance": provenance,
        "verification": result.report,
        "compilation": result.timings,
        "device_report": str(output / "device.json") if args.physical is not None else None,
    }
    (output / "runner.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.physical is None:
        print(
            json.dumps(
                {
                    "shape": [rows, columns],
                    "artifact_directory": str(artifact),
                    "provenance": provenance,
                    "verification": result.report,
                    "compilation": result.timings,
                },
                indent=2,
            )
        )
        return 0
    run(
        result,
        library,
        output,
        args.physical,
        cases,
        args.repeats,
        args.warmup,
        rows=rows,
        columns=columns,
        provenance=provenance,
        manifest=manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

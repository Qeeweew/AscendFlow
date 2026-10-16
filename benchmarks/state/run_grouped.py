"""Runtime parent/child state example and independent fixed AscendC comparison."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from benchmarks.artifacts import read_artifact
from benchmarks.references import ascendc_source
from benchmarks.validation.state import inputs_and_oracle
from benchmarks.work import structure_inventory, work_record
from examples.state.frame_snapshot import frame_state_sum
from examples.state.grouped import grouped_state_sum
from examples.state.shared_inputs import shared_state_sum
from xdsl_npu.build import build_module
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.diagnostics import require
from xdsl_npu.performance import hardware_snapshot


def reference_source(tile, groups, *, shared_inputs=False, frame_snapshot=False):
    source = (
        ascendc_source(
            "ascendc_frame_state.asc.in" if frame_snapshot else "ascendc_grouped_state.asc.in"
        )
        .replace("@B@", str(tile))
        .replace("@GROUPS@", str(groups))
    )
    if shared_inputs:
        # Change only the independently handwritten GM input bases. The output
        # still has a distinct frame per call; storage and issue policy agree.
        for pointer in ("initial", "input"):
            before = f"(__gm__ float*){pointer} + (int64_t(GetBlockIdx()) * outputs + call)"
            require(source.count(before) == 1, "reference input base changed")
            source = source.replace(before, f"(__gm__ float*){pointer} + int64_t(GetBlockIdx())")
        source = source.replace("grouped_state_reference", "shared_state_reference")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["0:0", "0:7", "1:0", "3:0", "1:1", "3:1", "2:2", "3:7", "16:128"],
    )
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shared-inputs", action="store_true")
    mode.add_argument("--frame-snapshot", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-rounding-case", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    cases = [tuple(map(int, case.split(":"))) for case in args.cases]
    require(
        1 <= args.groups <= 40
        and args.tile > 0
        and args.tile % 8 == 0
        and args.tile * 12 <= 192 * 1024
        and args.warmup >= 0
        and args.repeats > 0,
        "invalid grouped-state policy",
    )
    require(
        all(
            len(case) == 2
            and all(0 <= v <= 65536 for v in case)
            and args.groups * (1 if args.shared_inputs else case[0]) * case[1] * args.tile * 4
            <= 4 * 1024**3
            for case in cases
        ),
        "validation cases require nonnegative bounded counts and at most 4 GiB input each",
    )
    definition = (
        frame_state_sum
        if args.frame_snapshot
        else shared_state_sum
        if args.shared_inputs
        else grouped_state_sum
    )
    variants = ["generated", "ascendc"] if args.variant == "both" else [args.variant]
    build_generated = args.build and "generated" in variants
    generated = args.output / "generated"
    generated_check = generated if build_generated else generated / "source-check"
    build_module(
        lambda: definition.specialize(B=args.tile, GROUPS=args.groups),
        generated_check,
        shared=build_generated,
        timeout_s=args.timeout,
    )
    artifact = read_artifact(generated_check)
    inventory = structure_inventory(artifact.plan)
    sources = {"generated": (artifact.name, artifact.asc)}
    manifests = {}
    for variant in variants:
        directory = args.output / variant
        if variant == "ascendc":
            name = (
                "frame_state_reference"
                if args.frame_snapshot
                else "shared_state_reference"
                if args.shared_inputs
                else "grouped_state_reference"
            )
            check = directory if args.build else directory / "source-check"
            build_module(
                lambda: reference_source(
                    args.tile,
                    args.groups,
                    shared_inputs=args.shared_inputs,
                    frame_snapshot=args.frame_snapshot,
                ),
                check,
                name=name,
                shared=args.build,
                timeout_s=args.timeout,
            )
            reference = read_artifact(check)
            sources[variant] = reference.name, reference.asc
        name, source = sources[variant]
        if args.physical is not None:
            manifest = json.loads((directory / f"{name}.compile.json").read_text())
            require(
                hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"]
                and (directory / f"{name}.asc").read_text() == source
                and hashlib.sha256((directory / f"{name}.so").read_bytes()).hexdigest()
                == manifest["shared_sha256"],
                "source/library changed since build",
            )
            manifests[variant] = manifest
    if args.physical is None:
        print(artifact.timings)
        return
    output = args.record or args.output / f"device-{args.variant}.json"
    health = health_snapshot(args.physical, output.parent)
    hardware = hardware_snapshot(args.physical)
    import numpy as np

    runtime, records = ACL(), []
    try:
        for variant in variants:
            name, _ = sources[variant]
            library = ct.CDLL(str((args.output / variant / f"{name}.so").resolve()))
            launch = getattr(library, "launch_" + name)
            launch.argtypes, launch.restype = [ct.c_void_p] * 4 + [ct.c_int64] * 2, ct.c_int
            for calls, tiles, rounding in [(p, n, False) for p, n in cases] + (
                [] if args.skip_rounding_case else [(3, 2, True)]
            ):
                initial, x, expected = inputs_and_oracle(
                    args.groups * (1 if args.shared_inputs else calls),
                    tiles,
                    args.tile,
                    rounding=rounding,
                )
                expected = (
                    np.repeat(expected[:, None, :], calls, axis=1)
                    if args.shared_inputs
                    else expected.reshape(args.groups, calls, args.tile)
                )
                if args.frame_snapshot:
                    expected = np.add(
                        expected, np.frombuffer(initial, dtype=np.float32).reshape(expected.shape)
                    )
                guard = b"\x5a" * 512
                original = b"\xcd" * expected.nbytes + guard
                initial, x = initial + guard, x + guard
                ip, xp, op = (
                    runtime.allocate(initial),
                    runtime.allocate(x),
                    runtime.allocate(original),
                )
                for _ in range(args.warmup):
                    require(
                        launch(runtime.stream, ip, xp, op, calls, tiles) == 0, "launch rejected"
                    )
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    require(
                        launch(runtime.stream, ip, xp, op, calls, tiles) == 0, "launch rejected"
                    )
                runtime.record(stop)
                runtime.synchronize()
                raw = runtime.read(op, len(original))
                actual = np.frombuffer(raw[: expected.nbytes], dtype=np.float32).reshape(
                    expected.shape
                )
                require(
                    np.array_equal(actual, expected), "ordered per-call state recurrence failed"
                )
                require(
                    raw[expected.nbytes :] == guard
                    and runtime.read(ip, len(initial)) == initial
                    and runtime.read(xp, len(x)) == x,
                    "input/output guard changed",
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
                    "outputs": calls,
                    "tiles": tiles,
                    "rounding_case": rounding,
                    "input_bytes": len(initial) + len(x) - 2 * len(guard),
                    "output_elements": expected.size,
                    "max_abs_error": 0.0,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed.value * 1000 / args.repeats,
                    "timing_scope": "ACL stream interval including host gaps; not kernel-only",
                    "cache_policy": (
                        "same group/tiles-major read-only inputs shared across calls; buffers reused, no flush"
                        if args.shared_inputs
                        else "same group/call/tiles-major buffers reused, no flush"
                    ),
                    "input_sha256": [hashlib.sha256(v).hexdigest() for v in (initial, x)],
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                    "work": work_record(
                        inventory,
                        algorithm_counts={
                            "fp32_add_operations": args.groups
                            * calls
                            * (tiles + int(args.frame_snapshot))
                            * args.tile,
                        },
                        algorithm_scope="ordered FP32 recurrence per output plus the optional original-frame add; no native count",
                    ),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "work"}),
                    flush=True,
                )
            for p, n in [(-1, 1), (1, -1), (1 << 40, 1 << 40)]:
                require(launch(runtime.stream, ip, xp, op, p, n) == -1, "invalid extent accepted")
    finally:
        runtime.close()
    output.write_text(
        json.dumps(
            {
                "policy": {
                    "B": args.tile,
                    "GROUPS": args.groups,
                    **({"shared_inputs": True} if args.shared_inputs else {}),
                    **({"frame_snapshot": True} if args.frame_snapshot else {}),
                },
                "health": health,
                "hardware": hardware,
                "variants": manifests,
                "cases": records,
                "compilation": artifact.timings,
                "contract": (
                    "ordered FP32 recurrence per call, then add the original GM snapshot; zero parent writes nothing"
                    if args.frame_snapshot
                    else "ordered FP32 recurrence per call; zero child copies Initial; zero parent writes nothing"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

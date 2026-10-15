"""Runtime parent/child state example and independent fixed AscendC comparison."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ast_grouped_state_sum import grouped_state_sum
from ast_shared_state_sum import shared_state_sum
from run_state_sum import inputs_and_oracle

from xdsl_npu import compile
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require
from xdsl_npu.performance import hardware_snapshot, runtime_work
from xdsl_npu.toolchain import offline_compile


def reference_source(tile, groups, *, shared_inputs=False):
    source = (
        Path(__file__)
        .with_name("ascendc_grouped_state.asc.in")
        .read_text()
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
    parser.add_argument("--shared-inputs", action="store_true")
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
    definition = shared_state_sum if args.shared_inputs else grouped_state_sum
    artifact = compile(definition.specialize(B=args.tile, GROUPS=args.groups))
    sources = {
        "generated": (artifact.name, artifact.asc),
        "ascendc": (
            "shared_state_reference" if args.shared_inputs else "grouped_state_reference",
            reference_source(args.tile, args.groups, shared_inputs=args.shared_inputs),
        ),
    }
    variants = list(sources) if args.variant == "both" else [args.variant]
    if args.build or args.physical is None:
        artifact.write(args.output / "generated")
    manifests = {}
    for variant in variants:
        name, source = sources[variant]
        directory = args.output / variant
        if args.build:
            offline_compile(source, directory, name, shared=True)
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
                    "requested_work": runtime_work(
                        artifact.plan, {"outputs": calls, "tiles": tiles}
                    ),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "requested_work"}),
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
                },
                "health": health,
                "hardware": hardware,
                "variants": manifests,
                "cases": records,
                "compilation": artifact.timings,
                "contract": "ordered FP32 recurrence per call; zero child copies Initial; zero parent writes nothing",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

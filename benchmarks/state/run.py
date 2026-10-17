"""Build/run the retained state recurrence and its independent AscendC policy."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ascendflow.build import build_module
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.references import ascendc_source
from benchmarks.validation.state import inputs_and_oracle
from benchmarks.work import structure_inventory, work_record
from examples.state.recurrence import state_sum


def reference_source(tile, groups):
    return (
        ascendc_source("ascendc_state_sum.asc.in")
        .replace("@B@", str(tile))
        .replace("@GROUPS@", str(groups))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--counts", nargs="+", type=int, default=[0, 1, 2, 3, 7, 17, 2048])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--skip-rounding-case", action="store_true", help="profile only the requested fixed counts"
    )
    args = parser.parse_args()
    require(
        1 <= args.groups <= 40
        and args.tile > 0
        and args.tile % 8 == 0
        and 12 * args.tile <= 192 * 1024
        and args.warmup >= 0
        and args.repeats > 0
        and all(0 <= count <= 65536 for count in args.counts),
        "invalid state validation policy",
    )
    variants = ["generated", "ascendc"] if args.variant == "both" else [args.variant]
    build_generated = args.build and "generated" in variants
    generated = args.output / "generated"
    generated_check = generated if build_generated else generated / "source-check"
    build_module(
        lambda: state_sum.specialize(B=args.tile, GROUPS=args.groups),
        generated_check,
        shared=build_generated,
        timeout_s=args.timeout,
    )
    artifact = read_artifact(generated_check)
    inventory = structure_inventory(artifact.plan)
    sources = {"generated": (artifact.name, artifact.asc)}
    manifests = {}
    for variant in variants:
        directory = (args.output / variant).resolve()
        if variant == "ascendc":
            check = directory if args.build else directory / "source-check"
            build_module(
                lambda: reference_source(args.tile, args.groups),
                check,
                name="state_sum_reference",
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
            launch.argtypes, launch.restype = [ct.c_void_p] * 4 + [ct.c_int64], ct.c_int
            for count, rounding in [(n, False) for n in args.counts] + (
                [] if args.skip_rounding_case else [(2, True)]
            ):
                initial, x, expected = inputs_and_oracle(
                    args.groups, count, args.tile, rounding=rounding
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
                    require(launch(runtime.stream, ip, xp, op, count) == 0, "launch rejected")
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    require(launch(runtime.stream, ip, xp, op, count) == 0, "launch rejected")
                runtime.record(stop)
                runtime.synchronize()
                raw = runtime.read(op, len(original))
                actual = np.frombuffer(raw[: expected.nbytes], dtype=np.float32).reshape(
                    expected.shape
                )
                require(np.array_equal(actual, expected), "ordered FP32 state recurrence failed")
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
                    "tiles": count,
                    "rounding_case": rounding,
                    "input_bytes": len(initial) + len(x) - 2 * len(guard),
                    "output_elements": expected.size,
                    "max_abs_error": 0.0,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed.value * 1000 / args.repeats,
                    "timing_scope": "ACL stream interval including submission gaps; not kernel-only",
                    "cache_policy": "same allocations reused, no flush",
                    "input_sha256": [hashlib.sha256(v).hexdigest() for v in (initial, x)],
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                    "work": work_record(
                        inventory,
                        algorithm_counts={"fp32_add_operations": args.groups * count * args.tile},
                        algorithm_scope="ordered FP32 state recurrence; no native or operand-byte count",
                    ),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "work"}),
                    flush=True,
                )
            for bad in (-1, 1 << 62):
                require(launch(runtime.stream, ip, xp, op, bad) == -1, "invalid extent accepted")
    finally:
        runtime.close()
    output.write_text(
        json.dumps(
            {
                "policy": {"B": args.tile, "GROUPS": args.groups},
                "health": health,
                "hardware": hardware,
                "variants": manifests,
                "cases": records,
                "compilation": artifact.timings,
                "contract": "ordered FP32 additions; zero count copies Initial to Output",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

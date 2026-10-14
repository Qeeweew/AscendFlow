"""Build/run the retained state recurrence and its independent AscendC policy."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ast_state_sum import state_sum

from xdsl_npu import compile
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require
from xdsl_npu.performance import hardware_snapshot, runtime_work
from xdsl_npu.toolchain import offline_compile


def reference_source(tile, groups):
    return (
        Path(__file__)
        .with_name("ascendc_state_sum.asc.in")
        .read_text()
        .replace("@B@", str(tile))
        .replace("@GROUPS@", str(groups))
    )


def inputs_and_oracle(groups, count, tile, *, rounding=False):
    import numpy as np

    if rounding:
        require(count == 2, "rounding counterexample uses two ordered additions")
        initial = np.full((groups, tile), 2**24, dtype=np.float32)
        x = np.empty((groups, 2, tile), dtype=np.float32)
        x[:, 0, :], x[:, 1, :] = 1.0, -(2**24)
        expected = np.zeros_like(initial)  # (2**24 + 1) rounds first, then subtract 2**24.
    else:
        require(0 <= count <= 65536, "exact dyadic oracle length exceeds its bound")
        rng = np.random.default_rng(9102201 + count)
        initial = rng.integers(-31, 32, size=(groups, tile), dtype=np.int16).astype(np.float32) / 32
        x = (
            rng.integers(-31, 32, size=(groups, count, tile), dtype=np.int16).astype(np.float32)
            / 32
        )
        expected = (initial.astype(np.float64) + x.sum(axis=1, dtype=np.float64)).astype(np.float32)
    return initial.tobytes(), x.tobytes(), expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--counts", nargs="+", type=int, default=[0, 1, 2, 3, 7, 17, 2048])
    parser.add_argument("--build", action="store_true")
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
    artifact = compile(state_sum.specialize(B=args.tile, GROUPS=args.groups))
    sources = {
        "generated": (artifact.name, artifact.asc),
        "ascendc": ("state_sum_reference", reference_source(args.tile, args.groups)),
    }
    variants = list(sources) if args.variant == "both" else [args.variant]
    if args.build or args.physical is None:
        artifact.write(args.output / "generated")
    manifests = {}
    for variant in variants:
        name, source = sources[variant]
        directory = (args.output / variant).resolve()
        if args.build:
            offline_compile(source, directory, name, shared=True)
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
                    "requested_work": runtime_work(artifact.plan, {"tiles": count}),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "requested_work"}),
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

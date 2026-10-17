"""Runtime K-panel reduction: exact oracle and fixed AscendC comparison.

Zero panels explicitly preserve C. Nonzero panels accumulate in source order.
The default large case has 960 MiB of b16 inputs across 20 AICs; inputs are
already panel-major, so neither implementation performs an extra GM packing.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ascendflow import language as N
from ascendflow.build import build_module
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.references import ascendc_source
from benchmarks.validation.gemm import inputs_and_oracle
from benchmarks.work import structure_inventory, work_record
from examples.gemm.reduction import reduction_gemm


def reference_source(policy):
    source = ascendc_source("ascendc_reduction_gemm.asc.in")
    values = {key: policy[key] for key in ("M", "N", "BK", "GROUPS")}
    values.update(
        TYPE="half" if policy["DTYPE"] == N.f16 else "bfloat16_t",
        TRANSPOSE=str(policy["TRANSPOSE_B"]).lower(),
    )
    for key, value in values.items():
        source = source.replace("@" + key + "@", str(value))
    require("@" not in source, "unbound reference parameter")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    parser.add_argument("--nn", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--panels", type=int, nargs="+", default=[0, 1, 2, 4, 7, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    require(
        1 <= args.groups <= 20
        and all(0 <= p <= 65536 for p in args.panels)
        and args.warmup >= 0
        and args.repeats > 0,
        "invalid fixed validation policy",
    )
    policy = {
        "M": args.m,
        "N": args.n,
        "BK": args.bk,
        "GROUPS": args.groups,
        "DTYPE": N.DType(args.dtype),
        "TRANSPOSE_B": not args.nn,
    }
    variants = ["generated", "ascendc"] if args.variant == "both" else [args.variant]
    build_generated = args.build and "generated" in variants
    generated = args.output / "generated"
    generated_check = generated if build_generated else generated / "source-check"
    build_module(
        lambda: reduction_gemm.specialize(**policy),
        generated_check,
        shared=build_generated,
        timeout_s=args.timeout,
    )
    result = read_artifact(generated_check)
    inventory = structure_inventory(result.plan)
    sources = {"generated": (result.name, result.asc)}
    manifests = {}
    for variant in variants:
        directory = (args.output / variant).resolve()
        if variant == "ascendc":
            check = directory if args.build else directory / "source-check"
            build_module(
                lambda: reference_source(policy),
                check,
                name="reduction_reference",
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
        print(result.timings)
        return
    record_path = args.record or args.output / f"device-{args.variant}.json"
    health = health_snapshot(args.physical, record_path.parent)
    hardware = hardware_snapshot(args.physical)
    import numpy as np

    runtime, records = ACL(), []
    try:
        for variant in variants:
            name, _ = sources[variant]
            module = ct.CDLL(str((args.output / variant / f"{name}.so").resolve()))
            launch = getattr(module, "launch_" + name)
            launch.argtypes, launch.restype = [ct.c_void_p] * 4 + [ct.c_int64], ct.c_int
            for panels in args.panels:
                a, b, expected = inputs_and_oracle(policy, panels)
                guard = b"\x5a" * 512
                a, b = a + guard, b + guard
                original = b"\xcd" * expected.nbytes + guard
                ap, bp, cp = runtime.allocate(a), runtime.allocate(b), runtime.allocate(original)
                for _ in range(args.warmup):
                    require(launch(runtime.stream, ap, bp, cp, panels) == 0, "launch rejected")
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    require(launch(runtime.stream, ap, bp, cp, panels) == 0, "launch rejected")
                runtime.record(stop)
                runtime.synchronize()
                raw = runtime.read(cp, len(original))
                actual = np.frombuffer(raw[: expected.nbytes], dtype=np.float32).reshape(
                    expected.shape
                )
                require(
                    np.array_equal(actual, expected) if panels else raw == original,
                    "ordered reduction/empty-output contract failed",
                )
                require(
                    raw[expected.nbytes :] == guard
                    and runtime.read(ap, len(a)) == a
                    and runtime.read(bp, len(b)) == b,
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
                    "panels": panels,
                    "effective_gemm_shape": [args.groups, args.m, args.n, panels * args.bk],
                    "output_elements": expected.size,
                    "max_abs_error": 0.0 if panels else None,
                    "empty_output_preserved": panels == 0,
                    "inputs_unchanged": True,
                    "guards_unchanged": True,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed.value * 1000 / args.repeats,
                    "timing_scope": "ACL stream interval including submission gaps; not kernel-only",
                    "cache_policy": "same panel-major allocations reused, no flush",
                    "input_sha256": [hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()],
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                    "work": work_record(
                        inventory,
                        algorithm_counts={
                            "fma_flops": 2 * args.groups * panels * args.m * args.n * args.bk,
                            "logical_input_bytes": len(a) + len(b) - 2 * len(guard),
                            "logical_output_bytes": expected.nbytes,
                        },
                        algorithm_scope="panel-reduction GEMM problem, two FLOPs per scalar FMA; zero panels preserve output",
                    ),
                }
                records.append(record)
                print(
                    json.dumps({key: value for key, value in record.items() if key != "work"}),
                    flush=True,
                )
    finally:
        runtime.close()
    record_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "policy": policy,
                "health": health,
                "hardware": hardware,
                "variants": manifests,
                "cases": records,
                "compilation": result.timings,
                "numerical_contract": "panel order preserved; integer/64 products and bounded exact f32 intermediates; zero panels preserve output",
                "scope": "explicit panel-major inputs; one local K stream per AIC, no nested parent pipeline or implicit GM packing",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

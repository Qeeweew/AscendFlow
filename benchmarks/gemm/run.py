"""Fixed explicit-operand Cube policy, generated ASC versus public AscendC.

Default large case: 20 groups × 1024 products of [32,64]×[64,64], 400 MiB
of GM inputs/output. Inputs are nonsymmetric exact dyadic values; every output
element is checked against a separate NumPy oracle, without tolerance tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path

from ascendflow import language as N
from ascendflow.build import build_module
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.references import ascendc_source, reference_kernel
from benchmarks.work import structure_inventory, work_record
from examples.gemm.batched import batched_matmul


def reference_source(policy):
    source = ascendc_source("ascendc_cube.asc.in")
    values = {
        "M": policy["M"],
        "N": policy["N"],
        "K": policy["K"],
        "GROUPS": policy["GROUPS"],
        "L1S": policy["L1_SLOTS"],
        "L0S": policy["L0_SLOTS"],
        "CS": policy["C_SLOTS"],
        "TYPE": "half" if policy["DTYPE"] == N.f16 else "bfloat16_t",
        "TRANSPOSE": str(policy["TRANSPOSE_B"]).lower(),
        "ACCUMULATE": str(policy["ACCUMULATE"]).lower(),
    }
    for key, value in values.items():
        source = source.replace("@" + key + "@", str(value))
    require("@" not in source, "unbound AscendC reference parameter")
    return source


def inputs_and_oracle(policy, tiles):
    import numpy as np

    m, n, k = (policy[axis] for axis in ("M", "N", "K"))
    # Exactly 17 different products repeat. Compute these independently and
    # gather their outputs; host oracle work/memory need not be cubic in tiles.
    a = ((np.arange(17 * m * k, dtype=np.int32) * 3 + 1) % 17 - 8).astype(np.float32).reshape(
        17, m, k
    ) / 8
    br, bc = (n, k) if policy["TRANSPOSE_B"] else (k, n)
    b = ((np.arange(17 * br * bc, dtype=np.int32) * 7 + 5) % 17 - 8).astype(np.float32).reshape(
        17, br, bc
    ) / 8
    expected = a @ (b.transpose(0, 2, 1) if policy["TRANSPOSE_B"] else b)
    if policy["ACCUMULATE"]:
        expected = expected + expected
    selection = np.arange(policy["GROUPS"] * tiles) % 17

    def pack(values):
        if policy["DTYPE"] == N.f16:
            return values.astype(np.float16)[selection].tobytes()
        # All generated values are exactly representable in bf16, including
        # zero. This is input serialization, not a numeric approximation.
        return (values.view(np.uint32) >> 16).astype(np.uint16)[selection].tobytes()

    return pack(a), pack(b), expected[selection]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["generated", "ascendc", "both"], default="both")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    parser.add_argument("--nn", action="store_true")
    parser.add_argument("--accumulate", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 1, 2, 4, 1024])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(
        1 <= args.groups <= 20
        and all(0 <= t <= 65536 for t in args.tiles)
        and args.warmup >= 0
        and args.repeats > 0,
        "invalid fixed test policy",
    )
    policy = {
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "GROUPS": args.groups,
        "DTYPE": N.DType(args.dtype),
        "TRANSPOSE_B": not args.nn,
        "L1_SLOTS": 3,
        "L0_SLOTS": 2,
        "C_SLOTS": 2,
        "ACCUMULATE": args.accumulate,
    }
    variants = ["generated", "ascendc"] if args.variant == "both" else [args.variant]
    build_generated = args.build and "generated" in variants
    generated = args.output / "generated"
    generated_check = generated if build_generated else generated / "source-check"
    build_module(
        lambda: batched_matmul.specialize(**policy),
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
                name="cube_reference",
                shared=args.build,
                timeout_s=args.timeout,
            )
            reference = read_artifact(check)
            sources[variant] = reference.name, reference.asc
        name, source = sources[variant]
        if args.physical is not None:
            manifest = json.loads((directory / f"{name}.compile.json").read_text())
            require(
                hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"],
                "source changed after CANN build",
            )
            require(
                hashlib.sha256((directory / f"{name}.so").read_bytes()).hexdigest()
                == manifest["shared_sha256"],
                "library changed after CANN build",
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
            library = args.output / variant / f"{name}.so"
            for tiles in args.tiles:
                launch = (
                    runtime.kernel(library, name, parameters=(tiles,))
                    if variant == "generated"
                    else reference_kernel(runtime, library, name, 3, parameters=(tiles,))
                )
                a, b, expected = inputs_and_oracle(policy, tiles)
                guard = b"\x5a" * 512
                a, b = a + guard, b + guard
                ap, bp, cp = (
                    runtime.allocate(a),
                    runtime.allocate(b),
                    runtime.allocate(b"\xcd" * expected.nbytes + guard),
                )
                for _ in range(args.warmup):
                    launch(ap, bp, cp)
                runtime.synchronize()
                start, stop = runtime.event(), runtime.event()
                runtime.record(start)
                for _ in range(args.repeats):
                    launch(ap, bp, cp)
                runtime.record(stop)
                runtime.synchronize()
                raw = runtime.read(cp, expected.nbytes + len(guard))
                actual = np.frombuffer(raw[: expected.nbytes], dtype=np.float32).reshape(
                    expected.shape
                )
                require(
                    np.array_equal(actual, expected), "Cube result differs from exact dyadic oracle"
                )
                require(
                    raw[expected.nbytes :] == guard
                    and runtime.read(ap, len(a)) == a
                    and runtime.read(bp, len(b)) == b,
                    "input/output guard changed",
                )
                elapsed_us = runtime.elapsed_us(start, stop)
                record = {
                    "variant": variant,
                    "kernel": name,
                    "tiles_per_group": tiles,
                    "input_a_shape": [args.groups, tiles, args.m, args.k],
                    "input_b_shape": [args.groups, tiles, args.n, args.k]
                    if not args.nn
                    else [args.groups, tiles, args.k, args.n],
                    "output_elements": expected.size,
                    "max_abs_error": 0.0,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed_us / args.repeats,
                    "timing_scope": "ACL stream interval including submission gaps; not kernel-only",
                    "cache_policy": "same allocations reused; no cache flush",
                    "input_sha256": [hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()],
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                    "work": work_record(
                        inventory,
                        algorithm_counts={
                            "fma_flops": 2
                            * args.groups
                            * tiles
                            * args.m
                            * args.n
                            * args.k
                            * (2 if args.accumulate else 1),
                            "logical_input_bytes": len(a) + len(b) - 2 * len(guard),
                            "logical_output_bytes": expected.nbytes,
                        },
                        algorithm_scope="GEMM problem with two FLOPs per scalar FMA; no native instruction count",
                    ),
                }
                records.append(record)
                print(
                    json.dumps({key: value for key, value in record.items() if key != "work"}),
                    flush=True,
                )
    finally:
        runtime.close()
    report = {
        "policy": policy,
        "health": health,
        "hardware": hardware,
        "variants": manifests,
        "cases": records,
        "numerical_contract": "exact dyadic b16 inputs; f32 native MAD, optional second ordered MAD; f32 Fixpipe output",
        "comparison_scope": "fixed native batched GEMM policy; no tile search or full-GEMM/FA performance claim",
        "compilation": result.timings,
    }
    record_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

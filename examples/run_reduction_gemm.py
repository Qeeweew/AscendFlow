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

from ast_reduction_gemm import reduction_gemm

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require
from xdsl_npu.performance import hardware_snapshot, runtime_work
from xdsl_npu.toolchain import offline_compile


def reference_source(policy):
    source = Path(__file__).with_name("ascendc_reduction_gemm.asc.in").read_text()
    values = {key: policy[key] for key in ("M", "N", "BK", "GROUPS")}
    values.update(
        TYPE="half" if policy["DTYPE"] == N.f16 else "bfloat16_t",
        TRANSPOSE=str(policy["TRANSPOSE_B"]).lower(),
    )
    for key, value in values.items():
        source = source.replace("@" + key + "@", str(value))
    require("@" not in source, "unbound reference parameter")
    return source


def inputs_and_oracle(policy, panels):
    import numpy as np

    m, n, bk, groups = (policy[key] for key in ("M", "N", "BK", "GROUPS"))
    require(panels * bk * 4 <= 1 << 24, "exact-oracle input bound exceeded")
    # Each product is an integer/64. The conservative bound above makes every
    # possible intermediate sum exactly representable in f32, regardless of
    # MAD's internal grouping. Non-square, nonsymmetric inputs test the layout.
    a = ((np.arange(5 * m * bk, dtype=np.int64) * 3 + 1) % 5 - 2).reshape(5, m, bk)
    br, bc = (n, bk) if policy["TRANSPOSE_B"] else (bk, n)
    b = ((np.arange(5 * br * bc, dtype=np.int64) * 2 + 3) % 5 - 2).reshape(5, br, bc)
    products = a @ (b.transpose(0, 2, 1) if policy["TRANSPOSE_B"] else b)
    expected = np.zeros((groups, m, n), dtype=np.int64)
    for group in range(groups):
        start = group * panels
        for residue in range(5):
            occurrences = (start + panels - 1 - residue) // 5 - (start - 1 - residue) // 5
            expected[group] += occurrences * products[residue]
    selection = np.arange(groups * panels) % 5

    def pack(values):
        values = values.astype(np.float32) / 8
        encoded = (
            values.astype(np.float16)
            if policy["DTYPE"] == N.f16
            else (values.view(np.uint32) >> 16).astype(np.uint16)
        )
        return encoded[selection].tobytes()

    return pack(a), pack(b), expected.astype(np.float32) / 64


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
    result = compile(reduction_gemm.specialize(**policy))
    sources = {
        "generated": (result.name, result.asc),
        "ascendc": ("reduction_reference", reference_source(policy)),
    }
    variants = list(sources) if args.variant == "both" else [args.variant]
    if args.build or args.physical is None:
        result.write(args.output / "generated")
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
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "stream_interval_us": elapsed.value * 1000 / args.repeats,
                    "timing_scope": "ACL stream interval including submission gaps; not kernel-only",
                    "cache_policy": "same panel-major allocations reused, no flush",
                    "input_sha256": [hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()],
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                    "requested_work": runtime_work(result.plan, {"panels": panels}),
                }
                records.append(record)
                print(
                    json.dumps(
                        {key: value for key, value in record.items() if key != "requested_work"}
                    ),
                    flush=True,
                )
            require(launch(runtime.stream, ap, bp, cp, -1) == -1, "negative count accepted")
    finally:
        runtime.close()
    record_path.write_text(
        json.dumps(
            {
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

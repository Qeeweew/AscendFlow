"""Independent fixed-policy AscendC FA; no compiler/legacy operator construction."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
from time import perf_counter

from attention_oracle import decode, encode, online_attention

from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require
from xdsl_npu.performance import hardware_snapshot
from xdsl_npu.toolchain import offline_compile

NAME = "attention_reference"


def reference_source(*, batch=1, heads=8, kv_heads=8, groups=20, dtype="f16"):
    require(
        all(type(v) is int and v > 0 for v in (batch, heads, kv_heads, groups))
        and heads % kv_heads == 0
        and groups <= 20
        and batch * heads < 2**20
        and dtype in {"f16", "bf16"},
        "reference requires positive B/H/G, HQ divisible by HKV, groups<=20 and b16 dtype",
    )
    source = Path(__file__).with_name("ascendc_attention.asc.in").read_text()
    for key, value in {
        "B": batch,
        "HQ": heads,
        "HKV": kv_heads,
        "GROUPS": groups,
        "TYPE": "half" if dtype == "f16" else "bfloat16_t",
    }.items():
        source = source.replace("@" + key + "@", str(value))
    require("@" not in source, "unbound attention reference policy")
    return source


def inputs_and_oracle(policy, sq, sk, *, scope="full", data_case="standard"):
    import numpy as np

    require(sq >= 0 and sq % 128 == 0 and sk > 0 and sk % 1024 == 0, "complete FA tiles required")
    require(
        scope in {"full", "sampled"} and data_case in {"standard", "uniform", "sharp"},
        "unknown numerical validation policy",
    )
    batch, heads, kh = (policy[k] for k in ("batch", "heads", "kv_heads"))
    dtype = policy["dtype"]
    rng = np.random.default_rng(9102201)
    arrays, raw = {}, {}
    for name, shape in [
        ("q", (batch, heads, sq, 128)),
        ("k", (batch, kh, sk, 128)),
        ("v", (batch, kh, sk, 128)),
    ]:
        values = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.5)
        if data_case == "uniform" and name == "k":
            values.fill(0)
        if data_case == "sharp" and name in {"q", "k"}:
            values *= np.float32(3)
        raw[name] = encode(values, dtype)
        arrays[name] = decode(raw[name], dtype, shape)
    selected = (
        np.arange(sq)
        if scope == "full"
        else np.array(
            sorted({q * 128 + r for q in range(sq // 128) for r in (0, 7, 63, 64, 71, 127)}),
            dtype=np.int64,
        )
    )
    start = perf_counter()
    expected = np.empty((batch, heads, len(selected), 128), dtype=np.float32)
    for b in range(batch):
        for h in range(heads):
            for start_row in range(0, len(selected), 128):
                rows = selected[start_row : start_row + 128]
                expected[b, h, start_row : start_row + len(rows)] = online_attention(
                    arrays["q"][b, h, rows],
                    arrays["k"][b, h // (heads // kh)],
                    arrays["v"][b, h // (heads // kh)],
                    block_kv=1024,
                    dtype=dtype,
                    scale=np.float32(1 / np.sqrt(128)),
                )
    raw["o"] = b"\xff\x7f" * (batch * heads * sq * 128)
    raw["score"] = b"\xcd" * (policy["groups"] * 2 * 128 * 1024 * 4)
    raw["p"] = b"\xcd" * (policy["groups"] * 2 * 128 * 1024 * 2)
    raw["pv"] = b"\xcd" * (policy["groups"] * 2 * 128 * 128 * 4)
    return raw, expected, selected, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    parser.add_argument("--cases", nargs="+", default=["128:1024", "256:1024", "384:3072"])
    parser.add_argument("--scope", choices=["full", "sampled"], default="full")
    parser.add_argument("--data-case", choices=["standard", "uniform", "sharp"], default="standard")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    policy = {k: getattr(args, k) for k in ("batch", "heads", "kv_heads", "groups", "dtype")}
    source = reference_source(**policy)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build:
        offline_compile(source, args.output, NAME, shared=True)
    if args.physical is None:
        (args.output / (NAME + ".asc")).write_text(source)
        print(
            json.dumps(
                {"policy": policy, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            )
        )
        return
    require(args.warmup >= 1 and args.repeats > 0, "positive warmup/repeats required")
    manifest = json.loads((args.output / (NAME + ".compile.json")).read_text())
    library_path = args.output / (NAME + ".so")
    require(
        hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"]
        and (args.output / (NAME + ".asc")).read_text() == source
        and hashlib.sha256(library_path.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "reference source/library changed since build",
    )
    record_path = args.record or args.output / ("device-" + args.data_case + ".json")
    import numpy as np

    records, prepared = [], []
    for case in args.cases:
        sq, sk = map(int, case.split(":"))
        raw, expected, selected, oracle_s = inputs_and_oracle(
            policy, sq, sk, scope=args.scope, data_case=args.data_case
        )
        prepared.append((sq, sk, raw, expected, selected, oracle_s))
    # One all-card selection before this process initializes ACL. Reuse its
    # context across cases; a second selection would see our own registered PID.
    health = health_snapshot(args.physical, record_path.parent)
    hardware = hardware_snapshot(args.physical)
    runtime = ACL()
    try:
        for sq, sk, raw, expected, selected, oracle_s in prepared:
            guard = b"\x5a" * 512
            pointers = {k: runtime.allocate(v + guard) for k, v in raw.items()}
            library = ct.CDLL(str(library_path.resolve()))
            launch = getattr(library, "launch_" + NAME)
            launch.argtypes = [ct.c_void_p] * 8 + [ct.c_int64] * 2
            launch.restype = ct.c_int
            params = [pointers[k] for k in ("q", "k", "v", "o", "score", "p", "pv")]
            for _ in range(args.warmup):
                require(launch(runtime.stream, *params, sq, sk) == 0, "reference launch rejected")
            runtime.synchronize()
            start, stop = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(args.repeats):
                require(launch(runtime.stream, *params, sq, sk) == 0, "reference launch rejected")
            runtime.record(stop)
            runtime.synchronize()
            actual_raw = runtime.read(pointers["o"], len(raw["o"]) + len(guard))
            actual = decode(
                actual_raw[: len(raw["o"])], args.dtype, (args.batch, args.heads, sq, 128)
            )
            require(np.isfinite(actual).all(), "nonfinite/unwritten FA output")
            checked = actual[:, :, selected]
            rtol, atol = (0.002, 0.0002) if args.dtype == "f16" else (0.01, 0.001)
            error = float(np.max(np.abs(checked - expected))) if expected.size else 0.0
            if not np.allclose(checked, expected, rtol=rtol, atol=atol):
                bad = tuple(
                    np.argwhere(np.abs(checked - expected) > atol + rtol * np.abs(expected))[0]
                )
                (record_path.parent / "mismatch-output.bin").write_bytes(actual_raw)
                raise ValueError(
                    f"FA mismatch {bad}: actual={checked[bad]}, expected={expected[bad]}, max={error}"
                )
            for name, ptr in pointers.items():
                observed = runtime.read(ptr, len(raw[name]) + len(guard))
                require(observed[-len(guard) :] == guard, f"{name} guard changed")
                if name in {"q", "k", "v"}:
                    require(observed[: -len(guard)] == raw[name], f"{name} input changed")
            elapsed = ct.c_float()
            runtime.call(
                "aclrtEventElapsedTime",
                [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
                ct.byref(elapsed),
                start,
                stop,
            )
            row = {
                "sq": sq,
                "sk": sk,
                "kernel": NAME,
                "policy": policy,
                "health": health,
                "hardware": hardware,
                "max_abs_error": error,
                "rtol": rtol,
                "atol": atol,
                "finite_elements": int(actual.size),
                "oracle_scope": args.scope,
                "selected_rows": selected.tolist(),
                "oracle_s": oracle_s,
                "data_case": args.data_case,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "stream_interval_us": elapsed.value * 1000 / args.repeats,
                "timing_scope": "ACL stream interval includes host submission gaps; not kernel-only",
                "cache_policy": "same BNSD allocations reused; no flush",
                "input_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                "output_sha256": hashlib.sha256(actual_raw).hexdigest(),
            }
            records.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "sq",
                            "sk",
                            "max_abs_error",
                            "finite_elements",
                            "stream_interval_us",
                        )
                    }
                ),
                flush=True,
            )
            record_path.write_text(
                json.dumps(
                    {
                        "cases": records,
                        "compilation": manifest,
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "oracle_sha256": hashlib.sha256(
                        Path(__file__).with_name("attention_oracle.py").read_bytes()
                    ).hexdigest(),
                        "contract": "BK1024 online FP32; P and O RN to b16; reciprocal+mul; CK128 native accumulation; oracle BLAS/reduction differs",
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

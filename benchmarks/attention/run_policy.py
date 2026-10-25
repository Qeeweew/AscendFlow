"""Independent fixed-policy AscendC FA reference."""

import argparse
import hashlib
import json
from pathlib import Path

import benchmarks.validation.attention as attention_validation
from ascendflow.build import build_module
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.references import reference_kernel
from benchmarks.references.attention import NAME, reference_source
from benchmarks.validation.attention import decode, inputs_and_oracle


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
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    policy = {k: getattr(args, k) for k in ("batch", "heads", "kv_heads", "groups", "dtype")}
    check = args.output if args.build else args.output / "source-check"
    build_module(
        lambda: reference_source(**policy),
        check,
        name=NAME,
        shared=args.build,
        timeout_s=args.timeout,
    )
    source = read_artifact(check).asc
    if args.physical is None:
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
            launch = reference_kernel(runtime, library_path, NAME, 7, parameters=(sq, sk))
            params = [pointers[k] for k in ("q", "k", "v", "o", "score", "p", "pv")]
            for _ in range(args.warmup):
                launch(*params)
            runtime.synchronize()
            start, stop = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(args.repeats):
                launch(*params)
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
            elapsed_us = runtime.elapsed_us(start, stop)
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
                "stream_interval_us": elapsed_us / args.repeats,
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
                            Path(attention_validation.__file__).read_bytes()
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

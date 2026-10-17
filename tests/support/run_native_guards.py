"""Explicit first/last native correctness regression, not operator throughput."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3, 7, 31, 257])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    require(args.repeats > 0 and all(0 <= n <= 65536 for n in args.cases), "invalid cases")
    output = args.output.resolve()
    name = "native_guarded_recurrence"
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "guarded.py"
    if args.build:
        report = build_file(fixture, output, entry=name, shared=True, timeout_s=300)
        print(json.dumps({"build": "passed", "compilation": report["compilation"]}), flush=True)
    if args.physical is None:
        return
    manifest = json.loads((output / f"{name}.compile.json").read_text())
    library = output / f"{name}.so"
    require(
        hashlib.sha256((output / f"{name}.asc").read_bytes()).hexdigest()
        == manifest["source_sha256"]
        and hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "source/library changed since CANN compilation",
    )
    record = {
        "purpose": "native first/last and retained child correctness, not performance",
        "kernel": name,
        "health": health_snapshot(args.physical, output / "device"),
        "compile_manifest": manifest,
        "artifact_sha256": {
            file: hashlib.sha256((output / file).read_bytes()).hexdigest()
            for file in ("source.xdsl", "before.xdsl", "after.xdsl", f"{name}.asc")
        },
        "runs": [],
    }
    import numpy as np

    lib = ct.CDLL(str(library))
    launch = getattr(lib, "launch_" + name)
    launch.argtypes, launch.restype = [ct.c_void_p] * 3 + [ct.c_int64], ct.c_int
    guard = b"\x5a" * 512
    runtime = ACL()
    try:
        for n in args.cases:
            x = (((np.arange(n * 1024) * 7) % 31 - 15) / 8).astype(np.float32).reshape(n, 2, 8, 64)
            state = np.zeros((2, 8, 64), dtype=np.float32)
            for part in x:
                state = (state + part) * np.float32(0.5)
            expected = state.astype(np.float16).tobytes() if n else b"\xcd" * (1024 * 2)
            original = x.tobytes() + guard
            initial = b"\xcd" * (1024 * 2) + guard
            xp, op = runtime.allocate(original), runtime.allocate(initial)
            for _ in range(args.repeats):
                require(launch(runtime.stream, xp, op, n) == 0, "launch rejected")
                runtime.synchronize()
                require(
                    runtime.read(op, len(initial)) == expected + guard,
                    "native parent guard result or output guard mismatch",
                )
                require(runtime.read(xp, len(original)) == original, "input or input guard changed")
            result = {
                "n": n,
                "input_shape": list(x.shape),
                "output_shape": [2, 8, 64],
                "launches": args.repeats,
                "output_exact": True,
                "empty_output_preserved": n == 0,
                "guards_intact": True,
                "input_intact": True,
            }
            record["runs"].append(result)
            print(json.dumps(result), flush=True)
    finally:
        runtime.close()
    (output / "device/result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

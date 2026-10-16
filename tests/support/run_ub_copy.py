"""Bitwise device regression for the native dav-2201 UB-to-UB route."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from xdsl_npu.build import build_file
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.diagnostics import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3, 7, 4096])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    require(args.repeats > 0 and all(0 <= n <= 65536 for n in args.cases), "invalid cases")
    output, name = args.output.resolve(), "ub_copy"
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ub_copy.py"
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
        "purpose": "bitwise UB copy and buffer reuse correctness, not performance",
        "health": health_snapshot(args.physical, output / "device"),
        "compile_manifest": manifest,
        "runs": [],
    }
    import numpy as np

    lib = ct.CDLL(str(library))
    launch = getattr(lib, "launch_" + name)
    launch.argtypes, launch.restype = [ct.c_void_p] * 3 + [ct.c_int64], ct.c_int
    # Include negative zero, infinities, qNaN/sNaN payloads and a subnormal.
    # No floating-point arithmetic is used to construct or check these bits.
    special = np.array(
        [0, 0x80000000, 0x3F800000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 1],
        dtype=np.uint32,
    )
    guard = b"\x5a" * 512
    runtime = ACL()
    try:
        for n in args.cases:
            rows = (np.arange(n, dtype=np.uint64) * 0x9E3779B9).astype(np.uint32)
            raw = (special[None, :] ^ rows[:, None]).tobytes()
            original, initial = raw + guard, b"\xcd" * len(raw) + guard
            xp, op = runtime.allocate(original), runtime.allocate(initial)
            for _ in range(args.repeats):
                require(launch(runtime.stream, xp, op, n) == 0, "launch rejected")
                runtime.synchronize()
                require(runtime.read(op, len(initial)) == original, "output bits/guard changed")
                require(runtime.read(xp, len(original)) == original, "input bits/guard changed")
            result = {"n": n, "bytes": len(raw), "launches": args.repeats, "bitwise_exact": True}
            record["runs"].append(result)
            print(json.dumps(result), flush=True)
    finally:
        runtime.close()
    (output / "device/result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

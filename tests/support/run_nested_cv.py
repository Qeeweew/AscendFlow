"""Explicit device regression for the retained runtime-parent CV fixture.

This checks values, slot reuse, empty domains and guards. It is not an operator
performance benchmark. Run with --build first; --physical requires an explicit
physical-to-logical-zero environment mapping and checks every card's status.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from xdsl_npu.build import build_file
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import require


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
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "nested.py"
    if args.build:
        report = build_file(fixture, output, entry="nested_cv", shared=True, timeout_s=300)
        print(json.dumps({"build": "passed", "compilation": report["compilation"]}))
    if args.physical is None:
        return
    manifest = json.loads((output / "nested_cv.compile.json").read_text())
    library = output / "nested_cv.so"
    require(
        hashlib.sha256((output / "nested_cv.asc").read_bytes()).hexdigest()
        == manifest["source_sha256"]
        and hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "source/library changed since CANN compilation",
    )
    record = {
        "purpose": "nested CV correctness, not FA performance",
        "health": health_snapshot(args.physical, output / "device"),
        "compile_manifest": manifest,
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("source.xdsl", "before.xdsl", "after.xdsl", "nested_cv.asc")
        },
        "runs": [],
    }
    import numpy as np

    lib = ct.CDLL(str(library))
    launch = lib.launch_nested_cv
    launch.argtypes, launch.restype = [ct.c_void_p] * 4 + [ct.c_int64], ct.c_int
    guard = b"\x5a" * 512
    runtime = ACL()
    try:
        for n in args.cases:
            x = ((np.arange(2 * n * 2 * 16 * 16) % 7) - 3).astype(np.float16)
            x = x.reshape(2, n, 2, 16, 16)
            wide = x.astype(np.float32)
            gram = wide @ wide.swapaxes(-1, -2)
            raw_score = np.concatenate([gram[:, :, j] for j in range(2)], axis=-1)
            expected = raw_score * 2
            slot_bytes = bytearray(b"\xcd" * (2 * 2 * 16 * 32 * 4))
            slots = np.frombuffer(slot_bytes, dtype=np.float32).reshape(2, 2, 16, 32)
            for t in range(n):
                slots[:, t % 2] = raw_score[:, t]
            initial_score = b"\xcd" * len(slot_bytes) + guard
            original = x.tobytes() + guard
            xp = runtime.allocate(original)
            sp = runtime.allocate(initial_score)
            op = runtime.allocate(b"\xcd" * expected.nbytes + guard)
            for _ in range(args.repeats):
                require(launch(runtime.stream, xp, sp, op, n) == 0, "launch rejected")
                runtime.synchronize()
                actual = runtime.read(op, expected.nbytes + len(guard))
                require(actual == expected.tobytes() + guard, "CV output or output guard mismatch")
                require(
                    runtime.read(sp, len(initial_score)) == bytes(slot_bytes) + guard,
                    "parent double-buffer slot or workspace guard mismatch",
                )
                require(runtime.read(xp, len(original)) == original, "input or input guard changed")
            result = {
                "n": n,
                "input_shape": list(x.shape),
                "output_shape": list(expected.shape),
                "launches": args.repeats,
                "max_abs_error": 0.0,
                "output_exact": True,
                "final_slots_exact": True,
                "guards_intact": True,
                "input_intact": True,
            }
            record["runs"].append(result)
            print(json.dumps(result), flush=True)
    finally:
        runtime.close()
    (output / "device" / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

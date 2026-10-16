"""Build/run the AST prefix example with independent lengths and exact guards."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from examples.vector.prefix import scale_prefix
from xdsl_npu import compile
from xdsl_npu.backend.cann import offline_compile
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.diagnostics import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    result = compile(scale_prefix)
    result.write(args.output)
    library = args.library
    if library is None and (args.shared or args.physical is not None):
        offline_compile(result.asc, args.output.resolve(), result.name, shared=True)
        library = args.output / f"{result.name}.so"
    if args.physical is None:
        print(json.dumps({"verification": result.report, "compilation": result.timings}))
        return
    manifest = json.loads(library.with_suffix(".compile.json").read_text())
    require(
        hashlib.sha256(result.asc.encode()).hexdigest() == manifest["source_sha256"],
        "shared library does not match the current ASC",
    )
    require(
        hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "shared library hash differs from its compilation manifest",
    )
    health = health_snapshot(args.physical, args.output)
    import numpy as np

    runtime = ACL()
    records = []
    try:
        module = ct.CDLL(str(library.resolve()))
        launch = getattr(module, f"launch_{result.name}")
        launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int64]
        launch.restype = ct.c_int
        cases = [
            (0, 0),
            (0, 80),
            (80, 0),
            (8, 65),
            (65, 8),
            (56, 64),
            (64, 64),
            (72, 80),
            (80, 72),
            (136, 145),
            (264, 280),
            (4 * 64, 4 * 64 + 8),
            (4096 * 64, 4096 * 64 + 8),
            (65536 * 64, 65536 * 64 + 8),
            (4096 * 64 + 8, 4096 * 64 + 24),
            (65536 * 64 + 8, 65536 * 64 + 16),
            ((1 << 24) + 8, (1 << 24) + 16),
        ]
        guard = b"\x5a" * 128
        for n, m in cases:
            values = ((np.arange(n, dtype=np.int32) % 127 - 63) / 16).astype(np.float32)
            initial = np.full(m, -123.25, dtype=np.float32)
            expected = initial.copy()
            length = min(n, m)
            expected[:length] = values[:length] * np.float32(2)
            raw_x = values.tobytes() + guard
            raw_y = initial.tobytes() + guard
            x, y = runtime.allocate(raw_x), runtime.allocate(raw_y)
            for _ in range(5):
                require(launch(runtime.stream, x, y, n, m) == 0, "valid input contract rejected")
                runtime.synchronize()
            actual = runtime.read(y, len(raw_y))
            require(
                actual == expected.tobytes() + guard,
                "prefix output, untouched suffix or guard differs from independent oracle",
            )
            require(runtime.read(x, len(raw_x)) == raw_x, "read-only input or guard changed")
            record = {
                "n": n,
                "m": m,
                "processed_elements": length,
                "max_abs_error": 0,
                "repeats": 5,
                "suffix_and_guards": "unchanged",
                "input_sha256": hashlib.sha256(raw_x).hexdigest(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        # These calls must reject on the host without submitting device work.
        rejected = [
            launch(runtime.stream, x, y, -1, 64),
            launch(runtime.stream, x, y, 64, -1),
            launch(runtime.stream, x, y, 7, 64),
            launch(runtime.stream, x, y, (1 << 63) - 1, 64),
            launch(runtime.stream, x, x, 64, 64),
            launch(runtime.stream, ct.c_void_p(x.value + 4), y, 64, 64),
        ]
        require(rejected == [-1] * len(rejected), "invalid input contract submitted a kernel")
    finally:
        runtime.close()
    (args.output / "device.json").write_text(
        json.dumps(
            {
                "health": health,
                "source_sha256": manifest["source_sha256"],
                "shared_sha256": manifest["shared_sha256"],
                "cases": records,
                "same_binary_for_all_lengths": True,
                "invalid_contracts_rejected": len(rejected),
                "verification": result.report,
                "compilation": result.timings,
                "performance_measured": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

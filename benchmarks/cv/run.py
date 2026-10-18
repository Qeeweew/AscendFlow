"""Build and check the nested mixed example against a full, exact CPU oracle."""

import argparse
import ctypes as ct
import hashlib
import json
import re
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact, verified_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 7])
    args = parser.parse_args()
    require(all(0 <= n <= 64 for n in args.counts), "CV smoke counts must be in [0,64]")
    directory = args.output.resolve()
    if args.build:
        build_file(
            Path(__file__).resolve().parents[2] / "examples/cv/nested.py",
            directory,
            entry="nested_cv",
            shared=True,
            timeout_s=300,
        )
    artifact = read_artifact(directory)
    manifest = json.loads((directory / "nested_cv.compile.json").read_text())
    source = verified_source(manifest)
    require(source == directory / "nested_cv.asc", "CV build directory differs")
    if args.physical is None:
        return
    health = health_snapshot(args.physical, directory)
    hardware = hardware_snapshot(args.physical)
    import numpy as np

    # Bind the actual named launcher ABI, independently of declaration order.
    signature = re.search(r'extern "C" int launch_nested_cv\(([^)]+)\)', artifact.asc)
    require(signature is not None, "missing CV launcher")
    names = [parameter.strip().split()[-1].lstrip("*") for parameter in signature[1].split(",")]
    require(
        len(names) == 5 and set(names) == {"stream", "arg_X", "arg_Score", "arg_O", "npu_size_n"},
        "unknown CV launcher ABI",
    )
    module = ct.CDLL(str(source.with_suffix(".so")))
    launch = module.launch_nested_cv
    launch.argtypes = [ct.c_int64 if name == "npu_size_n" else ct.c_void_p for name in names]
    launch.restype = ct.c_int
    runtime, records = ACL(), []
    guard = b"\x5a" * 512
    try:
        for n in args.counts:
            values = np.random.default_rng(9102201 + n).integers(-2, 3, (2, n, 2, 16, 16))
            x = (values / 8).astype(np.float16)
            products = x.astype(np.float64) @ x.astype(np.float64).swapaxes(-1, -2)
            expected = (np.concatenate([products[:, :, j] for j in range(2)], axis=-1) * 2).astype(
                np.float32
            )
            raw = {
                "arg_X": x.tobytes(),
                "arg_Score": b"\xcd" * (2 * 2 * 16 * 32 * 4),
                "arg_O": b"\xcd" * expected.nbytes,
            }
            pointers = {name: runtime.allocate(data + guard) for name, data in raw.items()}
            bindings = {**pointers, "stream": runtime.stream, "npu_size_n": n}
            for _ in range(2):
                require(launch(*(bindings[name] for name in names)) == 0, "CV launch failed")
                runtime.synchronize()
                output = runtime.read(pointers["arg_O"], expected.nbytes + len(guard))
                require(output == expected.tobytes() + guard, "CV output or guard differs")
            for name, pointer in pointers.items():
                observed = runtime.read(pointer, len(raw[name]) + len(guard))
                require(observed[-len(guard) :] == guard, f"CV {name} guard changed")
                if name == "arg_X":
                    require(observed[: -len(guard)] == raw[name], "CV input changed")
            records.append(
                {"n": n, "checked_elements": expected.size, "repeats": 2, "max_abs_error": 0}
            )
    finally:
        runtime.close()
    report = {
        "status": "passed",
        "health": health,
        "hardware": hardware,
        "artifact": manifest,
        "input_contract": "disjoint FP16 X, FP32 Score/O; two groups and two private AIV owners",
        "oracle": "full FP64 X @ X.T, concatenate two panels, multiply by two; exact dyadic inputs",
        "inputs_unchanged": True,
        "guards_checked": True,
        "cases": records,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "performance_measured": False,
    }
    (directory / "device.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "cases": records}), flush=True)


if __name__ == "__main__":
    main()

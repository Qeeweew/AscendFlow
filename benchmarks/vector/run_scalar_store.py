"""Check the public scalar capture/store example against an exact FP32 oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from benchmarks.artifacts import read_artifact


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, required=True)
    args = parser.parse_args()
    artifact = read_artifact(args.artifact)
    require(artifact.name == "scalar_store", "runner requires the scalar_store example")
    library = (args.artifact / "scalar_store.so").resolve()
    manifest = json.loads(library.with_suffix(".compile.json").read_text())
    require(
        hashlib.sha256(artifact.asc.encode()).hexdigest() == manifest["source_sha256"]
        and digest(library) == manifest["shared_sha256"],
        "scalar store source/library differs from its published compilation",
    )
    health = health_snapshot(args.physical, args.output)
    values = (np.arange(64, dtype=np.float32) - 32) / np.float32(16)
    guard = b"\x5a" * 128
    source, destination = values.tobytes() + guard, b"\xcd" * values.nbytes + guard
    runtime = ACL()
    try:
        launch = runtime.kernel(library, artifact.name)
        x, y = runtime.allocate(source), runtime.allocate(destination)
        for _ in range(3):
            launch(x, y)
            runtime.synchronize()
            actual = runtime.read(y, len(destination))
            require(
                np.array_equal(np.frombuffer(actual[: values.nbytes], dtype=np.float32), values * 2)
                and actual[values.nbytes :] == guard,
                "scalar capture/store output or bounds guard differs from the exact oracle",
            )
            require(runtime.read(x, len(source)) == source, "scalar store changed its input")
    finally:
        runtime.close()
    (args.output / "device.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "kernel": artifact.name,
                "health": health,
                "source_sha256": manifest["source_sha256"],
                "shared_sha256": manifest["shared_sha256"],
                "input_sha256": hashlib.sha256(source).hexdigest(),
                "max_abs_error": 0.0,
                "guard_and_input": "unchanged",
                "launches": 3,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

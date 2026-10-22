"""Build the read-only L2 eviction workload used by sparse measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascendflow import language as T
from ascendflow.build import build_file

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@T.prim_func(disjoint_external=True)
def cache_sweep(
    X: T.Tensor((GROUPS, tiles, B), T.f32),
    Output: T.Tensor((GROUPS, B), T.f32, access="write"),
    tiles: T.int64,
    B: T.constexpr = 8192,
    GROUPS: T.constexpr = 40,
):
    with T.Kernel(GROUPS, kind="vector") as group:
        checksum = T.alloc_ub((B,), T.f32)
        inputs = T.alloc_ub((2, B), T.f32)
        T.fill(checksum, 0.0)
        for tile in T.Pipelined(tiles):
            with T.Stage(0, name="load"):
                T.copy(X[group, tile, :], inputs[tile % 2, :])
            with T.Stage(1, name="checksum"):
                T.add(checksum, inputs[tile % 2, :], out=checksum)
        T.copy(checksum, Output[group, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = {"B": 8192, "GROUPS": 40}
    build_file(
        __file__,
        args.output,
        entry="cache_sweep",
        parameters=parameters,
        shared=True,
        timeout_s=300,
    )
    (args.output / "eviction-policy.json").write_text(
        json.dumps({"parameters": parameters, "tiles": 819}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

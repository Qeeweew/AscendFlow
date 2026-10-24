"""AST frontend: explicit UB slots and one runtime tile-count parameter.

This example uses complete tiles; device validation lives in benchmarks/vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ascendflow import language as N
from ascendflow.build import build_file


@N.prim_func(disjoint_external=True)
def scale(
    X: N.Tensor((tiles, B), N.f32),  # noqa: F821 - resolved by the DSL signature pass
    Y: N.Tensor((tiles, B), N.f32, access="write"),  # noqa: F821
    tiles: N.i64,
    B: N.constexpr = 64,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((B,), N.f32, slots=2, name="inputs")
        y = N.alloc_ub((B,), N.f32, slots=2, name="outputs")
        for t in N.Pipelined(tiles, issue_order=("load", "compute", "store")):
            with N.Stage(0, name="load"):
                loaded = N.copy(X[t, :], x.at(t))
            with N.Vector(), N.Stage(1, name="compute"):
                scaled = N.mul(loaded, 2.0, out=y.at(t))
            with N.Stage(2, name="store"):
                N.copy(scaled, Y[t, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report = build_file(
        Path(__file__),
        args.output,
        entry="scale",
        parameters={"B": args.tile},
        compile_asc=args.compile,
        shared=args.shared,
        timeout_s=args.timeout,
    )
    print(report["compilation"])


if __name__ == "__main__":
    main()

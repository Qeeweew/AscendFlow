"""AST frontend: explicit UB slots and one runtime tile-count parameter.

This first migration slice uses complete tiles. Raw element lengths, tail
predicates and additional independent runtime arguments are subsequent work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.toolchain import offline_compile


@N.prim_func(disjoint_external=True)
def scale(
    X: N.Tensor((tiles, B), N.f32),  # noqa: F821 - resolved by the DSL signature pass
    Y: N.Tensor((tiles, B), N.f32, access="write"),  # noqa: F821
    tiles: N.int64,
    B: N.constexpr = 64,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((2, B), N.f32, name="inputs")
        y = N.alloc_ub((2, B), N.f32, name="outputs")
        for t in N.Pipelined(tiles, stages=3):
            slot = t % 2
            with N.Stage(0, name="load"):
                N.copy(X[t, :], x[slot, :])
            with N.Stage(1, name="compute"), N.Vector():
                N.mul(x[slot, :], 2.0, out=y[slot, :])
            with N.Stage(2, name="store"):
                N.copy(y[slot, :], Y[t, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    result = compile(scale.specialize(B=args.tile))
    result.write(args.output)
    if args.compile or args.shared:
        offline_compile(result.asc, args.output, result.name, shared=args.shared)
    print(result.timings)


if __name__ == "__main__":
    main()

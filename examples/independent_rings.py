"""Independent 5/7-slot address periods, with one runtime task stream.

The deliberately coprime ring sizes exercise compiler complexity and reuse
proofs. They are not a tuning recommendation. Defaults retain 64-element tiles
so the existing scale oracle and guarded device runner apply unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.toolchain import offline_compile


@N.prim_func(disjoint_external=True)
def independent_rings(
    X: N.Tensor((tiles, 64), N.f32),  # noqa: F821
    Y: N.Tensor((tiles, 64), N.f32, access="write"),  # noqa: F821
    tiles: N.int64,
    P: N.constexpr = 5,
    Q: N.constexpr = 7,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((P, 64), N.f32, name="inputs")
        y = N.alloc_ub((Q, 64), N.f32, name="outputs")
        for t in N.Pipelined(tiles, stages=3):
            with N.Stage(0, name="load"):
                N.copy(X[t, :], x[t % P, :])
            with N.Stage(1, name="compute"), N.Vector():
                N.mul(x[t % P, :], 2.0, out=y[t % Q, :])
            with N.Stage(2, name="store"):
                N.copy(y[t % Q, :], Y[t, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--physical", type=int)
    args = parser.parse_args()
    result = compile(independent_rings)
    result.write(args.output)
    library = args.library
    if args.compile or args.shared:
        offline_compile(result.asc, args.output.resolve(), result.name, shared=args.shared)
        if args.shared:
            library = args.output / f"{result.name}.so"
    if args.physical is not None:
        if library is None:
            parser.error("--physical requires --shared or --library")
        from runtime_vector import run

        run(
            result,
            library,
            args.output,
            args.physical,
            [0, 1, 2, 4, 5, 6, 7, 8, 14, 35, 65536, 262144],
            5,
        )
    print(result.timings)


if __name__ == "__main__":
    main()

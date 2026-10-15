"""Independent 5/7-slot address periods, with one runtime task stream.

The deliberately coprime ring sizes exercise compiler complexity and reuse
proofs. They are not a tuning recommendation. Defaults retain 64-element tiles
so the existing scale oracle and guarded device runner apply unchanged.
"""

from __future__ import annotations

from xdsl_npu import language as N


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

"""Independent 5/7-slot address periods, with one runtime task stream.

The deliberately coprime ring sizes exercise compiler complexity and reuse
proofs. They are not a tuning recommendation. Defaults retain 64-element tiles
so the existing scale oracle and guarded device runner apply unchanged.
"""

from __future__ import annotations

from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def independent_rings(
    X: N.Tensor((tiles, 64), N.f32),  # noqa: F821
    Y: N.Tensor((tiles, 64), N.f32, access="write"),  # noqa: F821
    tiles: N.i64,
    P: N.constexpr = 5,
    Q: N.constexpr = 7,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((64,), N.f32, slots=P, name="inputs")
        y = N.alloc_ub((64,), N.f32, slots=Q, name="outputs")
        for t in N.Pipelined(tiles, issue_order=("load", "compute", "store")):
            with N.Stage(0, name="load"):
                loaded = N.copy(X[t, :], x.at(t))
            with N.Vector(), N.Stage(1, name="compute"):
                scaled = N.mul(loaded, 2.0, out=y.at(t))
            with N.Stage(2, name="store"):
                N.copy(scaled, Y[t, :])

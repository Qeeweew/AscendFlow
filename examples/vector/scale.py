"""AST frontend: explicit UB slots and one runtime tile-count parameter.

This example uses complete tiles; device validation lives in benchmarks/vector.
"""

from __future__ import annotations

from ascendflow import language as N


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

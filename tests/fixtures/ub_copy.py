"""Public bitwise UB copy: one binary, runtime tile count, no arithmetic."""

from __future__ import annotations

# Runtime dimensions are captured by the public DSL frontend.
# ruff: noqa: F821
from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def ub_copy(X: N.Tensor((n, 8), N.f32), O: N.Tensor((n, 8), N.f32, access="write"), n: N.i64):
    with N.Kernel():
        source = N.alloc_ub((2, 8), N.f32)
        destination = N.alloc_ub((2, 8), N.f32)
        for tile in N.Pipelined(n):
            with N.Stage(0, name="load"):
                N.copy(X[tile, :], source[tile % 2, :])
            with N.Stage(1, name="copy"):
                N.copy(source[tile % 2, :], destination[tile % 2, :])
            with N.Stage(2, name="store"):
                N.copy(destination[tile % 2, :], O[tile, :])

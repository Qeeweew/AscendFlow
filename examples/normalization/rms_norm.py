"""Explicit FP32 RMS normalization primitive, without affine gamma/bias.

The source prescribes R64 partials, merge, mean, epsilon, sqrt, reciprocal and scale. No
Rsqrt substitution, reassociation or implicit state/buffer copies are allowed.
"""

from __future__ import annotations

from ascendflow import language as N


@N.inline
def rms_compute(x, slot, y, partial, statistics, D, EPS, constant):
    N.mul(x[slot, :], x[slot, :], out=y[slot, :])
    N.reduce_sum(
        y[slot, :],
        axis=0,
        scratch=partial,
        out=statistics[:1],
        order="r64_then_native_merge",
    )
    N.mul(statistics[:1], 1.0 / D, out=statistics[:1])
    N.add(statistics[:1], EPS, out=statistics[:1])
    N.sqrt(statistics[:1], out=statistics[:1])
    N.fill(constant[:1], 1.0)
    N.div(constant[:1], statistics[:1], out=statistics[:1])
    inverse = N.load_scalar(statistics[:1])
    N.mul(x[slot, :], inverse, out=y[slot, :])


@N.prim_func(disjoint_external=True)
def rms_normalize(
    X: N.Tensor((GROUPS, rows, D), N.f32),  # noqa: F821 - DSL signature scope
    Y: N.Tensor((GROUPS, rows, D), N.f32, access="write"),  # noqa: F821
    rows: N.i64,
    D: N.constexpr = 4096,
    GROUPS: N.constexpr = 32,
    EPS: N.constexpr = 1e-6,
):
    with N.Kernel(GROUPS, kind="vector") as group:
        x = N.alloc_ub((2, D), N.f32, name="inputs")
        y = N.alloc_ub((2, D), N.f32, name="work")
        partial = N.alloc_ub((D // 64,), N.f32, name="partial")
        statistics = N.alloc_ub((8,), N.f32, name="statistics")
        constant = N.alloc_ub((8,), N.f32, name="constant")
        for row in N.Pipelined(rows, issue_order=("store", "load", "compute")):
            slot = row % 2
            with N.Stage(0, name="load"):
                N.copy(X[group, row, :], x[slot, :])
            with N.Vector(), N.Stage(1, name="compute"):
                rms_compute(x, slot, y, partial, statistics, D, EPS, constant)
            with N.Stage(2, name="store"):
                N.copy(y[slot, :], Y[group, row, :])

"""Explicit FP32 RMS normalization primitive, without affine gamma/bias.

The source prescribes R64 partials, merge, mean, epsilon, sqrt, reciprocal and scale. No
Rsqrt substitution, reassociation or implicit state/buffer copies are allowed.
"""

from __future__ import annotations

from ascendflow import language as N


@N.inline
def rms_compute(x, y, partial, statistics, D, EPS, constant):
    squared = N.mul(x, x, out=y)
    total = N.reduce_sum(
        squared,
        axis=0,
        scratch=partial,
        out=statistics[:1],
        order="r64_then_native_merge",
    )
    mean = N.mul(total, 1.0 / D, out=statistics[:1])
    stabilized = N.add(mean, EPS, out=statistics[:1])
    root = N.sqrt(stabilized, out=statistics[:1])
    one = N.fill(constant[:1], 1.0)
    reciprocal = N.div(one, root, out=statistics[:1])
    inverse = N.load_scalar(reciprocal)
    return N.mul(x, inverse, out=y)


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
        x = N.alloc_ub((D,), N.f32, slots=2, name="inputs")
        y = N.alloc_ub((D,), N.f32, slots=2, name="work")
        partial = N.alloc_ub((D // 64,), N.f32, name="partial")
        statistics = N.alloc_ub((8,), N.f32, name="statistics")
        constant = N.alloc_ub((8,), N.f32, name="constant")
        for row in N.Pipelined(rows, issue_order=("store", "load", "compute")):
            with N.Stage(0, name="load"):
                loaded = N.copy(X[group, row, :], x.at(row))
            with N.Vector(), N.Stage(1, name="compute"):
                normalized = rms_compute(loaded, y.at(row), partial, statistics, D, EPS, constant)
            with N.Stage(2, name="store"):
                N.copy(normalized, Y[group, row, :])

"""Real DSL primitive: exact b16 fills with untouched output row guards."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured constexpr annotation names.
VALUES = (
    0.0,
    -0.0,
    1.0,
    -2.0,
    1.0 + 2.0**-8,
    1.0 + 3.0 * 2.0**-8,
    1.0 + 2.0**-11,
    1.0 + 3.0 * 2.0**-11,
    2.0**-133,
    2.0**-24,
    2.0**-25,
    65520.0,
    -65520.0,
    float.fromhex("0x1.fffffep+127"),
    float("inf"),
    float("-inf"),
)


@N.prim_func
def b16_fill(
    Y: N.Tensor((16, 96), DTYPE, access="write"),
    DTYPE: N.constexpr = N.bf16,
):
    with N.Kernel(1, kind="vector"):
        work = N.alloc_ub((96,), DTYPE)
        N.fill(work, -7.0)
        for row in N.unroll(16):
            N.fill(work[16:64], VALUES[row])
            # Export the UB guards too: an over-wide Dup cannot be hidden by
            # copying only its intended destination interval to GM.
            N.copy(work, Y[row, :])

"""Stable row softmax with packed Vector statistics and explicit BRCB scratch.

The runtime tile loop retains three stages and independent input/output slots.
The R64/packed merge order and FP32 division are part of this example's contract.
"""

from __future__ import annotations

from ascendflow import language as T


@T.inline
def softmax_compute(x, statistic, partial, D, broadcast, y):
    maximum = T.reduce_max(
        x,
        axis=1,
        out=statistic,
        scratch=partial[:, : D // 64],
        order="r64_then_packed_merge",
    )
    shifted = T.sub(x, maximum[:, None], out=y, scratch=broadcast)
    exponentials = T.exp(shifted, out=y)
    total = T.reduce_sum(
        exponentials,
        axis=1,
        out=statistic,
        scratch=partial[:, : D // 64],
        order="r64_then_packed_merge",
    )
    return T.div(exponentials, total[:, None], out=y, scratch=broadcast)


@T.prim_func(disjoint_external=True)
def softmax(
    X: T.Tensor((GROUPS, tiles, R, D), T.f32),  # noqa: F821
    Y: T.Tensor((GROUPS, tiles, R, D), T.f32, access="write"),  # noqa: F821
    tiles: T.i64,
    R: T.constexpr = 8,
    D: T.constexpr = 1024,
    GROUPS: T.constexpr = 32,
):
    with T.Kernel(GROUPS, kind="vector") as group:
        x = T.alloc_ub((R, D), T.f32, slots=2, name="inputs")
        y = T.alloc_ub((R, D), T.f32, slots=2, name="outputs")
        # Packed merge reads each partial row at a 32-byte boundary.
        partial = T.alloc_ub((R, ((D // 64 + 7) // 8) * 8), T.f32, name="partial")
        statistic = T.alloc_ub((R,), T.f32, name="statistic")
        broadcast = T.alloc_ub((R, 8), T.f32, name="broadcast")
        for tile in T.Pipelined(tiles, issue_order=("store", "load", "compute")):
            with T.Stage(0, name="load"):
                loaded = T.copy(X[group, tile, :, :], x.at(tile))
            with T.Vector(), T.Stage(1, name="compute"):
                normalized = softmax_compute(loaded, statistic, partial, D, broadcast, y.at(tile))
            with T.Stage(2, name="store"):
                T.copy(normalized, Y[group, tile, :, :])

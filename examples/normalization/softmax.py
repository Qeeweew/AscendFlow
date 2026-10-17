"""Stable row softmax with packed Vector statistics and explicit BRCB scratch.

The runtime tile loop retains three stages and independent input/output slots.
The R64/packed merge order and FP32 division are part of this example's contract.
"""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def softmax(
    X: T.Tensor((GROUPS, tiles, R, D), T.f32),  # noqa: F821
    Y: T.Tensor((GROUPS, tiles, R, D), T.f32, access="write"),  # noqa: F821
    tiles: T.int64,
    R: T.constexpr = 8,
    D: T.constexpr = 1024,
    GROUPS: T.constexpr = 32,
):
    with T.Kernel(GROUPS, kind="vector") as group:
        x = T.alloc_ub((2, R, D), T.f32, name="inputs")
        y = T.alloc_ub((2, R, D), T.f32, name="outputs")
        # Packed merge reads each partial row at a 32-byte boundary.
        partial = T.alloc_ub((R, ((D // 64 + 7) // 8) * 8), T.f32, name="partial")
        statistic = T.alloc_ub((R,), T.f32, name="statistic")
        broadcast = T.alloc_ub((R, 8), T.f32, name="broadcast")
        for tile in T.Pipelined(
            tiles, stages=3, issue_order={"vector": ("store", "load", "compute")}
        ):
            slot = tile % 2
            with T.Stage(0, name="load"):
                T.copy(X[group, tile, :, :], x[slot, :, :])
            with T.Stage(1, name="compute"), T.Vector():
                maximum = T.reduce_max(
                    x[slot, :, :],
                    axis=1,
                    out=statistic,
                    scratch=partial[:, : D // 64],
                    order="r64_then_packed_merge",
                )
                T.sub(x[slot, :, :], maximum[:, None], out=y[slot, :, :], scratch=broadcast)
                T.exp(y[slot, :, :], out=y[slot, :, :])
                total = T.reduce_sum(
                    y[slot, :, :],
                    axis=1,
                    out=statistic,
                    scratch=partial[:, : D // 64],
                    order="r64_then_packed_merge",
                )
                T.div(y[slot, :, :], total[:, None], out=y[slot, :, :], scratch=broadcast)
            with T.Stage(2, name="store"):
                T.copy(y[slot, :, :], Y[group, tile, :, :])

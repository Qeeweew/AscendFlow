"""BNSD indexing with cyclic owners and runtime sequence length.

One binary handles unequal owner counts and owners with no work. Each task
scales one [ROWS,D] tile, using explicit two-slot UB buffers. This is the same
runtime batch/head/query descriptor needed by attention, without its math.
"""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def grouped_scale(
    X: T.Tensor((B, H, S, D), T.f32),  # noqa: F821
    Y: T.Tensor((B, H, S, D), T.f32, access="write"),  # noqa: F821
    S: T.int64(minimum=1, multiple_of=ROWS),  # noqa: F821
    B: T.constexpr = 2,
    H: T.constexpr = 3,
    D: T.constexpr = 64,
    ROWS: T.constexpr = 16,
    GROUPS: T.constexpr = 20,
):
    with T.Kernel(GROUPS, kind="vector") as group:
        x = T.alloc_ub((2, ROWS, D), T.f32, name="inputs")
        y = T.alloc_ub((2, ROWS, D), T.f32, name="outputs")
        tiles = S // ROWS
        total = B * H * tiles
        count = T.ceildiv(T.max(0, total - group), GROUPS)
        for task in T.Pipelined(count, stages=3):
            flat = group + task * GROUPS
            batch = flat // (H * tiles)
            head = (flat // tiles) % H
            row = (flat % tiles) * ROWS
            slot = task % 2
            with T.Stage(0, name="load"):
                value = T.copy(X[batch, head, row : row + ROWS, :], x[slot, :, :])
            with T.Stage(1, name="scale"), T.Vector():
                value = T.mul(value, 2.0, out=y[slot, :, :])
            with T.Stage(2, name="store"):
                T.copy(value, Y[batch, head, row : row + ROWS, :])

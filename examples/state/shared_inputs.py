"""Shared read-only GM across serial calls; each call owns its output frame."""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def shared_state_sum(
    Initial: T.Tensor((GROUPS, B), T.f32),  # noqa: F821
    X: T.Tensor((GROUPS, tiles, B), T.f32),  # noqa: F821
    Output: T.Tensor((GROUPS, outputs, B), T.f32, access="write"),  # noqa: F821
    outputs: T.int64,
    tiles: T.int64,
    B: T.constexpr = 4096,
    GROUPS: T.constexpr = 32,
):
    with T.Kernel(GROUPS, kind="vector") as group:
        state = T.alloc_ub((B,), T.f32, name="state")
        x = T.alloc_ub((2, B), T.f32, name="inputs")
        for output in T.serial(outputs):
            T.copy(Initial[group, :], state)
            for t in T.Pipelined(tiles, stages=2):
                with T.Stage(0, name="load"):
                    T.copy(X[group, t, :], x[t % 2, :])
                with T.Stage(1, name="update"), T.Vector():
                    T.add(state, x[t % 2, :], out=state)
            T.copy(state, Output[group, output, :])

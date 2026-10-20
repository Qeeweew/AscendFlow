"""Two runtime axes with actual init/use and asynchronous scratch reuse per call."""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def grouped_state_sum(
    Initial: T.Tensor((GROUPS, outputs, B), T.f32),  # noqa: F821
    X: T.Tensor((GROUPS, outputs, tiles, B), T.f32),  # noqa: F821
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
            T.copy(Initial[group, output, :], state)
            for t in T.Pipelined(tiles):
                with T.Stage(0, name="load"):
                    T.copy(X[group, output, t, :], x[t % 2, :])
                with T.Vector(), T.Stage(1, name="update"):
                    T.add(state, x[t % 2, :], out=state)
            T.copy(state, Output[group, output, :])

"""Public source fixtures with native work on real parent first/last epochs."""

from __future__ import annotations

# Runtime dimensions in annotations are captured by the DSL frontend.
# ruff: noqa: F821
from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def native_guarded_recurrence(
    X: N.Tensor((n, 2, 8, 64), N.f32),
    O: N.Tensor((2, 8, 64), N.f16, access="write"),
    n: N.int64,
):
    with N.Kernel():
        state = N.alloc_ub((2, 8, 64), N.f32, name="state")
        factor = N.alloc_ub((8,), N.f32, name="factor")
        x = N.alloc_ub((2, 8, 64), N.f32, name="x")
        out = N.alloc_ub((2, 8, 64), N.f16, name="out")
        broadcast = N.alloc_ub((8, 8), N.f32, name="broadcast")
        for t in N.Pipelined(n, stages=2):
            with N.Stage(0, name="initialize"):
                if t == 0:
                    N.fill(state[0, :, :], 0.0)
                    N.fill(state[1, :, :], 0.0)
                    N.fill(factor, 0.5)
            with N.Stage(1, name="update"):
                for row in N.Pipelined(2, stages=3):
                    with N.Stage(0, name="load"):
                        N.copy(X[t, row, :, :], x[row % 2, :, :])
                    with N.Stage(1, name="compute"):
                        N.add(state[row, :, :], x[row % 2, :, :], out=state[row, :, :])
                        N.mul(
                            state[row, :, :],
                            factor[:, None],
                            out=state[row, :, :],
                            scratch=broadcast,
                        )
                        if t == n - 1:
                            N.cast(state[row, :, :], out=out[row, :, :])
                    with N.Stage(2, name="store"):
                        if t == n - 1:
                            N.copy(out[row, :, :], O[row, :, :])

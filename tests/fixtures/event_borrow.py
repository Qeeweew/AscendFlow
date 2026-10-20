"""Two real Vector children whose parent releases fill child event idle gaps."""

from __future__ import annotations

# Runtime dimensions in annotations are captured by the DSL frontend.
# ruff: noqa: F821
from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def vector_event_borrow(
    X: N.Tensor((n, 4, 16), N.f32),
    Z: N.Tensor((n, 4, 16), N.f32),
    Y: N.Tensor((n, 4, 16), N.f16, access="write"),
    O: N.Tensor((4, 16), N.f16, access="write"),
    n: N.int64,
):
    with N.Kernel():
        x = N.alloc_ub((2, 16), N.f32, name="x")
        work = N.alloc_ub((16,), N.f32, name="work")
        p = N.alloc_ub((2, 16), N.f16, name="p")
        z = N.alloc_ub((2, 16), N.f32, name="z")
        state = N.alloc_ub((4, 16), N.f32, name="state")
        out = N.alloc_ub((2, 16), N.f16, name="out")
        for t in N.Pipelined(n, issue_order=("produce", "update")):
            with N.Stage(1, name="produce"):
                for j in N.Pipelined(4):
                    with N.Stage(0, name="load"):
                        N.copy(X[t, j, :], x[j % 2, :])
                    with N.Stage(1, name="compute"):
                        N.mul(x[j % 2, :], 2.0, out=work)
                        N.cast(work, out=p[j % 2, :])
                    with N.Stage(2, name="store"):
                        N.copy(p[j % 2, :], Y[t, j, :])
            with N.Stage(2, name="update"):
                if t == 0:
                    N.fill(state, 0.0)
                for j in N.Pipelined(4):
                    with N.Stage(0, name="load"):
                        N.copy(Z[t, j, :], z[j % 2, :])
                    with N.Stage(1, name="compute"):
                        N.add(state[j, :], z[j % 2, :], out=state[j, :])
                        if t == n - 1:
                            N.mul(state[j, :], 2.0, out=z[j % 2, :])
                            N.cast(z[j % 2, :], out=out[j % 2, :])
                    with N.Stage(2, name="store"):
                        if t == n - 1:
                            N.copy(out[j % 2, :], O[j, :])

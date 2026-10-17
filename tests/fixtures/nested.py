"""Real nested pipeline programs with runtime parent clocks and fixed children."""

from __future__ import annotations

from ascendflow import language as N

# Runtime dimensions in annotations are captured by the DSL frontend.
# ruff: noqa: F821
from examples.cv.nested import nested_cv  # noqa: F401 - shared source fixture


@N.prim_func(disjoint_external=True)
def nested_copy(
    X: N.Tensor((n, 4, 8), N.f32),
    Y: N.Tensor((n, 4, 8), N.f32, access="write"),
    O: N.Tensor((n, 4, 8), N.f32, access="write"),
    n: N.int64,
):
    with N.Kernel():
        x = N.alloc_ub((2, 2, 8))
        for t in N.Pipelined(n, stages=2):
            with N.Stage(0, name="produce"):
                for j in N.Pipelined(4, stages=2):
                    with N.Stage(0, name="load"):
                        z = N.copy(X[t, j, :], x[t % 2, j % 2, :])
                    with N.Stage(1, name="store"):
                        N.copy(z, Y[t, j, :])
            with N.Stage(1, name="consume"):
                for j in N.Pipelined(4, stages=2):
                    with N.Stage(0, name="load"):
                        z = N.copy(Y[t, j, :], x[t % 2, j % 2, :])
                    with N.Stage(1, name="store"):
                        N.copy(z, O[t, j, :])


@N.prim_func(disjoint_external=True)
def shared_l0c_qk_pv(
    X: N.Tensor((1, n, 8, 16, 16), N.f16),
    Score: N.Tensor((1, 2, 16, 128), N.f32, access="write"),
    Y: N.Tensor((1, n, 16, 16), N.f32, access="write"),
    n: N.int64,
):
    with N.Kernel(1, kind="cube") as group:
        qk_l1 = N.alloc_l1((2, 16, 16), name="qk_l1")
        qk_a = N.alloc_l0a((2, 16, 16), name="qk_a")
        qk_b = N.alloc_l0b((2, 16, 16), name="qk_b")
        pv_l1 = N.alloc_l1((2, 16, 16), name="pv_l1")
        pv_a = N.alloc_l0a((2, 16, 16), name="pv_a")
        pv_b = N.alloc_l0b((2, 16, 16), name="pv_b")
        c = N.alloc_l0c((2, 16, 16), name="c")
        for t in N.Pipelined(n, stages=2, issue_order={"cube": ("qk", "pv")}):
            with N.Stage(0, name="qk"):
                for j in N.Pipelined(
                    8,
                    stages=4,
                    issue_order={"cube": ("load", "mad", "operands", "export")},
                ):
                    with N.Stage(0, name="load"):
                        x = N.copy(X[group, t, j, :, :], qk_l1[j % 2, :, :])
                    with N.Stage(1, name="operands"):
                        a = N.copy(x, qk_a[j % 2, :, :])
                        b = N.copy(x, qk_b[j % 2, :, :], transpose=True)
                    with N.Stage(2, name="mad"):
                        z = N.matmul(a, b, accum=c[j % 2, :, :], initialize=True)
                    with N.Stage(3, name="export"):
                        N.copy(z, Score[group, t % 2, :, j * 16 : (j + 1) * 16])
            with N.Stage(1, name="pv"):
                for j in N.Pipelined(8, stages=1):
                    x = N.copy(X[group, t, j, :, :], pv_l1[j % 2, :, :])
                    a = N.copy(x, pv_a[j % 2, :, :])
                    b = N.copy(x, pv_b[j % 2, :, :], transpose=True)
                    N.matmul(a, b, accum=c[0, :, :], initialize=(j == 0))
                N.copy(c[0, :, :], Y[group, t, :, :])

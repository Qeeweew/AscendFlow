"""Real nested pipeline programs with runtime parent clocks and fixed children."""

from __future__ import annotations

from ascendflow import language as N

# Runtime dimensions in annotations are captured by the DSL frontend.
# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def nested_cv(
    X: N.Tensor((2, n, 2, 16, 16), N.f16),
    Score: N.Tensor((2, 2, 16, 32), N.f32, access="write"),
    O: N.Tensor((2, n, 16, 32), N.f32, access="write"),
    n: N.int64,
):
    with N.Kernel(2, kind="mixed") as group:
        x1 = N.alloc_l1((2, 16, 16))
        a = N.alloc_l0a((2, 16, 16))
        b = N.alloc_l0b((2, 16, 16))
        c = N.alloc_l0c((2, 16, 16))
        v = N.alloc_ub((2, 4, 32))
        for t in N.Dataflow(n):
            with N.Cube(), N.Stage(0, name="cube"):
                for j in N.Pipelined(2):
                    with N.Stage(0, name="load"):
                        x = N.copy(X[group, t, j, :, :], x1[j % 2, :, :])
                    with N.Stage(1, name="operands"):
                        aa = N.copy(x, a[j % 2, :, :])
                        bb = N.copy(x, b[j % 2, :, :], transpose=True)
                    with N.Stage(2, name="mad"):
                        z = N.matmul(aa, bb, accum=c[j % 2, :, :], initialize=True)
                    with N.Stage(3, name="fix"):
                        N.copy(z, Score[group, t % 2, :, j * 16 : (j + 1) * 16])
            with N.Vector() as lane, N.Stage(1, name="vector"):
                for r in N.Pipelined(2):
                    row = lane * 8 + r * 4
                    with N.Stage(0, name="load"):
                        z = N.copy(Score[group, t % 2, row : row + 4, :], v[r % 2, :, :])
                    with N.Stage(1, name="scale"):
                        z = N.mul(z, 2.0, out=v[r % 2, :, :])
                    with N.Stage(2, name="store"):
                        N.copy(z, O[group, t, row : row + 4, :])


@N.prim_func(disjoint_external=True)
def partitioned_cv(
    X: N.Tensor((n, 2, 16, 16), N.f16),
    Exchange: N.Tensor((2, 32, 16), N.f32, access="write"),
    Back: N.Tensor((2, 32, 16), N.f16, access="write"),
    n: N.int64,
):
    """Two Cube operations address the separate halves of one AIV quorum."""
    with N.Kernel(1, kind="mixed"):
        l1 = N.alloc_l1((2, 16, 16), N.f16)
        a = N.alloc_l0a((2, 16, 16), N.f16)
        b = N.alloc_l0b((2, 16, 16), N.f16)
        c = N.alloc_l0c((2, 16, 16), N.f32)
        receive = N.alloc_l1((2, 16, 16), N.f16)
        u = N.alloc_ub((16, 16), N.f32)
        v = N.alloc_ub((16, 16), N.f16)
        for t in N.Dataflow(n):
            with N.Cube(), N.Stage(0, name="produce"):
                for part in N.unroll(2):
                    x = N.copy(X[t, part, :, :], l1[part, :, :])
                    aa = N.copy(x, a[part, :, :])
                    bb = N.copy(x, b[part, :, :], transpose=True)
                    z = N.matmul(aa, bb, accum=c[part, :, :], initialize=True)
                    N.copy(z, Exchange[t % 2, part * 16 : (part + 1) * 16, :])
            with N.Vector() as lane, N.Stage(1, name="transform"):
                x = N.copy(Exchange[t % 2, lane * 16 : (lane + 1) * 16, :], u)
                y = N.cast(x, out=v)
                N.copy(y, Back[t % 2, lane * 16 : (lane + 1) * 16, :])
            with N.Cube(), N.Stage(2, name="consume"):
                for part in N.unroll(2):
                    N.copy(Back[t % 2, part * 16 : (part + 1) * 16, :], receive[part, :, :])


@N.prim_func(disjoint_external=True)
def nested_copy(
    X: N.Tensor((n, 4, 8), N.f32),
    Y: N.Tensor((n, 4, 8), N.f32, access="write"),
    O: N.Tensor((n, 4, 8), N.f32, access="write"),
    n: N.int64,
):
    with N.Kernel():
        x = N.alloc_ub((2, 2, 8))
        for t in N.Pipelined(n):
            with N.Stage(0, name="produce"):
                for j in N.Pipelined(4):
                    with N.Stage(0, name="load"):
                        z = N.copy(X[t, j, :], x[t % 2, j % 2, :])
                    with N.Stage(1, name="store"):
                        N.copy(z, Y[t, j, :])
            with N.Stage(1, name="consume"):
                for j in N.Pipelined(4):
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
        for t in N.Pipelined(n, issue_order=("qk", "pv")):
            with N.Stage(0, name="qk"):
                for j in N.Pipelined(
                    8,
                    issue_order=("load", "mad", "operands", "export"),
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
                for j in N.Pipelined(8):
                    x = N.copy(X[group, t, j, :, :], pv_l1[j % 2, :, :])
                    a = N.copy(x, pv_a[j % 2, :, :])
                    b = N.copy(x, pv_b[j % 2, :, :], transpose=True)
                    N.matmul(a, b, accum=c[0, :, :], initialize=(j == 0))
                N.copy(c[0, :, :], Y[group, t, :, :])

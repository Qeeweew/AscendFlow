"""Independent panel words with reusable GM slots and both AIV readers."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - constexpr/runtime signature dimensions.


@N.prim_func(disjoint_external=True)
def fifo_panels(
    A: N.Tensor((16, 16), N.bf16),
    B: N.Tensor((16, 16), N.bf16),
    Work: N.Tensor((2, PANELS, 16, 16), N.f32, access="write"),
    Out: N.Tensor((EPOCHS, n, PANELS, 16, 16), N.f32, access="write", write_disjoint=True),
    n: N.i64(minimum=0),
    PANELS: N.constexpr = 9,
    EPOCHS: N.constexpr = 1,
):
    with N.Kernel(1, kind="mixed"):
        a1 = N.alloc_l1((16, 16), N.bf16)
        b1 = N.alloc_l1((16, 16), N.bf16)
        a0 = N.alloc_l0a((16, 16), N.bf16)
        b0 = N.alloc_l0b((16, 16), N.bf16)
        c0 = N.alloc_l0c((16, 16), N.f32)
        x = N.alloc_ub((8, 16), N.f32)
        with N.Cube():
            N.copy(A, a1)
            N.copy(B, b1)
            N.copy(a1, a0)
            N.copy(b1, b0)
            N.matmul(a0, b0, accum=c0, initialize=True)
        for epoch in N.unroll(EPOCHS):
            for i in N.Dataflow(n):
                with N.Cube(), N.Stage(0):
                    for panel in N.unroll(PANELS):
                        N.copy(c0, Work[i % 2, panel, :, :])
                with N.Vector() as lane, N.Stage(1):
                    for panel in N.unroll(PANELS):
                        N.copy(Work[i % 2, panel, lane * 8 : lane * 8 + 8, :], x)
                        N.copy(x, Out[epoch, i, panel, lane * 8 : lane * 8 + 8, :])


@N.prim_func(disjoint_external=True)
def fifo_feedback(
    A: N.Tensor((16, 16), N.bf16),
    B: N.Tensor((16, 16), N.bf16),
    Work: N.Tensor((EPOCHS, 2, PANELS, 16, 16), N.bf16, access="write"),
    Feedback: N.Tensor((EPOCHS, 16, 16), N.bf16, access="write"),
    Out: N.Tensor((EPOCHS, n, PANELS, 16, 16), N.bf16, access="write", write_disjoint=True),
    n: N.i64(minimum=1),
    PANELS: N.constexpr = 9,
    EPOCHS: N.constexpr = 2,
):
    with N.Kernel(1, kind="mixed"):
        a1 = N.alloc_l1((16, 16), N.bf16)
        b1 = N.alloc_l1((16, 16), N.bf16)
        a0 = N.alloc_l0a((16, 16), N.bf16)
        b0 = N.alloc_l0b((16, 16), N.bf16)
        c0 = N.alloc_l0c((16, 16), N.f32)
        x = N.alloc_ub((8, 16), N.bf16)
        with N.Cube():
            N.copy(A, a1)
            N.copy(B, b1)
            N.copy(a1, a0)
            N.copy(b1, b0)
            N.matmul(a0, b0, accum=c0, initialize=True)
        for epoch in N.unroll(EPOCHS):
            if n > 0:
                if epoch > 0:
                    with N.Cube():
                        N.copy(Feedback[epoch - 1, :, :], a1)
                        N.copy(a1, a0)
                        N.matmul(a0, b0, accum=c0, initialize=True)
                for i in N.Dataflow(n):
                    with N.Cube(), N.Stage(0):
                        for panel in N.unroll(PANELS):
                            N.copy(c0, Work[epoch, i % 2, panel, :, :])
                    with N.Vector() as lane, N.Stage(1):
                        for panel in N.unroll(PANELS):
                            N.copy(Work[epoch, i % 2, panel, lane * 8 : lane * 8 + 8, :], x)
                            N.copy(x, Out[epoch, i, panel, lane * 8 : lane * 8 + 8, :])
                with N.Vector() as lane:
                    N.copy(x, Feedback[epoch, lane * 8 : lane * 8 + 8, :])

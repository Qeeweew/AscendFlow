"""Real mixed recurrence with separate GM message credits at each phase."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured constexpr annotation names.


@N.prim_func(disjoint_external=True)
def counter_epochs(
    A: N.Tensor((16, 16), N.f16),
    B: N.Tensor((16, 16), N.f16),
    Work: N.Tensor((PHASES, 16, 16), N.f16, access="write"),
    Feedback: N.Tensor((PHASES, 16, 16), N.f16, access="write"),
    n: N.i64,
    PHASES: N.constexpr = 17,
):
    with N.Kernel(1, kind="mixed"):
        a1 = N.alloc_l1((16, 16), N.f16)
        b1 = N.alloc_l1((16, 16), N.f16)
        a0 = N.alloc_l0a((16, 16), N.f16)
        b0 = N.alloc_l0b((16, 16), N.f16)
        c0 = N.alloc_l0c((16, 16), N.f32)
        x = N.alloc_ub((8, 16), N.f16)
        with N.Cube():
            N.copy(A, a1)
            N.copy(B, b1)
            N.copy(a1, a0)
            N.copy(b1, b0)
        for outer in N.serial(n):
            for phase in N.unroll(PHASES):
                with N.Cube():
                    if phase > 0:
                        N.copy(Feedback[phase - 1, :, :], a1)
                        N.copy(a1, a0)
                    N.matmul(a0, b0, accum=c0, initialize=True)
                    N.copy(c0, Work[phase, :, :])
                with N.Vector() as lane:
                    N.copy(Work[phase, lane * 8 : lane * 8 + 8, :], x)
                    N.copy(x, Feedback[phase, lane * 8 : lane * 8 + 8, :])
            # The next outer iteration consumes the real previous result.
            # This closes reentry through both AIVs without a new barrier.
            with N.Cube():
                N.copy(Feedback[PHASES - 1, :, :], a1)
                N.copy(a1, a0)

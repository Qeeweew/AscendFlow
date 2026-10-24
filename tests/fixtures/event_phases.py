"""Real binary-event phases with a different overlapping pair in each step."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - runtime dimensions are captured by the frontend.


@N.inline
def phase_pair(X, Out, left, right, wide0, wide1, row):
    a = N.copy(X[row, :], left)
    b = N.copy(X[row + 1, :], right)
    af = N.cast(a, out=wide0)
    bf = N.cast(b, out=wide1)
    N.copy(af, Out[row, :])
    N.copy(bf, Out[row + 1, :])


@N.prim_func(disjoint_external=True)
def binary_phases(
    X: N.Tensor((rounds, 6, 64), N.bf16),
    Out: N.Tensor((rounds, 6, 64), N.f32, access="write"),
    rounds: N.i64(minimum=0),
):
    with N.Kernel(kind="vector"):
        a = N.alloc_ub((64,), N.bf16)
        b = N.alloc_ub((64,), N.bf16)
        c = N.alloc_ub((64,), N.bf16)
        wide0 = N.alloc_ub((64,), N.f32)
        wide1 = N.alloc_ub((64,), N.f32)
        for turn in N.serial(rounds):
            phase_pair(X[turn, :, :], Out[turn, :, :], a, b, wide0, wide1, 0)
            phase_pair(X[turn, :, :], Out[turn, :, :], b, c, wide0, wide1, 2)
            phase_pair(X[turn, :, :], Out[turn, :, :], c, a, wide0, wide1, 4)


@N.prim_func(disjoint_external=True)
def binary_phases_mixed(
    X: N.Tensor((rounds, 2, 6, 64), N.bf16),
    Out: N.Tensor((rounds, 2, 6, 64), N.f32, access="write"),
    rounds: N.i64(minimum=0),
):
    with N.Kernel(kind="mixed"):
        a = N.alloc_ub((64,), N.bf16)
        b = N.alloc_ub((64,), N.bf16)
        c = N.alloc_ub((64,), N.bf16)
        wide0 = N.alloc_ub((64,), N.f32)
        wide1 = N.alloc_ub((64,), N.f32)
        for turn in N.serial(rounds):
            with N.Vector() as sid:
                phase_pair(X[turn, sid, :, :], Out[turn, sid, :, :], a, b, wide0, wide1, 0)
                phase_pair(X[turn, sid, :, :], Out[turn, sid, :, :], b, c, wide0, wide1, 2)
                phase_pair(X[turn, sid, :, :], Out[turn, sid, :, :], c, a, wide0, wide1, 4)


@N.prim_func(disjoint_external=True)
def full_bank_phases(
    X: N.Tensor((rounds, 72, 64), N.bf16),
    Out: N.Tensor((rounds, 72, 64), N.f32, access="write"),
    rounds: N.i64(minimum=0),
    WORD: N.constexpr = (
        0,
        1,
        2,
        0,
        3,
        1,
        4,
        0,
        5,
        1,
        6,
        0,
        7,
        1,
        8,
        2,
        3,
        4,
        2,
        5,
        3,
        6,
        2,
        7,
        3,
        8,
        4,
        5,
        6,
        4,
        7,
        5,
        8,
        6,
        7,
        8,
        0,
    ),
):
    """Every pair overlaps once; a real return separates successive phases.

    The word is an Euler circuit of the nine-credit complete graph. Whole
    credits need nine distinct numbers under their supported constraints;
    each actual phase has only two members and keeps the original native work.
    """
    with N.Kernel(kind="vector"):
        payload = N.alloc_ub((64,), N.bf16, slots=9)
        wide0 = N.alloc_ub((64,), N.f32)
        wide1 = N.alloc_ub((64,), N.f32)
        for turn in N.serial(rounds):
            for step in N.unroll(36):
                phase_pair(
                    X[turn, :, :],
                    Out[turn, :, :],
                    payload.slot(WORD[step]),
                    payload.slot(WORD[step + 1]),
                    wide0,
                    wide1,
                    2 * step,
                )

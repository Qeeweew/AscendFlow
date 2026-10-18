"""Continuous FP32 Cube inter-block Gram pipeline with bounded gate factors."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def prefill_gram_inter(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    A: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    GramLeft: N.Tensor((GROUPS, 2, 32, K), N.f32, access="write"),
    GramRight: N.Tensor((GROUPS, 2, 16, K), N.f32, access="write"),
    GramScore: N.Tensor((GROUPS, 2, 32, 16), N.f32, access="write"),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64, "six lower-triangular block pairs"
        raw = N.alloc_ub((16, 64), DTYPE)
        q = N.alloc_ub((16, 64), N.f32)
        k = N.alloc_ub((16, 64), N.f32)
        gates = N.alloc_ub((16, 64), N.f32)
        factor = N.alloc_ub((16, 64), N.f32)
        anchor = N.alloc_ub((64,), N.f32)
        scores = N.alloc_ub((16, 16), N.f32)
        betas = N.alloc_ub((16,), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        a1 = N.alloc_l1((2, 32, K), N.f32)
        b1 = N.alloc_l1((2, 16, K), N.f32)
        a0 = N.alloc_l0a((32, K), N.f32)
        b0 = N.alloc_l0b((K, 16), N.f32)
        c0 = N.alloc_l0c((32, 16), N.f32)
        for item in N.Pipelined(N.ceildiv(N.max(0, CHUNKS * H * 6 - owner), GROUPS), stages=3):
            work = owner + item * GROUPS
            chunk, head = work // (H * 6), (work // 6) % H
            pair = work % 6
            # (1,0), (2,0), (2,1), (3,0), (3,1), (3,2).
            block = 1 + N.min(pair, 1) + pair // 3
            column_block = N.max(N.max(pair - 3, 0), N.max(N.min(pair, 2) // 2 - pair // 3, 0))
            row, column = block * 16, column_block * 16
            ring = item % 2
            with N.Stage(0, name="prepare"), N.Vector() as lane:
                col = lane * 64
                N.copy(Gp[chunk, head, row, col : col + 64], anchor)
                N.copy(Gp[chunk, head, row : row + 16, col : col + 64], gates)
                N.mul(anchor, -1.0, out=anchor)
                N.add(gates, anchor[None, :], out=factor)
                N.exp(factor, out=factor)
                N.copy(Kp[chunk, head, row : row + 16, col : col + 64], raw)
                N.cast(raw, out=k)
                N.mul(k, factor, out=k)
                N.copy(k, GramLeft[owner, ring, :16, col : col + 64])
                N.copy(Qp[chunk, head, row : row + 16, col : col + 64], raw)
                N.cast(raw, out=q)
                N.mul(q, factor, out=q)
                N.mul(q, SCALE, out=q)
                N.copy(q, GramLeft[owner, ring, 16:32, col : col + 64])
                N.copy(Gp[chunk, head, column : column + 16, col : col + 64], gates)
                N.add(gates, anchor[None, :], out=factor)
                N.mul(factor, -1.0, out=factor)
                N.exp(factor, out=factor)
                N.copy(Kp[chunk, head, column : column + 16, col : col + 64], raw)
                N.cast(raw, out=k)
                N.mul(k, factor, out=k)
                N.copy(k, GramRight[owner, ring, :, col : col + 64])
            with N.Stage(1, name="dot"), N.Cube():
                aa = N.copy(GramLeft[owner, ring, :, :], a1[ring, :, :])
                bb = N.copy(GramRight[owner, ring, :, :], b1[ring, :, :])
                aa = N.copy(aa, a0)
                bb = N.copy(bb, b0, transpose=True)
                N.matmul(aa, bb, accum=c0, initialize=True)
                N.copy(c0, GramScore[owner, ring, :, :])
            with N.Stage(2, name="store"), N.Vector() as lane:
                valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                N.copy(GramScore[owner, ring, lane * 16 : (lane + 1) * 16, :], scores)
                N.fill(scores, 0.0, mask=N.arange(16)[:, None] + row >= valid)
                if lane == 0:
                    N.copy(Bp[chunk, head, row : row + 16], betas)
                    N.mul(scores, betas[:, None], out=scores, scratch=rows)
                    N.copy(scores, A[chunk, head, row : row + 16, column : column + 16])
                else:
                    N.copy(scores, Aqk[chunk, head, row : row + 16, column : column + 16])

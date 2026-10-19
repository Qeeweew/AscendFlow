"""L2-sized Gram batches: Vector weights, pipelined FP32 Cube, Vector store.

The host orders the five phases of each slab on one stream. Workspace is
reused only after that slab's final reader has completed. No mixed-kernel
per-matrix C/V handshake is needed.
"""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def prefill_gram_prepare(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    GramLeft: N.Tensor((SLAB, 3, 32, K), N.f32, access="write", write_disjoint=True),
    GramRight: N.Tensor((SLAB, 96, K), N.f32, access="write", write_disjoint=True),
    SLAB_INDEX: N.int64(minimum=0),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    SLAB: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        assert K == 128 and BT == 64, "three strict lower-triangular row panels"
        raw = N.alloc_ub((48, K), DTYPE)
        q = N.alloc_ub((16, K), N.f32)
        k = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((16, K), N.f32)
        factor = N.alloc_ub((16, K), N.f32)
        anchor = N.alloc_ub((K,), N.f32)
        right_gates = N.alloc_ub((48, K), N.f32)
        right_factor = N.alloc_ub((48, K), N.f32)
        right = N.alloc_ub((48, K), N.f32)
        assert SLAB % H == 0 and CHUNKS % (SLAB // H) == 0, "complete L2 slabs"
        first = (SLAB_INDEX % (CHUNKS // (SLAB // H))) * (SLAB // H)
        for item in N.serial(N.ceildiv(N.max(0, SLAB - owner), GROUPS)):
            local = owner + item * GROUPS
            chunk, head = first + local // H, local % H
            for panel in N.unroll(1, 4):
                width = panel * 16
                offset = panel * (panel - 1) * 8
                N.copy(Gp[chunk, head, width, :], anchor)
                N.copy(Gp[chunk, head, width : width + 16, :], gates)
                N.mul(anchor, -1.0, out=anchor)
                N.add(gates, anchor[None, :], out=factor)
                N.exp(factor, out=factor)
                N.copy(Kp[chunk, head, width : width + 16, :], raw[:16, :])
                N.cast(raw[:16, :], out=k)
                N.mul(k, factor, out=k)
                N.copy(k, GramLeft[local, panel - 1, :16, :])
                N.copy(Qp[chunk, head, width : width + 16, :], raw[:16, :])
                N.cast(raw[:16, :], out=q)
                N.mul(q, factor, out=q)
                N.mul(q, SCALE, out=q)
                N.copy(q, GramLeft[local, panel - 1, 16:32, :])
                N.copy(Gp[chunk, head, :width, :], right_gates[:width, :])
                N.add(right_gates[:width, :], anchor[None, :], out=right_factor[:width, :])
                N.mul(right_factor[:width, :], -1.0, out=right_factor[:width, :])
                N.exp(right_factor[:width, :], out=right_factor[:width, :])
                N.copy(Kp[chunk, head, :width, :], raw[:width, :])
                N.cast(raw[:width, :], out=right[:width, :])
                N.mul(right[:width, :], right_factor[:width, :], out=right[:width, :])
                N.copy(right[:width, :], GramRight[local, offset : offset + width, :])


@N.prim_func(disjoint_external=True)
def prefill_gram_dot(
    GramLeft: N.Tensor((SLAB, 3, 32, K), N.f32),
    GramRight: N.Tensor((SLAB, 96, K), N.f32),
    GramScore: N.Tensor((SLAB, 32, 96), N.f32, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    SLAB: N.constexpr = 64,
    PANEL: N.constexpr = 1,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="cube") as owner:
        assert K == 128 and PANEL >= 1 and PANEL <= 3, "complete FP32 row panels"
        width = PANEL * 16
        offset = PANEL * (PANEL - 1) * 8
        a1 = N.alloc_l1((2, 32, K), N.f32)
        b1 = N.alloc_l1((2, width, K), N.f32)
        a0 = N.alloc_l0a((32, K), N.f32)
        b0 = N.alloc_l0b((K, width), N.f32)
        c0 = N.alloc_l0c((32, width), N.f32)
        for item in N.Pipelined(N.ceildiv(N.max(0, SLAB - owner), GROUPS), stages=2):
            local = owner + item * GROUPS
            ring = item % 2
            with N.Stage(0, name="load"):
                N.copy(GramLeft[local, PANEL - 1, :, :], a1[ring, :, :])
                N.copy(GramRight[local, offset : offset + width, :], b1[ring, :, :])
            with N.Stage(1, name="compute"):
                aa = N.copy(a1[ring, :, :], a0)
                bb = N.copy(b1[ring, :, :], b0, transpose=True)
                N.matmul(aa, bb, accum=c0, initialize=True)
                N.copy(c0, GramScore[local, :, offset : offset + width])


@N.prim_func(disjoint_external=True)
def prefill_gram_store(
    GramScore: N.Tensor((SLAB, 32, 96), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    Negative: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    SLAB_INDEX: N.int64(minimum=0),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    BT: N.constexpr = 64,
    SLAB: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        betas = N.alloc_ub((16,), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        scores16 = N.alloc_ub((16, 16), N.f32)
        scores32 = N.alloc_ub((16, 32), N.f32)
        scores48 = N.alloc_ub((16, 48), N.f32)
        assert SLAB % H == 0 and CHUNKS % (SLAB // H) == 0, "complete L2 slabs"
        first = (SLAB_INDEX % (CHUNKS // (SLAB // H))) * (SLAB // H)
        for item in N.serial(N.ceildiv(N.max(0, SLAB - owner), GROUPS)):
            local = owner + item * GROUPS
            chunk, head = first + local // H, local % H
            valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
            for panel in N.unroll(1, 4):
                width = panel * 16
                offset = panel * (panel - 1) * 8
                if panel == 1:
                    scores = scores16
                elif panel == 2:
                    scores = scores32
                else:
                    scores = scores48
                N.copy(GramScore[local, :16, offset : offset + width], scores)
                if width + 16 > valid:
                    N.fill(scores, 0.0, mask=N.arange(16)[:, None] + width >= valid)
                N.copy(Bp[chunk, head, width : width + 16], betas)
                N.mul(scores, betas[:, None], out=scores, scratch=rows)
                N.mul(scores, -1.0, out=scores)
                N.copy(scores, Negative[chunk, head, width : width + 16, :width])
                N.copy(GramScore[local, 16:32, offset : offset + width], scores)
                if width + 16 > valid:
                    N.fill(scores, 0.0, mask=N.arange(16)[:, None] + width >= valid)
                N.copy(scores, Aqk[chunk, head, width : width + 16, :width])

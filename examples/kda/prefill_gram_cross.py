"""Persistent hierarchical cross-Gram with buffered C/V cohorts."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def prepare_one_gram(inputs, buffers, geometry, coordinates):
    Gp, Kp, GramLeft, Qp, GramRight = inputs
    anchor, gates, factor, raw, k, q, right_factor, right = buffers
    (SCALE,) = geometry
    chunk, head, lane, owner, ring, item = coordinates
    for panel in N.unroll(3):
        if panel < 2:
            start, col, extent = 16 + panel * 32, panel * 32, 16
            left_offset, right_offset = panel * 32, panel * 16
        else:
            start, col, extent = 32, 0, 32
            left_offset, right_offset = 64, 32
        half = extent // 2
        row = start + lane * half
        right_row = col + lane * half
        N.copy(Gp[chunk, head, start, :], anchor)
        N.copy(Gp[chunk, head, row : row + half, :], gates[:half, :])
        N.mul(anchor, -1.0, out=anchor)
        N.add(gates[:half, :], anchor[None, :], out=factor[:half, :])
        N.exp(factor[:half, :], out=factor[:half, :])
        N.copy(Kp[chunk, head, row : row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=k[:half, :])
        N.mul(k[:half, :], factor[:half, :], out=k[:half, :])
        N.copy(
            k[:half, :],
            GramLeft[
                owner,
                ring,
                item,
                left_offset + lane * half : left_offset + lane * half + half,
                :,
            ],
        )
        N.copy(Qp[chunk, head, row : row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=q[:half, :])
        N.mul(q[:half, :], factor[:half, :], out=q[:half, :])
        N.mul(q[:half, :], SCALE, out=q[:half, :])
        N.copy(
            q[:half, :],
            GramLeft[
                owner,
                ring,
                item,
                left_offset + extent + lane * half : left_offset + extent + lane * half + half,
                :,
            ],
        )
        N.copy(Gp[chunk, head, right_row : right_row + half, :], gates[:half, :])
        N.add(gates[:half, :], anchor[None, :], out=right_factor[:half, :])
        N.mul(right_factor[:half, :], -1.0, out=right_factor[:half, :])
        N.exp(right_factor[:half, :], out=right_factor[:half, :])
        N.copy(Kp[chunk, head, right_row : right_row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=right[:half, :])
        N.mul(right[:half, :], right_factor[:half, :], out=right[:half, :])
        N.copy(
            right[:half, :],
            GramRight[
                owner,
                ring,
                item,
                right_offset + lane * half : right_offset + lane * half + half,
                :,
            ],
        )


@N.inline
def prepare_gram_operands(inputs, buffers, geometry, coordinates):
    PACK, H, SCALE = geometry
    base, lane, owner, ring = coordinates
    for item in N.serial(PACK):
        flat = base + item
        prepare_one_gram(inputs, buffers, (SCALE,), (flat // H, flat % H, lane, owner, ring, item))


@N.inline
def import_gram_item(inputs, buffers, coordinates):
    GramLeft, GramRight = inputs
    a1, b1_small, b1_large = buffers
    owner, ring, item, native_ring = coordinates
    N.copy(GramLeft[owner, ring, item, :, :], a1[native_ring, :, :])
    for panel in N.unroll(2):
        N.copy(
            GramRight[owner, ring, item, panel * 16 : panel * 16 + 16, :],
            b1_small[native_ring, panel, :, :],
        )
    N.copy(GramRight[owner, ring, item, 32:64, :], b1_large[native_ring, :, :])


@N.inline
def compute_gram_item(outputs, buffers, coordinates):
    GramSmall, GramLarge = outputs
    b0_large, a1, b1_large, a0_small, b0_small, c0_small, a0_large, c0_large, b1_small = buffers
    owner, ring, item, native_ring = coordinates
    for panel in N.unroll(2):
        aa = N.copy(a1[native_ring, panel * 32 : panel * 32 + 32, :], a0_small)
        bb = N.copy(
            b1_small[native_ring, panel, :, :],
            b0_small,
            transpose=True,
        )
        N.matmul(aa, bb, accum=c0_small, initialize=True)
        N.copy(c0_small, GramSmall[owner, ring, item, panel, :, :])
    bb = N.copy(b1_large[native_ring, :, :], b0_large, transpose=True)
    for part in N.unroll(2):
        aa = N.copy(a1[native_ring, 64 + part * 32 : 96 + part * 32, :], a0_large)
        N.matmul(aa, bb, accum=c0_large, initialize=True)
        N.copy(
            c0_large,
            GramLarge[owner, ring, item, part * 32 : part * 32 + 32, :],
        )


@N.inline
def gram_products(inputs, buffers, geometry, coordinates):
    GramLeft, GramRight, GramSmall, GramLarge = inputs
    _b0_large, a1, b1_large, _a0_small, _b0_small, _c0_small, _a0_large, _c0_large, b1_small = (
        buffers
    )
    (PACK,) = geometry
    owner, ring = coordinates
    for item in N.Pipelined(PACK):
        native_ring = item % 2
        with N.Stage(0, name="load"):
            import_gram_item(
                (GramLeft, GramRight), (a1, b1_small, b1_large), (owner, ring, item, native_ring)
            )
        with N.Stage(1, name="dot"):
            compute_gram_item((GramSmall, GramLarge), buffers, (owner, ring, item, native_ring))


@N.inline
def store_one_gram(inputs, buffers, coordinates):
    GramSmall, GramLarge, Bp, Negative, Aqk = inputs
    scores32, scores16, betas, rows = buffers
    chunk, head, valid, lane, owner, ring, item = coordinates
    for panel in N.unroll(3):
        if panel < 2:
            start, col, extent = 16 + panel * 32, panel * 32, 16
            scores = scores16[:8, :]
            score_source = GramSmall[owner, ring, item, panel, :, :]
        else:
            start, col, extent = 32, 0, 32
            scores = scores32
            score_source = GramLarge[owner, ring, item, :, :]
        half = extent // 2
        row = start + lane * half
        N.copy(score_source[lane * half : lane * half + half, :], scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(Bp[chunk, head, row : row + half], betas[:half])
        N.mul(scores, betas[:half, None], out=scores, scratch=rows[:half, :])
        N.mul(scores, -1.0, out=scores)
        N.copy(scores, Negative[chunk, head, row : row + half, col : col + extent])
        N.copy(
            score_source[extent + lane * half : extent + lane * half + half, :],
            scores,
        )
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(scores, Aqk[chunk, head, row : row + half, col : col + extent])


@N.inline
def store_gram_blocks(inputs, buffers, geometry, coordinates):
    Lengths, GramSmall, GramLarge, Bp, Negative, Aqk = inputs
    PACK, H = geometry
    base, lane, owner, ring = coordinates
    for item in N.serial(PACK):
        flat = base + item
        chunk, head = flat // H, flat % H
        valid = N.load_index(Lengths[chunk : chunk + 1])
        store_one_gram(
            (GramSmall, GramLarge, Bp, Negative, Aqk),
            buffers,
            (chunk, head, valid, lane, owner, ring, item),
        )


@N.prim_func(disjoint_external=True)
def prefill_gram_cross(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    Negative: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    GramLeft: N.Tensor((GROUPS, 2, PACK, 128, K), N.f32, access="write"),
    GramRight: N.Tensor((GROUPS, 2, PACK, 64, K), N.f32, access="write"),
    GramSmall: N.Tensor((GROUPS, 2, PACK, 2, 32, 16), N.f32, access="write"),
    GramLarge: N.Tensor((GROUPS, 2, PACK, 64, 32), N.f32, access="write"),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    PACK: N.constexpr = 16,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.bf16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64, "hierarchical 16/32-token Gram rectangles"
        assert PACK > 0 and CHUNKS * H % PACK == 0, "complete task cohorts"
        raw = N.alloc_ub((16, K), DTYPE)
        q = N.alloc_ub((16, K), N.f32)
        k = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((16, K), N.f32)
        factor = N.alloc_ub((16, K), N.f32)
        anchor = N.alloc_ub((K,), N.f32)
        right_factor = N.alloc_ub((16, K), N.f32)
        right = N.alloc_ub((16, K), N.f32)
        betas = N.alloc_ub((16,), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        scores16 = N.alloc_ub((16, 16), N.f32)
        scores32 = N.alloc_ub((16, 32), N.f32)
        # One actual DMA defines the complete left slab. Its L1 reuse waits
        # for all four readers; right operands retain compact transpose planes.
        a1 = N.alloc_l1((2, 128, K), N.f32)
        b1_small = N.alloc_l1((2, 2, 16, K), N.f32)
        b1_large = N.alloc_l1((2, 32, K), N.f32)
        a0_small = N.alloc_l0a((32, K), N.f32)
        a0_large = N.alloc_l0a((32, K), N.f32)
        b0_small = N.alloc_l0b((K, 16), N.f32)
        b0_large = N.alloc_l0b((K, 32), N.f32)
        c0_small = N.alloc_l0c((32, 16), N.f32)
        c0_large = N.alloc_l0c((32, 32), N.f32)
        for cohort in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H // PACK - owner), GROUPS),
            issue_order={"vector": ("store", "prepare")},
        ):
            base = (owner + cohort * GROUPS) * PACK
            ring = cohort % 2
            with N.Vector() as lane, N.Stage(0, name="prepare"):
                prepare_gram_operands(
                    (Gp, Kp, GramLeft, Qp, GramRight),
                    (anchor, gates, factor, raw, k, q, right_factor, right),
                    (PACK, H, SCALE),
                    (base, lane, owner, ring),
                )
            with N.Cube(), N.Stage(1, name="compute"):
                gram_products(
                    (GramLeft, GramRight, GramSmall, GramLarge),
                    (
                        b0_large,
                        a1,
                        b1_large,
                        a0_small,
                        b0_small,
                        c0_small,
                        a0_large,
                        c0_large,
                        b1_small,
                    ),
                    (PACK,),
                    (owner, ring),
                )
            with N.Vector() as lane, N.Stage(2, name="store"):
                store_gram_blocks(
                    (Lengths, GramSmall, GramLarge, Bp, Negative, Aqk),
                    (scores32, scores16, betas, rows),
                    (PACK, H),
                    (base, lane, owner, ring),
                )

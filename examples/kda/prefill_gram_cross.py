"""Persistent hierarchical cross-Gram with buffered C/V cohorts."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def prepare_one_gram(inputs, buffers, geometry, coordinates):
    Gp, Kp, GramLeft, Qp, GramRight = inputs
    anchor, gates, factor, raw, k, q, right_factor, right = buffers
    (SCALE,) = geometry
    (lane,) = coordinates
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
        N.copy(Gp[start, :], anchor)
        N.copy(Gp[row : row + half, :], gates[:half, :])
        N.mul(anchor, -1.0, out=anchor)
        N.add(gates[:half, :], anchor[None, :], out=factor[:half, :])
        N.exp(factor[:half, :], out=factor[:half, :])
        N.copy(Kp[row : row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=k[:half, :])
        N.mul(k[:half, :], factor[:half, :], out=k[:half, :])
        N.copy(
            k[:half, :],
            GramLeft[
                left_offset + lane * half : left_offset + lane * half + half,
                :,
            ],
        )
        N.copy(Qp[row : row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=q[:half, :])
        N.mul(q[:half, :], factor[:half, :], out=q[:half, :])
        N.mul(q[:half, :], SCALE, out=q[:half, :])
        N.copy(
            q[:half, :],
            GramLeft[
                left_offset + extent + lane * half : left_offset + extent + lane * half + half,
                :,
            ],
        )
        N.copy(Gp[right_row : right_row + half, :], gates[:half, :])
        N.add(gates[:half, :], anchor[None, :], out=right_factor[:half, :])
        N.mul(right_factor[:half, :], -1.0, out=right_factor[:half, :])
        N.exp(right_factor[:half, :], out=right_factor[:half, :])
        N.copy(Kp[right_row : right_row + half, :], raw[:half, :])
        N.cast(raw[:half, :], out=right[:half, :])
        N.mul(right[:half, :], right_factor[:half, :], out=right[:half, :])
        N.copy(
            right[:half, :],
            GramRight[
                right_offset + lane * half : right_offset + lane * half + half,
                :,
            ],
        )


@N.inline
def prepare_gram_operands(inputs, buffers, geometry, coordinates):
    Gp, Kp, GramLeft, Qp, GramRight = inputs
    PACK, H, SCALE = geometry
    base, lane = coordinates
    for item in N.serial(PACK):
        flat = base + item
        chunk, head = flat // H, flat % H
        prepare_one_gram(
            (
                Gp[chunk, head, :, :],
                Kp[chunk, head, :, :],
                GramLeft[item, :, :],
                Qp[chunk, head, :, :],
                GramRight[item, :, :],
            ),
            buffers,
            (SCALE,),
            (lane,),
        )


@N.inline
def import_gram_item(inputs, buffers):
    GramLeft, GramRight = inputs
    a1, b1_small, b1_large = buffers
    with N.CopyGroup(name="gram_operands"):
        left = N.copy(GramLeft, a1)
        for panel in N.unroll(2):
            N.copy(GramRight[panel * 16 : panel * 16 + 16, :], b1_small[panel, :, :])
        right = N.copy(GramRight[32:64, :], b1_large)
    return left, right


@N.inline
def compute_gram_item(outputs, operands, buffers):
    GramSmall, GramLarge = outputs
    a1, b1_small, b1_large = operands
    b0_large, a0_small, b0_small, c0_small, a0_large, c0_large = buffers
    for panel in N.unroll(2):
        aa = N.copy(a1[panel * 32 : panel * 32 + 32, :], a0_small)
        bb = N.copy(b1_small[panel, :, :], b0_small, transpose=True)
        result = N.matmul(aa, bb, accum=c0_small, initialize=True)
        N.copy(result, GramSmall[panel, :, :])
    bb = N.copy(b1_large, b0_large, transpose=True)
    for part in N.unroll(2):
        aa = N.copy(a1[64 + part * 32 : 96 + part * 32, :], a0_large)
        result = N.matmul(aa, bb, accum=c0_large, initialize=True)
        N.copy(
            result,
            GramLarge[part * 32 : part * 32 + 32, :],
        )


@N.inline
def gram_products(inputs, buffers, geometry):
    GramLeft, GramRight, GramSmall, GramLarge = inputs
    b0_large, a1, b1_large, a0_small, b0_small, c0_small, a0_large, c0_large, b1_small = buffers
    (PACK,) = geometry
    for item in N.Pipelined(PACK, issue_order=("load", "dot")):
        with N.Stage(0, name="load"):
            left, right = import_gram_item(
                (GramLeft[item, :, :], GramRight[item, :, :]),
                (a1.at(item), b1_small.at(item), b1_large.at(item)),
            )
        with N.Stage(1, name="dot"):
            compute_gram_item(
                (GramSmall[item, :, :, :], GramLarge[item, :, :]),
                (left, b1_small.at(item), right),
                (b0_large, a0_small, b0_small, c0_small, a0_large, c0_large),
            )


@N.inline
def store_one_gram(inputs, buffers, coordinates):
    GramSmall, GramLarge, Bp, Negative, Aqk = inputs
    scores32, scores16, betas, rows = buffers
    valid, lane = coordinates
    for panel in N.unroll(3):
        if panel < 2:
            start, col, extent = 16 + panel * 32, panel * 32, 16
            scores = scores16[:8, :]
            score_source = GramSmall[panel, :, :]
        else:
            start, col, extent = 32, 0, 32
            scores = scores32
            score_source = GramLarge
        half = extent // 2
        row = start + lane * half
        N.copy(score_source[lane * half : lane * half + half, :], scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(Bp[row : row + half], betas[:half])
        N.mul(scores, betas[:half, None], out=scores, scratch=rows[:half, :])
        N.mul(scores, -1.0, out=scores)
        N.copy(scores, Negative[row : row + half, col : col + extent])
        N.copy(
            score_source[extent + lane * half : extent + lane * half + half, :],
            scores,
        )
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(scores, Aqk[row : row + half, col : col + extent])


@N.inline
def store_gram_blocks(inputs, buffers, geometry, coordinates):
    Lengths, GramSmall, GramLarge, Bp, Negative, Aqk = inputs
    PACK, H = geometry
    base, lane = coordinates
    for item in N.serial(PACK):
        flat = base + item
        chunk, head = flat // H, flat % H
        valid = N.load_index(Lengths[chunk : chunk + 1])
        store_one_gram(
            (
                GramSmall[item, :, :, :],
                GramLarge[item, :, :],
                Bp[chunk, head, :],
                Negative[chunk, head, :, :],
                Aqk[chunk, head, :, :],
            ),
            buffers,
            (valid, lane),
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
        a1 = N.alloc_l1((128, K), N.f32, slots=2)
        b1_small = N.alloc_l1((2, 16, K), N.f32, slots=2)
        b1_large = N.alloc_l1((32, K), N.f32, slots=2)
        a0_small = N.alloc_l0a((32, K), N.f32)
        a0_large = N.alloc_l0a((32, K), N.f32)
        b0_small = N.alloc_l0b((K, 16), N.f32)
        b0_large = N.alloc_l0b((K, 32), N.f32)
        c0_small = N.alloc_l0c((32, 16), N.f32)
        c0_large = N.alloc_l0c((32, 32), N.f32)
        left_values = N.slotted_view(GramLeft[owner, :, :, :, :])
        right_values = N.slotted_view(GramRight[owner, :, :, :, :])
        small_values = N.slotted_view(GramSmall[owner, :, :, :, :, :])
        large_values = N.slotted_view(GramLarge[owner, :, :, :, :])
        for cohort in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H // PACK - owner), GROUPS),
            issue_order={"cube": ("compute",), "vector": ("store", "prepare")},
        ):
            base = (owner + cohort * GROUPS) * PACK
            with N.Vector() as lane, N.Stage(0, name="prepare"):
                prepare_gram_operands(
                    (Gp, Kp, left_values.at(cohort), Qp, right_values.at(cohort)),
                    (anchor, gates, factor, raw, k, q, right_factor, right),
                    (PACK, H, SCALE),
                    (base, lane),
                )
            with N.Cube(), N.Stage(1, name="compute"):
                gram_products(
                    (
                        left_values.at(cohort),
                        right_values.at(cohort),
                        small_values.at(cohort),
                        large_values.at(cohort),
                    ),
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
                )
            with N.Vector() as lane, N.Stage(2, name="store"):
                store_gram_blocks(
                    (Lengths, small_values.at(cohort), large_values.at(cohort), Bp, Negative, Aqk),
                    (scores32, scores16, betas, rows),
                    (PACK, H),
                    (base, lane),
                )

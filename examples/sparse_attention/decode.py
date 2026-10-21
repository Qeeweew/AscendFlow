"""TND sparse decode, including multiple verification tokens per request.

Sparse rows are gathered once per token/head tile; window pages go directly to L1.
QK and PV keep explicit L1/L0 rings, with one persistent query pipeline.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.sparse_attention.gather import copy_window_pages, gather_compressed, query_metadata

# ruff: noqa: F821 - captured constexpr annotation names.
NEG_INF = float("-inf")


@N.inline
def load_cache(cache, geometry, position, destination, first, feature, IS_WINDOW, PAGE, C, MAX_SEQ):
    ori_kv, ori_block_table, KV = cache
    batch, length, visible = geometry
    group, cache_slot, _slot = position
    if IS_WINDOW:
        copy_window_pages(
            ori_kv,
            ori_block_table,
            destination,
            batch=batch,
            length=length,
            visible=visible,
            feature=feature,
            PAGE=PAGE,
            C=C,
            MAX_SEQ=MAX_SEQ,
        )
    else:
        N.copy(KV[group, cache_slot, first : first + 128, feature : feature + C], destination)


@N.inline
def qk_tiles(cache, geometry, position, Score, l1, l0, S, PAGE, D, C, MAX_SEQ, KEEP, CACHE_Q):
    q1, k1, resident = l1
    q0, b0, c0 = l0
    group, _cache_slot, slot = position
    if CACHE_Q:
        for part in N.unroll(D // C):
            N.copy(q1[:, part * C : (part + 1) * C], q0[part, :, :])
    # Window first, then sparse rows: window QK can overlap the AIV gather.
    for plane in N.unroll(2):
        WIDTH = 128 if plane == 0 else S - 128
        OFFSET = plane * 128
        # Transfer one L1 panel for two consecutive 128-wide L0 operands.
        LC = 2 * C if D % (2 * C) == 0 else C
        for n in N.unroll(WIDTH // 128):
            for panel in N.Pipelined(D // LC, issue_order=("qk_load", "qk_dot", "qk_operand")):
                d = panel * LC
                with N.Stage(0, name="qk_load"):
                    if n + OFFSET // 128 < KEEP:
                        load_cache(
                            cache,
                            geometry,
                            position,
                            resident[n + OFFSET // 128, :, d : d + LC],
                            first=n * 128,
                            feature=d,
                            IS_WINDOW=plane == 0,
                            PAGE=PAGE,
                            C=LC,
                            MAX_SEQ=MAX_SEQ,
                        )
                    else:
                        load_cache(
                            cache,
                            geometry,
                            position,
                            k1[panel % 2, :, 0:LC],
                            first=n * 128,
                            feature=d,
                            IS_WINDOW=plane == 0,
                            PAGE=PAGE,
                            C=LC,
                            MAX_SEQ=MAX_SEQ,
                        )
                with N.Stage(1, name="qk_operand"):
                    for part in N.unroll(LC // C):
                        bank = part if LC > C else panel % 2
                        k = d + part * C
                        if CACHE_Q == 0:
                            N.copy(q1[:, k : k + C], q0[bank, :, :])
                        if n + OFFSET // 128 < KEEP:
                            N.copy(
                                resident[n + OFFSET // 128, :, k : k + C],
                                b0[bank, :, :],
                                transpose=True,
                            )
                        else:
                            N.copy(
                                k1[panel % 2, :, part * C : (part + 1) * C],
                                b0[bank, :, :],
                                transpose=True,
                            )
                with N.Stage(2, name="qk_dot"):
                    for part in N.unroll(LC // C):
                        bank = part if LC > C else panel % 2
                        k = d + part * C
                        N.matmul(
                            q0[k // C if CACHE_Q else bank, :, :],
                            b0[bank, :, :],
                            accum=c0[:, (n % 2) * 128 : (n % 2 + 1) * 128],
                            initialize=panel == 0 if part == 0 else False,
                        )
            N.copy(
                c0[:, (n % 2) * 128 : (n % 2 + 1) * 128],
                Score[group, slot, :, OFFSET + n * 128 : OFFSET + (n + 1) * 128],
            )


@N.inline
def pv_tile(cache, geometry, position, P, PV, l1, p1, l0, S, PAGE, D, C, MAX_SEQ, KEEP):
    _ori_kv, _ori_block_table, KV = cache
    cached_p, v1, resident = l1
    a0, b0, c0 = l0
    group, cache_slot, slot = position
    NC = 2 if D % (2 * C) == 0 else 1
    PC = N.min(D, S)
    # Reuse Q storage after its last MTE1 reader; each cached P panel serves all columns.
    N.copy(P[group, slot, :, 0:PC], cached_p[:, 0:PC])
    for column in N.serial(D // (NC * C)):
        d = column * (NC * C)
        for chunk in N.Pipelined(S // 128, issue_order=("pv_load", "pv_dot", "pv_operand")):
            j = chunk * 128
            bank = chunk % 2
            with N.Stage(0, name="pv_load"):
                if chunk >= PC // 128:
                    N.copy(P[group, slot, :, j : j + 128], p1[bank, :, :])
                if KEEP > 0:
                    if chunk >= KEEP:
                        first = N.max(0, j - 128)
                        N.copy(
                            KV[group, cache_slot, first : first + 128, d : d + NC * C],
                            v1[bank, :, :],
                        )
                else:
                    load_cache(
                        cache,
                        geometry,
                        position,
                        v1[bank, :, :],
                        first=N.max(0, j - 128),
                        feature=d,
                        IS_WINDOW=chunk == 0,
                        PAGE=PAGE,
                        C=NC * C,
                        MAX_SEQ=MAX_SEQ,
                    )
            with N.Stage(1, name="pv_operand"):
                if chunk >= PC // 128:
                    N.copy(p1[bank, :, :], a0[bank, :, :])
                else:
                    N.copy(cached_p[:, j : j + 128], a0[bank, :, :])
                for c in N.unroll(NC):
                    b = bank if NC == 1 else c
                    if chunk < KEEP:
                        N.copy(resident[chunk, :, d + c * C : d + (c + 1) * C], b0[b, :, :])
                    else:
                        N.copy(v1[bank, :, c * C : (c + 1) * C], b0[b, :, :])
            with N.Stage(2, name="pv_dot"):
                for c in N.unroll(NC):
                    b = bank if NC == 1 else c
                    result = column % 2 if NC == 1 else c
                    N.matmul(
                        a0[bank, :, :],
                        b0[b, :, :],
                        accum=c0[:, result * C : (result + 1) * C],
                        initialize=chunk == 0,
                    )
        if NC == 2:
            N.copy(c0[:, :], PV[group, slot, :, d : d + 2 * C])
        else:
            N.copy(c0[:, (column % 2) * C : (column % 2 + 1) * C], PV[group, slot, :, d : d + C])


@N.prim_func(disjoint_external=True)
def sparse_mla_decode(
    q: N.Tensor((T, H, D), DTYPE),
    ori_kv: N.Tensor((ORI_BLOCKS, PAGE, 1, D), DTYPE),
    cmp_kv: N.Tensor((CMP_BLOCKS, PAGE // RATIO, 1, D), DTYPE),
    cmp_sparse_indices: N.Tensor((T, 1, TOPK), N.i32),
    ori_block_table: N.Tensor((B, MAX_SEQ // PAGE), N.i32),
    cmp_block_table: N.Tensor((B, MAX_SEQ // PAGE), N.i32),
    cu_seqlens_q: N.Tensor((B + 1,), N.i32),
    seqused_ori_kv: N.Tensor((B,), N.i32),
    seqused_cmp_kv: N.Tensor((B,), N.i32),
    cmp_residual_kv: N.Tensor((B,), N.i32),
    ori_topk_length: N.Tensor((T, 1), N.i32),
    sinks: N.Tensor((H,), N.f32),
    attention_out: N.Tensor((T, H, D), DTYPE, access="write", write_disjoint=True),
    KV: N.Tensor((N.min(GROUPS, T * (H // BM)), 3, S - WINDOW, D), DTYPE, access="write"),
    Bias: N.Tensor((N.min(GROUPS, T * (H // BM)), 3, S - WINDOW), N.f32, access="write"),
    Score: N.Tensor((N.min(GROUPS, T * (H // BM)), 2, BM, S), N.f32, access="write"),
    P: N.Tensor((N.min(GROUPS, T * (H // BM)), 2, BM, S), DTYPE, access="write"),
    PV: N.Tensor((N.min(GROUPS, T * (H // BM)), 2, BM, D), N.f32, access="write"),
    B: N.constexpr = 4,
    T: N.constexpr = 16,
    H: N.constexpr = 64,
    D: N.constexpr = 512,
    PAGE: N.constexpr = 128,
    RATIO: N.constexpr = 2,
    MAX_SEQ: N.constexpr = 4096,
    ORI_BLOCKS: N.constexpr = 128,
    CMP_BLOCKS: N.constexpr = 128,
    WINDOW: N.constexpr = 128,
    TOPK: N.constexpr = 512,
    ROWS: N.constexpr = 32,
    S: N.constexpr = 640,
    BM: N.constexpr = 64,
    PARTS: N.constexpr = 1,
    GROUPS: N.constexpr = 20,
    VR: N.constexpr = 16,
    BK: N.constexpr = 512,
    C: N.constexpr = 128,
    DTYPE: N.constexpr = N.bf16,
    SCALE: N.constexpr = 0.04419417382415922,
):
    with N.Kernel(N.min(GROUPS, T * (H // BM)), kind="mixed") as group:
        assert B > 0 and B < 2147483648 and T > 0 and T < 2147483648, "nonempty INT32 TND prefix"
        assert H % BM == 0 and (BM == 32 or BM == 64), "complete head tiles"
        assert WINDOW == 128 and PAGE == 128 and BK == 512 and C == 128, "cache/tile geometry"
        assert D % C == 0 and S == WINDOW + N.ceildiv(TOPK, BK) * BK, "complete selected capacities"
        assert PARTS == 1 and (ROWS == 16 or ROWS == 32) and GROUPS > 0, "sparse query policy"
        assert (VR == 4 or VR == 8 or VR == 16) and (BM // 2) % VR == 0, "softmax row groups"
        assert (RATIO == 1 or RATIO == 2) and MAX_SEQ % PAGE == 0, "complete C1/C2 pages"
        WIDTH = S - WINDOW
        assert N.ceildiv(S, 512) * 512 <= 4096, "resident softmax capacity"
        assert SCALE > 0.0, "positive attention scale"
        UR = 8
        NC = 2 if D % (2 * C) == 0 else 1
        BASE_L1 = 2 * (BM * D + 2 * 128 * NC * C + 2 * BM * 128)
        KEEP = N.min(S // 128, N.min(2, (512 * 1024 - BASE_L1) // (128 * D * 2)))
        q1 = N.alloc_l1((BM, D), DTYPE)
        kv1 = N.alloc_l1((2, 128, NC * C), DTYPE)
        if KEEP > 0:
            resident = N.alloc_l1((KEEP, 128, D), DTYPE, name="resident_compressed")
        else:
            resident = kv1
        p1 = N.alloc_l1((2, BM, 128), DTYPE)
        CACHE_Q = BM * D * 2 <= 64 * 1024
        Q_SLOTS = N.max(2, D // C) if CACHE_Q else 2
        a0 = N.alloc_l0a((Q_SLOTS, BM, C), DTYPE)
        b0 = N.alloc_l0b((2, 128, C), DTYPE)
        c0 = N.alloc_l0c((BM, 2 * C), N.f32)
        gather_zeros = N.alloc_ub((2, 8), N.f32)
        GATHER_SLOTS = 2
        gather_rows = N.alloc_ub((GATHER_SLOTS, ROWS, D), DTYPE)
        gather_masks = N.alloc_ub((GATHER_SLOTS, ROWS), N.f32)
        indices = N.alloc_ub((TOPK,), N.i32, name="cached_indices")
        pages = N.alloc_ub((MAX_SEQ // PAGE,), N.i32, name="cached_pages")
        score = N.alloc_ub((VR, WIDTH), N.f32)
        window_score = N.alloc_ub((VR, WINDOW), N.f32)
        bias = N.alloc_ub((WIDTH,), N.f32)
        window_bias = N.alloc_ub((WINDOW,), N.f32)
        probabilities = N.alloc_ub((VR, WIDTH), DTYPE)
        window_probabilities = N.alloc_ub((VR, WINDOW), DTYPE)
        window_stat = N.alloc_ub((VR,), N.f32)
        sink = N.alloc_ub((VR,), N.f32)
        maximum = N.alloc_ub((VR,), N.f32)
        row_sum = N.alloc_ub((VR,), N.f32)
        final_l = N.alloc_ub((2, BM // 2), N.f32)
        partial = N.alloc_ub((VR * (WIDTH // 8 + (WIDTH // 64 if WIDTH > 512 else 0)),), N.f32)
        broadcast = N.alloc_ub((VR, 8), N.f32)
        if WIDTH > 1024:
            softmax_work = N.alloc_ub((VR, 512), N.f32)
        pv = N.alloc_ub((UR, D), N.f32)
        output = N.alloc_ub((UR, D), DTYPE)
        inverse = N.alloc_ub((UR,), N.f32)
        ones = N.alloc_ub((UR,), N.f32)
        update_broadcast = N.alloc_ub((UR, 8), N.f32)
        with N.Cube():
            for bank in N.unroll(2):
                N.fill(kv1[bank, :, :], 0.0)
            if KEEP > 0:
                for bank in N.unroll(KEEP):
                    N.fill(resident[bank, :, :], 0.0)
        with N.Vector():
            N.fill(gather_zeros[0, :], 0.0)
            N.fill(gather_zeros[1, :], NEG_INF)
            for bank in N.unroll(GATHER_SLOTS):
                N.fill(gather_rows[bank, :, :], 0.0)
            N.fill(ones, 1.0)
        # Inline bundles pass existing GM and L1/L0 handles; storage stays explicit above.
        cache = (ori_kv, ori_block_table, KV)
        l1 = (q1, kv1, resident)
        l0 = (a0, b0, c0)
        WORKERS = N.min(GROUPS, T * (H // BM))
        for query in N.Dataflow(
            N.ceildiv(T * (H // BM) - group, WORKERS),
            issue_order={"cube": ("pv", "qk"), "vector": ("gather", "softmax", "finish")},
        ):
            owner = group + query * WORKERS
            token = owner // (H // BM)
            head = (owner % (H // BM)) * BM
            slot = query % 2
            cache_slot = query % 3
            position = (group, cache_slot, slot)
            with N.Vector() as sid, N.Stage(0, name="gather"):
                batch, qlen, offset, length, visible = query_metadata(
                    cu_seqlens_q, seqused_ori_kv, ori_topk_length, token, B, WINDOW
                )
                available = N.load_index(seqused_cmp_kv[batch : batch + 1])
                residual = N.load_index(cmp_residual_kv[batch : batch + 1])
                cutoff = N.max(0, available * RATIO + residual - qlen + offset + 1) // RATIO
                N.copy(cmp_sparse_indices[token, 0, :], indices)
                N.copy(cmp_block_table[batch, :], pages)
                gather_compressed(
                    Cmp=cmp_kv,
                    CmpTable=pages,
                    Indices=indices,
                    available=available,
                    cutoff=cutoff,
                    KV=KV,
                    Bias=Bias,
                    zeros=gather_zeros,
                    rows=gather_rows,
                    masks=gather_masks,
                    group=group,
                    slot=cache_slot,
                    sid=sid,
                    BK=S - WINDOW,
                    ROWS=ROWS,
                    SLOTS=GATHER_SLOTS,
                    PAGE=PAGE,
                    RATIO=RATIO,
                    TOPK=TOPK,
                )
            with N.Cube(), N.Stage(0, name="qk"):
                batch, qlen, offset, length, visible = query_metadata(
                    cu_seqlens_q, seqused_ori_kv, ori_topk_length, token, B, WINDOW
                )
                N.copy(q[token, head : head + BM, :], q1)
                # Compute the direct window while the two AIVs gather sparse
                # rows. The final compressed Score prefix joins both gather
                # publishers before the full Score acquire and Bias read.
                qk_tiles(
                    cache,
                    (batch, length, visible),
                    position,
                    Score,
                    l1,
                    l0,
                    S=S,
                    PAGE=PAGE,
                    D=D,
                    C=C,
                    MAX_SEQ=MAX_SEQ,
                    KEEP=KEEP,
                    CACHE_Q=CACHE_Q,
                )
            with N.Vector() as sid, N.Stage(1, name="softmax"):
                batch, qlen, offset, length, visible = query_metadata(
                    cu_seqlens_q, seqused_ori_kv, ori_topk_length, token, B, WINDOW
                )
                # Acquire the complete compressed score prefix before Bias;
                # this includes both gather publishers, even for window-only inputs.
                N.copy(Score[group, slot, sid * (BM // 2) : sid * (BM // 2) + VR, WINDOW:S], score)
                N.copy(
                    Score[group, slot, sid * (BM // 2) : sid * (BM // 2) + VR, 0:WINDOW],
                    window_score,
                )
                N.fill(window_bias, 0.0)
                N.fill(window_bias, NEG_INF, mask=N.arange(WINDOW) >= visible)
                N.copy(Bias[group, cache_slot, :], bias)
                for block in N.serial(BM // 2 // VR):
                    local = block * VR
                    r = sid * (BM // 2) + local
                    if block > 0:
                        N.copy(Score[group, slot, r : r + VR, WINDOW:S], score)
                        N.copy(Score[group, slot, r : r + VR, 0:WINDOW], window_score)
                    N.copy(sinks[head + r : head + r + VR], sink)
                    N.mul(score, SCALE, out=score)
                    N.mul(window_score, SCALE, out=window_score)
                    if WIDTH <= 1024:
                        N.add(score, bias[None, :], out=score)
                    else:
                        for row in N.serial(VR):
                            N.add(score[row, :], bias, out=score[row, :])
                    N.add(window_score, window_bias[None, :], out=window_score)
                    N.reduce_max(
                        window_score,
                        axis=1,
                        out=window_stat,
                        scratch=partial[0 : VR * (WINDOW // 8)],
                        order="block8_then_packed_merge",
                    )
                    N.reduce_max(
                        score,
                        axis=1,
                        out=maximum,
                        scratch=partial,
                        order="block8_then_packed_merge",
                    )
                    N.maximum(maximum, window_stat, out=maximum)
                    N.maximum(maximum, sink, out=maximum)
                    N.sub(sink, maximum, out=sink)
                    N.exp(sink, out=sink)
                    N.sub(window_score, maximum[:, None], out=window_score, scratch=broadcast)
                    if WIDTH <= 1024:
                        N.sub(score, maximum[:, None], out=score, scratch=broadcast)
                    else:
                        for part in N.serial(WIDTH // 512):
                            for row in N.serial(VR):
                                N.copy(
                                    score[row, part * 512 : (part + 1) * 512], softmax_work[row, :]
                                )
                            N.sub(
                                softmax_work, maximum[:, None], out=softmax_work, scratch=broadcast
                            )
                            for row in N.serial(VR):
                                N.copy(
                                    softmax_work[row, :], score[row, part * 512 : (part + 1) * 512]
                                )
                    N.exp(score, out=score)
                    N.exp(window_score, out=window_score)
                    N.cast(score, out=probabilities)
                    N.cast(window_score, out=window_probabilities)
                    N.reduce_sum(
                        window_score,
                        axis=1,
                        out=window_stat,
                        scratch=partial[0 : VR * (WINDOW // 8)],
                        order="block8_then_packed_merge",
                    )
                    N.reduce_sum(
                        score,
                        axis=1,
                        out=row_sum,
                        scratch=partial,
                        order="block8_then_packed_merge",
                    )
                    N.add(row_sum, window_stat, out=row_sum)
                    N.add(row_sum, sink, out=row_sum)
                    N.copy(row_sum, final_l[slot, local : local + VR])
                    N.copy(window_probabilities, P[group, slot, r : r + VR, 0:WINDOW])
                    N.copy(probabilities, P[group, slot, r : r + VR, WINDOW:S])
            with N.Cube(), N.Stage(1, name="pv"):
                if KEEP == 0:
                    batch, qlen, offset, length, visible = query_metadata(
                        cu_seqlens_q, seqused_ori_kv, ori_topk_length, token, B, WINDOW
                    )
                else:
                    batch = 0
                    length = 0
                    visible = 0
                pv_tile(
                    cache,
                    (batch, length, visible),
                    position,
                    P,
                    PV,
                    l1,
                    p1,
                    l0,
                    S=S,
                    PAGE=PAGE,
                    D=D,
                    C=C,
                    MAX_SEQ=MAX_SEQ,
                    KEEP=KEEP,
                )
            with N.Vector() as sid, N.Stage(2, name="finish"):
                for block in N.serial(BM // 2 // UR):
                    local = block * UR
                    r = sid * (BM // 2) + local
                    N.copy(PV[group, slot, r : r + UR, :], pv)
                    N.div(ones, final_l[slot, local : local + UR], out=inverse)
                    N.mul(pv, inverse[:, None], out=pv, scratch=update_broadcast)
                    N.cast(pv, out=output)
                    N.copy(output, attention_out[token, head + r : head + r + UR, :])

"""CSA sparse MLA: direct window, compressed gather, QK/softmax/PV/update.

Follows vllm-ascend's dav-2201 CSA structure: one shared KV gather for all
heads, 512-row compressed tiles, triple GM slots and double 16-row UB streams.
Window pages go directly to L1. FP32 online state includes exactly one sink.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.sparse_attention.gather import copy_window_pages, gather_compressed

# ruff: noqa: F821 - captured constexpr annotation names.
NEG_INF = float("-inf")


@N.inline
def load_cache(
    Ori,
    OriTable,
    length,
    visible,
    KV,
    destination,
    batch,
    group,
    cache_slot,
    first,
    feature,
    IS_WINDOW,
    PAGE,
    C,
    MAX_SEQ,
    ORI_BLOCKS,
):
    if IS_WINDOW:
        copy_window_pages(
            Ori,
            OriTable,
            destination,
            batch=batch,
            length=length,
            visible=visible,
            feature=feature,
            PAGE=PAGE,
            C=C,
            MAX_SEQ=MAX_SEQ,
            ORI_BLOCKS=ORI_BLOCKS,
        )
    else:
        N.copy(KV[group, cache_slot, first : first + 128, feature : feature + C], destination)


@N.inline
def qk_tile(
    Ori,
    OriTable,
    Lengths,
    WindowLengths,
    KV,
    Score,
    q1,
    k1,
    q0,
    b0,
    c0,
    batch,
    group,
    cache_slot,
    slot,
    IS_WINDOW,
    WIDTH,
    PAGE,
    D,
    C,
    WINDOW,
    MAX_SEQ,
    ORI_BLOCKS,
):
    length = N.load_index(Lengths[batch : batch + 1], minimum=0, maximum=MAX_SEQ)
    visible = N.load_index(
        WindowLengths[batch : batch + 1], minimum=0, maximum=N.min(WINDOW, length)
    )
    for n in N.serial(WIDTH // 128):
        for chunk in N.Pipelined(D // C, issue_order=("qk_load", "qk_dot", "qk_operand")):
            d = chunk * C
            bank = chunk % 2
            with N.Stage(0, name="qk_load"):
                load_cache(
                    Ori=Ori,
                    OriTable=OriTable,
                    length=length,
                    visible=visible,
                    KV=KV,
                    destination=k1[chunk % 2, :, 0:C],
                    batch=batch,
                    group=group,
                    cache_slot=cache_slot,
                    first=n * 128,
                    feature=d,
                    IS_WINDOW=IS_WINDOW,
                    PAGE=PAGE,
                    C=C,
                    MAX_SEQ=MAX_SEQ,
                    ORI_BLOCKS=ORI_BLOCKS,
                )
            with N.Stage(1, name="qk_operand"):
                N.copy(q1[:, d : d + C], q0[bank, :, :])
                N.copy(k1[chunk % 2, :, 0:C], b0[bank, :, :], transpose=True)
            with N.Stage(2, name="qk_dot"):
                N.matmul(
                    q0[bank, :, :],
                    b0[bank, :, :],
                    accum=c0[:, (n % 2) * 128 : (n % 2 + 1) * 128],
                    initialize=chunk == 0,
                )
        N.copy(
            c0[:, (n % 2) * 128 : (n % 2 + 1) * 128], Score[group, slot, :, n * 128 : (n + 1) * 128]
        )


@N.inline
def pv_tile(
    Ori,
    OriTable,
    Lengths,
    WindowLengths,
    KV,
    P,
    PV,
    p1,
    v1,
    a0,
    b0,
    c0,
    batch,
    group,
    cache_slot,
    slot,
    IS_WINDOW,
    WIDTH,
    PAGE,
    D,
    C,
    WINDOW,
    MAX_SEQ,
    ORI_BLOCKS,
):
    length = N.load_index(Lengths[batch : batch + 1], minimum=0, maximum=MAX_SEQ)
    visible = N.load_index(
        WindowLengths[batch : batch + 1], minimum=0, maximum=N.min(WINDOW, length)
    )
    NC = 2 if D % (2 * C) == 0 else 1
    for column in N.serial(D // (NC * C)):
        d = column * (NC * C)
        for chunk in N.Pipelined(WIDTH // 128, issue_order=("pv_load", "pv_dot", "pv_operand")):
            j = chunk * 128
            bank = chunk % 2
            with N.Stage(0, name="pv_load"):
                N.copy(P[group, slot, :, j : j + 128], p1[bank, :, :])
                for c in N.unroll(NC):
                    load_cache(
                        Ori=Ori,
                        OriTable=OriTable,
                        length=length,
                        visible=visible,
                        KV=KV,
                        destination=v1[bank, :, c * C : (c + 1) * C],
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        first=j,
                        feature=d + c * C,
                        IS_WINDOW=IS_WINDOW,
                        PAGE=PAGE,
                        C=C,
                        MAX_SEQ=MAX_SEQ,
                        ORI_BLOCKS=ORI_BLOCKS,
                    )
            with N.Stage(1, name="pv_operand"):
                N.copy(p1[bank, :, :], a0[bank, :, :])
                for c in N.unroll(NC):
                    b = bank if NC == 1 else c
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


@N.inline
def update_online_softmax(
    score,
    probabilities,
    m,
    l,
    alpha,
    next_m,
    row_sum,
    partial,
    broadcast,
    local,
    slot,
    VR,
    WIDTH,
):
    N.reduce_max(
        score,
        axis=1,
        out=next_m,
        scratch=partial[0 : VR * (WIDTH // 8)],
        order="block8_then_packed_merge",
    )
    N.maximum(m[local : local + VR], next_m, out=next_m)
    N.sub(m[local : local + VR], next_m, out=alpha[slot, local : local + VR])
    N.exp(alpha[slot, local : local + VR], out=alpha[slot, local : local + VR])
    N.sub(score, next_m[:, None], out=score, scratch=broadcast)
    N.exp(score, out=score)
    N.cast(score, out=probabilities)
    N.reduce_sum(
        score,
        axis=1,
        out=row_sum,
        scratch=partial[0 : VR * (WIDTH // 8)],
        order="block8_then_packed_merge",
    )
    N.mul(l[local : local + VR], alpha[slot, local : local + VR], out=l[local : local + VR])
    N.add(l[local : local + VR], row_sum, out=l[local : local + VR])
    N.copy(next_m, m[local : local + VR])


@N.inline
def softmax_tile(
    Score,
    Bias,
    P,
    WindowLengths,
    Lengths,
    score,
    bias,
    bias_rows,
    probabilities,
    m,
    l,
    alpha,
    next_m,
    row_sum,
    partial,
    broadcast,
    batch,
    group,
    cache_slot,
    slot,
    sid,
    IS_WINDOW,
    BM,
    WIDTH,
    VR,
    MAX_SEQ,
    WINDOW,
    SCALE,
):
    N.copy(Score[group, slot, sid * (BM // 2) : sid * (BM // 2) + VR, 0:WIDTH], score)
    if IS_WINDOW:
        length = N.load_index(Lengths[batch : batch + 1], minimum=0, maximum=MAX_SEQ)
        visible = N.load_index(
            WindowLengths[batch : batch + 1], minimum=0, maximum=N.min(WINDOW, length)
        )
    else:
        N.copy(Bias[group, cache_slot, :], bias)
        N.broadcast(bias[None, :], out=bias_rows)
    for block in N.serial(BM // 2 // VR):
        local = block * VR
        r = sid * (BM // 2) + local
        if block > 0:
            N.copy(Score[group, slot, r : r + VR, 0:WIDTH], score)
        N.mul(score, SCALE, out=score)
        if IS_WINDOW:
            N.fill(score, NEG_INF, mask=N.arange(WIDTH)[None, :] >= visible)
        else:
            N.add(score, bias_rows, out=score)
        update_online_softmax(
            score=score,
            probabilities=probabilities,
            m=m,
            l=l,
            alpha=alpha,
            next_m=next_m,
            row_sum=row_sum,
            partial=partial,
            broadcast=broadcast,
            local=local,
            slot=slot,
            VR=VR,
            WIDTH=WIDTH,
        )
        N.copy(probabilities, P[group, slot, r : r + VR, 0:WIDTH])


@N.prim_func(disjoint_external=True)
def sparse_mla_decode(
    Q: N.Tensor((B, H, D), DTYPE),
    Ori: N.Tensor((ORI_BLOCKS, PAGE, D), DTYPE),
    Cmp: N.Tensor((CMP_BLOCKS, PAGE // RATIO, D), DTYPE),
    OriTable: N.Tensor((B, MAX_SEQ // PAGE), N.i32),
    CmpTable: N.Tensor((B, MAX_SEQ // PAGE), N.i32),
    Lengths: N.Tensor((B,), N.i32),
    WindowLengths: N.Tensor((B,), N.i32),
    Counts: N.Tensor((B,), N.i32),
    Indices: N.Tensor((B, TOPK), N.i32),
    Sinks: N.Tensor((H,), N.f32),
    O: N.Tensor((B, H, D), DTYPE, access="write", write_disjoint=True),
    KV: N.Tensor((B * (H // BM), 3, BK, D), DTYPE, access="write"),
    Bias: N.Tensor((B * (H // BM), 3, BK), N.f32, access="write"),
    Score: N.Tensor((B * (H // BM), 2, BM, BK), N.f32, access="write"),
    P: N.Tensor((B * (H // BM), 2, BM, BK), DTYPE, access="write"),
    PV: N.Tensor((B * (H // BM), 2, BM, D), N.f32, access="write"),
    B: N.constexpr = 4,
    H: N.constexpr = 64,
    D: N.constexpr = 512,
    PAGE: N.constexpr = 128,
    RATIO: N.constexpr = 2,
    MAX_SEQ: N.constexpr = 4096,
    ORI_BLOCKS: N.constexpr = 128,
    CMP_BLOCKS: N.constexpr = 128,
    WINDOW: N.constexpr = 128,
    TOPK: N.constexpr = 512,
    ROWS: N.constexpr = 16,
    S: N.constexpr = 640,
    BM: N.constexpr = 64,
    PARTS: N.constexpr = 1,
    GROUPS: N.constexpr = 20,
    VR: N.constexpr = 8,
    BK: N.constexpr = 512,
    C: N.constexpr = 128,
    DTYPE: N.constexpr = N.bf16,
    SCALE: N.constexpr = 0.04419417382415922,
):
    with N.Kernel(N.min(GROUPS, B * (H // BM)), kind="mixed") as group:
        assert H % BM == 0 and (BM == 32 or BM == 64), "CSA requires complete head tiles"
        assert WINDOW == 128 and PAGE == 128 and BK == 512 and C == 128, "CSA page/tile geometry"
        assert D % C == 0 and S == WINDOW + N.ceildiv(TOPK, BK) * BK, "complete CSA capacities"
        assert PARTS == 1 and ROWS == 16 and VR == 8, "CSA baseline policy"
        assert GROUPS > 0, "positive CSA worker count"
        assert (RATIO == 1 or RATIO == 2) and MAX_SEQ % PAGE == 0, "complete C1/C2 pages"
        q1 = N.alloc_l1((BM, D), DTYPE)
        # QK and PV run as consecutive Cube children. Their MTE1 reads
        # share one KV pool. At this example's 128x128 load granularity,
        # Stage 1 is its last reader, so two planes overlap Stage 0 loads
        # with MTE1 without tying L1 retirement to the Stage 2 MAD.
        PV_COLUMNS = 2 if D % (2 * C) == 0 else 1
        kv1 = N.alloc_l1((2, 128, PV_COLUMNS * C), DTYPE)
        p1 = N.alloc_l1((2, BM, 128), DTYPE)
        a0 = N.alloc_l0a((2, BM, C), DTYPE)
        b0 = N.alloc_l0b((2, 128, C), DTYPE)
        c0 = N.alloc_l0c((BM, 2 * C), N.f32)
        gather_zeros = N.alloc_ub((8, D), N.f32)
        gather_rows = N.alloc_ub((2, ROWS, D), DTYPE)
        gather_masks = N.alloc_ub((2, ROWS), N.f32)
        window_score = N.alloc_ub((VR, WINDOW), N.f32)
        window_p = N.alloc_ub((VR, WINDOW), DTYPE)
        score = N.alloc_ub((VR, BK), N.f32)
        bias = N.alloc_ub((BK,), N.f32)
        bias_rows = N.alloc_ub((VR, BK), N.f32)
        probabilities = N.alloc_ub((VR, BK), DTYPE)
        pv = N.alloc_ub((VR, D), N.f32)
        output = N.alloc_ub((VR, D), DTYPE)
        m = N.alloc_ub((BM // 2,), N.f32)
        l = N.alloc_ub((BM // 2,), N.f32)
        final_l = N.alloc_ub((2, BM // 2), N.f32)
        o = N.alloc_ub((BM // 2, D), N.f32)
        alpha = N.alloc_ub((2, BM // 2), N.f32)
        next_m = N.alloc_ub((VR,), N.f32)
        row_sum = N.alloc_ub((VR,), N.f32)
        partial = N.alloc_ub((VR * (BK // 8),), N.f32)
        broadcast = N.alloc_ub((VR, 8), N.f32)
        update_broadcast = N.alloc_ub((VR, 8), N.f32)
        ones = N.alloc_ub((VR,), N.f32)
        inverse = N.alloc_ub((VR,), N.f32)
        initial_batch = group // (H // BM)
        initial_head = (group % (H // BM)) * BM
        with N.Cube():
            N.copy(Q[initial_batch, initial_head : initial_head + BM, :], q1)
        with N.Vector() as initial_sid:
            N.fill(gather_zeros, 0.0)
            N.copy(
                Sinks[
                    initial_head + initial_sid * (BM // 2) : initial_head
                    + (initial_sid + 1) * (BM // 2)
                ],
                m,
            )
            N.fill(l, 1.0)
            N.fill(o, 0.0)
            N.fill(ones, 1.0)
        WORKERS = N.min(GROUPS, B * (H // BM))
        TILES = 1 + N.ceildiv(TOPK, BK)
        # Continue one tile pipeline across requests owned by this worker.
        # Its fill/drain occurs once, as in CSA's persistent core schedule.
        for tile in N.Dataflow(
            N.ceildiv(B * (H // BM) - group, WORKERS) * TILES,
            issue_order={"cube": ("pv", "qk"), "vector": ("gather", "softmax", "update")},
        ):
            local_tile = tile % TILES
            owner = group + (tile // TILES) * WORKERS
            batch = owner // (H // BM)
            head = (owner % (H // BM)) * BM
            slot = tile % 2
            cache_slot = tile % 3
            with N.Vector() as sid, N.Stage(0, name="gather"):
                if local_tile == 0:
                    pass
                else:
                    gather_compressed(
                        Cmp=Cmp,
                        CmpTable=CmpTable,
                        Lengths=Lengths,
                        Counts=Counts,
                        Indices=Indices,
                        KV=KV,
                        Bias=Bias,
                        zeros=gather_zeros,
                        rows=gather_rows,
                        masks=gather_masks,
                        batch=batch,
                        group=group,
                        slot=cache_slot,
                        tile=local_tile - 1,
                        sid=sid,
                        BK=BK,
                        ROWS=ROWS,
                        PAGE=PAGE,
                        RATIO=RATIO,
                        MAX_SEQ=MAX_SEQ,
                        CMP_BLOCKS=CMP_BLOCKS,
                        TOPK=TOPK,
                    )
            with N.Cube(), N.Stage(0, name="qk"):
                if local_tile == 0:
                    if tile > 0:
                        N.copy(Q[batch, head : head + BM, :], q1)
                    qk_tile(
                        Ori=Ori,
                        OriTable=OriTable,
                        Lengths=Lengths,
                        WindowLengths=WindowLengths,
                        KV=KV,
                        Score=Score,
                        q1=q1,
                        k1=kv1,
                        q0=a0,
                        b0=b0,
                        c0=c0,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        IS_WINDOW=True,
                        WIDTH=WINDOW,
                        PAGE=PAGE,
                        D=D,
                        C=C,
                        WINDOW=WINDOW,
                        MAX_SEQ=MAX_SEQ,
                        ORI_BLOCKS=ORI_BLOCKS,
                    )
                else:
                    qk_tile(
                        Ori=Ori,
                        OriTable=OriTable,
                        Lengths=Lengths,
                        WindowLengths=WindowLengths,
                        KV=KV,
                        Score=Score,
                        q1=q1,
                        k1=kv1,
                        q0=a0,
                        b0=b0,
                        c0=c0,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        IS_WINDOW=False,
                        WIDTH=BK,
                        PAGE=PAGE,
                        D=D,
                        C=C,
                        WINDOW=WINDOW,
                        MAX_SEQ=MAX_SEQ,
                        ORI_BLOCKS=ORI_BLOCKS,
                    )
            with N.Vector() as sid, N.Stage(1, name="softmax"):
                if local_tile == 0:
                    if tile > 0:
                        N.copy(Sinks[head + sid * (BM // 2) : head + (sid + 1) * (BM // 2)], m)
                        N.fill(l, 1.0)
                    softmax_tile(
                        Score=Score,
                        Bias=Bias,
                        P=P,
                        WindowLengths=WindowLengths,
                        Lengths=Lengths,
                        score=window_score,
                        bias=bias,
                        bias_rows=bias_rows,
                        probabilities=window_p,
                        m=m,
                        l=l,
                        alpha=alpha,
                        next_m=next_m,
                        row_sum=row_sum,
                        partial=partial,
                        broadcast=broadcast,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        sid=sid,
                        IS_WINDOW=True,
                        BM=BM,
                        WIDTH=WINDOW,
                        VR=VR,
                        MAX_SEQ=MAX_SEQ,
                        WINDOW=WINDOW,
                        SCALE=SCALE,
                    )
                else:
                    softmax_tile(
                        Score=Score,
                        Bias=Bias,
                        P=P,
                        WindowLengths=WindowLengths,
                        Lengths=Lengths,
                        score=score,
                        bias=bias,
                        bias_rows=bias_rows,
                        probabilities=probabilities,
                        m=m,
                        l=l,
                        alpha=alpha,
                        next_m=next_m,
                        row_sum=row_sum,
                        partial=partial,
                        broadcast=broadcast,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        sid=sid,
                        IS_WINDOW=False,
                        BM=BM,
                        WIDTH=BK,
                        VR=VR,
                        MAX_SEQ=MAX_SEQ,
                        WINDOW=WINDOW,
                        SCALE=SCALE,
                    )
                # Like alpha, each tile's exact FP32 denominator is a value
                # for the following update, separate from softmax's carry.
                N.copy(l, final_l[slot, :])
            with N.Cube(), N.Stage(1, name="pv"):
                if local_tile == 0:
                    pv_tile(
                        Ori=Ori,
                        OriTable=OriTable,
                        Lengths=Lengths,
                        WindowLengths=WindowLengths,
                        KV=KV,
                        P=P,
                        PV=PV,
                        p1=p1,
                        v1=kv1,
                        a0=a0,
                        b0=b0,
                        c0=c0,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        IS_WINDOW=True,
                        WIDTH=WINDOW,
                        PAGE=PAGE,
                        D=D,
                        C=C,
                        WINDOW=WINDOW,
                        MAX_SEQ=MAX_SEQ,
                        ORI_BLOCKS=ORI_BLOCKS,
                    )
                else:
                    pv_tile(
                        Ori=Ori,
                        OriTable=OriTable,
                        Lengths=Lengths,
                        WindowLengths=WindowLengths,
                        KV=KV,
                        P=P,
                        PV=PV,
                        p1=p1,
                        v1=kv1,
                        a0=a0,
                        b0=b0,
                        c0=c0,
                        batch=batch,
                        group=group,
                        cache_slot=cache_slot,
                        slot=slot,
                        IS_WINDOW=False,
                        WIDTH=BK,
                        PAGE=PAGE,
                        D=D,
                        C=C,
                        WINDOW=WINDOW,
                        MAX_SEQ=MAX_SEQ,
                        ORI_BLOCKS=ORI_BLOCKS,
                    )
            with N.Vector() as sid, N.Stage(2, name="update"):
                if local_tile == 0:  # noqa: SIM102 - keep ordered DSL control regions
                    if tile > 0:
                        # Old output state retires only at the previous
                        # request's final update, after its last softmax.
                        N.fill(o, 0.0)
                for block in N.serial(BM // 2 // VR):
                    local = block * VR
                    r = sid * (BM // 2) + local
                    N.copy(PV[group, slot, r : r + VR, :], pv)
                    N.mul(
                        o[local : local + VR, :],
                        alpha[slot, local : local + VR, None],
                        out=o[local : local + VR, :],
                        scratch=update_broadcast,
                    )
                    N.add(o[local : local + VR, :], pv, out=o[local : local + VR, :])
                    if local_tile == N.ceildiv(TOPK, BK):
                        N.div(ones, final_l[slot, local : local + VR], out=inverse)
                        N.mul(
                            o[local : local + VR, :],
                            inverse[:, None],
                            out=pv,
                            scratch=update_broadcast,
                        )
                        N.cast(pv, out=output)
                        N.copy(output, O[batch, head + r : head + r + VR, :])

"""BNSD Flash Attention with retained query/KV/child pipelines.

Defaults select dense B1/Hq8/Hkv8/D128 and BQ256/BM128/BK1024/CK128/VR8.
CAUSAL selects top-left j<=i masking; causal_policy chooses explicit host tiles.
Score, P and PV use double-buffered GM workspaces. Qualification belongs to
the actual artifacts and cases recorded in docs/operators/flash-attention.md.
"""

from __future__ import annotations

from ascendflow import language as N

# Tensor dimensions in annotations are interpreted by the DSL signature pass.
# ruff: noqa: F821

NEG_INF = float("-inf")


def causal_policy(sq, sk):
    """Host tiling policy: amortize state updates while bounding diagonal waste."""
    prefix = min(sq, sk)
    if prefix < 2048:
        return {"kv_tile": 256, "vector_rows": 16}
    if prefix < 8192:
        return {"kv_tile": 512, "vector_rows": 16}
    return {"kv_tile": 1024, "vector_rows": 8}


@N.inline
def qk_tiles(inputs, buffers, task, geometry):
    K, Score = inputs
    q1, k1, a0, b0, c0 = buffers
    batch, kv_head, group, s, kv = task
    parts, BM, BK, CK = geometry
    if parts == 2:
        # PV overwrites these slots between QK phases, but
        # Q is unchanged across every CK chunk of this child.
        for part in N.unroll(parts):
            N.copy(q1[part, :, :], a0[part, :, :])
    for chunk in N.Pipelined(
        BK // CK,
        issue_order=("load_k", "qk_emit", "operand_k"),
    ):
        j = kv * BK + chunk * CK
        with N.Stage(0, name="load_k"):
            N.copy(K[batch, kv_head, j : j + CK, :], k1[chunk % 2, :, :])
        with N.Stage(1, name="operand_k"):
            N.copy(k1[chunk % 2, :, :], b0[chunk % 2, :, :], transpose=True)
        with N.Stage(2, name="qk_emit"):
            for part in N.unroll(parts):
                plane = (chunk * parts + part) % 2
                N.matmul(
                    a0[part, :, :],
                    b0[chunk % 2, :, :],
                    accum=c0[plane, :, :],
                    initialize=True,
                )
                # Publish each native result at its producer;
                # its FIX read retires independently of peers.
                N.copy(
                    c0[plane, :, :],
                    Score[
                        group,
                        s,
                        part * BM : (part + 1) * BM,
                        chunk * CK : (chunk + 1) * CK,
                    ],
                )


@N.inline
def online_softmax_rows(inputs, buffers, state, scratch, task, geometry):
    Score, P = inputs
    score_in, score_work, p_out = buffers
    m, l, alpha = state
    partial, next_m, row_sum, soft_broadcast = scratch
    group, s, kv, query_row, lane = task
    BQ, BM, BK, VR, SCALE, CAUSAL = geometry
    for row in N.Pipelined(BQ // 2 // VR):
        r = row * VR
        g = lane * (BQ // 2) + r
        with N.Stage(0, name="import_score"):
            N.copy(Score[group, s, g : g + VR, :], score_in[row % 2, :, :])
        with N.Stage(1, name="online_softmax"):
            if BQ == BM:
                work = score_work
            else:
                # Reuse the input slot through its last Vector read;
                # the other slot remains available to MTE2.
                work = score_in[row % 2, :, :]
            N.mul(score_in[row % 2, :, :], SCALE, out=work)
            if CAUSAL:
                # Ordinary logical lane coordinates, not a
                # triangular primitive: lane (r, c) is masked
                # when its global key column is after the
                # query row. Work is already fully initialized.
                rows = N.arange(VR)[:, None]
                cols = N.arange(BK)[None, :]
                N.fill(
                    work,
                    NEG_INF,
                    mask=(kv * BK + cols) > (query_row + g + rows),
                )
            N.reduce_max(
                work,
                axis=1,
                out=next_m,
                scratch=partial,
                order="block8_then_packed_merge",
            )
            N.maximum(m[r : r + VR], next_m, out=next_m)
            N.sub(m[r : r + VR], next_m, out=alpha[s, r : r + VR])
            N.sub(work, next_m[:, None], out=work, scratch=soft_broadcast)
            N.exp(alpha[s, r : r + VR], out=alpha[s, r : r + VR])
            N.exp(work, out=work)
            # P has separate storage. Publish its conversion
            # before the denominator reduction so work's final
            # reader is the first reduction, not this cast.
            # The denominator still consumes unrounded f32 P.
            N.cast(work, out=p_out[row % 2, :, :])
            N.reduce_sum(
                work,
                axis=1,
                out=row_sum,
                scratch=partial,
                order="block8_then_packed_merge",
            )
            N.mul(l[r : r + VR], alpha[s, r : r + VR], out=l[r : r + VR])
            N.add(l[r : r + VR], row_sum, out=l[r : r + VR])
            N.copy(next_m, m[r : r + VR])
        with N.Stage(2, name="export_p"):
            N.copy(p_out[row % 2, :, :], P[group, s, g : g + VR, :])


@N.inline
def pv_tiles(inputs, buffers, task, geometry):
    V, P, PV = inputs
    v1, p1, a0, b0, c0 = buffers
    batch, kv_head, group, s, kv, query_row = task
    parts, BQ, BM, BK, CK, CAUSAL = geometry
    if CAUSAL:
        # P is explicitly zero outside the causal prefix.
        # Keep both operand slots populated, including the
        # all-masked padding tile at startup.
        pv_chunks = N.min(BK // CK, N.max(2, N.ceildiv(N.max(0, query_row + BQ - kv * BK), CK)))
    else:
        pv_chunks = BK // CK
    if parts == 1:
        pv_order = ("load_pv", "operand_v", "pv_0", "operand_p0")
    else:
        pv_order = (
            "load_pv",
            "operand_v",
            "pv_0",
            "operand_p0",
            "pv_1",
            "operand_p1",
        )
    for chunk in N.Pipelined(
        pv_chunks,
        issue_order=pv_order,
    ):
        j = kv * BK + chunk * CK
        with N.Stage(0, name="load_pv"):
            for part in N.unroll(parts):
                N.copy(
                    P[
                        group,
                        s,
                        part * BM : (part + 1) * BM,
                        chunk * CK : (chunk + 1) * CK,
                    ],
                    p1[chunk % 2, part, :, :],
                )
            N.copy(V[batch, kv_head, j : j + CK, :], v1[chunk % 2, :, :])
        with N.Stage(1, name="operand_v"):
            N.copy(v1[chunk % 2, :, :], b0[chunk % 2, :, :])
        with N.Stage(1, name="operand_p0"):
            N.copy(p1[chunk % 2, 0, :, :], a0[1 if parts == 1 else 0, :, :])
        with N.Stage(2, name="pv_0"):
            N.matmul(
                a0[1 if parts == 1 else 0, :, :],
                b0[chunk % 2, :, :],
                accum=c0[0, :, :],
                initialize=(chunk == 0),
            )
        if parts == 2:
            with N.Stage(1, name="operand_p1"):
                N.copy(p1[chunk % 2, 1, :, :], a0[1, :, :])
            with N.Stage(2, name="pv_1"):
                N.matmul(
                    a0[1, :, :],
                    b0[chunk % 2, :, :],
                    accum=c0[1, :, :],
                    initialize=(chunk == 0),
                )
    for part in N.unroll(parts):
        N.copy(c0[part, :, :], PV[group, s, part * BM : (part + 1) * BM, :])


@N.inline
def update_output_rows(inputs, buffers, state, scratch, task, geometry):
    PV, O = inputs
    pv_in, o_out = buffers
    o, l, alpha = state
    out_broadcast, ones, inverse = scratch
    batch, head, group, s, kv, query_row, lane, active_kv_tiles = task
    BQ, VR, CAUSAL = geometry
    N.mul(o, alpha[s, :, None], out=o, scratch=out_broadcast)
    for row in N.Pipelined(BQ // 2 // VR):
        r = row * VR
        g = lane * (BQ // 2) + r
        with N.Stage(0, name="import_pv"):
            N.copy(PV[group, s, g : g + VR, :], pv_in[row % 2, :, :])
        with N.Stage(1, name="accumulate_o"):
            N.add(o[r : r + VR, :], pv_in[row % 2, :, :], out=o[r : r + VR, :])
            if kv == active_kv_tiles - 1:
                N.div(ones, l[r : r + VR], out=inverse)
                N.mul(
                    o[r : r + VR, :],
                    inverse[:, None],
                    out=pv_in[row % 2, :, :],
                    scratch=out_broadcast[:VR, :],
                )
                if CAUSAL:
                    output_slot = 0
                else:
                    output_slot = row % 2
                N.cast(pv_in[row % 2, :, :], out=o_out[output_slot, :, :])
                N.copy(
                    o_out[output_slot, :, :],
                    O[batch, head, query_row + g : query_row + g + VR, :],
                )


@N.prim_func(disjoint_external=True)
def attention_target(
    Q: N.Tensor((B, HQ, Sq, D), DTYPE),
    K: N.Tensor((B, HKV, Sk, D), DTYPE),
    V: N.Tensor((B, HKV, Sk, D), DTYPE),
    O: N.Tensor((B, HQ, Sq, D), DTYPE, access="write", write_disjoint=True),
    Score: N.Tensor((GROUPS, 2, BQ, BK), N.f32, access="write"),
    P: N.Tensor((GROUPS, 2, BQ, BK), DTYPE, access="write"),
    PV: N.Tensor((GROUPS, 2, BQ, D), N.f32, access="write"),
    Sq: N.int64(minimum=1, multiple_of=BQ),
    Sk: N.int64(minimum=2 * BK if CAUSAL else 1, multiple_of=1024),
    B: N.constexpr = 1,
    HQ: N.constexpr = 8,
    HKV: N.constexpr = 8,
    D: N.constexpr = 128,
    BQ: N.constexpr = 256,
    BM: N.constexpr = 128,
    BK: N.constexpr = 1024,
    CK: N.constexpr = 128,
    VR: N.constexpr = 8,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
    CAUSAL: N.constexpr = False,
):
    with N.Kernel(GROUPS, kind="mixed") as group:
        assert B > 0, "batch count must be positive"
        assert HQ > 0, "query head count must be positive"
        assert HKV > 0, "KV head count must be positive"
        assert HQ % HKV == 0, "query heads must be divisible by KV heads"
        assert VR > 0, "Vector row tile must be positive"
        assert BQ % (2 * VR) == 0, "query tile must cover complete rows on both AIVs"
        assert BM > 0, "native query row tile must be positive"
        assert BQ % BM == 0, "query tile must contain complete native row tiles"
        assert BQ <= 2 * BM, "query tile must fit the two shared accumulator planes"
        assert CK > 0, "Cube reduction tile must be positive"
        assert BK % CK == 0, "KV tile must contain complete Cube reduction tiles"
        assert CK == D, "shared QK/PV accumulator planes require CK equal to D"
        # Declarations dominate all actor regions that use their storage.
        parts = BQ // BM
        q1 = N.alloc_l1((parts, BM, D), DTYPE)
        k1 = N.alloc_l1((2, CK, D), DTYPE)
        p1 = N.alloc_l1((2, parts, BM, CK), DTYPE)
        v1 = N.alloc_l1((2, CK, D), DTYPE)
        # One row part: Q/P have separate slots. Two parts: QK/PV share
        # both slots by phase. Q stays resident throughout each QK child.
        a0 = N.alloc_l0a((2, BM, D), DTYPE)
        # QK and PV share the physical L0B bank. Two slots let MTE1
        # prepare the next operand while MAD still reads the other slot.
        b0 = N.alloc_l0b((2, D, CK), DTYPE)
        # Shared QK/PV L0C: two native planes, not three allocations.
        c0 = N.alloc_l0c((2, BM, CK), N.f32)
        score_in = N.alloc_ub((2, VR, BK), N.f32)
        if BQ == BM:
            score_work = N.alloc_ub((VR, BK), N.f32)
        p_out = N.alloc_ub((2, VR, BK), DTYPE)
        pv_in = N.alloc_ub((2, VR, D), N.f32)
        # Causal output uses one slot, leaving room for wider softmax rows
        # within the same private UB. Release follows its actual GM store.
        o_out = N.alloc_ub((1 if CAUSAL else 2, VR, D), DTYPE)
        m = N.alloc_ub((BQ // 2,), N.f32)
        l = N.alloc_ub((BQ // 2,), N.f32)
        o = N.alloc_ub((BQ // 2, D), N.f32)
        alpha = N.alloc_ub((2, BQ // 2), N.f32)
        # Independent block8 planes; packed merge reads the last one.
        partial = N.alloc_ub((VR * (BK // 8 + (BK // 64 if BK > 512 else 0)),), N.f32)
        next_m = N.alloc_ub((VR,), N.f32)
        row_sum = N.alloc_ub((VR,), N.f32)
        ones = N.alloc_ub((VR,), N.f32)
        inverse = N.alloc_ub((VR,), N.f32)
        # Physical BRCB scratch is explicit; no full-size broadcast tile.
        soft_broadcast = N.alloc_ub((VR, 8), N.f32)
        out_broadcast = N.alloc_ub((BQ // 2, 8), N.f32)
        with N.Vector():
            N.fill(ones, 1.0)
        query_tiles = Sq // BQ
        kv_tiles = Sk // BK
        local_queries = N.ceildiv(N.max(0, B * HQ * query_tiles - group), GROUPS)
        # Sq/Sk contracts are checked before launch; head/tile relations are checked above.
        # Empty owners do no work. No implicit padding or global Q/K/V packing.
        for query in N.serial(local_queries):
            global_query = group + query * GROUPS
            if CAUSAL:
                # Center the rotation on the rising key prefix. Queries
                # beyond Sk have a flat cost, rather than a longer triangle.
                waves = N.ceildiv(B * HQ * query_tiles, GROUPS)
                dense_waves = N.ceildiv(B * HQ * N.ceildiv(N.max(0, Sq - Sk), BQ), GROUPS)
                global_query = (
                    global_query
                    + N.max(0, B * HQ * query_tiles - ((waves + dense_waves + 1) // 2) * GROUPS)
                ) % (B * HQ * query_tiles)
                # Consecutive groups serve different heads of the same Q
                # block, spreading the triangular lengths across the grid.
                query_row = ((global_query // HQ) % query_tiles) * BQ
                head = global_query % HQ
            else:
                query_row = (global_query % query_tiles) * BQ
                head = (global_query // query_tiles) % HQ
            batch = global_query // (HQ * query_tiles)
            kv_head = head // (HQ // HKV)
            with N.Cube():
                for part in N.unroll(parts):
                    N.copy(
                        Q[
                            batch,
                            head,
                            query_row + part * BM : query_row + (part + 1) * BM,
                            :,
                        ],
                        q1[part, :, :],
                    )
                if parts == 1:
                    N.copy(q1[0, :, :], a0[0, :, :])
            with N.Vector():
                N.fill(m, NEG_INF)
                N.fill(l, 0.0)
                N.fill(o, 0.0)
            if CAUSAL:
                # Both physical handoff slots are visited in every parent.
                # The extra startup tile, when present, is fully masked.
                active_kv_tiles = N.min(kv_tiles, N.max(2, N.ceildiv(query_row + BQ, BK)))
                kv_order = {"cube": ("qk", "pv"), "vector": ("softmax", "update")}
            else:
                active_kv_tiles = kv_tiles
                kv_order = {"cube": ("qk", "pv"), "vector": ("softmax", "update")}
            for kv in N.Dataflow(
                active_kv_tiles,
                issue_order=kv_order,
            ):
                task = query * kv_tiles + kv
                if CAUSAL:
                    s = kv % 2
                else:
                    s = task % 2
                with N.Cube(), N.Stage(0, name="qk"):
                    qk_tiles(
                        (K, Score),
                        (q1, k1, a0, b0, c0),
                        (batch, kv_head, group, s, kv),
                        (parts, BM, BK, CK),
                    )
                with N.Vector() as lane, N.Stage(0, name="softmax"):
                    online_softmax_rows(
                        (Score, P),
                        (score_in, score_work if BQ == BM else None, p_out),
                        (m, l, alpha),
                        (partial, next_m, row_sum, soft_broadcast),
                        (group, s, kv, query_row, lane),
                        (BQ, BM, BK, VR, SCALE, CAUSAL),
                    )
                with N.Cube(), N.Stage(1, name="pv"):
                    pv_tiles(
                        (V, P, PV),
                        (v1, p1, a0, b0, c0),
                        (batch, kv_head, group, s, kv, query_row),
                        (parts, BQ, BM, BK, CK, CAUSAL),
                    )
                with N.Vector() as lane, N.Stage(1, name="update"):
                    # All rows rescale independently before adding this PV.
                    # Broadcast once for the complete state; retain separate
                    # FP32 multiply/add and the later normalization boundary.
                    update_output_rows(
                        (PV, O),
                        (pv_in, o_out),
                        (o, l, alpha),
                        (out_broadcast, ones, inverse),
                        (batch, head, group, s, kv, query_row, lane, active_kv_tiles),
                        (BQ, VR, CAUSAL),
                    )

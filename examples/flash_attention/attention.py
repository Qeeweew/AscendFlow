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
    kv, Sk = task
    parts, BM, BK, CK = geometry
    if parts == 2:
        # PV overwrites these slots between QK phases, but
        # Q is unchanged across every CK chunk of this child.
        for part in N.unroll(parts):
            N.copy(q1.slot(part), a0.slot(part))
    for chunk in N.Pipelined(
        BK // CK,
        issue_order=("load_k", "qk_emit", "operand_k"),
    ):
        j = kv * BK + chunk * CK
        with N.Stage(0, name="load_k"):
            keys = N.copy(
                K[j : j + CK, :],
                k1.at(chunk),
                valid_rows=N.min(CK, N.max(0, Sk - j)),
                pad_value=0,
            )
        with N.Stage(1, name="operand_k"):
            operand = N.copy(keys, b0.at(chunk), transpose=True)
        with N.Stage(2, name="qk_emit"):
            for part in N.unroll(parts):
                result = N.matmul(
                    a0.slot(part),
                    operand,
                    accum=c0.at(chunk * parts + part),
                    initialize=True,
                )
                # Publish each native result at its producer;
                # its FIX read retires independently of peers.
                N.copy(
                    result,
                    Score[
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
    kv, query_row, lane, Sk = task
    BQ, BM, BK, VR, SCALE, CAUSAL = geometry
    for row in N.Pipelined(
        BQ // 2 // VR, issue_order=("import_score", "online_softmax", "export_p")
    ):
        r = row * VR
        g = lane * (BQ // 2) + r
        with N.Stage(0, name="import_score"):
            scores = N.copy(Score[g : g + VR, :], score_in.at(row))
        with N.Stage(1, name="online_softmax"):
            if BQ == BM:
                work = score_work
            else:
                # Reuse the input slot through its last Vector read;
                # the other slot remains available to MTE2.
                work = score_in.at(row)
            N.mul(scores, SCALE, out=work)
            if kv * BK + BK > Sk:
                columns = N.arange(BK)[None, :]
                N.fill(work, NEG_INF, mask=(kv * BK + columns) >= Sk)
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
            N.sub(m[r : r + VR], next_m, out=alpha[r : r + VR])
            N.sub(work, next_m[:, None], out=work, scratch=soft_broadcast)
            N.exp(alpha[r : r + VR], out=alpha[r : r + VR])
            N.exp(work, out=work)
            # P has separate storage. Publish its conversion
            # before the denominator reduction so work's final
            # reader is the first reduction, not this cast.
            # The denominator still consumes unrounded f32 P.
            probabilities = N.cast(work, out=p_out.at(row))
            N.reduce_sum(
                work,
                axis=1,
                out=row_sum,
                scratch=partial,
                order="block8_then_packed_merge",
            )
            N.mul(l[r : r + VR], alpha[r : r + VR], out=l[r : r + VR])
            N.add(l[r : r + VR], row_sum, out=l[r : r + VR])
            N.copy(next_m, m[r : r + VR])
        with N.Stage(2, name="export_p"):
            N.copy(probabilities, P[g : g + VR, :])


@N.inline
def pv_tiles(inputs, buffers, task, geometry):
    V, P, PV = inputs
    v1, p1, a0, b0, c0 = buffers
    kv, query_row, Sk = task
    parts, BQ, BM, BK, CK, CAUSAL = geometry
    if CAUSAL:
        # P is explicitly zero outside the causal prefix.
        # Only active KV blocks reach this child. Its actual prefix,
        # including a single CK chunk, determines operand-slot visits.
        pv_chunks = N.min(BK // CK, N.ceildiv(N.max(0, query_row + BQ - kv * BK), CK))
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
                        part * BM : (part + 1) * BM,
                        chunk * CK : (chunk + 1) * CK,
                    ],
                    p1.at(chunk)[part, :, :],
                )
            values = N.copy(
                V[j : j + CK, :],
                v1.at(chunk),
                valid_rows=N.min(CK, N.max(0, Sk - j)),
                pad_value=0,
            )
        with N.Stage(1, name="operand_v"):
            operand_v = N.copy(values, b0.at(chunk))
        with N.Stage(1, name="operand_p0"):
            operand_p0 = N.copy(p1.at(chunk)[0, :, :], a0.slot(1 if parts == 1 else 0))
        with N.Stage(2, name="pv_0"):
            N.matmul(
                operand_p0,
                operand_v,
                accum=c0.slot(0),
                initialize=(chunk == 0),
            )
        if parts == 2:
            with N.Stage(1, name="operand_p1"):
                operand_p1 = N.copy(p1.at(chunk)[1, :, :], a0.slot(1))
            with N.Stage(2, name="pv_1"):
                N.matmul(
                    operand_p1,
                    operand_v,
                    accum=c0.slot(1),
                    initialize=(chunk == 0),
                )
    for part in N.unroll(parts):
        N.copy(c0.slot(part), PV[part * BM : (part + 1) * BM, :])


@N.inline
def update_output_rows(inputs, buffers, state, scratch, task, geometry):
    PV, O = inputs
    pv_in, o_out = buffers
    o, l, alpha = state
    out_broadcast, ones, inverse = scratch
    kv, query_row, lane, active_kv_tiles, Sq = task
    BQ, VR = geometry
    N.mul(o, alpha[:, None], out=o, scratch=out_broadcast)
    for row in N.Pipelined(BQ // 2 // VR, issue_order=("import_pv", "accumulate_o")):
        r = row * VR
        g = lane * (BQ // 2) + r
        with N.Stage(0, name="import_pv"):
            products = N.copy(PV[g : g + VR, :], pv_in.at(row))
        with N.Stage(1, name="accumulate_o"):
            N.add(o[r : r + VR, :], products, out=o[r : r + VR, :])
            if kv == active_kv_tiles - 1:
                reciprocal = N.div(ones, l[r : r + VR], out=inverse)
                normalized = N.mul(
                    o[r : r + VR, :],
                    reciprocal[:, None],
                    out=pv_in.at(row),
                    scratch=out_broadcast[:VR, :],
                )
                converted = N.cast(normalized, out=o_out.at(row))
                N.copy(
                    converted,
                    O[query_row + g : query_row + g + VR, :],
                    valid_rows=N.min(VR, N.max(0, Sq - query_row - g)),
                )


@N.prim_func(disjoint_external=True)
def attention_target(
    Q: N.Tensor((B, HQ, Sq, D), DTYPE),
    K: N.Tensor((B, HKV, Sk, D), DTYPE),
    V: N.Tensor((B, HKV, Sk, D), DTYPE),
    O: N.Tensor((B, HQ, Sq, D), DTYPE, access="write", write_disjoint=True),
    Sq: N.i64(minimum=1),
    Sk: N.i64(minimum=1),
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
        Score = N.alloc_workspace((GROUPS, 2, BQ, BK), N.f32, name="Score")
        P = N.alloc_workspace((GROUPS, 2, BQ, BK), DTYPE, name="P")
        PV = N.alloc_workspace((GROUPS, 2, BQ, D), N.f32, name="PV")
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
        q1 = N.alloc_l1((BM, D), DTYPE, slots=parts)
        k1 = N.alloc_l1((CK, D), DTYPE, slots=2)
        p1 = N.alloc_l1((parts, BM, CK), DTYPE, slots=2)
        v1 = N.alloc_l1((CK, D), DTYPE, slots=2)
        # One row part: Q/P have separate slots. Two parts: QK/PV share
        # both slots by phase. Q stays resident throughout each QK child.
        a0 = N.alloc_l0a((BM, D), DTYPE, slots=2)
        # QK and PV share the physical L0B bank. Two slots let MTE1
        # prepare the next operand while MAD still reads the other slot.
        b0 = N.alloc_l0b((D, CK), DTYPE, slots=2)
        # Shared QK/PV L0C: two native planes, not three allocations.
        c0 = N.alloc_l0c((BM, CK), N.f32, slots=2)
        score_in = N.alloc_ub((VR, BK), N.f32, slots=2)
        if BQ == BM:
            score_work = N.alloc_ub((VR, BK), N.f32)
        p_out = N.alloc_ub((VR, BK), DTYPE, slots=2)
        pv_in = N.alloc_ub((VR, D), N.f32, slots=2)
        # Causal output uses one slot, leaving room for wider softmax rows
        # within the same private UB. Release follows its actual GM store.
        o_out = N.alloc_ub((VR, D), DTYPE, slots=1 if CAUSAL else 2)
        m = N.alloc_ub((BQ // 2,), N.f32)
        l = N.alloc_ub((BQ // 2,), N.f32)
        o = N.alloc_ub((BQ // 2, D), N.f32)
        alpha = N.alloc_ub((BQ // 2,), N.f32, slots=2)
        scores = N.slotted_view(Score[group, :, :, :])
        probabilities = N.slotted_view(P[group, :, :, :])
        products = N.slotted_view(PV[group, :, :, :])
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
        query_tiles = N.ceildiv(Sq, BQ)
        kv_tiles = N.ceildiv(Sk, BK)
        local_queries = N.ceildiv(N.max(0, B * HQ * query_tiles - group), GROUPS)
        # Runtime lengths need no tile alignment. Empty owners do no work;
        # the final physical tiles have explicit bounded reads and zero padding.
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
                        q1.slot(part),
                        valid_rows=N.min(BM, N.max(0, Sq - query_row - part * BM)),
                        pad_value=0,
                    )
                if parts == 1:
                    N.copy(q1.slot(0), a0.slot(0))
            with N.Vector():
                N.fill(m, NEG_INF)
                N.fill(l, 0.0)
                N.fill(o, 0.0)
            kv_order = {"cube": ("qk", "pv"), "vector": ("softmax", "update")}
            if CAUSAL:
                active_kv_tiles = N.min(kv_tiles, N.ceildiv(query_row + BQ, BK))
            else:
                active_kv_tiles = kv_tiles
            for kv in N.Dataflow(
                active_kv_tiles,
                issue_order=kv_order,
            ):
                task = query * kv_tiles + kv
                if CAUSAL:
                    generation = kv
                else:
                    generation = task
                with N.Cube(), N.Stage(0, name="qk"):
                    qk_tiles(
                        (K[batch, kv_head, :, :], scores.at(generation)),
                        (q1, k1, a0, b0, c0),
                        (kv, Sk),
                        (parts, BM, BK, CK),
                    )
                with N.Vector() as lane, N.Stage(0, name="softmax"):
                    online_softmax_rows(
                        (scores.at(generation), probabilities.at(generation)),
                        (score_in, score_work if BQ == BM else None, p_out),
                        (m, l, alpha.at(generation)),
                        (partial, next_m, row_sum, soft_broadcast),
                        (kv, query_row, lane, Sk),
                        (BQ, BM, BK, VR, SCALE, CAUSAL),
                    )
                with N.Cube(), N.Stage(1, name="pv"):
                    pv_tiles(
                        (
                            V[batch, kv_head, :, :],
                            probabilities.at(generation),
                            products.at(generation),
                        ),
                        (v1, p1, a0, b0, c0),
                        (kv, query_row, Sk),
                        (parts, BQ, BM, BK, CK, CAUSAL),
                    )
                with N.Vector() as lane, N.Stage(1, name="update"):
                    # All rows rescale independently before adding this PV.
                    # Broadcast once for the complete state; retain separate
                    # FP32 multiply/add and the later normalization boundary.
                    update_output_rows(
                        (products.at(generation), O[batch, head, :, :]),
                        (pv_in, o_out),
                        (o, l, alpha.at(generation)),
                        (out_broadcast, ones, inverse),
                        (kv, query_row, lane, active_kv_tiles, Sq),
                        (BQ, VR),
                    )

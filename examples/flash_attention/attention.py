"""Dense f16 BNSD Flash Attention with retained query/KV/child pipelines.

The validated configuration uses B1/Hq8/Hkv8/D128 and BQ128/BK1024/CK128/VR8.
Score, P and PV are explicit double-buffered GM workspaces. Other constexpr
variants are not covered by the current device validation; see docs/attention-target.md.
"""

from __future__ import annotations

from xdsl_npu import language as N

# Tensor dimensions in annotations are interpreted by the DSL signature pass.
# ruff: noqa: F821

NEG_INF = float("-inf")


@N.prim_func(disjoint_external=True)
def attention_target(
    Q: N.Tensor((B, HQ, Sq, D), DTYPE),
    K: N.Tensor((B, HKV, Sk, D), DTYPE),
    V: N.Tensor((B, HKV, Sk, D), DTYPE),
    O: N.Tensor((B, HQ, Sq, D), DTYPE, access="write"),
    Score: N.Tensor((GROUPS, 2, BQ, BK), N.f32, access="write"),
    P: N.Tensor((GROUPS, 2, BQ, BK), DTYPE, access="write"),
    PV: N.Tensor((GROUPS, 2, BQ, D), N.f32, access="write"),
    Sq: N.int64(minimum=1, multiple_of=BQ),
    Sk: N.int64(minimum=1, multiple_of=BK),
    B: N.constexpr = 1,
    HQ: N.constexpr = 8,
    HKV: N.constexpr = 8,
    D: N.constexpr = 128,
    BQ: N.constexpr = 128,
    BK: N.constexpr = 1024,
    CK: N.constexpr = 128,
    VR: N.constexpr = 8,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="mixed") as group:
        # Declarations dominate all actor regions that use their storage.
        q1 = N.alloc_l1((BQ, D), DTYPE)
        k1 = N.alloc_l1((2, CK, D), DTYPE)
        p1 = N.alloc_l1((2, BQ, CK), DTYPE)
        v1 = N.alloc_l1((2, CK, D), DTYPE)
        qa = N.alloc_l0a((BQ, D), DTYPE)
        pa = N.alloc_l0a((BQ, CK), DTYPE)
        kb = N.alloc_l0b((D, CK), DTYPE)
        vb = N.alloc_l0b((CK, D), DTYPE)
        # Shared QK/PV L0C: two native planes, not three allocations.
        c0 = N.alloc_l0c((2, BQ, CK), N.f32)
        score_in = N.alloc_ub((2, VR, BK), N.f32)
        score_work = N.alloc_ub((VR, BK), N.f32)
        p_out = N.alloc_ub((2, VR, BK), DTYPE)
        pv_in = N.alloc_ub((2, VR, D), N.f32)
        o_out = N.alloc_ub((2, VR, D), DTYPE)
        m = N.alloc_ub((BQ // 2,), N.f32)
        l = N.alloc_ub((BQ // 2,), N.f32)
        o = N.alloc_ub((BQ // 2, D), N.f32)
        alpha = N.alloc_ub((2, BQ // 2), N.f32)
        partial = N.alloc_ub((VR, BK // 64), N.f32)
        next_m = N.alloc_ub((VR,), N.f32)
        row_sum = N.alloc_ub((VR,), N.f32)
        ones = N.alloc_ub((VR,), N.f32)
        inverse = N.alloc_ub((VR,), N.f32)
        # Physical BRCB scratch is explicit; no full-size broadcast tile.
        soft_broadcast = N.alloc_ub((VR, 8), N.f32)
        out_broadcast = N.alloc_ub((VR, 8), N.f32)
        query_tiles = Sq // BQ
        kv_tiles = Sk // BK
        local_queries = N.ceildiv(N.max(0, B * HQ * query_tiles - group), GROUPS)
        # Sq/Sk contracts are checked before launch; constexpr head/tile checks remain required.
        # Empty owners do no work. No implicit padding or global Q/K/V packing.
        for query in N.serial(local_queries):
            for kv in N.Pipelined(
                kv_tiles,
                stages=3,
                issue_order={"cube": ("qk", "pv"), "vector": ("update", "softmax")},
            ):
                task = query * kv_tiles + kv
                global_query = group + query * GROUPS
                batch = global_query // (HQ * query_tiles)
                head = (global_query // query_tiles) % HQ
                kv_head = head // (HQ // HKV)
                query_row = (global_query % query_tiles) * BQ
                s = task % 2
                with N.Stage(0, name="qk"), N.Cube():
                    if kv == 0:
                        N.copy(Q[batch, head, query_row : query_row + BQ, :], q1)
                        N.copy(q1, qa)
                    for chunk in N.Pipelined(
                        BK // CK,
                        stages=4,
                        issue_order={"cube": ("load_k", "qk_native", "operand_k", "export_score")},
                    ):
                        j = kv * BK + chunk * CK
                        with N.Stage(0, name="load_k"):
                            N.copy(K[batch, kv_head, j : j + CK, :], k1[chunk % 2, :, :])
                        with N.Stage(1, name="operand_k"):
                            N.copy(k1[chunk % 2, :, :], kb, transpose=True)
                        with N.Stage(2, name="qk_native"):
                            N.matmul(qa, kb, accum=c0[chunk % 2, :, :], initialize=True)
                        with N.Stage(3, name="export_score"):
                            N.copy(
                                c0[chunk % 2, :, :],
                                Score[group, s, :, chunk * CK : (chunk + 1) * CK],
                            )
                with N.Stage(1, name="softmax"), N.Vector() as lane:
                    if kv == 0:
                        N.fill(m, NEG_INF)
                        N.fill(l, 0.0)
                    for row in N.Pipelined(BQ // 2 // VR, stages=3):
                        r = row * VR
                        g = lane * (BQ // 2) + r
                        with N.Stage(0, name="import_score"):
                            N.copy(Score[group, s, g : g + VR, :], score_in[row % 2, :, :])
                        with N.Stage(1, name="online_softmax"):
                            N.mul(score_in[row % 2, :, :], SCALE, out=score_work)
                            N.reduce_max(
                                score_work,
                                axis=1,
                                out=next_m,
                                scratch=partial,
                                order="r64_then_packed_merge",
                            )
                            N.maximum(m[r : r + VR], next_m, out=next_m)
                            N.sub(m[r : r + VR], next_m, out=alpha[s, r : r + VR])
                            N.exp(alpha[s, r : r + VR], out=alpha[s, r : r + VR])
                            N.sub(
                                score_work, next_m[:, None], out=score_work, scratch=soft_broadcast
                            )
                            N.exp(score_work, out=score_work)
                            N.reduce_sum(
                                score_work,
                                axis=1,
                                out=row_sum,
                                scratch=partial,
                                order="r64_then_packed_merge",
                            )
                            N.mul(l[r : r + VR], alpha[s, r : r + VR], out=l[r : r + VR])
                            N.add(l[r : r + VR], row_sum, out=l[r : r + VR])
                            N.copy(next_m, m[r : r + VR])
                            N.cast(score_work, out=p_out[row % 2, :, :])
                        with N.Stage(2, name="export_p"):
                            N.copy(p_out[row % 2, :, :], P[group, s, g : g + VR, :])
                with N.Stage(1, name="pv"), N.Cube():
                    for chunk in N.Pipelined(
                        BK // CK,
                        stages=3,
                        issue_order={"cube": ("load_pv", "pv_native", "operand_pv")},
                    ):
                        j = kv * BK + chunk * CK
                        with N.Stage(0, name="load_pv"):
                            N.copy(
                                P[group, s, :, chunk * CK : (chunk + 1) * CK], p1[chunk % 2, :, :]
                            )
                            N.copy(V[batch, kv_head, j : j + CK, :], v1[chunk % 2, :, :])
                        with N.Stage(1, name="operand_pv"):
                            N.copy(p1[chunk % 2, :, :], pa)
                            N.copy(v1[chunk % 2, :, :], vb)
                        with N.Stage(2, name="pv_native"):
                            N.matmul(pa, vb, accum=c0[0, :, :], initialize=(chunk == 0))
                    N.copy(c0[0, :, :], PV[group, s, :, :])
                with N.Stage(2, name="update"), N.Vector() as lane:
                    if kv == 0:
                        N.fill(o, 0.0)
                    for row in N.Pipelined(BQ // 2 // VR, stages=3):
                        r = row * VR
                        g = lane * (BQ // 2) + r
                        with N.Stage(0, name="import_pv"):
                            N.copy(PV[group, s, g : g + VR, :], pv_in[row % 2, :, :])
                        with N.Stage(1, name="accumulate_o"):
                            N.mul(
                                o[r : r + VR, :],
                                alpha[s, r : r + VR, None],
                                out=o[r : r + VR, :],
                                scratch=out_broadcast,
                            )
                            N.add(o[r : r + VR, :], pv_in[row % 2, :, :], out=o[r : r + VR, :])
                            if kv == kv_tiles - 1:
                                N.fill(ones, 1.0)
                                N.div(ones, l[r : r + VR], out=inverse)
                                N.mul(
                                    o[r : r + VR, :],
                                    inverse[:, None],
                                    out=pv_in[row % 2, :, :],
                                    scratch=out_broadcast,
                                )
                                N.cast(pv_in[row % 2, :, :], out=o_out[row % 2, :, :])
                        with N.Stage(2, name="export_o"):
                            if kv == kv_tiles - 1:
                                N.copy(
                                    o_out[row % 2, :, :],
                                    O[batch, head, query_row + g : query_row + g + VR, :],
                                )

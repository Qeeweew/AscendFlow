"""Chunk-64 KDA prefill. Numerical boundaries follow the upstream Triton path.

Global phases are ordered kernel launches. Every kernel uses the ordinary DSL
and automatic synchronization; algorithms and packing policy live here.
"""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def prefill_prepare(
    Q: N.Tensor((TOKENS, H, K), DTYPE),
    Key: N.Tensor((TOKENS, H, K), DTYPE),
    Value: N.Tensor((TOKENS, H, K), DTYPE),
    Gate: N.Tensor((TOKENS, H, K), N.f32),
    Beta: N.Tensor((TOKENS, H), N.f32),
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    Vp: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32, access="write", write_disjoint=True),
    Starts: N.Tensor((CHUNKS,), N.i64),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    TOKENS: N.int64(minimum=1),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        assert CHUNKS == SEQUENCES * NC and NC > 0, "equal nonempty chunk domains"
        q16 = N.alloc_ub((BT, K), DTYPE)
        k16 = N.alloc_ub((BT, K), DTYPE)
        v16 = N.alloc_ub((BT, K), DTYPE)
        q = N.alloc_ub((BT, K), N.f32)
        k = N.alloc_ub((BT, K), N.f32)
        work = N.alloc_ub((BT, K), N.f32)
        gates = N.alloc_ub((2, K), N.f32)
        prefixes = N.alloc_ub((2, K), N.f32)
        cumulative = N.alloc_ub((K,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        partial = N.alloc_ub((BT * (K // 8),), N.f32)
        stat = N.alloc_ub((BT,), N.f32)
        rows = N.alloc_ub((BT, 8), N.f32)
        for task in N.serial(N.ceildiv(N.max(0, SEQUENCES * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = (flat // H) * NC + NC - 1, flat % H
            start = N.load_index(Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - 1)
            valid = N.load_index(
                Lengths[chunk : chunk + 1], minimum=1, maximum=N.min(BT, TOKENS - start)
            )
            N.fill(q, 0.0)
            N.cast(q, out=q16)
            N.cast(q, out=k16)
            N.cast(q, out=v16)
            N.copy(q, Gp[chunk, head, :, :])
            N.fill(cumulative, 0.0)
            N.fill(betas, 0.0)
            for row in N.Pipelined(valid, stages=3):
                token = start + row
                ring = row % 2
                with N.Stage(0, name="load"):
                    N.copy(Q[token, head, :], q16[row, :])
                    N.copy(Key[token, head, :], k16[row, :])
                    N.copy(Value[token, head, :], v16[row, :])
                    N.copy(Gate[token, head, :], gates[ring, :])
                    coefficient = N.load_scalar(Beta[token, head : head + 1])
                    N.store_scalar(coefficient, out=betas[row : row + 1])
                with N.Stage(1, name="prefix"):
                    N.add(cumulative, gates[ring, :], out=cumulative)
                    N.copy(cumulative, prefixes[ring, :])
                with N.Stage(2, name="store"):
                    N.copy(prefixes[ring, :], Gp[chunk, head, row, :])
            N.cast(q16, out=q)
            N.cast(k16, out=k)
            N.mul(q, q, out=work)
            N.reduce_sum(work, axis=1, out=stat, scratch=partial, order="block8_then_packed_merge")
            N.add(stat, 1e-6, out=stat)
            N.rsqrt(stat, out=stat)
            N.mul(q, stat[:, None], out=q, scratch=rows)
            N.cast(q, out=q16)
            N.mul(k, k, out=work)
            N.reduce_sum(work, axis=1, out=stat, scratch=partial, order="block8_then_packed_merge")
            N.add(stat, 1e-6, out=stat)
            N.rsqrt(stat, out=stat)
            N.mul(k, stat[:, None], out=k, scratch=rows)
            N.cast(k, out=k16)
            N.copy(q16, Qp[chunk, head, :, :])
            N.copy(k16, Kp[chunk, head, :, :])
            N.copy(v16, Vp[chunk, head, :, :])
            N.copy(betas, Bp[chunk, head, :])


@N.prim_func(disjoint_external=True)
def prefill_gram(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Diagonal: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Ai32: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        assert K == 128 and BT == 64, "four diagonal blocks share 64-wide Vector rows"
        keys16 = N.alloc_ub((16, K), DTYPE)
        keys = N.alloc_ub((16, K), N.f32)
        queries = N.alloc_ub((16, K), N.f32)
        weighted = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((16, K), N.f32)
        decay = N.alloc_ub((16, K), N.f32)
        steps = N.alloc_ub((16, K), N.f32)
        product = N.alloc_ub((16, K), N.f32)
        zeros = N.alloc_ub((BT, BT), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        partial = N.alloc_ub((16 * (K // 8),), N.f32)
        scores = N.alloc_ub((64,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        diagonal = N.alloc_ub((16, 16), N.f32)
        attention = N.alloc_ub((16, 16), N.f32)
        inverse = N.alloc_ub((16, BT), N.f32)
        coefficients = N.alloc_ub((BT,), N.f32)
        expanded = N.alloc_ub((BT, 64), N.f32)
        coefficient_matrix = N.alloc_ub((16, BT), N.f32)
        weighted_inverse = N.alloc_ub((16, BT), N.f32)
        inverse_tree = N.alloc_ub((8, BT), N.f32)
        inverse_rows = N.alloc_ub((BT, 8), N.f32)
        identity = N.alloc_ub((16, 16), N.f32)
        identities = N.alloc_ub((16, BT), N.f32)
        upper_storage = N.alloc_mask((16, 16))
        strict_upper_storage = N.alloc_mask((16, 16))
        mask_zero = N.alloc_ub((16, 16), N.f32)
        N.fill(scores, 0.0)
        # Coordinate masks are invariant across every chunk/head. Build their
        # packed words once, then consume them with Vector select in the loop.
        upper = N.materialize(N.arange(16)[:, None] <= N.arange(16)[None, :], out=upper_storage)
        strict_upper = N.materialize(
            N.arange(16)[:, None] < N.arange(16)[None, :], out=strict_upper_storage
        )
        N.fill(identity, 0.0)
        N.fill(identity, 1.0, mask=N.arange(16)[:, None] == N.arange(16)[None, :])
        for plane in N.unroll(BT // 16):
            N.copy(identity, identities[:, plane * 16 : plane * 16 + 16])
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
            N.copy(Bp[chunk, head, :], betas)
            N.fill(zeros, 0.0)
            N.copy(zeros, Aqk[chunk, head, :, :])
            N.copy(zeros, Diagonal[chunk, head, :, :])
            N.copy(zeros, Ai32[chunk, head, :, :])
            for block_index in N.serial(BT // 16):
                block = block_index * 16
                N.copy(Kp[chunk, head, block : block + 16, :], keys16)
                N.cast(keys16, out=keys)
                N.mul(keys, betas[block : block + 16, None], out=weighted, scratch=rows)
                N.copy(Qp[chunk, head, block : block + 16, :], keys16)
                N.cast(keys16, out=queries)
                N.copy(Gp[chunk, head, block : block + 16, :], gates)
                N.fill(steps[0, :], 1.0)
                N.mul(gates[:15, :], -1.0, out=steps[1:16, :])
                N.add(steps[1:16, :], gates[1:16, :], out=steps[1:16, :])
                N.exp(steps[1:16, :], out=steps[1:16, :])
                N.copy(keys, decay)
                N.fill(diagonal, 0.0)
                N.fill(attention, 0.0)
                # Four causal prefixes avoid evaluating most upper-triangle
                # exp/products, retaining the same FP32 per-score sum tree.
                for quarter in N.unroll(4):
                    extent = (quarter + 1) * 4
                    for local in N.serial(quarter * 4, N.min(extent, N.max(0, valid - block))):
                        key_row = weighted[local, :]
                        query_row = queries[local, :]
                        step_row = steps[local, :]
                        # Carry k_j * exp(G_i-G_j) forward one row. Each
                        # diagonal resets to k_i; upper entries are discarded
                        # by the final masks after both score reductions.
                        N.mul(decay[:extent, :], step_row[None, :], out=decay[:extent, :])
                        N.copy(keys[local, :], decay[local, :])
                        N.mul(decay[:extent, :], key_row[None, :], out=product[:extent, :])
                        N.reduce_sum(
                            product[:extent, :],
                            axis=1,
                            out=scores[:extent],
                            scratch=partial[: extent * (K // 8)],
                            order="block8_then_packed_merge",
                        )
                        N.copy(scores[:16], diagonal[local, :])
                        N.mul(decay[:extent, :], query_row[None, :], out=product[:extent, :])
                        N.reduce_sum(
                            product[:extent, :],
                            axis=1,
                            out=scores[:extent],
                            scratch=partial[: extent * (K // 8)],
                            order="block8_then_packed_merge",
                        )
                        N.mul(scores[:16], SCALE, out=scores[:16])
                        N.copy(scores[:16], attention[local, :])
                N.fill(diagonal, 0.0, mask=upper, scratch=mask_zero)
                N.fill(attention, 0.0, mask=strict_upper, scratch=mask_zero)
                N.copy(attention, Aqk[chunk, head, block : block + 16, block : block + 16])
                N.copy(diagonal, inverse[:, block : block + 16])
            # Four independent 16x16 blocks share one row recurrence. The
            # 64-wide row layout keeps the multiply and every sum-tree level
            # on complete Vector repeats, preserving each block's FP32 tree.
            N.mul(inverse, -1.0, out=inverse)
            for inverse_row in N.serial(2, 16):
                N.copy(inverse[inverse_row, :], coefficients)
                N.broadcast(coefficients[:, None], out=expanded, scratch=inverse_rows)
                for plane in N.unroll(BT // 16):
                    N.copy(
                        expanded[plane * 16 : plane * 16 + 16, :16],
                        coefficient_matrix[:, plane * 16 : plane * 16 + 16],
                    )
                N.mul(inverse, coefficient_matrix, out=weighted_inverse)
                N.add(weighted_inverse[:8, :], weighted_inverse[8:16, :], out=inverse_tree)
                N.add(inverse_tree[:4, :], inverse_tree[4:8, :], out=inverse_tree[:4, :])
                N.add(inverse_tree[:2, :], inverse_tree[2:4, :], out=inverse_tree[:2, :])
                N.add(inverse_tree[0, :], inverse_tree[1, :], out=inverse_tree[0, :])
                N.add(coefficients, inverse_tree[0, :], out=inverse[inverse_row, :])
            N.add(inverse, identities, out=inverse)
            for plane in N.unroll(BT // 16):
                block = plane * 16
                N.copy(
                    inverse[:, block : block + 16],
                    Ai32[chunk, head, block : block + 16, block : block + 16],
                )
                N.copy(
                    inverse[:, block : block + 16],
                    Diagonal[chunk, head, block : block + 16, block : block + 16],
                )


@N.prim_func(disjoint_external=True)
def prefill_transform(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Vp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    Right: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE, access="write", write_disjoint=True),
    KG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        raw = N.alloc_ub((BT, K), DTYPE)
        packed = N.alloc_ub((BT, K), DTYPE)
        work = N.alloc_ub((BT, K), N.f32)
        gates = N.alloc_ub((BT, K), N.f32)
        factor = N.alloc_ub((BT, K), N.f32)
        last = N.alloc_ub((K,), N.f32)
        beta = N.alloc_ub((BT,), N.f32)
        rows = N.alloc_ub((BT, 8), N.f32)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
            N.copy(Bp[chunk, head, :], beta)
            N.copy(Gp[chunk, head, :, :], gates)
            N.copy(Gp[chunk, head, valid - 1, :], last)
            N.exp(gates, out=factor)
            N.copy(Kp[chunk, head, :, :], raw)
            N.cast(raw, out=work)
            N.mul(work, beta[:, None], out=work, scratch=rows)
            N.mul(work, factor, out=work)
            N.cast(work, out=packed)
            N.copy(packed, Right[chunk, head, 0, :, :])
            N.copy(Vp[chunk, head, :, :], raw)
            N.cast(raw, out=work)
            N.mul(work, beta[:, None], out=work, scratch=rows)
            N.cast(work, out=packed)
            N.copy(packed, Right[chunk, head, 1, :, :])
            N.mul(gates, -1.0, out=factor)
            N.add(factor, last[None, :], out=factor)
            N.exp(factor, out=factor)
            N.copy(Kp[chunk, head, :, :], raw)
            N.cast(raw, out=work)
            N.mul(work, factor, out=work)
            N.cast(work, out=packed)
            N.copy(packed, KG[chunk, head, :, :])
            N.copy(Qp[chunk, head, :, :], raw)
            N.cast(raw, out=work)
            N.mul(work, SCALE, out=work)
            N.cast(work, out=packed)
            N.cast(packed, out=work)
            N.exp(gates, out=factor)
            N.mul(work, factor, out=work)
            N.cast(work, out=packed)
            N.copy(packed, QG[chunk, head, :, :])


@N.prim_func(disjoint_external=True)
def prefill_wu(
    Inverse: N.Tensor((CHUNKS, H, BT, BT), DTYPE),
    Right: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE),
    WU: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="cube") as owner:
        a1 = N.alloc_l1((2, BT, BT), DTYPE)
        b1 = N.alloc_l1((2, 2, BT, K), DTYPE)
        a0 = N.alloc_l0a((BT, BT), DTYPE)
        b0 = N.alloc_l0b((BT, K), DTYPE)
        c0 = N.alloc_l0c((BT, K), N.f32)
        for task in N.Pipelined(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS), stages=2):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            ring = task % 2
            with N.Stage(0, name="load"):
                N.copy(Inverse[chunk, head, :, :], a1[ring, :, :])
                for part in N.unroll(2):
                    N.copy(Right[chunk, head, part, :, :], b1[ring, part, :, :])
            with N.Stage(1, name="compute"):
                aa = N.copy(a1[ring, :, :], a0)
                for part in N.unroll(2):
                    bb = N.copy(b1[ring, part, :, :], b0)
                    N.matmul(aa, bb, accum=c0, initialize=True)
                    N.copy(c0, WU[chunk, head, part, :, :])


@N.prim_func(disjoint_external=True)
def prefill_scan(
    WU: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE),
    KG: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32),
    Initial: N.Tensor((SEQUENCES, H, K, K), N.f32),
    Final: N.Tensor((SEQUENCES, H, K, K), N.f32, access="write", write_disjoint=True),
    Output: N.Tensor((TOKENS, H, K), DTYPE, access="write", write_disjoint=True),
    History: N.Tensor((SEQUENCES, H, K, K), N.f32, access="write"),
    Snapshot: N.Tensor((CHUNKS, H, K, K), DTYPE, access="write"),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
    Projection: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
    Residual: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write"),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE, access="write"),
    Delta: N.Tensor((CHUNKS, H, K, K), N.f32, access="write"),
    Inter: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
    Intra: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
    Starts: N.Tensor((CHUNKS,), N.i64),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    TOKENS: N.int64(minimum=1),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    BV: N.constexpr = 64,
    CK: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64 and CK == 64 and BV == 64, "initial scan tile policy"
        assert CHUNKS == SEQUENCES * NC, "each sequence has NC nonempty chunks"
        carry = N.alloc_ub((BV, K), N.f32)
        state16 = N.alloc_ub((BV, K), DTYPE)
        w16 = N.alloc_ub((BT // 2, K), DTYPE)
        w32 = N.alloc_ub((BT // 2, K), N.f32)
        u16 = N.alloc_ub((BT, BV), DTYPE)
        scratch = N.alloc_ub((BT, BV), N.f32)
        other = N.alloc_ub((BT, BV), N.f32)
        r16 = N.alloc_ub((BT, BV), DTYPE)
        af32 = N.alloc_ub((BT // 2, BT), N.f32)
        af16 = N.alloc_ub((BT // 2, BT), DTYPE)
        update = N.alloc_ub((BV, K), N.f32)
        last_gate = N.alloc_ub((K,), N.f32)
        a32_1 = N.alloc_l1((BT, CK), N.f32)
        b32_1 = N.alloc_l1((K, CK), N.f32)
        a32_0 = N.alloc_l0a((BT, CK), N.f32)
        b32_0 = N.alloc_l0b((CK, K), N.f32)
        res1 = N.alloc_l1((BT, K), DTYPE)
        kg1 = N.alloc_l1((BT, K), DTYPE)
        q1 = N.alloc_l1((BT, CK), DTYPE)
        h1 = N.alloc_l1((K, CK), DTYPE)
        aq1 = N.alloc_l1((BT, BT), DTYPE)
        update_a0 = N.alloc_l0a((K, BT), DTYPE)
        out_a0 = N.alloc_l0a((BT, CK), DTYPE)
        right0 = N.alloc_l0b((BT, K), DTYPE)
        score0 = N.alloc_l0c((BT, K), N.f32)
        state0 = N.alloc_l0c((K, K), N.f32)
        for task in N.serial(N.ceildiv(N.max(0, SEQUENCES * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            seq, head = flat // H, flat % H
            with N.Vector() as initial_lane:
                N.copy(Initial[seq, head, initial_lane * BV : (initial_lane + 1) * BV, :], carry)
            for local in N.serial(NC):
                chunk = seq * NC + local
                for clock in N.Pipelined(1, stages=5):
                    with N.Stage(0, name="publish"), N.Vector() as lane:
                        N.copy(carry, History[seq, head, lane * BV : (lane + 1) * BV, :])
                        N.cast(carry, out=state16)
                        N.copy(state16, Snapshot[chunk, head, lane * BV : (lane + 1) * BV, :])
                        N.copy(
                            WU[chunk, head, 0, lane * (BT // 2) : (lane + 1) * (BT // 2), :], w16
                        )
                        N.cast(w16, out=w32)
                        N.copy(w32, W32[chunk, head, lane * (BT // 2) : (lane + 1) * (BT // 2), :])
                    with N.Stage(1, name="projection"), N.Cube():
                        for part in N.unroll(K // CK):
                            aa = N.copy(W32[chunk, head, :, part * CK : (part + 1) * CK], a32_1)
                            bb = N.copy(History[seq, head, :, part * CK : (part + 1) * CK], b32_1)
                            aa = N.copy(aa, a32_0)
                            bb = N.copy(bb, b32_0, transpose=True)
                            N.matmul(aa, bb, accum=score0, initialize=part == 0)
                        N.copy(score0, Projection[chunk, head, :, :])
                    with N.Stage(2, name="residual"), N.Vector() as lane:
                        N.copy(WU[chunk, head, 1, :, lane * BV : (lane + 1) * BV], u16)
                        N.cast(u16, out=other)
                        N.copy(Projection[chunk, head, :, lane * BV : (lane + 1) * BV], scratch)
                        N.sub(other, scratch, out=scratch)
                        N.cast(scratch, out=r16)
                        N.copy(r16, Residual[chunk, head, :, lane * BV : (lane + 1) * BV])
                        N.copy(Aqk[chunk, head, lane * (BT // 2) : (lane + 1) * (BT // 2), :], af32)
                        N.cast(af32, out=af16)
                        N.copy(af16, A16[chunk, head, lane * (BT // 2) : (lane + 1) * (BT // 2), :])
                    with N.Stage(3, name="products"), N.Cube():
                        rr = N.copy(Residual[chunk, head, :, :], res1)
                        kk = N.copy(KG[chunk, head, :, :], kg1)
                        aa = N.copy(rr, update_a0, transpose=True)
                        bb = N.copy(kk, right0)
                        N.matmul(aa, bb, accum=state0, initialize=True)
                        N.copy(state0, Delta[chunk, head, :, :])
                        for part in N.unroll(K // CK):
                            qq = N.copy(QG[chunk, head, :, part * CK : (part + 1) * CK], q1)
                            hh = N.copy(Snapshot[chunk, head, :, part * CK : (part + 1) * CK], h1)
                            aa = N.copy(qq, out_a0)
                            bb = N.copy(hh, right0, transpose=True)
                            N.matmul(aa, bb, accum=score0, initialize=part == 0)
                        N.copy(score0, Inter[chunk, head, :, :])
                        aq = N.copy(A16[chunk, head, :, :], aq1)
                        aa = N.copy(aq, out_a0)
                        bb = N.copy(res1, right0)
                        N.matmul(aa, bb, accum=score0, initialize=True)
                        N.copy(score0, Intra[chunk, head, :, :])
                    with N.Stage(4, name="commit"), N.Vector() as lane:
                        start = N.load_index(
                            Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - 1
                        )
                        valid = N.load_index(
                            Lengths[chunk : chunk + 1], minimum=1, maximum=N.min(BT, TOKENS - start)
                        )
                        N.copy(Gp[chunk, head, valid - 1, :], last_gate)
                        N.exp(last_gate, out=last_gate)
                        N.mul(carry, last_gate[None, :], out=carry)
                        N.copy(Delta[chunk, head, lane * BV : (lane + 1) * BV, :], update)
                        N.add(carry, update, out=carry)
                        N.copy(Inter[chunk, head, :, lane * BV : (lane + 1) * BV], scratch)
                        N.copy(Intra[chunk, head, :, lane * BV : (lane + 1) * BV], other)
                        N.add(scratch, other, out=scratch)
                        N.cast(scratch, out=r16)
                        if local < NC - 1:
                            full_start = N.load_index(
                                Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - BT
                            )
                            N.copy(
                                r16,
                                Output[
                                    full_start : full_start + BT, head, lane * BV : (lane + 1) * BV
                                ],
                            )
                        else:
                            for row in N.serial(valid):
                                N.copy(
                                    r16[row, :],
                                    Output[start + row, head, lane * BV : (lane + 1) * BV],
                                )
            with N.Vector() as final_lane:
                N.copy(carry, Final[seq, head, final_lane * BV : (final_lane + 1) * BV, :])

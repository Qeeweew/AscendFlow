"""Chunk/head-local KDA front end; all C/V handoffs use ordinary auto sync."""

from __future__ import annotations

from ascendflow import language as N
from examples.kda.prefill_gram import diagonal_gram, invert_diagonal
from examples.kda.prefill_gram_cross import (
    import_gram_item,
    prepare_one_gram,
)
from examples.kda.prefill_solve import (
    merge_inverse_blocks,
    publish_inverse_blocks,
    read_inverse_blocks,
    solve_import,
)
from examples.kda.prefill_wu import wu_import, wu_products

# ruff: noqa: F821


@N.inline
def normalize_panel(raw, value, work, statistic, partial, rows):
    N.cast(raw, out=value)
    N.mul(value, value, out=work)
    N.reduce_sum(work, axis=1, out=statistic, scratch=partial, order="block8_then_packed_merge")
    N.add(statistic, 1e-6, out=statistic)
    N.rsqrt(statistic, out=statistic)
    N.mul(value, statistic[:, None], out=value, scratch=rows)
    N.cast(value, out=raw)


@N.inline
def advance_gate(Gate, staging, cumulative, start, token, valid, head):
    if token < valid:
        N.copy(Gate[start + token, head, :], staging)
    else:
        N.fill(staging, 0.0)
    N.add(cumulative, staging, out=cumulative)


@N.inline
def prepare_inputs(inputs, outputs, buffers, geometry, coordinates):
    Q, Key, Value, Gate, Beta, Starts, Lengths = inputs
    Qp, Kp, Vp, Gp, Bp = outputs
    (
        q16,
        k16,
        v16,
        values,
        work,
        statistic,
        partial,
        rows,
        beta8,
        betas,
        gates,
        cumulative,
        tail_prefix,
    ) = buffers
    BT, TAIL = geometry
    chunk, head, lane, slot = coordinates
    start = N.load_index(Starts[chunk : chunk + 1])
    valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
    N.fill(betas, 0.0)
    N.fill(cumulative, 0.0)
    for panel in N.serial(2):
        row = lane * 32 + panel * 16
        N.fill(q16, 0.0)
        N.fill(k16, 0.0)
        N.fill(v16, 0.0)
        N.fill(gates, 0.0)
        if TAIL == 0 or row + 16 <= valid:
            N.copy(Q[start + row : start + row + 16, head, :], q16)
            N.copy(Key[start + row : start + row + 16, head, :], k16)
            N.copy(Value[start + row : start + row + 16, head, :], v16)
            N.copy(Gate[start + row : start + row + 16, head, :], gates)
            beta_head = (head // 8) * 8
            N.copy(Beta[start + row : start + row + 16, beta_head : beta_head + 8], beta8)
            N.fill(beta8, 0.0, mask=N.arange(8)[None, :] != head % 8)
            N.reduce_sum(
                beta8,
                axis=1,
                out=betas[row : row + 16],
                scratch=partial[:16],
                order="block8_then_packed_merge",
            )
        else:
            for token in N.serial(16):
                if row + token < valid:
                    N.copy(Q[start + row + token, head, :], q16[token, :])
                    N.copy(Key[start + row + token, head, :], k16[token, :])
                    N.copy(Value[start + row + token, head, :], v16[token, :])
                    N.copy(Gate[start + row + token, head, :], gates[token, :])
                    coefficient = N.load_scalar(Beta[start + row + token, head : head + 1])
                    N.store_scalar(coefficient, out=betas[row + token : row + token + 1])
        normalize_panel(q16, values, work, statistic, partial, rows)
        normalize_panel(k16, values, work, statistic, partial, rows)
        N.copy(q16, Qp[slot, 0, row : row + 16, :])
        N.copy(k16, Kp[slot, 0, row : row + 16, :])
        N.copy(v16, Vp[slot, 0, row : row + 16, :])
        N.copy(gates, Gp[slot, 0, row : row + 16, :])
    N.copy(betas[lane * 32 : lane * 32 + 32], Bp[slot, 0, lane * 32 : lane * 32 + 32])
    if TAIL == 1:
        # Keep the tail's FP32 sequential prefix. Both peers read the original
        # gates; each publishes only its own rows, with no peer-to-peer UB use.
        N.fill(cumulative, 0.0)
        if lane == 1:
            for token in N.serial(32):
                advance_gate(Gate, gates[0, :], cumulative, start, token, valid, head)
        for local in N.serial(32):
            token = lane * 32 + local
            advance_gate(Gate, gates[0, :], cumulative, start, token, valid, head)
            N.copy(cumulative, tail_prefix[local, :])
        if lane == 0:
            for local in N.serial(32):
                token = 32 + local
                advance_gate(Gate, gates[0, :], cumulative, start, token, valid, head)
        N.copy(tail_prefix, Gp[slot, 0, lane * 32 : lane * 32 + 32, :])


@N.inline
def transform_panels(inputs, outputs, buffers, geometry, coordinates):
    Qp, Kp, Vp, Gp, Bp, Aqk, tail_last = inputs
    Right, KG, QG, A16 = outputs
    raw, packed, work, gates, factor, last, beta, rows, af32, af16 = buffers
    SCALE, TAIL = geometry
    chunk, head, lane, slot = coordinates
    if TAIL == 0:
        N.copy(Gp[slot, 0, 63, :], last)
    else:
        N.copy(tail_last, last)
    for panel in N.serial(2):
        row = lane * 32 + panel * 16
        N.copy(Gp[slot, 0, row : row + 16, :], gates)
        N.copy(Bp[slot, 0, row : row + 16], beta[:16])
        N.exp(gates, out=factor)
        N.copy(Kp[slot, 0, row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, beta[:16, None], out=work, scratch=rows)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, Right[slot, 0, 0, row : row + 16, :])
        N.copy(Vp[slot, 0, row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, beta[:16, None], out=work, scratch=rows)
        N.cast(work, out=packed)
        N.copy(packed, Right[slot, 0, 1, row : row + 16, :])
        N.mul(gates, -1.0, out=factor)
        N.add(factor, last[None, :], out=factor)
        N.exp(factor, out=factor)
        N.copy(Kp[slot, 0, row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, KG[chunk, head, row : row + 16, :])
        N.copy(Qp[slot, 0, row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, SCALE, out=work)
        N.cast(work, out=packed)
        N.cast(packed, out=work)
        N.exp(gates, out=factor)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, QG[chunk, head, row : row + 16, :])
        N.copy(Aqk[slot, 0, row : row + 16, :], af32)
        N.cast(af32, out=af16)
        N.copy(af16, A16[chunk, head, row : row + 16, :])


@N.inline
def packed_gram_products(GramProducts, buffers, coordinates):
    a1, b1s, b1l, a0, b0s, b0l, c0 = buffers
    owner, ring = coordinates
    # Four independent GEMMs occupy disjoint columns of one accumulator.
    # Publish their complete word with one FIX, retaining FP32 arithmetic.
    for panel in N.unroll(2):
        aa = N.copy(a1[0, panel * 32 : panel * 32 + 32, :], a0)
        bb = N.copy(b1s[0, panel, :, :], b0s, transpose=True)
        N.matmul(aa, bb, accum=c0[:, panel * 16 : panel * 16 + 16], initialize=True)
    bb = N.copy(b1l[0, :, :], b0l, transpose=True)
    for part in N.unroll(2):
        aa = N.copy(a1[0, 64 + part * 32 : 96 + part * 32, :], a0)
        N.matmul(aa, bb, accum=c0[:, 32 + part * 32 : 64 + part * 32], initialize=True)
    N.copy(c0, GramProducts[owner, ring, :, :])


@N.inline
def store_packed_gram(inputs, buffers, coordinates):
    GramProducts, Bp, Negative, Aqk = inputs
    scores32, scores16, betas, rows = buffers
    valid, lane, owner, ring = coordinates
    for panel in N.unroll(3):
        if panel < 2:
            start, col, extent = 16 + panel * 32, panel * 32, 16
            scores = scores16[:8, :]
            key_source = GramProducts[
                owner, ring, lane * 8 : lane * 8 + 8, panel * 16 : panel * 16 + 16
            ]
            query_source = GramProducts[
                owner, ring, 16 + lane * 8 : 24 + lane * 8, panel * 16 : panel * 16 + 16
            ]
        else:
            start, col, extent = 32, 0, 32
            scores = scores32
            key_source = GramProducts[owner, ring, lane * 16 : lane * 16 + 16, 32:64]
            query_source = GramProducts[owner, ring, lane * 16 : lane * 16 + 16, 64:96]
        half = extent // 2
        row = start + lane * half
        # Publish Aqk before the Negative handshake into Cube. Transform
        # later acquires Cube's final inverse result, retaining both AIVs'
        # writes even where Gram row ownership differs from output ownership.
        N.copy(query_source, scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(scores, Aqk[owner, 0, row : row + half, col : col + extent])
        N.copy(key_source, scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(Bp[owner, 0, row : row + half], betas[:half])
        N.mul(scores, betas[:half, None], out=scores, scratch=rows[:half, :])
        N.mul(scores, -1.0, out=scores)
        N.copy(scores, Negative[owner, 0, row : row + half, col : col + extent])


@N.prim_func(disjoint_external=True)
def prefill_prepare(
    Q: N.Tensor((TOKENS, H, K), DTYPE),
    Key: N.Tensor((TOKENS, H, K), DTYPE),
    Value: N.Tensor((TOKENS, H, K), DTYPE),
    Gate: N.Tensor((TOKENS, H, K), N.f32),
    Beta: N.Tensor((TOKENS, H), N.f32),
    Qp: N.Tensor((GROUPS, 1, BT, K), DTYPE, access="write"),
    Kp: N.Tensor((GROUPS, 1, BT, K), DTYPE, access="write"),
    Vp: N.Tensor((GROUPS, 1, BT, K), DTYPE, access="write"),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    LocalGp: N.Tensor((GROUPS, 1, BT, K), N.f32, access="write"),
    Bp: N.Tensor((GROUPS, 1, BT), N.f32, access="write"),
    Aqk: N.Tensor((GROUPS, 1, BT, BT), N.f32, access="write"),
    Diagonal: N.Tensor((GROUPS, 1, BT, BT), N.f32, access="write"),
    Ai32: N.Tensor((GROUPS, 1, BT, BT), N.f32, access="write"),
    Negative: N.Tensor((GROUPS, 1, BT, BT), N.f32, access="write"),
    Inverse: N.Tensor((GROUPS, 1, BT, BT), DTYPE, access="write"),
    Right: N.Tensor((GROUPS, 1, 2, BT, K), DTYPE, access="write"),
    KG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE, access="write", write_disjoint=True),
    U: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    GramLeft: N.Tensor((GROUPS, 2, 1, 128, K), N.f32, access="write"),
    GramRight: N.Tensor((GROUPS, 2, 1, 64, K), N.f32, access="write"),
    GramProducts: N.Tensor((GROUPS, 2, 32, 96), N.f32, access="write"),
    Merge16: N.Tensor((GROUPS, 2, 16, 16), N.f32, access="write"),
    Merge32: N.Tensor((GROUPS, 2, 32, 32), N.f32, access="write"),
    Off32: N.Tensor((GROUPS, 2, 32, 32), N.f32, access="write"),
    W16Ring: N.Tensor((GROUPS, 2, BT, K), DTYPE, access="write"),
    Starts: N.Tensor((CHUNKS,), N.i64),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    Lower: N.Tensor((2, BT, BT), N.f32),
    TOKENS: N.i64(minimum=1),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.bf16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert CHUNKS == SEQUENCES * NC and NC > 0, "equal nonempty chunk domains"
        assert H % 8 == 0 and K == 128 and BT == 64, "KDA front-end tile policy"
        keys16 = N.alloc_ub((2, 16, K), DTYPE)
        queries16 = N.alloc_ub((2, 16, K), DTYPE)
        keys = N.alloc_ub((16, K), N.f32)
        queries = N.alloc_ub((16, K), N.f32)
        weighted = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((2, 16, K), N.f32)
        decay = N.alloc_ub((16, K), N.f32)
        steps = N.alloc_ub((16, K), N.f32)
        product = N.alloc_ub((32, K), N.f32)
        zeros = N.alloc_ub((BT, BT), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        partial = N.alloc_ub((32 * (K // 8),), N.f32)
        scores = N.alloc_ub((64,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        diagonal = N.alloc_ub((16, 16), N.f32)
        attention = N.alloc_ub((2, 16, 16), N.f32)
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
        anchor = N.alloc_ub((K,), N.f32)
        tail_last = N.alloc_ub((K,), N.f32)
        tail_prefix = N.alloc_ub((32, K), N.f32)
        packed_inverse = N.alloc_ub((32, BT), DTYPE)
        inverse_raw = N.alloc_ub((32, BT), N.f32)
        cross_scores = N.alloc_ub((16, 32), N.f32)
        # Different subphases reuse UB and native storage. The gate prefix
        # uses two 64-column passes to keep L0A/B within their 64 KiB banks.
        front_a1 = N.alloc_l1((1, 128, K), N.f32)
        front_b1 = N.alloc_l1((1, 64, K), N.f32)
        lower1 = front_a1[0, :, :BT]
        lower0 = N.alloc_l0a((BT, BT), N.f32)
        gate1 = front_b1[0, :, :64]
        gate0 = N.alloc_l0b((BT, 64), N.f32)
        prefix0 = N.alloc_l0c((BT, K), N.f32)
        ga1 = front_a1
        gb1s = N.alloc_l1((1, 2, 16, K), N.f32)
        gb1l = N.alloc_l1((1, 32, K), N.f32)
        ga0s = N.alloc_l0a((32, K), N.f32)
        gb0s = N.alloc_l0b((K, 16), N.f32)
        gb0l = N.alloc_l0b((K, 32), N.f32)
        gc0l = N.alloc_l0c((32, 96), N.f32)
        sa16 = front_a1[:, :, :16]
        sb16 = front_b1[:, :, :16]
        sa32 = front_a1[:, :, :32]
        sb32 = front_b1[:, :, :32]
        sa016 = N.alloc_l0a((16, 16), N.f32)
        sb016 = N.alloc_l0b((16, 16), N.f32)
        sc016 = N.alloc_l0c((16, 16), N.f32)
        sa032 = N.alloc_l0a((32, 32), N.f32)
        sb032 = N.alloc_l0b((32, 32), N.f32)
        sc032 = gc0l[:, :32]
        wa1 = N.alloc_l1((1, 128, BT), DTYPE)
        wb1 = N.alloc_l1((1, 2, BT, K), DTYPE)
        wa0 = N.alloc_l0a((BT, BT), DTYPE)
        wb0 = N.alloc_l0b((BT, K), DTYPE)
        wc0 = prefix0
        with N.Vector():
            N.fill(scores, 0.0)
            N.fill(identity, 0.0)
            N.fill(identity, 1.0, mask=N.arange(16)[:, None] == N.arange(16)[None, :])
            for plane in N.unroll(BT // 16):
                N.copy(identity, identities[:, plane * 16 : plane * 16 + 16])
            N.fill(zeros, 0.0)
            N.fill(product, 0.0)
        for tail in N.unroll(2):
            if tail == 1 or NC > 1:
                for task in N.Dataflow(
                    N.ceildiv(
                        N.max(0, SEQUENCES * (1 if tail == 1 else NC - 1) * H - owner), GROUPS
                    )
                ):
                    work = owner + task * GROUPS
                    head = work % H
                    if tail == 1:
                        chunk = (work // H) * NC + NC - 1
                    else:
                        full = work // H
                        chunk = (full // (NC - 1)) * NC + full % (NC - 1)
                    ring = 0
                    with N.Vector() as lane, N.Stage(0, name="inputs"):
                        prepare_inputs(
                            (Q, Key, Value, Gate, Beta, Starts, Lengths),
                            (Qp, Kp, Vp, LocalGp, Bp),
                            (
                                queries16[0, :, :],
                                keys16[0, :, :],
                                keys16[1, :, :],
                                keys,
                                weighted,
                                scores[:16],
                                partial[: 16 * (K // 8)],
                                rows,
                                inverse_rows[:16, :],
                                betas,
                                gates[0, :, :],
                                anchor,
                                tail_prefix,
                            ),
                            (BT, tail),
                            (chunk, head, lane, owner),
                        )
                        N.copy(anchor, tail_last)
                    with N.Cube(), N.Stage(0, name="prefix"):
                        N.copy(Lower[tail, :, :], lower1[:BT, :])
                        N.copy(lower1[:BT, :], lower0)
                        for part in N.unroll(2):
                            N.copy(LocalGp[owner, 0, :, part * 64 : part * 64 + 64], gate1)
                            gg = N.copy(gate1, gate0)
                            N.matmul(
                                lower0,
                                gg,
                                accum=prefix0[:, part * 64 : part * 64 + 64],
                                initialize=True,
                            )
                        N.copy(prefix0, LocalGp[owner, 0, :, :])
                        N.copy(prefix0, Gp[chunk, head, :, :])
                    with N.Vector() as lane, N.Stage(0, name="diagonal"):
                        valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                        N.fill(zeros, 0.0)
                        N.copy(zeros[:32, :], Aqk[owner, 0, lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], Diagonal[owner, 0, lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], Ai32[owner, 0, lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], Negative[owner, 0, lane * 32 : lane * 32 + 32, :])
                        N.fill(inverse, 0.0)
                        upper = N.materialize(
                            N.arange(16)[:, None] <= N.arange(16)[None, :], out=upper_storage
                        )
                        strict_upper = N.materialize(
                            N.arange(16)[:, None] < N.arange(16)[None, :], out=strict_upper_storage
                        )
                        for local in N.serial(2):
                            block = lane * 32 + local * 16
                            N.copy(Kp[owner, 0, block : block + 16, :], keys16[0, :, :])
                            N.copy(Qp[owner, 0, block : block + 16, :], queries16[0, :, :])
                            N.copy(LocalGp[owner, 0, block : block + 16, :], gates[0, :, :])
                            diagonal_gram(
                                (
                                    keys16,
                                    keys,
                                    betas,
                                    weighted,
                                    rows,
                                    queries16,
                                    queries,
                                    steps,
                                    gates,
                                    decay,
                                    diagonal,
                                    attention,
                                    product,
                                    scores,
                                    partial,
                                    upper,
                                    mask_zero,
                                    strict_upper,
                                    inverse,
                                ),
                                (SCALE, K),
                                (0, block, valid),
                            )
                            N.copy(
                                attention[0, :, :],
                                Aqk[owner, 0, block : block + 16, block : block + 16],
                            )
                        invert_diagonal(
                            (
                                inverse,
                                coefficients,
                                coefficient_matrix,
                                expanded,
                                inverse_rows,
                                weighted_inverse,
                                inverse_tree,
                                identities,
                            ),
                            (BT,),
                        )
                        for local in N.unroll(2):
                            block = lane * 32 + local * 16
                            N.copy(
                                inverse[:, block : block + 16],
                                Ai32[owner, 0, block : block + 16, block : block + 16],
                            )
                            N.copy(
                                inverse[:, block : block + 16],
                                Diagonal[owner, 0, block : block + 16, block : block + 16],
                            )
                        prepare_one_gram(
                            (LocalGp, Kp, GramLeft, Qp, GramRight),
                            (
                                anchor,
                                gates[0, :, :],
                                weighted,
                                keys16[0, :, :],
                                keys,
                                queries,
                                steps,
                                decay,
                            ),
                            (SCALE,),
                            (owner, 0, lane, owner, ring, 0),
                        )
                    with N.Cube(), N.Stage(0, name="cross_gram"):
                        import_gram_item(
                            (GramLeft, GramRight), (ga1, gb1s, gb1l), (owner, ring, 0, 0)
                        )
                        packed_gram_products(
                            GramProducts,
                            (ga1, gb1s, gb1l, ga0s, gb0s, gb0l, gc0l),
                            (owner, ring),
                        )
                    with N.Vector() as lane, N.Stage(0, name="gram_store"):
                        valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                        # Reuse an already retired inverse/identity scratch for scores.
                        store_packed_gram(
                            (GramProducts, Bp, Negative, Aqk),
                            (cross_scores, diagonal, betas[:16], rows),
                            (valid, lane, owner, ring),
                        )
                    with N.Cube(), N.Stage(0, name="solve"):
                        for pair in N.unroll(2):
                            left, right = pair * 32, pair * 32 + 16
                            solve_import(
                                (Diagonal, Negative),
                                (sa16[:, :16, :], sb16[:, :16, :]),
                                (16,),
                                (owner, 0, ring, right, left),
                            )
                            merge_inverse_blocks(
                                (Merge16, Diagonal, Ai32),
                                (sa016, sa16[:, :16, :], sb016, sb16[:, :16, :], sc016),
                                (16,),
                                (ring, owner, owner, 0, left, right),
                            )
                        solve_import(
                            (Ai32, Negative),
                            (sa32[:, :32, :], sb32[:, :32, :]),
                            (32,),
                            (owner, 0, ring, 32, 0),
                        )
                        merge_inverse_blocks(
                            (Merge32, Ai32, Off32),
                            (sa032, sa32[:, :32, :], sb032, sb32[:, :32, :], sc032),
                            (32,),
                            (ring, owner, owner, 0, 0, 32),
                        )
                    with N.Vector() as lane, N.Stage(0, name="transform"):
                        read_inverse_blocks(
                            (Ai32, Off32), inverse_raw, (lane, owner, 0, owner, ring)
                        )
                        transform_panels(
                            (Qp, Kp, Vp, LocalGp, Bp, Aqk, tail_last),
                            (Right, KG, QG, A16),
                            (
                                keys16[0, :, :],
                                queries16[0, :, :],
                                keys,
                                gates[0, :, :],
                                weighted,
                                anchor,
                                betas,
                                rows,
                                inverse,
                                packed_inverse[:16, :],
                            ),
                            (SCALE, tail),
                            (chunk, head, lane, owner),
                        )
                        publish_inverse_blocks(
                            Inverse,
                            (inverse_raw, packed_inverse),
                            (lane, owner, 0, owner, ring),
                        )
                    with N.Cube(), N.Stage(0, name="wu"):
                        wu_import(Inverse, wa1[:, :BT, :], owner, 0, ring, Right, wb1)
                        wu_products(
                            (W16Ring, U),
                            (wa0, wa1[:, :BT, :], wb0, wb1, wc0),
                            (ring, owner, chunk, head),
                        )
                    with N.Vector() as lane, N.Stage(0, name="publish_w"):
                        for panel in N.serial(2):
                            row = lane * 32 + panel * 16
                            N.copy(W16Ring[owner, ring, row : row + 16, :], keys16[0, :, :])
                            N.cast(keys16[0, :, :], out=keys)
                            N.copy(keys, W32[chunk, head, row : row + 16, :])

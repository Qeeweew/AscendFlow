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
    values = N.cast(raw, out=value)
    squares = N.mul(values, values, out=work)
    total = N.reduce_sum(
        squares, axis=1, out=statistic, scratch=partial, order="block8_then_packed_merge"
    )
    stable = N.add(total, 1e-6, out=statistic)
    inverse = N.rsqrt(stable, out=statistic)
    normalized = N.mul(values, inverse[:, None], out=value, scratch=rows)
    return N.cast(normalized, out=raw)


@N.inline
def advance_gate(Gate, staging, cumulative, start, token, valid):
    if token < valid:
        N.copy(Gate[start + token, :], staging)
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
    lane, beta_column = coordinates
    start = N.load_index(Starts)
    valid = N.load_index(Lengths, minimum=1, maximum=BT)
    N.fill(betas, 0.0)
    N.fill(cumulative, 0.0)
    for panel in N.serial(2):
        row = lane * 32 + panel * 16
        N.fill(q16, 0.0)
        N.fill(k16, 0.0)
        N.fill(v16, 0.0)
        N.fill(gates, 0.0)
        if TAIL == 0 or row + 16 <= valid:
            N.copy(Q[start + row : start + row + 16, :], q16)
            N.copy(Key[start + row : start + row + 16, :], k16)
            N.copy(Value[start + row : start + row + 16, :], v16)
            N.copy(Gate[start + row : start + row + 16, :], gates)
            N.copy(Beta[start + row : start + row + 16, :], beta8)
            N.fill(beta8, 0.0, mask=N.arange(8)[None, :] != beta_column)
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
                    N.copy(Q[start + row + token, :], q16[token, :])
                    N.copy(Key[start + row + token, :], k16[token, :])
                    N.copy(Value[start + row + token, :], v16[token, :])
                    N.copy(Gate[start + row + token, :], gates[token, :])
                    coefficient = N.load_scalar(
                        Beta[start + row + token, beta_column : beta_column + 1]
                    )
                    N.store_scalar(coefficient, out=betas[row + token : row + token + 1])
        query_values = normalize_panel(q16, values, work, statistic, partial, rows)
        key_values = normalize_panel(k16, values, work, statistic, partial, rows)
        N.copy(query_values, Qp[row : row + 16, :])
        N.copy(key_values, Kp[row : row + 16, :])
        N.copy(v16, Vp[row : row + 16, :])
        N.copy(gates, Gp[row : row + 16, :])
    N.copy(betas[lane * 32 : lane * 32 + 32], Bp[lane * 32 : lane * 32 + 32])
    if TAIL == 1:
        # Keep the tail's FP32 sequential prefix. Both peers read the original
        # gates; each publishes only its own rows, with no peer-to-peer UB use.
        N.fill(cumulative, 0.0)
        if lane == 1:
            for token in N.serial(32):
                advance_gate(Gate, gates[0, :], cumulative, start, token, valid)
        for local in N.serial(32):
            token = lane * 32 + local
            advance_gate(Gate, gates[0, :], cumulative, start, token, valid)
            N.copy(cumulative, tail_prefix[local, :])
        if lane == 0:
            for local in N.serial(32):
                token = 32 + local
                advance_gate(Gate, gates[0, :], cumulative, start, token, valid)
        N.copy(tail_prefix, Gp[lane * 32 : lane * 32 + 32, :])


@N.inline
def transform_panels(inputs, outputs, buffers, geometry, coordinates):
    Qp, Kp, Vp, Gp, Bp, Aqk, tail_last = inputs
    Right, KG, QG, A16 = outputs
    raw, packed, work, gates, factor, last, beta, rows, af32, af16 = buffers
    SCALE, TAIL = geometry
    (lane,) = coordinates
    if TAIL == 0:
        N.copy(Gp[63, :], last)
    else:
        N.copy(tail_last, last)
    for panel in N.serial(2):
        row = lane * 32 + panel * 16
        N.copy(Gp[row : row + 16, :], gates)
        N.copy(Bp[row : row + 16], beta[:16])
        N.exp(gates, out=factor)
        N.copy(Kp[row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, beta[:16, None], out=work, scratch=rows)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, Right[0, row : row + 16, :])
        N.copy(Vp[row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, beta[:16, None], out=work, scratch=rows)
        N.cast(work, out=packed)
        N.copy(packed, Right[1, row : row + 16, :])
        N.mul(gates, -1.0, out=factor)
        N.add(factor, last[None, :], out=factor)
        N.exp(factor, out=factor)
        N.copy(Kp[row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, KG[row : row + 16, :])
        N.copy(Qp[row : row + 16, :], raw)
        N.cast(raw, out=work)
        N.mul(work, SCALE, out=work)
        N.cast(work, out=packed)
        N.cast(packed, out=work)
        N.exp(gates, out=factor)
        N.mul(work, factor, out=work)
        N.cast(work, out=packed)
        N.copy(packed, QG[row : row + 16, :])
        N.copy(Aqk[row : row + 16, :], af32)
        N.cast(af32, out=af16)
        N.copy(af16, A16[row : row + 16, :])


@N.inline
def packed_gram_products(GramProducts, operands, buffers):
    a1, b1s, b1l = operands
    a0, b0s, b0l, c0 = buffers
    # Four independent GEMMs occupy disjoint columns of one accumulator.
    # Publish their complete word with one FIX, retaining FP32 arithmetic.
    for panel in N.unroll(2):
        aa = N.copy(a1[panel * 32 : panel * 32 + 32, :], a0)
        bb = N.copy(b1s[panel, :, :], b0s, transpose=True)
        N.matmul(aa, bb, accum=c0[:, panel * 16 : panel * 16 + 16], initialize=True)
    bb = N.copy(b1l, b0l, transpose=True)
    for part in N.unroll(2):
        aa = N.copy(a1[64 + part * 32 : 96 + part * 32, :], a0)
        N.matmul(aa, bb, accum=c0[:, 32 + part * 32 : 64 + part * 32], initialize=True)
    N.copy(c0, GramProducts)


@N.inline
def store_packed_gram(inputs, buffers, coordinates):
    GramProducts, Bp, Negative, Aqk = inputs
    scores32, scores16, betas, rows = buffers
    valid, lane = coordinates
    for panel in N.unroll(3):
        if panel < 2:
            start, col, extent = 16 + panel * 32, panel * 32, 16
            scores = scores16[:8, :]
            key_source = GramProducts[lane * 8 : lane * 8 + 8, panel * 16 : panel * 16 + 16]
            query_source = GramProducts[16 + lane * 8 : 24 + lane * 8, panel * 16 : panel * 16 + 16]
        else:
            start, col, extent = 32, 0, 32
            scores = scores32
            key_source = GramProducts[lane * 16 : lane * 16 + 16, 32:64]
            query_source = GramProducts[lane * 16 : lane * 16 + 16, 64:96]
        half = extent // 2
        row = start + lane * half
        # Publish Aqk before the Negative handshake into Cube. Transform
        # later acquires Cube's final inverse result, retaining both AIVs'
        # writes even where Gram row ownership differs from output ownership.
        N.copy(query_source, scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(scores, Aqk[row : row + half, col : col + extent])
        N.copy(key_source, scores)
        if start + extent > valid:
            N.fill(scores, 0.0, mask=N.arange(half)[:, None] + row >= valid)
        N.copy(Bp[row : row + half], betas[:half])
        N.mul(scores, betas[:half, None], out=scores, scratch=rows[:half, :])
        N.mul(scores, -1.0, out=scores)
        N.copy(scores, Negative[row : row + half, col : col + extent])


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
        keys16 = N.alloc_ub((16, K), DTYPE, slots=2)
        queries16 = N.alloc_ub((16, K), DTYPE, slots=2)
        keys = N.alloc_ub((16, K), N.f32)
        queries = N.alloc_ub((16, K), N.f32)
        weighted = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((16, K), N.f32, slots=2)
        decay = N.alloc_ub((16, K), N.f32)
        steps = N.alloc_ub((16, K), N.f32)
        product = N.alloc_ub((32, K), N.f32)
        zeros = N.alloc_ub((BT, BT), N.f32)
        rows = N.alloc_ub((16, 8), N.f32)
        partial = N.alloc_ub((32 * (K // 8),), N.f32)
        scores = N.alloc_ub((64,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        diagonal = N.alloc_ub((16, 16), N.f32)
        attention = N.alloc_ub((16, 16), N.f32, slots=2)
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
        front_a1 = N.alloc_l1((128, K), N.f32, slots=1)
        front_b1 = N.alloc_l1((64, K), N.f32, slots=1)
        lower1 = front_a1.slot(0)[:, :BT]
        lower0 = N.alloc_l0a((BT, BT), N.f32)
        gate1 = front_b1.slot(0)[:, :64]
        gate0 = N.alloc_l0b((BT, 64), N.f32)
        prefix0 = N.alloc_l0c((BT, K), N.f32)
        ga1 = front_a1.slot(0)
        gb1s = N.alloc_l1((2, 16, K), N.f32, slots=1)
        gb1l = N.alloc_l1((32, K), N.f32, slots=1)
        ga0s = N.alloc_l0a((32, K), N.f32)
        gb0s = N.alloc_l0b((K, 16), N.f32)
        gb0l = N.alloc_l0b((K, 32), N.f32)
        gc0l = N.alloc_l0c((32, 96), N.f32)
        sa16 = front_a1.slot(0)[:, :16]
        sb16 = front_b1.slot(0)[:, :16]
        sa32 = front_a1.slot(0)[:, :32]
        sb32 = front_b1.slot(0)[:, :32]
        sa016 = N.alloc_l0a((16, 16), N.f32)
        sb016 = N.alloc_l0b((16, 16), N.f32)
        sc016 = N.alloc_l0c((16, 16), N.f32)
        sa032 = N.alloc_l0a((32, 32), N.f32)
        sb032 = N.alloc_l0b((32, 32), N.f32)
        sc032 = gc0l[:, :32]
        wa1 = N.alloc_l1((128, BT), DTYPE, slots=1)
        wb1 = N.alloc_l1((2, BT, K), DTYPE, slots=1)
        wa0 = N.alloc_l0a((BT, BT), DTYPE)
        wb0 = N.alloc_l0b((BT, K), DTYPE)
        wc0 = prefix0
        # Reuse the original fixed workspace planes across full/tail phases.
        q_values = N.slotted_view(Qp[owner, :, :, :]).slot(0)
        k_values = N.slotted_view(Kp[owner, :, :, :]).slot(0)
        v_values = N.slotted_view(Vp[owner, :, :, :]).slot(0)
        gate_values = N.slotted_view(LocalGp[owner, :, :, :]).slot(0)
        beta_values = N.slotted_view(Bp[owner, :, :]).slot(0)
        attention_values = N.slotted_view(Aqk[owner, :, :, :]).slot(0)
        diagonal_values = N.slotted_view(Diagonal[owner, :, :, :]).slot(0)
        inverse32_values = N.slotted_view(Ai32[owner, :, :, :]).slot(0)
        negative_values = N.slotted_view(Negative[owner, :, :, :]).slot(0)
        inverse_values = N.slotted_view(Inverse[owner, :, :, :]).slot(0)
        right_values = N.slotted_view(Right[owner, :, :, :, :]).slot(0)
        gram_left = N.slotted_view(GramLeft[owner, :, :, :, :]).slot(0)[0, :, :]
        gram_right = N.slotted_view(GramRight[owner, :, :, :, :]).slot(0)[0, :, :]
        gram_products = N.slotted_view(GramProducts[owner, :, :, :]).slot(0)
        merge16 = N.slotted_view(Merge16[owner, :, :, :]).slot(0)
        merge32 = N.slotted_view(Merge32[owner, :, :, :]).slot(0)
        off32 = N.slotted_view(Off32[owner, :, :, :]).slot(0)
        weights = N.slotted_view(W16Ring[owner, :, :, :]).slot(0)
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
                    ),
                    issue_order={
                        "cube": ("prefix", "cross_gram", "solve", "wu"),
                        "vector": ("inputs", "diagonal", "gram_store", "transform", "publish_w"),
                    },
                ):
                    work = owner + task * GROUPS
                    head = work % H
                    if tail == 1:
                        chunk = (work // H) * NC + NC - 1
                    else:
                        full = work // H
                        chunk = (full // (NC - 1)) * NC + full % (NC - 1)
                    with N.Vector() as lane, N.Stage(0, name="inputs"):
                        beta_head = (head // 8) * 8
                        prepare_inputs(
                            (
                                Q[:, head, :],
                                Key[:, head, :],
                                Value[:, head, :],
                                Gate[:, head, :],
                                Beta[:, beta_head : beta_head + 8],
                                Starts[chunk : chunk + 1],
                                Lengths[chunk : chunk + 1],
                            ),
                            (q_values, k_values, v_values, gate_values, beta_values),
                            (
                                queries16.slot(0),
                                keys16.slot(0),
                                keys16.slot(1),
                                keys,
                                weighted,
                                scores[:16],
                                partial[: 16 * (K // 8)],
                                rows,
                                inverse_rows[:16, :],
                                betas,
                                gates.slot(0),
                                anchor,
                                tail_prefix,
                            ),
                            (BT, tail),
                            (lane, head % 8),
                        )
                        N.copy(anchor, tail_last)
                    with N.Cube(), N.Stage(0, name="prefix"):
                        lower_values = N.copy(Lower[tail, :, :], lower1[:BT, :])
                        lower_operand = N.copy(lower_values, lower0)
                        for part in N.unroll(2):
                            loaded_gate = N.copy(gate_values[:, part * 64 : part * 64 + 64], gate1)
                            gg = N.copy(loaded_gate, gate0)
                            N.matmul(
                                lower_operand,
                                gg,
                                accum=prefix0[:, part * 64 : part * 64 + 64],
                                initialize=True,
                            )
                        N.copy(prefix0, gate_values)
                        N.copy(prefix0, Gp[chunk, head, :, :])
                    with N.Vector() as lane, N.Stage(0, name="diagonal"):
                        valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                        N.fill(zeros, 0.0)
                        N.copy(zeros[:32, :], attention_values[lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], diagonal_values[lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], inverse32_values[lane * 32 : lane * 32 + 32, :])
                        N.copy(zeros[:32, :], negative_values[lane * 32 : lane * 32 + 32, :])
                        N.fill(inverse, 0.0)
                        upper = N.materialize(
                            N.arange(16)[:, None] <= N.arange(16)[None, :], out=upper_storage
                        )
                        strict_upper = N.materialize(
                            N.arange(16)[:, None] < N.arange(16)[None, :], out=strict_upper_storage
                        )
                        for local in N.serial(2):
                            block = lane * 32 + local * 16
                            N.copy(k_values[block : block + 16, :], keys16.slot(0))
                            N.copy(q_values[block : block + 16, :], queries16.slot(0))
                            N.copy(gate_values[block : block + 16, :], gates.slot(0))
                            diagonal_gram(
                                (
                                    keys16.slot(0),
                                    keys,
                                    betas[block : block + 16],
                                    weighted,
                                    rows,
                                    queries16.slot(0),
                                    queries,
                                    steps,
                                    gates.slot(0),
                                    decay,
                                    diagonal,
                                    attention.slot(0),
                                    product,
                                    scores,
                                    partial,
                                    upper,
                                    mask_zero,
                                    strict_upper,
                                    inverse,
                                ),
                                (SCALE, K),
                                (block, valid),
                            )
                            N.copy(
                                attention.slot(0),
                                attention_values[block : block + 16, block : block + 16],
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
                                inverse32_values[block : block + 16, block : block + 16],
                            )
                            N.copy(
                                inverse[:, block : block + 16],
                                diagonal_values[block : block + 16, block : block + 16],
                            )
                        prepare_one_gram(
                            (
                                gate_values,
                                k_values,
                                gram_left,
                                q_values,
                                gram_right,
                            ),
                            (
                                anchor,
                                gates.slot(0),
                                weighted,
                                keys16.slot(0),
                                keys,
                                queries,
                                steps,
                                decay,
                            ),
                            (SCALE,),
                            (lane,),
                        )
                    with N.Cube(), N.Stage(0, name="cross_gram"):
                        left_value, right_value = import_gram_item(
                            (gram_left, gram_right),
                            (ga1, gb1s.slot(0), gb1l.slot(0)),
                        )
                        packed_gram_products(
                            gram_products,
                            (left_value, gb1s.slot(0), right_value),
                            (ga0s, gb0s, gb0l, gc0l),
                        )
                    with N.Vector() as lane, N.Stage(0, name="gram_store"):
                        valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                        # Reuse an already retired inverse/identity scratch for scores.
                        store_packed_gram(
                            (gram_products, beta_values, negative_values, attention_values),
                            (cross_scores, diagonal, betas[:16], rows),
                            (valid, lane),
                        )
                    with N.Cube(), N.Stage(0, name="solve"):
                        for pair in N.unroll(2):
                            left, right = pair * 32, pair * 32 + 16
                            diagonal_value, negative_value = solve_import(
                                (diagonal_values, negative_values),
                                (sa16[:16, :], sb16[:16, :]),
                                (16,),
                                (right, left),
                            )
                            merge_inverse_blocks(
                                (
                                    merge16,
                                    diagonal_values[left : left + 16, left : left + 16],
                                    inverse32_values[right : right + 16, left : left + 16],
                                ),
                                (diagonal_value, negative_value),
                                (sa016, sa16[:16, :], sb016, sb16[:16, :], sc016),
                            )
                        diagonal_value, negative_value = solve_import(
                            (inverse32_values, negative_values),
                            (sa32[:32, :], sb32[:32, :]),
                            (32,),
                            (32, 0),
                        )
                        merge_inverse_blocks(
                            (
                                merge32,
                                inverse32_values[:32, :32],
                                off32,
                            ),
                            (diagonal_value, negative_value),
                            (sa032, sa32[:32, :], sb032, sb32[:32, :], sc032),
                        )
                    with N.Vector() as lane, N.Stage(0, name="transform"):
                        read_inverse_blocks((inverse32_values, off32), inverse_raw, (lane,))
                        transform_panels(
                            (
                                q_values,
                                k_values,
                                v_values,
                                gate_values,
                                beta_values,
                                attention_values,
                                tail_last,
                            ),
                            (
                                right_values,
                                KG[chunk, head, :, :],
                                QG[chunk, head, :, :],
                                A16[chunk, head, :, :],
                            ),
                            (
                                keys16.slot(0),
                                queries16.slot(0),
                                keys,
                                gates.slot(0),
                                weighted,
                                anchor,
                                betas,
                                rows,
                                inverse,
                                packed_inverse[:16, :],
                            ),
                            (SCALE, tail),
                            (lane,),
                        )
                        publish_inverse_blocks(
                            inverse_values,
                            (inverse_raw, packed_inverse),
                            (lane,),
                        )
                    with N.Cube(), N.Stage(0, name="wu"):
                        inverse_value, weight_value, rhs_value = wu_import(
                            (inverse_values, right_values),
                            (wa1.slot(0)[:BT, :], wb1.slot(0)),
                        )
                        wu_products(
                            (weights, U[chunk, head, :, :]),
                            (inverse_value, weight_value, rhs_value),
                            (wa0, wb0, wc0),
                        )
                    with N.Vector() as lane, N.Stage(0, name="publish_w"):
                        for panel in N.serial(2):
                            row = lane * 32 + panel * 16
                            N.copy(weights[row : row + 16, :], keys16.slot(0))
                            N.cast(keys16.slot(0), out=keys)
                            N.copy(keys, W32[chunk, head, row : row + 16, :])

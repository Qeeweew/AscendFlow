"""Prepare one 64-token chunk/head on one Cube and its two Vector peers.

The causal matrix has four 16-token diagonal blocks. Each Vector peer owns
two diagonal blocks. Cube computes the two 16x16 off-diagonal blocks and
the one 32x32 off-diagonal block, then merges inverses 16 -> 32 -> 64.
Cross-Gram and diagonal work are independent after the gate prefix.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.kda.prefill_gram import owned_diagonal_inverse
from examples.kda.prefill_gram_cross import (
    packed_gram_products,
    prepare_owned_gram,
    store_packed_gram,
)
from examples.kda.prefill_solve import (
    publish_inverse_blocks,
    read_inverse_blocks,
    solve_chunk_inverse,
)
from examples.kda.prefill_transform import form_transform_inputs, publish_attention
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
def prefetch_gate_panel(Gate, staging, start, row, valid):
    if row + 16 <= valid:
        N.copy(Gate[start + row : start + row + 16, :], staging)
    else:
        N.fill(staging, 0.0)
        for token in N.serial(16):
            if row + token < valid:
                N.copy(Gate[start + row + token, :], staging[token, :])


@N.inline
def prepare_inputs(inputs, outputs, buffers, geometry, coordinates):
    Q, Key, Value, Gate, Beta, Starts, Lengths = inputs
    Qp, Kp, Vp, Gp, Bp = outputs
    (
        raw_q,
        raw_k,
        raw_v,
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
    BT, task, tail_start = geometry
    lane, beta_column = coordinates
    start = N.load_index(Starts)
    valid = N.load_index(Lengths, minimum=1, maximum=BT)
    N.fill(betas, 0.0)
    N.fill(cumulative, 0.0)
    row = lane * 32
    complete_rows = task < tail_start or row + 32 <= valid
    # Prefix is independent of Q/K/V loading and normalization. Put its
    # DMA/publication first so Cube can run during both input operations.
    if task < tail_start:
        N.copy(Gate[start + row : start + row + 32, :], gates)
        N.copy(gates, Gp[row : row + 32, :])
    else:
        # Keep the tail's FP32 sequential prefix. Both peers read the original
        # gates; each publishes only its own rows, with no peer-to-peer UB use.
        N.fill(cumulative, 0.0)
        N.fill(tail_prefix, 0.0)
        for panel in N.serial(4):
            prefetch_gate_panel(Gate, gates[:16, :], start, panel * 16, valid)
            for local in N.serial(16):
                N.add(cumulative, gates[local, :], out=cumulative)
                if panel // 2 == lane:
                    N.copy(cumulative, tail_prefix[(panel % 2) * 16 + local, :])
        N.copy(tail_prefix, Gp[lane * 32 : lane * 32 + 32, :])

    if complete_rows:
        with N.CopyGroup(name="input_rows"):
            N.copy(Q[start + row : start + row + 32, :], raw_q)
            N.copy(Key[start + row : start + row + 32, :], raw_k)
            N.copy(Value[start + row : start + row + 32, :], raw_v)
            N.copy(Beta[start + row : start + row + 32, :], beta8)
    else:
        N.fill(raw_q, 0.0)
        N.fill(raw_k, 0.0)
        N.fill(raw_v, 0.0)
        # Retain complete 16-row tail DMAs; only the final incomplete
        # subpanel uses guarded rows, never an out-of-bounds 32-row read.
        for panel in N.serial(2):
            offset = panel * 16
            if row + offset + 16 <= valid:
                with N.CopyGroup(name="input_tail_panel"):
                    N.copy(
                        Q[start + row + offset : start + row + offset + 16, :],
                        raw_q[offset : offset + 16, :],
                    )
                    N.copy(
                        Key[start + row + offset : start + row + offset + 16, :],
                        raw_k[offset : offset + 16, :],
                    )
                    N.copy(
                        Value[start + row + offset : start + row + offset + 16, :],
                        raw_v[offset : offset + 16, :],
                    )
                    N.copy(Beta[start + row + offset : start + row + offset + 16, :], beta8[:16, :])
                N.fill(beta8[:16, :], 0.0, mask=N.arange(8)[None, :] != beta_column)
                N.reduce_sum(
                    beta8[:16, :],
                    axis=1,
                    out=betas[row + offset : row + offset + 16],
                    scratch=partial[:16],
                    order="block8_then_packed_merge",
                )
            else:
                for token in N.serial(16):
                    local = offset + token
                    if row + local < valid:
                        with N.CopyGroup(name="input_tail_row"):
                            N.copy(Q[start + row + local, :], raw_q[local, :])
                            N.copy(Key[start + row + local, :], raw_k[local, :])
                            N.copy(Value[start + row + local, :], raw_v[local, :])
                        coefficient = N.load_scalar(
                            Beta[start + row + local, beta_column : beta_column + 1]
                        )
                        N.store_scalar(coefficient, out=betas[row + local : row + local + 1])

    if complete_rows:
        N.fill(beta8, 0.0, mask=N.arange(8)[None, :] != beta_column)
        N.reduce_sum(
            beta8,
            axis=1,
            out=betas[row : row + 32],
            scratch=partial[:32],
            order="block8_then_packed_merge",
        )
    query_values = normalize_panel(raw_q, values, work, statistic, partial, rows)
    key_values = normalize_panel(raw_k, values, work, statistic, partial, rows)
    with N.CopyGroup(name="prepared_rows"):
        N.copy(query_values, Qp[row : row + 32, :])
        N.copy(key_values, Kp[row : row + 32, :])
        N.copy(raw_v, Vp[row : row + 32, :])
    N.copy(betas[lane * 32 : lane * 32 + 32], Bp[lane * 32 : lane * 32 + 32])


@N.prim_func(disjoint_external=True)
def prefill_prepare(
    Q: N.Tensor((TOKENS, H, K), DTYPE),
    Key: N.Tensor((TOKENS, H, K), DTYPE),
    Value: N.Tensor((TOKENS, H, K), DTYPE),
    Gate: N.Tensor((TOKENS, H, K), N.f32),
    Beta: N.Tensor((TOKENS, H), N.f32),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    KG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE, access="write", write_disjoint=True),
    U: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
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
        Qp = N.alloc_workspace((GROUPS, 1, BT, K), DTYPE, name="Qp")
        Kp = N.alloc_workspace((GROUPS, 1, BT, K), DTYPE, name="Kp")
        Vp = N.alloc_workspace((GROUPS, 1, BT, K), DTYPE, name="Vp")
        LocalGp = N.alloc_workspace((GROUPS, 1, BT, K), N.f32, name="LocalGp")
        Bp = N.alloc_workspace((GROUPS, 1, BT), N.f32, name="Bp")
        Aqk = N.alloc_workspace((GROUPS, 1, BT, BT), N.f32, name="Aqk")
        Diagonal = N.alloc_workspace((GROUPS, 1, BT, BT), N.f32, name="Diagonal")
        Ai32 = N.alloc_workspace((GROUPS, 1, BT, BT), N.f32, name="Ai32")
        Negative = N.alloc_workspace((GROUPS, 1, BT, BT), N.f32, name="Negative")
        Inverse = N.alloc_workspace((GROUPS, 1, BT, BT), DTYPE, name="Inverse")
        Right = N.alloc_workspace((GROUPS, 1, 2, BT, K), DTYPE, name="Right")
        GramLeft = N.alloc_workspace((GROUPS, 2, 1, 128, K), N.f32, name="GramLeft")
        GramRight = N.alloc_workspace((GROUPS, 2, 1, 64, K), N.f32, name="GramRight")
        GramProducts = N.alloc_workspace((GROUPS, 2, 32, 96), N.f32, name="GramProducts")
        Merge16 = N.alloc_workspace((GROUPS, 2, 16, 16), N.f32, name="Merge16")
        Merge32 = N.alloc_workspace((GROUPS, 2, 32, 32), N.f32, name="Merge32")
        Off32 = N.alloc_workspace((GROUPS, 2, 32, 32), N.f32, name="Off32")
        W16Ring = N.alloc_workspace((GROUPS, 2, BT, K), DTYPE, name="W16Ring")
        assert CHUNKS == SEQUENCES * NC and NC > 0, "equal nonempty chunk domains"
        assert H % 8 == 0 and K == 128 and BT == 64, "KDA front-end tile policy"
        raw_keys = N.alloc_ub((32, K), DTYPE)
        raw_queries = N.alloc_ub((32, K), DTYPE)
        raw_values = N.alloc_ub((32, K), DTYPE)
        # W publication is two ticks behind input loading. Its BF16 reader
        # must own storage independent of all three input planes.
        raw_weights = N.alloc_ub((32, K), DTYPE)
        keys = N.alloc_ub((16, K), N.f32)
        queries = N.alloc_ub((16, K), N.f32)
        weighted = N.alloc_ub((32, K), N.f32)
        gate_panels = N.alloc_ub((32, K), N.f32)
        decay = N.alloc_ub((16, K), N.f32)
        steps = N.alloc_ub((16, K), N.f32)
        product = N.alloc_ub((32, K), N.f32)
        # Successive lifetimes: zero publication, coefficient expansion,
        # then A16 import. Gram's real MTE3 retirement separates the first
        # two; inverse acquisition separates the latter two.
        plane_scratch = N.alloc_ub((BT // 2, BT), N.f32)
        rows = N.alloc_ub((32, 8), N.f32)
        partial = N.alloc_ub((32 * (K // 8),), N.f32)
        scores = N.alloc_ub((64,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        diagonal = N.alloc_ub((16, 16), N.f32)
        attention = N.alloc_ub((16, 16), N.f32, slots=1)
        inverse = N.alloc_ub((16, 32), N.f32)
        coefficients = N.alloc_ub((32,), N.f32)
        inverse_rows = N.alloc_ub((32, 8), N.f32)
        coefficient_matrix = N.alloc_ub((16, 32), N.f32)
        weighted_inverse = N.alloc_ub((16, 32), N.f32)
        inverse_tree = N.alloc_ub((8, 32), N.f32)
        identities = N.alloc_ub((16, 32), N.f32)
        upper_storage = N.alloc_mask((16, 16))
        strict_upper_storage = N.alloc_mask((16, 16))
        mask_zero = N.alloc_ub((16, 16), N.f32)
        anchor = N.alloc_ub((K,), N.f32)
        tail_last = N.alloc_ub((K,), N.f32)
        tail_prefix = N.alloc_ub((32, K), N.f32)
        packed_inverse = N.alloc_ub((32, BT), DTYPE)
        inverse_raw = N.alloc_ub((32, BT), N.f32)
        cross_scores = N.alloc_ub((16, 32), N.f32)
        # This work plane follows normalization, then Gram publication.
        # Its MTE3 readers retire before diagonal work; delayed W publication
        # uses product instead and never holds this plane across input tasks.
        gram_keys = N.alloc_ub((32, K), N.f32)
        # Different subphases reuse UB and native storage. The gate prefix
        # uses two 64-column passes to keep L0A/B within their 64 KiB banks.
        shared_left_l1 = N.alloc_l1((128, K), N.f32, slots=1)
        shared_right_l1 = N.alloc_l1((64, K), N.f32, slots=1)
        prefix_lower_l1 = shared_left_l1.slot(0)[:, :BT]
        prefix_lower_l0 = N.alloc_l0a((BT, BT), N.f32)
        prefix_gate_l1 = shared_right_l1.slot(0)
        prefix_gate_l0 = N.alloc_l0b((BT, 64), N.f32)
        shared_results_l0 = N.alloc_l0c((BT, K), N.f32)
        gram_left_l1 = shared_left_l1.slot(0)
        gram_small_right_l1 = N.alloc_l1((2, 16, K), N.f32, slots=1)
        gram_large_right_l1 = N.alloc_l1((32, K), N.f32, slots=1)
        gram_left_l0 = N.alloc_l0a((32, K), N.f32)
        gram_small_right_l0 = N.alloc_l0b((K, 16), N.f32)
        gram_large_right_l0 = N.alloc_l0b((K, 32), N.f32)
        gram_products_l0 = N.alloc_l0c((32, 96), N.f32)
        merge16_left_l0 = N.alloc_l0a((16, 16), N.f32)
        merge16_right_l0 = N.alloc_l0b((16, 16), N.f32)
        merge16_result_l0 = N.alloc_l0c((16, 16), N.f32)
        merge32_left_l0 = N.alloc_l0a((32, 32), N.f32)
        merge32_right_l0 = N.alloc_l0b((32, 32), N.f32)
        merge32_result_l0 = gram_products_l0[:, :32]
        wu_inverse_l1 = N.alloc_l1((128, BT), DTYPE, slots=1)
        wu_right_l1 = N.alloc_l1((2, BT, K), DTYPE, slots=1)
        wu_inverse_l0 = N.alloc_l0a((BT, BT), DTYPE)
        wu_right_l0 = N.alloc_l0b((BT, K), DTYPE)
        wu_result_l0 = shared_results_l0
        # Input/transform phases share one plane; only W publication spans tasks.
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
        weights = N.slotted_view(W16Ring[owner, :, :, :])
        with N.Vector():
            upper = N.materialize(N.arange(16)[:, None] <= N.arange(16)[None, :], out=upper_storage)
            strict_upper = N.materialize(
                N.arange(16)[:, None] < N.arange(16)[None, :], out=strict_upper_storage
            )
            N.fill(scores, 0.0)
            N.fill(diagonal, 0.0)
            N.fill(diagonal, 1.0, mask=N.arange(16)[:, None] == N.arange(16)[None, :])
            for plane in N.unroll(2):
                N.copy(diagonal, identities[:, plane * 16 : plane * 16 + 16])
            N.fill(product, 0.0)
        # Process all non-tail chunks, then each sequence's tail, in one
        # work domain. Only input padding and prefix arithmetic differ.
        # After transform(i), inputs(i+1) can overlap Cube W/U(i). Publish W
        # one tick later through the existing two workspace planes.
        full_work = SEQUENCES * (NC - 1) * H
        tail_start = N.ceildiv(N.max(0, full_work - owner), GROUPS)
        for task in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS),
            issue_order={
                "cube": ("prefix", "cross_gram", "solve", "wu"),
                "vector": (
                    "publish_w",
                    "gram_operands",
                    "diagonal",
                    "gram_store",
                    "transform",
                    "inputs",
                ),
            },
        ):
            work = owner + task * GROUPS
            head = work % H
            if NC == 1:
                chunk = work // H
            else:
                full = work // H
                chunk = (
                    (full // (NC - 1)) * NC + full % (NC - 1)
                    if task < tail_start
                    else ((work - full_work) // H) * NC + NC - 1
                )
            # Triangular Cube prefix for full chunks; identity publication
            # preserves the sequential FP32 Vector prefix for sequence tails.
            tail = 0 if task < tail_start else 1
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
                        raw_queries,
                        raw_keys,
                        raw_values,
                        gram_keys,
                        weighted,
                        scores[:32],
                        partial,
                        rows,
                        inverse_rows,
                        betas,
                        gate_panels,
                        anchor,
                        tail_prefix,
                    ),
                    (BT, task, tail_start),
                    (lane, head % 8),
                )
                N.copy(anchor, tail_last)
            with N.Cube(), N.Stage(1, name="prefix"):
                with N.CopyGroup(name="prefix_operands"):
                    lower_values = N.copy(Lower[tail, :, :], prefix_lower_l1[:BT, :])
                    loaded_gate = N.copy(gate_values, prefix_gate_l1)
                lower_operand = N.copy(lower_values, prefix_lower_l0)
                for part in N.unroll(2):
                    gg = N.copy(loaded_gate[:, part * 64 : part * 64 + 64], prefix_gate_l0)
                    N.matmul(
                        lower_operand,
                        gg,
                        accum=shared_results_l0[:, part * 64 : part * 64 + 64],
                        initialize=True,
                    )
                N.copy(shared_results_l0, gate_values)
                published_gate = N.copy(shared_results_l0, Gp[chunk, head, :, :])
            with N.Vector() as lane, N.Stage(1, name="gram_operands"):
                N.fill(plane_scratch, 0.0)
                N.copy(plane_scratch, attention_values[lane * 32 : lane * 32 + 32, :])
                N.copy(plane_scratch, inverse32_values[lane * 32 : lane * 32 + 32, :])
                # Diagonal is read only at its four fully written 16x16
                # blocks. Negative is read only at the three fully written
                # cross blocks. Neither needs a full-plane zero publication.
                # Clear before publishing G: Gram's row partition differs
                # from the two peers' 32-row ownership of the cleared planes.
                prepare_owned_gram(
                    Gp=gate_values,
                    Kp=k_values,
                    Qp=q_values,
                    Left=gram_left,
                    Right=gram_right,
                    anchor=anchor,
                    factors=gate_panels,
                    raw=raw_keys,
                    keys=gram_keys,
                    queries=tail_prefix,
                    SCALE=SCALE,
                    lane=lane,
                )
            with N.Vector() as lane, N.Stage(1, name="diagonal"):
                owned_diagonal_inverse(
                    inputs=(k_values, q_values, gate_values, Lengths[chunk : chunk + 1], betas),
                    outputs=(attention_values, inverse32_values, diagonal_values),
                    panels=(raw_keys, raw_queries, gate_panels),
                    gram_work=(
                        keys,
                        queries,
                        weighted,
                        rows,
                        steps,
                        decay,
                        diagonal,
                        attention,
                        product,
                        scores,
                        partial,
                    ),
                    inverse_work=(
                        inverse,
                        coefficients,
                        coefficient_matrix,
                        plane_scratch,
                        inverse_rows,
                        weighted_inverse,
                        inverse_tree,
                        identities,
                    ),
                    masks=(upper, mask_zero, strict_upper),
                    geometry=(SCALE, K, BT),
                    lane=lane,
                )
            with N.Cube(), N.Stage(1, name="cross_gram"):
                # Both peers publish their last Left rows after all Right
                # operands. Importing the full Left slab captures the quorum.
                with N.CopyGroup(name="gram_operands"):
                    left_value = N.copy(gram_left, gram_left_l1)
                    right_value = N.copy(gram_right[32:64, :], gram_large_right_l1.slot(0))
                    for panel in N.unroll(2):
                        N.copy(
                            gram_right[panel * 16 : panel * 16 + 16, :],
                            gram_small_right_l1.slot(0)[panel, :, :],
                        )
                packed_gram_products(
                    GramProducts=gram_products,
                    a1=left_value,
                    b1s=gram_small_right_l1.slot(0),
                    b1l=right_value,
                    a0=gram_left_l0,
                    b0s=gram_small_right_l0,
                    b0l=gram_large_right_l0,
                    c0=gram_products_l0,
                )
            with N.Vector() as lane, N.Stage(1, name="gram_store"):
                valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
                store_packed_gram(
                    GramProducts=gram_products,
                    Bp=beta_values,
                    Negative=negative_values,
                    Aqk=attention_values,
                    scores32=cross_scores,
                    scores16=diagonal,
                    betas=betas[:16],
                    rows=rows,
                    valid=valid,
                    lane=lane,
                )
            with N.Cube(), N.Stage(1, name="solve"):
                solve_chunk_inverse(
                    Diagonal=diagonal_values,
                    Negative=negative_values,
                    Ai32=inverse32_values,
                    Merge16=merge16,
                    Merge32=merge32,
                    Off32=off32,
                    a1=shared_left_l1.slot(0),
                    b1=shared_right_l1.slot(0),
                    cube16=(merge16_left_l0, merge16_right_l0, merge16_result_l0),
                    cube32=(merge32_left_l0, merge32_right_l0, merge32_result_l0),
                )
            with N.Vector() as lane, N.Stage(1, name="transform"):
                # RHS/KG/QG depend on the prepared inputs, so their Vector
                # work runs while Cube merges the inverse. Only Aqk's peer
                # visibility and final inverse packing require that acquire.
                form_transform_inputs(
                    (
                        q_values,
                        k_values,
                        v_values,
                        gate_values,
                        beta_values,
                        tail_last,
                        published_gate[63, :],
                    ),
                    (
                        right_values,
                        KG[chunk, head, :, :],
                        QG[chunk, head, :, :],
                    ),
                    (
                        raw_keys,
                        raw_queries,
                        tail_prefix,
                        gate_panels,
                        weighted,
                        product,
                        anchor,
                        betas,
                        rows,
                    ),
                    (SCALE, task, tail_start),
                    (lane,),
                )
                read_inverse_blocks((inverse32_values, off32), inverse_raw, (lane,))
                publish_attention(
                    Aqk=attention_values,
                    A16=A16[chunk, head, :, :],
                    raw=plane_scratch,
                    packed=packed_inverse,
                    lane=lane,
                )
                publish_inverse_blocks(
                    inverse_values,
                    (inverse_raw, packed_inverse),
                    (lane,),
                )
            with N.Cube(), N.Stage(1, name="wu"):
                inverse_value, weight_value, rhs_value = wu_import(
                    (inverse_values, right_values),
                    (wu_inverse_l1.slot(0)[:BT, :], wu_right_l1.slot(0)),
                )
                wu_products(
                    (weights.at(task), U[chunk, head, :, :]),
                    (inverse_value, weight_value, rhs_value),
                    (wu_inverse_l0, wu_right_l0, wu_result_l0),
                )
            with N.Vector() as lane, N.Stage(2, name="publish_w"):
                row = lane * 32
                N.copy(weights.at(task)[row : row + 32, :], raw_weights)
                N.cast(raw_weights, out=product)
                N.copy(product, W32[chunk, head, row : row + 32, :])

"""FP32 diagonal Gram blocks and their causal inverses."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def diagonal_gram(buffers, geometry, coordinates):
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
    ) = buffers
    SCALE, K = geometry
    block, valid = coordinates
    key_values = N.cast(keys16, out=keys)
    N.mul(key_values, betas[:, None], out=weighted, scratch=rows)
    N.cast(queries16, out=queries)
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
        panel = N.ceildiv(extent, 8) * 8
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
            N.mul(
                decay[:extent, :],
                query_row[None, :],
                out=product[panel : panel + extent, :],
            )
            N.reduce_sum(
                product[: 2 * panel, :],
                axis=1,
                out=scores[: 2 * panel],
                scratch=partial[: 2 * panel * (K // 8)],
                order="block8_then_packed_merge",
            )
            N.copy(scores[:16], diagonal[local, :])
            N.mul(
                scores[panel : panel + 16],
                SCALE,
                out=scores[panel : panel + 16],
            )
            N.copy(scores[panel : panel + 16], attention[local, :])
    N.fill(diagonal, 0.0, mask=upper, scratch=mask_zero)
    result = N.fill(attention, 0.0, mask=strict_upper, scratch=mask_zero)
    N.copy(diagonal, inverse[:, block : block + 16])
    return result


@N.inline
def invert_diagonal(buffers, geometry):
    (
        inverse,
        coefficients,
        coefficient_matrix,
        expanded,
        inverse_rows,
        weighted_inverse,
        inverse_tree,
        identities,
    ) = buffers
    (BT,) = geometry
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
    return N.add(inverse, identities, out=inverse)


@N.inline
def diagonal_inverse(inputs, buffers, geometry):
    Ai32, Diagonal = inputs
    (BT,) = geometry
    inverted = invert_diagonal(buffers, geometry)
    for plane in N.unroll(BT // 16):
        block = plane * 16
        N.copy(
            inverted[:, block : block + 16],
            Ai32[block : block + 16, block : block + 16],
        )
        N.copy(
            inverted[:, block : block + 16],
            Diagonal[block : block + 16, block : block + 16],
        )


@N.prim_func(disjoint_external=True)
def prefill_gram(
    Qp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Kp: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Bp: N.Tensor((CHUNKS, H, BT), N.f32),
    # Full-plane clears overlap the later diagonal-block updates.
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write"),
    Diagonal: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write"),
    Ai32: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write"),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.bf16,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        assert K == 128 and BT == 64, "four diagonal blocks share 64-wide Vector rows"
        keys16 = N.alloc_ub((16, K), DTYPE, slots=2)
        queries16 = N.alloc_ub((16, K), DTYPE, slots=2)
        keys = N.alloc_ub((16, K), N.f32)
        queries = N.alloc_ub((16, K), N.f32)
        weighted = N.alloc_ub((16, K), N.f32)
        gates = N.alloc_ub((16, K), N.f32, slots=2)
        decay = N.alloc_ub((16, K), N.f32)
        steps = N.alloc_ub((16, K), N.f32)
        # K and Q products occupy independent panels so both can be issued
        # before one batched reduction with the unchanged per-score sum tree.
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
        N.fill(zeros, 0.0)
        # Packed Q scores start on a full 32-byte block. Padding rows are
        # defined here and their scores are removed by the causal masks.
        N.fill(product, 0.0)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            valid = N.load_index(Lengths[chunk : chunk + 1])
            beta_values = N.copy(Bp[chunk, head, :], betas)
            N.copy(zeros, Aqk[chunk, head, :, :])
            N.copy(zeros, Diagonal[chunk, head, :, :])
            N.copy(zeros, Ai32[chunk, head, :, :])
            for block_index in N.Pipelined(BT // 16, issue_order=("prefetch", "store", "gram")):
                block = block_index * 16
                with N.Stage(0, name="prefetch"), N.CopyGroup(name="diagonal_operands"):
                    key_values = N.copy(
                        Kp[chunk, head, block : block + 16, :], keys16.at(block_index)
                    )
                    query_values = N.copy(
                        Qp[chunk, head, block : block + 16, :], queries16.at(block_index)
                    )
                    gate_values = N.copy(
                        Gp[chunk, head, block : block + 16, :], gates.at(block_index)
                    )
                with N.Stage(1, name="gram"):
                    result = diagonal_gram(
                        (
                            key_values,
                            keys,
                            beta_values[block : block + 16],
                            weighted,
                            rows,
                            query_values,
                            queries,
                            steps,
                            gate_values,
                            decay,
                            diagonal,
                            attention.at(block_index),
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
                with N.Stage(2, name="store"):
                    N.copy(
                        result,
                        Aqk[chunk, head, block : block + 16, block : block + 16],
                    )
            # Four independent 16x16 blocks share one row recurrence. The
            # 64-wide row layout keeps the multiply and every sum-tree level
            # on complete Vector repeats, preserving each block's FP32 tree.
            diagonal_inverse(
                (Ai32[chunk, head, :, :], Diagonal[chunk, head, :, :]),
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

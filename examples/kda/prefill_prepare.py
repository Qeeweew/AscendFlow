"""Overlap FP32 Cube gate prefixes with Vector normalization and bulk beta DMA."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def prepare_gate_load(Starts, chunk, BT, Gate, gates1, head, ring):
    start = N.load_index(Starts[chunk : chunk + 1])
    N.copy(Gate[start : start + BT, head, :], gates1[ring, :, :])


@N.inline
def prepare_input_load(inputs, buffers, geometry, coordinates):
    Starts, Q, Key, Value, Beta = inputs
    q16, k16, v16, beta8 = buffers
    (BT,) = geometry
    (chunk, lane, head, ring) = coordinates
    start = N.load_index(Starts[chunk : chunk + 1])
    row = lane * (BT // 2)
    N.copy(Q[start + row : start + row + BT // 2, head, :], q16[ring, :, :])
    N.copy(Key[start + row : start + row + BT // 2, head, :], k16[ring, :, :])
    N.copy(Value[start + row : start + row + BT // 2, head, :], v16[ring, :, :])
    beta_head = (head // 8) * 8
    N.copy(
        Beta[start + row : start + row + BT // 2, beta_head : beta_head + 8],
        beta8[ring, :, :],
    )


@N.inline
def prepare_gate_prefix(gates0, gates1, ring, lower0, prefixes0, Gp, chunk, head):
    gg = N.copy(gates1[ring, :, :], gates0)
    N.matmul(lower0, gg, accum=prefixes0, initialize=True)
    N.copy(prefixes0, Gp[chunk, head, :, :])


@N.inline
def prepare_normalize(buffers, coordinates):
    (
        q16,
        values,
        work,
        stat,
        partial,
        rows,
        k16,
        selectors,
        beta_mask,
        beta8,
        beta_zero,
        beta_partial,
        betas,
    ) = buffers
    ring, head = coordinates
    N.cast(q16[ring, :, :], out=values)
    N.mul(values, values, out=work)
    N.reduce_sum(
        work,
        axis=1,
        out=stat,
        scratch=partial,
        order="block8_then_packed_merge",
    )
    N.add(stat, 1e-6, out=stat)
    N.rsqrt(stat, out=stat)
    N.mul(values, stat[:, None], out=values, scratch=rows)
    N.cast(values, out=q16[ring, :, :])
    N.cast(k16[ring, :, :], out=values)
    N.mul(values, values, out=work)
    N.reduce_sum(
        work,
        axis=1,
        out=stat,
        scratch=partial,
        order="block8_then_packed_merge",
    )
    N.add(stat, 1e-6, out=stat)
    N.rsqrt(stat, out=stat)
    N.mul(values, stat[:, None], out=values, scratch=rows)
    N.cast(values, out=k16[ring, :, :])
    ignored = N.compare(selectors[head % 8, :, :], 0.0, kind="eq", out=beta_mask)
    N.fill(beta8[ring, :, :], 0.0, mask=ignored, scratch=beta_zero)
    N.reduce_sum(
        beta8[ring, :, :],
        axis=1,
        out=betas[ring, :],
        scratch=beta_partial,
        order="block8_then_packed_merge",
    )


@N.inline
def prepare_store(inputs, buffers, geometry, coordinates):
    Qp, Kp, Vp, Bp = inputs
    q16, k16, v16, betas = buffers
    (BT,) = geometry
    lane, ring, chunk, head = coordinates
    row = lane * (BT // 2)
    N.copy(q16[ring, :, :], Qp[chunk, head, row : row + BT // 2, :])
    N.copy(k16[ring, :, :], Kp[chunk, head, row : row + BT // 2, :])
    N.copy(v16[ring, :, :], Vp[chunk, head, row : row + BT // 2, :])
    N.copy(betas[ring, :], Bp[chunk, head, row : row + BT // 2])


@N.prim_func(disjoint_external=True)
def prefill_prepare_full(
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
    Lower: N.Tensor((BT, BT), N.f32),
    BetaSelectors: N.Tensor((8, BT // 2, 8), N.f32),
    TOKENS: N.int64(minimum=1),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert CHUNKS == SEQUENCES * NC and NC > 0, "equal nonempty chunk domains"
        assert H % 8 == 0 and BT == 64, "aligned groups of eight beta heads"
        q16 = N.alloc_ub((2, BT // 2, K), DTYPE)
        k16 = N.alloc_ub((2, BT // 2, K), DTYPE)
        v16 = N.alloc_ub((2, BT // 2, K), DTYPE)
        values = N.alloc_ub((BT // 2, K), N.f32)
        work = N.alloc_ub((BT // 2, K), N.f32)
        beta8 = N.alloc_ub((2, BT // 2, 8), N.f32)
        selectors = N.alloc_ub((8, BT // 2, 8), N.f32)
        beta_zero = N.alloc_ub((BT // 2, 8), N.f32)
        beta_mask = N.alloc_mask((BT // 2, 8))
        beta_partial = N.alloc_ub((BT // 2,), N.f32)
        betas = N.alloc_ub((2, BT // 2), N.f32)
        partial = N.alloc_ub(((BT // 2) * (K // 8),), N.f32)
        stat = N.alloc_ub((BT // 2,), N.f32)
        rows = N.alloc_ub((BT // 2, 8), N.f32)
        lower1 = N.alloc_l1((BT, BT), N.f32)
        lower0 = N.alloc_l0a((BT, BT), N.f32)
        gates1 = N.alloc_l1((2, BT, K), N.f32)
        gates0 = N.alloc_l0b((BT, K), N.f32)
        prefixes0 = N.alloc_l0c((BT, K), N.f32)
        if NC > 1:
            with N.Cube():
                N.copy(Lower, lower1)
                N.copy(lower1, lower0)
            with N.Vector():
                for selector in N.unroll(8):
                    N.copy(BetaSelectors[selector, :, :], selectors[selector, :, :])
            for task in N.Dataflow(
                N.ceildiv(N.max(0, SEQUENCES * (NC - 1) * H - owner), GROUPS),
                issue_order={"vector": ("store", "input_load", "normalize")},
            ):
                flat = owner + task * GROUPS
                full, head = flat // H, flat % H
                chunk = (full // (NC - 1)) * NC + full % (NC - 1)
                ring = task % 2
                with N.Cube(), N.Stage(0, name="gate_load"):
                    prepare_gate_load(Starts, chunk, BT, Gate, gates1, head, ring)
                with N.Vector() as lane, N.Stage(0, name="input_load"):
                    prepare_input_load(
                        (Starts, Q, Key, Value, Beta),
                        (q16, k16, v16, beta8),
                        (BT,),
                        (chunk, lane, head, ring),
                    )
                with N.Cube(), N.Stage(1, name="gate_prefix"):
                    prepare_gate_prefix(gates0, gates1, ring, lower0, prefixes0, Gp, chunk, head)
                with N.Vector(), N.Stage(1, name="normalize"):
                    prepare_normalize(
                        (
                            q16,
                            values,
                            work,
                            stat,
                            partial,
                            rows,
                            k16,
                            selectors,
                            beta_mask,
                            beta8,
                            beta_zero,
                            beta_partial,
                            betas,
                        ),
                        (ring, head),
                    )
                with N.Vector() as lane, N.Stage(2, name="store"):
                    prepare_store(
                        (Qp, Kp, Vp, Bp), (q16, k16, v16, betas), (BT,), (lane, ring, chunk, head)
                    )


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
    # Full-plane clearing precedes overlapping stores of the valid prefix rows.
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
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
            start = N.load_index(Starts[chunk : chunk + 1])
            # The tail chunk is nonempty; this bounds its pipeline clock.
            valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
            N.fill(q, 0.0)
            N.cast(q, out=q16)
            N.cast(q, out=k16)
            N.cast(q, out=v16)
            N.copy(q, Gp[chunk, head, :, :])
            N.fill(cumulative, 0.0)
            N.fill(betas, 0.0)
            for row in N.Pipelined(valid):
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

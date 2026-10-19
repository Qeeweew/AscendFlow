"""Bulk DMA for the complete chunks preceding each runtime tail chunk."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


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
        values = N.alloc_ub((BT, K), N.f32)
        work = N.alloc_ub((BT, K), N.f32)
        gates = N.alloc_ub((BT, K), N.f32)
        prefixes = N.alloc_ub((BT, K), N.f32)
        cumulative = N.alloc_ub((K,), N.f32)
        betas = N.alloc_ub((BT,), N.f32)
        partial = N.alloc_ub((BT * (K // 8),), N.f32)
        stat = N.alloc_ub((BT,), N.f32)
        rows = N.alloc_ub((BT, 8), N.f32)
        if NC > 1:
            for task in N.serial(N.ceildiv(N.max(0, SEQUENCES * (NC - 1) * H - owner), GROUPS)):
                flat = owner + task * GROUPS
                full, head = flat // H, flat % H
                chunk = (full // (NC - 1)) * NC + full % (NC - 1)
                # All but the final chunk of a sequence are complete. The
                # caller's metadata contract bounds the actual native rows.
                start = N.load_index(Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - BT)
                N.copy(Q[start : start + BT, head, :], q16)
                N.copy(Key[start : start + BT, head, :], k16)
                N.copy(Value[start : start + BT, head, :], v16)
                N.copy(Gate[start : start + BT, head, :], gates)
                N.fill(cumulative, 0.0)
                for row in N.serial(BT):
                    coefficient = N.load_scalar(Beta[start + row, head : head + 1])
                    N.store_scalar(coefficient, out=betas[row : row + 1])
                    N.add(cumulative, gates[row, :], out=cumulative)
                    N.copy(cumulative, prefixes[row, :])
                N.copy(prefixes, Gp[chunk, head, :, :])
                N.cast(q16, out=values)
                N.mul(values, values, out=work)
                N.reduce_sum(
                    work, axis=1, out=stat, scratch=partial, order="block8_then_packed_merge"
                )
                N.add(stat, 1e-6, out=stat)
                N.rsqrt(stat, out=stat)
                N.mul(values, stat[:, None], out=values, scratch=rows)
                N.cast(values, out=q16)
                N.cast(k16, out=values)
                N.mul(values, values, out=work)
                N.reduce_sum(
                    work, axis=1, out=stat, scratch=partial, order="block8_then_packed_merge"
                )
                N.add(stat, 1e-6, out=stat)
                N.rsqrt(stat, out=stat)
                N.mul(values, stat[:, None], out=values, scratch=rows)
                N.cast(values, out=k16)
                N.copy(q16, Qp[chunk, head, :, :])
                N.copy(k16, Kp[chunk, head, :, :])
                N.copy(v16, Vp[chunk, head, :, :])
                N.copy(betas, Bp[chunk, head, :])

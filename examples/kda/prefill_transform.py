"""Form the rounded RHS, decayed keys and queries for chunk KDA."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def transform_prefix(inputs, buffers):
    Lengths, Bp, Gp = inputs
    beta, gates, last, factor = buffers
    valid = N.load_index(Lengths)
    beta_value = N.copy(Bp, beta)
    gate_values = N.copy(Gp, gates)
    last_value = N.copy(Gp[valid - 1, :], last)
    factors = N.exp(gate_values, out=factor)
    return beta_value, gate_values, last_value, factors


@N.inline
def form_right_inputs(inputs, buffers):
    Kp, Right, Vp = inputs
    raw, work, beta, rows, factor, packed = buffers
    loaded = N.copy(Kp, raw)
    value = N.cast(loaded, out=work)
    weighted = N.mul(value, beta[:, None], out=work, scratch=rows)
    decayed = N.mul(weighted, factor, out=work)
    rounded = N.cast(decayed, out=packed)
    N.copy(rounded, Right[0, :, :])
    loaded = N.copy(Vp, raw)
    value = N.cast(loaded, out=work)
    weighted = N.mul(value, beta[:, None], out=work, scratch=rows)
    rounded = N.cast(weighted, out=packed)
    N.copy(rounded, Right[1, :, :])


@N.inline
def form_decayed_keys(inputs, buffers):
    Kp, KG = inputs
    gates, factor, last, raw, work, packed = buffers
    negative = N.mul(gates, -1.0, out=factor)
    difference = N.add(negative, last[None, :], out=factor)
    factors = N.exp(difference, out=factor)
    loaded = N.copy(Kp, raw)
    value = N.cast(loaded, out=work)
    decayed = N.mul(value, factors, out=work)
    rounded = N.cast(decayed, out=packed)
    N.copy(rounded, KG)


@N.inline
def form_scaled_queries(inputs, buffers, geometry):
    Qp, QG = inputs
    raw, work, packed, gates, factor = buffers
    (SCALE,) = geometry
    loaded = N.copy(Qp, raw)
    value = N.cast(loaded, out=work)
    scaled = N.mul(value, SCALE, out=work)
    rounded = N.cast(scaled, out=packed)
    value = N.cast(rounded, out=work)
    factors = N.exp(gates, out=factor)
    decayed = N.mul(value, factors, out=work)
    rounded = N.cast(decayed, out=packed)
    N.copy(rounded, QG)


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
    Aqk: N.Tensor((CHUNKS, H, BT, BT), N.f32),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE, access="write", write_disjoint=True),
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
        raw = N.alloc_ub((BT, K), DTYPE)
        packed = N.alloc_ub((BT, K), DTYPE)
        work = N.alloc_ub((BT, K), N.f32)
        gates = N.alloc_ub((BT, K), N.f32)
        factor = N.alloc_ub((BT, K), N.f32)
        last = N.alloc_ub((K,), N.f32)
        beta = N.alloc_ub((BT,), N.f32)
        rows = N.alloc_ub((BT, 8), N.f32)
        af32 = N.alloc_ub((BT, BT), N.f32)
        af16 = N.alloc_ub((BT, BT), DTYPE)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            beta_value, gate_values, last_value, factors = transform_prefix(
                (Lengths[chunk : chunk + 1], Bp[chunk, head, :], Gp[chunk, head, :, :]),
                (beta, gates, last, factor),
            )
            form_right_inputs(
                (Kp[chunk, head, :, :], Right[chunk, head, :, :, :], Vp[chunk, head, :, :]),
                (raw, work, beta_value, rows, factors, packed),
            )
            form_decayed_keys(
                (Kp[chunk, head, :, :], KG[chunk, head, :, :]),
                (gate_values, factor, last_value, raw, work, packed),
            )
            form_scaled_queries(
                (Qp[chunk, head, :, :], QG[chunk, head, :, :]),
                (raw, work, packed, gate_values, factor),
                (SCALE,),
            )
            attention = N.copy(Aqk[chunk, head, :, :], af32)
            rounded = N.cast(attention, out=af16)
            N.copy(rounded, A16[chunk, head, :, :])

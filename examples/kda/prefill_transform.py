"""Form the rounded RHS, decayed keys and queries for chunk KDA."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def transform_prefix(inputs, buffers, coordinates):
    Lengths, Bp, Gp = inputs
    beta, gates, last, factor = buffers
    chunk, head = coordinates
    valid = N.load_index(Lengths[chunk : chunk + 1])
    N.copy(Bp[chunk, head, :], beta)
    N.copy(Gp[chunk, head, :, :], gates)
    N.copy(Gp[chunk, head, valid - 1, :], last)
    N.exp(gates, out=factor)


@N.inline
def form_right_inputs(inputs, buffers, coordinates):
    Kp, Right, Vp = inputs
    raw, work, beta, rows, factor, packed = buffers
    chunk, head = coordinates
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


@N.inline
def form_decayed_keys(inputs, buffers, coordinates):
    Kp, KG = inputs
    gates, factor, last, raw, work, packed = buffers
    chunk, head = coordinates
    N.mul(gates, -1.0, out=factor)
    N.add(factor, last[None, :], out=factor)
    N.exp(factor, out=factor)
    N.copy(Kp[chunk, head, :, :], raw)
    N.cast(raw, out=work)
    N.mul(work, factor, out=work)
    N.cast(work, out=packed)
    N.copy(packed, KG[chunk, head, :, :])


@N.inline
def form_scaled_queries(inputs, buffers, geometry, coordinates):
    Qp, QG = inputs
    raw, work, packed, gates, factor = buffers
    (SCALE,) = geometry
    chunk, head = coordinates
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
        af32 = N.alloc_ub((BT, BT), N.f32)
        af16 = N.alloc_ub((BT, BT), DTYPE)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            transform_prefix((Lengths, Bp, Gp), (beta, gates, last, factor), (chunk, head))
            form_right_inputs(
                (Kp, Right, Vp), (raw, work, beta, rows, factor, packed), (chunk, head)
            )
            form_decayed_keys((Kp, KG), (gates, factor, last, raw, work, packed), (chunk, head))
            form_scaled_queries(
                (Qp, QG), (raw, work, packed, gates, factor), (SCALE,), (chunk, head)
            )
            N.copy(Aqk[chunk, head, :, :], af32)
            N.cast(af32, out=af16)
            N.copy(af16, A16[chunk, head, :, :])

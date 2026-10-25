"""Form the rounded RHS, decayed keys and queries for chunk KDA."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def form_transform_rows(inputs, outputs, buffers, geometry, lane):
    """Reuse FP32 K and exp(G), preserving all BF16 rounding boundaries."""
    Qp, Kp, Vp, Gp, Bp = inputs
    Right, KG, QG = outputs
    raw, packed, work, gates, factor, exponential, last, beta, rows = buffers
    SCALE, ROWS = geometry
    row = lane * ROWS
    with N.CopyGroup(name="transform_inputs"):
        N.copy(Gp[row : row + ROWS, :], gates)
        N.copy(Bp[row : row + ROWS], beta[:ROWS])
        N.copy(Kp[row : row + ROWS, :], raw)
    N.exp(gates, out=exponential)
    key_values = N.cast(raw, out=work)
    # The FP32 key is now independent of raw. Prefetch V while computing
    # the two key transforms; later do the same for Q after casting V.
    N.copy(Vp[row : row + ROWS, :], raw)
    N.mul(key_values, beta[:ROWS, None], out=factor, scratch=rows)
    N.mul(factor, exponential, out=factor)
    N.cast(factor, out=packed)
    N.copy(packed, Right[0, row : row + ROWS, :])
    N.mul(gates, -1.0, out=factor)
    N.add(factor, last[None, :], out=factor)
    N.exp(factor, out=factor)
    N.mul(key_values, factor, out=factor)
    N.cast(factor, out=packed)
    N.copy(packed, KG[row : row + ROWS, :])
    N.cast(raw, out=work)
    N.copy(Qp[row : row + ROWS, :], raw)
    N.mul(work, beta[:ROWS, None], out=work, scratch=rows)
    N.cast(work, out=packed)
    N.copy(packed, Right[1, row : row + ROWS, :])
    N.cast(raw, out=work)
    N.mul(work, SCALE, out=work)
    N.cast(work, out=packed)
    N.cast(packed, out=work)
    N.mul(work, exponential, out=work)
    N.cast(work, out=packed)
    N.copy(packed, QG[row : row + ROWS, :])


@N.inline
def form_transform_inputs(inputs, outputs, buffers, geometry, coordinates):
    Qp, Kp, Vp, Gp, Bp, tail_last, full_last = inputs
    Right, KG, QG = outputs
    last = buffers[6]
    SCALE, task, tail_start = geometry
    (lane,) = coordinates
    if task < tail_start:
        # Both AIVs read this row. The chunk output remains live while the
        # next inputs may already overwrite each peer's LocalGp half.
        N.copy(full_last, last)
    else:
        N.copy(tail_last, last)
    form_transform_rows(
        (Qp, Kp, Vp, Gp, Bp),
        (Right, KG, QG),
        buffers,
        (SCALE, 32),
        lane,
    )


@N.inline
def publish_attention(Aqk, A16, raw, packed, lane):
    # Called after inverse acquisition, which carries both peers' Aqk writes.
    # Both 16-row panels form one contiguous peer-owned 32-row plane.
    row = lane * 32
    N.copy(Aqk[row : row + 32, :], raw)
    N.cast(raw, out=packed)
    N.copy(packed, A16[row : row + 32, :])


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
        # Keep exp(G) and the FP32 K while forming both key transforms.
        # The temporary is reused for RHS weighting and last-G decay.
        transformed = N.alloc_ub((BT, K), N.f32)
        last = N.alloc_ub((K,), N.f32)
        beta = N.alloc_ub((BT,), N.f32)
        rows = N.alloc_ub((BT, 8), N.f32)
        af32 = N.alloc_ub((BT, BT), N.f32)
        af16 = N.alloc_ub((BT, BT), DTYPE)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            valid = N.load_index(Lengths[chunk : chunk + 1], minimum=1, maximum=BT)
            N.copy(Gp[chunk, head, valid - 1, :], last)
            form_transform_rows(
                (
                    Qp[chunk, head, :, :],
                    Kp[chunk, head, :, :],
                    Vp[chunk, head, :, :],
                    Gp[chunk, head, :, :],
                    Bp[chunk, head, :],
                ),
                (Right[chunk, head, :, :, :], KG[chunk, head, :, :], QG[chunk, head, :, :]),
                (raw, packed, work, gates, transformed, factor, last, beta, rows),
                (SCALE, BT),
                0,
            )
            attention = N.copy(Aqk[chunk, head, :, :], af32)
            rounded = N.cast(attention, out=af16)
            N.copy(rounded, A16[chunk, head, :, :])

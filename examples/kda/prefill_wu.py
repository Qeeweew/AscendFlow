"""Generate rounded W/U and publish FP32 projection weights."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def wu_import(Inverse, a1, chunk, head, ring, Right, b1):
    N.copy(Inverse[chunk, head, :, :], a1[ring, :, :])
    for part in N.unroll(2):
        N.copy(Right[chunk, head, part, :, :], b1[ring, part, :, :])


@N.inline
def wu_products(inputs, buffers, coordinates):
    W16Ring, U = inputs
    a0, a1, b0, b1, c0 = buffers
    ring, owner, chunk, head = coordinates
    aa = N.copy(a1[ring, :, :], a0)
    ww = N.copy(b1[ring, 0, :, :], b0)
    N.matmul(aa, ww, accum=c0, initialize=True)
    N.copy(c0, W16Ring[owner, ring, :, :])
    uu = N.copy(b1[ring, 1, :, :], b0)
    N.matmul(aa, uu, accum=c0, initialize=True)
    N.copy(c0, U[chunk, head, :, :])


@N.inline
def publish_rounded_weights(inputs, buffers, geometry, coordinates):
    W16Ring, W32 = inputs
    w16, w32 = buffers
    (BT,) = geometry
    owner, ring, lane, chunk, head = coordinates
    N.copy(
        W16Ring[owner, ring, lane * (BT // 2) : (lane + 1) * (BT // 2), :],
        w16[ring, :, :],
    )
    N.cast(w16[ring, :, :], out=w32[ring, :, :])
    N.copy(w32[ring, :, :], W32[chunk, head, lane * (BT // 2) : (lane + 1) * (BT // 2), :])


@N.prim_func(disjoint_external=True)
def prefill_wu(
    Inverse: N.Tensor((CHUNKS, H, BT, BT), DTYPE),
    Right: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE),
    U: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    W16Ring: N.Tensor((GROUPS, 2, BT, K), DTYPE, access="write"),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        a1 = N.alloc_l1((2, BT, BT), DTYPE)
        b1 = N.alloc_l1((2, 2, BT, K), DTYPE)
        a0 = N.alloc_l0a((BT, BT), DTYPE)
        b0 = N.alloc_l0b((BT, K), DTYPE)
        c0 = N.alloc_l0c((BT, K), N.f32)
        w16 = N.alloc_ub((2, BT // 2, K), DTYPE)
        w32 = N.alloc_ub((2, BT // 2, K), N.f32)
        for task in N.Dataflow(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            ring = task % 2
            with N.Cube(), N.Stage(0, name="load"):
                wu_import(Inverse, a1, chunk, head, ring, Right, b1)
            with N.Cube(), N.Stage(1, name="compute"):
                wu_products((W16Ring, U), (a0, a1, b0, b1, c0), (ring, owner, chunk, head))
            with N.Vector() as lane, N.Stage(2, name="cast"):
                # Preserve the low-precision W rounding before its FP32 projection.
                publish_rounded_weights(
                    (W16Ring, W32), (w16, w32), (BT,), (owner, ring, lane, chunk, head)
                )

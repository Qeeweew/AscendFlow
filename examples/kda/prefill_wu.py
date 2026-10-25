"""Generate rounded W/U and publish FP32 projection weights."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def wu_import(inputs, buffers):
    Inverse, Right = inputs
    a1, b1 = buffers
    with N.CopyGroup(name="wu_operands"):
        inverse = N.copy(Inverse, a1)
        weight = N.copy(Right[0, :, :], b1[0, :, :])
        value = N.copy(Right[1, :, :], b1[1, :, :])
    return inverse, weight, value


@N.inline
def wu_products(inputs, operands, buffers):
    W16Ring, U = inputs
    inverse, weight, value = operands
    a0, b0, c0 = buffers
    aa = N.copy(inverse, a0)
    ww = N.copy(weight, b0)
    result = N.matmul(aa, ww, accum=c0, initialize=True)
    N.copy(result, W16Ring)
    uu = N.copy(value, b0)
    result = N.matmul(aa, uu, accum=c0, initialize=True)
    N.copy(result, U)


@N.inline
def publish_rounded_weights(inputs, buffers):
    W16Ring, W32 = inputs
    w16, w32 = buffers
    rounded = N.copy(W16Ring, w16)
    promoted = N.cast(rounded, out=w32)
    N.copy(promoted, W32)


@N.prim_func(disjoint_external=True)
def prefill_wu(
    Inverse: N.Tensor((CHUNKS, H, BT, BT), DTYPE),
    Right: N.Tensor((CHUNKS, H, 2, BT, K), DTYPE),
    U: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write", write_disjoint=True),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.bf16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        W16Ring = N.alloc_workspace((GROUPS, 2, BT, K), DTYPE, name="W16Ring")
        a1 = N.alloc_l1((BT, BT), DTYPE, slots=2)
        b1 = N.alloc_l1((2, BT, K), DTYPE, slots=2)
        a0 = N.alloc_l0a((BT, BT), DTYPE)
        b0 = N.alloc_l0b((BT, K), DTYPE)
        c0 = N.alloc_l0c((BT, K), N.f32)
        w16 = N.alloc_ub((BT // 2, K), DTYPE, slots=2)
        w32 = N.alloc_ub((BT // 2, K), N.f32, slots=2)
        weights = N.slotted_view(W16Ring[owner, :, :, :])
        for task in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS),
            issue_order={"cube": ("load", "compute"), "vector": ("cast",)},
        ):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            with N.Cube(), N.Stage(0, name="load"):
                inverse, weight, value = wu_import(
                    (Inverse[chunk, head, :, :], Right[chunk, head, :, :, :]),
                    (a1.at(task), b1.at(task)),
                )
            with N.Cube(), N.Stage(1, name="compute"):
                wu_products(
                    (weights.at(task), U[chunk, head, :, :]),
                    (inverse, weight, value),
                    (a0, b0, c0),
                )
            with N.Vector() as lane, N.Stage(2, name="cast"):
                # Preserve the low-precision W rounding before its FP32 projection.
                publish_rounded_weights(
                    (
                        weights.at(task)[lane * (BT // 2) : (lane + 1) * (BT // 2), :],
                        W32[chunk, head, lane * (BT // 2) : (lane + 1) * (BT // 2), :],
                    ),
                    (w16.at(task), w32.at(task)),
                )

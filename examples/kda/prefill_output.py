"""Chunk/head output tasks overlap double-buffered import, Cube products and stores."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def output_load(inputs, buffers, task):
    QG, Snapshot, Residual, A16 = inputs
    q1, h1, r1, a1 = buffers
    chunk, head, ring = task
    N.copy(QG[chunk, head, :, :], q1[ring, :, :])
    N.copy(Snapshot[chunk, head, :, :], h1[ring, :, :])
    N.copy(Residual[chunk, head, :, :], r1[ring, :, :])
    N.copy(A16[chunk, head, :, :], a1[ring, :, :])


@N.inline
def output_products(operands, buffers, destination, geometry, task):
    q1, h1, r1, a1 = operands
    a0, b0, products0 = buffers
    OutputProducts, owner = destination
    K, CK = geometry
    (ring,) = task
    for part in N.unroll(K // CK):
        aa = N.copy(q1[ring, :, part * CK : (part + 1) * CK], a0[part, :, :])
        bb = N.copy(h1[ring, :, part * CK : (part + 1) * CK], b0[part, :, :], transpose=True)
        N.matmul(aa, bb, accum=products0[ring, :, :K], initialize=part == 0)
    aa = N.copy(a1[ring, :, :], a0[0, :, :])
    bb = N.copy(r1[ring, :, :], b0[0, :, :])
    N.matmul(aa, bb, accum=products0[ring, :, K:], initialize=True)
    N.copy(products0[ring, :, :], OutputProducts[owner, ring, :, :])


@N.inline
def output_import(buffers, task, geometry):
    OutputProducts, products = buffers
    owner, ring, lane = task
    (BT,) = geometry
    row = lane * (BT // 2)
    N.copy(OutputProducts[owner, ring, row : row + BT // 2, :], products[ring, :, :])


@N.inline
def output_merge(buffers, task, geometry):
    products, total, output = buffers
    (ring,) = task
    (K,) = geometry
    N.add(products[ring, :, :K], products[ring, :, K:], out=total[ring, :, :])
    N.cast(total[ring, :, :], out=output[ring, :, :])


@N.inline
def output_store(buffers, task, geometry):
    output, Output, Starts, Lengths = buffers
    chunk, head, ring, lane = task
    (BT, NC) = geometry
    row = lane * (BT // 2)
    if chunk % NC < NC - 1:
        full_start = N.load_index(Starts[chunk : chunk + 1])
        N.copy(
            output[ring, :, :],
            Output[full_start + row : full_start + row + BT // 2, head, :],
        )
    else:
        tail_start = N.load_index(Starts[chunk : chunk + 1])
        valid = N.load_index(Lengths[chunk : chunk + 1])
        for token in N.serial(N.min(BT // 2, N.max(0, valid - row))):
            N.copy(
                output[ring, token, :],
                Output[tail_start + row + token, head, :],
            )


@N.prim_func(disjoint_external=True)
def prefill_output(
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE),
    Snapshot: N.Tensor((CHUNKS, H, K, K), DTYPE),
    Residual: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Output: N.Tensor((TOKENS, H, K), DTYPE, access="write", write_disjoint=True),
    OutputProducts: N.Tensor((GROUPS, 2, BT, 2 * K), N.f32, access="write"),
    Starts: N.Tensor((CHUNKS,), N.i64),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    TOKENS: N.i64(minimum=1),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    CK: N.constexpr = 64,
    BV: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64 and CK == 64 and BV == 64, "output tile policy"
        assert CHUNKS == SEQUENCES * NC, "each sequence has NC nonempty chunks"
        q1 = N.alloc_l1((2, BT, K), DTYPE)
        h1 = N.alloc_l1((2, K, K), DTYPE)
        r1 = N.alloc_l1((2, BT, K), DTYPE)
        a1 = N.alloc_l1((2, BT, BT), DTYPE)
        a0 = N.alloc_l0a((2, BT, CK), DTYPE)
        b0 = N.alloc_l0b((2, CK, K), DTYPE)
        # Adjacent NZ column intervals let one FIX publish both products.
        # Each AIV imports a contiguous half of the rows, then adds in FP32.
        products0 = N.alloc_l0c((2, BT, 2 * K), N.f32)
        products = N.alloc_ub((2, BT // 2, 2 * K), N.f32)
        total = N.alloc_ub((2, BT // 2, K), N.f32)
        output = N.alloc_ub((2, BT // 2, K), DTYPE)
        for task in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS),
        ):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            ring = task % 2
            with N.Cube(), N.Stage(0, name="load"):
                output_load((QG, Snapshot, Residual, A16), (q1, h1, r1, a1), (chunk, head, ring))
            with N.Cube(), N.Stage(1, name="products"):
                output_products(
                    (q1, h1, r1, a1), (a0, b0, products0), (OutputProducts, owner), (K, CK), (ring,)
                )
            with N.Vector() as lane, N.Stage(0, name="import"):
                output_import((OutputProducts, products), (owner, ring, lane), (BT,))
            with N.Vector(), N.Stage(1, name="merge"):
                output_merge((products, total, output), (ring,), (K,))
            with N.Vector() as lane, N.Stage(2, name="store"):
                output_store((output, Output, Starts, Lengths), (chunk, head, ring, lane), (BT, NC))

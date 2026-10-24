"""Chunk/head output tasks overlap double-buffered import, Cube products and stores."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def output_load(inputs, buffers):
    QG, Snapshot, Residual, A16 = inputs
    q1, h1, r1, a1 = buffers
    with N.CopyGroup(name="output_operands"):
        query = N.copy(QG, q1)
        history = N.copy(Snapshot, h1)
        residual = N.copy(Residual, r1)
        attention = N.copy(A16, a1)
    return query, history, residual, attention


@N.inline
def output_products(operands, buffers, destination, geometry):
    q1, h1, r1, a1 = operands
    a0, b0, products0 = buffers
    K, CK = geometry
    for part in N.unroll(K // CK):
        aa = N.copy(q1[:, part * CK : (part + 1) * CK], a0.slot(part))
        bb = N.copy(h1[:, part * CK : (part + 1) * CK], b0.slot(part), transpose=True)
        N.matmul(aa, bb, accum=products0[:, :K], initialize=part == 0)
    aa = N.copy(a1, a0.slot(0))
    bb = N.copy(r1, b0.slot(0))
    N.matmul(aa, bb, accum=products0[:, K:], initialize=True)
    N.copy(products0, destination)


@N.inline
def output_merge(buffers, geometry):
    products, total, output = buffers
    (K,) = geometry
    merged = N.add(products[:, :K], products[:, K:], out=total)
    return N.cast(merged, out=output)


@N.inline
def output_store(buffers, task, geometry):
    output, Output, Starts, Lengths = buffers
    chunk, lane = task
    (BT, NC) = geometry
    row = lane * (BT // 2)
    if chunk % NC < NC - 1:
        full_start = N.load_index(Starts)
        N.copy(
            output,
            Output[full_start + row : full_start + row + BT // 2, :],
        )
    else:
        tail_start = N.load_index(Starts)
        valid = N.load_index(Lengths)
        for token in N.serial(N.min(BT // 2, N.max(0, valid - row))):
            N.copy(
                output[token, :],
                Output[tail_start + row + token, :],
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
    DTYPE: N.constexpr = N.bf16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64 and CK == 64 and BV == 64, "output tile policy"
        assert CHUNKS == SEQUENCES * NC, "each sequence has NC nonempty chunks"
        q1 = N.alloc_l1((BT, K), DTYPE, slots=2)
        h1 = N.alloc_l1((K, K), DTYPE, slots=2)
        r1 = N.alloc_l1((BT, K), DTYPE, slots=2)
        a1 = N.alloc_l1((BT, BT), DTYPE, slots=2)
        a0 = N.alloc_l0a((BT, CK), DTYPE, slots=2)
        b0 = N.alloc_l0b((CK, K), DTYPE, slots=2)
        # Adjacent NZ column intervals let one FIX publish both products.
        # Each AIV imports a contiguous half of the rows, then adds in FP32.
        products0 = N.alloc_l0c((BT, 2 * K), N.f32, slots=2)
        products = N.alloc_ub((BT // 2, 2 * K), N.f32, slots=2)
        total = N.alloc_ub((BT // 2, K), N.f32, slots=2)
        output = N.alloc_ub((BT // 2, K), DTYPE, slots=2)
        product_values = N.slotted_view(OutputProducts[owner, :, :, :])
        for task in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS),
            issue_order={"cube": ("load", "products"), "vector": ("import", "merge", "store")},
        ):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            with N.Cube(), N.Stage(0, name="load"):
                query, history, residual, attention = output_load(
                    (
                        QG[chunk, head, :, :],
                        Snapshot[chunk, head, :, :],
                        Residual[chunk, head, :, :],
                        A16[chunk, head, :, :],
                    ),
                    (q1.at(task), h1.at(task), r1.at(task), a1.at(task)),
                )
            with N.Cube(), N.Stage(1, name="products"):
                output_products(
                    (query, history, residual, attention),
                    (a0, b0, products0.at(task)),
                    product_values.at(task),
                    (K, CK),
                )
            with N.Vector() as lane, N.Stage(0, name="import"):
                row = lane * (BT // 2)
                imported = N.copy(
                    product_values.at(task)[row : row + BT // 2, :], products.at(task)
                )
            with N.Vector(), N.Stage(1, name="merge"):
                result = output_merge((imported, total.at(task), output.at(task)), (K,))
            with N.Vector() as lane, N.Stage(2, name="store"):
                output_store(
                    (
                        result,
                        Output[:, head, :],
                        Starts[chunk : chunk + 1],
                        Lengths[chunk : chunk + 1],
                    ),
                    (chunk, lane),
                    (BT, NC),
                )

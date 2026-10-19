"""Chunk/head output tasks overlap double-buffered import, Cube products and stores."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def prefill_output(
    QG: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    A16: N.Tensor((CHUNKS, H, BT, BT), DTYPE),
    Snapshot: N.Tensor((CHUNKS, H, K, K), DTYPE),
    Residual: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Output: N.Tensor((TOKENS, H, K), DTYPE, access="write", write_disjoint=True),
    OutputProducts: N.Tensor((GROUPS, 2, 2, BT, K), N.f32, access="write"),
    Starts: N.Tensor((CHUNKS,), N.i64),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    TOKENS: N.int64(minimum=1),
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
        products0 = N.alloc_l0c((2, 2, BT, K), N.f32)
        inter = N.alloc_ub((2, BT, BV), N.f32)
        intra = N.alloc_ub((2, BT, BV), N.f32)
        output = N.alloc_ub((2, BT, BV), DTYPE)
        for task in N.Pipelined(
            N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS),
            stages=3,
        ):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            ring = task % 2
            with N.Stage(0, name="load"), N.Cube():
                N.copy(QG[chunk, head, :, :], q1[ring, :, :])
                N.copy(Snapshot[chunk, head, :, :], h1[ring, :, :])
                N.copy(Residual[chunk, head, :, :], r1[ring, :, :])
                N.copy(A16[chunk, head, :, :], a1[ring, :, :])
            with N.Stage(1, name="compute"), N.Cube():
                for part in N.unroll(K // CK):
                    aa = N.copy(q1[ring, :, part * CK : (part + 1) * CK], a0[part, :, :])
                    bb = N.copy(
                        h1[ring, :, part * CK : (part + 1) * CK], b0[part, :, :], transpose=True
                    )
                    N.matmul(aa, bb, accum=products0[ring, 0, :, :], initialize=part == 0)
                N.copy(products0[ring, 0, :, :], OutputProducts[owner, ring, 0, :, :])
                aa = N.copy(a1[ring, :, :], a0[0, :, :])
                bb = N.copy(r1[ring, :, :], b0[0, :, :])
                N.matmul(aa, bb, accum=products0[ring, 1, :, :], initialize=True)
                N.copy(products0[ring, 1, :, :], OutputProducts[owner, ring, 1, :, :])
            with N.Stage(2, name="store"), N.Vector() as lane:
                column = lane * BV
                N.copy(OutputProducts[owner, ring, 0, :, column : column + BV], inter[ring, :, :])
                N.copy(OutputProducts[owner, ring, 1, :, column : column + BV], intra[ring, :, :])
                N.add(inter[ring, :, :], intra[ring, :, :], out=inter[ring, :, :])
                N.cast(inter[ring, :, :], out=output[ring, :, :])
                if chunk % NC < NC - 1:
                    full_start = N.load_index(
                        Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - BT
                    )
                    N.copy(
                        output[ring, :, :],
                        Output[full_start : full_start + BT, head, column : column + BV],
                    )
                else:
                    tail_start = N.load_index(
                        Starts[chunk : chunk + 1], minimum=0, maximum=TOKENS - 1
                    )
                    valid = N.load_index(
                        Lengths[chunk : chunk + 1],
                        minimum=1,
                        maximum=N.min(BT, TOKENS - tail_start),
                    )
                    for token in N.serial(valid):
                        N.copy(
                            output[ring, token, :],
                            Output[tail_start + token, head, column : column + BV],
                        )

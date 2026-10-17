"""Contiguous NT GEMM: f16 A[M,K] @ Bt[N,K].T, f32 C[M,N].

Each Cube owns complete output tiles and reduces K in order. Two column
tiles share A; their accumulator planes retire independently. GM inputs need
no packing. Shapes are constexpr because native row pitches are currently
static; extents must be positive multiples of the explicit tiles.
"""

from __future__ import annotations

from ascendflow import language as T

# Tensor dimensions in annotations are interpreted by the DSL signature pass.
# ruff: noqa: F821


@T.prim_func(disjoint_external=True)
def dense_gemm(
    A: T.Tensor((M, K), T.f16),
    Bt: T.Tensor((N, K), T.f16),
    C: T.Tensor(
        (M, N),
        T.f32,
        access="write",
        cache=("bypass" if M * N * 4 > L2_BYTES else "normal")
        if OUTPUT_CACHE == "auto"
        else OUTPUT_CACHE,
    ),
    M: T.constexpr = 4096,
    N: T.constexpr = 4096,
    K: T.constexpr = 4096,
    BM: T.constexpr = 128,
    BN: T.constexpr = 128,
    BK: T.constexpr = 256,
    CK: T.constexpr = 128,
    A_SLOTS: T.constexpr = 2,
    B_SLOTS: T.constexpr = 2,
    C_SLOTS: T.constexpr = 1,
    PREFETCH: T.constexpr = 1,
    COL_PARTS: T.constexpr = 2,
    OUTPUT_CACHE: T.constexpr = "auto",
    L2_BYTES: T.constexpr = 201326592,
    GROUPS: T.constexpr = 20,
):
    with T.Kernel(GROUPS, kind="cube") as group:
        assert M > 0, "M must be positive"
        assert N > 0, "N must be positive"
        assert K > 0, "K must be positive"
        assert M % BM == 0, "M must contain complete output row tiles"
        assert N % (BN * COL_PARTS) == 0, "N must contain complete output column groups"
        assert K % BK == 0, "K must contain complete reduction panels"
        assert BK % CK == 0, "L1 panels must contain complete Cube chunks"
        # The default K panel contains two contiguous full-row NZ slices.
        a1 = T.alloc_l1((A_SLOTS, BM, BK), T.f16)
        b1 = T.alloc_l1((B_SLOTS, COL_PARTS, BN, BK), T.f16)
        a0 = T.alloc_l0a((2, BM, CK), T.f16)
        b0 = T.alloc_l0b((2, CK, BN), T.f16)
        # Each column part has its own last MAD and FIX retirement point.
        c0 = T.alloc_l0c((C_SLOTS, COL_PARTS, BM, BN), T.f32)
        ntiles = N // (BN * COL_PARTS)
        ktiles = K // BK
        chunks = BK // CK
        tiles = (M // BM) * ntiles
        local_tiles = T.ceildiv(T.max(0, tiles - group), GROUPS)
        for tile in T.serial(local_tiles):
            index = group + tile * GROUPS
            row = (index // ntiles) * BM
            column = (index % ntiles) * BN * COL_PARTS
            for panel in T.Pipelined(ktiles, stages=PREFETCH + 1):
                k = panel * BK
                with T.Stage(0, name="load"):
                    T.copy(A[row : row + BM, k : k + BK], a1[panel % A_SLOTS, :, :])
                    for part in T.unroll(COL_PARTS):
                        T.copy(
                            Bt[column + part * BN : column + (part + 1) * BN, k : k + BK],
                            b1[panel % B_SLOTS, part, :, :],
                        )
                with T.Stage(PREFETCH, name="compute"):
                    for chunk in T.unroll(chunks):
                        aa = T.copy(
                            a1[panel % A_SLOTS, :, chunk * CK : (chunk + 1) * CK],
                            a0[chunk % 2, :, :],
                        )
                        for part in T.unroll(COL_PARTS):
                            # Both output parts borrow the same A operand.
                            bb = T.copy(
                                b1[panel % B_SLOTS, part, :, chunk * CK : (chunk + 1) * CK],
                                b0[(chunk * COL_PARTS + part) % 2, :, :],
                                transpose=True,
                            )
                            if chunk == 0:
                                T.matmul(
                                    aa,
                                    bb,
                                    accum=c0[tile % C_SLOTS, part, :, :],
                                    initialize=(panel == 0),
                                )
                            else:
                                T.matmul(
                                    aa, bb, accum=c0[tile % C_SLOTS, part, :, :], initialize=False
                                )
                    if panel == ktiles - 1:
                        for part in T.unroll(COL_PARTS):
                            T.copy(
                                c0[tile % C_SLOTS, part, :, :],
                                C[row : row + BM, column + part * BN : column + (part + 1) * BN],
                            )

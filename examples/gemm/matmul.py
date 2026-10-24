"""Contiguous b16 C[M,N] = A[M,K] @ B[K,N], including clipped edge tiles."""

from __future__ import annotations

from ascendflow import language as T
from examples.gemm.schedule import l2_tile

# Dimensions and dtype in annotations are interpreted by the DSL signature pass.
# ruff: noqa: F821


@T.inline
def import_panel(inputs, buffers, geometry, coordinates):
    A, B = inputs
    a1, b1 = buffers
    mvalid, kvalid, nvalid = geometry
    row, k, column = coordinates
    with T.CopyGroup(name="panels"):
        av = T.copy(A[row : row + mvalid, k : k + kvalid], a1, pad_value=0)
        bv = T.copy(B[k : k + kvalid, column : column + nvalid], b1, pad_value=0)
    return av, bv


@T.inline
def accumulate_panel(inputs, buffers, geometry, coordinates):
    C, av, bv = inputs
    a0, b0, c0 = buffers
    BK, CK, parts, BN, final, nvalid, mvalid = geometry
    panel, kbase, panels, column, row = coordinates
    for chunk in T.unroll(BK // CK):
        aa = T.copy(
            av[:, chunk * CK : (chunk + 1) * CK],
            a0.at(chunk),
        )
        for part in T.unroll(parts):
            bb = T.copy(
                bv[
                    chunk * CK : (chunk + 1) * CK,
                    part * BN : (part + 1) * BN,
                ],
                b0.at(chunk * parts + part),
            )
            T.matmul(
                aa,
                bb,
                accum=c0[part, :, :],
                initialize=(panel == 0) if chunk == 0 and kbase == 0 else False,
            )
    if final:  # noqa: SIM102 - constexpr then device predicate
        if panel == panels - 1:
            for part in T.unroll(parts):
                left = column + part * BN
                valid = T.min(BN, nvalid - part * BN)
                T.copy(
                    c0[part, :, :],
                    C[row : row + mvalid, left : left + valid],
                )


@T.prim_func(disjoint_external=True)
def gemm(
    A: T.Tensor((M, K), DTYPE),
    B: T.Tensor((K, N), DTYPE),
    C: T.Tensor(
        (M, N),
        DTYPE,
        access="write",
        cache=("bypass" if (M * K + K * N + M * N) * 2 > L2_BYTES else "normal")
        if OUTPUT_CACHE == "auto"
        else OUTPUT_CACHE,
    ),
    M: T.constexpr = 4096,
    N: T.constexpr = 4096,
    K: T.constexpr = 4096,
    DTYPE: T.constexpr = "f16",
    BM: T.constexpr = 128,
    BN: T.constexpr = 128,
    BK: T.constexpr = 256,
    CK: T.constexpr = 128,
    COL_PARTS: T.constexpr = 2,
    L2_TILE: T.constexpr = 0,
    GROUPS: T.constexpr = 20,
    OUTPUT_CACHE: T.constexpr = "auto",
    L2_BYTES: T.constexpr = 201326592,
):
    with T.Kernel(GROUPS, kind="cube") as group:
        assert M > 0 and N > 0 and K > 0, "M/N/K must be positive"
        assert DTYPE == "f16" or DTYPE == "bf16", "GEMM input/output dtype must be f16 or bf16"
        assert BM % 16 == 0 and BN % 16 == 0 and BK % 16 == 0, "tiles need complete Cube fractals"
        assert CK > 0 and BK % CK == 0, "BK must contain complete Cube chunks"
        assert L2_TILE >= 0, "L2_TILE must be nonnegative; zero selects the cache budget"
        a1 = T.alloc_l1((BM, BK), DTYPE, slots=2)
        b1 = T.alloc_l1((BK, BN * COL_PARTS), DTYPE, slots=2)
        a0 = T.alloc_l0a((BM, CK), DTYPE, slots=2)
        b0 = T.alloc_l0b((CK, BN), DTYPE, slots=2)
        c0 = T.alloc_l0c((COL_PARTS, BM, BN), T.f32)
        # Unaligned output rows stay with one Cube to avoid overlapping byte fragments.
        balanced = M % BM == 0 and N % (BN * COL_PARTS) == 0
        for mbase, mcount, mvalid in T.tile_ranges(M, BM):
            for nbase, ncount, nvalid in T.tile_ranges(N, BN * COL_PARTS):
                parts = T.ceildiv(nvalid, BN)
                count = mcount * ncount if balanced else mbase // BM + mcount
                first = 0 if balanced else T.ceildiv(T.max(0, mbase // BM - group), GROUPS)
                last = T.ceildiv(T.max(0, count - group), GROUPS)
                for tile in T.serial(first, last):
                    index = group + tile * GROUPS
                    row, col_start = (
                        l2_tile(index, mcount, ncount, BM, BN * COL_PARTS, K, L2_TILE, L2_BYTES)
                        if balanced
                        else (index * BM, 0)
                    )
                    for col in T.serial(1 if balanced else ncount):
                        column = nbase + col_start + (0 if balanced else col * BN * COL_PARTS)
                        for kbase, panels, kvalid in T.tile_ranges(K, BK):
                            final = kbase + panels * BK >= K
                            for panel in T.Pipelined(panels, issue_order=("load", "compute")):
                                k = kbase + panel * BK
                                with T.Stage(0, name="load"):
                                    av, bv = import_panel(
                                        (A, B),
                                        (a1.at(panel), b1.at(panel)),
                                        (mvalid, kvalid, nvalid),
                                        (row, k, column),
                                    )
                                with T.Stage(1, name="compute"):
                                    accumulate_panel(
                                        (C, av, bv),
                                        (a0, b0, c0),
                                        (BK, CK, parts, BN, final, nvalid, mvalid),
                                        (panel, kbase, panels, column, row),
                                    )

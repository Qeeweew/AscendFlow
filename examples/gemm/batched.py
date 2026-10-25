"""Explicit L1/L0A/L0B/L0C rings for a runtime batch of native GEMMs.

This closes the AST Cube storage API. It is not a nested K-reduction GEMM or
Flash Attention: every task computes one independent complete matrix product.
"""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def batched_matmul(
    A: T.Tensor((GROUPS, tiles, M, K), DTYPE),  # noqa: F821
    B: T.Tensor((GROUPS, tiles, N if TRANSPOSE_B else K, K if TRANSPOSE_B else N), DTYPE),  # noqa: F821
    C: T.Tensor((GROUPS, tiles, M, N), T.f32, access="write"),  # noqa: F821
    tiles: T.i64,
    M: T.constexpr = 32,
    N: T.constexpr = 64,
    K: T.constexpr = 64,
    GROUPS: T.constexpr = 20,
    DTYPE: T.constexpr = T.f16,
    TRANSPOSE_B: T.constexpr = True,
    L1_SLOTS: T.constexpr = 3,
    L0_SLOTS: T.constexpr = 2,
    C_SLOTS: T.constexpr = 2,
    ACCUMULATE: T.constexpr = False,
):
    with T.Kernel(GROUPS, kind="cube") as group:
        a1 = T.alloc_l1((M, K), DTYPE, slots=L1_SLOTS, name="a1")
        b1 = T.alloc_l1(
            (N if TRANSPOSE_B else K, K if TRANSPOSE_B else N), DTYPE, slots=L1_SLOTS, name="b1"
        )
        a0 = T.alloc_l0a((M, K), DTYPE, slots=L0_SLOTS, name="a0")
        b0 = T.alloc_l0b((K, N), DTYPE, slots=L0_SLOTS, name="b0")
        c0 = T.alloc_l0c((M, N), T.f32, slots=C_SLOTS, name="c0")
        for t in T.Pipelined(tiles, issue_order=("export", "load", "operands", "compute")):
            with T.Stage(0, name="load"), T.CopyGroup(name="panels"):
                av = T.copy(A[group, t, :, :], a1.at(t))
                bv = T.copy(B[group, t, :, :], b1.at(t))
            with T.Stage(1, name="operands"):
                aa = T.copy(av, a0.at(t))
                bb = T.copy(bv, b0.at(t), transpose=TRANSPOSE_B)
            with T.Stage(2, name="compute"):
                value = T.matmul(aa, bb, accum=c0.at(t), initialize=True)
                if ACCUMULATE:
                    value = T.matmul(aa, bb, accum=value, initialize=False)
            with T.Stage(3, name="export"):
                T.copy(value, C[group, t, :, :])

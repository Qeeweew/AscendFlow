"""Runtime K-panel reduction with one retained pipeline and terminal Fixpipe.

Inputs explicitly use panel-major storage: [group, panel, M, BK] and either
[group, panel, N, BK] (NT) or [group, panel, BK, N] (NN). No compiler-side GM
packing is performed. For zero panels the program leaves output untouched.
"""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def reduction_gemm(
    A: T.Tensor((GROUPS, panels, M, BK), DTYPE),  # noqa: F821
    B: T.Tensor((GROUPS, panels, N if TRANSPOSE_B else BK, BK if TRANSPOSE_B else N), DTYPE),  # noqa: F821
    C: T.Tensor((GROUPS, M, N), T.f32, access="write"),  # noqa: F821
    panels: T.i64,
    M: T.constexpr = 32,
    N: T.constexpr = 64,
    BK: T.constexpr = 64,
    GROUPS: T.constexpr = 20,
    DTYPE: T.constexpr = T.f16,
    TRANSPOSE_B: T.constexpr = True,
):
    with T.Kernel(GROUPS, kind="cube") as group:
        a1 = T.alloc_l1((M, BK), DTYPE, slots=3, name="a1")
        b1 = T.alloc_l1(
            (N if TRANSPOSE_B else BK, BK if TRANSPOSE_B else N), DTYPE, slots=3, name="b1"
        )
        a0 = T.alloc_l0a((M, BK), DTYPE, slots=2, name="a0")
        b0 = T.alloc_l0b((BK, N), DTYPE, slots=2, name="b0")
        c0 = T.alloc_l0c((M, N), T.f32, name="c0")
        for k in T.Pipelined(panels, issue_order=("load", "operands", "compute")):
            with T.Stage(0, name="load"), T.CopyGroup(name="panels"):
                av = T.copy(A[group, k, :, :], a1.at(k))
                bv = T.copy(B[group, k, :, :], b1.at(k))
            with T.Stage(1, name="operands"):
                aa = T.copy(av, a0.at(k))
                bb = T.copy(bv, b0.at(k), transpose=TRANSPOSE_B)
            with T.Stage(2, name="compute"):
                value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))
                if k == panels - 1:
                    T.copy(value, C[group, :, :])

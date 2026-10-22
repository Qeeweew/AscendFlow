"""R2a source: runtime output loop around the runtime K reduction.

Panel-major inputs are explicit. Parent and child loops are retained through
physical IR; per-slot credits and final Fixpipe release protect repeated calls.
An empty child preserves its output. No per-invocation whole-pipe drain.
"""

from __future__ import annotations

from ascendflow import language as T


@T.prim_func(disjoint_external=True)
def hierarchical_gemm(
    A: T.Tensor((GROUPS, outputs, panels, M, BK), DTYPE),  # noqa: F821
    B: T.Tensor((GROUPS, outputs, panels, N, BK), DTYPE),  # noqa: F821
    C: T.Tensor((GROUPS, outputs, M, N), T.f32, access="write"),  # noqa: F821
    outputs: T.i64,
    panels: T.i64,
    M: T.constexpr = 32,
    N: T.constexpr = 64,
    BK: T.constexpr = 64,
    GROUPS: T.constexpr = 20,
    DTYPE: T.constexpr = T.f16,
):
    with T.Kernel(GROUPS, kind="cube") as group:
        a1 = T.alloc_l1((3, M, BK), DTYPE, name="a1")
        b1 = T.alloc_l1((3, N, BK), DTYPE, name="b1")
        a0 = T.alloc_l0a((2, M, BK), DTYPE, name="a0")
        b0 = T.alloc_l0b((2, BK, N), DTYPE, name="b0")
        c0 = T.alloc_l0c((M, N), T.f32, name="c0")
        for output in T.serial(outputs):
            for k in T.Pipelined(panels):
                with T.Stage(0, name="load"):
                    av = T.copy(A[group, output, k, :, :], a1[k % 3, :, :])
                    bv = T.copy(B[group, output, k, :, :], b1[k % 3, :, :])
                with T.Stage(1, name="operands"):
                    aa = T.copy(av, a0[k % 2, :, :])
                    bb = T.copy(bv, b0[k % 2, :, :], transpose=True)
                with T.Stage(2, name="compute"):
                    value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))
                    if k == panels - 1:
                        T.copy(value, C[group, output, :, :])

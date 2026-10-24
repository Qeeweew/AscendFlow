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
        a1 = T.alloc_l1((M, BK), DTYPE, slots=3, name="a1")
        b1 = T.alloc_l1((N, BK), DTYPE, slots=3, name="b1")
        a0 = T.alloc_l0a((M, BK), DTYPE, slots=2, name="a0")
        b0 = T.alloc_l0b((BK, N), DTYPE, slots=2, name="b0")
        c0 = T.alloc_l0c((M, N), T.f32, name="c0")
        for output in T.serial(outputs):
            for k in T.Pipelined(panels, issue_order=("load", "operands", "compute")):
                with T.Stage(0, name="load"), T.CopyGroup(name="panels"):
                    av = T.copy(A[group, output, k, :, :], a1.at(k))
                    bv = T.copy(B[group, output, k, :, :], b1.at(k))
                with T.Stage(1, name="operands"):
                    aa = T.copy(av, a0.at(k))
                    bb = T.copy(bv, b0.at(k), transpose=True)
                with T.Stage(2, name="compute"):
                    value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))
                    if k == panels - 1:
                        T.copy(value, C[group, output, :, :])

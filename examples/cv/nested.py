"""Two AIVs share one Vector body across nested Cube/Vector pipelines.

Score is a double-buffered GM handoff; each AIV owns half its rows and private
UB. Runtime n stays symbolic. Scheduling and storage policy live in this example.
"""

from __future__ import annotations

# Runtime dimensions in annotations are captured by the DSL frontend.
# ruff: noqa: F821
from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def nested_cv(
    X: N.Tensor((2, n, 2, 16, 16), N.f16),
    Score: N.Tensor((2, 2, 16, 32), N.f32, access="write"),
    O: N.Tensor((2, n, 16, 32), N.f32, access="write"),
    n: N.int64,
):
    with N.Kernel(2, kind="mixed") as group:
        x1 = N.alloc_l1((2, 16, 16))
        a = N.alloc_l0a((2, 16, 16))
        b = N.alloc_l0b((2, 16, 16))
        c = N.alloc_l0c((2, 16, 16))
        v = N.alloc_ub((2, 4, 32))
        for t in N.Dataflow(n):
            with N.Cube(), N.Stage(0, name="cube"):
                for j in N.Pipelined(2):
                    with N.Stage(0, name="load"):
                        x = N.copy(X[group, t, j, :, :], x1[j % 2, :, :])
                    with N.Stage(1, name="operands"):
                        aa = N.copy(x, a[j % 2, :, :])
                        bb = N.copy(x, b[j % 2, :, :], transpose=True)
                    with N.Stage(2, name="mad"):
                        z = N.matmul(aa, bb, accum=c[j % 2, :, :], initialize=True)
                    with N.Stage(3, name="fix"):
                        N.copy(z, Score[group, t % 2, :, j * 16 : (j + 1) * 16])
            with N.Vector() as lane, N.Stage(1, name="vector"):
                for r in N.Pipelined(2):
                    row = lane * 8 + r * 4
                    with N.Stage(0, name="load"):
                        z = N.copy(Score[group, t % 2, row : row + 4, :], v[r % 2, :, :])
                    with N.Stage(1, name="scale"):
                        z = N.mul(z, 2.0, out=v[r % 2, :, :])
                    with N.Stage(2, name="store"):
                        N.copy(z, O[group, t, row : row + 4, :])

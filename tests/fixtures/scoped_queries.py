"""A small public matrix/vector recurrence with serial query and runtime KV clocks."""

from __future__ import annotations

# Runtime dimensions in annotations are captured by the public DSL frontend.
# ruff: noqa: F821
from ascendflow import language as N

GROUPS = 2


@N.prim_func(disjoint_external=True)
def scoped_query_recurrence(
    Q: N.Tensor((total_queries, 2, 16, 16), N.f16),
    K: N.Tensor((kv_tiles, 2, 16, 16), N.f16),
    Score: N.Tensor((GROUPS, 2, 16, 32), N.f32, access="write"),
    O: N.Tensor((total_queries, 16, 32), N.f32, access="write"),
    total_queries: N.i64(minimum=0),
    kv_tiles: N.i64(minimum=1),
):
    with N.Kernel(GROUPS, kind="mixed") as group:
        q_l1 = N.alloc_l1((2, 16, 16), N.f16, name="q_l1")
        k_l1 = N.alloc_l1((2, 16, 16), N.f16, name="k_l1")
        a = N.alloc_l0a((2, 16, 16), N.f16, name="a")
        b = N.alloc_l0b((2, 16, 16), N.f16, name="b")
        c = N.alloc_l0c((2, 16, 16), N.f32, name="c")
        loaded = N.alloc_ub((2, 4, 32), N.f32, name="loaded")
        state = N.alloc_ub((2, 4, 32), N.f32, name="state")
        for query in N.serial(N.ceildiv(N.max(0, total_queries - group), GROUPS)):
            flat = group + query * GROUPS
            for kv in N.Dataflow(kv_tiles):
                task = query * kv_tiles + kv
                slot = task % 2
                with N.Cube(), N.Stage(0, name="cube"):
                    for j in N.Pipelined(2):
                        with N.Stage(0, name="load"):
                            q = N.copy(Q[flat, j, :, :], q_l1[j % 2, :, :])
                            k = N.copy(K[kv, j, :, :], k_l1[j % 2, :, :])
                        with N.Stage(1, name="operands"):
                            aa = N.copy(q, a[j % 2, :, :])
                            bb = N.copy(k, b[j % 2, :, :], transpose=True)
                        with N.Stage(2, name="mad"):
                            z = N.matmul(aa, bb, accum=c[j % 2, :, :], initialize=True)
                        with N.Stage(3, name="export"):
                            N.copy(z, Score[group, slot, :, j * 16 : (j + 1) * 16])
                with N.Vector() as lane, N.Stage(1, name="vector"):
                    for r in N.Pipelined(2):
                        row = lane * 8 + r * 4
                        with N.Stage(0, name="load"):
                            N.copy(Score[group, slot, row : row + 4, :], loaded[r % 2, :, :])
                        with N.Stage(1, name="update"):
                            if kv == 0:
                                N.fill(state[r, :, :], 0.0)
                            N.add(state[r, :, :], loaded[r % 2, :, :], out=state[r, :, :])
                            N.mul(state[r, :, :], 0.5, out=state[r, :, :])
                        with N.Stage(2, name="store"):
                            if kv == kv_tiles - 1:
                                N.copy(state[r, :, :], O[flat, row : row + 4, :])

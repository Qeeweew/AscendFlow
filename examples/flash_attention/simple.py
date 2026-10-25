"""Teaching example: aligned, non-causal attention with one resident KV tile.

Each head computes softmax(Q @ K.T / sqrt(128)) @ V: D=128, Sk=128, BQ=64.
S is a positive multiple of 64. FP16 Q/K/V/P/O, FP32 scores and reductions.
Score, P and PV are two-slot GM workspaces, reused by each bounded worker.
"""

from __future__ import annotations

from ascendflow import language as N

# Dimensions in annotations are resolved by the DSL signature pass.
# ruff: noqa: F821


@N.inline
def qk(query, key, scores, buffers):
    q1, q0, score0 = buffers
    loaded = N.copy(query, q1)
    operand = N.copy(loaded, q0)
    result = N.matmul(operand, key, accum=score0, initialize=True)
    N.copy(result, scores)


@N.inline
def softmax(scores, probabilities, buffers):
    score, probability, maximum, total, partial, broadcast = buffers
    loaded = N.copy(scores, score)
    scaled = N.mul(loaded, 0.08838834764831845, out=score)
    row_max = N.reduce_max(
        scaled, axis=1, out=maximum, scratch=partial[:, :2], order="r64_then_packed_merge"
    )
    shifted = N.sub(scaled, row_max[:, None], out=score, scratch=broadcast)
    exponentials = N.exp(shifted, out=score)
    row_sum = N.reduce_sum(
        exponentials, axis=1, out=total, scratch=partial[:, :2], order="r64_then_packed_merge"
    )
    normalized = N.div(exponentials, row_sum[:, None], out=score, scratch=broadcast)
    converted = N.cast(normalized, out=probability)
    N.copy(converted, probabilities)


@N.inline
def pv(probabilities, value, products, buffers):
    p1, p0, output0 = buffers
    loaded = N.copy(probabilities, p1)
    operand = N.copy(loaded, p0)
    result = N.matmul(operand, value, accum=output0, initialize=True)
    N.copy(result, products)


@N.inline
def store(products, destination, buffers):
    output, result = buffers
    loaded = N.copy(products, output)
    converted = N.cast(loaded, out=result)
    N.copy(converted, destination)


@N.prim_func(disjoint_external=True)
def simple_attention(
    Q: N.Tensor((H, S, 128), N.f16),
    K: N.Tensor((H, 128, 128), N.f16),
    V: N.Tensor((H, 128, 128), N.f16),
    O: N.Tensor((H, S, 128), N.f16, access="write", write_disjoint=True),
    S: N.constexpr = 256,
    H: N.constexpr = 4,
    GROUPS: N.constexpr = 20,
):
    workers = N.min(H, GROUPS)
    with N.Kernel(workers, kind="mixed") as group:
        assert S > 0 and S % 64 == 0
        assert H > 0 and GROUPS > 0
        Score = N.alloc_workspace((workers, 2, 64, 128), N.f32, name="Score")
        P = N.alloc_workspace((workers, 2, 64, 128), N.f16, name="P")
        PV = N.alloc_workspace((workers, 2, 64, 128), N.f32, name="PV")
        q1 = N.alloc_l1((64, 128), N.f16)
        k1 = N.alloc_l1((128, 128), N.f16)
        p1 = N.alloc_l1((64, 128), N.f16)
        v1 = N.alloc_l1((128, 128), N.f16)
        q0 = N.alloc_l0a((64, 128), N.f16)
        k0 = N.alloc_l0b((128, 128), N.f16)
        p0 = N.alloc_l0a((64, 128), N.f16)
        v0 = N.alloc_l0b((128, 128), N.f16)
        score0 = N.alloc_l0c((64, 128), N.f32)
        output0 = N.alloc_l0c((64, 128), N.f32)
        score = N.alloc_ub((32, 128), N.f32)
        probability = N.alloc_ub((32, 128), N.f16)
        output = N.alloc_ub((32, 128), N.f32)
        result = N.alloc_ub((32, 128), N.f16)
        maximum = N.alloc_ub((32,), N.f32)
        total = N.alloc_ub((32,), N.f32)
        partial = N.alloc_ub((32, 8), N.f32)
        broadcast = N.alloc_ub((32, 8), N.f32)

        qk_buffers = (q1, q0, score0)
        softmax_buffers = (score, probability, maximum, total, partial, broadcast)
        pv_buffers = (p1, p0, output0)
        store_buffers = (output, result)
        # A worker with exactly one head needs no serial task scope.
        for head_task in (N.unroll if H <= GROUPS else N.serial)(
            1 if H <= GROUPS else N.ceildiv(N.max(0, H - group), workers)
        ):
            head = group + head_task * workers
            scores = N.slotted_view(Score[group, :, :, :])
            probabilities = N.slotted_view(P[group, :, :, :])
            products = N.slotted_view(PV[group, :, :, :])

            # K and V stay resident across all query tiles of this head.
            with N.Cube():
                N.copy(K[head, :, :], k1)
                N.copy(k1, k0, transpose=True)
                N.copy(V[head, :, :], v1)
                N.copy(v1, v0)

            for tile in N.Dataflow(
                S // 64, issue_order={"cube": ("qk", "pv"), "vector": ("softmax", "store")}
            ):
                q = tile * 64

                with N.Cube(), N.Stage(0, name="qk"):
                    qk(Q[head, q : q + 64, :], k0, scores.at(tile), qk_buffers)

                with N.Vector() as lane, N.Stage(0, name="softmax"):
                    r = lane * 32
                    softmax(
                        scores.at(tile)[r : r + 32, :],
                        probabilities.at(tile)[r : r + 32, :],
                        softmax_buffers,
                    )

                with N.Cube(), N.Stage(1, name="pv"):
                    pv(probabilities.at(tile), v0, products.at(tile), pv_buffers)

                with N.Vector() as lane, N.Stage(1, name="store"):
                    r = lane * 32
                    store(
                        products.at(tile)[r : r + 32, :],
                        O[head, q + r : q + r + 32, :],
                        store_buffers,
                    )

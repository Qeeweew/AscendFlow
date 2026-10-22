"""Teaching example: aligned, non-causal attention with one resident KV tile.

Each head computes softmax(Q @ K.T / sqrt(128)) @ V: D=128, Sk=128, BQ=64.
S is a positive multiple of 64. FP16 Q/K/V/P/O, FP32 scores and reductions.
Score, P and PV are two-slot GM workspaces, private to each head.
"""

from __future__ import annotations

from ascendflow import language as N

# Dimensions in annotations are resolved by the DSL signature pass.
# ruff: noqa: F821


@N.inline
def qk(query, key, scores, buffers):
    q1, q0, score0 = buffers
    N.copy(query, q1)
    N.copy(q1, q0)
    N.matmul(q0, key, accum=score0, initialize=True)
    N.copy(score0, scores)


@N.inline
def softmax(scores, probabilities, buffers):
    score, probability, maximum, total, partial, broadcast = buffers
    N.copy(scores, score)
    N.mul(score, 0.08838834764831845, out=score)
    N.reduce_max(score, axis=1, out=maximum, scratch=partial[:, :2], order="r64_then_packed_merge")
    N.sub(score, maximum[:, None], out=score, scratch=broadcast)
    N.exp(score, out=score)
    N.reduce_sum(score, axis=1, out=total, scratch=partial[:, :2], order="r64_then_packed_merge")
    N.div(score, total[:, None], out=score, scratch=broadcast)
    N.cast(score, out=probability)
    N.copy(probability, probabilities)


@N.inline
def pv(probabilities, value, products, buffers):
    p1, p0, output0 = buffers
    N.copy(probabilities, p1)
    N.copy(p1, p0)
    N.matmul(p0, value, accum=output0, initialize=True)
    N.copy(output0, products)


@N.inline
def store(products, destination, buffers):
    output, result = buffers
    N.copy(products, output)
    N.cast(output, out=result)
    N.copy(result, destination)


@N.prim_func(disjoint_external=True)
def simple_attention(
    Q: N.Tensor((H, S, 128), N.f16),
    K: N.Tensor((H, 128, 128), N.f16),
    V: N.Tensor((H, 128, 128), N.f16),
    O: N.Tensor((H, S, 128), N.f16, access="write", write_disjoint=True),
    Score: N.Tensor((H, 2, 64, 128), N.f32, access="write"),
    P: N.Tensor((H, 2, 64, 128), N.f16, access="write"),
    PV: N.Tensor((H, 2, 64, 128), N.f32, access="write"),
    S: N.constexpr = 256,
    H: N.constexpr = 4,
):
    with N.Kernel(H, kind="mixed") as head:
        assert S > 0 and S % 64 == 0
        assert H > 0
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

        # K and V stay resident across all query tiles of this head.
        with N.Cube():
            N.copy(K[head, :, :], k1)
            N.copy(k1, k0, transpose=True)
            N.copy(V[head, :, :], v1)
            N.copy(v1, v0)

        for tile in N.Dataflow(S // 64):
            q = tile * 64
            slot = tile % 2

            with N.Cube(), N.Stage(0, name="qk"):
                qk(Q[head, q : q + 64, :], k0, Score[head, slot, :, :], qk_buffers)

            with N.Vector() as lane, N.Stage(0, name="softmax"):
                r = lane * 32
                softmax(
                    Score[head, slot, r : r + 32, :],
                    P[head, slot, r : r + 32, :],
                    softmax_buffers,
                )

            with N.Cube(), N.Stage(1, name="pv"):
                pv(P[head, slot, :, :], v0, PV[head, slot, :, :], pv_buffers)

            with N.Vector() as lane, N.Stage(1, name="store"):
                r = lane * 32
                store(PV[head, slot, r : r + 32, :], O[head, q + r : q + r + 32, :], store_buffers)

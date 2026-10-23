"""Chunk-ordered FP32 carry, history projection and low precision snapshots."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def scan_prefetch_vector(inputs, buffers, geometry, coordinates):
    Lengths, Gp, U = inputs
    last_gate, u16 = buffers
    (BV,) = geometry
    chunk, head, ring, lane = coordinates
    valid = N.load_index(Lengths[chunk : chunk + 1])
    N.copy(Gp[chunk, head, valid - 1, :], last_gate[ring, :])
    N.exp(last_gate[ring, :], out=last_gate[ring, :])
    N.copy(U[chunk, head, :, lane * BV : (lane + 1) * BV], u16[ring, :, :])


@N.inline
def scan_prefetch_cube(K, CK, W32, a32_1, chunk, head, ring, KG, kg1):
    for part in N.unroll(K // CK):
        N.copy(
            W32[chunk, head, :, part * CK : (part + 1) * CK],
            a32_1[ring, part, :, :],
        )
    N.copy(KG[chunk, head, :, :], kg1[ring, :, :])


@N.inline
def publish_history_and_snapshot(carry, History, seq, head, lane, BV, state16, Snapshot, chunk):
    N.copy(carry, History[seq, head, lane * BV : (lane + 1) * BV, :])
    N.cast(carry, out=state16)
    N.copy(state16, Snapshot[chunk, head, lane * BV : (lane + 1) * BV, :])


@N.inline
def project_history(inputs, buffers, geometry, coordinates):
    History, Projection = inputs
    b32_1, a32_0, b32_0, a32_1, score0 = buffers
    K, CK = geometry
    seq, head, ring, chunk = coordinates
    for part in N.unroll(K // CK):
        bb = N.copy(History[seq, head, :, part * CK : (part + 1) * CK], b32_1)
        aa = N.copy(a32_1[ring, part, :, :], a32_0)
        bb = N.copy(bb, b32_0, transpose=True)
        N.matmul(aa, bb, accum=score0, initialize=part == 0)
    N.copy(score0, Projection[chunk, head, :, :])


@N.inline
def form_residual(inputs, buffers, geometry, coordinates):
    Projection, Residual = inputs
    u16, other, scratch, r16 = buffers
    (BV,) = geometry
    ring, chunk, head, lane = coordinates
    N.cast(u16[ring, :, :], out=other)
    N.copy(Projection[chunk, head, :, lane * BV : (lane + 1) * BV], scratch)
    N.sub(other, scratch, out=scratch)
    N.cast(scratch, out=r16)
    N.copy(r16, Residual[chunk, head, :, lane * BV : (lane + 1) * BV])


@N.inline
def form_state_delta(inputs, buffers, coordinates):
    Residual, Delta = inputs
    res1, update_a0, right0, kg1, state0 = buffers
    chunk, head, ring = coordinates
    rr = N.copy(Residual[chunk, head, :, :], res1)
    aa = N.copy(rr, update_a0, transpose=True)
    bb = N.copy(kg1[ring, :, :], right0)
    N.matmul(aa, bb, accum=state0, initialize=True)
    N.copy(state0, Delta[chunk, head, :, :])


@N.inline
def update_fp32_carry(last_gate, ring, carry, update, Delta, chunk, head, lane, BV):
    decay = last_gate[ring, :]
    N.mul(carry, decay[None, :], out=carry)
    N.copy(Delta[chunk, head, lane * BV : (lane + 1) * BV, :], update)
    N.add(carry, update, out=carry)


@N.prim_func(disjoint_external=True)
def prefill_scan(
    U: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    KG: N.Tensor((CHUNKS, H, BT, K), DTYPE),
    Gp: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Initial: N.Tensor((SEQUENCES, H, K, K), N.f32),
    Final: N.Tensor((SEQUENCES, H, K, K), N.f32, access="write", write_disjoint=True),
    History: N.Tensor((SEQUENCES, H, K, K), N.f32, access="write"),
    Snapshot: N.Tensor((CHUNKS, H, K, K), DTYPE, access="write", write_disjoint=True),
    W32: N.Tensor((CHUNKS, H, BT, K), N.f32),
    Projection: N.Tensor((CHUNKS, H, BT, K), N.f32, access="write"),
    Residual: N.Tensor((CHUNKS, H, BT, K), DTYPE, access="write"),
    Delta: N.Tensor((CHUNKS, H, K, K), N.f32, access="write"),
    Lengths: N.Tensor((CHUNKS,), N.i64),
    SEQUENCES: N.constexpr = 1,
    NC: N.constexpr = 2,
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    K: N.constexpr = 128,
    BT: N.constexpr = 64,
    BV: N.constexpr = 64,
    CK: N.constexpr = 64,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.bf16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert K == 128 and BT == 64 and CK == 64 and BV == 64, "scan tile policy"
        assert CHUNKS == SEQUENCES * NC, "each sequence has NC nonempty chunks"
        carry = N.alloc_ub((BV, K), N.f32)
        state16 = N.alloc_ub((BV, K), DTYPE)
        u16 = N.alloc_ub((2, BT, BV), DTYPE)
        scratch = N.alloc_ub((BT, BV), N.f32)
        other = N.alloc_ub((BT, BV), N.f32)
        r16 = N.alloc_ub((BT, BV), DTYPE)
        update = N.alloc_ub((BV, K), N.f32)
        last_gate = N.alloc_ub((2, K), N.f32)
        a32_1 = N.alloc_l1((2, K // CK, BT, CK), N.f32)
        b32_1 = N.alloc_l1((K, CK), N.f32)
        a32_0 = N.alloc_l0a((BT, CK), N.f32)
        b32_0 = N.alloc_l0b((CK, K), N.f32)
        res1 = N.alloc_l1((BT, K), DTYPE)
        kg1 = N.alloc_l1((2, BT, K), DTYPE)
        update_a0 = N.alloc_l0a((K, BT), DTYPE)
        right0 = N.alloc_l0b((BT, K), DTYPE)
        score0 = N.alloc_l0c((BT, K), N.f32)
        state0 = N.alloc_l0c((K, K), N.f32)
        for task in N.serial(N.ceildiv(N.max(0, SEQUENCES * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            seq, head = flat // H, flat % H
            with N.Vector() as initial_lane:
                N.copy(Initial[seq, head, initial_lane * BV : (initial_lane + 1) * BV, :], carry)
            for local in N.Dataflow(NC):
                chunk = seq * NC + local
                ring = local % 2
                with N.Vector() as lane, N.Stage(0, name="prefetch_vector"):
                    scan_prefetch_vector(
                        (Lengths, Gp, U), (last_gate, u16), (BV,), (chunk, head, ring, lane)
                    )
                with N.Cube(), N.Stage(0, name="prefetch_cube"):
                    scan_prefetch_cube(K, CK, W32, a32_1, chunk, head, ring, KG, kg1)
                with N.Vector() as lane, N.Stage(1, name="history"):
                    publish_history_and_snapshot(
                        carry, History, seq, head, lane, BV, state16, Snapshot, chunk
                    )
                with N.Cube(), N.Stage(1, name="projection"):
                    project_history(
                        (History, Projection),
                        (b32_1, a32_0, b32_0, a32_1, score0),
                        (K, CK),
                        (seq, head, ring, chunk),
                    )
                with N.Vector() as lane, N.Stage(1, name="residual"):
                    form_residual(
                        (Projection, Residual),
                        (u16, other, scratch, r16),
                        (BV,),
                        (ring, chunk, head, lane),
                    )
                with N.Cube(), N.Stage(1, name="delta"):
                    form_state_delta(
                        (Residual, Delta),
                        (res1, update_a0, right0, kg1, state0),
                        (chunk, head, ring),
                    )
                with N.Vector() as lane, N.Stage(1, name="carry"):
                    update_fp32_carry(last_gate, ring, carry, update, Delta, chunk, head, lane, BV)
            with N.Vector() as final_lane:
                N.copy(carry, Final[seq, head, final_lane * BV : (final_lane + 1) * BV, :])

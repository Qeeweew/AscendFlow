"""Chunk-ordered FP32 carry, history projection and low precision snapshots."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821 - captured runtime and constexpr dimensions.


@N.inline
def scan_prefetch_vector(inputs, buffers):
    Lengths, Gp, U = inputs
    last_gate, u16 = buffers
    valid = N.load_index(Lengths)
    gate = N.copy(Gp[valid - 1, :], last_gate)
    decay = N.exp(gate, out=last_gate)
    value = N.copy(U, u16)
    return decay, value


@N.inline
def scan_prefetch_cube(K, CK, W32, a32_1, KG, kg1):
    for part in N.unroll(K // CK):
        N.copy(
            W32[:, part * CK : (part + 1) * CK],
            a32_1[part, :, :],
        )
    return N.copy(KG, kg1)


@N.inline
def publish_history_and_snapshot(carry, History, state16, Snapshot):
    N.copy(carry, History)
    snapshot = N.cast(carry, out=state16)
    N.copy(snapshot, Snapshot)


@N.inline
def project_history(inputs, buffers, geometry):
    History, Projection = inputs
    b32_1, a32_0, b32_0, a32_1, score0 = buffers
    K, CK = geometry
    for part in N.unroll(K // CK):
        bb = N.copy(History[:, part * CK : (part + 1) * CK], b32_1)
        aa = N.copy(a32_1[part, :, :], a32_0)
        bb = N.copy(bb, b32_0, transpose=True)
        result = N.matmul(aa, bb, accum=score0, initialize=part == 0)
    N.copy(result, Projection)


@N.inline
def form_residual(inputs, buffers):
    Projection, Residual = inputs
    u16, other, scratch, r16 = buffers
    value = N.cast(u16, out=other)
    projection = N.copy(Projection, scratch)
    residual = N.sub(value, projection, out=scratch)
    rounded = N.cast(residual, out=r16)
    N.copy(rounded, Residual)


@N.inline
def form_state_delta(inputs, buffers):
    Residual, Delta = inputs
    res1, update_a0, right0, kg1, state0 = buffers
    rr = N.copy(Residual, res1)
    aa = N.copy(rr, update_a0, transpose=True)
    bb = N.copy(kg1, right0)
    result = N.matmul(aa, bb, accum=state0, initialize=True)
    N.copy(result, Delta)


@N.inline
def update_fp32_carry(decay, carry, update, Delta):
    decayed = N.mul(carry, decay[None, :], out=carry)
    delta = N.copy(Delta, update)
    N.add(decayed, delta, out=carry)


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
        u16 = N.alloc_ub((BT, BV), DTYPE, slots=2)
        scratch = N.alloc_ub((BT, BV), N.f32)
        other = N.alloc_ub((BT, BV), N.f32)
        r16 = N.alloc_ub((BT, BV), DTYPE)
        update = N.alloc_ub((BV, K), N.f32)
        last_gate = N.alloc_ub((K,), N.f32, slots=2)
        a32_1 = N.alloc_l1((K // CK, BT, CK), N.f32, slots=2)
        b32_1 = N.alloc_l1((K, CK), N.f32)
        a32_0 = N.alloc_l0a((BT, CK), N.f32)
        b32_0 = N.alloc_l0b((CK, K), N.f32)
        res1 = N.alloc_l1((BT, K), DTYPE)
        kg1 = N.alloc_l1((BT, K), DTYPE, slots=2)
        update_a0 = N.alloc_l0a((K, BT), DTYPE)
        right0 = N.alloc_l0b((BT, K), DTYPE)
        score0 = N.alloc_l0c((BT, K), N.f32)
        state0 = N.alloc_l0c((K, K), N.f32)
        for task in N.serial(N.ceildiv(N.max(0, SEQUENCES * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            seq, head = flat // H, flat % H
            with N.Vector() as initial_lane:
                N.copy(Initial[seq, head, initial_lane * BV : (initial_lane + 1) * BV, :], carry)
            for local in N.Dataflow(
                NC,
                issue_order={
                    "cube": ("prefetch_cube", "projection", "delta"),
                    "vector": ("prefetch_vector", "history", "residual", "carry"),
                },
            ):
                chunk = seq * NC + local
                with N.Vector() as lane, N.Stage(0, name="prefetch_vector"):
                    decay, values = scan_prefetch_vector(
                        (
                            Lengths[chunk : chunk + 1],
                            Gp[chunk, head, :, :],
                            U[chunk, head, :, lane * BV : (lane + 1) * BV],
                        ),
                        (last_gate.at(local), u16.at(local)),
                    )
                with N.Cube(), N.Stage(0, name="prefetch_cube"):
                    keys = scan_prefetch_cube(
                        K,
                        CK,
                        W32[chunk, head, :, :],
                        a32_1.at(local),
                        KG[chunk, head, :, :],
                        kg1.at(local),
                    )
                with N.Vector() as lane, N.Stage(1, name="history"):
                    publish_history_and_snapshot(
                        carry,
                        History[seq, head, lane * BV : (lane + 1) * BV, :],
                        state16,
                        Snapshot[chunk, head, lane * BV : (lane + 1) * BV, :],
                    )
                with N.Cube(), N.Stage(1, name="projection"):
                    project_history(
                        (History[seq, head, :, :], Projection[chunk, head, :, :]),
                        (b32_1, a32_0, b32_0, a32_1.at(local), score0),
                        (K, CK),
                    )
                with N.Vector() as lane, N.Stage(1, name="residual"):
                    form_residual(
                        (
                            Projection[chunk, head, :, lane * BV : (lane + 1) * BV],
                            Residual[chunk, head, :, lane * BV : (lane + 1) * BV],
                        ),
                        (values, other, scratch, r16),
                    )
                with N.Cube(), N.Stage(1, name="delta"):
                    form_state_delta(
                        (Residual[chunk, head, :, :], Delta[chunk, head, :, :]),
                        (res1, update_a0, right0, keys, state0),
                    )
                with N.Vector() as lane, N.Stage(1, name="carry"):
                    update_fp32_carry(
                        decay, carry, update, Delta[chunk, head, lane * BV : (lane + 1) * BV, :]
                    )
            with N.Vector() as final_lane:
                N.copy(carry, Final[seq, head, final_lane * BV : (final_lane + 1) * BV, :])

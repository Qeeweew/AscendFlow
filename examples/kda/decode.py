"""Single-token KDA: a three-stage head pipeline and tiled FP32 updates.

Each group owns a sequence and a cyclic subset of heads. Q, K, decay and
beta are prepared once per head.
Whole-head import, tiled FP32 update, and whole-head export use three
stages across the cyclic head stream, with explicit physical slots. There are no user-written events or barriers.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.kda.ops import delta_step, normalize_qk

# ruff: noqa: F821 - captured constexpr annotation names.


@N.prim_func(disjoint_external=True)
def decode_kda(
    Q: N.Tensor((TOKENS, H, K), DTYPE),
    Key: N.Tensor((TOKENS, H, K), DTYPE),
    Value: N.Tensor((TOKENS, HV, V), DTYPE),
    Gate: N.Tensor((TOKENS, HV, K), GATE_DTYPE),
    Beta: N.Tensor((TOKENS, HV), BETA_DTYPE),
    State: N.Tensor((SLOTS, HV, V, K), STATE_DTYPE, access="read_write"),
    Snapshots: N.Tensor((TOKENS, HV, V, K), STATE_DTYPE, access="write", write_disjoint=True),
    Output: N.Tensor((TOKENS, HV, V), DTYPE, access="write", write_disjoint=True),
    Cu: N.Tensor((SEQUENCES + 1,), N.i64),
    Indices: N.Tensor((SEQUENCES, 1), N.i64),
    Accepted: N.Tensor((SEQUENCES,), N.i64),
    TOKENS: N.i64,
    SEQUENCES: N.constexpr = 128,
    SLOTS: N.constexpr = 129,
    HEAD_GROUPS: N.constexpr = 8,
    H: N.constexpr = 32,
    HV: N.constexpr = 32,
    K: N.constexpr = 128,
    V: N.constexpr = 128,
    BV: N.constexpr = 64,
    PIPE_SLOTS: N.constexpr = 2,
    DTYPE: N.constexpr = N.bf16,
    GATE_DTYPE: N.constexpr = N.f32,
    BETA_DTYPE: N.constexpr = N.f32,
    STATE_DTYPE: N.constexpr = N.f32,
    NORMALIZE: N.constexpr = True,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(SEQUENCES * HEAD_GROUPS, kind="vector") as owner:
        assert HV % HEAD_GROUPS == 0 and SEQUENCES * HEAD_GROUPS <= 1024, (
            "invalid head distribution"
        )
        assert HV % H == 0, "value heads must be a multiple of query heads"
        assert V % BV == 0 and BV % 16 == 0, "V partitions must contain complete b16 DMA blocks"
        assert K % 64 == 0 and K <= 512, "K requires complete supported reduction blocks"
        assert PIPE_SLOTS >= 2, "overlapped import/update requires at least two state slots"
        q = N.alloc_ub((PIPE_SLOTS, K), N.f32, name="q")
        k = N.alloc_ub((PIPE_SLOTS, K), N.f32, name="k")
        decay = N.alloc_ub((PIPE_SLOTS, K), N.f32, name="decay")
        norm_work = N.alloc_ub((K,), N.f32, name="norm_work")
        norm_partial = N.alloc_ub((N.max(8, K // 64),), N.f32, name="norm_partial")
        norm_stat = N.alloc_ub((8,), N.f32, name="norm_stat")
        states = N.alloc_ub((PIPE_SLOTS, V, K), N.f32, name="states")
        product = N.alloc_ub((BV, K), N.f32, name="product")
        reduction = N.alloc_ub((BV * (K // 8),), N.f32, name="reduction")
        projected = N.alloc_ub((BV,), N.f32, name="projected")
        row_scratch = N.alloc_ub((BV, 8), N.f32, name="row_scratch")
        inputs = N.alloc_ub((PIPE_SLOTS, V), DTYPE, name="inputs")
        outputs = N.alloc_ub((PIPE_SLOTS, V), N.f32, name="outputs")
        if DTYPE != N.f32:
            q_input = N.alloc_ub((PIPE_SLOTS, K), DTYPE, name="q_input")
            k_input = N.alloc_ub((PIPE_SLOTS, K), DTYPE, name="k_input")
            v_work = N.alloc_ub((BV,), N.f32, name="v_work")
            output_store = N.alloc_ub((PIPE_SLOTS, V), DTYPE, name="output_store")
        if STATE_DTYPE != N.f32:
            state_input = N.alloc_ub((PIPE_SLOTS, V, K), STATE_DTYPE, name="state_input")
            state_store = N.alloc_ub((PIPE_SLOTS, V, K), STATE_DTYPE, name="state_store")
        if GATE_DTYPE != N.f32:
            gate_input = N.alloc_ub((PIPE_SLOTS, K), GATE_DTYPE, name="gate_input")
        seq = owner // HEAD_GROUPS
        # This clock covers at most one token; slot addresses need no range hints.
        bos = N.load_index(Cu[seq : seq + 1], minimum=0, maximum=TOKENS)
        eos = N.load_index(Cu[seq + 1 : seq + 2], minimum=bos, maximum=N.min(TOKENS, bos + 1))
        slot = N.load_index(Indices[seq, :])
        if slot > 0:
            for token in N.serial(bos, eos):
                for item in N.Pipelined(
                    HV // HEAD_GROUPS,
                    issue_order=("store", "load", "compute"),
                ):
                    head = owner % HEAD_GROUPS + item * HEAD_GROUPS
                    qhead = head // (HV // H)
                    ring = item % PIPE_SLOTS
                    with N.Stage(0, name="load"):
                        if DTYPE == N.f32:
                            N.copy(Q[token, qhead, :], q[ring, :])
                            N.copy(Key[token, qhead, :], k[ring, :])
                        else:
                            N.copy(Q[token, qhead, :], q_input[ring, :])
                            N.copy(Key[token, qhead, :], k_input[ring, :])
                        if GATE_DTYPE == N.f32:
                            N.copy(Gate[token, head, :], decay[ring, :])
                        else:
                            N.copy(Gate[token, head, :], gate_input[ring, :])
                        if STATE_DTYPE == N.f32:
                            N.copy(State[slot, head, :, :], states[ring, :, :])
                        else:
                            N.copy(State[slot, head, :, :], state_input[ring, :, :])
                        N.copy(Value[token, head, :], inputs[ring, :])
                    with N.Vector(), N.Stage(1, name="compute"):
                        if DTYPE != N.f32:
                            N.cast(q_input[ring, :], out=q[ring, :])
                            N.cast(k_input[ring, :], out=k[ring, :])
                        if GATE_DTYPE != N.f32:
                            N.cast(gate_input[ring, :], out=decay[ring, :])
                        if NORMALIZE:
                            normalize_qk(
                                q[ring, :], k[ring, :], norm_work, norm_partial, norm_stat, K
                            )
                        N.mul(q[ring, :], SCALE, out=q[ring, :])
                        N.exp(decay[ring, :], out=decay[ring, :])
                        beta = N.load_scalar(Beta[token, head : head + 1])
                        for part in N.serial(V // BV):
                            row = part * BV
                            if STATE_DTYPE != N.f32:
                                N.cast(
                                    state_input[ring, row : row + BV, :],
                                    out=states[ring, row : row + BV, :],
                                )
                            if DTYPE == N.f32:
                                value = inputs[ring, row : row + BV]
                            else:
                                N.cast(inputs[ring, row : row + BV], out=v_work)
                                value = v_work
                            delta_step(
                                states[ring, row : row + BV, :],
                                q[ring, :],
                                k[ring, :],
                                decay[ring, :],
                                value,
                                beta,
                                product,
                                reduction,
                                projected,
                                row_scratch,
                                outputs[ring, row : row + BV],
                            )
                            if DTYPE != N.f32:
                                N.cast(
                                    outputs[ring, row : row + BV],
                                    out=output_store[ring, row : row + BV],
                                )
                            if STATE_DTYPE != N.f32:
                                N.cast(
                                    states[ring, row : row + BV, :],
                                    out=state_store[ring, row : row + BV, :],
                                )
                    with N.Stage(2, name="store"):
                        if DTYPE == N.f32:
                            N.copy(outputs[ring, :], Output[token, head, :])
                        else:
                            N.copy(output_store[ring, :], Output[token, head, :])
                        if STATE_DTYPE == N.f32:
                            N.copy(states[ring, :, :], State[slot, head, :, :])
                        else:
                            N.copy(state_store[ring, :, :], State[slot, head, :, :])

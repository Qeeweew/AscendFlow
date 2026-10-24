"""Token-pipelined KDA with a whole-head FP32 carry and per-token snapshots.

State is V-first, like the AscendC reference. Distinct sequences must
own disjoint positive state slots, including initial and snapshot slots.
Slot zero is the inactive sentinel. Sequence boundaries and slot IDs are
immutable device metadata, not host-expanded token schedules.
Each task owns a complete head, prefetches token inputs, and exports a
separate snapshot while the next token updates the FP32 carry.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.kda.ops import delta_step, normalize_qk

# ruff: noqa: F821 - constexpr names in captured DSL annotations.


@N.prim_func(disjoint_external=True)
def recurrent_kda(
    Q: N.Tensor((TOKENS, H, K), DTYPE),
    Key: N.Tensor((TOKENS, H, K), DTYPE),
    Value: N.Tensor((TOKENS, HV, V), DTYPE),
    Gate: N.Tensor((TOKENS, HV, K), GATE_DTYPE),
    Beta: N.Tensor((TOKENS, HV), BETA_DTYPE),
    State: N.Tensor((SLOTS, HV, V, K), STATE_DTYPE, access="read_write"),
    Snapshots: N.Tensor((TOKENS, HV, V, K), STATE_DTYPE, access="write", write_disjoint=True),
    Output: N.Tensor((TOKENS, HV, V), DTYPE, access="write", write_disjoint=True),
    Cu: N.Tensor((SEQUENCES + 1,), N.i64),
    Indices: N.Tensor((SEQUENCES, INDEX_WIDTH), N.i64),
    Accepted: N.Tensor((SEQUENCES,), N.i64),
    TOKENS: N.constexpr = 4,
    SEQUENCES: N.constexpr = 4,
    SLOTS: N.constexpr = 5,
    INDEX_WIDTH: N.constexpr = 1,
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
    INDEXED: N.constexpr = True,
    SPECULATIVE: N.constexpr = False,
    INPLACE: N.constexpr = True,
    USE_INITIAL: N.constexpr = True,
    SCALE: N.constexpr = 0.08838834764831845,
):
    with N.Kernel(SEQUENCES * HV, kind="vector") as task:
        assert HV % H == 0, "value heads must be a multiple of query heads"
        assert V % BV == 0 and BV % 16 == 0, "value partitions must contain whole b16 DMA blocks"
        assert K % 64 == 0 and K <= 1024, "K must contain complete vector reduction blocks"
        assert PIPE_SLOTS >= 2, "token prefetch requires at least two input slots"
        assert INDEXED or INPLACE == False, "inplace snapshots require explicit state indices"
        state = N.alloc_ub((V, K), N.f32, name="state")
        product = N.alloc_ub((BV, K), N.f32, name="product")
        q = N.alloc_ub((K,), N.f32, slots=PIPE_SLOTS, name="q")
        k = N.alloc_ub((K,), N.f32, slots=PIPE_SLOTS, name="k")
        g = N.alloc_ub((K,), N.f32, slots=PIPE_SLOTS, name="gate")
        norm_work = N.alloc_ub((K,), N.f32, name="norm_work")
        norm_partial = N.alloc_ub((N.max(8, K // 64),), N.f32, name="norm_partial")
        norm_stat = N.alloc_ub((8,), N.f32, name="norm_stat")
        inputs = N.alloc_ub((V,), DTYPE, slots=PIPE_SLOTS, name="inputs")
        reduction = N.alloc_ub(
            (BV * (K // 8 + (K // 64 if K > 512 else 0)),), N.f32, name="reduction"
        )
        projected = N.alloc_ub((BV,), N.f32, name="projected")
        out = N.alloc_ub((V,), N.f32, slots=PIPE_SLOTS, name="output")
        row_scratch = N.alloc_ub((BV, 8), N.f32, name="row_scratch")
        if DTYPE != N.f32:
            q_input = N.alloc_ub((K,), DTYPE, slots=PIPE_SLOTS, name="q_input")
            k_input = N.alloc_ub((K,), DTYPE, slots=PIPE_SLOTS, name="k_input")
            v_work = N.alloc_ub((BV,), N.f32, name="v_work")
            out_store = N.alloc_ub((V,), DTYPE, slots=PIPE_SLOTS, name="out_store")
        if GATE_DTYPE != N.f32:
            gate_input = N.alloc_ub((K,), GATE_DTYPE, slots=PIPE_SLOTS, name="gate_input")
        if STATE_DTYPE != N.f32:
            state_input = N.alloc_ub((V, K), STATE_DTYPE, name="state_input")
        state_store = N.alloc_ub((V, K), STATE_DTYPE, name="state_store")
        seq = task // HV
        head = task % HV
        qhead = head // (HV // H)
        # Bound the token clock for generated notification arithmetic. State
        # indices below use ordinary caller-owned access contracts.
        bos = N.load_index(Cu[seq : seq + 1], minimum=0, maximum=TOKENS)
        eos = N.load_index(Cu[seq + 1 : seq + 2], minimum=bos, maximum=TOKENS)
        if eos > bos:
            if INDEXED:
                if SPECULATIVE:
                    accepted = N.load_index(Accepted[seq : seq + 1])
                    initial_column = accepted - 1
                else:
                    initial_column = 0
                slot = N.load_index(Indices[seq, initial_column : initial_column + 1])
            else:
                slot = bos
            active = slot > 0 if INDEXED else True
            if active:
                if USE_INITIAL:
                    if STATE_DTYPE == N.f32:
                        N.copy(State[slot, head, :, :], state)
                    else:
                        N.copy(State[slot, head, :, :], state_input)
                        N.cast(state_input, out=state)
                else:
                    N.fill(state, 0.0)
                for item in N.Pipelined(eos - bos, issue_order=("store", "load", "compute")):
                    token = bos + item
                    with N.Stage(0, name="load"), N.CopyGroup(name="inputs"):
                        if DTYPE == N.f32:
                            q_value = N.copy(Q[token, qhead, :], q.at(item))
                            k_value = N.copy(Key[token, qhead, :], k.at(item))
                        else:
                            q_loaded = N.copy(Q[token, qhead, :], q_input.at(item))
                            k_loaded = N.copy(Key[token, qhead, :], k_input.at(item))
                        input_value = N.copy(Value[token, head, :], inputs.at(item))
                        if GATE_DTYPE == N.f32:
                            gate_value = N.copy(Gate[token, head, :], g.at(item))
                        else:
                            gate_loaded = N.copy(Gate[token, head, :], gate_input.at(item))
                    with N.Vector(), N.Stage(1, name="compute"):
                        if DTYPE != N.f32:
                            q_value = N.cast(q_loaded, out=q.at(item))
                            k_value = N.cast(k_loaded, out=k.at(item))
                        if GATE_DTYPE != N.f32:
                            gate_value = N.cast(gate_loaded, out=g.at(item))
                        if NORMALIZE:
                            q_value, k_value = normalize_qk(
                                q_value, k_value, norm_work, norm_partial, norm_stat, K
                            )
                        scaled_q = N.mul(q_value, SCALE, out=q.at(item))
                        decay_value = N.exp(gate_value, out=g.at(item))
                        beta = N.load_scalar(Beta[token, head : head + 1])
                        for part in N.serial(V // BV):
                            row = part * BV
                            if DTYPE == N.f32:
                                value = input_value[row : row + BV]
                            else:
                                value = N.cast(input_value[row : row + BV], out=v_work)
                            delta_step(
                                state[row : row + BV, :],
                                scaled_q,
                                k_value,
                                decay_value,
                                value,
                                beta,
                                product,
                                reduction,
                                projected,
                                row_scratch,
                                out.at(item)[row : row + BV],
                            )
                        if DTYPE != N.f32:
                            converted_output = N.cast(out.at(item), out=out_store.at(item))
                        # Retiring this snapshot must not block the next carry
                        # update or feed a low-precision export back into it.
                        if STATE_DTYPE != N.f32:
                            snapshot = N.cast(state, out=state_store)
                        else:
                            snapshot = N.copy(state, state_store)
                    with N.Stage(2, name="store"):
                        if DTYPE == N.f32:
                            N.copy(out.at(item), Output[token, head, :])
                        else:
                            N.copy(converted_output, Output[token, head, :])
                        if INPLACE:
                            destination = N.load_index(Indices[seq, token - bos : token - bos + 1])
                            if destination > 0:
                                N.copy(snapshot, State[destination, head, :, :])
                        else:
                            N.copy(snapshot, Snapshots[token, head, :, :])

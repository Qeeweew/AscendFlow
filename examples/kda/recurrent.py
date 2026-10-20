"""Token-ordered KDA with FP32 UB carry and independent state snapshots.

State is V-first, like vllm-ascend's Triton kernel. Distinct sequences must
own disjoint positive state slots, including initial and snapshot slots.
Slot zero is the inactive sentinel. Sequence boundaries and slot IDs are
immutable device metadata, not host-expanded token schedules.
"""

from __future__ import annotations

from ascendflow import language as N
from examples.kda.ops import normalize_qk

# ruff: noqa: F821 - constexpr names in captured DSL annotations.


@N.inline
def recurrent_update(inputs, buffers, geometry, coordinates):
    (Beta,) = inputs
    q, g, state, k, product, projected, reduction, v, update, row_scratch, out = buffers
    (SCALE,) = geometry
    token, head = coordinates
    N.mul(q, SCALE, out=q)
    N.exp(g, out=g)
    N.mul(state, g[None, :], out=state)
    N.mul(state, k[None, :], out=product)
    N.reduce_sum(
        product,
        axis=1,
        out=projected,
        scratch=reduction,
        order="block8_then_packed_merge",
    )
    N.sub(v, projected, out=v)
    beta = N.load_scalar(Beta[token, head : head + 1])
    N.mul(v, beta, out=v)
    N.broadcast(k[None, :], out=update)
    N.mul(update, v[:, None], out=update, scratch=row_scratch)
    N.add(state, update, out=state)
    N.mul(state, q[None, :], out=product)
    N.reduce_sum(
        product,
        axis=1,
        out=out,
        scratch=reduction,
        order="block8_then_packed_merge",
    )


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
    BV: N.constexpr = 16,
    DTYPE: N.constexpr = N.f16,
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
    with N.Kernel(SEQUENCES * HV * (V // BV), kind="vector") as task:
        assert HV % H == 0, "value heads must be a multiple of query heads"
        assert V % BV == 0 and BV % 16 == 0, "value partitions must contain whole b16 DMA blocks"
        assert K % 64 == 0 and K <= 1024, "K must contain complete vector reduction blocks"
        assert INDEXED or INPLACE == False, "inplace snapshots require explicit state indices"
        state = N.alloc_ub((BV, K), N.f32, name="state")
        product = N.alloc_ub((BV, K), N.f32, name="product")
        update = N.alloc_ub((BV, K), N.f32, name="update")
        q = N.alloc_ub((K,), N.f32, name="q")
        k = N.alloc_ub((K,), N.f32, name="k")
        g = N.alloc_ub((K,), N.f32, name="gate")
        norm_work = N.alloc_ub((K,), N.f32, name="norm_work")
        norm_partial = N.alloc_ub((N.max(8, K // 64),), N.f32, name="norm_partial")
        norm_stat = N.alloc_ub((8,), N.f32, name="norm_stat")
        v = N.alloc_ub((BV,), N.f32, name="v")
        reduction = N.alloc_ub(
            (BV * (K // 8 + (K // 64 if K > 512 else 0)),), N.f32, name="reduction"
        )
        projected = N.alloc_ub((BV,), N.f32, name="projected")
        out = N.alloc_ub((BV,), N.f32, name="output")
        row_scratch = N.alloc_ub((BV, 8), N.f32, name="row_scratch")
        if DTYPE != N.f32:
            q_input = N.alloc_ub((K,), DTYPE, name="q_input")
            k_input = N.alloc_ub((K,), DTYPE, name="k_input")
            v_input = N.alloc_ub((BV,), DTYPE, name="v_input")
            out_store = N.alloc_ub((BV,), DTYPE, name="out_store")
        if GATE_DTYPE != N.f32:
            gate_input = N.alloc_ub((K,), GATE_DTYPE, name="gate_input")
        if STATE_DTYPE != N.f32:
            state_input = N.alloc_ub((BV, K), STATE_DTYPE, name="state_input")
            state_store = N.alloc_ub((BV, K), STATE_DTYPE, name="state_store")
        seq = task // (HV * (V // BV))
        head = (task // (V // BV)) % HV
        qhead = head // (HV // H)
        col = (task % (V // BV)) * BV
        bos = N.load_index(Cu[seq : seq + 1], minimum=0, maximum=TOKENS)
        end_limit = N.min(TOKENS, bos + INDEX_WIDTH) if INPLACE else TOKENS
        eos = N.load_index(Cu[seq + 1 : seq + 2], minimum=bos, maximum=end_limit)
        if eos > bos:
            if INDEXED:
                if SPECULATIVE:
                    accepted = N.load_index(Accepted[seq : seq + 1], minimum=1, maximum=INDEX_WIDTH)
                    initial_column = accepted - 1
                else:
                    initial_column = 0
                slot = N.load_index(
                    Indices[seq, initial_column : initial_column + 1],
                    minimum=0,
                    maximum=SLOTS - 1,
                )
            else:
                slot = bos
            active = slot > 0 if INDEXED else True
            if active:
                if USE_INITIAL:
                    if STATE_DTYPE == N.f32:
                        N.copy(State[slot, head, col : col + BV, :], state)
                    else:
                        N.copy(State[slot, head, col : col + BV, :], state_input)
                        N.cast(state_input, out=state)
                else:
                    N.fill(state, 0.0)
                for token in N.serial(bos, eos):
                    if DTYPE == N.f32:
                        N.copy(Q[token, qhead, :], q)
                        N.copy(Key[token, qhead, :], k)
                        N.copy(Value[token, head, col : col + BV], v)
                    else:
                        N.copy(Q[token, qhead, :], q_input)
                        N.copy(Key[token, qhead, :], k_input)
                        N.copy(Value[token, head, col : col + BV], v_input)
                        N.cast(q_input, out=q)
                        N.cast(k_input, out=k)
                        N.cast(v_input, out=v)
                    if GATE_DTYPE == N.f32:
                        N.copy(Gate[token, head, :], g)
                    else:
                        N.copy(Gate[token, head, :], gate_input)
                        N.cast(gate_input, out=g)
                    if NORMALIZE:
                        normalize_qk(q, k, norm_work, norm_partial, norm_stat, K)
                    recurrent_update(
                        (Beta,),
                        (
                            q,
                            g,
                            state,
                            k,
                            product,
                            projected,
                            reduction,
                            v,
                            update,
                            row_scratch,
                            out,
                        ),
                        (SCALE,),
                        (token, head),
                    )
                    if DTYPE == N.f32:
                        N.copy(out, Output[token, head, col : col + BV])
                    else:
                        N.cast(out, out=out_store)
                        N.copy(out_store, Output[token, head, col : col + BV])
                    if INPLACE:
                        destination = N.load_index(
                            Indices[seq, token - bos : token - bos + 1],
                            minimum=0,
                            maximum=SLOTS - 1,
                        )
                        if destination > 0:
                            if STATE_DTYPE == N.f32:
                                N.copy(state, State[destination, head, col : col + BV, :])
                            else:
                                N.cast(state, out=state_store)
                                N.copy(state_store, State[destination, head, col : col + BV, :])
                    else:
                        if STATE_DTYPE == N.f32:
                            N.copy(state, Snapshots[token, head, col : col + BV, :])
                        else:
                            N.cast(state, out=state_store)
                            N.copy(state_store, Snapshots[token, head, col : col + BV, :])

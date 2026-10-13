"""Static f32 gated-delta recurrence: private carry and explicit snapshot rings.

The 64-key recipe retains the upstream arch22 WholeReduceSum and MulAddDst
instructions. Dynamic state IDs, ragged streams and lower precision snapshots
need separate contracts and are not silently substituted for this program.
"""

from math import sqrt

from .builder import Kernel
from .kernels import align32
from .model import require, round_f32


def recurrent_gdn(
    sequences, steps, *, key_dim=64, value_dim=64, slots=2, snapshot_slots=2, scale=None
):
    """Per-sequence S[t] = exp(g[t])*S[t-1] + k[t] outer delta[t].

    Physical state is f32 [sequences,value_dim,key_dim]. Q/K/V and output are
    sequence-major. G/Beta have align8(steps) f32 words per sequence, including
    explicit readable padding. The two outputs are f32 attention[seq,T,Dv] and
    snapshots[seq,T,Dv,Dk]; snapshots never replace the resident FP32 carry.
    """
    require(
        isinstance(sequences, int) and 1 <= sequences <= 1024,
        "GDN sequences must be a static independent grid in [1,1024]",
    )
    require(
        isinstance(steps, int) and 0 <= steps <= 128,
        "GDN requires a static sequence length in [0,128]",
    )
    require(
        isinstance(key_dim, int) and key_dim == 64,
        "GDN currently requires 64 integer keys to preserve native reduction order",
    )
    require(
        isinstance(value_dim, int) and 8 <= value_dim <= 128 and value_dim % 8 == 0,
        "GDN value dimension must be a multiple of eight in [8,128]",
    )
    require(
        all(isinstance(n, int) and 1 <= n <= 8 for n in (slots, snapshot_slots)),
        "GDN input and snapshot rings require explicit slot counts in [1,8]",
    )
    scale = round_f32(1 / sqrt(key_dim) if scale is None else scale)
    qb, vb = key_dim * 4, value_dim * 4
    sb, gb = value_dim * qb, max(32, align32(steps * 4))
    k = Kernel("recurrent_gdn", blocks=sequences, disjoint_external=True)
    q = k.buffer("q", "GM", sequences * steps * qb, external=True, initialized=True)
    keys = k.buffer("k", "GM", sequences * steps * qb, external=True, initialized=True)
    v = k.buffer("v", "GM", sequences * steps * vb, external=True, initialized=True)
    gamma = k.buffer("log_decay", "GM", sequences * gb, external=True, initialized=True)
    beta = k.buffer("beta", "GM", sequences * gb, external=True, initialized=True)
    initial = k.buffer("initial_state", "GM", sequences * sb, external=True, initialized=True)
    y = k.buffer("y", "GM", sequences * steps * vb, external=True)
    snapshots = k.buffer("snapshots", "GM", sequences * steps * sb, external=True)
    qu = k.buffer("q_slots", "UB", slots * qb)
    ku = k.buffer("k_slots", "UB", slots * qb)
    vu = k.buffer("v_slots", "UB", slots * vb)
    carry = k.buffer("carry", "UB", sb)
    work = k.buffer("product", "UB", sb)
    delta = k.buffer("delta", "UB", vb)
    replicated = k.buffer("replicated", "UB", value_dim * 32)
    gamma_cache = k.buffer("gamma_cache", "UB", gb)
    beta_cache = k.buffer("beta_cache", "UB", gb)
    yu = k.buffer("attention_slots", "UB", slots * vb)
    su = k.buffer("snapshot_slots", "UB", snapshot_slots * sb)
    k.loop(steps)
    qr = k.view(qu, qb, stride=qb, slots=slots)
    kr = k.view(ku, qb, stride=qb, slots=slots)
    vr = k.view(vu, vb, stride=vb, slots=slots)
    state = k.view(carry, sb)
    product = k.view(work, sb)
    dr = k.view(delta, vb)
    br = k.view(replicated, value_dim * 32)
    gr, betar = k.view(gamma_cache, gb), k.view(beta_cache, gb)
    attn = k.view(yu, vb, stride=vb, slots=slots)
    snapshot = k.view(su, sb, stride=sb, slots=snapshot_slots)
    k.task(
        "load_initial",
        "copy",
        k.view(initial, sb, core_stride=sb),
        state,
        active=(0, 1),
        issue_order=90,
    )
    k.task("load_g", "copy", k.view(gamma, gb, core_stride=gb), gr, active=(0, 1), issue_order=91)
    k.task(
        "load_beta", "copy", k.view(beta, gb, core_stride=gb), betar, active=(0, 1), issue_order=92
    )
    for name, source, dest, size, order in (
        ("q", q, qr, qb, 100),
        ("k", keys, kr, qb, 101),
        ("v", v, vr, vb, 102),
    ):
        k.task(
            "load_" + name,
            "copy",
            k.view(source, size, stride=size, core_stride=steps * size),
            dest,
            issue_order=order,
        )
    k.task("scale_q", "muls", qr, qr, scalar=scale, stage=1, issue_order=10)
    k.task("exp_g", "exp", gr, gr, active=(0, 1), stage=1, issue_order=11)
    decay = k.scalar_load("capture_g", k.view(gr, 4, stride=4), stage=1, issue_order=20)
    weight = k.scalar_load("capture_beta", k.view(betar, 4, stride=4), stage=1, issue_order=21)
    k.scale_value("decay_state", state, state, decay, stage=1, issue_order=30)
    k.task(
        "key_product",
        "mul_columns64",
        state,
        kr,
        product,
        params=(value_dim,),
        stage=1,
        issue_order=31,
    )
    k.task("key_sum", "reduce_rows64", product, dr, params=(value_dim,), stage=1, issue_order=32)
    k.task("delta_sub", "sub", vr, dr, dr, stage=1, issue_order=33)
    k.scale_value("delta_scale", dr, dr, weight, stage=1, issue_order=34)
    k.task("delta_blocks", "brcb8", dr, br, params=(value_dim,), stage=1, issue_order=35)
    k.task(
        "delta_broadcast",
        "broadcast_rows64",
        br,
        product,
        params=(value_dim,),
        stage=1,
        issue_order=36,
    )
    k.task(
        "update_state",
        "fma_columns64",
        product,
        kr,
        state,
        params=(value_dim,),
        stage=1,
        issue_order=37,
    )
    k.task(
        "query_product",
        "mul_columns64",
        state,
        qr,
        product,
        params=(value_dim,),
        stage=1,
        issue_order=38,
    )
    k.task(
        "query_sum", "reduce_rows64", product, attn, params=(value_dim,), stage=1, issue_order=39
    )
    k.task("capture_snapshot", "vector_copy", state, snapshot, stage=1, issue_order=40)
    # Both earlier stores must dispatch before the current Scalar gates.
    k.task(
        "store_y",
        "copy",
        attn,
        k.view(y, vb, stride=vb, core_stride=steps * vb),
        stage=2,
        issue_order=0,
    )
    k.task(
        "store_snapshot",
        "copy",
        snapshot,
        k.view(snapshots, sb, stride=sb, core_stride=steps * sb),
        stage=2,
        issue_order=1,
    )
    return k.finish()

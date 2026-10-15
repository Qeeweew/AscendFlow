"""A2-style C/V communication tiles, independently tiled Cube and Vector work.

The algorithm remains an example. Segments, physical storage, native calls
and fixed issue stages use only the compiler's public physical builder.
No event IDs, release hints or private verifier shortcuts are supplied here.
"""

from math import sqrt

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, f32

from xdsl_npu.builder import Kernel
from xdsl_npu.mixed import compose
from xdsl_npu.model import require, round_f32


def flash_attention_tiled(
    batch,
    query_heads,
    kv_heads,
    query_length,
    kv_length,
    head_dim,
    *,
    dtype="f16",
    block_q=128,
    block_kv=1024,
    cube_kv=128,
    vector_rows=8,
    gm_slots=2,
    l1_slots=2,
    softmax_scale=None,
    query_tiles_per_group=1,
    pipeline=True,
):
    """Dense MHA/GQA/MQA with explicit logical BK and complete native tiles.

    QK segments publish one logical score block. Each AIV visits its rows
    in vector_rows chunks, updating each row's m/l/O once per logical BK.
    P is rounded ONCE before PV; PV keeps FP32 Cube accumulation over the
    smaller physical K blocks. QK/PV have independent L0 operands and C.
    """
    shape = (batch, query_heads, kv_heads, query_length, kv_length, head_dim)
    require(
        all(type(n) is int and n > 0 for n in shape), "attention needs positive static dimensions"
    )
    require(query_heads % kv_heads == 0, "attention requires integral Hq/Hkv")
    require(dtype in {"f16", "bf16"} and head_dim in {64, 128}, "tiled attention needs b16 D64/128")
    require(
        block_q in {64, 128} and block_kv in {256, 512, 1024},
        "tiled attention supports BQ64/128, logical BK256/512/1024",
    )
    require(cube_kv in {64, 128} and block_kv % cube_kv == 0, "native KV must divide logical BK")
    require(vector_rows == 8, "initial tiled Vector policy requires eight rows per native tile")
    require(
        query_length % block_q == kv_length % block_kv == 0,
        "tiled attention requires complete static tiles",
    )
    require(
        gm_slots in {2, 3} and l1_slots in {1, 2, 3}, "tiled policy requires GM2/3, L1 rings1..3"
    )
    require(not pipeline or l1_slots >= 2, "V lookahead requires at least two declared L1 slots")
    scale = round_f32(1 / sqrt(head_dim) if softmax_scale is None else softmax_scale)
    epochs, chunks = kv_length // block_kv, block_kv // cube_kv
    half, vr_count = block_q // 2, block_q // 2 // vector_rows
    qt, ratio = query_length // block_q, query_heads // kv_heads
    require(
        query_tiles_per_group in {1, 2, 4} and qt % query_tiles_per_group == 0,
        "query groups require 1/2/4 complete consecutive tiles within each head",
    )
    grouped = query_tiles_per_group != 1
    qt //= query_tiles_per_group
    cube_query_period, vector_query_period = epochs * chunks, epochs * vr_count
    stream_epochs = epochs * query_tiles_per_group
    groups = batch * query_heads * qt
    require(
        groups <= 1024 and 2 * chunks + 1 <= 32,
        "tiled static grid or issue stage exceeds verified bounds",
    )
    qbytes, kbytes, pbytes = block_q * head_dim * 2, cube_kv * head_dim * 2, block_q * cube_kv * 2
    sb, pb, ob = block_q * block_kv * 4, block_q * block_kv * 2, block_q * head_dim * 4
    sizes = {
        "q": batch * query_heads * query_length * head_dim * 2,
        "k": batch * kv_heads * kv_length * head_dim * 2,
        "v": batch * kv_heads * kv_length * head_dim * 2,
        "scores": groups * gm_slots * sb,
        "probabilities": groups * gm_slots * pb,
        "pv": groups * gm_slots * ob,
        "output": batch * query_heads * query_length * head_dim * 2,
    }

    def arguments(kernel):
        return {
            name: kernel.buffer(
                name,
                "GM",
                size,
                dtype if name in {"q", "k", "v", "probabilities", "output"} else "f32",
                external=True,
                initialized=name in {"q", "k", "v"},
            )
            for name, size in sizes.items()
        }

    cube = Kernel("cube", "aic", blocks=groups, disjoint_external=True)
    gm = arguments(cube)
    ql1 = cube.buffer("q_cache", "L1", qbytes, dtype, layout="NZ")
    kl1 = cube.buffer("k_slots", "L1", l1_slots * kbytes, dtype, layout="NZ")
    vl1 = cube.buffer("v_slots", "L1", l1_slots * kbytes, dtype, layout="NZ")
    pl1 = cube.buffer("p_slots", "L1", l1_slots * pbytes, dtype, layout="NZ")
    qa = cube.buffer("q_operand", "L0A", qbytes, dtype, layout="ZZ")
    pa = cube.buffer("p_operand", "L0A", pbytes, dtype, layout="ZZ")
    kb = cube.buffer("k_operand", "L0B", kbytes, dtype, layout="ZN")
    vb = cube.buffer("v_operand", "L0B", kbytes, dtype, layout="ZN")
    qc = cube.buffer("qk_accum", "L0C", block_q * cube_kv * 4, layout="NZ")
    pc = cube.buffer("pv_accum", "L0C", ob, layout="NZ")
    cube.init_fixpipe()
    cube.loop(stream_epochs * chunks)
    qr = cube.view(
        gm["q"],
        head_dim * 2,
        rows=block_q,
        row_stride=head_dim * 2,
        core_stride=qbytes * query_tiles_per_group,
        stride=qbytes if grouped else 0,
        divisor=cube_query_period,
    )
    kr = cube.view(
        gm["k"],
        head_dim * 2,
        rows=cube_kv,
        row_stride=head_dim * 2,
        stride=kbytes,
        slots=cube_query_period if grouped else 0,
        core_stride=kv_length * head_dim * 2,
        core_divisor=qt * ratio,
    )
    vv = cube.view(
        gm["v"],
        head_dim * 2,
        rows=cube_kv,
        row_stride=head_dim * 2,
        stride=kbytes,
        slots=cube_query_period if grouped else 0,
        core_stride=kv_length * head_dim * 2,
        core_divisor=qt * ratio,
    )

    def cube_ring(root, size, width):
        parent = cube.view(
            gm[root],
            width,
            rows=block_q,
            row_stride=width,
            stride=size,
            slots=gm_slots,
            divisor=chunks,
            core_stride=gm_slots * size,
        )
        return parent

    scores = cube.view(
        cube_ring("scores", sb, block_kv * 4),
        cube_kv * 4,
        rows=block_q,
        row_stride=block_kv * 4,
        stride=cube_kv * 4,
        slots=chunks,
    )
    probabilities = cube.view(
        cube_ring("probabilities", pb, block_kv * 2),
        cube_kv * 2,
        rows=block_q,
        row_stride=block_kv * 2,
        stride=cube_kv * 2,
        slots=chunks,
    )
    pv = cube_ring("pv", ob, head_dim * 4)
    qcache, qop, pop = cube.view(ql1, qbytes), cube.view(qa, qbytes), cube.view(pa, pbytes)
    kc = cube.view(kl1, kbytes, stride=kbytes, slots=l1_slots)
    vc = cube.view(vl1, kbytes, stride=kbytes, slots=l1_slots)
    pp = cube.view(pl1, pbytes, stride=pbytes, slots=l1_slots)
    kop, vop = cube.view(kb, kbytes), cube.view(vb, kbytes)
    sc, oc = cube.view(qc, block_q * cube_kv * 4), cube.view(pc, ob)
    cube.task(
        "load_q",
        "nd2nz",
        qr,
        qcache,
        params=(block_q, head_dim),
        active=(0, -1) if grouped else (0, 1),
        period=cube_query_period if grouped else 1,
        issue_order=100,
    )
    cube.task(
        "operand_q",
        "load2d_a",
        qcache,
        qop,
        params=(block_q, head_dim),
        active=(0, -1) if grouped else (0, 1),
        period=cube_query_period if grouped else 1,
        stage=1 if grouped else 0,
        issue_order=101,
    )
    cube.task("load_k", "nd2nz", kr, kc, params=(cube_kv, head_dim), issue_order=200)
    cube.task(
        "operand_k", "load2d_b_nt", kc, kop, params=(cube_kv, head_dim), stage=1, issue_order=201
    )
    cube.task(
        "qk", "mad", qop, kop, sc, params=(block_q, cube_kv, head_dim, 1), stage=1, issue_order=202
    )
    cube.task(
        "export_scores", "fix", sc, scores, params=(block_q, cube_kv), stage=1, issue_order=203
    )
    # Issue the next QK chunk BEFORE the controller's P-ready gate. The
    # independent V input can be prefetched one chunk earlier with its own
    # last-MTE1-read credit. Two logical tiles of lead-in let the next FULL
    # QK tile be published while Vector computes the older P. Merely issuing
    # its first microchunk before the gate leaves Vector waiting for the rest.
    pv_stage = 2 * chunks + 1
    pv_order = 300 if pipeline else 0
    cube.task(
        "load_v",
        "nd2nz",
        vv,
        vc,
        params=(cube_kv, head_dim),
        stage=pv_stage - int(pipeline),
        issue_order=0,
        logical_phase=2,
    )
    cube.task(
        "load_p",
        "nd2nz",
        probabilities,
        pp,
        params=(block_q, cube_kv),
        stage=pv_stage,
        issue_order=pv_order + 1,
        logical_phase=2,
    )
    cube.task(
        "operand_p",
        "load2d_a",
        pp,
        pop,
        params=(block_q, cube_kv),
        stage=pv_stage,
        issue_order=pv_order + 2,
        logical_phase=2,
    )
    cube.task(
        "operand_v",
        "load2d_b_nn",
        vc,
        vop,
        params=(cube_kv, head_dim),
        stage=pv_stage,
        issue_order=pv_order + 3,
        logical_phase=2,
    )
    cube.task(
        "pv",
        "mad_cycle",
        pop,
        vop,
        oc,
        params=(block_q, head_dim, cube_kv, chunks),
        stage=pv_stage,
        issue_order=pv_order + 4,
        logical_phase=2,
    )
    cube.task(
        "export_pv",
        "fix",
        oc,
        pv,
        params=(block_q, head_dim),
        stage=pv_stage,
        issue_order=pv_order + 5,
        period=chunks,
        phase=chunks - 1,
        logical_phase=2,
    )

    vectors = []
    for owner in range(2):
        vector = Kernel("vector", blocks=groups, disjoint_external=True)
        vg = arguments(vector)
        r = vector_rows
        row_s, row_p, row_o = r * block_kv * 4, r * block_kv * 2, r * head_dim * 4
        bridge = vector.buffer("bridge_input", "UB", 2 * row_s)
        pv_input = vector.buffer("pv_input", "UB", 2 * row_o)
        work = vector.buffer("score_work", "UB", row_s)
        rounded = vector.buffer("p_output", "UB", 2 * row_p, dtype)
        carry = vector.buffer("output_carry", "UB", half * head_dim * 4)
        partial_pitch = max(32, (block_kv // 64) * 4)
        partials = vector.buffer("partials", "UB", r * partial_pitch)
        stat_roots = {
            name: vector.buffer(name, "UB", half * 4) for name in ("maximum", "denominator")
        }
        temp_roots = {
            name: vector.buffer(name, "UB", r * 4) for name in ("next_maximum", "row_sum", "ones")
        }
        alpha = vector.buffer("alpha", "UB", 2 * half * 4)
        broad = vector.buffer("expanded", "UB", r * 32)
        ab = vector.buffer("alpha_expanded", "UB", r * 32)
        invb = vector.buffer("inverse_expanded", "UB", r * 32) if grouped else broad
        inverse = vector.buffer("output_inverse", "UB", r * 4) if grouped else temp_roots["row_sum"]
        cast_output = (
            vector.buffer("output_rounded", "UB", 2 * r * head_dim * 2, dtype)
            if grouped
            else rounded
        )
        vector.loop(stream_epochs * vr_count)

        def vring(name, size, width, *, vector=vector, vg=vg, r=r, owner=owner):
            parent = vector.view(
                vg[name],
                width,
                rows=block_q,
                row_stride=width,
                stride=size,
                slots=gm_slots,
                divisor=vr_count,
                core_stride=gm_slots * size,
            )
            return vector.view(
                parent, r * width, offset=owner * half * width, stride=r * width, slots=vr_count
            )

        sgm, pgm, pvgm = (
            vring("scores", sb, block_kv * 4),
            vring("probabilities", pb, block_kv * 2),
            vring("pv", ob, head_dim * 4),
        )
        si = vector.view(bridge, row_s, stride=row_s, slots=2)
        vi = vector.view(pv_input, row_o, stride=row_o, slots=2)
        sw = vector.view(work, row_s)
        po = vector.view(rounded, row_p, stride=row_p, slots=2)
        out = vector.view(
            cast_output, r * head_dim * 2, stride=r * head_dim * 2 if grouped else row_p, slots=2
        )
        state = vector.view(carry, row_o, stride=row_o, slots=vr_count)
        part = vector.view(partials, block_kv // 64 * 4, rows=r, row_stride=partial_pitch)
        mx, den = (
            vector.view(stat_roots[name], r * 4, stride=r * 4, slots=vr_count)
            for name in ("maximum", "denominator")
        )
        nm, total, one = (
            vector.view(temp_roots[name], r * 4) for name in ("next_maximum", "row_sum", "ones")
        )
        aring = vector.view(alpha, half * 4, stride=half * 4, slots=2, divisor=vr_count)
        al = vector.view(aring, r * 4, stride=r * 4, slots=vr_count)
        br, abr = vector.view(broad, r * 32), vector.view(ab, r * 32)
        invbr = vector.view(invb, r * 32)
        inv = vector.view(inverse, r * 4)
        output_parent = vector.view(
            vg["output"],
            qbytes,
            core_stride=qbytes * query_tiles_per_group,
            stride=qbytes if grouped else 0,
            divisor=vector_query_period,
        )
        ogm = vector.view(
            output_parent,
            r * head_dim * 2,
            offset=owner * half * head_dim * 2,
            stride=r * head_dim * 2,
            slots=vr_count,
        )
        for uid, native, ref, scalar in (
            ("init_m", "fill_neg_inf", mx, 0),
            ("init_l", "fill", den, 0),
            ("init_o", "fill", state, 0),
        ):
            vector.task(
                uid,
                native,
                ref,
                active=(0, -1) if grouped else (0, vr_count),
                period=vector_query_period if grouped else 1,
                phase_width=vr_count if grouped else 1,
                stage=vr_count if grouped else 0,
                issue_order=50 + vector.num_tasks,
                scalar=scalar,
                logical_phase=1,
            )
        vector.task("init_one", "fill", one, active=(0, 1), scalar=1, logical_phase=1)

        def soft(uid, native, *refs, vector=vector, **kwargs):
            return vector.task(
                uid,
                native,
                *refs,
                stage=vr_count,
                issue_order=100 + vector.num_tasks,
                logical_phase=1,
                **kwargs,
            )

        soft("import_scores", "copy", sgm, si)
        soft("scale_scores", "muls", si, sw, scalar=scale)
        soft("partial_max", "reduce_max64", sw, part, params=(r, block_kv))
        soft("tile_max", "merge_max_packed", part, nm, params=(r, block_kv // 64))
        soft("next_max", "max", mx, nm, nm)
        soft("max_delta", "sub", mx, nm, al)
        soft("alpha", "exp", al, al)
        soft("broadcast_max", "brcb8", nm, br, params=(r,))
        soft("shift_scores", "sub_row_broadcast", sw, br, sw, params=(r, block_kv))
        soft("exponential", "exp", sw, sw)
        soft("round_p", "convert_rn", sw, po)
        soft("export_p", "copy", po, pgm)
        soft("partial_sum", "reduce_sum64", sw, part, params=(r, block_kv))
        soft("tile_sum", "merge_sum_packed", part, total, params=(r, block_kv // 64))
        soft("rescale_l", "mul", den, al, den)
        soft("update_l", "add", den, total, den)
        soft("update_m", "vector_copy", nm, mx)

        def update(uid, native, *refs, vector=vector, final=False, **kwargs):
            if not grouped:
                return vector.task(
                    uid,
                    native,
                    *refs,
                    stage=2 * vr_count,
                    issue_order=300 + vector.num_tasks,
                    logical_phase=3,
                    active=(vector_query_period - vr_count, vector_query_period)
                    if final
                    else (0, -1),
                    **kwargs,
                )
            for tail in (True,) if final else (False, True):
                width = vr_count if tail else vector_query_period - vr_count
                if not width:
                    continue
                vector.task(
                    uid + ("_tail" if tail else ""),
                    native,
                    *refs,
                    stage=2 * vr_count,
                    issue_order=(0 if tail else 300) + vector.num_tasks,
                    logical_phase=3,
                    period=vector_query_period,
                    phase=vector_query_period - vr_count if tail else 0,
                    phase_width=width,
                    **kwargs,
                )

        update("broadcast_alpha", "brcb8", al, abr, params=(r,))
        update("rescale_o", "mul_row_broadcast", state, abr, state, params=(r, head_dim))
        update("import_pv", "copy", pvgm, vi)
        update("update_o", "add", state, vi, state)
        update("reciprocal", "div", one, den, inv, final=True)
        update("broadcast_inv", "brcb8", inv, invbr, params=(r,), final=True)
        update(
            "normalize", "mul_row_broadcast", state, invbr, state, params=(r, head_dim), final=True
        )
        update("round_output", "convert_rn", state, out, final=True)
        update("store", "copy", out, ogm, final=True)
        vectors.append(vector.finish())
    result = compose(
        "flash_attention",
        cube.finish(),
        *vectors,
        cube_epoch=chunks,
        vector_epoch=vr_count,
        disjoint_external=True,
    )
    result.body.block.first_op.attributes.update(
        attention_shape=ArrayAttr(IntAttr(n) for n in shape),
        attention_tile=ArrayAttr(IntAttr(n) for n in (block_q, block_kv, gm_slots, l1_slots)),
        attention_native_tile=ArrayAttr(IntAttr(n) for n in (cube_kv, vector_rows)),
        attention_scale=FloatAttr(scale, f32),
        attention_causal=IntAttr(0),
        attention_start=IntAttr(0),
        attention_packed_decode=IntAttr(0),
        attention_query_tiles_per_group=IntAttr(query_tiles_per_group),
        attention_pipeline=IntAttr(int(pipeline)),
    )
    return result


def main():
    import argparse
    import json
    from pathlib import Path

    from xdsl_npu.cli import build_module

    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("batch", 1),
        ("query-heads", 8),
        ("kv-heads", 8),
        ("query-length", 4096),
        ("kv-length", 4096),
        ("head-dim", 128),
        ("block-q", 128),
        ("block-kv", 1024),
        ("cube-kv", 128),
        ("vector-rows", 8),
        ("gm-slots", 2),
        ("l1-slots", 2),
        ("query-tiles-per-group", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--no-pipeline", dest="pipeline", action="store_false")
    args = vars(parser.parse_args())
    output, compile_asc, shared = args.pop("output"), args.pop("compile"), args.pop("shared")
    print(
        json.dumps(
            build_module(
                flash_attention_tiled(**args), output, compile_asc=compile_asc, shared=shared
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

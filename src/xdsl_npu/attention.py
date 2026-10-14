"""Online forward attention recipes; all Cube, Vector and bridge effects are IR.

Q/K/V/O are BNSD. A mixed group owns one query tile, its private FP32
statistics/output carry and three bounded GM rings. K/V stay in their original
layout. P is explicitly rounded before PV; the denominator reads unrounded P.
No event, global barrier or scheduling certificate is supplied by this recipe.
"""

from math import sqrt

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, f32

from .builder import Kernel
from .mixed import compose
from .model import require, round_f32


def _flash_attention_group(
    batch,
    query_heads,
    kv_heads,
    query_length,
    kv_length,
    head_dim,
    *,
    dtype="f16",
    block_q=64,
    block_kv=128,
    gm_slots=2,
    local_slots=2,
    softmax_scale=None,
    q_cache="l0",
    reuse_ub=False,
    pipeline=False,
    packed_reductions=False,
    causal=False,
    query_start=0,
    _query_first=0,
    _query_tiles=None,
    _kv_tiles=None,
    _group_offset=0,
    _total_groups=None,
    _decode_prefix=None,
):
    """Dense MHA/GQA/MQA, with finite inputs and complete static native tiles.

    FP32 QK, row max, exp, denominator and O carry. Each multiply/add of
    the online update remains a distinct FP32 instruction. This first recipe
    accepts D64/128 and BQ32/64/128, BK64/128; unsupported tails fail explicitly.
    Each group owns a query tile; persistent/dynamic ownership is separate.
    """
    shape = (batch, query_heads, kv_heads, query_length, kv_length, head_dim)
    require(
        all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape),
        "attention dimensions must be positive static integers",
    )
    require(query_heads % kv_heads == 0, "attention requires an integer Hq/Hkv ratio")
    require(dtype in {"f16", "bf16"}, "attention operands/output must be f16 or bf16")
    require(
        block_q in {32, 64, 128} and block_kv in {64, 128},
        "attention supports BQ32/64/128, BK64/128",
    )
    require(q_cache in {"l0", "l1"}, "attention Q cache must be l0 or l1")
    require(
        all(isinstance(v, bool) for v in (reuse_ub, pipeline, packed_reductions)),
        "attention recipe policies must be boolean",
    )
    require(
        not pipeline or local_slots == 2,
        "lookahead attention pipeline requires two local slots; choose its issue policy explicitly",
    )
    require(head_dim in {64, 128}, "attention supports D64/128")
    require(
        query_length % block_q == kv_length % block_kv == 0,
        "attention currently requires complete Q/KV tiles; no masked or padded fallback",
    )
    require(
        isinstance(gm_slots, int)
        and not isinstance(gm_slots, bool)
        and isinstance(local_slots, int)
        and not isinstance(local_slots, bool)
        and gm_slots in {1, 2, 3}
        and local_slots in {1, 2},
        "attention supports GM rings1..3 and local rings1..2",
    )
    scale = round_f32(1 / sqrt(head_dim) if softmax_scale is None else softmax_scale)
    qt, epochs = query_length // block_q, kv_length // block_kv
    partitioned = _query_tiles is not None
    qt = qt if _query_tiles is None else _query_tiles
    epochs = epochs if _kv_tiles is None else _kv_tiles
    groups = batch * query_heads * qt
    require(groups <= 1024, "attention query grid exceeds 1024; use a verified persistent recipe")
    ratio, rows = query_heads // kv_heads, block_q // 2
    qbytes, kbytes = block_q * head_dim * 2, block_kv * head_dim * 2
    pbytes, sbytes, obytes = block_q * block_kv * 2, block_q * block_kv * 4, block_q * head_dim * 4
    workspace_groups = groups if _total_groups is None else _total_groups
    sizes = {
        "q": batch * query_heads * query_length * head_dim * 2,
        "k": batch * kv_heads * kv_length * head_dim * 2,
        "v": batch * kv_heads * kv_length * head_dim * 2,
        "scores": workspace_groups * gm_slots * sbytes,
        "probabilities": workspace_groups * gm_slots * pbytes,
        "pv": workspace_groups * gm_slots * obytes,
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

    cube = Kernel("attention_cube", "aic", blocks=groups, disjoint_external=True)
    gm = arguments(cube)
    ql1 = cube.buffer("q_cache", "L1", qbytes, dtype, layout="NZ")
    kl1 = cube.buffer("k_slots", "L1", local_slots * kbytes, dtype, layout="NZ")
    vl1 = cube.buffer("v_slots", "L1", local_slots * kbytes, dtype, layout="NZ")
    pl1 = cube.buffer("p_slots", "L1", local_slots * pbytes, dtype, layout="NZ")
    if q_cache == "l0":
        qa = cube.buffer("q_operand", "L0A", qbytes, dtype, layout="ZZ")
        pa = cube.buffer("p_operand", "L0A", local_slots * pbytes, dtype, layout="ZZ")
    else:
        qa = pa = cube.buffer(
            "a_operand", "L0A", local_slots * max(qbytes, pbytes), dtype, layout="ZZ"
        )
    b = cube.buffer("b_operand", "L0B", local_slots * kbytes, dtype, layout="ZN")
    cbytes = max(sbytes, obytes)
    c = cube.buffer("c_slots", "L0C", local_slots * cbytes, layout="NZ")
    cube.init_fixpipe()
    cube.loop(epochs)
    if partitioned:
        qrows = cube.view(
            gm["q"],
            head_dim * 2,
            rows=qt * block_q,
            row_stride=head_dim * 2,
            offset=_query_first * head_dim * 2,
            core_stride=query_length * head_dim * 2,
            core_divisor=qt,
        )
        qr = cube.view(
            qrows,
            head_dim * 2,
            rows=block_q,
            row_stride=head_dim * 2,
            core_stride=qbytes,
            core_modulus=qt,
        )
    else:
        qr = cube.view(
            gm["q"], head_dim * 2, rows=block_q, row_stride=head_dim * 2, core_stride=qbytes
        )
    kr = cube.view(
        gm["k"],
        head_dim * 2,
        rows=block_kv,
        row_stride=head_dim * 2,
        stride=kbytes,
        core_stride=kv_length * head_dim * 2,
        core_divisor=qt * ratio,
    )
    vr = cube.view(
        gm["v"],
        head_dim * 2,
        rows=block_kv,
        row_stride=head_dim * 2,
        stride=kbytes,
        core_stride=kv_length * head_dim * 2,
        core_divisor=qt * ratio,
    )
    sr = cube.view(
        gm["scores"],
        block_kv * 4,
        rows=block_q,
        row_stride=block_kv * 4,
        stride=sbytes,
        slots=gm_slots,
        core_stride=gm_slots * sbytes,
        offset=_group_offset * gm_slots * sbytes,
    )
    pr = cube.view(
        gm["probabilities"],
        block_kv * 2,
        rows=block_q,
        row_stride=block_kv * 2,
        stride=pbytes,
        slots=gm_slots,
        core_stride=gm_slots * pbytes,
        offset=_group_offset * gm_slots * pbytes,
    )
    pvr = cube.view(
        gm["pv"],
        head_dim * 4,
        rows=block_q,
        row_stride=head_dim * 4,
        stride=obytes,
        slots=gm_slots,
        core_stride=gm_slots * obytes,
        offset=_group_offset * gm_slots * obytes,
    )
    qcache = cube.view(ql1, qbytes)
    qoperand = cube.view(
        qa,
        qbytes,
        stride=max(qbytes, pbytes) if q_cache == "l1" else 0,
        slots=local_slots if q_cache == "l1" else 0,
    )
    kc = cube.view(kl1, kbytes, stride=kbytes, slots=local_slots)
    vc = cube.view(vl1, kbytes, stride=kbytes, slots=local_slots)
    pc = cube.view(pl1, pbytes, stride=pbytes, slots=local_slots)
    pp = cube.view(
        pa, pbytes, stride=max(qbytes, pbytes) if q_cache == "l1" else pbytes, slots=local_slots
    )
    bp = cube.view(b, kbytes, stride=kbytes, slots=local_slots)
    sc = cube.view(c, sbytes, stride=cbytes, slots=local_slots)
    oc = cube.view(c, obytes, stride=cbytes, slots=local_slots)
    cube.task(
        "load_q", "nd2nz", qr, qcache, params=(block_q, head_dim), active=(0, 1), issue_order=100
    )
    cube.task(
        "operand_q",
        "load2d_a",
        qcache,
        qoperand,
        params=(block_q, head_dim),
        active=(0, 1) if q_cache == "l0" else (0, -1),
        stage=0 if q_cache == "l0" else 1,
        issue_order=101 if q_cache == "l0" else 99,
    )
    cube.task(
        "load_k", "nd2nz", kr, kc, params=(block_kv, head_dim), issue_order=400 if pipeline else 200
    )
    # A single V slot cannot accept V[g+1] ahead of PV[g]'s MTE1 read:
    # that would block the MTE2 queue ahead of P[g] and form a local cycle.
    # This is an explicit recipe issue policy, chosen before AutoSync; slot
    # count is preserved and no synchronization pass moves an instruction.
    cube.task(
        "load_v",
        "nd2nz",
        vr,
        vc,
        params=(block_kv, head_dim),
        stage=1 if local_slots == 1 else 0,
        issue_order=99 if local_slots == 1 else (401 if pipeline else 201),
    )
    cube.task(
        "operand_k", "load2d_b_nt", kc, bp, params=(block_kv, head_dim), stage=1, issue_order=100
    )
    cube.task(
        "qk",
        "mad",
        qoperand,
        bp,
        sc,
        params=(block_q, block_kv, head_dim, 1),
        stage=1,
        issue_order=101,
    )
    cube.task("export_scores", "fix", sc, sr, params=(block_q, block_kv), stage=1, issue_order=102)
    cube.task(
        "load_p",
        "nd2nz",
        pr,
        pc,
        params=(block_q, block_kv),
        stage=2,
        issue_order=200 if pipeline else 0,
        logical_phase=2,
    )
    cube.task(
        "operand_p",
        "load2d_a",
        pc,
        pp,
        params=(block_q, block_kv),
        stage=2,
        issue_order=201 if pipeline else 1,
        logical_phase=2,
    )
    cube.task(
        "operand_v",
        "load2d_b_nn",
        vc,
        bp,
        params=(block_kv, head_dim),
        stage=2,
        issue_order=202 if pipeline else 2,
        logical_phase=2,
    )
    cube.task(
        "pv",
        "mad",
        pp,
        bp,
        oc,
        params=(block_q, head_dim, block_kv, 1),
        stage=2,
        issue_order=203 if pipeline else 3,
        logical_phase=2,
    )
    cube.task(
        "export_pv",
        "fix",
        oc,
        pvr,
        params=(block_q, head_dim),
        stage=2,
        issue_order=204 if pipeline else 4,
        logical_phase=2,
    )

    vectors = []
    for owner in range(2):
        vector = Kernel(f"attention_vector{owner}", blocks=groups, disjoint_external=True)
        vg = arguments(vector)
        sb, pb, ob = rows * block_kv * 4, rows * block_kv * 2, rows * head_dim * 4
        input_pitch = max(sb, ob) if reuse_ub else sb
        p_pitch = max(pb, rows * head_dim * 2) if reuse_ub else pb
        inp = vector.buffer(
            "bridge_input" if reuse_ub else "score_input", "UB", local_slots * input_pitch
        )
        work = vector.buffer("score_work", "UB", sb)
        prob = vector.buffer("p_output", "UB", local_slots * p_pitch, dtype)
        pv_input = inp if reuse_ub else vector.buffer("pv_input", "UB", local_slots * ob)
        output_carry = vector.buffer("output_carry", "UB", ob)
        out = (
            prob if reuse_ub else vector.buffer("output_rounded", "UB", rows * head_dim * 2, dtype)
        )
        partials = vector.buffer(
            "partials", "UB", rows * (block_kv // 64) * 4 if packed_reductions else rows * 32
        )
        maximum = vector.buffer("maximum", "UB", rows * 4)
        next_max = vector.buffer("next_maximum", "UB", rows * 4)
        denominator = vector.buffer("denominator", "UB", rows * 4)
        row_sum = vector.buffer("row_sum", "UB", rows * 4)
        alpha = vector.buffer("alpha", "UB", local_slots * rows * 4)
        ones = vector.buffer("ones", "UB", rows * 4)
        expanded = vector.buffer("expanded", "UB", rows * 32)
        alpha_expanded = vector.buffer("alpha_expanded", "UB", rows * 32)
        vector.loop(epochs)
        si = vector.view(inp, sb, stride=input_pitch, slots=local_slots)
        sw = vector.view(work, sb)
        po = vector.view(prob, pb, stride=p_pitch, slots=local_slots)
        vi = vector.view(pv_input, ob, stride=input_pitch if reuse_ub else ob, slots=local_slots)
        carry = vector.view(output_carry, ob)
        rounded = vector.view(
            out,
            rows * head_dim * 2,
            stride=p_pitch if reuse_ub else 0,
            slots=local_slots if reuse_ub else 0,
        )
        if packed_reductions:
            part = vector.view(partials, rows * (block_kv // 64) * 4)
            planes = [
                vector.view(part, rows * 4, offset=i * rows * 4) for i in range(block_kv // 64)
            ]
        else:
            part = vector.view(partials, block_kv // 64 * 4, rows=rows, row_stride=32)
        mx, nm, den, total, al, one = (
            vector.view(
                root,
                rows * 4,
                stride=rows * 4 if root is alpha else 0,
                slots=local_slots if root is alpha else 0,
            )
            for root in (maximum, next_max, denominator, row_sum, alpha, ones)
        )
        broad, ab = vector.view(expanded, rows * 32), vector.view(alpha_expanded, rows * 32)
        sgm = vector.view(
            vg["scores"],
            sb,
            offset=_group_offset * gm_slots * sbytes + owner * sb,
            stride=sbytes,
            slots=gm_slots,
            core_stride=gm_slots * sbytes,
        )
        pgm = vector.view(
            vg["probabilities"],
            pb,
            offset=_group_offset * gm_slots * pbytes + owner * pb,
            stride=pbytes,
            slots=gm_slots,
            core_stride=gm_slots * pbytes,
        )
        vgm = vector.view(
            vg["pv"],
            ob,
            offset=_group_offset * gm_slots * obytes + owner * ob,
            stride=obytes,
            slots=gm_slots,
            core_stride=gm_slots * obytes,
        )
        if partitioned:
            output_rows = vector.view(
                vg["output"],
                head_dim * 2,
                rows=qt * block_q,
                row_stride=head_dim * 2,
                offset=_query_first * head_dim * 2,
                core_stride=query_length * head_dim * 2,
                core_divisor=qt,
            )
            ogm = vector.view(
                output_rows,
                rows * head_dim * 2,
                offset=owner * rows * head_dim * 2,
                core_stride=qbytes,
                core_modulus=qt,
            )
        else:
            ogm = vector.view(
                vg["output"],
                rows * head_dim * 2,
                offset=owner * rows * head_dim * 2,
                core_stride=qbytes,
            )

        def soft(uid, native, *refs, vector=vector, **kwargs):
            return vector.task(
                uid,
                native,
                *refs,
                stage=1,
                issue_order=(10 if pipeline else 100) + vector.num_tasks,
                logical_phase=1,
                **kwargs,
            )

        for uid, native, ref, value in (
            ("init_m", "fill_neg_inf", mx, 0),
            ("init_l", "fill", den, 0),
            ("init_o", "fill", carry, 0),
            ("init_one", "fill", one, 1),
        ):
            vector.task(
                uid,
                native,
                ref,
                active=(0, 1),
                scalar=value,
                logical_phase=1,
                issue_order=10 + vector.num_tasks,
            )
        soft("import_scores", "copy", sgm, si)
        soft("scale_scores", "muls", si, sw, scalar=scale)
        if causal:
            if _decode_prefix is not None:
                if _decode_prefix % block_kv:
                    soft(
                        "causal_mask",
                        "mask_prefix",
                        sw,
                        params=(rows, block_kv, _decode_prefix),
                        active=(epochs - 1, epochs),
                    )
            else:
                soft(
                    "causal_mask",
                    "mask_causal",
                    sw,
                    params=(rows, block_kv, block_q, qt, _query_first, owner * rows, query_start),
                    active=(min(epochs, (query_start + _query_first + 1) // block_kv), epochs),
                )
        soft(
            "partial_max",
            "reduce_tile_max64" if packed_reductions else "reduce_max64",
            sw,
            part,
            params=(rows, block_kv),
        )
        if packed_reductions:
            soft("tile_max", "vector_copy" if block_kv == 64 else "max", *planes, nm)
        else:
            soft("tile_max", "merge_max_packed", part, nm, params=(rows, block_kv // 64))
        soft("next_max", "max", mx, nm, nm)
        soft("max_delta", "sub", mx, nm, al)
        soft("alpha", "exp", al, al)
        soft("broadcast_max", "brcb8", nm, broad, params=(rows,))
        soft("shift_scores", "sub_row_broadcast", sw, broad, sw, params=(rows, block_kv))
        soft("exponential", "exp", sw, sw)
        soft("round_p", "convert_rn", sw, po)
        soft("export_p", "copy", po, pgm)
        soft(
            "partial_sum",
            "reduce_tile_sum64" if packed_reductions else "reduce_sum64",
            sw,
            part,
            params=(rows, block_kv),
        )
        if packed_reductions:
            soft("tile_sum", "vector_copy" if block_kv == 64 else "add", *planes, total)
        else:
            soft("tile_sum", "merge_sum_packed", part, total, params=(rows, block_kv // 64))
        soft("rescale_l", "mul", den, al, den)
        soft("update_l", "add", den, total, den)
        soft("update_m", "vector_copy", nm, mx)
        vector.task(
            "broadcast_alpha",
            "brcb8",
            al,
            ab,
            params=(rows,),
            stage=2,
            issue_order=300 if pipeline else 0,
            logical_phase=3,
        )
        vector.task(
            "rescale_o",
            "mul_row_broadcast",
            carry,
            ab,
            carry,
            params=(rows, head_dim),
            stage=2,
            issue_order=301 if pipeline else 1,
            logical_phase=3,
        )
        vector.task(
            "import_pv",
            "copy",
            vgm,
            vi,
            stage=2,
            issue_order=302 if pipeline else 2,
            logical_phase=3,
        )
        vector.task(
            "update_o",
            "add",
            carry,
            vi,
            carry,
            stage=2,
            issue_order=303 if pipeline else 3,
            logical_phase=3,
        )
        last = (epochs - 1, epochs)
        vector.task(
            "reciprocal",
            "div",
            one,
            den,
            total,
            stage=2,
            active=last,
            issue_order=304 if pipeline else 4,
            logical_phase=3,
        )
        vector.task(
            "broadcast_inv",
            "brcb8",
            total,
            broad,
            params=(rows,),
            stage=2,
            active=last,
            issue_order=305 if pipeline else 5,
            logical_phase=3,
        )
        vector.task(
            "normalize",
            "mul_row_broadcast",
            carry,
            broad,
            carry,
            params=(rows, head_dim),
            stage=2,
            active=last,
            issue_order=306 if pipeline else 6,
            logical_phase=3,
        )
        vector.task(
            "round_output",
            "convert_rn",
            carry,
            rounded,
            stage=2,
            active=last,
            issue_order=307 if pipeline else 7,
            logical_phase=3,
        )
        vector.task(
            "store",
            "copy",
            rounded,
            ogm,
            stage=2,
            active=last,
            issue_order=308 if pipeline else 8,
            logical_phase=3,
        )
        vectors.append(vector.finish())
    result = compose("flash_attention", cube.finish(), *vectors, disjoint_external=True)
    op = result.body.block.first_op
    op.attributes.update(
        attention_shape=ArrayAttr(IntAttr(n) for n in shape),
        attention_tile=ArrayAttr(IntAttr(n) for n in (block_q, block_kv, gm_slots, local_slots)),
        attention_scale=FloatAttr(scale, f32),
        attention_causal=IntAttr(int(causal)),
        attention_start=IntAttr(query_start),
        attention_packed_decode=IntAttr(int(_decode_prefix is not None)),
        attention_policies=ArrayAttr(
            IntAttr(int(n)) for n in (q_cache == "l1", reuse_ub, pipeline, packed_reductions)
        ),
        attention_domain=ArrayAttr(IntAttr(n) for n in (_query_first, qt, epochs, _group_offset)),
    )
    return result


def flash_attention(
    batch,
    query_heads,
    kv_heads,
    query_length,
    kv_length,
    head_dim,
    *,
    dtype="f16",
    block_q=None,
    block_kv=128,
    gm_slots=2,
    local_slots=2,
    softmax_scale=None,
    q_cache="l0",
    reuse_ub=False,
    pipeline=False,
    packed_reductions=False,
    causal=False,
    query_start=0,
):
    """Forward dense/causal MHA, GQA or MQA in BNSD, with explicit P RN.

    Causal visibility is key <= query_start + query_row. Statically partition
    query tiles by their *actual* KV prefix extent. Each selected group runs
    one complete mixed protocol; fully future KV tiles are never issued.
    The same physical IR, synchronization and verifier serve every variant.
    Sq1 packs adjacent GQA heads as Cube rows when Hq/Hkv is a multiple of BQ32/64.
    All packed heads retain the same absolute query position for masking.
    """
    args = (batch, query_heads, kv_heads, query_length, kv_length, head_dim)
    require(
        all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in args),
        "attention dimensions must be positive static integers",
    )
    require(query_heads % kv_heads == 0, "attention requires an integer Hq/Hkv ratio")
    if block_q is None:
        block_q = 32 if query_length == 1 and query_heads // kv_heads == 32 else 64
    options = {
        "dtype": dtype,
        "block_q": block_q,
        "block_kv": block_kv,
        "gm_slots": gm_slots,
        "local_slots": local_slots,
        "softmax_scale": softmax_scale,
        "q_cache": q_cache,
        "reuse_ub": reuse_ub,
        "pipeline": pipeline,
        "packed_reductions": packed_reductions,
    }
    require(isinstance(causal, bool), "attention causal must be boolean")
    require(
        isinstance(query_start, int) and not isinstance(query_start, bool) and query_start >= 0,
        "attention query_start must be a nonnegative static integer",
    )
    if query_length == 1:
        ratio = query_heads // kv_heads
        require(
            block_q in {32, 64} and ratio % block_q == 0,
            "packed decode requires complete BQ32/64 groups of adjacent GQA heads; padded heads unsupported",
        )
        require(not causal or query_start < kv_length, "decode query position must lie inside KV")
        result = _flash_attention_group(
            batch,
            kv_heads,
            kv_heads,
            ratio,
            kv_length,
            head_dim,
            **options,
            causal=causal,
            query_start=query_start,
            _kv_tiles=(query_start + block_kv) // block_kv if causal else None,
            _decode_prefix=query_start + 1 if causal else None,
        )
        result.body.block.first_op.attributes["attention_shape"] = ArrayAttr(
            IntAttr(n) for n in args
        )
        result.body.block.first_op.attributes["attention_packed_decode"] = IntAttr(1)
        return result
    require(
        block_q in {32, 64, 128} and block_kv in {64, 128},
        "attention supports BQ32/64/128, BK64/128",
    )
    require(
        query_length % block_q == 0,
        "attention currently requires complete Q/KV tiles; no masked or padded fallback",
    )
    require(
        not causal or query_start + query_length <= kv_length,
        "causal query rows must have positions inside KV",
    )
    if not causal and batch * query_heads * (query_length // block_q) <= 1024:
        return _flash_attention_group(*args, **options, query_start=query_start)
    from .dispatch import static_dispatch

    tiles = query_length // block_q
    total_groups = batch * query_heads * tiles
    max_tiles = 1024 // (batch * query_heads)
    require(max_tiles > 0, "static attention requires at most 1024 batch/head owners per case")
    cases, first, cursor = [], 0, 0
    while first < tiles:
        extent = (
            (query_start + (first + 1) * block_q + block_kv - 1) // block_kv
            if causal
            else kv_length // block_kv
        )
        stop = first + 1
        while (
            stop < tiles
            and stop - first < max_tiles
            and (
                not causal
                or (query_start + (stop + 1) * block_q + block_kv - 1) // block_kv == extent
            )
        ):
            stop += 1
        cases.append(
            _flash_attention_group(
                *args,
                **options,
                causal=causal,
                query_start=query_start,
                _query_first=first * block_q,
                _query_tiles=stop - first,
                _kv_tiles=extent,
                _group_offset=cursor,
                _total_groups=total_groups,
            )
        )
        cursor += batch * query_heads * (stop - first)
        first = stop
    return static_dispatch("flash_attention", cases)

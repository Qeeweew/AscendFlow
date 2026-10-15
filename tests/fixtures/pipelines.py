"""Small real recipes, plus storage-lifetime counterexamples used by regressions."""

from xdsl_npu._physical_builder import Kernel


def scalar_pipeline(count=3, blocks=1):
    """A real Scalar UB read; the scalar operand is captured before Vector issue."""
    k = Kernel("scalar_pipeline", blocks=blocks, disjoint_external=True)
    x = k.buffer("x", "GM", count * blocks * 32, external=True, initialized=True)
    y = k.buffer("y", "GM", count * blocks * 32, external=True)
    a = k.buffer("a", "UB", 64)
    b = k.buffer("b", "UB", 64)
    k.loop(count)
    xr = k.view(x, 32, stride=32, core_stride=count * 32)
    yr = k.view(y, 32, stride=32, core_stride=count * 32)
    ar, br = k.view(a, 32, stride=32, slots=2), k.view(b, 32, stride=32, slots=2)
    k.copy("load", xr, ar)
    word = k.view(ar, 4)
    value = k.scalar_load("factor", word, stage=1)
    k.scale_value("multiply", ar, br, value, stage=1)
    k.copy("store", br, yr, stage=2)
    return k.finish()


def vector_pipeline(count=6, tile=128, slots=2, *, stages=(0, 1, 2), late_reader=False, tail=0):
    nbytes = tile * 4
    total_bytes = (count - 1) * nbytes + tail * 4 if count and tail else count * nbytes
    k = Kernel("vector_pipeline", disjoint_external=True)
    x = k.buffer("x", "GM", total_bytes, external=True, initialized=True)
    y = k.buffer("y", "GM", total_bytes, external=True)
    inp = k.buffer("input_slots", "UB", slots * nbytes)
    out = k.buffer("output_slots", "UB", slots * nbytes)
    k.loop(count)
    xr = k.view(x, nbytes, stride=nbytes, tail_nbytes=tail * 4)
    yr = k.view(y, nbytes, stride=nbytes, tail_nbytes=tail * 4)
    a, b = (
        k.view(inp, nbytes, stride=nbytes, slots=slots, tail_nbytes=tail * 4),
        k.view(out, nbytes, stride=nbytes, slots=slots, tail_nbytes=tail * 4),
    )
    k.copy("load", xr, a, stage=stages[0])
    k.scale("scale", a, b, 2.0, stage=stages[1])
    if late_reader:
        k.add("last_read", a, b, b, stage=stages[1])
    k.add_scalar("postprocess", b, b, 1.0, stage=stages[1])
    k.copy("store", b, yr, stage=stages[2])
    return k.finish()


def cube_pipeline(count=5, slots=2, *, stages=(0, 1, 2, 3)):
    k = Kernel("cube_pipeline", core="aic", disjoint_external=True)
    a = k.buffer("a", "GM", count * 512, "f16", layout="NZ", external=True, initialized=True)
    b = k.buffer("b", "GM", count * 512, "f16", layout="NZ", external=True, initialized=True)
    c = k.buffer("c", "GM", count * 1024, external=True)
    l1a = k.buffer("l1a", "L1", slots * 512, "f16", layout="NZ")
    l1b = k.buffer("l1b", "L1", slots * 512, "f16", layout="NZ")
    l0a = k.buffer("l0a", "L0A", slots * 512, "f16", layout="ZZ")
    l0b = k.buffer("l0b", "L0B", slots * 512, "f16", layout="ZN")
    l0c = k.buffer("l0c", "L0C", slots * 1024, layout="NZ")
    k.init_fixpipe()
    k.loop(count)
    ar, br = k.view(a, 512, stride=512), k.view(b, 512, stride=512)
    cr = k.view(c, 1024, stride=1024)
    la = k.view(l1a, 512, stride=512, slots=slots)
    lb = k.view(l1b, 512, stride=512, slots=slots)
    aa = k.view(l0a, 512, stride=512, slots=slots)
    bb = k.view(l0b, 512, stride=512, slots=slots)
    cc = k.view(l0c, 1024, stride=1024, slots=slots)
    k.copy("load_a", ar, la, stage=stages[0])
    k.copy("load_b", br, lb, stage=stages[0])
    k.copy("operand_a", la, aa, stage=stages[1])
    k.copy("operand_b", lb, bb, stage=stages[1])
    k.mad("mad", aa, bb, cc, stage=stages[2])
    k.fixpipe("export", cc, cr, stage=stages[3])
    return k.finish()


def phased_pipeline(phases=9, phase_length=1, tile=16):
    """Static phases share buffers; logical channel count is not flag usage.

    Input has one slot; output has two. Each phase deliberately has distinct
    task identities while preserving the same native pipeline and aliases.
    There is no phase barrier or buffer expansion at the boundaries.
    """
    from xdsl_npu.model import require

    require(isinstance(phases, int) and 1 <= phases <= 64, "phases must be in [1,64]")
    require(
        isinstance(phase_length, int) and 1 <= phase_length <= 16, "phase length must be in [1,16]"
    )
    require(
        isinstance(tile, int) and 8 <= tile <= 4096 and tile % 8 == 0,
        "phase tile must be a multiple of eight in [8,4096]",
    )
    count, nbytes = phases * phase_length, tile * 4
    k = Kernel("phased_pipeline", disjoint_external=True)
    x = k.buffer("x", "GM", count * nbytes, external=True, initialized=True)
    y = k.buffer("y", "GM", count * nbytes, external=True)
    a = k.buffer("input_slots", "UB", nbytes)
    b = k.buffer("output_slots", "UB", 2 * nbytes)
    k.loop(count)
    src = k.view(x, nbytes, stride=nbytes)
    dst = k.view(y, nbytes, stride=nbytes)
    inp = k.view(a, nbytes)
    out = k.view(b, nbytes, stride=nbytes, slots=2)
    for phase in range(phases):
        active = (phase * phase_length, (phase + 1) * phase_length)
        k.task(f"load_{phase}", "copy", src, inp, active=active)
        k.task(f"scale_{phase}", "muls", inp, out, scalar=2, stage=1, active=active)
        k.task(f"store_{phase}", "copy", out, dst, stage=2, issue_order=0, active=active)
    return k.finish()


def mixed_pipeline(count=3, slots=2, *, aiv1_delay=0, segments=1, local_slots=None):
    """Minimal real FIX -> GM ring -> two AIVs, including zero/reuse boundaries."""
    from xdsl.dialects.builtin import IntAttr, StringAttr

    from xdsl_npu.ir import BufferOp, ViewOp
    from xdsl_npu.mixed import compose
    from xdsl_npu.model import require
    from xdsl_npu.sync import parts

    require(isinstance(aiv1_delay, int) and 0 <= aiv1_delay <= 64, "AIV delay must be in [0,64]")
    require(type(segments) is int and 1 <= segments <= 16, "static segments must be in [1,16]")
    local_slots = slots if local_slots is None else local_slots
    native_count, gm_slots = count * segments, slots * segments

    cube = cube_pipeline(native_count, local_slots)
    kernel, stream = parts(cube)
    root = next(
        o for o in kernel.body.block.ops if isinstance(o, BufferOp) and o.sym_name.data == "c"
    )
    root.attributes.update({"sym_name": StringAttr("scratch"), "nbytes": IntAttr(gm_slots * 1024)})
    view = next(o for o in stream.body.block.ops if isinstance(o, ViewOp) and o.source is root.ref)
    view.attributes["modulus"] = IntAttr(gm_slots)
    vectors = []
    for subblock in range(2):
        v = Kernel("consumer", disjoint_external=True)
        gm = v.buffer("scratch", "GM", gm_slots * 1024, external=True)
        y = v.buffer("y", "GM", native_count * 1024, external=True)
        ub = v.buffer("tile", "UB", local_slots * 512)
        delay = v.buffer("delay_word", "UB", 32) if subblock == 1 and aiv1_delay else None
        v.loop(native_count)
        src = v.view(gm, 512, offset=subblock * 512, stride=1024, slots=gm_slots)
        local = v.view(ub, 512, stride=512, slots=local_slots)
        dst = v.view(y, 512, offset=subblock * 512, stride=1024)
        if delay is not None:
            word = v.view(delay, 4)
            v.task("delay_init", "stat_fill", word, params=(1,), issue_order=30)
            for i in range(aiv1_delay):
                v.task(
                    f"delay_{i}", "stat_adds", word, word, params=(1,), scalar=1, issue_order=31 + i
                )
            # A real synchronous, volatile UB load gates subsequent dispatch.
            # Its result is intentionally unused; the extra private work only
            # stresses uneven collective arrival, without altering Y.
            v.scalar_load("capture_delay", word, issue_order=31 + aiv1_delay)
        v.task("import_tile", "copy", src, local, issue_order=200)
        v.task("scale", "muls", local, local, scalar=2, stage=1, issue_order=10)
        v.task("store", "copy", local, dst, stage=2, issue_order=0)
        vectors.append(v.finish())
    return compose(
        "mixed_pipeline",
        cube,
        *vectors,
        cube_epoch=segments,
        vector_epoch=segments,
        disjoint_external=True,
    )

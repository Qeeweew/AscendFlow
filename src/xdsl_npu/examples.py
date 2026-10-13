"""Small real recipes, plus storage-lifetime counterexamples used by regressions."""

from .builder import Kernel


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


def mixed_pipeline(count=3, slots=2, *, aiv1_delay=0):
    """Minimal real FIX -> GM ring -> two AIVs, including zero/reuse boundaries."""
    from xdsl.dialects.builtin import IntAttr, StringAttr

    from .ir import BufferOp, ViewOp
    from .mixed import compose
    from .model import require
    from .sync import parts

    require(isinstance(aiv1_delay, int) and 0 <= aiv1_delay <= 64, "AIV delay must be in [0,64]")

    cube = cube_pipeline(count, slots)
    kernel, stream = parts(cube)
    root = next(
        o for o in kernel.body.block.ops if isinstance(o, BufferOp) and o.sym_name.data == "c"
    )
    root.attributes.update({"sym_name": StringAttr("scratch"), "nbytes": IntAttr(slots * 1024)})
    view = next(o for o in stream.body.block.ops if isinstance(o, ViewOp) and o.source is root.ref)
    view.attributes["modulus"] = IntAttr(slots)
    vectors = []
    for subblock in range(2):
        v = Kernel("consumer", disjoint_external=True)
        gm = v.buffer("scratch", "GM", slots * 1024, external=True)
        y = v.buffer("y", "GM", count * 1024, external=True)
        ub = v.buffer("tile", "UB", slots * 512)
        delay = v.buffer("delay_word", "UB", 32) if subblock == 1 and aiv1_delay else None
        v.loop(count)
        src = v.view(gm, 512, offset=subblock * 512, stride=1024, slots=slots)
        local = v.view(ub, 512, stride=512, slots=slots)
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
    return compose("mixed_pipeline", cube, *vectors, disjoint_external=True)

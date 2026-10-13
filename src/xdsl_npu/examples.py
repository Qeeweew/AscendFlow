"""Small real recipes, plus storage-lifetime counterexamples used by regressions."""

from .builder import Kernel


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

"""Operator recipes expose all native work before the shared AutoSync pass.

These factories use static shapes and native arithmetic, not arbitrary Python tracing. No user
events or hidden scratch/communication are introduced after synchronization.
"""

from math import isfinite

from .builder import Kernel
from .model import CAPACITY, CompileError, require, round_f32


def align32(n):
    return (n + 31) // 32 * 32


def row_policy(rows, columns, blocks, row_block, size):
    require(
        isinstance(rows, int) and rows >= 0 and isinstance(columns, int) and columns > 0,
        "rows must be a nonnegative static integer; columns must be positive",
    )
    require(
        isinstance(blocks, int) and 1 <= blocks <= 1024 and rows % blocks == 0,
        "static rows must divide the independent core grid",
    )
    if row_block is not None:
        require(
            isinstance(row_block, int)
            and 1 <= row_block <= 8
            and (rows // blocks) % row_block == 0,
            "fixed row_block must divide each core's rows",
        )
        require(size(row_block) <= CAPACITY["UB"], "fixed row_block exceeds the 192 KiB private UB")
        return row_block
    for candidate in (8, 4, 2, 1):
        if (rows // blocks) % candidate == 0 and size(candidate) <= CAPACITY["UB"]:
            return candidate
    raise CompileError("no supported row grain fits private UB with the requested slot count")


def rms_norm(rows, columns, *, epsilon=1e-6, blocks=1, row_block=None, slots=2):
    """Y = round_f32(round_f32(X * (1/sqrt(mean(X^2)+eps))) * Gamma).

    Sum uses native contiguous 64-element partials then native per-row merging.
    Only finite inputs, f32 and 64-aligned columns up to 4096 are supported.
    Gamma stays resident; partials and statistics have lifetimes separate from
    the input/output rings. Scalar capture releases statistics before Muls ends.
    """
    require(
        isinstance(columns, int) and 64 <= columns <= 4096 and columns % 64 == 0,
        "RMSNorm supports f32 columns divisible by 64, up to 4096",
    )
    require(isfinite(epsilon) and epsilon > 0, "RMSNorm epsilon must be finite and positive")
    epsilon = round_f32(epsilon)
    require(epsilon > 0, "RMSNorm epsilon must remain positive after f32 rounding")
    require(isinstance(slots, int) and 1 <= slots <= 8, "RMSNorm slots must be in [1,8]")
    partial_pitch = align32(columns // 64 * 4)
    grain = row_policy(
        rows,
        columns,
        blocks,
        row_block,
        lambda r: slots * 2 * r * columns * 4 + columns * 4 + r * partial_pitch + r * 64,
    )
    count = rows // blocks // grain
    tile_bytes = grain * columns * 4
    k = Kernel("rms_norm", blocks=blocks, disjoint_external=True)
    x = k.buffer("x", "GM", rows * columns * 4, external=True, initialized=True)
    gamma = k.buffer("gamma", "GM", columns * 4, external=True, initialized=True)
    y = k.buffer("y", "GM", rows * columns * 4, external=True)
    inp = k.buffer("input_slots", "UB", slots * tile_bytes)
    work = k.buffer("work_slots", "UB", slots * tile_bytes)
    weights = k.buffer("gamma_cache", "UB", columns * 4)
    partials = k.buffer("partials", "UB", grain * partial_pitch)
    stats = k.buffer("stats", "UB", grain * 32)
    ones = k.buffer("ones", "UB", grain * 32)
    k.loop(count)
    xr = k.view(x, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes)
    yr = k.view(y, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes)
    a = k.view(inp, tile_bytes, stride=tile_bytes, slots=slots)
    b = k.view(work, tile_bytes, stride=tile_bytes, slots=slots)
    gr = k.view(weights, columns * 4)
    pr = k.view(partials, columns // 64 * 4, rows=grain, row_stride=partial_pitch)
    sr = k.view(stats, 4, rows=grain, row_stride=32)
    one = k.view(ones, 4, rows=grain, row_stride=32)
    k.task("load_gamma", "copy", k.view(gamma, columns * 4), gr, active=(0, 1), issue_order=1)
    k.task("load", "copy", xr, a, issue_order=2)
    k.task("init_one", "stat_fill", one, stage=1, params=(grain,), scalar=1.0, active=(0, 1))
    k.task("square", "mul", a, a, b, stage=1, issue_order=3)
    k.task("partial_sum", "reduce_sum64", b, pr, stage=1, params=(grain, columns))
    k.task("row_sum", "merge_sum", pr, sr, stage=1, params=(grain, columns // 64))
    k.task("mean", "stat_muls", sr, sr, stage=1, params=(grain,), scalar=1 / columns)
    k.task("epsilon", "stat_adds", sr, sr, stage=1, params=(grain,), scalar=epsilon)
    k.task("sqrt", "stat_sqrt", sr, sr, stage=1, params=(grain,))
    k.task("inverse", "stat_div", one, sr, sr, stage=1, params=(grain,))
    for r in range(grain):
        value = k.scalar_load(f"capture_r{r}", k.view(sr, 4, offset=r * 32), stage=1)
        k.scale_value(
            f"normalize_r{r}",
            k.view(a, columns * 4, offset=r * columns * 4),
            k.view(b, columns * 4, offset=r * columns * 4),
            value,
            stage=1,
        )
    k.task("gamma", "mul_gamma", b, gr, b, stage=1, params=(grain, columns))
    # Previous tile Store must be dispatched before a current Scalar wait.
    k.task("store", "copy", b, yr, stage=2, issue_order=0)
    return k.finish()


def rope(rows, columns, *, blocks=1, row_block=None, slots=2):
    """Rotate-half RoPE with f32 X/Y[rows,D], Cos/Sin[rows,D/2].

    Keeps each multiplication and add/subtract rounding boundary. Cos and Sin
    share a real UB allocation; every Cos reader retires before the Sin DMA.
    """
    require(
        isinstance(columns, int) and 16 <= columns <= 4096 and columns % 16 == 0,
        "RoPE supports f32 columns divisible by 16, up to 4096",
    )
    require(isinstance(slots, int) and 1 <= slots <= 8, "RoPE slots must be in [1,8]")
    grain = row_policy(rows, columns, blocks, row_block, lambda r: slots * 3 * r * columns * 4)
    count = rows // blocks // grain
    half = columns // 2
    tile_bytes, trig_bytes = grain * columns * 4, grain * half * 4
    k = Kernel("rope", blocks=blocks, disjoint_external=True)
    x = k.buffer("x", "GM", rows * columns * 4, external=True, initialized=True)
    cos = k.buffer("cos", "GM", rows * half * 4, external=True, initialized=True)
    sin = k.buffer("sin", "GM", rows * half * 4, external=True, initialized=True)
    y = k.buffer("y", "GM", rows * columns * 4, external=True)
    inp = k.buffer("input_slots", "UB", slots * tile_bytes)
    out = k.buffer("output_slots", "UB", slots * tile_bytes)
    trig = k.buffer("trig_slots", "UB", slots * trig_bytes)
    temp = k.buffer("temp_slots", "UB", slots * trig_bytes)
    k.loop(count)
    ar = k.view(inp, tile_bytes, stride=tile_bytes, slots=slots)
    br = k.view(out, tile_bytes, stride=tile_bytes, slots=slots)
    tr = k.view(trig, trig_bytes, stride=trig_bytes, slots=slots)
    tm = k.view(temp, trig_bytes, stride=trig_bytes, slots=slots)
    xl = k.view(ar, half * 4, rows=grain, row_stride=columns * 4)
    xh = k.view(ar, half * 4, offset=half * 4, rows=grain, row_stride=columns * 4)
    yl = k.view(br, half * 4, rows=grain, row_stride=columns * 4)
    yh = k.view(br, half * 4, offset=half * 4, rows=grain, row_stride=columns * 4)
    tv = k.view(tr, half * 4, rows=grain, row_stride=half * 4)
    tmp = k.view(tm, half * 4, rows=grain, row_stride=half * 4)
    k.copy("load", k.view(x, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes), ar)
    k.copy(
        "load_cos", k.view(cos, trig_bytes, stride=trig_bytes, core_stride=count * trig_bytes), tr
    )
    k.task("cos_low", "mul_rows", xl, tv, yl, stage=1, params=(grain, half))
    k.task("cos_high", "mul_rows", xh, tv, yh, stage=1, params=(grain, half))
    k.task(
        "load_sin",
        "copy",
        k.view(sin, trig_bytes, stride=trig_bytes, core_stride=count * trig_bytes),
        tr,
        stage=1,
    )
    k.task("sin_high", "mul_rows", xh, tv, tmp, stage=1, params=(grain, half))
    k.task("rotate_low", "sub_rows", yl, tmp, yl, stage=1, params=(grain, half))
    k.task("sin_low", "mul_rows", xl, tv, tmp, stage=1, params=(grain, half))
    k.task("rotate_high", "add_rows", yh, tmp, yh, stage=1, params=(grain, half))
    k.task(
        "store",
        "copy",
        br,
        k.view(y, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes),
        stage=2,
    )
    return k.finish()


def softmax(rows, columns, *, blocks=1, row_block=None, slots=2):
    """Stable row Softmax with explicit f32 max, exp, sum, reciprocal and Mul.

    Finite inputs only. The mathematical reciprocal is a separate f32 node;
    normalization is multiplication by that value, with no reassociation.
    Max and sum share the same physical partial/statistic buffers.
    """
    require(
        isinstance(columns, int) and 64 <= columns <= 4096 and columns % 64 == 0,
        "Softmax supports f32 columns divisible by 64, up to 4096",
    )
    require(isinstance(slots, int) and 1 <= slots <= 8, "Softmax slots must be in [1,8]")
    partial_pitch = align32(columns // 64 * 4)
    grain = row_policy(
        rows,
        columns,
        blocks,
        row_block,
        lambda r: slots * 2 * r * columns * 4 + r * partial_pitch + r * 64,
    )
    count, tile_bytes = rows // blocks // grain, grain * columns * 4
    k = Kernel("softmax", blocks=blocks, disjoint_external=True)
    x = k.buffer("x", "GM", rows * columns * 4, external=True, initialized=True)
    y = k.buffer("y", "GM", rows * columns * 4, external=True)
    inp = k.buffer("input_slots", "UB", slots * tile_bytes)
    out = k.buffer("output_slots", "UB", slots * tile_bytes)
    partials = k.buffer("partials", "UB", grain * partial_pitch)
    stats = k.buffer("stats", "UB", grain * 32)
    ones = k.buffer("ones", "UB", grain * 32)
    k.loop(count)
    a = k.view(inp, tile_bytes, stride=tile_bytes, slots=slots)
    b = k.view(out, tile_bytes, stride=tile_bytes, slots=slots)
    pr = k.view(partials, columns // 64 * 4, rows=grain, row_stride=partial_pitch)
    sr = k.view(stats, 4, rows=grain, row_stride=32)
    one = k.view(ones, 4, rows=grain, row_stride=32)
    k.copy("load", k.view(x, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes), a)
    k.task("init_one", "stat_fill", one, stage=1, params=(grain,), scalar=1.0, active=(0, 1))
    k.task("partial_max", "reduce_max64", a, pr, stage=1, params=(grain, columns))
    k.task("row_max", "merge_max", pr, sr, stage=1, params=(grain, columns // 64))
    for r in range(grain):
        value = k.scalar_load(f"capture_max_r{r}", k.view(sr, 4, offset=r * 32), stage=1)
        k.subtract_value(
            f"shift_r{r}",
            k.view(a, columns * 4, offset=r * columns * 4),
            k.view(b, columns * 4, offset=r * columns * 4),
            value,
            stage=1,
        )
    k.task("exponential", "exp", b, b, stage=1)
    k.task("partial_sum", "reduce_sum64", b, pr, stage=1, params=(grain, columns))
    k.task("row_sum", "merge_sum", pr, sr, stage=1, params=(grain, columns // 64))
    k.task("reciprocal", "stat_div", one, sr, sr, stage=1, params=(grain,))
    for r in range(grain):
        value = k.scalar_load(f"capture_inv_r{r}", k.view(sr, 4, offset=r * 32), stage=1)
        br = k.view(b, columns * 4, offset=r * columns * 4)
        k.scale_value(f"normalize_r{r}", br, br, value, stage=1)
    k.task(
        "store",
        "copy",
        b,
        k.view(y, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes),
        stage=2,
        issue_order=0,
    )
    return k.finish()


def swiglu(rows, columns, *, blocks=1, row_block=None, slots=2):
    """Separate gate/up tensors: Y = round(round(G / (1 + exp(-G))) * U).

    Gate, Up and Y are f32 [rows,columns] with disjoint GM allocations. Gate
    input retires at Div; Up retires at Mul. The denominator allocation becomes
    the output in place, so no additional tensor transfer or buffer is needed.
    """
    require(
        isinstance(columns, int) and 8 <= columns <= 4096 and columns % 8 == 0,
        "SwiGLU supports f32 columns divisible by 8, up to 4096",
    )
    require(isinstance(slots, int) and 1 <= slots <= 8, "SwiGLU slots must be in [1,8]")
    grain = row_policy(rows, columns, blocks, row_block, lambda r: slots * 3 * r * columns * 4)
    count, tile_bytes = rows // blocks // grain, grain * columns * 4
    k = Kernel("swiglu", blocks=blocks, disjoint_external=True)
    gate = k.buffer("gate", "GM", rows * columns * 4, external=True, initialized=True)
    up = k.buffer("up", "GM", rows * columns * 4, external=True, initialized=True)
    y = k.buffer("y", "GM", rows * columns * 4, external=True)
    ga = k.buffer("gate_slots", "UB", slots * tile_bytes)
    ua = k.buffer("up_slots", "UB", slots * tile_bytes)
    work = k.buffer("work_slots", "UB", slots * tile_bytes)
    k.loop(count)
    g = k.view(ga, tile_bytes, stride=tile_bytes, slots=slots)
    u = k.view(ua, tile_bytes, stride=tile_bytes, slots=slots)
    b = k.view(work, tile_bytes, stride=tile_bytes, slots=slots)
    k.copy(
        "load_gate", k.view(gate, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes), g
    )
    k.copy("load_up", k.view(up, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes), u)
    k.scale("negative_gate", g, b, -1.0, stage=1)
    k.task("exponential", "exp", b, b, stage=1)
    k.add_scalar("denominator", b, b, 1.0, stage=1)
    k.task("silu", "div", g, b, b, stage=1)
    br = k.view(b, columns * 4, rows=grain, row_stride=columns * 4)
    ur = k.view(u, columns * 4, rows=grain, row_stride=columns * 4)
    k.task("multiply_up", "mul_rows", br, ur, br, stage=1, params=(grain, columns))
    k.copy(
        "store",
        b,
        k.view(y, tile_bytes, stride=tile_bytes, core_stride=count * tile_bytes),
        stage=2,
    )
    return k.finish()


def batched_matmul_nt(batch, m, n, reduction, *, k_block=64, slots=2):
    """C[B,M,N] = A[B,M,K] @ Bt[B,N,K].T, with f16 inputs/f32 C.

    One independent AIC per matrix, complete 16-aligned rectangular tiles.
    GM is normal row-major ND; native ND2NZ packs each K slice directly in L1.
    K slices are the stream generations. A single f32 L0C carry is initialized
    by the first MAD, updated by every later MAD and exported exactly once.
    """
    require(isinstance(batch, int) and 1 <= batch <= 1024, "batch must be static in [1,1024]")
    require(
        all(isinstance(d, int) and 16 <= d <= 256 and d % 16 == 0 for d in (m, n, k_block)),
        "matmul M,N,K-block must be complete 16-aligned tiles in [16,256]",
    )
    require(
        isinstance(reduction, int) and k_block <= reduction <= 65520 and reduction % k_block == 0,
        "matmul K must be a positive multiple of K-block and fit the ND2NZ leading dimension",
    )
    require(isinstance(slots, int) and 1 <= slots <= 8, "matmul slots must be in [1,8]")
    count = reduction // k_block
    ab, bb, cb = m * k_block * 2, n * k_block * 2, m * n * 4
    k = Kernel("batched_matmul_nt", core="aic", blocks=batch, disjoint_external=True)
    a = k.buffer("a", "GM", batch * m * reduction * 2, "f16", external=True, initialized=True)
    bt = k.buffer("bt", "GM", batch * n * reduction * 2, "f16", external=True, initialized=True)
    c = k.buffer("c", "GM", batch * cb, external=True)
    l1a = k.buffer("l1a", "L1", slots * ab, "f16", layout="NZ")
    l1b = k.buffer("l1b", "L1", slots * bb, "f16", layout="NZ")
    l0a = k.buffer("l0a", "L0A", slots * ab, "f16", layout="ZZ")
    l0b = k.buffer("l0b", "L0B", slots * bb, "f16", layout="ZN")
    l0c = k.buffer("accumulator", "L0C", cb, layout="NZ")
    k.init_fixpipe()
    k.loop(count)
    ar = k.view(
        a,
        k_block * 2,
        rows=m,
        row_stride=reduction * 2,
        stride=k_block * 2,
        core_stride=m * reduction * 2,
    )
    br = k.view(
        bt,
        k_block * 2,
        rows=n,
        row_stride=reduction * 2,
        stride=k_block * 2,
        core_stride=n * reduction * 2,
    )
    cr = k.view(c, n * 4, rows=m, row_stride=n * 4, core_stride=cb)
    la = k.view(l1a, ab, stride=ab, slots=slots)
    lb = k.view(l1b, bb, stride=bb, slots=slots)
    aa = k.view(l0a, ab, stride=ab, slots=slots)
    ba = k.view(l0b, bb, stride=bb, slots=slots)
    cc = k.view(l0c, cb)
    k.task("load_a", "nd2nz", ar, la, params=(m, k_block))
    k.task("load_bt", "nd2nz", br, lb, params=(n, k_block))
    k.task("operand_a", "load2d_a", la, aa, stage=1, params=(m, k_block))
    k.task("operand_b", "load2d_b_nt", lb, ba, stage=1, params=(n, k_block))
    k.task("mad_init", "mad", aa, ba, cc, stage=2, params=(m, n, k_block, 1), active=(0, 1))
    k.task("mad_acc", "mad", aa, ba, cc, stage=2, params=(m, n, k_block, 0), active=(1, count))
    k.task("export", "fix", cc, cr, stage=3, params=(m, n), active=(count - 1, count))
    return k.finish()

"""ASC emission consumes verified final IR; no instruction selection or hidden DMA."""

from .ir import TaskOp
from .model import ITEMSIZE, Ref, optional_int, storage_layout
from .sync import execution, parts, verify_plan

QUALIFIER = {
    "GM": "__gm__",
    "UB": "__ubuf__",
    "L1": "__cbuf__",
    "L0A": "__ca__",
    "L0B": "__cb__",
    "L0C": "__cc__",
}
CTYPE = {"f32": "float", "f16": "half", "bf16": "bfloat16_t"}


def pointer(ref: Ref, g="g", *, core="0", extra=""):
    terms = []
    if ref.count_offset:
        terms.append(f"n * {ref.count_offset}")
    if ref.count_core_stride:
        terms.append(f"n * ({core}) * {ref.count_core_stride}")
    for offset, stride, modulus, divisor in ref.terms:
        if offset:
            terms.append(str(offset))
        if stride:
            index = f"({g} / {divisor})" if divisor != 1 else g
            index = f"({index} % {modulus})" if modulus else index
            terms.append(f"{stride} * {index}")
    if core != "0":
        for stride, divisor, modulus in ref.core_terms:
            index = f"({core} / {divisor})" if divisor != 1 else core
            index = f"({index} % {modulus})" if modulus else index
            terms.append(f"{stride} * {index}")
    if extra:
        terms.append(extra)
    address = " + ".join(terms) or "0"
    q = QUALIFIER[ref.storage.bank]
    return f"({q} {CTYPE[ref.dtype]}*)({ref.storage.name} + ({address}))"


def native_line(task, count, core="0"):
    refs = task.refs
    p = [pointer(r, core=core) for r in refs]

    def length(ref, unit=1):
        base = ref.total_nbytes // unit
        if ref.tail_nbytes:
            return f"(g == {count - 1} ? {ref.tail_nbytes // unit} : {base})"
        return str(base)

    if task.native == "copy":
        route = tuple(r.storage.bank for r in refs)
        function = {
            ("GM", "UB"): "asc_copy_gm2ub",
            ("UB", "GM"): "asc_copy_ub2gm",
            ("GM", "L1"): "asc_copy_gm2l1",
        }.get(route)
        if function:
            return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0])}));"
        if route == ("L1", "L0A"):
            # CATLASS AtlasA2 zN->zZ uses classic Load2D with ifTranspose=false.
            return f"asc_copy_l12l0a({p[1]}, {p[0]}, 0, 1, 1, 0);"
        return f"asc_copy_l12l0b({p[1]}, {p[0]}, 0, 1, 1, 0);"
    if task.native == "copy_rows":
        function = "asc_copy_gm2ub" if refs[0].storage.bank == "GM" else "asc_copy_ub2gm"
        gaps = [(r.row_stride - r.nbytes) // 32 if r.rows > 1 else 0 for r in refs]
        return (
            f"{function}({p[1]}, {p[0]}, uint16_t({refs[0].rows}), "
            f"uint16_t({refs[0].nbytes // 32}), uint16_t({gaps[0]}), uint16_t({gaps[1]}));"
        )
    if task.native in {"convert_f32", "convert_rn"}:
        source, destination = refs[0].dtype, refs[1].dtype
        function = (
            ("asc_half2float" if source == "f16" else "asc_bfloat162float")
            if task.native == "convert_f32"
            else ("asc_float2half_rn" if destination == "f16" else "asc_float2bfloat16_rn")
        )
        return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0], ITEMSIZE[source])}));"
    if task.native in {"muls", "adds"}:
        function = "asc_mul_scalar" if task.native == "muls" else "asc_add_scalar"
        return (
            f"{function}({p[-1]}, {p[0]}, float({task.scalar!r}), "
            f"uint32_t({length(refs[0], ITEMSIZE[refs[0].dtype])}));"
        )
    if task.native in {"add", "mul", "sub", "div", "max"}:
        return f"asc_{task.native}({p[2]}, {p[0]}, {p[1]}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"exp", "rsqrt"}:
        return f"asc_{task.native}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.native == "scalar_load":
        source = pointer(refs[0], core=core).replace("__ubuf__ float*", "__ubuf__ volatile float*")
        return f"npu_value_{task.uid} = *({source});"
    if task.native in {"vector_copy", "brcb8", "broadcast_rows64"}:
        bits = [value.replace("float*", "uint32_t*") for value in p]
        if task.native == "vector_copy":
            return f"asc_copy({bits[1]}, {bits[0]}, uint32_t({length(refs[0], 4)}));"
        if task.native == "brcb8":
            return f"asc_brcb({bits[1]}, {bits[0]}, 1, 8, uint8_t({task.params[0] // 8}));"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"asc_copy({bits[1]}, {bits[0]}, uint8_t({task.params[0]}), 1, 0, 8, 1);"
        )
    if task.native in {"fill", "fill_neg_inf"}:
        value = "-__builtin_inff()" if task.native == "fill_neg_inf" else f"float({task.scalar!r})"
        return f"asc_duplicate({p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.native == "mask_causal":
        rows, cols, block_q, qt, first, owner_row, start = task.params
        dst = pointer(refs[0], core=core, extra=f"npu_row * {cols * 4} + npu_chunk * 256")
        return (
            "asc_set_mask_norm(); "
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"int64_t npu_visible = {start + first + owner_row + 1} + ({core} % {qt}) * {block_q} + npu_row - g * {cols} - npu_chunk * 64; "
            "uint64_t npu_mask = npu_visible >= 64 ? 0 : (npu_visible <= 0 ? UINT64_MAX : UINT64_MAX << npu_visible); "
            "if (npu_mask) { asc_set_vector_mask(0, npu_mask); "
            f"asc_duplicate({dst}, -__builtin_inff(), uint8_t(1), 1, 8); }} }} }}"
        )
    if task.native == "mask_prefix":
        rows, cols, prefix = task.params
        dst = pointer(refs[0], core=core, extra="npu_chunk * 256")
        return (
            "asc_set_mask_norm(); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"int64_t npu_visible = {prefix} - g * {cols} - npu_chunk * 64; "
            "uint64_t npu_mask = npu_visible >= 64 ? 0 : (npu_visible <= 0 ? UINT64_MAX : UINT64_MAX << npu_visible); "
            "if (npu_mask) { asc_set_vector_mask(0, npu_mask); "
            f"asc_duplicate({dst}, -__builtin_inff(), uint8_t({rows}), 1, {cols // 8}); }} }}"
        )
    if task.native in {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast"}:
        rows, cols = task.params
        function = task.native.removesuffix("_row_broadcast")
        if rows < cols // 64:
            # Repeat across the columns of one row. The eight identical
            # scalar words stay fixed (block/repeat strides 0/0); the full
            # matrix footprint and the exact in-place policy are unchanged.
            src = pointer(refs[0], core=core, extra=f"npu_row * {cols * 4}")
            scalar = pointer(refs[1], core=core, extra="npu_row * 32")
            dst = pointer(refs[2], core=core, extra=f"npu_row * {cols * 4}")
            return (
                "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
                f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
                f"asc_{function}({dst}, {src}, {scalar}, uint8_t({cols // 64}), "
                "1, 1, 0, 8, 8, 0); }"
            )
        src = pointer(refs[0], core=core, extra="npu_chunk * 256")
        dst = pointer(refs[2], core=core, extra="npu_chunk * 256")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"asc_{function}({dst}, {src}, {p[1]}, uint8_t({rows}), "
            f"1, 1, 0, {cols // 8}, {cols // 8}, 1); }}"
        )
    if task.native in {"merge_sum_packed", "merge_max_packed"}:
        rows, partials = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "merge_sum_packed"
            else "asc_repeat_reduce_max_only_value"
        )
        return (
            f"asc_set_mask_norm(); asc_set_vector_mask(0, uint64_t({(1 << partials) - 1})); "
            f"{function}({p[1]}, {p[0]}, uint8_t({rows}), 1, 1, {refs[0].row_stride // 32});"
        )
    if task.native in {"mul_columns64", "fma_columns64"}:
        function = "asc_fma" if task.native == "fma_columns64" else "asc_mul"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"{function}({p[2]}, {p[0]}, {p[1]}, uint8_t({task.params[0]}), 1, 1, 1, 8, 8, 0);"
        )
    if task.native == "reduce_rows64":
        return f"asc_repeat_reduce_sum({p[1]}, {p[0]}, uint32_t({task.params[0] * 64}));"
    if task.native in {"scale_value", "sub_value"}:
        function = "asc_mul_scalar" if task.native == "scale_value" else "asc_add_scalar"
        value = f"npu_value_{task.scalar_source}"
        if task.native == "sub_value":
            value = f"-{value}"
        return f"{function}({p[1]}, {p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"mul_rows", "sub_rows", "add_rows", "div_rows"}:
        rows, cols = task.params
        pp = [pointer(r, core=core, extra=f"npu_row * {r.row_stride}") for r in refs]
        function = task.native.removesuffix("_rows")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{function}({pp[2]}, {pp[0]}, {pp[1]}, uint32_t({cols})); }}"
        )
    if task.native in {"reduce_tile_sum64", "reduce_tile_max64"}:
        rows, cols = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "reduce_tile_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        src = pointer(refs[0], core=core, extra="npu_chunk * 256")
        dst = pointer(refs[1], core=core, extra=f"npu_chunk * {rows * 4}")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"{function}({dst}, {src}, uint8_t({rows}), 1, 1, {cols // 8}); }}"
        )
    if task.native in {"reduce_sum64", "reduce_max64"}:
        rows, cols = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "reduce_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        if refs[1].contiguous:
            return f"{function}({p[1]}, {p[0]}, uint32_t({rows * cols}));"
        src = pointer(refs[0], core=core, extra=f"npu_row * {cols * 4}")
        dst = pointer(refs[1], core=core, extra=f"npu_row * {refs[1].row_stride}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"{function}({dst}, {src}, uint32_t({cols})); }}"
        )
    if task.native in {
        "merge_sum",
        "merge_max",
        "stat_muls",
        "stat_adds",
        "stat_rsqrt",
        "stat_sqrt",
    }:
        rows = task.params[0]
        src = pointer(refs[0], core=core, extra=f"npu_row * {refs[0].row_stride}")
        dst = pointer(refs[1], core=core, extra=f"npu_row * {refs[1].row_stride}")
        if task.native.startswith("merge"):
            function = (
                "asc_repeat_reduce_sum"
                if task.native == "merge_sum"
                else "asc_repeat_reduce_max_only_value"
            )
            call = f"{function}({dst}, {src}, uint32_t({task.params[1]}));"
        elif task.native in {"stat_rsqrt", "stat_sqrt"}:
            call = f"asc_{task.native.removeprefix('stat_')}({dst}, {src}, uint32_t(1));"
        else:
            function = "asc_mul_scalar" if task.native == "stat_muls" else "asc_add_scalar"
            call = f"{function}({dst}, {src}, float({task.scalar!r}), uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.native in {"mul_gamma", "add_bias"}:
        rows, cols = task.params
        src = pointer(refs[0], core=core, extra=f"npu_row * {cols * 4}")
        dst = pointer(refs[2], core=core, extra=f"npu_row * {cols * 4}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{'mul' if task.native == 'mul_gamma' else 'add'}"
            f"({dst}, {src}, {p[1]}, uint32_t({cols})); }}"
        )
    if task.native in {"stat_fill", "stat_div"}:
        rows = task.params[0]
        pp = [pointer(r, core=core, extra=f"npu_row * {r.row_stride}") for r in refs]
        if task.native == "stat_fill":
            call = f"asc_duplicate({pp[0]}, float({task.scalar!r}), uint32_t(1));"
        else:
            call = f"asc_div({pp[2]}, {pp[0]}, {pp[1]}, uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.native == "nd2nz":
        rows, cols = task.params
        return (
            f"asc_copy_gm2l1_nd2nz({p[1]}, {p[0]}, 1, {rows}, {cols}, 0, "
            f"{refs[0].row_stride // 2}, {rows}, 1, 0);"
        )
    if task.native == "load2d_a":
        rows, cols = task.params
        src = pointer(refs[0], core=core, extra="npu_chunk * 512")
        dst = pointer(refs[1], core=core, extra=f"npu_chunk * {cols * 32}")
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"asc_copy_l12l0a({dst}, {src}, 0, {cols // 16}, {rows // 16}, 0); }}"
        )
    if task.native == "load2d_b_nt":
        rows, cols = task.params
        return f"asc_copy_l12l0b({p[1]}, {p[0]}, 0, {rows * cols // 256}, 1, 0);"
    if task.native == "load2d_b_nn":
        rows, cols = task.params  # B[K,N] compact zN -> L0B ZN
        src = pointer(refs[0], core=core, extra="npu_chunk * 512")
        dst = pointer(refs[1], core=core, extra=f"npu_chunk * {cols * 32}")
        # CANN 9.1/2201's public asc_copy_l12l0b has no classic transpose
        # parameter; the similarly named *_transpose is a 3510 API. Use
        # the exact audited dav_c220 LoadData2D intrinsic with transpose=true,
        # not the distinct LoadDataWithTranspose instruction/footprint.
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"load_cbuf_to_cb({dst}, {src}, 0, {cols // 16}, {rows // 16}, 0, 0, true, inc); }}"
        )
    if task.native in {"mad", "mad_cycle"}:
        m, n, k, clear = task.params
        flag = (
            f"(g % {clear} == 0)" if task.native == "mad_cycle" else ("true" if clear else "false")
        )
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, {m}, {k}, {n}, 0, false, false, {flag});"
    if task.native == "fix":
        m, n = task.params
        return (
            f"asc_copy_l0c2gm({p[1]}, {p[0]}, {n}, {m}, {refs[1].row_stride // 4}, "
            f"{m}, 0, 0, 0, false, true);"
        )
    if task.native == "mad16":
        clear = "true" if task.params[0] else "false"
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, 16, 16, 16, 0, false, false, {clear});"
    if task.native == "fix16":
        return f"asc_copy_l0c2gm({p[1]}, {p[0]}, 16, 16, 16, 16, 0, 0, 0, false, true);"
    raise AssertionError("unvalidated native recipe")


def _body_lines(p, ex, stream, *, cross=None, drains=(), core_expression=None):
    """Backend helper; callers must verify the entire enclosing program first."""
    tasks = {
        t.uid: t for _, t in sorted(enumerate(p.tasks), key=lambda x: (x[1].issue_order, x[0]))
    }
    # Read actual sync placements, then retain the compact physical step loop.
    before, after, pending, previous = {}, {}, [], None
    from .ir import CrossSyncOp, SyncOp

    for op in stream.body.block.ops:
        if isinstance(op, TaskOp):
            before[op.uid.data] = pending
            pending = []
            previous = op.uid.data
        elif isinstance(op, (SyncOp, CrossSyncOp)):
            if op.action.data == "set":
                after.setdefault(previous, []).append(op)
            else:
                pending.append(op)
    lines = [
        "  asc_init();",
        "  asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));",
    ]
    offsets, _ = storage_layout(p.storages)
    if p.blocks > 1:
        lines.append(f"  const int64_t core_id = {core_expression or 'asc_get_block_idx()'};")
    for s in p.storages:
        if s.external:
            continue
        offset = offsets[s.name]
        q = QUALIFIER[s.bank]
        lines.append(f"  {q} uint8_t* {s.name} = ({q} uint8_t*){offset};")
    if p.config:
        lines += [
            "  // Explicit invariant ND_PARA state readiness, including an empty stream.",
            "  asc_set_l0c_copy_params(1, 0, 0);",
            "  asc_sync_notify(PIPE_S, PIPE_FIX, 0);",
            "  asc_sync_wait(PIPE_S, PIPE_FIX, 0);",
        ]

    def sync_line(op):
        comment = ""
        if isinstance(op, CrossSyncOp):
            ch = cross[op.channel.data]
            pipe = current_task.pipe
            function = "asc_sync_block_arrive" if op.action.data == "set" else "asc_sync_block_wait"
            code = f"{function}(PIPE_{pipe}, {ch.flag_id});"
            comment = f" // {ch.name}: epoch g/{op.epoch_divisor.data}{op.epoch_offset.data:+d}"
        elif op.action.data == "barrier":
            code = f"asc_sync_pipe(PIPE_{op.pipe.data});"
        else:
            ch = ex.channels[op.channel.data]
            a, b = p.tasks[ch.source].pipe, p.tasks[ch.destination].pipe
            epoch = f"g - {op.distance.data} - {ch.lower}"
            if ch.stride != 1:
                epoch = f"({epoch}) / {ch.stride}"
            event_id = f"{ch.base} + (({epoch}) % {ch.width})"
            if ch.event_ids is not None:
                from .events import selection_expression

                event_id = selection_expression(ch.event_ids, epoch)
            function = "asc_sync_notify" if op.action.data == "set" else "asc_sync_wait"
            code = f"{function}(PIPE_{a}, PIPE_{b}, {event_id});"
            comment = f" // {ch.name}"
        lo, hi = op.lower.data, op.upper.data
        stride = optional_int(op.stride, 1)
        if (lo, hi) != (0, p.count) or stride != 1:
            # The surrounding task guard already establishes 0 <= g < count.
            clauses = ([f"g >= {lo}"] if lo else []) + ([f"g < {hi}"] if hi != p.count else [])
            if stride != 1:
                clauses.append(f"(g - {lo}) % {stride} == 0")
            code = f"if ({' && '.join(clauses)}) {{ {code} }}"
        return code + comment

    if p.count:
        end = p.count + max(t.stage for t in p.tasks)
        lines.append(f"  for (int64_t step = 0; step < {end}; ++step) {{")
        for task in p.tasks:
            if task.native == "scalar_load":
                lines.append(f"    float npu_value_{task.uid} = 0.0f;")
        for task in tasks.values():
            current_task = task
            guard = f"step >= {task.lower + task.stage} && step < {task.upper + task.stage}"
            if task.period != 1:
                position = f"(step - {task.stage}) % {task.period}"
                if task.phase_width == 1:
                    guard += f" && {position} == {task.phase}"
                elif task.phase_width != task.period:
                    guard += f" && {position} >= {task.phase} && {position} < {task.phase + task.phase_width}"
            lines += [
                f"    // {task.uid}: {task.pipe}, logical generation = step - {task.stage}",
                f"    if ({guard}) {{",
                f"      const int64_t g = step - {task.stage};",
            ]
            lines.extend("      " + sync_line(op) for op in before[task.uid])
            lines.append("      " + native_line(task, p.count, "core_id" if p.blocks > 1 else "0"))
            lines.extend("      " + sync_line(op) for op in after.get(task.uid, []))
            lines.append("    }")
        lines.append("  }")
    for drain in drains:
        ch = cross[drain.channel.data]
        lines += [
            "  // Retire outstanding GM credits, including the final partial ring.",
            f"  for (int64_t epoch = {drain.lower.data}; epoch < {drain.upper.data}; ++epoch) {{",
            f"    asc_sync_block_wait(PIPE_S, {ch.flag_id}); // {ch.name}",
            "  }",
        ]
    lines += [
        "  // Required kernel lifetime boundary; no PIPE_ALL inside the task stream.",
        "  asc_sync_pipe(PIPE_ALL);",
    ]
    return lines


def emit_asc(module, *, host_launcher=True):
    _, analysis = verify_plan(module, _details=True)
    return _emit_checked(module, analysis, host_launcher=host_launcher)


def _emit_checked(module, analysis, *, host_launcher=True):
    """Format the exact final extraction owned by this compilation invocation.

    External/mutable modules enter through emit_asc and are always checked.
    No verified flag, historical witness, or user analysis is accepted there.
    """
    from .ir import MixedOp, StaticDispatchOp
    from .loop_model import is_symbolic

    if is_symbolic(module):
        from .loop_codegen import emit_loop
        return emit_loop(analysis, host_launcher=host_launcher)

    if isinstance(module.body.block.first_op, StaticDispatchOp):
        return _emit_dispatch(analysis, host_launcher=host_launcher)

    if isinstance(module.body.block.first_op, MixedOp):
        return _emit_mixed(analysis, host_launcher=host_launcher)
    ex = analysis
    p = ex.program
    _, stream = parts(module)
    args = [s for s in p.storages if s.external]
    signature = ", ".join(f"__gm__ uint8_t* {s.name}" for s in args)
    kind = "__vector__" if p.core == "aiv" else "__cube__"
    lines = [
        "// xdsl-npu v0.1: dav-2201, single physical core, fixed issue plan",
        '#include "c_api/asc_simd.h"',
        "#include <cstdint>",
        "",
        f'extern "C" __global__ {kind} void {p.name}({signature}) {{',
        *_body_lines(p, ex, stream),
        "}",
    ]
    if host_launcher:
        sig = ", ".join(["void* stream"] + [f"void* arg_{s.name}" for s in args])
        call = ", ".join(f"(uint8_t*)arg_{s.name}" for s in args)
        lines += [
            "",
            f'extern "C" void launch_{p.name}({sig}) {{',
            f"  {p.name}<<<{p.blocks}, nullptr, stream>>>({call});",
            "}",
        ]
    return "\n".join(lines) + "\n"


def _emit_mixed(analysis, *, host_launcher):
    m, channels, _ = analysis
    signature = ", ".join(f"__gm__ uint8_t* {name}" for name in m.roots)
    name = m.op.sym_name.data
    lines = [
        "// xdsl-npu: verified dav-2201 1 AIC : 2 AIV GM ownership protocol",
        '#include "c_api/asc_simd.h"',
        "#include <cstdint>",
        "",
        f'extern "C" __global__ __mix__(1, 2) void {name}({signature}) {{',
    ]
    lines.extend(_mixed_body_lines(m, channels))
    lines.append("}")
    if host_launcher:
        sig = ", ".join(["void* stream"] + [f"void* arg_{root}" for root in m.roots])
        call = ", ".join(f"(uint8_t*)arg_{root}" for root in m.roots)
        lines += [
            "",
            f'extern "C" void launch_{name}({sig}) {{',
            f"  {name}<<<{m.op.blocks.data}, nullptr, stream>>>({call});",
            "}",
        ]
    return "\n".join(lines) + "\n"


def _mixed_body_lines(m, channels, *, core_expression=None):
    from .ir import CrossDrainOp, StreamOp

    lines = []
    for actor, kernel in m.actors.items():
        guard = "ASC_IS_AIC" if actor == "aic" else "ASC_IS_AIV"
        # ASC_IS_* expands to constexpr(...), part of the if syntax itself.
        lines.append(f"  if {guard} {{ // {actor}: private local banks")
        if actor != "aic":
            lines.append(f"  if (asc_get_sub_block_id() == {int(actor[-1])}) {{")
        ex = execution(m.projections[actor], _program=m.programs[actor])
        stream = next(o for o in kernel.body.block.ops if isinstance(o, StreamOp))
        drains = [o for o in kernel.body.block.ops if isinstance(o, CrossDrainOp)]
        lines.extend(
            "  " + line
            for line in _body_lines(
                ex.program,
                ex,
                stream,
                cross=channels,
                drains=drains,
                core_expression=core_expression,
            )
        )
        if actor != "aic":
            lines.append("  }")
        lines.append("  }")
    return lines


def _emit_dispatch(analysis, *, host_launcher):
    dispatch, checked = analysis
    name, roots = dispatch.op.sym_name.data, dispatch.roots
    signature = ", ".join(f"__gm__ uint8_t* {root}" for root in roots)
    lines = [
        "// xdsl-npu: verified static mixed launch partition for dav-2201",
        '#include "c_api/asc_simd.h"',
        "#include <cstdint>",
        "",
        f'extern "C" __global__ __mix__(1, 2) void {name}({signature}) {{',
    ]
    for (case, _), (m, channels, _) in zip(dispatch.cases, checked):
        lo, hi = case.lower.data, case.upper.data
        lines.append(f"  if (asc_get_block_idx() >= {lo} && asc_get_block_idx() < {hi}) {{")
        lines.extend(
            "  " + line
            for line in _mixed_body_lines(
                m, channels, core_expression=f"asc_get_block_idx() - {lo}"
            )
        )
        lines.append("  }")
    lines.append("}")
    if host_launcher:
        sig = ", ".join(["void* stream"] + [f"void* arg_{root}" for root in roots])
        call = ", ".join(f"(uint8_t*)arg_{root}" for root in roots)
        lines += [
            "",
            f'extern "C" void launch_{name}({sig}) {{',
            f"  {name}<<<{dispatch.op.blocks.data}, nullptr, stream>>>({call});",
            "}",
        ]
    return "\n".join(lines) + "\n"

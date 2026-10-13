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
CTYPE = {"f32": "float", "f16": "half"}


def pointer(ref: Ref, g="g", *, core="0", extra=""):
    terms = []
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
    if task.native in {"muls", "adds"}:
        function = "asc_mul_scalar" if task.native == "muls" else "asc_add_scalar"
        return (
            f"{function}({p[-1]}, {p[0]}, float({task.scalar!r}), "
            f"uint32_t({length(refs[0], ITEMSIZE[refs[0].dtype])}));"
        )
    if task.native in {"add", "mul", "sub", "div"}:
        return f"asc_{task.native}({p[2]}, {p[0]}, {p[1]}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"exp", "rsqrt"}:
        return f"asc_{task.native}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.native == "scalar_load":
        source = pointer(refs[0], core=core).replace("__ubuf__ float*", "__ubuf__ volatile float*")
        return f"npu_value_{task.uid} = *({source});"
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
    if task.native == "mul_gamma":
        rows, cols = task.params
        src = pointer(refs[0], core=core, extra=f"npu_row * {cols * 4}")
        dst = pointer(refs[2], core=core, extra=f"npu_row * {cols * 4}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_mul({dst}, {src}, {p[1]}, uint32_t({cols})); }}"
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


def emit_asc(module, *, host_launcher=True):
    verify_plan(module)
    ex = execution(module)
    p = ex.program
    _, stream = parts(module)
    tasks = {
        t.uid: t for _, t in sorted(enumerate(p.tasks), key=lambda x: (x[1].issue_order, x[0]))
    }
    # Read actual sync placements, then retain the compact physical step loop.
    before, after, pending, previous = {}, {}, [], None
    from .ir import SyncOp

    for op in stream.body.block.ops:
        if isinstance(op, TaskOp):
            before[op.uid.data] = pending
            pending = []
            previous = op.uid.data
        elif isinstance(op, SyncOp):
            if op.action.data == "set":
                after.setdefault(previous, []).append(op)
            else:
                pending.append(op)
    args = [s for s in p.storages if s.external]
    signature = ", ".join(f"__gm__ uint8_t* {s.name}" for s in args)
    kind = "__vector__" if p.core == "aiv" else "__cube__"
    lines = [
        "// xdsl-npu v0.1: dav-2201, single physical core, fixed issue plan",
        '#include "c_api/asc_simd.h"',
        "#include <cstdint>",
        "",
        f'extern "C" __global__ {kind} void {p.name}({signature}) {{',
        "  asc_init();",
        "  asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));",
    ]
    offsets, _ = storage_layout(p.storages)
    if p.blocks > 1:
        lines.append("  const int64_t core_id = asc_get_block_idx();")
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
        if op.action.data == "barrier":
            code = f"asc_sync_pipe(PIPE_{op.pipe.data});"
        else:
            ch = ex.channels[op.channel.data]
            a, b = p.tasks[ch.source].pipe, p.tasks[ch.destination].pipe
            epoch = f"g - {op.distance.data} - {ch.lower}"
            if ch.stride != 1:
                epoch = f"({epoch}) / {ch.stride}"
            event_id = f"{ch.base} + (({epoch}) % {ch.width})"
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
            guard = f"step >= {task.lower + task.stage} && step < {task.upper + task.stage}"
            if task.period != 1:
                guard += f" && (step - {task.stage}) % {task.period} == {task.phase}"
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
    lines += [
        "  // Required kernel lifetime boundary; no PIPE_ALL inside the task stream.",
        "  asc_sync_pipe(PIPE_ALL);",
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

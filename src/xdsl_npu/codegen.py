"""ASC emission consumes verified final IR; no instruction selection or hidden DMA."""

from .ir import TaskOp
from .model import ITEMSIZE, Ref, storage_layout
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


def pointer(ref: Ref, g="g"):
    terms = []
    for offset, stride, modulus in ref.terms:
        if offset:
            terms.append(str(offset))
        if stride:
            index = f"({g} % {modulus})" if modulus else g
            terms.append(f"{stride} * {index}")
    address = " + ".join(terms) or "0"
    q = QUALIFIER[ref.storage.bank]
    return f"({q} {CTYPE[ref.dtype]}*)({ref.storage.name} + ({address}))"


def native_line(task, count):
    refs = task.refs
    p = [pointer(r) for r in refs]

    def length(ref, unit=1):
        base = ref.nbytes // unit
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
    if task.native == "add":
        return f"asc_add({p[2]}, {p[0]}, {p[1]}, uint32_t({length(refs[0], 4)}));"
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
    tasks = {t.uid: t for t in p.tasks}
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
            event_id = f"{ch.base} + (({epoch}) % {ch.width})"
            function = "asc_sync_notify" if op.action.data == "set" else "asc_sync_wait"
            code = f"{function}(PIPE_{a}, PIPE_{b}, {event_id});"
            comment = f" // {ch.name}"
        lo, hi = op.lower.data, op.upper.data
        if (lo, hi) != (0, p.count):
            # The surrounding task guard already establishes 0 <= g < count.
            clauses = ([f"g >= {lo}"] if lo else []) + ([f"g < {hi}"] if hi != p.count else [])
            code = f"if ({' && '.join(clauses)}) {{ {code} }}"
        return code + comment

    if p.count:
        end = p.count + max(t.stage for t in p.tasks)
        lines.append(f"  for (int64_t step = 0; step < {end}; ++step) {{")
        for task in tasks.values():
            lines += [
                f"    // {task.uid}: {task.pipe}, logical generation = step - {task.stage}",
                f"    if (step >= {task.stage} && step < {p.count + task.stage}) {{",
                f"      const int64_t g = step - {task.stage};",
            ]
            lines.extend("      " + sync_line(op) for op in before[task.uid])
            lines.append("      " + native_line(task, p.count))
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
            f"  {p.name}<<<1, nullptr, stream>>>({call});",
            "}",
        ]
    return "\n".join(lines) + "\n"

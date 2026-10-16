"""ASC emission consumes verified final IR; no instruction selection or hidden DMA."""

from .asc_scalar import ScalarScope
from .backend.native import QUALIFIER, load3d_init, native_line
from .dialects.physical import TaskOp
from .model import optional_int, storage_layout
from .sync import execution, parts, verify_plan


def _body_lines(p, ex, stream, *, cross=None, drains=(), core_expression=None):
    """Backend helper; callers must verify the entire enclosing program first."""
    tasks = {
        t.uid: t for _, t in sorted(enumerate(p.tasks), key=lambda x: (x[1].issue_order, x[0]))
    }
    # Read actual sync placements, then retain the compact physical step loop.
    before, after, pending, previous = {}, {}, [], None
    from .dialects.physical import CrossSyncOp, SyncOp

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

    lines += load3d_init(p.load3d)

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
            epoch = scalar.div(scalar.sub(scalar.sub("g", op.distance.data), ch.lower), ch.stride)
            event_id = scalar.add(ch.base, scalar.rem(epoch, ch.width))
            if ch.event_ids is not None:
                from .events import selection_expression

                event_id = selection_expression(ch.event_ids, epoch, scalar=scalar)
            function = "asc_sync_notify" if op.action.data == "set" else "asc_sync_wait"
            code = f"{function}(PIPE_{a}, PIPE_{b}, {scalar.use(event_id)});"
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
            scalar = ScalarScope()
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
            body = [sync_line(op) for op in before[task.uid]]
            body.append(
                native_line(task, p.count, "core_id" if p.blocks > 1 else "0", scalar=scalar)
            )
            body.extend(sync_line(op) for op in after.get(task.uid, []))
            lines.extend("      " + line for line in scalar.finish(body))
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
    from .scope_model import is_scoped

    if is_scoped(module):
        from .scope_codegen import emit_scopes

        return emit_scopes(analysis, host_launcher=host_launcher)
    from .dialects.physical import MixedOp, StaticDispatchOp
    from .loop_model import is_symbolic
    from .sequence_model import is_sequence

    if is_sequence(module):
        from .sequence_codegen import emit_sequence

        return emit_sequence(analysis, host_launcher=host_launcher)

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
    from .dialects.physical import CrossDrainOp, StreamOp

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

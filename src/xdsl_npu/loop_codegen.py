"""ASC loops with uniform runtime counts and guarded event ordinals."""

from .codegen import QUALIFIER, native_line
from .ir import MixedOp
from .model import ITEMSIZE, storage_layout
from .scalar import bindings, from_ssa


def _guard(command):
    terms = []
    if command.lower:
        terms.append(f"g >= {command.lower}")
    if command.upper:
        terms.append(f"g < n - {command.upper}")
    if command.period != 1:
        terms.append(
            "(" + " || ".join(f"g % {command.period} == {r}" for r in command.residues) + ")"
        )
    return " && ".join(terms)


def _ordinal(channel, epoch):
    if channel.period == 1:
        return epoch
    rank = " + ".join(f"(({epoch}) % {channel.period} >= {r})" for r in channel.residues)
    return f"(({epoch}) / {channel.period} * {len(channel.residues)} + {rank} - 1)"


def _event(channel, epoch):
    if len(channel.ids) == 1:
        return str(channel.ids[0])
    ordinal = _ordinal(channel, epoch)
    choices = str(channel.ids[-1])
    for i in range(len(channel.ids) - 2, -1, -1):
        choices = f"(({ordinal}) % {len(channel.ids)} == {i} ? {channel.ids[i]} : {choices})"
    return choices


def emit_loop(analysis, *, host_launcher=True):
    loop, channel_map, sequences, _, _ = analysis
    name = loop.op.sym_name.data
    roots = list(loop.roots)
    size_args = ["npu_size_" + size.sym_name.data for size in loop.parameters]
    signature = [f"__gm__ uint8_t* {root}" for root in roots]
    signature.extend(f"int64_t {arg}" for arg in size_args)
    expressions = [loop.bound] + [
        from_ssa(root.extent_count) for root in loop.roots.values() if root.extent_count is not None
    ]
    expressions += [
        ref.prefix_extent
        for data in loop.actors.values()
        for task in data.program.tasks
        for ref in task.refs
        if ref.prefix_extent is not None
    ]
    uniform_names, uniform_lines = bindings(expressions)
    kind = (
        "__mix__(1, 2)"
        if isinstance(loop.op, MixedOp)
        else ("__cube__" if loop.op.core.data == "aic" else "__vector__")
    )
    lines = [
        "// xdsl-npu: parametric dav-2201 loop; no runtime-length expansion",
        '#include "c_api/asc_simd.h"',
        "#include <cstdint>",
        "",
        f'extern "C" __global__ {kind} void {name}({", ".join(signature)}) {{',
        *uniform_lines,
        f"  const int64_t n = {loop.bound.c(uniform_names) if loop.bound is not None else loop.constant_count};",
    ]
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    for actor, data in loop.actors.items():
        mixed = isinstance(loop.op, MixedOp)
        if mixed:
            lines.append(f"  if {'ASC_IS_AIC' if actor == 'aic' else 'ASC_IS_AIV'} {{ // {actor}")
            if actor != "aic":
                lines.append(f"  if (asc_get_sub_block_id() == {actor[-1]}) {{")
        lines += [
            "  asc_init();",
            "  asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));",
            "  const int64_t core_id = asc_get_block_idx();",
        ]
        offsets, _ = storage_layout(data.program.storages)
        for storage in data.program.storages:
            if not storage.external:
                qualifier = QUALIFIER[storage.bank]
                lines.append(
                    f"  {qualifier} uint8_t* {storage.name} = ({qualifier} uint8_t*){offsets[storage.name]};"
                )
        if data.program.config:
            lines += [
                "  asc_set_l0c_copy_params(1, 0, 0);",
                "  asc_sync_notify(PIPE_S, PIPE_FIX, event_t(0));",
                "  asc_sync_wait(PIPE_S, PIPE_FIX, event_t(0));",
            ]
        maximum = max(task.stage for task in data.program.tasks)
        for task in data.program.tasks:
            if task.native == "scalar_load":
                lines.append(f"  float npu_value_{task.uid} = 0.0f;")
        lines.append(f"  for (int64_t step = 0; step < n + {maximum}; ++step) {{")
        for command in sequences[actor]:
            task = command.task
            lines += [
                f"    {{ // {actor}/{task.uid}: {command.kind}, stage {task.stage}",
                f"      const int64_t g = step - {task.stage};",
                "      if (g >= 0 && g < n) {",
            ]
            if command.kind == "task":
                code = native_line(task, "n", "core_id", uniform_names=uniform_names)
            elif command.kind == "barrier":
                code = f"asc_sync_pipe(PIPE_{task.pipe});"
            else:
                channel = channel_map[command.channel]
                epoch = "g" if command.kind.endswith("set") else f"g - {channel.distance}"
                if channel.kind == "cross":
                    function = (
                        "asc_sync_block_arrive"
                        if command.kind == "cross_set"
                        else "asc_sync_block_wait"
                    )
                    code = f"{function}(PIPE_{task.pipe}, {channel.ids[0]}); // {channel.name}"
                else:
                    a, b = tasks[channel.sources[0]].pipe, tasks[channel.destinations[0]].pipe
                    function = "asc_sync_notify" if command.kind == "set" else "asc_sync_wait"
                    code = f"{function}(PIPE_{a}, PIPE_{b}, event_t({_event(channel, epoch)})); // {channel.name}"
            guard = _guard(command)
            if guard:
                body, marker, comment = code.partition("//")
                code = f"if ({guard}) {{ {body.strip()} }}" + (f" //{comment}" if marker else "")
            lines.append(f"        {code}")
            lines += ["      }", "    }"]
        lines += [
            "  }",
            "  // Final lifetime boundary; no PIPE_ALL in the loop.",
            "  asc_sync_pipe(PIPE_ALL);",
        ]
        if mixed:
            if actor != "aic":
                lines.append("  }")
            lines.append("  }")
    lines.append("}")
    if host_launcher:
        signature = ["void* stream"] + [f"void* arg_{root}" for root in roots]
        arguments = [f"(uint8_t*)arg_{root}" for root in roots]
        signature.extend(f"int64_t {arg}" for arg in size_args)
        arguments.extend(size_args)
        lines += ["", f'extern "C" int launch_{name}({", ".join(signature)}) {{']
        coefficient = max(
            [1] + [s.count_scale for actor in loop.actors.values() for s in actor.program.storages]
        )
        margin = max(
            [64]
            + [
                r.prefix_elements
                for actor in loop.actors.values()
                for task in actor.program.tasks
                for r in task.refs
            ]
        )
        limit = ((1 << 63) - 1 - margin) // coefficient
        for size_arg, size in zip(size_args, loop.parameters):
            lines.append(
                f"  if ({size_arg} < 0 || {size_arg} > {limit} || {size_arg} % {size.multiple.data} != 0) return -1;"
            )
        lines += uniform_lines
        if loop.bound is not None:
            stage_margin = max(
                task.stage for actor in loop.actors.values() for task in actor.program.tasks
            )
            lines.append(
                f"  if ({loop.bound.c(uniform_names)} > {(1 << 63) - 1 - stage_margin}) return -1; // loop step overflow"
            )
        alignments = sorted(
            {
                (ref.prefix_extent.c(uniform_names), 32 // ITEMSIZE[ref.dtype])
                for actor in loop.actors.values()
                for task in actor.program.tasks
                if task.pipe in {"MTE2", "MTE3"}
                for ref in task.refs
                if ref.prefix_extent is not None
            }
        )
        for extent, multiple in alignments:
            lines.append(
                f"  if ({extent} % {multiple} != 0) return -1; // exact DMA prefix alignment"
            )
        # Check the declared extents, not padded loop envelopes. This is a host
        # input contract check, never a device barrier or a hidden allocation.
        for root, allocation in loop.roots.items():
            if allocation.extent_count is not None:
                extent = from_ssa(allocation.extent_count).c(uniform_names)
                lines.append(
                    f"  if ({extent} > {((1 << 63) - 1) // allocation.count_scale.data}) return -1; // extent multiplication overflow"
                )
                nbytes = f"({from_ssa(allocation.extent_count).c(uniform_names)} * {allocation.count_scale.data})"
            else:
                nbytes = str(allocation.nbytes.data)
            lines += [
                f"  const uint64_t npu_bytes_{root} = {nbytes};",
                f"  const uintptr_t npu_ptr_{root} = (uintptr_t)arg_{root};",
                (
                    f"  if (npu_bytes_{root} && (!npu_ptr_{root} || npu_ptr_{root} % 32 || "
                    f"npu_ptr_{root} > UINTPTR_MAX - npu_bytes_{root})) return -1;"
                ),
            ]
        if loop.op.disjoint_external.data:
            for i, left in enumerate(roots):
                for right in roots[i + 1 :]:
                    lines.append(
                        f"  if (npu_bytes_{left} && npu_bytes_{right} && "
                        f"npu_ptr_{left} < npu_ptr_{right} + npu_bytes_{right} && "
                        f"npu_ptr_{right} < npu_ptr_{left} + npu_bytes_{left}) return -1;"
                    )
        lines += [
            f"  {name}<<<{loop.groups}, nullptr, stream>>>({', '.join(arguments)});",
            "  return 0;",
            "}",
        ]
    return "\n".join(lines) + "\n"

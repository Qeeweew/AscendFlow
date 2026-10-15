"""ASC loops with uniform runtime counts and guarded event ordinals."""

from itertools import groupby

from .asc_scalar import ScalarScope
from .codegen import QUALIFIER, load3d_init, native_line
from .ir import InvocationChannelOp, MixedOp, OnceOp
from .model import ITEMSIZE, storage_layout
from .scalar import bindings, facts_from_sizes, from_ssa


def _guard(command, channel=None):
    terms = []
    if command.last_gap is not None:
        terms.append(f"g == n - {command.last_gap + 1}")
    if command.lower:
        terms.append(
            f"(g >= {command.lower} || npu_invocation > 0)"
            if channel is not None and channel.carry
            else f"g >= {command.lower}"
        )
    if command.upper:
        terms.append(
            f"(g < n - {command.upper} || npu_invocation + 1 < npu_invocations)"
            if channel is not None and channel.carry
            else f"g < n - {command.upper}"
        )
    if command.period != 1:
        terms.append(
            "(" + " || ".join(f"g % {command.period} == {r}" for r in command.residues) + ")"
        )
    return " && ".join(terms)


def _event(channel, epoch, scalar):
    if len(channel.ids) == 1:
        return scalar.value(channel.ids[0])
    if channel.period == 1:
        ordinal = epoch
    else:
        rank = scalar.value(0)
        for residue in channel.residues:
            rank = scalar.add(
                rank, scalar.select_lt(scalar.rem(epoch, channel.period), residue, 0, 1)
            )
        ordinal = scalar.sub(
            scalar.add(scalar.mul(scalar.div(epoch, channel.period), len(channel.residues)), rank),
            1,
        )
    index = scalar.rem(ordinal, len(channel.ids))
    base, step = channel.ids[0], channel.ids[1] - channel.ids[0]
    if all(value == base + i * step for i, value in enumerate(channel.ids)):
        return scalar.add(base, scalar.mul(step, index))
    choices = scalar.value(channel.ids[-1])
    for i in range(len(channel.ids) - 2, -1, -1):
        choices = scalar.select_eq(index, i, channel.ids[i], choices)
    return choices


def _region_lines(
    actor,
    data,
    sequences,
    channel_map,
    tasks,
    invocation,
    invocation_channels,
    uniform_names,
    *,
    boundary=None,
):
    lines = []
    maximum = max(task.stage for task in data.program.tasks)
    for task in data.program.tasks:
        if task.native == "scalar_load":
            lines.append(f"  float npu_value_{task.uid} = 0.0f;")
    if invocation is not None:
        lines.append(
            "  for (int64_t npu_invocation = 0; n > 0 && npu_invocation < npu_invocations; ++npu_invocation) {"
        )
        for frame in invocation.frames:
            lines.append(
                f"    __gm__ uint8_t* {frame.name} = {frame.root.sym_name.data} + "
                f"(core_id * npu_invocations + npu_invocation) * "
                f"{frame.extent.c(uniform_names)} * {frame.scale};"
            )
    if isinstance(data.stream, OnceOp):
        lines += [
            "  { // ordinary region: execute once even for an empty child",
            "    const int64_t step = 0;",
        ]
    else:
        lines.append(f"  for (int64_t step = 0; step < n + {maximum}; ++step) {{")
    # Adjacent commands with the same stage have exactly the same generation
    # and activation. Share that guard and a pure scalar scope, preserving every
    # native/sync call and its inner protocol guard in the original order.
    for stage, stage_commands in groupby(sequences[actor], key=lambda c: c.task.stage):
        scalar, inner = ScalarScope(), []
        generation = "step" if stage == 0 else f"step - {stage}"
        lines += [
            f"    {{ // {actor}: stage {stage}",
            f"      const int64_t g = {generation};",
            "      if (g >= 0 && g < n) {",
        ]
        for command in stage_commands:
            task = command.task
            if command.kind == "task":
                code = native_line(task, "n", "core_id", uniform_names=uniform_names, scalar=scalar)
                if boundary is not None:
                    code = boundary(task, code)
                for protocol in invocation_channels:
                    source = tuple(protocol.source.data.split("/"))
                    destination = tuple(protocol.destination.data.split("/"))
                    gap, index = protocol.source_gap.data, protocol.destination_index.data
                    a, b = tasks[source].pipe, tasks[destination].pipe
                    if (actor, task.uid) == destination:
                        code = (
                            f"if (npu_invocation > 0 && n > {max(gap, index)} && g == {index}) {{ "
                            f"asc_sync_wait(PIPE_{a}, PIPE_{b}, event_t({protocol.event_id.data})); }} "
                            + code
                        )
                    if (actor, task.uid) == source:
                        code += (
                            f" if (npu_invocation + 1 < npu_invocations && n > {max(gap, index)} "
                            f"&& g == n - {gap + 1}) {{ asc_sync_notify(PIPE_{a}, PIPE_{b}, "
                            f"event_t({protocol.event_id.data})); }}"
                        )
            elif command.kind == "barrier":
                code = f"asc_sync_pipe(PIPE_{task.pipe});"
            else:
                channel = channel_map[command.channel]
                epoch = (
                    scalar.value("g")
                    if command.kind.endswith("set") or channel.carry
                    else scalar.sub("g", channel.distance)
                )
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
                    code = f"{function}(PIPE_{a}, PIPE_{b}, event_t({scalar.use(_event(channel, epoch, scalar))})); // {channel.name}"
            guard = _guard(command, channel_map.get(command.channel))
            if guard:
                body, marker, comment = code.partition("//")
                code = f"if ({guard}) {{ {body.strip()} }}" + (f" //{comment}" if marker else "")
            inner += [f"// {actor}/{task.uid}: {command.kind}, stage {stage}", code]
        lines.extend(f"        {line}" for line in scalar.finish(inner))
        lines += ["      }", "    }"]
    lines.append("  }")
    if invocation is not None:
        lines.append("  } // serial invocation: no whole-child drain")
    return lines


def emit_loop(analysis, *, host_launcher=True, sequence=None):
    loop, channel_map, sequences, _, _ = analysis
    all_data = [
        d
        for item in ([analysis] if sequence is None else sequence.analyses.values())
        for d in item[0].actors.values()
    ]
    name = loop.op.sym_name.data
    roots = list(loop.roots)
    facts = facts_from_sizes(loop.parameters)
    size_args = ["npu_size_" + size.sym_name.data for size in loop.parameters]
    signature = [f"__gm__ uint8_t* {root}" for root in roots]
    signature.extend(f"int64_t {arg}" for arg in size_args)
    expressions = (
        [loop.host_bound]
        if sequence is None
        else [a[0].host_bound for a in sequence.analyses.values()]
    ) + [
        from_ssa(root.extent_count, facts=facts)
        for root in loop.roots.values()
        if root.extent_count is not None
    ]
    expressions += [
        ref.prefix_extent
        for data in all_data
        for task in data.program.tasks
        for ref in task.refs
        if ref.prefix_extent is not None
    ]
    invocation = loop.invocation
    invocation_channels = [
        op for op in loop.op.body.block.ops if isinstance(op, InvocationChannelOp)
    ]
    if invocation is not None:
        expressions += [invocation.bound] + [frame.extent for frame in invocation.frames]
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
    ]
    if sequence is not None:
        lines.append(
            f"  const int64_t npu_sequence_count = {sequence.sequence.summary.bound.c(uniform_names) if sequence.sequence.summary.bound is not None else 0};"
        )
    elif loop.cyclic is None:
        lines.append(
            f"  const int64_t n = {loop.bound.c(uniform_names) if loop.bound is not None else loop.constant_count};"
        )
    if invocation is not None:
        lines.append(f"  const int64_t npu_invocations = {invocation.bound.c(uniform_names)};")
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
        if loop.cyclic is not None:
            # The descriptor is evaluated once per owner. Its SSA division is
            # known nonnegative after clamping; do not duplicate a nested max
            # in both quotient and remainder of the ceildiv spelling.
            total = loop.cyclic.total.c(uniform_names)
            lines += [
                f"  const int64_t npu_remaining = {total} - core_id;",
                "  const int64_t npu_owned = npu_remaining > 0 ? npu_remaining : 0;",
                f"  const int64_t n = npu_owned / {loop.groups} + (npu_owned % {loop.groups} != 0);",
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
        lines += load3d_init(data.program.load3d)
        if sequence is None:
            lines += _region_lines(
                actor,
                data,
                sequences,
                channel_map,
                tasks,
                invocation,
                invocation_channels,
                uniform_names,
            )
        else:
            from .sequence_codegen import region_boundary

            if sequence.sequence.invocation is not None:
                lines.append(
                    "  for (int64_t npu_invocation = 0; npu_invocation < npu_invocations; ++npu_invocation) {"
                )
                for frame in invocation.frames:
                    lines.append(
                        f"    __gm__ uint8_t* {frame.name} = {frame.root.sym_name.data} + "
                        f"(core_id * npu_invocations + npu_invocation) * "
                        f"{frame.extent.c(uniform_names)} * {frame.scale};"
                    )
            for region_name, leaf in sequence.analyses.items():
                region_loop, region_channels, region_commands, _, _ = leaf
                region_data = region_loop.actors[actor]
                region_tasks = {(actor, t.uid): t for t in region_data.program.tasks}
                bound = (
                    region_loop.bound.c(uniform_names)
                    if region_loop.bound is not None
                    else region_loop.constant_count
                )
                lines += [f"  {{ // region {region_name}", f"  const int64_t n = {bound};"]
                lines += _region_lines(
                    actor,
                    region_data,
                    region_commands,
                    region_channels,
                    region_tasks,
                    None,
                    (),
                    uniform_names,
                    boundary=region_boundary(sequence, region_name),
                )
                lines.append("  } // asynchronous region exit")
            if sequence.sequence.invocation is not None:
                lines.append("  } // serial sequence: no per-call drain")
        lines += [
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
                for actor in all_data
                for task in actor.program.tasks
                for r in task.refs
            ]
        )
        limit = ((1 << 63) - 1 - margin) // coefficient
        for size_arg, size in zip(size_args, loop.parameters):
            lines.append(
                f"  if ({size_arg} < {size.minimum.data} || {size_arg} > {limit} || {size_arg} % {size.multiple.data} != 0) return -1;"
            )
        _, checked_uniform_lines = bindings(expressions, checked=True)
        lines += checked_uniform_lines
        if loop.host_bound is not None:
            stage_margin = max(task.stage for actor in all_data for task in actor.program.tasks)
            lines.append(
                f"  if ({loop.host_bound.c(uniform_names)} > {(1 << 63) - 1 - stage_margin}) return -1; // loop step overflow"
            )
        alignments = sorted(
            {
                (ref.prefix_extent.c(uniform_names), 32 // ITEMSIZE[ref.dtype])
                for actor in all_data
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
                extent = from_ssa(allocation.extent_count, facts=facts).c(uniform_names)
                lines.append(
                    f"  if ({extent} > {((1 << 63) - 1) // allocation.count_scale.data}) return -1; // extent multiplication overflow"
                )
                nbytes = f"({from_ssa(allocation.extent_count, facts=facts).c(uniform_names)} * {allocation.count_scale.data})"
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

"""Static 910B AIC/two-AIV ownership, counting notifications and final proof.

Actor clocks map to a common semantic epoch, independently of physical issue
stages. GM references capture the SAME outer SSA allocation. Local analysis has
conditional imports; only this module's whole-group proof can discharge them.
There is no user event ID, hidden copy, polling, or internal global barrier.
"""

import itertools
import re
from collections import defaultdict, deque
from dataclasses import dataclass, replace

from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from .ir import (
    BufferOp,
    ChannelOp,
    CrossChannelOp,
    CrossDrainOp,
    CrossSyncOp,
    EndOp,
    KernelOp,
    MixedOp,
    StreamOp,
    SyncOp,
    TaskOp,
)
from .model import (
    CompileError,
    Instance,
    Program,
    Ref,
    dependencies,
    extract,
    optional_int,
    ranges_overlap,
    require,
)
from .sync import Graph, autosync, execution, parts

ACTORS = ("aic", "aiv0", "aiv1")
MAX_CROSS_BACKLOG = 15


def compose(name, cube, vector0, vector1, *, cube_epoch=1, disjoint_external=False):
    """Capture matching external declarations into explicit shared SSA roots.

    Source order within each common epoch is AIC, AIV0, AIV1. This declares
    logical effects, not a physical dispatch/completion order between actors.
    All three source kernels are cloned; the caller's modules stay untouched.
    """
    block = Block()
    roots = {}
    kernels = []
    for actor, module, divisor in zip(ACTORS, (cube, vector0, vector1), (cube_epoch, 1, 1)):
        clone = module.clone()
        program = extract(clone)
        kernel, _ = parts(clone)
        require(program.core == ("aic" if actor == "aic" else "aiv"), "mixed actor mismatch")
        kernel.attributes["sym_name"] = StringAttr(actor)
        kernel.attributes["epoch_divisor"] = IntAttr(divisor)
        for op in list(kernel.body.block.ops):
            if not isinstance(op, BufferOp) or not op.external.data:
                continue
            key = op.sym_name.data
            if key not in roots:
                root = op.clone()
                roots[key] = root
                block.add_op(root)
            root = roots[key]
            require(
                root.ref.type == op.ref.type
                and root.nbytes == op.nbytes
                and root.initialized == op.initialized,
                f"shared allocation {key}: incompatible actor declarations",
            )
            op.ref.replace_all_uses_with(root.ref)
            kernel.body.block.erase_op(op)
        clone.body.block.detach_op(kernel)
        kernels.append(kernel)
    require(len({optional_int(k.blocks, 1) for k in kernels}) == 1, "mixed group grids differ")
    block.add_ops(kernels)
    block.add_op(EndOp())
    mixed = MixedOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "target": StringAttr("dav-2201"),
            "blocks": kernels[0].blocks,
            "disjoint_external": IntAttr(int(disjoint_external)),
        },
        regions=[Region(block)],
    )
    result = ModuleOp([mixed])
    extract_mixed(result)
    return result


@dataclass
class MixedProgram:
    op: MixedOp
    roots: dict[str, BufferOp]
    actors: dict[str, KernelOp]
    projections: dict[str, ModuleOp]
    programs: dict[str, Program]
    divisors: dict[str, int]
    epochs: int

    @property
    def imports(self):
        return tuple(name for name, op in self.roots.items() if not op.initialized.data)

    def semantic_instances(self):
        return sorted(
            ((actor, inst) for actor, p in self.programs.items() for inst in p.instances()),
            key=lambda key: (
                key[1].generation // self.divisors[key[0]],
                ACTORS.index(key[0]),
                key[1].generation,
                key[1].task,
            ),
        )

    def physical_effects(self, key, core=0):
        actor, inst = key
        task = self.programs[actor].tasks[inst.task]
        return [
            (
                ref.storage.name if ref.storage.external else f"{actor}/{ref.storage.name}",
                a,
                b,
                mode,
            )
            for ref, mode in zip(task.refs, task.modes_for(inst.generation))
            for a, b in ref.segments(inst.generation, core)
        ]


def project_actor(mixed, actor):
    mapper = {}
    buffers = []
    for op in mixed.body.block.ops:
        if isinstance(op, BufferOp):
            clone = op.clone()
            buffers.append(clone)
            mapper[op.ref] = clone.ref
    kernel = actor.clone(value_mapper=mapper)
    for op in list(kernel.walk()):
        if isinstance(op, (CrossSyncOp, CrossDrainOp)):
            op.parent.erase_op(op)
    first = kernel.body.block.first_op
    kernel.body.block.insert_ops_before(buffers, first)
    return ModuleOp([kernel])


def extract_mixed(module):
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "one mixed kernel per compilation")
    op = module.body.block.first_op
    require(isinstance(op, MixedOp), "expected npu.mixed")
    require(op.target.data == "dav-2201", "mixed protocol is specific to dav-2201")
    require(re.fullmatch(r"[A-Za-z_]\w*", op.sym_name.data), "invalid mixed kernel name")
    require(op.disjoint_external.data == 1, "mixed GM argument bases require a disjoint contract")
    roots, actors = {}, {}
    seen_actor = False
    for child in op.body.block.ops:
        if isinstance(child, BufferOp):
            require(not seen_actor and child.external.data == 1, "shared roots precede actors")
            name = child.sym_name.data
            require(name not in roots, "duplicate shared allocation")
            roots[name] = child
        elif isinstance(child, KernelOp):
            seen_actor = True
            actor = child.sym_name.data
            require(actor in ACTORS and actor not in actors, "expected one AIC and two AIV actors")
            require(
                child.core.data == ("aic" if actor == "aic" else "aiv")
                and optional_int(child.blocks, 1) == op.blocks.data,
                "mixed actor core/grid mismatch",
            )
            require(
                not any(isinstance(o, BufferOp) and o.external.data for o in child.walk()),
                "actor GM must capture an outer shared SSA allocation",
            )
            # Projection removes cross effects only to build a conditional
            # local proof. Every removed effect must have an actual place in
            # the whole-group command stream and the backend.
            streams = [o for o in child.body.block.ops if isinstance(o, StreamOp)]
            require(len(streams) == 1, "mixed actor requires one finite stream")
            for effect in child.walk():
                if isinstance(effect, CrossSyncOp):
                    require(
                        effect.parent is streams[0].body.block,
                        "cross_sync must belong directly to the actor stream",
                    )
                elif isinstance(effect, CrossDrainOp):
                    require(
                        effect.parent is child.body.block,
                        "cross_drain must belong directly to the actor epilogue",
                    )
            actors[actor] = child
        else:
            require(isinstance(child, (CrossChannelOp, EndOp)), "unsupported mixed operation")
    require(tuple(actors) == ACTORS, "mixed semantic actor order must be AIC, AIV0, AIV1")
    projections = {a: project_actor(op, k) for a, k in actors.items()}
    programs = {a: extract(m) for a, m in projections.items()}
    divisors = {a: optional_int(k.epoch_divisor, 1) for a, k in actors.items()}
    require(
        all(1 <= d <= 4096 and programs[a].count % d == 0 for a, d in divisors.items()),
        "actor generation clock must divide into complete static semantic epochs",
    )
    epochs = {p.count // divisors[a] for a, p in programs.items()}
    require(len(epochs) == 1, "mixed actors must participate in every common epoch, including zero")
    result = MixedProgram(op, roots, actors, projections, programs, divisors, epochs.pop())
    verify_group_owners(result)
    return result


def merge_ranges(parts):
    result = []
    for a, b in sorted(parts):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return tuple(result)


def verify_group_owners(m):
    """The group-0 proof is reusable only for identical internal GM translations."""
    writes = {
        root
        for key in m.semantic_instances()
        for root, _, _, mode in m.physical_effects(key)
        if "W" in mode and root in m.roots
    }
    translations = defaultdict(set)
    effects = defaultdict(list)
    for core in range(m.op.blocks.data):
        for actor, p in m.programs.items():
            for task in p.tasks:
                for ref in task.refs:
                    if ref.storage.name in writes:
                        translations[(core, ref.storage.name)].add(
                            ref.address(0, core) - ref.address(0)
                        )
            for inst in p.instances():
                for root, a, b, mode in m.physical_effects((actor, inst), core):
                    if root in m.roots:
                        for access in ("R", "W"):
                            if access in mode:
                                effects[(core, root, access)].append((a, b))
    require(
        all(len(t) == 1 for t in translations.values()),
        "shared written views need identical group translations; core-dependent aliases rejected",
    )
    effects = {key: merge_ranges(parts) for key, parts in effects.items()}
    for (core, root, mode), footprint in effects.items():
        if mode != "W":
            continue
        for other in range(m.op.blocks.data):
            if core != other:
                require(
                    all(
                        not ranges_overlap(footprint, effects.get((other, root, access), ()))
                        for access in ("R", "W")
                    ),
                    f"shared GM {root}: different mixed groups overlap; no inter-group protocol",
                )


def global_dependencies(m):
    """Recompute the physical frontier without any actor import assumptions."""
    origins = m.semantic_instances()
    storages, tasks = {}, []
    for actor, inst in origins:
        task = m.programs[actor].tasks[inst.task]
        refs = []
        for ref in task.refs:
            root = ref.storage.name if ref.storage.external else f"{actor}/{ref.storage.name}"
            storage = replace(ref.storage, name=root)
            storages[root] = storage
            refs.append(
                Ref(
                    storage,
                    ((ref.address(inst.generation), 0, 0, 1),),
                    ref.length(inst.generation),
                    ref.dtype,
                    ref.layout,
                    rows=ref.rows,
                    row_stride=ref.row_stride,
                )
            )
        tasks.append(
            replace(
                task,
                native="materialized_effects",
                refs=tuple(refs),
                modes=task.modes_for(inst.generation),
                lower=0,
                upper=1,
                period=1,
                phase=0,
            )
        )
    flat = Program(m.op.sym_name.data, "mixed", 1, tuple(storages.values()), tuple(tasks), False)
    return [(origins[o.source.task], origins[o.destination.task], o) for o in dependencies(flat)]


@dataclass(frozen=True)
class CrossChannel:
    name: str
    sources: tuple[str, ...]
    destinations: tuple[str, ...]
    flag_id: int
    count: int


@dataclass(frozen=True)
class MixedCommand:
    actor: str
    kind: str
    pipe: str
    instance: Instance | None
    channel: str = ""
    epoch: int = 0
    event: tuple[str, str, int] | None = None


def mixed_execution(module):
    m = extract_mixed(module)
    channels, ids = {}, set()
    for op in m.op.body.block.ops:
        if not isinstance(op, CrossChannelOp):
            continue
        name = op.sym_name.data
        sources = tuple(s.data for s in op.sources)
        destinations = tuple(s.data for s in op.destinations)
        require(
            (sources, destinations) in {(("aic",), ("aiv0", "aiv1")), (("aiv0", "aiv1"), ("aic",))},
            "mode-2 requires the real 1 AIC : 2 AIV collective, with both AIV participants",
        )
        flag_id, count = op.flag_id.data, op.count.data
        require(
            name not in channels and 0 <= flag_id <= 7 and flag_id not in ids,
            "cross-counter namespace requires distinct IDs in [0,7]; no truncation/reserved IDs",
        )
        require(0 < count <= 4096, "cross-channel has invalid finite count")
        ids.add(flag_id)
        channels[name] = CrossChannel(name, sources, destinations, flag_id, count)
    commands = {}
    for actor, kernel in m.actors.items():
        ex = execution(m.projections[actor])
        local = defaultdict(deque)
        for cmd in ex.commands:
            local[(cmd.instance, cmd.kind, cmd.channel)].append(cmd)
        stream = next(o for o in kernel.body.block.ops if isinstance(o, StreamOp))
        before, after, pending, previous = {}, defaultdict(list), [], None
        for op in stream.body.block.ops:
            if isinstance(op, TaskOp):
                before[op.uid.data] = pending
                previous, pending = op.uid.data, []
            elif isinstance(op, (SyncOp, CrossSyncOp)):
                if isinstance(op, CrossSyncOp):
                    # Validate even if this actor has zero instances or none
                    # of the adjacent task's activations select the effect.
                    require(
                        op.generation is stream.body.block.args[0], "wrong cross generation clock"
                    )
                    require(
                        op.action.data in {"set", "wait"}, "cross operation must be Set or Wait"
                    )
                    require(
                        0 <= op.lower.data < op.upper.data <= ex.program.count
                        and 1 <= op.stride.data <= 4096
                        and 1 <= op.epoch_divisor.data <= 4096,
                        "cross operation has invalid activation/epoch clock",
                    )
                    require(
                        op.channel.data in channels, "cross operation references missing channel"
                    )
                if op.action.data == "set":
                    require(previous is not None and not pending, "arrival must follow its source")
                    after[previous].append(op)
                else:
                    pending.append(op)
        require(not pending, "mixed Wait must precede a task; use cross_drain for final returns")
        out = []

        def convert(op, inst, *, actor=actor, local=local, ex=ex):
            g = inst.generation
            stride = optional_int(op.stride, 1)
            if isinstance(op, SyncOp):
                if not (op.lower.data <= g < op.upper.data and (g - op.lower.data) % stride == 0):
                    return None
                cmd = local[(inst, op.action.data, op.channel.data)].popleft()
                return MixedCommand(
                    actor, cmd.kind, cmd.pipe, inst, cmd.channel, cmd.epoch, cmd.event
                )
            if not (op.lower.data <= g < op.upper.data and (g - op.lower.data) % stride == 0):
                return None
            name = op.channel.data
            epoch = g // op.epoch_divisor.data + op.epoch_offset.data
            require(0 <= epoch < channels[name].count, "cross notification epoch out of bounds")
            # On 2201 wait_flag_dev ignores the named pipe. This is a
            # controller instruction, not an entry in the MTE2/V/FIX queue.
            pipe = "S" if op.action.data == "wait" else ex.program.tasks[inst.task].pipe
            return MixedCommand(actor, "cross_" + op.action.data, pipe, inst, name, epoch)

        for inst in ex.program.issued():
            task = ex.program.tasks[inst.task]
            for op in before[task.uid]:
                cmd = convert(op, inst)
                if cmd:
                    out.append(cmd)
            out.append(MixedCommand(actor, "task", task.pipe, inst))
            local[(inst, "task", "")].popleft()
            for op in after[task.uid]:
                cmd = convert(op, inst)
                if cmd:
                    out.append(cmd)
        require(not any(local.values()), "actual local sync placements were not fully consumed")
        seen_stream = False
        for op in kernel.body.block.ops:
            if isinstance(op, StreamOp):
                seen_stream = True
            if isinstance(op, CrossDrainOp):
                name = op.channel.data
                require(
                    seen_stream
                    and name in channels
                    and 0 <= op.lower.data < op.upper.data <= channels[name].count,
                    "cross drain must follow the stream and consume valid return epochs",
                )
                out.extend(
                    MixedCommand(actor, "cross_wait", "S", None, name, g)
                    for g in range(op.lower.data, op.upper.data)
                )
        commands[actor] = out
        for name in channels:
            require(
                len({c.pipe for c in out if c.kind == "cross_set" and c.channel == name}) <= 1,
                "one cross-counter source must notify from one real FIFO pipe",
            )
    return m, channels, commands


def mixed_graph(m, channels, commands):
    graph = Graph()
    begin, end, sends, receives = {}, {}, {}, {}
    local_sends, local_waits = defaultdict(list), {}
    for actor, sequence in commands.items():
        previous_issue, previous_fence, previous_native = {}, {}, {}
        prefixes = defaultdict(list)
        gate = None
        cross_order = {}
        for cmd in sequence:
            inst, pipe = cmd.instance, cmd.pipe
            label = f"{actor}/{inst}.{cmd.kind}:{cmd.channel}[{cmd.epoch}]"
            start = graph.node(label + ".begin")
            finish = (
                graph.node(label + ".end") if cmd.kind in {"task", "set", "cross_set"} else start
            )
            graph.edge(start, finish)
            if gate is not None:
                graph.edge(gate, start)
            if pipe in previous_issue:
                graph.edge(previous_issue[pipe], start)
            if pipe in previous_fence:
                graph.edge(previous_fence[pipe], start)
            previous_issue[pipe] = start
            if cmd.kind == "task":
                key = (actor, inst)
                begin[key], end[key] = start, finish
                if pipe in previous_native and previous_native[pipe].implicit_completion:
                    graph.edge(prefixes[pipe][-1], start)
                prefixes[pipe].append(finish)
                previous_native[pipe] = m.programs[actor].tasks[inst.task]
                if pipe == "S":
                    gate = finish
            else:
                if cmd.kind in {"set", "cross_set", "barrier"}:
                    for done in prefixes[pipe]:
                        graph.edge(done, finish)
                if cmd.kind in {"wait", "barrier"}:
                    previous_fence[pipe] = finish
                if cmd.kind == "set":
                    local_sends[(actor, cmd.event)].append((finish, cmd.channel, cmd.epoch))
                elif cmd.kind == "wait":
                    local_waits[(actor, cmd.channel, cmd.epoch)] = finish
                    if pipe == "S":
                        gate = finish
                elif cmd.kind.startswith("cross_"):
                    key = (actor, cmd.channel, cmd.epoch)
                    records = sends if cmd.kind == "cross_set" else receives
                    require(key not in records, "duplicate cross epoch/participant")
                    records[key] = finish
                    if cmd.kind == "cross_wait":
                        # dav-2201: blocks every subsequent dispatch on THIS core.
                        gate = finish
                    else:
                        # Notifications from one source pipe are FIFO; source
                        # tasks remain asynchronous after the arrival instruction.
                        if cmd.channel in cross_order:
                            graph.edge(cross_order[cmd.channel], finish)
                        cross_order[cmd.channel] = finish
        # Literal sequence ordinals must match counting hardware's untagged tokens.
        for name, channel in channels.items():
            for kind, participants in (
                ("cross_set", channel.sources),
                ("cross_wait", channel.destinations),
            ):
                epochs = [c.epoch for c in sequence if c.kind == kind and c.channel == name]
                require(
                    epochs == (list(range(channel.count)) if actor in participants else []),
                    f"{name}/{actor}: missing, reordered or inactive collective epoch",
                )
    for name, channel in channels.items():
        for epoch in range(channel.count):
            for src in channel.sources:
                for dst in channel.destinations:
                    graph.edge(sends[(src, name, epoch)], receives[(dst, name, epoch)])
    for (actor, name, epoch), wait in local_waits.items():
        matches = [
            node
            for (a, _), items in local_sends.items()
            if a == actor
            for node, ch, g in items
            if ch == name and g == epoch
        ]
        require(len(matches) == 1, "local event pair is missing in the mixed graph")
        graph.edge(matches[0], wait)
    graph.close()
    return graph, begin, end, sends, receives, local_sends, local_waits


def verify_mixed(module):
    m, channels, commands = mixed_execution(module)
    obligations = global_dependencies(m)
    graph, begin, end, sends, receives, local_sends, local_waits = mixed_graph(
        m, channels, commands
    )
    for source, destination, o in sorted(obligations, key=lambda x: x[2].hazard != "WAR"):
        require(
            graph.precedes(end[source], begin[destination]),
            f"missing mixed {o.hazard} {o.milestone}: {source} -> {destination} "
            f"on {o.storage}[{o.start}:{o.stop}]",
        )
    for actor, program in m.programs.items():
        ids = {task.uid: index for index, task in enumerate(program.tasks)}
        for inst in program.instances():
            source = program.tasks[inst.task].scalar_source
            if source is not None:
                require(
                    graph.precedes(
                        end[(actor, Instance(inst.generation, ids[source]))], begin[(actor, inst)]
                    ),
                    "mixed Scalar value was not captured before dispatch",
                )
    for (actor, event), sets in local_sends.items():
        for (_, name, epoch), (next_set, _, _) in itertools.pairwise(sets):
            require(
                graph.precedes(local_waits[(actor, name, epoch)], next_set),
                f"{actor}: binary re-Set before Wait on {event}",
            )
    peaks = {}
    for name, channel in channels.items():
        peak = 0
        for src in channel.sources:
            for dst in channel.destinations:
                consumed = 0
                for i in range(channel.count):
                    # Both source notification and controller Wait ordinals
                    # are FIFO, hence completed receives form a prefix.
                    while consumed < i and graph.precedes(
                        receives[(dst, name, consumed)], sends[(src, name, i)]
                    ):
                        consumed += 1
                    peak = max(peak, i + 1 - consumed)
                    require(
                        i + 1 - consumed <= MAX_CROSS_BACKLOG,
                        f"{name}: notification backlog can exceed 15; no unsafe counter overflow",
                    )
        peaks[name] = peak
    return {
        "kernel": m.op.sym_name.data,
        "groups": m.op.blocks.data,
        "physical_actors_per_group": {"aic": 1, "aiv": 2},
        "epochs": m.epochs,
        "tasks_per_group": len(begin),
        "obligations_per_group": len(obligations),
        "cross_counter_ids": {name: ch.flag_id for name, ch in channels.items()},
        "cross_backlog_upper_bounds": peaks,
        "scratch_bytes": sum(
            op.nbytes.data
            for op in m.roots.values()
            if not op.initialized.data and op.sym_name.data == "scratch"
        ),
        "verification": "all static instances and exact group translations; trusted dav-2201 contracts",
        "terminal_drain": "every cross return consumed; each actor drains its issued pipelines",
    }


def cross_op(g, action, channel, lo, hi, stride=1, divisor=1, offset=0):
    return CrossSyncOp.create(
        operands=[g],
        attributes={
            "action": StringAttr(action),
            "channel": StringAttr(channel),
            "lower": IntAttr(lo),
            "upper": IntAttr(hi),
            "stride": IntAttr(stride),
            "epoch_divisor": IntAttr(divisor),
            "epoch_offset": IntAttr(offset),
        },
    )


def explain_mixed(module):
    m, channels, commands = mixed_execution(module)
    verification = verify_mixed(module)

    def label(key):
        actor, inst = key
        return f"{actor}/{m.programs[actor].tasks[inst.task].uid}[{inst.generation}]"

    return {
        "verification": verification,
        "cross_channels": {
            name: {
                "sources": ch.sources,
                "destinations": ch.destinations,
                "flag_id": ch.flag_id,
                "count": ch.count,
            }
            for name, ch in channels.items()
        },
        "cross_obligations": [
            {
                "source": label(src),
                "destination": label(dst),
                "hazard": o.hazard,
                "storage": o.storage,
                "bytes": [o.start, o.stop],
                "milestone": o.milestone,
            }
            for src, dst, o in global_dependencies(m)
            if src[0] != dst[0]
        ],
        "commands": {
            actor: [
                {
                    "kind": c.kind,
                    "pipe": c.pipe,
                    "task": label((actor, c.instance)) if c.instance else "epilogue",
                    "channel": c.channel,
                    "epoch": c.epoch,
                }
                for c in sequence
            ]
            for actor, sequence in commands.items()
        },
    }


def install_actor(m, name, module):
    kernel, _ = parts(module)
    old = m.actors[name]
    for op in list(kernel.body.block.ops):
        if isinstance(op, BufferOp) and op.external.data:
            op.ref.replace_all_uses_with(m.roots[op.sym_name.data].ref)
            kernel.body.block.erase_op(op)
    module.body.block.detach_op(kernel)
    m.op.body.block.insert_op_before(kernel, old)
    m.op.body.block.erase_op(old)


def autosync_mixed(module):
    result = module.clone()
    m = extract_mixed(result)
    require(
        not any(
            isinstance(o, (ChannelOp, SyncOp, CrossChannelOp, CrossSyncOp, CrossDrainOp))
            for o in result.walk()
        ),
        "mixed AutoSync expects unsynchronized physical IR",
    )
    obligations = global_dependencies(
        m
    )  # Undefined imports fail here, before conditional analysis.
    cross = [(src, dst, o) for src, dst, o in obligations if src[0] != dst[0]]
    for actor, projection in m.projections.items():
        install_actor(m, actor, autosync(projection, _imports=m.imports))
    m = extract_mixed(result)
    if not m.epochs:
        verify_mixed(result)
        return result
    roots = {o.storage for _, _, o in cross}
    require(len(roots) == 1, "first mixed model requires exactly one shared GM handoff ring")
    raw = [(src, dst, o) for src, dst, o in cross if o.hazard == "RAW"]
    require(
        raw and all(src[0] == "aic" and dst[0] in {"aiv0", "aiv1"} for src, dst, _ in raw),
        "mixed handoff must be FIX -> two AIV GM imports",
    )
    writers = {src[1].task for src, _, _ in raw}
    readers = {a: {dst[1].task for _, dst, _ in raw if dst[0] == a} for a in ACTORS[1:]}
    require(
        len(writers) == 1 and all(len(r) == 1 for r in readers.values()),
        "handoff needs one producer and one last GM reader per AIV/epoch",
    )
    writer = writers.pop()
    readers = {a: indices.pop() for a, indices in readers.items()}
    endpoints = {"aic": writer, **readers}
    for actor, index in endpoints.items():
        task = m.programs[actor].tasks[index]
        active = [g for g in range(m.programs[actor].count) if task.active(g)]
        divisor = m.divisors[actor]
        require(
            len(active) == m.epochs
            and active == [active[0] + i * divisor for i in range(m.epochs)]
            and active[0] < divisor
            and task.pipe == ("FIX" if actor == "aic" else "MTE2"),
            "handoff endpoint must execute once per epoch on its real producer/read pipe",
        )
    require(
        all(
            src[1].generation // m.divisors[src[0]] == dst[1].generation // m.divisors[dst[0]]
            for src, dst, _ in raw
        ),
        "GM import must read its own logical epoch",
    )
    war = [(src, dst, o) for src, dst, o in cross if o.hazard == "WAR"]
    require(
        len(war) == len(cross) - len(raw)
        and all(
            src[0] in readers
            and src[1].task == readers[src[0]]
            and dst[0] == "aic"
            and dst[1].task == writer
            for src, dst, _ in war
        ),
        "unsupported shared ownership: only producer/last-reader ring RAW and WAR are implemented",
    )
    distances = {
        dst[1].generation // m.divisors[dst[0]] - src[1].generation // m.divisors[src[0]]
        for src, dst, _ in war
    }
    require(len(distances) <= 1, "handoff ring reuse must have one constant epoch distance")
    slots = distances.pop() if distances else m.epochs
    require(
        1 <= slots <= 15, "handoff in-flight window must fit the 15-notification hardware limit"
    )
    for index, (name, sources, destinations) in enumerate(
        (
            ("gm_ready", ("aic",), ACTORS[1:]),
            ("gm_credit", ACTORS[1:], ("aic",)),
        )
    ):
        channel = CrossChannelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "sources": ArrayAttr(StringAttr(a) for a in sources),
                "destinations": ArrayAttr(StringAttr(a) for a in destinations),
                "flag_id": IntAttr(index),
                "count": IntAttr(m.epochs),
            }
        )
        m.op.body.block.insert_op_before(channel, m.actors["aic"])
    for actor, index in endpoints.items():
        kernel = m.actors[actor]
        stream = next(o for o in kernel.body.block.ops if isinstance(o, StreamOp))
        task = [o for o in stream.body.block.ops if isinstance(o, TaskOp)][index]
        g = stream.body.block.args[0]
        divisor = m.divisors[actor]
        phase = next(i for i in range(divisor) if m.programs[actor].tasks[index].active(i))
        hi = m.programs[actor].count
        if actor == "aic":
            stream.body.block.insert_op_after(
                cross_op(g, "set", "gm_ready", phase, hi, divisor, divisor), task
            )
            if slots < m.epochs:
                stream.body.block.insert_op_before(
                    cross_op(
                        g,
                        "wait",
                        "gm_credit",
                        slots * divisor + phase,
                        hi,
                        divisor,
                        divisor,
                        -slots,
                    ),
                    task,
                )
            drain = CrossDrainOp.create(
                attributes={
                    "channel": StringAttr("gm_credit"),
                    "lower": IntAttr(max(0, m.epochs - slots)),
                    "upper": IntAttr(m.epochs),
                }
            )
            kernel.body.block.insert_op_before(drain, kernel.body.block.last_op)
        else:
            stream.body.block.insert_op_before(
                cross_op(g, "wait", "gm_ready", phase, hi, divisor, divisor), task
            )
            stream.body.block.insert_op_after(
                cross_op(g, "set", "gm_credit", phase, hi, divisor, divisor), task
            )
    verify_mixed(result)
    # Cross read-retirement can also cover actor-local ring WAW. Remove a
    # conflict barrier only after the whole-group graph proves it redundant.
    for op in list(result.walk()):
        if not isinstance(op, SyncOp) or op.action.data != "barrier":
            continue
        block, following = op.parent, op.next_op
        block.detach_op(op)
        try:
            verify_mixed(result)
        except CompileError:
            block.insert_op_before(op, following)
    verify_mixed(result)
    return result

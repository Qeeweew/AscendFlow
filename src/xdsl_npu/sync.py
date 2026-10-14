"""Finite local AutoSync and a final verifier that re-reads actual xDSL PlanIR.

Proposal and checking are separate. No certificate string grants a completion
edge. Event ID reuse is checked only after real command placement is known.
"""

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass

from xdsl.dialects.builtin import IntAttr, ModuleOp, StringAttr

from .ir import ChannelOp, KernelOp, StreamOp, SyncOp, TaskOp
from .model import (
    IMPLICIT_COMPLETION,
    CompileError,
    Instance,
    Program,
    dependencies,
    extract,
    optional_int,
    require,
    storage_layout,
)


@dataclass(frozen=True)
class Channel:
    name: str
    source: int
    destination: int
    distance: int
    lower: int
    upper: int
    width: int
    base: int
    reason: str
    stride: int = 1
    event_ids: tuple[int, ...] | None = None

    def physical_id(self, epoch):
        ordinal = (epoch - self.lower) // self.stride
        if self.event_ids is not None:
            return self.event_ids[ordinal]
        return self.base + ordinal % self.width


@dataclass(frozen=True)
class Command:
    kind: str
    pipe: str
    instance: Instance
    channel: str = ""
    epoch: int = 0
    event: tuple[str, str, int] | None = None


@dataclass
class Execution:
    program: Program
    commands: list[Command]
    channels: dict[str, Channel]

    def queues(self):
        queues = defaultdict(list)
        for command in self.commands:
            queues[command.pipe].append(command)
        return dict(queues)


def parts(module):
    kernel = module.body.block.first_op
    assert isinstance(kernel, KernelOp)
    stream = next(op for op in kernel.body.block.ops if isinstance(op, StreamOp))
    return kernel, stream


def execution(module: ModuleOp, *, _program=None) -> Execution:
    # Internal planning reuses the already verified immutable native effects
    # while changing only sync placements. Final verification always extracts
    # those effects again. This is not an input certificate or a fast mode.
    p = extract(module) if _program is None else _program
    kernel, stream = parts(module)
    task_indices = {t.uid: i for i, t in enumerate(p.tasks)}
    channels = {}
    usage = defaultdict(set)
    for op in kernel.body.block.ops:
        if not isinstance(op, ChannelOp):
            continue
        name, src, dst = op.sym_name.data, op.source.data, op.destination.data
        require(
            name not in channels and src in task_indices and dst in task_indices,
            f"invalid or duplicate channel {name}",
        )
        source, destination = task_indices[src], task_indices[dst]
        d, lo, hi, w, base, stride = (
            op.distance.data,
            op.lower.data,
            op.upper.data,
            op.width.data,
            op.base.data,
            optional_int(op.stride, 1),
        )
        require(
            0 <= d and 0 <= lo < hi <= p.count and hi + d <= p.count and 1 <= stride <= 4096,
            f"{name}: unsupported activation/generation span",
        )
        require(
            1 <= w <= 8 and 0 <= base and base + w <= 8,
            f"{name}: local binary event bank has 8 IDs",
        )
        event_ids = None if op.event_ids is None else tuple(a.data for a in op.event_ids)
        if event_ids is not None:
            require(
                (base, w) == (0, 1),
                f"{name}: explicit event IDs require neutral base=0,width=1",
            )
            require(
                len(event_ids) == len(range(lo, hi, stride))
                and all(isinstance(e, int) and 0 <= e < 8 for e in event_ids),
                f"{name}: explicit event IDs must select [0,7] for every notification ordinal",
            )
        a, b = p.tasks[source].pipe, p.tasks[destination].pipe
        require(
            a != b, f"{name}: same-pipe synchronization must use its native completion contract"
        )
        channels[name] = Channel(
            name, source, destination, d, lo, hi, w, base, op.reason.data, stride, event_ids
        )
        usage[(a, b)].update(range(base, base + w))
    before = defaultdict(list)
    after = defaultdict(list)
    pending = []
    previous = None
    for op in stream.body.block.ops:
        if isinstance(op, TaskOp):
            index = task_indices[op.uid.data]
            before[index] = pending
            pending = []
            previous = index
        elif isinstance(op, SyncOp):
            require(
                op.generation is stream.body.block.args[0], "sync uses the wrong generation clock"
            )
            action = op.action.data
            require(action in {"set", "wait", "barrier"}, f"unknown sync action {action}")
            require(
                0 <= op.lower.data < op.upper.data <= p.count
                and op.distance.data >= 0
                and 1 <= optional_int(op.stride, 1) <= 4096,
                "sync has invalid active generation span",
            )
            if action == "set":
                require(previous is not None and not pending, "Set must follow its source task")
                after[previous].append(op)
            else:
                pending.append(op)
    require(not pending, "Wait/barrier must precede its consuming task")
    commands = []
    seen_sets, seen_waits = set(), set()

    def convert(op, inst):
        action, name = op.action.data, op.channel.data
        stride = optional_int(op.stride, 1)
        if not (
            op.lower.data <= inst.generation < op.upper.data
            and (inst.generation - op.lower.data) % stride == 0
        ):
            return None
        pipe = p.tasks[inst.task].pipe
        if action == "barrier":
            require(
                not name and op.pipe.data == pipe and pipe not in IMPLICIT_COMPLETION,
                "invalid same-pipe barrier",
            )
            return Command("barrier", pipe, inst)
        require(name in channels, f"sync references missing channel {name}")
        ch = channels[name]
        require(op.pipe.data == pipe, f"{name}: sync is on the wrong execution pipe")
        require(stride == ch.stride, f"{name}: sync uses the wrong periodic epoch stride")
        if action == "set":
            require(
                inst.task == ch.source
                and op.distance.data == 0
                and (op.lower.data, op.upper.data) == (ch.lower, ch.upper),
                f"{name}: Set has wrong source, generation or activation",
            )
            epoch = inst.generation
            seen = seen_sets
        else:
            require(
                inst.task == ch.destination
                and op.distance.data == ch.distance
                and (op.lower.data, op.upper.data)
                == (ch.lower + ch.distance, ch.upper + ch.distance),
                f"{name}: Wait has wrong destination, generation or activation",
            )
            epoch = inst.generation - ch.distance
            seen = seen_waits
        require((name, epoch) not in seen, f"{name}[{epoch}]: duplicate {action}")
        seen.add((name, epoch))
        event = (
            p.tasks[ch.source].pipe,
            p.tasks[ch.destination].pipe,
            ch.physical_id(epoch),
        )
        return Command(action, pipe, inst, name, epoch, event)

    for inst in p.issued():
        for op in before[inst.task]:
            command = convert(op, inst)
            if command:
                commands.append(command)
        commands.append(Command("task", p.tasks[inst.task].pipe, inst))
        for op in after[inst.task]:
            command = convert(op, inst)
            if command:
                commands.append(command)
    expected = {
        (ch.name, g) for ch in channels.values() for g in range(ch.lower, ch.upper, ch.stride)
    }
    require(
        seen_sets == seen_waits == expected,
        "event pairs missing: incomplete ready/release lifecycle",
    )
    return Execution(p, commands, channels)


class Graph:
    def __init__(self):
        self.edges: list[set[int]] = []
        self.labels: list[str] = []
        self.reach: list[int] | None = None

    def node(self, label):
        n = len(self.edges)
        self.edges.append(set())
        self.labels.append(label)
        return n

    def edge(self, a, b):
        if a != b:
            self.edges[a].add(b)

    def close(self):
        indegree = [0] * len(self.edges)
        for targets in self.edges:
            for b in targets:
                indegree[b] += 1
        queue = deque(i for i, d in enumerate(indegree) if not d)
        order = []
        while queue:
            a = queue.popleft()
            order.append(a)
            for b in self.edges[a]:
                indegree[b] -= 1
                if not indegree[b]:
                    queue.append(b)
        if len(order) != len(self.edges):
            blocked = [self.labels[i] for i, d in enumerate(indegree) if d][:6]
            raise CompileError(f"fixed issue plan has a completion/wait cycle: {blocked}")
        reach = [0] * len(self.edges)
        for a in reversed(order):
            for b in self.edges[a]:
                reach[a] |= (1 << b) | reach[b]
        self.reach = reach

    def precedes(self, a, b):
        assert self.reach is not None
        return bool(self.reach[a] & (1 << b))


def completion_graph(ex: Execution):
    graph = Graph()
    begin, end = {}, {}
    notifications, waits = {}, {}
    physical_sets = defaultdict(list)
    previous_issue, previous_fence = {}, {}
    dispatch_gate = None
    tasks_in_pipe = defaultdict(list)
    previous_native = {}
    for cmd in ex.commands:
        inst, pipe = cmd.instance, cmd.pipe
        suffix = f"({cmd.channel})" if cmd.channel else ""
        label = f"{ex.program.tasks[inst.task].uid}[{inst.generation}].{cmd.kind}{suffix}"
        start = graph.node(label + ".begin")
        finish = graph.node(label + ".end") if cmd.kind in {"task", "set"} else start
        graph.edge(start, finish)
        if dispatch_gate is not None:
            graph.edge(dispatch_gate, start)
        if pipe in previous_issue:
            graph.edge(previous_issue[pipe], start)  # issue order is deliberately weak.
        if pipe in previous_fence:
            graph.edge(previous_fence[pipe], start)
        previous_issue[pipe] = start
        if cmd.kind == "task":
            begin[inst], end[inst] = start, finish
            if pipe in previous_native and previous_native[pipe].implicit_completion:
                graph.edge(tasks_in_pipe[pipe][-1], start)
            tasks_in_pipe[pipe].append(finish)
            previous_native[pipe] = ex.program.tasks[inst.task]
            if pipe == "S":
                # Scalar UB load is synchronous and its by-value result is
                # captured before the controller can dispatch later work.
                dispatch_gate = finish
        else:
            if cmd.kind in {"set", "barrier"}:
                for done in tasks_in_pipe[pipe]:
                    graph.edge(done, finish)
            # Set publishes after prior work, but is NOT a source-pipe fence.
            # Following native instructions may begin before the Set takes effect.
            if cmd.kind in {"wait", "barrier"}:
                previous_fence[pipe] = finish
            if cmd.kind == "set":
                notifications[(cmd.channel, cmd.epoch)] = finish
                physical_sets[cmd.event].append((finish, cmd.channel, cmd.epoch))
            elif cmd.kind == "wait":
                waits[(cmd.channel, cmd.epoch)] = finish
                if pipe == "S":
                    # A Scalar hard-event wait stops later core dispatch;
                    # instructions already dispatched remain independent.
                    dispatch_gate = finish
    for key, producer in notifications.items():
        graph.edge(producer, waits[key])
    graph.close()
    return graph, begin, end, waits, physical_sets


def verify_plan(module: ModuleOp, *, _imports=()) -> dict:
    from .ir import MixedOp, StaticDispatchOp

    if isinstance(module.body.block.first_op, StaticDispatchOp):
        from .dispatch import verify_dispatch

        require(not _imports, "whole launch proofs cannot assume imports")
        return verify_dispatch(module)

    if isinstance(module.body.block.first_op, MixedOp):
        from .mixed import verify_mixed

        require(not _imports, "whole mixed proofs cannot assume imports")
        return verify_mixed(module)
    ex = execution(module)
    obligations = dependencies(ex.program, _imports=_imports)  # recomputed from final effects.
    graph, begin, end, waits, physical_sets = completion_graph(ex)
    # Diagnose a missing physical release before its secondary lost ready paths.
    for o in sorted(obligations, key=lambda o: o.hazard != "WAR"):
        require(
            graph.precedes(end[o.source], begin[o.destination]),
            f"missing {o.hazard} {o.milestone}: "
            f"{ex.program.tasks[o.source.task].uid}[{o.source.generation}] -> "
            f"{ex.program.tasks[o.destination.task].uid}[{o.destination.generation}] "
            f"on {o.storage}[{o.start}:{o.stop}]",
        )
    indices = {task.uid: i for i, task in enumerate(ex.program.tasks)}
    for inst in ex.program.instances():
        task = ex.program.tasks[inst.task]
        if task.scalar_source is not None:
            provider = Instance(inst.generation, indices[task.scalar_source])
            require(
                graph.precedes(end[provider], begin[inst]),
                f"Scalar value {task.scalar_source} is not captured before {task.uid}",
            )
    for event, sets in physical_sets.items():
        # Same physical direction means one source pipe, hence totally ordered sets.
        for (_, name, epoch), (next_set, next_name, next_epoch) in itertools.pairwise(sets):
            require(
                graph.precedes(waits[(name, epoch)], next_set),
                f"binary event {event} may be re-set before old Wait consumes it: "
                f"{name}[{epoch}] -> {next_name}[{next_epoch}]",
            )
    bank = defaultdict(set)
    for event in physical_sets:
        bank[f"{ex.program.core}:{event[0]}->{event[1]}"].add(event[2])
    if ex.program.config:
        bank["aic:S->FIX"].add(0)
    _, peaks = storage_layout(ex.program.storages)
    return {
        "tasks": len(begin),
        "independent_cores": ex.program.blocks,
        "obligations": len(obligations),
        "channels": len(ex.channels),
        "event_ids": {k: sorted(v) for k, v in bank.items()},
        "event_selection_tables": {
            ch.name: list(ch.event_ids) for ch in ex.channels.values() if ch.event_ids is not None
        },
        "storage_bytes": {**peaks, "GM": sum(s.nbytes for s in ex.program.storages if s.external)},
        "verification": (
            "conditional actor-local proof; shared imports still need a whole-group proof"
            if _imports
            else "all instances of this static finite stream; trusted native contracts"
        ),
        "conditional_imports": list(_imports),
        "terminal_drain": "kernel return drains all issued pipelines",
    }


def sync_op(g, action, channel, pipe, lower, upper, distance=0, stride=1):
    return SyncOp.create(
        operands=[g],
        attributes={
            "action": StringAttr(action),
            "channel": StringAttr(channel),
            "pipe": StringAttr(pipe),
            "lower": IntAttr(lower),
            "upper": IntAttr(upper),
            "distance": IntAttr(distance),
            "stride": IntAttr(stride),
        },
    )


def arithmetic_span(epochs):
    """A finite arithmetic progression; no invented notifications in its holes."""
    lo, hi = epochs[0], epochs[-1] + 1
    stride = epochs[1] - lo if len(epochs) > 1 else 1
    require(
        epochs == list(range(lo, hi, stride)),
        "irregular activation needs a richer stream contract; cannot invent event pairs",
    )
    return lo, hi, stride


def explain_plan(module: ModuleOp) -> dict:
    from .ir import MixedOp, StaticDispatchOp

    if isinstance(module.body.block.first_op, StaticDispatchOp):
        from .dispatch import explain_dispatch

        return explain_dispatch(module)

    if isinstance(module.body.block.first_op, MixedOp):
        from .mixed import explain_mixed

        return explain_mixed(module)
    """Concrete finite witnesses, regenerated from final command locations."""
    verify_plan(module)
    ex = execution(module)
    graph, begin, end, waits, physical_sets = completion_graph(ex)

    def path(source, destination):
        previous = {source: None}
        queue = deque([source])
        while queue and destination not in previous:
            node = queue.popleft()
            for target in sorted(graph.edges[node]):
                if target not in previous:
                    previous[target] = node
                    queue.append(target)
        nodes = []
        node = destination
        while node is not None:
            nodes.append(graph.labels[node])
            node = previous[node]
        return list(reversed(nodes))

    hazards = [
        {
            "hazard": o.hazard,
            "storage": o.storage,
            "bytes": [o.start, o.stop],
            "source_generation": o.source.generation,
            "destination_generation": o.destination.generation,
            "required_milestone": o.milestone,
            "completion_path": path(end[o.source], begin[o.destination]),
        }
        for o in dependencies(ex.program)
    ]
    reuse = []
    for event, sets in physical_sets.items():
        for a, b in itertools.pairwise(sets):
            reuse.append(
                {
                    "physical_event": event,
                    "old_channel": a[1],
                    "old_epoch": a[2],
                    "next_channel": b[1],
                    "next_epoch": b[2],
                    "consumption_path": path(waits[(a[1], a[2])], b[0]),
                }
            )
    return {
        "physical_obligations": hazards,
        "binary_event_reuse": reuse,
        "scope": "exact static finite instances, under the documented native contracts",
    }


def autosync(module: ModuleOp, *, _imports=()) -> ModuleOp:
    """Preserve issue/stage/storage. Reduce redundant *whole channels* with witnesses.

    The finite proof permits testing ring widths and removing channels, with no
    assumption that the buffer ring equals the event ring. Capacity failure is
    a diagnostic, not permission to serialize or enlarge buffers.
    """
    from .ir import MixedOp, StaticDispatchOp

    if isinstance(module.body.block.first_op, StaticDispatchOp):
        from .dispatch import autosync_dispatch

        require(not _imports, "whole launch planning cannot assume imports")
        return autosync_dispatch(module)

    if isinstance(module.body.block.first_op, MixedOp):
        from .mixed import autosync_mixed

        require(not _imports, "whole mixed planning cannot assume imports")
        return autosync_mixed(module)
    result = module.clone()
    p = extract(result)
    kernel, stream = parts(result)
    require(
        not any(isinstance(o, (ChannelOp, SyncOp)) for o in result.walk()),
        "AutoSync expects unsynchronized physical IR",
    )
    obs = dependencies(p, _imports=_imports)
    groups = defaultdict(list)
    for o in obs:
        a, b = p.tasks[o.source.task], p.tasks[o.destination.task]
        if a.pipe == b.pipe and a.implicit_completion:
            continue
        d = o.destination.generation - o.source.generation
        groups[(o.source.task, o.destination.task, d)].append(o)
    task_ops = [o for o in stream.body.block.ops if isinstance(o, TaskOp)]
    g = stream.body.block.args[0]
    proposals = []
    for (src, dst, d), records in groups.items():
        active = sorted({o.source.generation for o in records})
        reason = "; ".join(
            sorted({f"{o.hazard}:{o.storage}[{o.start}:{o.stop}]:{o.milestone}" for o in records})
        )
        try:
            spans = [arithmetic_span(active)]
        except CompileError:
            # A finite union of contiguous runs, e.g. accumulator dependencies
            # inside each output's K loop with independent first-write holes.
            spans = [
                arithmetic_span([value for _, value in group])
                for _, group in itertools.groupby(enumerate(active), lambda pair: pair[1] - pair[0])
            ]
        proposals.extend((src, dst, d, lo, hi, stride, reason) for lo, hi, stride in spans)

    def materialize(proposals, widths=None, *, virtual=False):
        for op in list(kernel.body.block.ops):
            if isinstance(op, ChannelOp):
                kernel.body.block.erase_op(op)
        for op in list(stream.body.block.ops):
            if isinstance(op, SyncOp):
                stream.body.block.erase_op(op)
        banks = defaultdict(int)
        for i, (src, dst, d, lo, hi, stride, reason) in enumerate(proposals):
            a, b = p.tasks[src], p.tasks[dst]
            if a.pipe == b.pipe:
                stream.body.block.insert_op_before(
                    sync_op(g, "barrier", "", b.pipe, lo + d, hi + d, d, stride), task_ops[dst]
                )
                continue
            name = f"c{i}_{a.uid}_{b.uid}_d{d}"
            epochs = len(range(lo, hi, stride))
            w = widths[i] if widths is not None else min(8, epochs)
            # Before allocation, logical pairs are independent in the completion graph.
            # ID reuse is deliberately not inferred until the final allocation is checked.
            base = 0 if virtual else banks[(a.pipe, b.pipe)]
            banks[(a.pipe, b.pipe)] += w
            channel = ChannelOp.create(
                attributes={
                    "sym_name": StringAttr(name),
                    "source": StringAttr(a.uid),
                    "destination": StringAttr(b.uid),
                    "distance": IntAttr(d),
                    "lower": IntAttr(lo),
                    "upper": IntAttr(hi),
                    "stride": IntAttr(stride),
                    "width": IntAttr(w),
                    "base": IntAttr(base),
                    "reason": StringAttr(reason),
                }
            )
            kernel.body.block.insert_op_before(channel, stream)
            stream.body.block.insert_op_before(
                sync_op(g, "wait", name, b.pipe, lo + d, hi + d, d, stride), task_ops[dst]
            )
            stream.body.block.insert_op_after(
                sync_op(g, "set", name, a.pipe, lo, hi, stride=stride), task_ops[src]
            )
        return banks

    # Allocate provisional banks independently of final hardware IDs to test coverage.
    # Width=1 can initially fail reuse, so coverage/cycle checking precedes reuse checking.
    def covered(candidate):
        materialize(candidate, [1] * len(candidate), virtual=True)
        # Temporarily give each logical channel a private direction namespace by
        # checking only the completion graph. Final physical reuse is checked later.
        try:
            ex = execution(result, _program=p)
            graph, begin, end, _, _ = completion_graph(ex)
            return all(graph.precedes(end[o.source], begin[o.destination]) for o in obs)
        except CompileError:
            return False

    # Large provisional banks are unnecessary: remove redundant channels first.
    require(
        covered(proposals),
        "fixed schedule cannot cover its physical dependencies within local banks",
    )
    for proposal in list(reversed(proposals)):
        candidate = proposals.copy()
        candidate.remove(proposal)
        if covered(candidate):
            proposals = candidate
    # Restrict a barrier only when a real destination completion path covers
    # the removed epochs. Source notifications alone cannot provide that path.
    for proposal in list(proposals):
        src, dst, d, lo, hi, stride, reason = proposal
        if p.tasks[src].pipe != p.tasks[dst].pipe:
            continue
        candidate = proposals.copy()
        candidate.remove(proposal)
        materialize(candidate, [1] * len(candidate), virtual=True)
        ex = execution(result, _program=p)
        graph, begin, end, _, _ = completion_graph(ex)
        missing = sorted(
            {
                o.source.generation
                for o in obs
                if (
                    o.source.task,
                    o.destination.task,
                    o.destination.generation - o.source.generation,
                )
                == (src, dst, d)
                and not graph.precedes(end[o.source], begin[o.destination])
            }
        )
        if missing:
            try:
                span = arithmetic_span(missing)
            except CompileError:
                continue
            narrowed = (src, dst, d, *span, reason)
            if covered(candidate + [narrowed]):
                proposals = candidate + [narrowed]
    # Find width by actual old-Wait->next-Set paths, never by storage count.
    widths = [1] * len(proposals)
    needs_lifetime_packing = False
    for i, (src, dst, d, lo, hi, stride, reason) in enumerate(proposals):
        if p.tasks[src].pipe == p.tasks[dst].pipe:
            continue
        for width in range(1, min(8, len(range(lo, hi, stride))) + 1):
            widths[i] = width
            materialize(proposals, widths, virtual=True)
            try:
                ex = execution(result, _program=p)
                graph, _, _, waits, physical_sets = completion_graph(ex)
            except CompileError:
                continue
            name = f"c{i}_{p.tasks[src].uid}_{p.tasks[dst].uid}_d{d}"
            pairs = [[item for item in seq if item[1] == name] for seq in physical_sets.values()]
            if all(
                graph.precedes(waits[(a[1], a[2])], b[0])
                for seq in pairs
                for a, b in itertools.pairwise(seq)
            ):
                break
        else:
            # A finite nonperiodic ID assignment can fit even when this
            # channel cannot use a rotating ring. Do not change its commands.
            widths[i] = 1
            needs_lifetime_packing = True
    banks = defaultdict(int)
    for i, (src, dst, *_) in enumerate(proposals):
        a, b = p.tasks[src].pipe, p.tasks[dst].pipe
        if a != b:
            banks[(a, b)] += widths[i]
    needs_lifetime_packing |= any(size > 8 for size in banks.values())
    materialize(proposals, widths, virtual=needs_lifetime_packing)
    if needs_lifetime_packing:
        from .events import pack_event_lifetimes

        pack_event_lifetimes(result)
    verify_plan(result, _imports=_imports)
    return result

"""Finite single-AIV oracle for actual retained parent/child instructions.

Only physical IR and elementary scalar arithmetic are interpreted here. No
production completion graph, memory summary, event allocator, or proof is an
input. Small runtime counts exercise boundaries; this is not a parametric
compiler proof. A child return creates no asynchronous completion edge.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

from xdsl.dialects import arith

from xdsl_npu.ir import (
    BufferOp,
    CallOp,
    CaptureOp,
    CoreIdOp,
    EndOp,
    KernelOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    RuntimeSizeOp,
    ScopeChannelOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    ViewOp,
)

MODES = {
    "copy": ("R", "W"),
    "copy_rows": ("R", "W"),
    "fill": ("W",),
    "muls": ("R", "W"),
    "adds": ("R", "W"),
    "exp": ("R", "W"),
    "add": ("R", "R", "W"),
    "mul": ("R", "R", "W"),
    "sub": ("R", "R", "W"),
    "div": ("R", "R", "W"),
    "max": ("R", "R", "W"),
    "scalar_load": ("R",),
}


@dataclass(frozen=True)
class Event:
    action: str
    resource: tuple[str, str, int]
    token: tuple
    parent: int
    call: str
    child: int


@dataclass(frozen=True)
class FiniteResult:
    issued_tasks: tuple[tuple[int, str, int, str], ...]
    events: tuple[Event, ...]
    command_count: int
    bytes_read: int
    bytes_written: int


def _integer(attribute, default):
    return default if attribute is None else attribute.data


def _scalar(value, environment, parameters, group):
    if value in environment:
        return environment[value]
    op = value.owner
    if isinstance(op, RuntimeSizeOp):
        result = parameters[op.sym_name.data]
        assert result >= op.minimum.data and result % op.multiple.data == 0
        return result
    if isinstance(op, CoreIdOp):
        return group
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    functions = {
        arith.AddiOp: lambda a, b: a + b,
        arith.SubiOp: lambda a, b: a - b,
        arith.MuliOp: lambda a, b: a * b,
        arith.DivUIOp: lambda a, b: a // b,
        arith.RemUIOp: lambda a, b: a % b,
        arith.CeilDivUIOp: lambda a, b: (a + b - 1) // b,
        arith.MinSIOp: min,
        arith.MaxSIOp: max,
    }
    assert type(op) in functions, f"unsupported actual scalar: {op}"
    a, b = (_scalar(v, environment, parameters, group) for v in (op.lhs, op.rhs))
    if isinstance(op, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
        assert a >= 0 and b > 0, "invalid unsigned address arithmetic"
    return functions[type(op)](a, b)


def pipe(task):
    native = task.native.data
    assert native in MODES, f"unsupported actual native: {native}"
    if native in {"copy", "copy_rows"}:
        banks = tuple(ref.type.bank.data for ref in task.refs)
        assert banks in {("GM", "UB"), ("UB", "GM")}, banks
        return "MTE2" if banks[0] == "GM" else "MTE3"
    return "S" if native == "scalar_load" else "V"


def _extent(root, environment, parameters, group):
    return (
        root.nbytes.data
        if root.extent_count is None
        else _scalar(root.extent_count, environment, parameters, group) * root.count_scale.data
    )


def _captures(call, environment, parameters, group):
    result = {}
    for op in call.body.block.ops:
        if not isinstance(op, CaptureOp):
            continue
        root = op.source.owner
        assert isinstance(root, BufferOp), "capture must retain an actual allocation"
        offset = _scalar(op.offset, environment, parameters, group)
        assert 0 <= offset and offset + op.nbytes.data <= _extent(
            root, environment, parameters, group
        ), "actual capture out of bounds"
        result[op.ref] = root, offset, op.nbytes.data
    return result


def _cells(value, environment, captures, count, parameters, group):
    view = value.owner
    assert isinstance(view, ViewOp), "native access must be an actual ViewOp"
    assert not _integer(view.prefix_elements, 0), "prefix clipping is outside this oracle"
    root = view.source.owner
    if isinstance(root, BufferOp):
        base, span = 0, _extent(root, environment, parameters, group)
    else:
        root, base, span = captures[view.source]
    generation = _scalar(view.generation, environment, parameters, group)
    divisor = _integer(view.divisor, 1)
    assert divisor > 0
    coordinate = generation // divisor
    if view.modulus.data:
        coordinate %= view.modulus.data
    offset = view.offset.data + view.stride.data * coordinate
    for term in view.index_terms or ():
        coefficient, divisor, modulus = (v.data for v in term)
        assert divisor > 0 and modulus >= 0
        offset += coefficient * (
            generation // divisor % modulus if modulus else generation // divisor
        )
    divisor = _integer(view.core_divisor, 1)
    assert divisor > 0
    core = group // divisor
    modulus = _integer(view.core_modulus, 0)
    offset += _integer(view.core_stride, 0) * (core % modulus if modulus else core)
    for term in view.core_terms or ():
        coefficient, divisor, modulus = (v.data for v in term)
        assert divisor > 0 and modulus >= 0
        offset += coefficient * (group // divisor % modulus if modulus else group // divisor)
    extent_count = (
        count
        if view.extent_count is None
        else _scalar(view.extent_count, environment, parameters, group)
    )
    offset += _integer(view.count_offset, 0) * extent_count
    offset += _integer(view.count_core_stride, 0) * extent_count * group
    width = (
        view.tail_nbytes.data
        if view.tail_nbytes.data and generation == count - 1
        else view.nbytes.data
    )
    extent = _extent(root, environment, parameters, group)
    for row in range(_integer(view.rows, 1)):
        start = offset + row * _integer(view.row_stride, 0)
        for byte in range(start, start + width):
            assert 0 <= byte < span, "actual native byte outside capture"
            assert 0 <= base + byte < extent, "actual native byte out of bounds"
            yield root, base + byte


def _active(task, generation, count):
    upper = _integer(task.upper, -1)
    period, phase = _integer(task.period, 1), _integer(task.phase, 0)
    return _integer(task.lower, 0) <= generation < (
        count if upper == -1 else upper
    ) and phase <= generation % period < phase + _integer(task.phase_width, 1)


def _event_id(channel, epoch):
    period, residues = channel.period.data, tuple(v.data for v in channel.residues)
    assert epoch >= 0 and epoch % period in residues, "notification has no source epoch"
    ordinal = epoch // period * len(residues) + sum(r <= epoch % period for r in residues) - 1
    ids = tuple(v.data for v in channel.event_ids)
    assert ids and all(0 <= event < 8 for event in ids), "invalid physical event ID"
    return ids[ordinal % len(ids)]


def _templates(child):
    tasks = [op for op in child.body.block.ops if isinstance(op, TaskOp)]
    assert tasks, "child has no native instructions"
    for task in tasks:
        pipe(task)
    before, after, pending, previous = {}, defaultdict(list), [], None
    for op in child.body.block.ops:
        if isinstance(op, TaskOp):
            before[op], pending, previous = pending, [], op
        elif isinstance(op, (LoopSyncOp, ScopeSyncOp)):
            if op.action.data == "set":
                assert previous is not None, "Set has no actual native predecessor"
                after[previous].append(op)
            else:
                assert op.action.data in {"wait", "barrier"}
                pending.append(op)
        else:
            assert isinstance(op, (ViewOp, LoopChannelOp, EndOp)) or op.name.startswith("arith."), (
                f"unsupported actual child instruction: {op.name}"
            )
    assert not pending, "Wait has no actual native successor"
    return tasks, before, after


def finite_check(plan, n, *, group=0, parameters=None):
    """Check actual issue, binary-event consumption, and every byte dependency."""
    parameters = {"n": n} | ({} if parameters is None else parameters)
    kernels = [op for op in plan.walk() if isinstance(op, KernelOp)]
    assert len(kernels) == 1 and kernels[0].core.data == "aiv", "single-AIV oracle only"
    kernel = kernels[0]
    assert 0 <= group < _integer(kernel.blocks, 1)
    parents = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
    assert len(parents) == 1, "one retained parent stream is required"
    parent = parents[0]
    count = (
        _scalar(parent.limit, {}, parameters, group)
        if parent.limit is not None
        else parent.count.data
    )
    assert count >= 0
    calls = [op for op in parent.body.block.ops if isinstance(op, CallOp)]
    assert calls and len({op.uid.data for op in calls}) == len(calls)
    lexical_calls = {call: index for index, call in enumerate(calls)}
    scope_channels = {op.sym_name.data: op for op in plan.walk() if isinstance(op, ScopeChannelOp)}
    children, templates, channels, tables = {}, {}, {}, {}
    for call in calls:
        assert call.generation is parent.body.block.args[0], "call uses a different parent clock"
        regions = [op for op in call.body.block.ops if isinstance(op, (StreamOp, OnceOp))]
        assert len(regions) == 1
        children[call] = child = regions[0]
        templates[call] = _templates(child)
        tables[call.uid.data] = {task.uid.data: task for task in templates[call][0]}
        assert len(tables[call.uid.data]) == len(templates[call][0])
        channels[call] = {
            op.sym_name.data: op for op in child.body.block.ops if isinstance(op, LoopChannelOp)
        }

    def scope_endpoint(encoded):
        actor, uid, task, position = encoded.data.split("/")
        assert actor == "aiv0", "single-AIV endpoint required"
        return uid, tables[uid][task], int(position)

    def sync_info(op, call, task, pg, g, child_count, environment):
        action = op.action.data
        if isinstance(op, LoopSyncOp):
            assert op.generation is children[call].body.block.args[0]
            if (
                not op.lower.data <= g < child_count - op.upper_gap.data
                or g % op.period.data not in tuple(v.data for v in op.residues)
                or op.last_gap is not None
                and g != child_count - 1 - op.last_gap.data
            ):
                return None
            if action == "barrier":
                # asc_sync_pipe has no event channel; normal physical IR uses
                # an empty channel name and no LoopChannelOp declaration.
                return action, None, None
            channel = channels[call][op.channel.data]
            assert not _integer(channel.last, 0) and not _integer(channel.carry, 0)
            assert channel.kind.data == "local" and action in {"set", "wait"}
            source, destination = (
                tables[call.uid.data][values.data[0].data.split("/")[1]]
                for values in (channel.sources, channel.destinations)
            )
            assert task is (source if action == "set" else destination)
            epoch = g if action == "set" else g - channel.distance.data
            token = "child", call.uid.data, pg, channel.sym_name.data, epoch
        else:
            assert op.parent_generation is call.generation
            assert op.child_generation is children[call].body.block.args[0]
            assert _scalar(op.parent_generation, environment, parameters, group) == pg
            if g != op.position.data:
                return None
            channel = scope_channels[op.channel.data]
            distance = channel.distance.data
            if (action == "set" and pg >= count - distance) or (action != "set" and pg < distance):
                return None
            epoch = pg - distance if action == "wait" else pg
            if epoch % channel.period.data not in tuple(v.data for v in channel.residues):
                return None
            endpoints = channel.sources if action == "set" else channel.destinations
            assert any(
                (uid, native, position) == (call.uid.data, task, g)
                for uid, native, position in map(scope_endpoint, endpoints)
            ), "scope synchronization is not beside its actual endpoint"
            if channel.kind.data == "barrier":
                assert action == "barrier"
                return action, None, None
            assert channel.kind.data == "local" and action in {"set", "wait"}
            source, destination = (
                scope_endpoint(values.data[0])[1]
                for values in (channel.sources, channel.destinations)
            )
            token = "parent", channel.sym_name.data, epoch
        assert pipe(source) != pipe(destination)
        return action, (pipe(source), pipe(destination), _event_id(channel, epoch)), token

    edges = []

    def node():
        edges.append(set())
        return len(edges) - 1

    def edge(a, b):
        if a is not None and a != b:
            edges[a].add(b)

    queue, prefixes, begins, ends, accesses = {}, {}, {}, {}, {}
    sends, waits = defaultdict(list), defaultdict(list)
    dispatch, gate, command_count = None, None, 0
    instances, event_trace, issued = [], [], []
    maximum = max(call.stage.data for call in calls)
    for parent_step in range(count + maximum):
        for call in sorted(calls, key=lambda c: (c.issue_order.data, lexical_calls[c])):
            pg = parent_step - call.stage.data
            if not 0 <= pg < count:
                continue
            child = children[call]
            call_environment = {parent.body.block.args[0]: pg}
            captures = _captures(call, call_environment, parameters, group)
            child_count = (
                _scalar(child.limit, call_environment, parameters, group)
                if child.limit is not None
                else child.count.data
            )
            tasks, before, after = templates[call]
            lexical_tasks = {task: index for index, task in enumerate(tasks)}
            for step in range(child_count + max(task.stage.data for task in tasks)):
                for task in sorted(
                    tasks, key=lambda t: (_integer(t.issue_order, 0), lexical_tasks[t])
                ):
                    g = step - task.stage.data
                    if not 0 <= g < child_count or not _active(task, g, child_count):
                        continue
                    environment = call_environment | {child.body.block.args[0]: g}
                    key = pg, call, g, task
                    for op in (*before[task], None, *after[task]):
                        info = (
                            ("task", None, None)
                            if op is None
                            else sync_info(op, call, task, pg, g, child_count, environment)
                        )
                        if info is None:
                            continue
                        kind, resource, token = info
                        command_count += 1
                        issue, begin = node(), node()
                        end = node() if kind in {"task", "set"} else begin
                        edge(dispatch, issue)
                        edge(gate, issue)
                        edge(issue, begin)
                        edge(begin, end)
                        native_pipe = pipe(task)
                        edge(queue.get(native_pipe), begin)
                        queue[native_pipe] = begin
                        if kind == "task":
                            prefix = node()
                            edge(prefixes.get(native_pipe), prefix)
                            edge(end, prefix)
                            prefixes[native_pipe] = prefix
                            begins[key], ends[key] = begin, end
                            modes = MODES[task.native.data]
                            touched = defaultdict(set)
                            for value, mode in zip(task.refs, modes, strict=True):
                                for byte in _cells(
                                    value, environment, captures, child_count, parameters, group
                                ):
                                    touched[byte].update(mode)
                            accesses[key] = touched
                            instances.append(
                                (
                                    pg,
                                    call.logical_phase.data,
                                    lexical_calls[call],
                                    g,
                                    _integer(task.logical_phase, 0),
                                    lexical_tasks[task],
                                    key,
                                )
                            )
                            issued.append((pg, call.uid.data, g, task.uid.data))
                        elif kind in {"set", "barrier"}:
                            edge(prefixes.get(native_pipe), end)
                        if native_pipe == "S" and kind in {"task", "wait"}:
                            gate = end
                        if kind in {"set", "wait"}:
                            (sends if kind == "set" else waits)[resource].append((token, end))
                            event_trace.append(Event(kind, resource, token, pg, call.uid.data, g))
                        dispatch = issue

    assert sends.keys() == waits.keys(), "unmatched physical Set/Wait bank"
    for resource, sent in sends.items():
        received = waits[resource]
        assert len(sent) == len(received), "unmatched physical Set/Wait count"
        for (sent_token, send), (expected_token, wait) in zip(sent, received, strict=True):
            assert sent_token == expected_token, (
                "physical event paired a different logical notification",
                resource,
                sent_token,
                expected_token,
            )
            edge(send, wait)
    degree = [0] * len(edges)
    for following in edges:
        for target in following:
            degree[target] += 1
    pending = deque(index for index, incoming in enumerate(degree) if not incoming)
    order = []
    while pending:
        source = pending.popleft()
        order.append(source)
        for target in edges[source]:
            degree[target] -= 1
            if degree[target] == 0:
                pending.append(target)
    assert len(order) == len(edges), "actual command deadlock"
    reach = [0] * len(edges)
    for source in reversed(order):
        for target in edges[source]:
            reach[source] |= (1 << target) | reach[target]
    for resource, sent in sends.items():
        for (_, wait), (_, send) in zip(waits[resource], sent[1:]):
            assert reach[wait] & (1 << send), ("binary event overwrite", resource)

    writers, readers = {}, defaultdict(set)
    bytes_read = bytes_written = 0
    for *_, key in sorted(instances, key=lambda instance: instance[:-1]):
        for byte, modes in accesses[key].items():
            hazards = []
            if "R" in modes:
                bytes_read += 1
                assert byte in writers or byte[0].initialized.data, "undefined actual read"
                if byte in writers:
                    hazards.append(("RAW", writers[byte]))
            if "W" in modes:
                bytes_written += 1
                if byte in writers:
                    hazards.append(("WAW", writers[byte]))
                hazards.extend(("WAR", old) for old in readers[byte])
            for kind, old in hazards:
                if old != key:
                    assert reach[ends[old]] & (1 << begins[key]), (
                        "unordered " + kind,
                        byte[0].sym_name.data,
                        byte[1],
                        old,
                        key,
                    )
            if "R" in modes:
                readers[byte].add(key)
            if "W" in modes:
                writers[byte], readers[byte] = key, set()
    return FiniteResult(tuple(issued), tuple(event_trace), command_count, bytes_read, bytes_written)

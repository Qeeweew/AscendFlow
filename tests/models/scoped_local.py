"""Finite single-actor oracle for actual retained parent/child instructions.

Only physical IR and elementary scalar arithmetic are interpreted here. No
production completion graph, memory summary, event allocator, or proof is an
input. Small runtime counts exercise boundaries; this is not a parametric
compiler proof. A child return creates no asynchronous completion edge.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise

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
    ScopeGuardOp,
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
    "brcb8": ("R", "W"),
    "mul_row_broadcast": ("R", "R", "W"),
    "convert_rn": ("R", "W"),
    "nd2nz": ("R", "W"),
    "load2d_a": ("R", "W"),
    "load2d_b_nt": ("R", "W"),
    "mad": ("R", "R", "W"),
    "mad_first": ("R", "R", "W"),
    "fix": ("R", "W"),
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
    cube = {
        "nd2nz": ("MTE2", ("GM", "L1")),
        "load2d_a": ("MTE1", ("L1", "L0A")),
        "load2d_b_nt": ("MTE1", ("L1", "L0B")),
        "mad": ("M", ("L0A", "L0B", "L0C")),
        "mad_first": ("M", ("L0A", "L0B", "L0C")),
        "fix": ("FIX", ("L0C", "GM")),
    }
    if native in cube:
        native_pipe, banks = cube[native]
        assert tuple(ref.type.bank.data for ref in task.refs) == banks
        dimensions = tuple(value.data for value in task.params)
        assert dimensions[:2] == (16, 16), "Cube oracle supports whole 16x16 fractals"
        if native in {"mad", "mad_first"}:
            assert dimensions[:3] == (16, 16, 16), "Cube oracle supports short 16x16 MAD"
        return native_pipe
    return "S" if native == "scalar_load" else "V"


def _modes(task, generation):
    if task.native.data == "mad_first":
        return "R", "R", "W" if generation == 0 else "RW"
    if task.native.data == "mad":
        assert len(task.params) == 4 and task.params.data[3].data in (0, 1)
        return "R", "R", "W" if task.params.data[3].data else "RW"
    return MODES[task.native.data]


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


def _spans(value, environment, captures, count, parameters, group):
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
        assert 0 <= start < start + width <= span, "actual native byte outside capture"
        assert 0 <= base + start < base + start + width <= extent, (
            "actual native byte out of bounds"
        )
        yield root, base + start, width


def _cells(value, environment, captures, count, parameters, group):
    for root, start, width in _spans(value, environment, captures, count, parameters, group):
        for byte in range(start, start + width):
            yield root, byte


def _atomize(accesses):
    """Exact finite row partition: bytes in an atom have identical access histories."""
    boundaries = defaultdict(set)
    for spans in accesses.values():
        for root, start, width, _ in spans:
            boundaries[root].update((start, start + width))
    atoms = {root: tuple(pairwise(sorted(points))) for root, points in boundaries.items()}
    result = {}
    for key, spans in accesses.items():
        touched = defaultdict(set)
        for root, start, width, mode in spans:
            for first, last in atoms[root]:
                if start <= first and last <= start + width:
                    touched[root, first, last].update(mode)
        result[key] = touched
    return result


def _scope_epoch(channel, action, generation, count):
    distance = channel.distance.data
    activation = channel.activation
    if activation is None:
        assert distance >= 0, "signed distance needs an actual activation"
        if (action == "set" and generation >= count - distance) or (
            action != "set" and generation < distance
        ):
            return None
        epoch = generation - distance if action == "wait" else generation
    else:
        fields = {name: value.data for name, value in activation.data.items()}
        required = {
            "count_period",
            "count_residue",
            "first_end",
            "first_offset",
            "last_end",
            "last_offset",
        }
        assert (
            required
            <= fields.keys()
            <= required | {"count_exact", "count_minimum", "count_maximum"}
        )
        period, residue = fields["count_period"], fields["count_residue"]
        assert period > 0 and 0 <= residue < period
        if "count_exact" in fields:
            if count != fields["count_exact"]:
                return None
        elif (
            count < fields.get("count_minimum", period)
            or count % period != residue
            or "count_maximum" in fields
            and count > fields["count_maximum"]
        ):
            return None
        epoch = generation if action == "set" else generation - distance
        first = (
            count - 1 - fields["first_offset"] if fields["first_end"] else fields["first_offset"]
        )
        last = count - 1 - fields["last_offset"] if fields["last_end"] else fields["last_offset"]
        if not first <= epoch <= last:
            return None
        assert 0 <= epoch < count and 0 <= epoch + distance < count, "scope epoch out of bounds"
    if epoch % channel.period.data not in tuple(value.data for value in channel.residues):
        return None
    return epoch


def _physical_versions(logical, physical, accesses, begins, ends, reach):
    """Keep source RAW/RW versions; order independent definitions physically.

    Every adjacent version needs WAW retirement, every read needs its exact
    source writer, and all that version's readers must retire before the next
    version can overwrite it. RW updates remain in their original definition.
    This proves a unique writer at each read without imposing source WAR/WAW
    between independent clears whose physical occupation order was changed.
    """
    writers, owners = {}, {}
    versions = defaultdict(list)
    read_bindings = []
    bytes_read = bytes_written = 0
    for key in logical:
        for cell, modes in accesses[key].items():
            previous = writers.get(cell)
            if "R" in modes:
                bytes_read += cell[2] - cell[1]
                assert previous is not None or cell[0].initialized.data, "undefined actual read"
                read_bindings.append((cell, key, previous))
            if "W" in modes:
                bytes_written += cell[2] - cell[1]
                owner = owners[previous, cell] if "R" in modes and previous is not None else key
                owners[key, cell] = owner
                versions[cell, owner].append(key)
                writers[cell] = key
    definitions = defaultdict(list)
    for key in physical:
        for cell, modes in accesses[key].items():
            if "W" in modes and owners[key, cell] == key:
                definitions[cell].append(key)
    following = {}
    for cell, definitions_ in definitions.items():
        ordered = [key for owner in definitions_ for key in versions[cell, owner]]
        if cell[0].external.data:
            assert ordered[-1] == writers[cell], "final external writer changed"
        following[cell, None] = ordered[0]
        for old, new in pairwise(ordered):
            following[cell, old] = new
            assert reach[ends[old]] & (1 << begins[new]), (
                "unordered physical WAW",
                cell[0].sym_name.data,
                cell[1],
                old,
                new,
            )
    for cell, reader, writer in read_bindings:
        if writer is not None:
            assert reach[ends[writer]] & (1 << begins[reader]), (
                "logical RAW producer unavailable",
                cell[0].sym_name.data,
                cell[1],
                writer,
                reader,
            )
        next_writer = following.get((cell, writer))
        if next_writer is not None and next_writer != reader:
            assert reach[ends[reader]] & (1 << begins[next_writer]), (
                "unordered physical WAR",
                cell[0].sym_name.data,
                cell[1],
                reader,
                next_writer,
            )
    return bytes_read, bytes_written


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


PARENT_CASES = ("single", "first", "middle", "last")


def _cases(op):
    value = op.attributes.get("scope_cases")
    if value is None:
        return PARENT_CASES
    result = tuple(item.data for item in value)
    assert result and result == tuple(case for case in PARENT_CASES if case in result), (
        "invalid actual scope_cases"
    )
    return result


def _parent_case(generation, count):
    assert 0 <= generation < count
    return (
        "single"
        if count == 1
        else "first"
        if generation == 0
        else "last"
        if generation == count - 1
        else "middle"
    )


def _selected(op, contexts, pg, count):
    for guard in contexts[op]:
        if guard.kind.data == "first" and pg != 0:
            return False
        if guard.kind.data == "last" and pg != count - 1:
            return False
    return _parent_case(pg, count) in _cases(op)


def _templates(child, call):
    contexts, operations = {}, []

    def collect(block, guards=()):
        for op in block.ops:
            if isinstance(op, ScopeGuardOp):
                assert op.kind.data in {"first", "last"}, "unknown actual parent guard"
                assert op.parent_generation is call.generation, (
                    "guard uses a different parent clock"
                )
                assert op.child_generation is child.body.block.args[0], (
                    "guard uses a different child clock"
                )
                collect(op.body.block, (*guards, op))
            elif not isinstance(op, EndOp):
                contexts[op] = guards
                operations.append(op)

    collect(child.body.block)
    tasks = [op for op in operations if isinstance(op, TaskOp)]
    assert tasks, "child has no native instructions"
    for task in tasks:
        pipe(task)
    before, after, pending, previous = {}, defaultdict(list), [], None
    for op in operations:
        if isinstance(op, TaskOp):
            before[op], pending, previous = pending, [], op
        elif isinstance(op, (LoopSyncOp, ScopeSyncOp)):
            if isinstance(op, LoopSyncOp):
                assert op.generation is child.body.block.args[0], (
                    "sync uses a different child clock"
                )
                assert (
                    op.parent_generation is call.generation
                    if "scope_cases" in op.attributes
                    else op.parent_generation is None
                ), "sync uses a different parent clock"
                _cases(op)
            else:
                assert op.parent_generation is call.generation, "sync uses a different parent clock"
                assert op.child_generation is child.body.block.args[0], (
                    "sync uses a different child clock"
                )
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
    channels = {op.sym_name.data: op for op in operations if isinstance(op, LoopChannelOp)}
    for op in operations:
        if isinstance(op, LoopSyncOp) and op.action.data != "barrier":
            assert op.channel.data in channels, "sync has no actual child channel"
            assert _cases(op) == _cases(channels[op.channel.data]), "actual scope_cases mismatch"
    return tasks, before, after, contexts, channels


def finite_check(plan, n, *, group=0, parameters=None, _execute=None):
    """Check actual issue, binary-event consumption, and every byte dependency."""
    parameters = {"n": n} | ({} if parameters is None else parameters)
    kernels = [op for op in plan.walk() if isinstance(op, KernelOp)]
    assert len(kernels) == 1 and kernels[0].core.data in {"aiv", "aic"}, "single-actor oracle only"
    kernel = kernels[0]
    cube = kernel.core.data == "aic"
    actor_name = "aic" if cube else "aiv0"
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
        templates[call] = _templates(child, call)
        tables[call.uid.data] = {task.uid.data: task for task in templates[call][0]}
        assert len(tables[call.uid.data]) == len(templates[call][0])
        channels[call] = templates[call][4]

    def scope_endpoint(encoded):
        actor, uid, task, position = encoded.data.split("/")
        assert actor == actor_name, "endpoint belongs to a different actual actor"
        return uid, tables[uid][task], int(position)

    def sync_info(op, call, task, pg, g, child_count, environment):
        action = op.action.data
        if not _selected(op, templates[call][3], pg, count):
            return None
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
            assert _selected(channel, templates[call][3], pg, count), (
                "actual sync executes outside its channel case"
            )
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
            epoch = _scope_epoch(channel, action, pg, count)
            if epoch is None:
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
    instances, event_trace, issued, executions = [], [], [], {}
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
            tasks, before, after, contexts, _ = templates[call]
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
                        if op is None and not _selected(task, contexts, pg, count):
                            continue
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
                            if _execute is not None:
                                executions[end] = task, environment, captures, child_count
                            modes = _modes(task, g)
                            touched = [] if cube else defaultdict(set)
                            for value, mode in zip(task.refs, modes, strict=True):
                                if cube:
                                    touched.extend(
                                        (root, start, width, mode)
                                        for root, start, width in _spans(
                                            value,
                                            environment,
                                            captures,
                                            child_count,
                                            parameters,
                                            group,
                                        )
                                    )
                                else:
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

    logical = [instance[-1] for instance in sorted(instances, key=lambda instance: instance[:-1])]
    if cube:
        accesses = _atomize(accesses)
        bytes_read, bytes_written = _physical_versions(
            logical, list(begins), accesses, begins, ends, reach
        )
        return FiniteResult(
            tuple(issued), tuple(event_trace), command_count, bytes_read, bytes_written
        )
    # Rebuild Vector source order independently of physical issue/Stage order.
    # Inactive first/last natives never create a writer or a reader frontier.
    logical = []
    for pg in range(count):
        for call in sorted(calls, key=lambda op: (op.logical_phase.data, lexical_calls[op])):
            child = children[call]
            environment = {parent.body.block.args[0]: pg}
            child_count = (
                _scalar(child.limit, environment, parameters, group)
                if child.limit is not None
                else child.count.data
            )
            tasks, _, _, contexts, _ = templates[call]
            source_tasks = sorted(
                enumerate(tasks), key=lambda item: (_integer(item[1].logical_phase, 0), item[0])
            )
            for generation in range(child_count):
                for _, task in source_tasks:
                    if _active(task, generation, child_count) and _selected(
                        task, contexts, pg, count
                    ):
                        logical.append((pg, call, generation, task))
    assert set(logical) == set(begins), "physical and logical actual native instances disagree"
    writers, readers = {}, defaultdict(set)
    bytes_read = bytes_written = 0
    for key in logical:
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
    if _execute is not None:
        for end in order:
            if end in executions:
                _execute(*executions[end], parameters, group)
    return FiniteResult(tuple(issued), tuple(event_trace), command_count, bytes_read, bytes_written)


def execute(plan, n, arrays, *, group=0, parameters=None):
    """Execute supported actual Vector natives in a checked physical DAG order.

    Supplied external arrays are copied, including output sentinels. This is a
    numerical test interpreter, not a native instruction bit-level simulator.
    """
    import numpy as np

    assert all(op.core.data == "aiv" for op in plan.walk() if isinstance(op, KernelOp))
    parameters = {"n": n} | ({} if parameters is None else parameters)
    roots = [op for op in plan.walk() if isinstance(op, BufferOp)]
    memory = {}
    for root in roots:
        name = root.sym_name.data
        extent = _extent(root, {}, parameters, group)
        if name in arrays:
            data = np.ascontiguousarray(arrays[name]).view(np.uint8).reshape(-1).copy()
            assert data.nbytes == extent, "external value has the wrong actual allocation size"
            memory[root] = data
        else:
            assert not root.external.data, "missing actual external array"
            memory[root] = np.zeros(extent, dtype=np.uint8)

    def values(value, environment, captures, count, parameters, group):
        dtype = {"f16": np.float16, "f32": np.float32}[value.type.dtype.data]
        return np.concatenate(
            [
                memory[root][start : start + width].view(dtype)
                for root, start, width in _spans(
                    value, environment, captures, count, parameters, group
                )
            ]
        )

    def run(task, environment, captures, count, parameters, group):
        native = task.native.data
        assert native in {
            "copy",
            "copy_rows",
            "fill",
            "add",
            "brcb8",
            "mul_row_broadcast",
            "convert_rn",
        }, "unsupported numerical native: " + native
        inputs = [
            values(value, environment, captures, count, parameters, group)
            for value in task.refs[:-1]
        ]
        dtype = {"f16": np.float16, "f32": np.float32}[task.refs[-1].type.dtype.data]
        if native in {"copy", "copy_rows", "convert_rn"}:
            result = inputs[0].astype(dtype)
        elif native == "fill":
            width = sum(
                width
                for _, _, width in _spans(
                    task.refs[-1], environment, captures, count, parameters, group
                )
            )
            result = np.full(width // np.dtype(dtype).itemsize, task.scalar.value.data, dtype=dtype)
        elif native == "add":
            result = np.add(inputs[0], inputs[1], dtype=np.float32)
        elif native == "brcb8":
            rows = task.params.data[0].data
            assert inputs[0].size == rows
            result = np.repeat(inputs[0], 8)
        else:
            rows, columns = (value.data for value in task.params)
            factors = inputs[1].reshape(rows, 8)
            assert np.all(factors == factors[:, :1]), "BRCB scratch has inconsistent row words"
            result = np.multiply(
                inputs[0].reshape(rows, columns), factors[:, :1], dtype=np.float32
            ).reshape(-1)
        raw, offset = np.ascontiguousarray(result).view(np.uint8).reshape(-1), 0
        for root, start, width in _spans(
            task.refs[-1], environment, captures, count, parameters, group
        ):
            memory[root][start : start + width] = raw[offset : offset + width]
            offset += width
        assert offset == raw.size, "native numerical result has the wrong actual footprint"

    checked = finite_check(plan, n, group=group, parameters=parameters, _execute=run)
    outputs = {
        root.sym_name.data: memory[root]
        .view(arrays[root.sym_name.data].dtype)
        .reshape(arrays[root.sym_name.data].shape)
        .copy()
        for root in roots
        if root.sym_name.data in arrays
    }
    return checked, outputs

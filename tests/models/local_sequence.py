"""Bounded actual-command/event DAG and byte oracle for serial local regions.

This interpreter consumes emitted IR, never production completion queries or
memory summaries. Enumerating small counts here is a test oracle, not compiler
loop expansion or a proof for arbitrary runtime lengths.
"""

from collections import defaultdict, deque
from itertools import product

from xdsl.dialects import arith, scf

from xdsl_npu.ir import (
    BoundaryOp,
    BufferOp,
    FrameOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    RegionChannelOp,
    RegionSyncOp,
    RuntimeSizeOp,
    StreamOp,
    TaskOp,
    ViewOp,
)


def physical_regions(plan):
    kernel = plan.body.block.first_op
    parent = next((op for op in kernel.body.block.ops if isinstance(op, scf.ForOp)), kernel)
    return [op for op in parent.body.block.ops if isinstance(op, (OnceOp, StreamOp))]


def operations(region):
    for op in region.body.block.ops:
        if isinstance(op, BoundaryOp):
            yield from op.body.block.ops
        else:
            yield op


def pipe(task):
    if task.native.data in {"copy", "copy_rows"}:
        return "MTE2" if task.refs[0].type.bank.data == "GM" else "MTE3"
    return "S" if task.native.data == "scalar_load" else "V"


def scalar(value, n, calls):
    op = value.owner
    if isinstance(op, RuntimeSizeOp):
        return calls if op.sym_name.data == "outputs" else n
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    assert isinstance(op, arith.MuliOp)
    return scalar(op.lhs, n, calls) * scalar(op.rhs, n, calls)


def cells(value, g, n, *, call=0, calls=1, group=0):
    view = value.owner
    assert isinstance(view, ViewOp)
    root = view.source.owner
    frame = root if isinstance(root, FrameOp) else None
    if frame is not None:
        root = frame.source.owner
    assert isinstance(root, BufferOp)
    address = view.offset.data + view.stride.data * (
        g % view.modulus.data if view.modulus.data else g
    )
    for term in view.index_terms or ():
        coefficient, divisor, modulus = [v.data for v in term]
        address += coefficient * ((g // divisor) % modulus if modulus else g // divisor)
    core = group // view.core_divisor.data
    address += view.core_stride.data * (
        core % view.core_modulus.data if view.core_modulus.data else core
    )
    for term in view.core_terms or ():
        coefficient, divisor, modulus = [v.data for v in term]
        address += coefficient * ((group // divisor) % modulus if modulus else group // divisor)
    address += (0 if view.count_offset is None else view.count_offset.data) * n
    address += (0 if view.count_core_stride is None else view.count_core_stride.data) * n * group
    if frame is not None:
        address += (group * calls + call) * scalar(frame.extent, n, calls) * frame.scale.data
    extent = (
        root.nbytes.data
        if root.extent_count is None
        else scalar(root.extent_count, n, calls) * root.count_scale.data
    )
    for row in range(1 if view.rows is None else view.rows.data):
        start = address + row * (0 if view.row_stride is None else view.row_stride.data)
        for byte in range(start, start + view.nbytes.data):
            assert 0 <= byte < extent, f"{root.sym_name.data}: actual native byte out of bounds"
            yield root, byte


def finite_check(plan, n, *, invocations=1, group=0):
    regions = physical_regions(plan)
    boundary = {o.sym_name.data: o for o in plan.walk() if isinstance(o, RegionChannelOp)}
    ids, edges = {}, []
    begins, ends = {}, {}
    queue, prefixes = {}, {}
    dispatch, gate = None, None
    sets, waits = defaultdict(list), defaultdict(list)
    serial = 0

    def node(key):
        if key not in ids:
            ids[key] = len(edges)
            edges.append(set())
        return ids[key]

    def edge(a, b):
        if a is not None and a != b:
            edges[a].add(b)

    counts, native = {}, {}
    for call, region in product(range(invocations), regions):
        count = (
            1
            if isinstance(region, OnceOp)
            else n
            if region.limit is not None
            else region.count.data
        )
        counts[region] = count
        tasks = [op for op in operations(region) if isinstance(op, TaskOp)]
        native[region] = tasks
        channels = {
            op.sym_name.data: op for op in region.body.block.ops if isinstance(op, LoopChannelOp)
        }
        table = {t.uid.data: t for t in tasks}
        before, after, pending, prior = {}, defaultdict(list), [], None
        for op in operations(region):
            if isinstance(op, TaskOp):
                before[op], pending, prior = pending, [], op
            elif isinstance(op, (LoopSyncOp, RegionSyncOp)):
                if op.action.data == "set":
                    after[prior].append(op)
                else:
                    pending.append(op)
        assert not pending
        for step in range(count + max(t.stage.data for t in tasks)):
            for task in sorted(tasks, key=lambda t: (t.issue_order.data, tasks.index(t))):
                g = step - task.stage.data
                if (
                    not 0 <= g < count
                    or isinstance(task.parent_op(), BoundaryOp)
                    and g != count - 1
                ):
                    continue
                for op in [*before[task], None, *after[task]]:
                    resource = None
                    kind = "task" if op is None else op.action.data
                    if isinstance(op, LoopSyncOp):
                        if (
                            not op.lower.data <= g < count - op.upper_gap.data
                            or g % op.period.data not in [r.data for r in op.residues]
                        ):
                            continue
                        if op.last_gap is not None and g != count - 1 - op.last_gap.data:
                            continue
                        if kind in {"set", "wait"}:
                            c = channels[op.channel.data]
                            epoch = g if kind == "set" else g - c.distance.data
                            residues = [r.data for r in c.residues]
                            ordinal = (
                                epoch // c.period.data * len(residues)
                                + sum(epoch % c.period.data >= r for r in residues)
                                - 1
                            )
                            event_id = c.event_ids.data[ordinal % len(c.event_ids)].data
                            src = table[c.sources.data[0].data.split("/")[1]]
                            dst = table[c.destinations.data[0].data.split("/")[1]]
                            resource = pipe(src), pipe(dst), event_id
                    elif isinstance(op, RegionSyncOp):
                        c = boundary[op.channel.data]
                        carried = c.carried is not None and c.carried.data == 1
                        if carried and (
                            (kind == "set" and call + 1 == invocations)
                            or (kind != "set" and call == 0)
                        ):
                            continue
                        active = False
                        for row in c.domains:
                            p, r, exact, minimum, *upper = [v.data for v in row]
                            active |= (
                                n == exact
                                if exact >= 0
                                else (n >= minimum and n % p == r and (not upper or n <= upper[0]))
                            )
                        end = c.source_end.data if kind == "set" else c.destination_end.data
                        offset = (
                            c.source_offset.data if kind == "set" else c.destination_offset.data
                        )
                        if not active or g != (count - 1 - offset if end else offset):
                            continue
                        src_region = next(r for r in regions if r.uid.data == c.source_region.data)
                        dst_region = next(
                            r for r in regions if r.uid.data == c.destination_region.data
                        )
                        src = next(
                            t
                            for t in operations(src_region)
                            if isinstance(t, TaskOp) and t.uid == c.source_task
                        )
                        dst = next(
                            t
                            for t in operations(dst_region)
                            if isinstance(t, TaskOp) and t.uid == c.destination_task
                        )
                        if kind != "barrier":
                            resource = pipe(src), pipe(dst), c.event_id.data
                    serial += 1
                    issue = node((serial, "issue"))
                    begin = node((serial, "begin"))
                    end = node((serial, "end")) if kind in {"task", "set"} else begin
                    edge(dispatch, issue)
                    edge(gate, issue)
                    edge(issue, begin)
                    edge(begin, end)
                    p = pipe(task)
                    edge(queue.get(p), begin)
                    queue[p] = begin
                    if kind == "task":
                        prefix = node((serial, "prefix"))
                        edge(prefixes.get(p), prefix)
                        edge(end, prefix)
                        prefixes[p] = prefix
                        begins[call, region, g, task] = begin
                        ends[call, region, g, task] = end
                    elif kind in {"set", "barrier"}:
                        edge(prefixes.get(p), end)
                    if p == "S" and kind in {"task", "wait"}:
                        gate = end
                    if kind == "set":
                        sets[resource].append(end)
                    elif kind == "wait":
                        waits[resource].append(begin)
                    dispatch = issue
    assert sets.keys() == waits.keys()
    for key in sets:
        assert len(sets[key]) == len(waits[key])
        for send, wait in zip(sets[key], waits[key]):
            edge(send, wait)
    degree = [0] * len(edges)
    for successors in edges:
        for b in successors:
            degree[b] += 1
    work = deque(i for i, d in enumerate(degree) if not d)
    order = []
    while work:
        a = work.popleft()
        order.append(a)
        for b in edges[a]:
            degree[b] -= 1
            if not degree[b]:
                work.append(b)
    assert len(order) == len(edges), "actual command deadlock"
    reach = [0] * len(edges)
    for a in reversed(order):
        for b in edges[a]:
            reach[a] |= (1 << b) | reach[b]
    for key in sets:
        for wait, send in zip(waits[key], sets[key][1:]):
            assert reach[wait] & (1 << send), "binary event overwrite"
    writers, readers = {}, defaultdict(set)
    for call, region in product(range(invocations), regions):
        for g in range(counts[region]):
            for task in native[region]:
                if isinstance(task.parent_op(), BoundaryOp) and g != counts[region] - 1:
                    continue
                modes = {
                    "copy": ("R", "W"),
                    "copy_rows": ("R", "W"),
                    "add": ("R", "R", "W"),
                    "fill": ("W",),
                    "muls": ("R", "W"),
                    "scalar_load": ("R",),
                }[task.native.data]
                touched = {}
                for value, mode in zip(task.refs, modes):
                    for byte in cells(
                        value, g, counts[region], call=call, calls=invocations, group=group
                    ):
                        touched.setdefault(byte, set()).update(mode)
                key = call, region, g, task
                for byte, mode in touched.items():
                    hazards = set()
                    if "R" in mode:
                        assert byte in writers or byte[0].initialized.data, "undefined actual read"
                        if byte in writers:
                            hazards.add(writers[byte])
                    if "W" in mode:
                        if byte in writers:
                            hazards.add(writers[byte])
                        hazards.update(readers[byte])
                    for old in hazards:
                        if old != key:
                            assert reach[ends[old]] & (1 << begins[key]), (byte[1], old, key)
                    if "R" in mode:
                        readers[byte].add(key)
                    if "W" in mode:
                        writers[byte], readers[byte] = key, set()

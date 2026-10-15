"""Independent finite actual-command DAG and byte oracle for local sequences."""

from collections import defaultdict, deque
from itertools import product

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from examples.state.recurrence import state_sum
from xdsl_npu import Kernel, autosync, compile, verify_plan
from xdsl_npu import language as N
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
    parse,
)
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.fixture(scope="module")
def artifact():
    return compile(state_sum.specialize(B=16, GROUPS=1))


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
    if task.native.data == "copy":
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
                            p, r, exact, minimum = [v.data for v in row]
                            active |= n == exact if exact >= 0 else n >= minimum and n % p == r
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


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 17])
def test_actual_instructions_cover_bytes_events_and_progress(artifact, count):
    finite_check(artifact.plan, count)


def test_entry_capture_zero_exit_and_single_drain_are_explicit(artifact):
    plan = artifact.plan
    assert len(physical_regions(plan)) == 3
    assert artifact.report["native_templates"] == 4
    assert artifact.report["sequence_channels"] == 2
    assert artifact.report["sequence_inherited_paths"] > 0
    assert artifact.asc.count("for (int64_t step") == 1
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert verify_plan(parse(artifact.plan_ir)) == artifact.report
    for count in (0, 1, 7, 1048576):
        w = runtime_work(plan, {"tiles": count})
        assert w["native_invocations"]["copy"] == count + 2
        assert w["native_invocations"]["add"] == count
        assert w["terminal_drains"] == 1
        assert w["region_tasks_per_group"] == {"r0": 1, "r1": count, "r2": 1}


@pytest.mark.parametrize("which", [0, 1])
def test_missing_real_boundary_endpoint_is_rejected(artifact, which):
    plan = artifact.plan
    declaration = [o for o in plan.walk() if isinstance(o, RegionChannelOp)][which]
    wait = next(
        o
        for o in plan.walk()
        if isinstance(o, RegionSyncOp)
        and o.channel == declaration.sym_name
        and o.action.data == "wait"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(CompileError, match="unpaired|wrong native endpoint"):
        verify_plan(plan)


def test_removed_initialization_cannot_be_replaced_by_live_in_assumptions(artifact):
    plan = artifact.plan
    region = physical_regions(plan)[0]
    region.parent.erase_op(region)
    with pytest.raises(CompileError, match="undefined sequence entry read"):
        verify_plan(plan)


def test_final_structure_check_includes_late_regions(artifact):
    plan = artifact.plan
    task = next(o for o in physical_regions(plan)[-1].walk() if isinstance(o, TaskOp))
    task.attributes["issue_order"] = StringAttr("invalid")
    with pytest.raises(VerifyException, match="issue_order"):
        verify_plan(plan)


def test_wrong_clock_and_resource_collision_are_rejected(artifact):
    plan = artifact.plan
    wait = next(o for o in plan.walk() if isinstance(o, RegionSyncOp) and o.action.data == "wait")
    wait.operands = [physical_regions(plan)[1].body.block.args[0]]
    with pytest.raises(CompileError, match="wrong resource or coordinate"):
        verify_plan(plan)
    plan = artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp))
    channel.attributes["event_id"] = IntAttr(8)
    with pytest.raises(CompileError, match="overlap/exceed"):
        verify_plan(plan)


def test_zero_trip_guard_cannot_publish_a_nonexistent_final_writer(artifact):
    plan = artifact.plan
    c = [o for o in plan.walk() if isinstance(o, RegionChannelOp)][1]
    c.attributes["domains"] = next(o for o in plan.walk() if isinstance(o, RegionChannelOp)).domains
    with pytest.raises(CompileError, match="no actual producer/consumer"):
        verify_plan(plan)


def readers():
    k = Kernel("readers", disjoint_external=True)
    initial = k.buffer("initial", "GM", 64, external=True, initialized=True)
    output = k.buffer("output", "GM", 64, external=True)
    state = k.buffer("state", "UB", 64)
    scratch = k.buffer("scratch", "UB", 64)
    count = k.runtime_size("n")
    k.region("entry")
    k.copy("init", k.view(initial, 64), k.view(state, 64))
    k.region("readers", count)
    k.task("scalar", "scalar_load", k.view(state, 4))
    k.scale("vector", k.view(state, 64), k.view(scratch, 64), 2.0)
    k.copy("dma", k.view(state, 64), k.view(output, 64))
    k.region("exit")
    k.task("overwrite", "fill", k.view(state, 64))
    return k.finish()


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 17])
def test_all_reader_families_retire_before_overwrite(count):
    finite_check(autosync(readers()), count)


def test_two_retained_children_use_same_allocation_and_checked_resources():
    definition = N.from_source(
        """
        def twice(A:N.Tensor((64,),N.f32), O:N.Tensor((64,),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.copy(A,x)
                for t in N.Pipelined(n,stages=1):
                    N.mul(x,2.0,out=x)
                for t in N.Pipelined(n,stages=1):
                    N.add(x,x,out=x)
                N.copy(x,O)
    """,
        disjoint_external=True,
    )
    result = compile(definition)
    for n in (0, 1, 2, 7):
        finite_check(result.plan, n)


def test_borrowed_cross_region_value_is_diagnosed_until_version_binding_is_supported():
    source = state_sum.source.replace(
        "T.copy(Initial[group, :], state)", "saved = T.copy(Initial[group, :], state)"
    ).replace("T.add(state,", "T.add(saved,")
    # Capture's actual module alias is preserved by the definition environment.
    from dataclasses import replace

    with pytest.raises(CompileError, match="ambiguous ownership|producer"):
        compile(replace(state_sum, source=source))


def test_large_constant_bound_uses_boundary_families_instead_of_finite_history():
    source = """
        def large(A:N.Tensor((64,),N.f32), X:N.Tensor((COUNT,64),N.f32),
                  O:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((2,64))
                s=N.alloc_ub((64,))
                N.copy(A,s)
                for t in N.Pipelined(COUNT,stages=2,issue_order={"vector":("compute","load")}):
                    with N.Stage(0,name="load"):
                        N.copy(X[t,:],x[t%2,:])
                    with N.Stage(1,name="compute"):
                        N.add(s,x[t%2,:],out=s)
                N.copy(s,O)
    """
    reports = [
        compile(
            N.from_source(source, constants={"COUNT": count}, disjoint_external=True), timeout_s=3
        ).report
        for count in (1024, 1048576)
    ]
    assert all(r["native_templates"] == 4 for r in reports)
    assert reports[0]["proof_domain_states"] == reports[1]["proof_domain_states"]
    assert reports[1]["proof_domain_states"] < 10000


def test_two_dma_pipelines_share_readonly_gm_and_release_only_actual_readers():
    source = state_sum.source.replace(
        "T.copy(state, Output[group, :])",
        """
        for u in T.Pipelined(tiles, stages=2):
            with T.Stage(0, name="reload"):
                T.copy(X[group, u, :], x[u % 2, :])
            with T.Stage(1, name="again"), T.Vector():
                T.add(state, x[u % 2, :], out=state)
        T.copy(state, Output[group, :])""",
    )
    from dataclasses import replace

    result = compile(replace(state_sum, source=source).specialize(B=16, GROUPS=1))
    for n in (0, 1, 2, 3, 7, 17):
        finite_check(result.plan, n)
    assert "asc_sync_pipe(PIPE_MTE2)" not in result.asc
    assert len(physical_regions(result.plan)) == 4


def test_cross_region_gm_ownership_is_checked_as_a_union():
    k = Kernel("bad_owners", disjoint_external=True, blocks=2)
    gm = k.buffer("gm", "GM", 256, external=True)
    ub = k.buffer("ub", "UB", 64)
    k.region("init")
    k.task("fill", "fill", k.view(ub, 64))
    k.region("left")
    k.copy("left_write", k.view(ub, 64), k.view(gm, 64, core_stride=64))
    k.region("right")
    k.copy("right_write", k.view(ub, 64), k.view(gm, 64, offset=64, core_stride=128))
    with pytest.raises(CompileError, match="sequential regions disagree on GM ownership"):
        autosync(k.finish())

"""Parent completion tested against finite actual instructions and byte hazards.

The oracle does not consume boundary summaries or the production graph. It
retains native FIFO/capture rules and pairs the real binary-event occurrences.
"""

import re
from collections import defaultdict, deque

import pytest
from xdsl.dialects import scf
from xdsl.dialects.builtin import IntAttr, StringAttr

from examples.gemm.hierarchical import hierarchical_gemm
from tests.report_contract import semantic_report
from xdsl_npu import compile, verify_plan
from xdsl_npu import language as N
from xdsl_npu.ir import (
    BoundaryOp,
    FrameOp,
    InvocationChannelOp,
    InvocationSyncOp,
    LoopChannelOp,
    LoopSyncOp,
    TaskOp,
    parse,
)
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.fixture(scope="module")
def artifact():
    return compile(hierarchical_gemm.specialize(M=16, N=16, BK=16, GROUPS=2))


def finite_parent(loop, outputs, panels):
    _, data = next(iter(loop.actors.items()))
    native = {t.uid: t for t in data.program.tasks}
    declarations = {
        op.sym_name.data: op
        for op in loop.op.body.block.ops
        if isinstance(op, (LoopChannelOp, InvocationChannelOp))
    }
    before, after, previous, pending = {}, defaultdict(list), None, []
    for block in [data.stream.body.block] + [
        op.body.block for op in data.stream.walk() if isinstance(op, BoundaryOp)
    ]:
        previous, pending = None, []
        for op in block.ops:
            if isinstance(op, TaskOp):
                before[op.uid.data], pending, previous = pending, [], op.uid.data
            elif isinstance(op, (LoopSyncOp, InvocationSyncOp)):
                if op.action.data == "set":
                    after[previous].append(op)
                else:
                    pending.append(op)
        assert not pending
    tasks = sorted(data.program.tasks, key=lambda t: t.issue_order)
    ids, edges = {}, []
    issue, prefixes, implicit, fences = {}, {}, {}, {}
    sends, waits = defaultdict(list), defaultdict(list)
    serial = 0

    def node(key):
        if key not in ids:
            ids[key] = len(edges)
            edges.append(set())
        return ids[key]

    def edge(a, b):
        if a != b:
            edges[a].add(b)

    for p in range(outputs):
        for step in range(panels + max(t.stage for t in tasks)):
            for task in tasks:
                g = step - task.stage
                if not 0 <= g < panels or task.last and g != panels - 1:
                    continue
                for op in [*before[task.uid], None, *after[task.uid]]:
                    resource = None
                    if op is None:
                        action = "task"
                    else:
                        action = op.action.data
                        if isinstance(op, LoopSyncOp):
                            low, high = op.lower.data, panels - op.upper_gap.data
                            protocol = declarations.get(op.channel.data)
                            carry = (
                                protocol is not None
                                and protocol.carry is not None
                                and protocol.carry.data
                            )
                            active = low <= g < high
                            if carry:
                                active = (
                                    (g < high or p + 1 < outputs)
                                    if action == "set"
                                    else (g >= low or p > 0)
                                )
                            if not active or g % op.period.data not in [
                                v.data for v in op.residues
                            ]:
                                continue
                            if op.last_gap is not None and g != panels - 1 - op.last_gap.data:
                                continue
                            if action in {"set", "wait"}:
                                source = protocol.sources.data[0].data.split("/")[1]
                                destination = protocol.destinations.data[0].data.split("/")[1]
                                epoch = (
                                    g if action == "set" or carry else g - protocol.distance.data
                                )
                                residues = [v.data for v in protocol.residues]
                                ordinal = (
                                    (epoch // protocol.period.data) * len(residues)
                                    + sum(epoch % protocol.period.data >= r for r in residues)
                                    - 1
                                )
                                event_id = protocol.event_ids.data[
                                    ordinal % len(protocol.event_ids)
                                ].data
                                resource = native[source].pipe, native[destination].pipe, event_id
                        else:
                            protocol = declarations[op.channel.data]
                            gap, index = protocol.source_gap.data, protocol.destination_index.data
                            if panels <= max(gap, index):
                                continue
                            if action == "set" and not (p + 1 < outputs and g == panels - 1 - gap):
                                continue
                            if action == "wait" and not (p > 0 and g == index):
                                continue
                            source = protocol.source.data.split("/")[1]
                            destination = protocol.destination.data.split("/")[1]
                            resource = (
                                native[source].pipe,
                                native[destination].pipe,
                                protocol.event_id.data,
                            )
                    key = (p, g, task.uid) if op is None else ("sync", serial)
                    serial += 1
                    begin = node((*key, "begin"))
                    end = node((*key, "end")) if action in {"task", "set"} else begin
                    edge(begin, end)
                    pipe = task.pipe
                    if pipe in issue:
                        edge(issue[pipe], begin)
                    if pipe in fences:
                        edge(fences[pipe], begin)
                    if action == "task":
                        prefix = node((*key, "prefix"))
                        edge(end, prefix)
                        if pipe in prefixes:
                            edge(prefixes[pipe], prefix)
                        if pipe in implicit:
                            edge(implicit[pipe], begin)
                        prefixes[pipe] = prefix
                        if task.implicit_completion:
                            implicit[pipe] = end
                    elif action in {"set", "barrier"} and pipe in prefixes:
                        edge(prefixes[pipe], end)
                    if action in {"wait", "barrier"}:
                        fences[pipe] = begin
                    if action == "set":
                        sends[resource].append(end)
                    elif action == "wait":
                        waits[resource].append(begin)
                    issue[pipe] = begin
    assert sends.keys() == waits.keys()
    for resource in sends:
        assert len(sends[resource]) == len(waits[resource]), resource
        for sent, consumed in zip(sends[resource], waits[resource]):
            edge(sent, consumed)
    degrees = [0] * len(edges)
    for targets in edges:
        for target in targets:
            degrees[target] += 1
    ready = deque(i for i, degree in enumerate(degrees) if not degree)
    order = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for target in edges[current]:
            degrees[target] -= 1
            if degrees[target] == 0:
                ready.append(target)
    assert len(order) == len(edges), "finite parent has a completion cycle"
    reach = [0] * len(edges)
    for current in reversed(order):
        for target in edges[current]:
            reach[current] |= 1 << target | reach[target]
    for resource in sends:
        for consumed, next_set in zip(waits[resource], sends[resource][1:]):
            assert reach[consumed] & (1 << next_set), ("event reused early", resource)
    return ids, reach


def test_retained_physical_parent_frames_and_final_ir_revalidation(artifact):
    plan = parse(artifact.plan_ir)
    assert len([op for op in plan.walk() if isinstance(op, scf.ForOp)]) == 1
    assert len([op for op in plan.walk() if isinstance(op, FrameOp)]) == 3
    assert semantic_report(verify_plan(plan)) == semantic_report(artifact.report)
    assert artifact.report["invocation_bound"] == "outputs"
    assert "core_id * npu_invocations + npu_invocation" in artifact.asc
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert artifact.asc.index("serial invocation: no whole-child drain") < artifact.asc.index(
        "asc_sync_pipe(PIPE_ALL)"
    )
    assert "uniform product overflow" in artifact.asc


@pytest.mark.parametrize(
    "outputs,panels",
    [(0, 0), (0, 7), (1, 0), (3, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4), (2, 7), (3, 17)],
)
def test_independent_native_hazards_and_event_lifetimes(artifact, outputs, panels):
    loop = extract_loop(artifact.plan)
    ids, reach = finite_parent(loop, outputs, panels)
    _, data = next(iter(loop.actors.items()))
    frames = {frame.name: frame for frame in loop.invocation.frames}
    ordered = sorted(
        enumerate(data.program.tasks), key=lambda item: (item[1].logical_phase, item[0])
    )
    writers, readers = {}, defaultdict(set)
    for p in range(outputs):
        for g in range(panels):
            for _, task in ordered:
                if task.last and g != panels - 1:
                    continue
                current = (p, g, task.uid)
                for index, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                    if task.native == "mad_first" and g == 0 and index == 2:
                        mode = "W"
                    address = sum(
                        o + s * ((g // d) % m if m else g // d) for o, s, m, d in ref.terms
                    )
                    root = ref.storage.name
                    if ref.storage.external:
                        frame = frames[root]
                        address += (
                            p
                            * frame.extent.evaluate({"outputs": outputs, "panels": panels})
                            * frame.scale
                        )
                        root = frame.root.sym_name.data
                    for row in range(ref.rows):
                        for byte in range(
                            address + row * ref.row_stride,
                            address + row * ref.row_stride + ref.nbytes,
                        ):
                            cell = root, byte
                            obligations = []
                            if "R" in mode:
                                assert cell in writers or ref.storage.initialized
                                if cell in writers:
                                    obligations.append(writers[cell])
                                readers[cell].add(current)
                            if "W" in mode:
                                if cell in writers:
                                    obligations.append(writers[cell])
                                obligations.extend(readers[cell])
                                writers[cell], readers[cell] = current, set()
                            target = ids[*current, "begin"]
                            for source in obligations:
                                if source != current:
                                    assert reach[ids[*source, "end"]] & (1 << target), (
                                        cell,
                                        source,
                                        current,
                                    )


def test_removed_fix_release_is_rejected(artifact):
    plan = artifact.plan
    for op in list(plan.walk()):
        if isinstance(op, (InvocationChannelOp, InvocationSyncOp)):
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match="missing invocation"):
        verify_plan(plan)


def test_removed_carried_operand_release_is_rejected(artifact):
    plan = artifact.plan
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopChannelOp) and op.carry is not None and op.distance.data == 2
    )
    channel.attributes.pop("carry")
    with pytest.raises(CompileError, match="missing invocation"):
        verify_plan(plan)


@pytest.mark.parametrize("change", ["extent", "scale", "coordinate", "missing_set"])
def test_parent_contract_or_actual_sync_mutation_is_rejected(artifact, change):
    plan = artifact.plan
    frame = next(op for op in plan.walk() if isinstance(op, FrameOp))
    if change == "scale":
        frame.attributes["scale"] = IntAttr(frame.scale.data + 32)
    elif change == "extent":
        frame.operands = [frame.source, frame.invocation, frame.parent_op().ub]
    elif change == "coordinate":
        stream = next(op for op in plan.walk() if op.name == "npu.stream")
        frame.operands = [frame.source, stream.body.block.args[0], frame.extent]
    else:
        op = next(
            op for op in plan.walk() if isinstance(op, InvocationSyncOp) and op.action.data == "set"
        )
        op.parent.erase_op(op)
    with pytest.raises(CompileError, match="frame|synchronization sites"):
        verify_plan(plan)


@pytest.mark.parametrize(
    "outputs,panels", [(0, 7), (17, 0), (1, 1), (17, 7), (16, 256), (1048576, 1048576)]
)
def test_work_accounting_preserves_both_axes_without_expansion(artifact, outputs, panels):
    work = runtime_work(artifact.plan, {"outputs": outputs, "panels": panels})
    assert work["native_invocations"]["mad_first"] == 2 * outputs * panels
    assert work["native_invocations"]["fix"] == 2 * outputs * int(panels > 0)
    assert (
        work["memory_requested_bytes_by_pipe_bank_mode"]["M:L0C:R"]
        == 2 * outputs * max(panels - 1, 0) * 16 * 16 * 4
    )
    assert work["sync_invocations"].get("set:FIX", 0) == 2 * max(outputs - 1, 0) * int(panels > 0)
    assert work["terminal_drains"] == 2


@pytest.mark.parametrize("outputs,panels", [(1, 1), (2, 7), (1048576, 1048576)])
def test_constant_parent_and_child_keep_one_native_body(outputs, panels):
    source = (
        re.sub(r"\bN\b", "COLS", hierarchical_gemm.source)
        .replace("T.", "N.")
        .replace("outputs: N.int64,", "outputs: N.constexpr = OUTPUTS,")
        .replace("panels: N.int64,", "panels: N.constexpr = PANELS,")
    )
    result = compile(
        N.from_source(
            source, constants={"OUTPUTS": outputs, "PANELS": panels}, disjoint_external=True
        )
    )
    assert result.report["native_templates"] == 6
    assert result.report["runtime_parameters"] == []
    assert 0 <= result.report["completion_statistics"]["condition_facts"] < 100000
    assert result.asc.count("asc_mmad(") == 1
    assert result.asc.count("asc_copy_l0c2gm(") == 1


def test_invocation_protocol_is_not_specific_to_cube_or_gemm():
    source = N.from_source(
        """
        def vector_rows(X: N.Tensor((2, outputs, tiles, 64), N.f32),
                        Y: N.Tensor((2, outputs, tiles, 64), N.f32, access="write"),
                        outputs: N.int64, tiles: N.int64):
            with N.Kernel(2, kind="vector") as group:
                x=N.alloc_ub((2,64))
                y=N.alloc_ub((3,64))
                for output in N.serial(outputs):
                    for i in N.Pipelined(tiles,stages=3):
                        with N.Stage(0,name="load"):
                            N.copy(X[group,output,i,:],x[i%2,:])
                        with N.Stage(1,name="compute"):
                            N.mul(x[i%2,:],2.0,out=y[i%3,:])
                        with N.Stage(2,name="store"):
                            N.copy(y[i%3,:],Y[group,output,i,:])
    """,
        disjoint_external=True,
    )
    result = compile(source)
    loop = extract_loop(result.plan)
    assert result.report["native_templates"] == 3
    for outputs, tiles in ((0, 0), (1, 0), (2, 1), (3, 2), (3, 7)):
        finite_parent(loop, outputs, tiles)


def test_invocation_sync_cannot_use_parent_as_child_generation(artifact):
    plan = artifact.plan
    frame = next(op for op in plan.walk() if isinstance(op, FrameOp))
    sync = next(op for op in plan.walk() if isinstance(op, InvocationSyncOp))
    sync.operands = [frame.invocation]
    with pytest.raises(CompileError, match="wrong scope"):
        verify_plan(plan)


def test_invocation_event_cannot_overlap_child_bank_assignment(artifact):
    plan = artifact.plan
    child = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopChannelOp) and op.carry is not None and op.distance.data == 2
    )
    parent = next(op for op in plan.walk() if isinstance(op, InvocationChannelOp))
    parent.attributes["source"] = StringAttr(child.sources.data[0].data)
    parent.attributes["destination"] = StringAttr(child.destinations.data[0].data)
    parent.attributes["event_id"] = IntAttr(child.event_ids.data[0].data)
    with pytest.raises(CompileError, match="event overlaps"):
        verify_plan(plan)

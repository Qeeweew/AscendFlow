"""Terminal-region and first-clear proofs checked against finite actual accesses."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block, Region

from examples.gemm.reduction import reduction_gemm
from tests.report_contract import semantic_report
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile, verify_plan
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.ir import (
    BoundaryOp,
    BufferOp,
    EndOp,
    LoopChannelOp,
    LoopSyncOp,
    TaskOp,
    ViewOp,
    parse,
)
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.fixture(scope="module", params=[False, True])
def artifact(request):
    return compile(
        reduction_gemm.specialize(M=16, N=32, BK=48, GROUPS=2, TRANSPOSE_B=request.param)
    )


def test_retained_terminal_region_and_native_initialization(artifact):
    assert len([o for o in artifact.plan.walk() if isinstance(o, BoundaryOp)]) == 1
    tasks = [o for o in artifact.plan.walk() if isinstance(o, TaskOp)]
    assert len(tasks) == 6
    assert [o.native.data for o in tasks][-2:] == ["mad_first", "fix"]
    assert artifact.report["terminal_templates"] == 1
    assert semantic_report(verify_plan(parse(artifact.plan_ir))) == semantic_report(artifact.report)
    assert "false, false, (g == 0)" in artifact.asc
    assert "if (g >= 1) { asc_sync_pipe(PIPE_M); }" in artifact.asc
    assert "if (g == n - 1) { asc_copy_l0c2gm" in artifact.asc
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    channels = [
        o for o in artifact.plan.walk() if isinstance(o, LoopChannelOp) and o.last is not None
    ]
    assert len(channels) == 1 and len(channels[0].event_ids) == 1


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 17])
def test_boundary_byte_conflicts_and_flags_have_actual_hardware_paths(artifact, count):
    _, (loop, channels, commands, graph, _) = verify_loop(artifact.plan, _details=True)
    ids, reach = finite_hardware_dag(loop, commands, channels, count)
    native = {c.task.uid: c for c in commands["aic"] if c.kind == "task"}
    tasks = [op for op in artifact.plan.walk() if isinstance(op, TaskOp)]
    writers, readers, conflicts = {}, {}, set()
    writes = 0
    for g in range(count):
        for task in tasks:
            terminal = isinstance(task.parent_op(), BoundaryOp)
            if terminal and g != count - 1:
                continue
            current = task.uid.data, g
            modes = (
                ("R", "R", "W" if g == 0 else "RW")
                if task.native.data == "mad_first"
                else ("R", "W")
            )
            if task.native.data == "fix":
                writes += 1
            for value, mode in zip(task.refs, modes):
                view, root = value.owner, value.owner.source.owner
                assert isinstance(view, ViewOp) and isinstance(root, BufferOp)
                address = view.offset.data
                for term in () if view.index_terms is None else view.index_terms:
                    coefficient, divisor, modulus = (v.data for v in term)
                    index = g // divisor
                    address += coefficient * (index % modulus if modulus else index)
                for row in range(view.rows.data):
                    start = address + row * view.row_stride.data
                    for byte in range(start, start + view.nbytes.data):
                        key = root.sym_name.data, byte
                        if "R" in mode:
                            assert key in writers or root.initialized.data
                            if key in writers:
                                conflicts.add((writers[key], current))
                            readers.setdefault(key, set()).add(current)
                        if "W" in mode:
                            if key in writers:
                                conflicts.add((writers[key], current))
                            conflicts.update((reader, current) for reader in readers.get(key, ()))
                            writers[key], readers[key] = current, set()
    assert writes == int(count > 0)
    for (a, ag), (b, bg) in conflicts:
        if (a, ag) != (b, bg):
            assert reach[ids[native[a].index, ag, "end"]] & (1 << ids[native[b].index, bg, "begin"])
    # Every claimed arc between real commands must be an actual finite path.
    maximum = max(c.task.stage for c in commands["aic"])
    for a, b, delta in graph.edges:
        source, destination = graph.nodes[a], graph.nodes[b]
        for cycle in range(-maximum // graph.period - 1, count // graph.period + 1):
            sg = source.residue + cycle * graph.period
            dg = destination.residue + (cycle + delta) * graph.period
            i, j = (
                ids.get((source.command.index, sg, source.point)),
                ids.get((destination.command.index, dg, destination.point)),
            )
            if i is not None and j is not None and i != j:
                assert reach[i] & (1 << j)


@pytest.mark.parametrize("count", [0, 1, 2, 7, 1048576])
def test_counting_is_exact_without_count_expansion(artifact, count):
    work = runtime_work(artifact.plan, {"panels": count})
    assert work["native_invocations"]["mad_first"] == count * 2
    assert work["native_invocations"]["fix"] == 2 * int(count > 0)
    assert (
        work["memory_requested_bytes_by_pipe_bank_mode"]["M:L0C:R"]
        == max(0, count - 1) * 16 * 32 * 4 * 2
    )
    assert work["transfer_requested_bytes"]["L0C->GM"] == int(count > 0) * 16 * 32 * 4 * 2
    assert work["sync_invocations"]["wait:FIX"] == 2 * int(count > 0)


def test_deleted_terminal_ready_is_not_repaired_by_final_drain(artifact):
    plan = artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, LoopChannelOp) and o.last is not None)
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing loop RAW"):
        verify_plan(plan)


def test_terminal_send_cannot_be_published_each_iteration(artifact):
    plan = artifact.plan
    source = next(
        o
        for o in plan.walk()
        if isinstance(o, LoopSyncOp) and o.last_gap is not None and o.action.data == "set"
    )
    source.attributes.pop("last_gap")
    with pytest.raises(CompileError, match="terminal activation"):
        verify_plan(plan)


def test_first_clear_does_not_initialize_other_ring_slots():
    source = reduction_gemm.source.replace(
        "c0 = T.alloc_l0c((M, N)", "c0 = T.alloc_l0c((2, M, N)"
    ).replace("c0[:, :]", "c0[k % 2, :, :]")
    with pytest.raises(CompileError, match="undefined first read"):
        compile(replace(reduction_gemm, source=source))


def test_conditional_clear_keeps_the_accumulator_numerical_reader():
    source = reduction_gemm.source.replace(
        "value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))",
        "prior = T.matmul(aa, bb, accum=c0[:, :], initialize=True)\n"
        "                value = T.matmul(aa, bb, accum=prior, initialize=(k == 0))",
    )
    result = compile(replace(reduction_gemm, source=source))
    assert result.report["native_templates"] == 7
    assert result.asc.count("asc_mmad(") == 2


def test_unsupported_boundary_and_late_first_clear_are_not_assumed_safe():
    for original, changed, message in [
        ("k == panels - 1", "k == panels - 2", "pipeline bound - 1"),
        ("k == 0", "k == 1", "own generation == 0"),
    ]:
        with pytest.raises(CompileError, match=message):
            compile(
                replace(reduction_gemm, source=reduction_gemm.source.replace(original, changed))
            )


def test_boundary_cannot_change_to_an_unverified_predicate(artifact):
    plan = artifact.plan
    boundary = next(o for o in plan.walk() if isinstance(o, BoundaryOp))
    boundary.attributes["kind"] = StringAttr("first")
    with pytest.raises(CompileError, match="boundary requires last"):
        verify_plan(plan)


def test_terminal_write_does_not_shadow_steady_readers_or_writers():
    definition = N.from_source(
        """
        def final_clear(X: N.Tensor((n, 64), N.f32),
                        Y: N.Tensor((n, 64), N.f32, access="write"), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,), name="input")
                y = N.alloc_ub((64,), name="output")
                for t in N.Pipelined(n, stages=4):
                    with N.Stage(0, name="load"):
                        N.copy(X[t, :], x)
                    with N.Stage(1, name="compute"):
                        N.mul(x, 2.0, out=y)
                    with N.Stage(2, name="store"):
                        N.copy(y, Y[t, :])
                    with N.Stage(3, name="clear"):
                        if t == n - 1:
                            N.fill(x, 0.0)
    """,
        disjoint_external=True,
    )
    result = compile(definition)
    from xdsl_npu.loop_model import extract_loop, loop_dependencies

    loop = extract_loop(result.plan)
    war = [
        o
        for o in loop_dependencies(loop)
        if o.storage == "aiv0/input" and o.hazard == "WAR" and not o.last
    ]
    assert war and {o.distance for o in war} == {1}
    assert all(o.source[1].startswith("compute_") for o in war)
    # In a faulty periodic frontier, final_clear(g) would erase these readers
    # even in the nonfinal iterations where final_clear does not execute.
    plan = result.plan
    release = next(
        o
        for o in plan.walk()
        if isinstance(o, LoopChannelOp)
        and o.sources.data[0].data.split("/")[1].startswith("compute_")
        and o.destinations.data[0].data.split("/")[1].startswith("load_")
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == release.sym_name:
            op.parent.erase_op(op)
    release.parent.erase_op(release)
    with pytest.raises(CompileError, match="missing loop WAR"):
        verify_plan(plan)


def test_terminal_writer_cannot_initialize_a_steady_first_read():
    definition = N.from_source("""
        def uninitialized(n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                y = N.alloc_ub((64,))
                for t in N.Pipelined(n, stages=2):
                    with N.Stage(0):
                        N.mul(x, 2.0, out=y)
                    with N.Stage(1):
                        if t == n - 1:
                            N.fill(x, 0.0)
    """)
    with pytest.raises(CompileError, match="undefined first read"):
        compile(definition)


def test_terminal_overwrite_of_read_only_stream_requires_retirement_prefix():
    definition = N.from_source("""
        def overwrite(X: N.Tensor((64,), N.f32, access="read_write"), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((3, 64), name="x")
                y = N.alloc_ub((64,), name="y")
                for t in N.Pipelined(n, stages=3):
                    with N.Stage(0, name="read"):
                        N.copy(X, x[t % 3, :])
                    with N.Stage(2, name="last_write"):
                        if t == n - 1:
                            N.fill(y, 0.0)
                            N.copy(y, X)
    """)
    result = compile(definition)
    from xdsl_npu.loop_model import extract_loop, loop_dependencies

    loop = extract_loop(result.plan)
    war = [o for o in loop_dependencies(loop) if o.storage == "X" and o.hazard == "WAR"]
    assert war and all(o.last and o.prefix for o in war)
    for count in [1, 2, 3, 4, 7, 17]:
        _, (loop, channels, commands, _, _) = verify_loop(result.plan, _details=True)
        ids, reach = finite_hardware_dag(loop, commands, channels, count)
        native = {c.task.native: c for c in commands["aiv0"] if c.kind == "task"}
        source = next(
            c for c in commands["aiv0"] if c.kind == "task" and c.task.uid.startswith("read_")
        )
        target = ids[native["copy"].index, count - 1, "begin"]
        for g in range(count):
            assert reach[ids[source.index, g, "end"]] & (1 << target)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 1048576])
def test_static_terminal_domains_keep_only_the_real_final_residue(count):
    kernel = Kernel("last_tile", disjoint_external=True)
    x = kernel.buffer("x", "GM", max(count, 1) * 32, external=True, initialized=True)
    out = kernel.buffer("out", "GM", 32, external=True)
    buf = kernel.buffer("buf", "UB", 64)
    kernel.loop(count, symbolic=True)
    local = kernel.view(buf, 32, index_terms=((32, 1, 2),))
    kernel.copy("load", kernel.view(x, 32, index_terms=((32, 1, 0),)), local)
    stream_block = kernel.task_block
    boundary = BoundaryOp.create(
        operands=[stream_block.args[0]],
        attributes={"kind": StringAttr("last")},
        regions=[Region(Block())],
    )
    stream_block.add_op(boundary)
    kernel.task_block = boundary.body.block
    kernel.copy("store", local, kernel.view(out, 32), stage=1)
    kernel.task_block.add_op(EndOp())
    kernel.task_block = stream_block
    result = compile(kernel.finish())
    assert result.report["native_templates"] == 2
    assert result.report["terminal_templates"] == 1
    if count < 100:
        _, (loop, channels, commands, _, _) = verify_loop(result.plan, _details=True)
        ids, reach = finite_hardware_dag(loop, commands, channels, count)
        native = {c.task.uid: c for c in commands["aiv0"] if c.kind == "task"}
        stores = [key for key in ids if key[0] == native["store"].index and key[2] == "begin"]
        assert len(stores) == int(count > 0)
        if count:
            assert stores[0][1] == count - 1
            assert reach[ids[native["load"].index, count - 1, "end"]] & (1 << ids[stores[0]])

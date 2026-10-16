"""Real local integration and counterexamples for the explicit structured entry.

The isolated legacy-verifier test is a migration cross-check. It is not an
independent hardware-semantics oracle; other tests forbid that verifier and
the old completion solver throughout synchronization and ASC formatting.
"""

from dataclasses import asdict, replace
from importlib import import_module

import pytest
from xdsl.dialects.builtin import IntAttr

from examples.gemm.batched import batched_matmul
from examples.gemm.reduction import reduction_gemm
from examples.vector.scale import scale
from tests.fixtures.pipelines import scalar_pipeline
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis.protocols import LocalMechanism
from xdsl_npu.backend.local import emit_local
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp, TaskOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.passes.insert_sync import synchronize_local
from xdsl_npu.target.ascend910b import Engine


def _cube(entry=batched_matmul, **parameters):
    return entry.specialize(
        **{"M": 16, "N": 16, "K": 16, "GROUPS": 1, "L1_SLOTS": 2, "L0_SLOTS": 2, "C_SLOTS": 2}
        | parameters
    )


def _forbid_old_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("structured local entry called the old completion solver")

    # Patch the class itself so already-imported class aliases cannot evade
    # the check. Patch function aliases owned by legacy production modules too.
    targets = (
        ("xdsl_npu", ("autosync", "verify_plan")),
        ("xdsl_npu.sync", ("autosync", "completion_graph", "verify_plan")),
        ("xdsl_npu.loop_graph", ("completion_graph",)),
        (
            "xdsl_npu.loop_sync",
            ("autosync_loop", "completion_graph", "verify_loop", "_verify_leaf"),
        ),
        ("xdsl_npu.compilation", ("autosync", "verify_plan")),
    )
    # Import all aliases before mutating any function, so monkeypatch teardown
    # cannot leave a late-imported module holding the temporary forbidden stub.
    modules = [(import_module(name), attributes) for name, attributes in targets]
    planning = import_module("xdsl_npu.loop_planning")
    monkeypatch.setattr(planning.DirectCompletion, "__init__", forbidden)
    for module, attributes in modules:
        for attribute in attributes:
            monkeypatch.setattr(module, attribute, forbidden)


def _assert_progress_certificate(certificate):
    assert len(certificate.points) == len(certificate.offsets)
    assert certificate.rounds <= len(certificate.points)
    assert certificate.edge_visits <= len(certificate.points) * len(certificate.arcs)
    for arc in certificate.arcs:
        assert (
            certificate.offsets[arc.source] + 1
            <= certificate.stride * arc.steps + certificate.offsets[arc.destination]
        )


def _native(plan, native):
    return tuple(task for task in plan.selection.tasks if task.native == native)


def _actual_sets(task):
    """Read the consecutive Set operations after the actual native anchor."""
    result, operation = [], task.operation.next_op
    while isinstance(operation, LoopSyncOp) and operation.action.data == "set":
        assert operation.generation is task.coordinate
        result.append(operation.channel.data)
        operation = operation.next_op
    return tuple(result)


def _steady_lines(asc):
    lines = asc.splitlines()
    start = next(index for index, line in enumerate(lines) if "// steady" in line)
    stop = next(index for index, line in enumerate(lines) if "// epilogue" in line)
    return lines[start:stop]


@pytest.mark.parametrize("entry", [scale.specialize(B=64), _cube()])
def test_real_dsl_entry_and_asc_never_call_old_completion_solver(entry, monkeypatch):
    # Source admission still uses its existing logical-value checks. The ban
    # starts at the new physical entry and includes final ASC formatting.
    physical = lower_source(entry.to_ir())
    before = dump(physical)
    _forbid_old_solver(monkeypatch)
    plan = synchronize_local(physical)
    asc = emit_local(plan)
    assert plan.module is not physical and dump(physical) == before
    assert plan.selection.protocols and plan.resources.assignments
    assert "asc_sync_notify(" in asc and "asc_sync_wait(" in asc
    # One final issued-pipe drain is part of the kernel ABI.
    assert asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert "for (int64_t step = 0; step < n + " in asc
    _assert_progress_certificate(plan.progress)


def test_real_scale_waw_is_removed_only_with_retained_actual_read_chains():
    plan = synchronize_local(lower_source(scale.specialize(B=64).to_ir()))
    waws = tuple(d for d in plan.dependencies.obligations if d.hazard == "WAW")
    assert len(waws) == 4
    assert not any(d.hazard == "WAW" for d in plan.dependencies.required)
    assert set(waws) == {carry.waw for carry in plan.dependencies.read_carries}
    assert not any(h.kind.value == "WAW" for h in plan.selection.requirements)
    assert {h.kind.value for h in plan.selection.requirements} == {"RAW", "WAR"}
    for carry in plan.dependencies.read_carries:
        assert carry.war is not None
        assert carry.raw.source == carry.waw.source
        assert carry.raw.destination == carry.war.source
        assert carry.war.destination == carry.waw.destination
        assert carry.waw.distance == carry.raw.distance + carry.war.distance == 2
        assert carry.raw.storage == carry.war.storage == carry.waw.storage
        assert (carry.raw.start, carry.raw.stop) == (carry.waw.start, carry.waw.stop)
        assert (carry.war.start, carry.war.stop) == (carry.waw.start, carry.waw.stop)
        assert not carry.domain.contains(1) and carry.domain.contains(2)


def test_real_last_input_reader_selects_the_release_capture():
    entry = replace(
        scale,
        source=scale.source.replace(
            "N.mul(x[slot, :], 2.0, out=y[slot, :])",
            "N.mul(x[slot, :], 2.0, out=y[slot, :])\n"
            "                N.add(x[slot, :], y[slot, :], out=y[slot, :])",
        ),
    ).specialize(B=64)
    plan = synchronize_local(lower_source(entry.to_ir()))
    last = _native(plan, "add")[0]
    release = next(
        protocol
        for protocol in plan.selection.protocols
        if protocol.source.engine is Engine.V and protocol.destination.engine is Engine.MTE2
    )
    assert release.source is last and release.distance == 2
    reads = [
        hazard
        for hazard in release.obligations
        if hazard.kind.value == "WAR" and hazard.memory.name == "aiv0/inputs"
    ]
    assert {hazard.producer.native for hazard in reads} == {"muls", "add"}
    assert all(hazard.consumer is release.destination for hazard in reads)
    assert not any(
        protocol.source.engine is Engine.MTE3 and protocol.destination.engine is Engine.MTE2
        for protocol in plan.selection.protocols
    )


def test_short_mad_rmw_requires_an_actual_same_pipe_fence():
    plan = synchronize_local(lower_source(_cube(ACCUMULATE=True).to_ir()))
    first, second = _native(plan, "mad")
    assert not first.implicit_completion and not second.implicit_completion
    fence = next(
        protocol
        for protocol in plan.selection.protocols
        if protocol.source is first and protocol.destination is second
    )
    assert fence.mechanism is LocalMechanism.FENCE and fence.distance == 0
    assert any(h.kind.value == "RAW" for h in fence.obligations)
    assert not any(a.protocol is fence for a in plan.resources.assignments)
    commands = plan.commands
    native_index = next(c.index for c in commands if c.kind == "task" and c.task is second)
    assert commands[native_index - 1].kind == "barrier"
    assert commands[native_index - 1].protocol is fence
    assert "asc_sync_pipe(PIPE_M);" in emit_local(plan)
    _assert_progress_certificate(plan.progress)


def test_same_physical_step_wait_allows_a_later_fix_set():
    later_fix = replace(
        batched_matmul,
        source=batched_matmul.source.replace(
            '("export", "load", "operands", "compute")',
            '("load", "operands", "compute", "export")',
        ),
    )
    plan = synchronize_local(lower_source(_cube(later_fix, C_SLOTS=1).to_ir()))
    release = next(
        p
        for p in plan.selection.protocols
        if p.source.engine is Engine.FIX and p.destination.engine is Engine.M
    )
    assert release.distance == 1
    assert release.destination.stage + release.distance - release.source.stage == 0
    assert release.destination.lexical < release.source.lexical
    wait = next(c for c in plan.commands if c.kind == "wait" and c.protocol is release)
    notify = next(c for c in plan.commands if c.kind == "set" and c.protocol is release)
    assert wait.index < notify.index
    notification = next(
        arc
        for arc in plan.progress.arcs
        if plan.progress.points[arc.source].command is notify
        and plan.progress.points[arc.destination].command is wait
    )
    assert notification.steps == 0
    assert "matching Wait" in notification.reason
    assert not any("Scalar" in arc.reason for arc in plan.progress.arcs)
    _assert_progress_certificate(plan.progress)
    asc = emit_local(plan)
    endpoint_lines = [
        line
        for line in _steady_lines(asc)
        if "asc_sync_wait(PIPE_FIX, PIPE_M" in line or "asc_sync_notify(PIPE_FIX, PIPE_M" in line
    ]
    assert len(endpoint_lines) == 2
    assert "asc_sync_wait(PIPE_FIX, PIPE_M" in endpoint_lines[0]
    assert "asc_sync_notify(PIPE_FIX, PIPE_M" in endpoint_lines[1]


def test_same_generation_reuse_self_lock_reports_the_actual_cycle(monkeypatch):
    physical = lower_source(scale.specialize(B=64).to_ir())
    insertion = import_module("xdsl_npu.passes.insert_sync")
    select = insertion.select_local_protocols
    corrupted = []

    def bad_pair(*args, **kwargs):
        selected = select(*args, **kwargs)
        release = next(
            p
            for p in selected.protocols
            if p.source.engine is Engine.V and p.destination.engine is Engine.MTE2
        )
        wrong = replace(release, activation=replace(release.activation, distance=0))
        corrupted.append(wrong)
        return replace(
            selected, protocols=tuple(wrong if p is release else p for p in selected.protocols)
        )

    def no_resources(*args, **kwargs):
        raise AssertionError("cyclic protocol reached event allocation before progress rejection")

    monkeypatch.setattr(insertion, "select_local_protocols", bad_pair)
    monkeypatch.setattr(insertion, "allocate_local_events", no_resources)
    with pytest.raises(CompileError, match="no increasing dispatch/completion rank") as failure:
        synchronize_local(physical)
    error = failure.value
    assert error.kind is ErrorKind.UNSUPPORTED and error.phase == "local-progress"
    assert error.witness
    release = corrupted[0]
    witness = "\n".join(error.witness)
    assert release.source.operation.uid.data + ":set/end" in witness
    assert release.destination.operation.uid.data + ":wait/begin" in witness
    assert "step distance -1" in witness
    assert "selected notification effect before matching Wait" in witness
    assert "actual source retirement capture" in witness


def test_materialized_sets_and_asc_keep_the_selected_order_at_every_native():
    plan = synchronize_local(lower_source(_cube().to_ir()))
    expected = {}
    for command in plan.commands:
        if command.kind == "set":
            expected.setdefault(command.task.operation, []).append(command.protocol.name)
    assert any(len(names) > 1 for names in expected.values())
    for task in plan.selection.tasks:
        assert _actual_sets(task) == tuple(expected.get(task.operation, ()))
    declarations = {
        op.sym_name.data: tuple(flag.data for flag in op.event_ids)
        for op in plan.module.walk()
        if isinstance(op, LoopChannelOp)
    }
    assert declarations == {a.protocol.name: a.ids for a in plan.resources.assignments}
    emitted_sets, native = [], None
    for line in _steady_lines(emit_local(plan)):
        if line.strip().startswith("// "):
            native = line.strip()[3:].rsplit(": ", 1)[0]
        if "asc_sync_notify(" in line:
            source, destination, _ = line.split("asc_sync_notify(", 1)[1].split(", ", 2)
            emitted_sets.append((native, source, destination))
    assert emitted_sets == [
        (
            c.task.operation.uid.data,
            "PIPE_" + c.protocol.source.engine.value,
            "PIPE_" + c.protocol.destination.engine.value,
        )
        for c in plan.commands
        if c.kind == "set"
    ]


def test_terminal_native_is_explicitly_rejected_without_old_fallback(monkeypatch):
    physical = lower_source(reduction_gemm.specialize(M=16, N=16, BK=16, GROUPS=1).to_ir())
    before = dump(physical)
    _forbid_old_solver(monkeypatch)
    with pytest.raises(CompileError, match="terminal-only natives") as failure:
        synchronize_local(physical)
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.phase == "local-sync"
    assert dump(physical) == before


def test_scalar_spr_setup_is_explicitly_rejected_without_old_fallback(monkeypatch):
    k = Kernel("spr_setup")
    partials = k.buffer("partials", "UB", 64)
    statistics = k.buffer("statistics", "UB", 32)
    k.loop(3)
    k.task("initialize", "fill", k.view(partials, 64))
    k.task(
        "merge",
        "merge_sum_packed",
        k.view(partials, 12, rows=2, row_stride=32),
        k.view(statistics, 8),
        params=(2, 3),
    )
    physical = k.finish()
    before = dump(physical)
    _forbid_old_solver(monkeypatch)
    with pytest.raises(CompileError, match="Scalar register setup") as failure:
        synchronize_local(physical)
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.phase == "local-sync"
    assert dump(physical) == before


def test_scalar_wait_carries_ready_to_later_vector_dispatch(monkeypatch):
    physical = scalar_pipeline(count=5)
    _forbid_old_solver(monkeypatch)
    plan = synchronize_local(physical)
    tasks = {task.operation.uid.data: task for task in plan.selection.tasks}
    load, factor, multiply = (tasks[name] for name in ("load", "factor", "multiply"))
    assert factor.engine is Engine.S and multiply.scalar_source is factor.operation
    ready = next(
        protocol
        for protocol in plan.selection.protocols
        if protocol.source is load and protocol.destination is factor
    )
    assert ready.distance == 0
    assert not any(
        protocol.source is load and protocol.destination is multiply
        for protocol in plan.selection.protocols
    )
    inherited = [
        coverage
        for coverage in plan.selection.coverage
        if coverage.requirement.producer is load
        and coverage.requirement.consumer is multiply
        and coverage.requirement.kind.value == "RAW"
    ]
    assert inherited and all(coverage.protocol is ready for coverage in inherited)
    assert all("blocking controller" in coverage.rule for coverage in inherited)
    assert plan.resource_work.controller_comparisons > 0
    _assert_progress_certificate(plan.progress)
    asc = emit_local(plan)
    assert "__ubuf__ volatile float*" in asc
    assert "asc_mul_scalar(" in asc and "npu_value_factor" in asc


def test_single_scalar_result_cannot_be_used_after_next_iteration_overwrites_it(monkeypatch):
    physical = scalar_pipeline(count=5)
    multiply = next(
        op for op in physical.walk() if isinstance(op, TaskOp) and op.uid.data == "multiply"
    )
    # At physical step g+2, factor(g+1) runs before multiply(g). The emitted
    # single C scalar slot would therefore contain the next generation's value.
    multiply.attributes["stage"] = IntAttr(2)
    _forbid_old_solver(monkeypatch)
    # Current physical admission rejects this before local selection's scalar
    # lifetime check. Its diagnostic must demand actual versioned storage.
    with pytest.raises(CompileError, match="delayed captures require explicit versioned storage"):
        synchronize_local(physical)


@pytest.mark.parametrize("model", [scale, batched_matmul])
def test_fixed_trip_count_does_not_expand_local_structure_or_work(model):
    measurements = []
    for count in (2**10, 2**20):
        # Bind tensor extents and the actual Pipelined count together. The
        # original runtime scalar argument can remain unused in this variant.
        entry = replace(model, source=model.source.replace("tiles,", f"{count},"))
        entry = entry.specialize(B=64) if model is scale else _cube(entry)
        plan = synchronize_local(lower_source(entry.to_ir()))
        assert plan.loop.constant_count == count
        measurements.append(
            (
                len(plan.selection.tasks),
                len(plan.dependencies.obligations),
                len(plan.dependencies.required),
                len(plan.dependencies.read_carries),
                len(plan.selection.protocols),
                len(plan.commands),
                len(plan.progress.points),
                len(plan.progress.arcs),
                plan.progress.offsets,
                plan.progress.rounds,
                plan.progress.edge_visits,
                plan.contracts.statistics,
                asdict(plan.selection_work),
                asdict(plan.resource_work),
            )
        )
        _assert_progress_certificate(plan.progress)
    assert measurements[0] == measurements[1]


@pytest.mark.parametrize("entry", [scale.specialize(B=64), _cube()])
def test_materialized_ir_legacy_debug_cross_check_is_not_an_independent_oracle(entry):
    # Deliberately isolated from the solver-ban tests: this merely checks
    # compatibility of the actual materialized physical encoding during M2.
    from xdsl_npu.loop_sync import verify_loop

    plan = synchronize_local(lower_source(entry.to_ir()))
    report = verify_loop(plan.module)
    assert report["native_templates"] == len(plan.selection.tasks)
    assert report["channels"] == len(plan.resources.assignments)

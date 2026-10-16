"""Parametric-loop regressions and an independent finite hardware DAG.

The finite comparisons below test boundaries; they are not a proof for all N.
The normal compiler must never invoke finite task instantiation.
"""

from collections import deque

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.fixtures.cv import cv_source, segmented_cv_program, vector_source
from tests.report_contract import semantic_report
from xdsl_npu import compile, emit_asc, verify_plan
from xdsl_npu import language as N
from xdsl_npu.check import bounded_check
from xdsl_npu.ir import LoopChannelOp, LoopSyncOp, parse
from xdsl_npu.loop_model import loop_dependencies
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.model import CompileError, Program


@pytest.fixture(scope="module", params=[vector_source, cv_source, segmented_cv_program])
def artifact(request):
    return compile(request.param(N.dynamic("tasks")))


def finite_hardware_dag(loop, sequences, channel_map, count):
    """Instantiate actual commands and FIFO/capture/gate contracts, not family arcs."""
    ids, successors = {}, []

    def node(command, g, point):
        key = command.index, g, point
        if key not in ids:
            ids[key] = len(ids)
            successors.append(set())
        return ids[key]

    def edge(a, b):
        if a != b:
            successors[a].add(b)

    sends, waits = {}, {}
    for actor, sequence in sequences.items():
        issue, native, fences, notifications = {}, {}, {}, {}
        gate = None
        maximum = max(command.task.stage for command in sequence)
        for step in range(count + maximum):
            for command in sequence:
                g = step - command.task.stage
                if not command.lower <= g < count - command.upper:
                    continue
                if g % command.period not in command.residues:
                    continue
                if command.last_gap is not None and g != count - 1 - command.last_gap:
                    continue
                pipe = command.task.pipe
                begin = node(command, g, "begin")
                end = (
                    node(command, g, "end")
                    if command.kind in {"task", "set", "cross_set"}
                    else begin
                )
                edge(begin, end)
                if pipe in issue:
                    edge(issue[pipe], begin)
                if pipe in fences:
                    edge(fences[pipe], begin)
                if gate is not None:
                    edge(gate, begin)
                if command.kind == "task":
                    prefix = node(command, g, "prefix")
                    edge(end, prefix)
                    if pipe in native:
                        previous_command, previous_end, previous_prefix = native[pipe]
                        edge(previous_prefix, prefix)
                        if previous_command.task.implicit_completion:
                            edge(previous_end, begin)
                    native[pipe] = command, end, prefix
                elif command.kind in {"set", "cross_set", "barrier"} and pipe in native:
                    edge(native[pipe][2], end)
                if command.kind == "cross_set":
                    if command.channel in notifications:
                        edge(notifications[command.channel], end)
                    notifications[command.channel] = end
                if command.kind in {"wait", "barrier"}:
                    fences[pipe] = begin
                if command.kind == "cross_wait" or (
                    command.kind in {"task", "wait"} and pipe == "S"
                ):
                    gate = end
                if command.kind in {"set", "cross_set"}:
                    sends[actor, command.channel, g] = end
                elif command.kind in {"wait", "cross_wait"}:
                    waits[actor, command.channel, g] = begin
                issue[pipe] = begin
    for name, channel in channel_map.items():
        for g in range(count - channel.distance):
            if channel.last and g != count - 1 - channel.distance:
                continue
            if g % channel.period not in channel.residues:
                continue
            for actor, _ in channel.sources:
                for destination_actor, _ in channel.destinations:
                    edge(
                        sends[actor, name, g], waits[destination_actor, name, g + channel.distance]
                    )
    incoming = [0] * len(ids)
    for targets in successors:
        for target in targets:
            incoming[target] += 1
    ready = deque(i for i, degree in enumerate(incoming) if degree == 0)
    order = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for target in successors[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    assert len(order) == len(ids), "actual finite hardware graph has a cycle"
    reach = [0] * len(ids)
    for current in reversed(order):
        for target in successors[current]:
            reach[current] |= (1 << target) | reach[target]
    return ids, reach


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 17])
def test_periodic_completion_arcs_are_actual_hardware_paths_at_boundaries(artifact, count):
    _, (loop, channel_map, sequences, graph, _) = verify_loop(artifact.plan, _details=True)
    ids, reach = finite_hardware_dag(loop, sequences, channel_map, count)
    maximum = max(command.task.stage for sequence in sequences.values() for command in sequence)
    for a, b, delta in graph.edges:
        source, destination = graph.nodes[a], graph.nodes[b]
        for cycle in range(-maximum // graph.period - 1, count // graph.period + 1):
            sg = source.residue + cycle * graph.period
            dg = destination.residue + (cycle + delta) * graph.period
            i = ids.get((source.command.index, sg, source.point))
            j = ids.get((destination.command.index, dg, destination.point))
            if i is not None and j is not None and i != j:
                assert reach[i] & (1 << j), (count, source, sg, destination, dg)
    # Check the WHOLE proved endpoint relation in a finite model containing
    # only actual commands. Analysis-only prefixes do not appear in that model.
    native = {
        (command.actor, command.task.uid): command
        for sequence in sequences.values()
        for command in sequence
        if command.kind == "task"
    }
    for obligation in loop_dependencies(loop):
        a, b = native[obligation.source], native[obligation.destination]
        for dg in range(obligation.distance, count):
            if dg % obligation.period in obligation.residues:
                i = ids[a.index, dg - obligation.distance, "end"]
                j = ids[b.index, dg, "begin"]
                assert reach[i] & (1 << j), (count, obligation, dg)
    endpoint = {
        (command.actor, command.channel, command.kind.removeprefix("cross_")): command
        for sequence in sequences.values()
        for command in sequence
        if command.kind in {"set", "wait", "cross_set", "cross_wait"}
    }
    for name, channel in channel_map.items():
        active = [
            g for g in range(count - channel.distance) if g % channel.period in channel.residues
        ]
        advance = len(channel.ids) if channel.kind == "local" else 15
        for old, following in zip(active, active[advance:]):
            for receiver, _ in channel.destinations:
                for sender, _ in channel.sources:
                    a, b = endpoint[receiver, name, "wait"], endpoint[sender, name, "set"]
                    i, j = (
                        ids[a.index, old + channel.distance, "begin"],
                        ids[b.index, following, "end"],
                    )
                    assert reach[i] & (1 << j), (count, name, old, following)


def test_normal_compilation_never_instantiates_input_length(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("normal source compilation expanded a runtime length")

    monkeypatch.setattr(Program, "instances", forbidden)
    sizes = []
    for count in [4, 4096, 1 << 26, N.dynamic("tasks")]:
        result = compile(vector_source(count))
        report = result.report
        sizes.append(
            (report["native_templates"], report["completion_node_families"], report["channels"])
        )
        assert "for (int64_t step" in result.asc
        assert result.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert len(set(sizes)) == 1


def test_selection_builds_only_planning_and_materialized_graphs_without_a_path_oracle(monkeypatch):
    import xdsl_npu.loop_sync as sync

    original, calls = sync.completion_graph, []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sync, "completion_graph", counted)
    from xdsl_npu.loop_graph import Graph

    def forbidden(*args, **kwargs):
        raise AssertionError("planning or final verification called the old path oracle")

    monkeypatch.setattr(Graph, "path", forbidden)
    result = compile(cv_source(N.dynamic("tasks")))
    assert result.report["channels"] > 10
    assert len(calls) == 2  # native insertion cuts and the materialized resource graph


@pytest.mark.parametrize("width", [1, 8, 24])
def test_guarded_prefix_graph_has_linear_edges_in_template_count(width):
    chain = N.from_source(
        """
        def chain(X:N.Tensor((tasks,64),N.f32),
                  Y:N.Tensor((tasks,64),N.f32,access="write"),tasks:N.int64):
            with N.Kernel():
                inputs=N.alloc_ub((2,64))
                outputs=N.alloc_ub((2,64))
                for task in N.Pipelined(tasks,stages=3):
                    with N.Stage(0):
                        value=N.copy(X[task,:],inputs[task%2,:])
                    with N.Stage(1),N.Vector():
                        for repeat in N.unroll(WIDTH):
                            value=N.mul(value,2.0,out=outputs[task%2,:])
                    with N.Stage(2):
                        N.copy(value,Y[task,:])
    """,
        constants={"WIDTH": width},
        disjoint_external=True,
    )

    report = compile(chain).report
    assert report["native_templates"] == width + 2
    assert report["completion_arc_families"] <= 3 * report["completion_node_families"]


def test_removed_release_and_shortened_binary_reuse_are_rejected():
    result = compile(vector_source(N.dynamic("tasks")))
    plan = result.plan
    release = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data > 0
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == release.sym_name:
            op.parent.erase_op(op)
    release.parent.erase_op(release)
    with pytest.raises(CompileError, match="missing loop WAR"):
        emit_asc(plan)
    plan = result.plan
    channel = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and len(op.event_ids) == 2
    )
    channel.attributes["event_ids"] = ArrayAttr([IntAttr(channel.event_ids.data[0].data)])
    with pytest.raises(CompileError, match="re-Set"):
        verify_plan(plan)


def test_runtime_ssa_roundtrip_and_explicit_finite_check_boundary(artifact):
    plan = parse(artifact.plan_ir)
    assert semantic_report(verify_plan(plan)) == semantic_report(artifact.report)
    assert emit_asc(plan) == artifact.asc
    with pytest.raises(CompileError, match="explicit finite test instance"):
        bounded_check(plan)


def test_war_records_last_reader_not_logical_result_completion():
    plan = compile(vector_source(N.dynamic("tasks"))).plan
    _, (loop, _, _, _, _) = verify_loop(plan, _details=True)
    hazards = loop_dependencies(loop)
    release = [o for o in hazards if o.hazard == "WAR" and o.storage.endswith("/inputs")]
    assert release and {o.distance for o in release} == {2}
    assert all(
        o.source[1].startswith("compute_") and o.destination[1].startswith("load_") for o in release
    )


def test_nested_runtime_loop_is_diagnosed_instead_of_expanded():
    nested = N.from_source(
        """
        def nested(X:N.Tensor((tasks,64),N.f32),tasks:N.int64):
            with N.Kernel():
                ub=N.alloc_ub((64,))
                for task in N.Pipelined(tasks,stages=1):
                    with N.Stage(0):
                        for inner in N.Pipelined(tasks,stages=1):
                            with N.Stage(0):
                                N.copy(X[inner,:],ub)
    """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="nested AST pipelines"):
        compile(nested)

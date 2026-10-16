"""Factorization against exhaustive arithmetic and finite byte/hardware oracles."""

from math import lcm

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from examples.vector.independent_rings import independent_rings
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile, verify_plan
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import BufferOp, LoopChannelOp, LoopSyncOp, TaskOp, ViewOp
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.lowering import lower_source
from xdsl_npu.periodic import normalize, project, union


def test_crt_projection_preserves_first_occurrence_and_all_protocol_residues():
    for period in range(1, 13):
        for clock in range(1, 13):
            for residues in [(r,) for r in range(period)] + [tuple(range(period))]:
                for minimum in [0, 1, period - 1, period, period + 1, 3 * period + clock]:
                    expected = {}
                    for g in range(minimum, minimum + lcm(period, clock)):
                        if g % period in residues:
                            expected.setdefault(g % clock, g)
                    assert dict(project(period, residues, clock, minimum)) == expected


def test_residue_union_normalizes_before_coupling_periods():
    assert union({29: range(29), 31: range(31)}, "test") == (1, (0,))
    assert union({4: (0, 2), 6: (1, 3, 5)}, "test") == (1, (0,))
    with pytest.raises(CompileError, match="coupled period 899"):
        union({29: (0,), 31: (0,)}, "test")
    # Compare every nonempty subset of small periods, not just uniform rings.
    for p in range(1, 9):
        for mask in range(1, 1 << p):
            selected = tuple(r for r in range(p) if mask & (1 << r))
            q, rs = normalize(p, selected)
            assert p % q == 0
            assert {r for r in range(p) if r % q in rs} == set(selected)


@pytest.fixture(scope="module")
def artifact():
    return compile(independent_rings)


def test_unrelated_storage_periods_do_not_multiply(artifact):
    report = artifact.report
    assert report["storage_periods"] == {"X": 1, "aiv0/inputs": 5, "aiv0/outputs": 7, "Y": 1}
    assert report["protocol_period"] == 1
    small = compile(independent_rings.specialize(P=2, Q=2)).report
    assert report["completion_node_families"] == small["completion_node_families"]
    assert report["native_templates"] == 3
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    loop = extract_loop(artifact.plan)
    hazards = loop_dependencies(loop)
    assert {o.distance for o in hazards if o.hazard == "WAR"} == {5, 7}
    assert {o.period for o in hazards} == {5, 7}


@pytest.mark.parametrize("count", [0, 1, 2, 4, 5, 6, 7, 8, 14, 35])
def test_coprime_ring_conflicts_against_independent_byte_oracle(artifact, count):
    _, (loop, channels, commands, _, _) = verify_loop(artifact.plan, _details=True)
    ids, reach = finite_hardware_dag(loop, commands, channels, count)
    native = {c.task.uid: c for c in commands["aiv0"] if c.kind == "task"}
    tasks = sorted(
        (op for op in artifact.plan.walk() if isinstance(op, TaskOp)),
        key=lambda op: 0 if op.logical_phase is None else op.logical_phase.data,
    )
    writers, readers, conflicts = {}, {}, set()
    for g in range(count):
        for task in tasks:
            current = task.uid.data, g
            assert task.native.data in {"copy", "muls"}
            for value, mode in zip(task.refs, ("R", "W")):
                view = value.owner
                assert isinstance(view, ViewOp)
                root = view.source.owner
                assert isinstance(root, BufferOp)
                address = view.offset.data
                for term in view.index_terms:
                    coefficient, divisor, modulus = (item.data for item in term)
                    index = g // divisor
                    address += coefficient * (index % modulus if modulus else index)
                for byte in range(address, address + view.nbytes.data):
                    key = root.sym_name.data, byte
                    if mode == "R":
                        assert key in writers or root.initialized.data
                        if key in writers:
                            conflicts.add((writers[key], current, "RAW"))
                        readers.setdefault(key, set()).add(current)
                    else:
                        if key in writers:
                            conflicts.add((writers[key], current, "WAW"))
                        conflicts.update(
                            (reader, current, "WAR") for reader in readers.get(key, ())
                        )
                        writers[key], readers[key] = current, set()
    for (a, ag), (b, bg), hazard in conflicts:
        if (a, ag) != (b, bg):
            assert reach[ids[native[a].index, ag, "end"]] & (
                1 << ids[native[b].index, bg, "begin"]
            ), (count, a, ag, b, bg, hazard)
    if count >= 8:
        assert {bg - ag for (_, ag), (_, bg), h in conflicts if h == "WAR"} == {5, 7}


def test_protocol_period_unrelated_to_address_period_cannot_hide_missing_ready(artifact):
    plan = artifact.plan
    channel = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data == 0
    )
    # A malformed plan only sends/consumes readiness on even generations.
    # Storage period 5 is odd: checking just residue zero would miss the race.
    for op in plan.walk():
        if op is channel or (isinstance(op, LoopSyncOp) and op.channel == channel.sym_name):
            op.attributes["period"] = IntAttr(2)
            op.attributes["residues"] = ArrayAttr([IntAttr(0)])
    with pytest.raises(CompileError, match="missing loop RAW"):
        verify_plan(plan)


def test_aliases_of_one_allocation_are_coupled_before_bounds_analysis():
    plan = lower_source(independent_rings.specialize(P=29, Q=31))
    task = next(op for op in plan.walk() if isinstance(op, TaskOp) and op.native.data == "muls")
    view = task.refs[0].owner
    view.attributes["index_terms"] = ArrayAttr([ArrayAttr([IntAttr(256), IntAttr(1), IntAttr(31)])])
    with pytest.raises(CompileError, match="storage aiv0/inputs: coupled period 899"):
        verify_plan(plan)


def test_large_independent_periods_pass_analysis_but_not_event_capacity():
    source = independent_rings.specialize(P=29, Q=31)
    physical = lower_source(source)
    loop = extract_loop(physical)
    assert loop.storage_periods == {"X": 1, "aiv0/inputs": 29, "aiv0/outputs": 31, "Y": 1}
    assert {o.period for o in loop_dependencies(loop)} == {29, 31}
    # Unthrottled MTE2 can issue >8 ready notifications before V consumes
    # them. Factorization must not hide a real event-lifetime failure.
    with pytest.raises(CompileError, match="no legal binary event reuse width <=8"):
        compile(source)


def test_proof_budget_is_checked_before_large_domain_materialization(monkeypatch):
    from xdsl_npu import loop_graph

    graph = loop_graph.Graph(1, None)
    # No physical instructions are needed to test the guarded-prefix bound.
    command = loop_graph.Command(0, "aiv0", None, "task")
    graph.node(command, 0, "begin")
    monkeypatch.setattr(loop_graph, "MAX_PROOF_STATES", 8)
    with pytest.raises(CompileError, match="requested=1000001.*first=1000000"):
        graph.domain(0, 1_000_000, 0)
    assert graph.domain_states == 0 and not graph.domains

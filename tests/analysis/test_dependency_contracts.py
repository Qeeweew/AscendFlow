"""Direct periodic read carries checked against independent actual byte effects."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block, Region

from tests.fixtures.pipelines import cube_pipeline, vector_pipeline
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis.dependencies import GenerationDomain, analyze_dependencies
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import BoundaryOp, EndOp
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.model import Ref, Task


def _matches(dependence, source, destination, kind):
    return (
        dependence.source[1] == source
        and dependence.destination[1] == destination
        and dependence.hazard == kind
    )


def _exists(dependence, generation, count, tasks):
    return (
        dependence.distance <= generation < count
        and generation % dependence.period in dependence.residues
        and (not dependence.last or generation == count - 1)
        and (not tasks[dependence.source].last or generation - dependence.distance == count - 1)
    )


def _actual_conflicts(loop, count):
    """Finite test oracle: no production access cutter, frontier, or modes_for."""
    ordered = sorted(
        (task.logical_phase, owner, index, actor, task)
        for owner, (actor, data) in enumerate(loop.actors.items())
        for index, task in enumerate(data.program.tasks)
    )
    writers, readers, conflicts = {}, {}, set()
    for generation in range(count):
        for terminal in (False, True):
            if terminal and generation != count - 1:
                continue
            for _, _, _, actor, task in ordered:
                if task.last != terminal:
                    continue
                current = generation, actor, task.uid
                touched = {}
                for index, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                    if index == 2:
                        if task.native == "mad_first" and generation == 0:
                            mode = "W"
                        if task.native == "mad_cycle" and generation % task.params[3] == 0:
                            mode = "W"
                    root = loop.root(actor, ref)
                    for start, stop in ref.segments(generation):
                        for byte in range(start, stop):
                            key = root, byte
                            before = touched.get(key, "")
                            touched[key] = before + mode
                for key, mode in touched.items():
                    writer = writers.get(key)
                    if "R" in mode:
                        if writer is not None and writer != current:
                            conflicts.add((writer, current, "RAW", *key))
                        readers.setdefault(key, set()).add(current)
                    if "W" in mode:
                        if writer is not None and writer != current:
                            conflicts.add((writer, current, "WAW", *key))
                        for reader in readers.get(key, ()):
                            if reader != current:
                                conflicts.add((reader, current, "WAR", *key))
                        writers[key], readers[key] = current, set()
    return conflicts


def _check_carries(loop, analysis, count):
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    conflicts = _actual_conflicts(loop, count)
    for carry in analysis.read_carries:
        waw, raw, war = carry.waw, carry.raw, carry.war
        assert raw.source == waw.source
        assert raw.storage == waw.storage
        assert (raw.start, raw.stop) == (waw.start, waw.stop)
        if war is None:
            assert raw.destination == waw.destination
            assert raw.distance == waw.distance
        else:
            assert raw.destination == war.source
            assert war.destination == waw.destination
            assert raw.distance + war.distance == waw.distance
            assert war.storage == waw.storage
            assert (war.start, war.stop) == (waw.start, waw.stop)
        for generation in range(count):
            if not _exists(waw, generation, count, tasks) or not carry.domain.contains(generation):
                continue
            reader_generation = generation - (0 if war is None else war.distance)
            source = generation - waw.distance, *waw.source
            reader = reader_generation, *raw.destination
            destination = generation, *waw.destination
            assert _exists(raw, reader_generation, count, tasks)
            if war is not None:
                assert _exists(war, generation, count, tasks)
            for byte in range(waw.start, waw.stop):
                assert (source, destination, "WAW", waw.storage, byte) in conflicts
                assert (source, reader, "RAW", raw.storage, byte) in conflicts
                if war is not None:
                    assert (reader, destination, "WAR", war.storage, byte) in conflicts

    # Independently check the result of every reduction: actual byte conflicts
    # remain reachable using required edges that correspond to actual reads.
    graph = {}
    for dependence in analysis.required:
        for generation in range(count):
            if not _exists(dependence, generation, count, tasks):
                continue
            source = generation - dependence.distance, *dependence.source
            destination = generation, *dependence.destination
            if any(
                (source, destination, dependence.hazard, dependence.storage, byte) in conflicts
                for byte in range(dependence.start, dependence.stop)
            ):
                graph.setdefault(source, set()).add(destination)
    for source, destination in {(source, destination) for source, destination, *_ in conflicts}:
        visited, pending = set(), [source]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(graph.get(current, ()))
        assert destination in visited


def _accumulator(native="mad_cycle", *, cycle=3, count=9, slots=1, export=False, initialize=False):
    k = Kernel("accumulator", core="aic", disjoint_external=True)
    x = k.buffer("x", "GM", 512, "f16", layout="NZ", external=True, initialized=True)
    y = k.buffer("y", "GM", 512, "f16", layout="NZ", external=True, initialized=True)
    la = k.buffer("la", "L1", 512, "f16", layout="NZ")
    lb = k.buffer("lb", "L1", 512, "f16", layout="NZ")
    a = k.buffer("a", "L0A", 512, "f16", layout="ZZ")
    b = k.buffer("b", "L0B", 512, "f16", layout="ZN")
    c = k.buffer("c", "L0C", slots * 1024, layout="NZ")
    output = k.buffer("output", "GM", 1024, external=True) if export else None
    if export:
        k.init_fixpipe()
    k.loop(count)
    ar, br = k.view(a, 512), k.view(b, 512)
    cr = k.view(c, 1024, stride=1024, slots=slots)
    lar, lbr = k.view(la, 512), k.view(lb, 512)
    k.copy("load_a", k.view(x, 512), lar)
    k.copy("load_b", k.view(y, 512), lbr)
    k.copy("operand_a", lar, ar)
    k.copy("operand_b", lbr, br)
    if initialize:
        k.task("clear", "mad", ar, br, cr, params=(16, 16, 16, 1))
    params = (16, 16, 16) if native == "mad_first" else (16, 16, 16, cycle)
    k.task("accumulate", native, ar, br, cr, params=params)
    if output is not None:
        k.fixpipe("export", cr, k.view(output, 1024))
    return extract_loop(k.finish())


def _terminal(k):
    steady = k.task_block
    boundary = BoundaryOp.create(
        operands=[steady.args[0]],
        attributes={"kind": StringAttr("last")},
        regions=[Region(Block())],
    )
    steady.add_op(boundary)
    k.task_block = boundary.body.block
    return steady


def _end_terminal(k, steady):
    k.task_block.add_op(EndOp())
    k.task_block = steady


@pytest.mark.parametrize("recipe", [vector_pipeline, cube_pipeline])
@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_real_pipeline_carries_have_actual_raw_war_endpoints(recipe, count):
    loop = extract_loop(recipe(count=max(1, count)))
    analysis = analyze_dependencies(loop)
    assert analysis.obligations == loop_dependencies(loop)
    assert any(d.hazard == "WAW" for d in analysis.obligations)
    assert not any(d.hazard == "WAW" for d in analysis.required)
    assert {d for d in analysis.obligations if d.hazard != "WAW"} <= set(analysis.required)
    _check_carries(loop, analysis, count)


def test_each_pending_reader_remains_an_actual_independent_carry():
    loop = extract_loop(vector_pipeline(late_reader=True))
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if _matches(c.waw, "load", "load", "WAW")]
    assert {c.raw.destination[1] for c in carries} == {"scale", "last_read"}
    assert all(c.war.source == c.raw.destination for c in carries)
    _check_carries(loop, analysis, 6)


def test_aliased_rmw_uses_its_real_raw_without_a_self_war():
    loop = extract_loop(vector_pipeline(late_reader=True))
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if _matches(c.waw, "scale", "last_read", "WAW")]
    assert len(carries) == 2
    assert all(c.war is None and c.raw.destination == c.waw.destination for c in carries)
    assert not any(_matches(d, "last_read", "last_read", "WAR") for d in analysis.obligations)
    _check_carries(loop, analysis, 6)


def test_unread_fragment_keeps_waw_and_physical_holes_stay_separate():
    k = Kernel("fragments", disjoint_external=True)
    source = k.buffer("source", "GM", 96, external=True, initialized=True)
    data = k.buffer("data", "UB", 96)
    output = k.buffer("output", "UB", 32)
    k.loop(5)
    src = k.view(source, 32, rows=2, row_stride=64)
    dst = k.view(data, 32, rows=2, row_stride=64)
    k.task("load", "copy_rows", src, dst)
    k.scale("read_first", k.view(data, 32), k.view(output, 32), 2.0)
    loop = extract_loop(k.finish())
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if c.waw.storage == "aiv0/data"]
    assert carries and {(c.waw.start, c.waw.stop) for c in carries} == {(0, 32)}
    remaining = [d for d in analysis.required if d.storage == "aiv0/data" and d.hazard == "WAW"]
    assert {(d.start, d.stop) for d in remaining} == {(64, 96)}
    assert all(not (d.start < 64 and d.stop > 32) for d in analysis.obligations)
    _check_carries(loop, analysis, 5)


@pytest.mark.parametrize("native,cycle", [("mad_first", 3), ("mad_cycle", 2), ("mad_cycle", 3)])
def test_accumulator_clear_generations_are_not_read_certificates(native, cycle):
    loop = _accumulator(native, cycle=cycle)
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if c.waw.storage == "aic/c"]
    assert len(carries) == 1 and carries[0].war is None
    carry = carries[0]
    for generation in range(9):
        read = generation > 0 if native == "mad_first" else generation % cycle != 0
        assert carry.domain.contains(generation) == (generation >= carry.waw.distance and read)
    retained = [d for d in analysis.required if d.storage == "aic/c" and d.hazard == "WAW"]
    assert bool(retained) == (native == "mad_cycle")
    _check_carries(loop, analysis, 9)


def test_cycle_one_has_no_accumulator_read_or_carry():
    analysis = analyze_dependencies(_accumulator(cycle=1))
    assert not any(c.waw.storage == "aic/c" for c in analysis.read_carries)
    assert any(d.storage == "aic/c" and d.hazard == "WAW" for d in analysis.required)


def test_first_clear_partial_domain_conservatively_keeps_same_generation_waw():
    loop = _accumulator("mad_first", initialize=True)
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if _matches(c.waw, "clear", "accumulate", "WAW")]
    assert len(carries) == 1
    assert not carries[0].domain.contains(0) and carries[0].domain.contains(1)
    assert carries[0].waw in analysis.required
    _check_carries(loop, analysis, 9)


def test_clear_residue_keeps_waw_while_always_read_slot_has_full_certificate():
    loop = _accumulator(cycle=2, slots=2, initialize=True)
    analysis = analyze_dependencies(loop)
    remaining = [d for d in analysis.required if _matches(d, "clear", "accumulate", "WAW")]
    assert {(d.start, d.stop, d.residues) for d in remaining} == {(0, 1024, (0,))}
    carries = [c for c in analysis.read_carries if _matches(c.waw, "clear", "accumulate", "WAW")]
    assert len(carries) == 1 and carries[0].waw.residues == (1,)
    assert carries[0].waw.period == 2 and carries[0].domain.period == 2
    _check_carries(loop, analysis, 9)


def test_real_earlier_full_reader_covers_waw_despite_later_guarded_rmw():
    loop = _accumulator(cycle=3, export=True)
    analysis = analyze_dependencies(loop)
    carries = [c for c in analysis.read_carries if c.waw.storage == "aic/c"]
    assert {c.raw.destination[1] for c in carries} == {"export", "accumulate"}
    assert not any(d.storage == "aic/c" and d.hazard == "WAW" for d in analysis.required)
    export = next(c for c in carries if c.raw.destination[1] == "export")
    assert export.war.distance == 1 and export.domain.contains(3)
    _check_carries(loop, analysis, 9)


@pytest.mark.parametrize("native", ["mad_first", "mad_cycle"])
def test_clear_does_not_initialize_another_accumulator_slot(native):
    with pytest.raises(CompileError, match=r"undefined first read on aic/c\[1024:2048\]"):
        analyze_dependencies(_accumulator(native, cycle=2, slots=2))


def test_undefined_first_read_and_explicit_live_in_are_preserved():
    k = Kernel("undefined")
    source = k.buffer("source", "UB", 32)
    destination = k.buffer("destination", "UB", 32)
    k.loop(3)
    k.scale("read", k.view(source, 32), k.view(destination, 32), 2.0)
    loop = extract_loop(k.finish())
    with pytest.raises(CompileError, match="undefined first read"):
        analyze_dependencies(loop)
    assert analyze_dependencies(loop, live_in={"aiv0/source"}).obligations


@pytest.mark.parametrize("count", [0, 1, 3])
def test_terminal_undefined_read_checks_only_real_final_activation(count):
    k = Kernel("terminal_undefined", disjoint_external=True)
    source = k.buffer("source", "UB", 32)
    output = k.buffer("output", "GM", 32, external=True)
    k.loop(count)
    steady = _terminal(k)
    k.copy("last_read", k.view(source, 32), k.view(output, 32))
    _end_terminal(k, steady)
    loop = extract_loop(k.finish())
    if count:
        with pytest.raises(CompileError, match="undefined terminal read"):
            analyze_dependencies(loop)
    else:
        assert not analyze_dependencies(loop).obligations
    analyze_dependencies(loop, live_in={"aiv0/source"})


def test_terminal_reader_does_not_shadow_steady_frontier_and_prefix_is_retained():
    k = Kernel("terminal_prefix", disjoint_external=True)
    state = k.buffer("state", "GM", 32, external=True, initialized=True)
    local = k.buffer("local", "UB", 32)
    k.loop(7)
    state_ref, local_ref = k.view(state, 32), k.view(local, 32)
    k.copy("read", state_ref, local_ref)
    steady = _terminal(k)
    k.copy("overwrite", local_ref, state_ref)
    _end_terminal(k, steady)
    loop = extract_loop(k.finish())
    analysis = analyze_dependencies(loop)
    wars = [d for d in analysis.required if d.storage == "state" and d.hazard == "WAR"]
    assert wars and all(d.last and d.prefix for d in wars)
    assert any(_matches(d, "read", "read", "WAW") for d in analysis.required)
    assert not any(c.waw.storage == "aiv0/local" for c in analysis.read_carries)


def test_one_scan_and_no_runtime_or_modes_for_expansion(monkeypatch):
    loops = [extract_loop(vector_pipeline(count=n)) for n in (2**10, 2**20)]
    original = Ref.segments
    calls = []

    def segments(ref, *args):
        calls.append(args)
        return original(ref, *args)

    def forbid_modes(*args):
        raise AssertionError("dependency analysis expanded modes_for")

    monkeypatch.setattr(Ref, "segments", segments)
    monkeypatch.setattr(Task, "modes_for", forbid_modes)
    results, work = [], []
    for loop in loops:
        calls.clear()
        results.append(analyze_dependencies(loop))
        work.append(len(calls))
    # Input ring: two refs x two slots; output ring: four refs x two slots;
    # streaming Y: one ref. Initialized readonly X is omitted by contract.
    assert work == [13, 13]
    assert results[0] == results[1]
    calls.clear()
    assert loop_dependencies(loops[0]) == results[0].obligations
    assert len(calls) == work[0]


def test_native_clear_period_is_separate_from_storage_and_large_n(monkeypatch):
    loop = _accumulator(cycle=4096, count=2**20)

    def forbid_modes(*args):
        raise AssertionError("dependency analysis expanded modes_for")

    monkeypatch.setattr(Task, "modes_for", forbid_modes)
    analysis = analyze_dependencies(loop)
    carry = next(c for c in analysis.read_carries if c.waw.storage == "aic/c")
    assert carry.waw.period == 1 and carry.domain.period == 4096
    assert carry.domain.contains(4095) and not carry.domain.contains(4096)
    assert analyze_dependencies(replace(loop, constant_count=2**40)) == analysis


def test_generation_domain_intersection_and_full_coverage_use_exact_residues():
    domain = GenerationDomain(minimum=5, period=6, residues=(1, 3, 5))
    assert domain.first(4, 1) == 5
    assert domain.first(4, 0) is None
    assert domain.covers(4, 1, 5)
    assert not domain.covers(4, 1, 0)
    assert not domain.covers(3, 1, 5)

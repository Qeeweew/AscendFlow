"""Guarded memory transfers compared with literal finite access frontiers."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import StringAttr

from tests.fixtures.nested import nested_cv
from xdsl_npu import language as N
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import ScopeGuardOp
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_domains import Activation, ScopedDependence
from xdsl_npu.scope_guarded_access import (
    AnchoredDependence,
    BoundaryEffect,
    analyze_guarded_access,
    dependent_frontiers,
)
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes

ALL = Activation()
FIRST = Activation(CountDomain(1, 0, minimum=1), Position(False, 0), Position(False, 0))
LAST = Activation(CountDomain(1, 0, minimum=1), Position(True, 0), Position(True, 0))


def effect(name, modes, rank, activation=ALL, *, start=0, stop=64, storage="state"):
    return BoundaryEffect(("aiv0", name), storage, start, stop, modes, activation, (rank,))


def _active(active, generation, count):
    return (
        active.count.contains(count)
        and active.first.evaluate(count) <= generation <= active.last.evaluate(count)
        and generation % active.period in active.residues
    )


def _literal(records, count, *, initialized=()):
    """Byte-by-byte execution; no production cuts, domains, or frontier scans."""
    written, pending, result = {}, {}, set()
    for generation in range(count):
        for record in sorted(records, key=lambda item: (item.logical_order, item.endpoint)):
            if not _active(record.activation, generation, count):
                continue
            target = record.endpoint, generation
            for byte in range(record.start, record.stop):
                key = record.storage, byte
                previous = written.get(key)
                readers = pending.setdefault(key, set())

                def edge(source, hazard, *, target=target, storage=record.storage, byte=byte):
                    if source is not None and source != target:
                        result.add((source, target, hazard, storage, byte))

                if "R" in record.modes:
                    assert previous is not None or record.storage in initialized
                    edge(previous, "RAW")
                    readers.add(target)
                if "W" in record.modes:
                    edge(previous, "WAW")
                    for reader in readers:
                        edge(reader, "WAR")
                    written[key], pending[key] = target, set()
    return result


def _expanded(dependencies, count):
    result = set()
    for dep in dependencies:
        if not dep.activation.count.contains(count):
            continue
        if isinstance(dep, ScopedDependence):
            pairs = (
                (generation, generation + dep.distance)
                for generation in range(count)
                if _active(dep.activation, generation, count)
            )
        elif dep.source_position is not None and dep.destination_position is not None:
            pairs = (
                (dep.source_position.evaluate(count), dep.destination_position.evaluate(count)),
            )
        else:
            pairs = (
                (
                    dep.source_position.evaluate(count)
                    if dep.source_position is not None
                    else generation,
                    dep.destination_position.evaluate(count)
                    if dep.destination_position is not None
                    else generation,
                )
                for generation in range(count)
                if _active(dep.activation, generation, count)
            )
        for source, destination in pairs:
            assert 0 <= source <= destination < count
            for byte in range(dep.start, dep.stop):
                result.add(
                    (
                        (dep.source, source),
                        (dep.destination, destination),
                        dep.hazard,
                        dep.storage,
                        byte,
                    )
                )
    return result


def _check(records, counts=range(12), *, initialized=()):
    dependencies = dependent_frontiers(records, initialized=initialized)
    for count in counts:
        assert _expanded(dependencies, count) == _literal(records, count, initialized=initialized)
    return dependencies


def test_first_init_rw_versions_and_last_reader_keep_single_count_overlap():
    records = (
        effect("init", "W", 0, FIRST),
        effect("mul", "RW", 1),
        effect("add", "RW", 2),
        effect("normalize", "R", 3, LAST),
    )
    dependencies = _check(records)
    assert any(
        dep.hazard == "RAW"
        and dep.source == ("aiv0", "init")
        and dep.destination == ("aiv0", "mul")
        and dep.activation.count.contains(1)
        for dep in dependencies
    )
    assert all(
        dep.source != ("aiv0", "init") or dep.activation.first == Position(False, 0)
        for dep in dependencies
    )


def test_first_cache_writer_reaches_every_native_reader_and_last_overwrite_waits_all():
    records = (
        effect("load_q", "W", 0, FIRST),
        effect("mad_j0", "R", 1),
        effect("mad_j6", "R", 2),
        effect("replace_q", "W", 3, LAST),
    )
    dependencies = _check(records)
    for endpoint in (("aiv0", "mad_j0"), ("aiv0", "mad_j6")):
        assert any(
            isinstance(dep, AnchoredDependence)
            and dep.source == ("aiv0", "load_q")
            and dep.source_position == Position(False, 0)
            and dep.destination == endpoint
            and dep.destination_position is None
            and dep.hazard == "RAW"
            for dep in dependencies
        )
        assert any(
            isinstance(dep, AnchoredDependence)
            and dep.source == endpoint
            and dep.source_position is None
            and dep.destination == ("aiv0", "replace_q")
            and dep.destination_position == Position(True, 0)
            and dep.hazard == "WAR"
            for dep in dependencies
        )


@pytest.mark.parametrize("period", (1, 2, 3, 5))
def test_periodic_steady_frontier_keeps_startup_and_short_terminal_fallback(period):
    records = (
        effect("init", "W", 0, FIRST),
        effect("reader", "R", 1),
        effect("update", "RW", 2, Activation(period=period, residues=(period - 1,))),
        effect("tail_read", "R", 3, LAST),
        effect("tail_write", "W", 4, LAST),
    )
    _check(records, range(2 * period + 5))


def test_disjoint_rows_preserve_incoming_value_for_the_unwritten_half():
    records = (
        effect("init", "W", 0, FIRST),
        effect("update", "RW", 1, stop=32),
        effect("read_rows", "R", 2),
        effect("store", "R", 3, LAST),
    )
    dependencies = _check(records)
    assert any(
        isinstance(dep, AnchoredDependence)
        and dep.source == ("aiv0", "init")
        and dep.start == 32
        and dep.stop == 64
        for dep in dependencies
    )


def test_count_predicates_are_partitioned_before_binding_writers():
    counts = CountDomain(2, 1, minimum=3, maximum=9)
    first = replace(FIRST, count=counts)
    active = Activation(counts)
    last = replace(LAST, count=counts)
    records = (
        effect("init", "W", 0, first),
        effect("rw", "RW", 1, active),
        effect("tail", "R", 2, last),
    )
    _check(records, range(13))


def test_missing_first_initializer_is_not_authorized_by_a_later_writer():
    records = (
        effect("rw", "RW", 0),
        effect("late_clear", "W", 1, LAST),
    )
    with pytest.raises(CompileError, match="undefined guarded first read"):
        dependent_frontiers(records)
    _check(records, initialized=("state",))


def test_no_runtime_count_enumeration_and_static_work_budget(monkeypatch):
    import xdsl_npu.scope_guarded_access as guarded

    huge = CountDomain(1, 0, exact=1 << 60)
    records = (
        effect("init", "W", 0, replace(FIRST, count=huge)),
        effect("rw", "RW", 1, Activation(huge)),
        effect("tail", "R", 2, replace(LAST, count=huge)),
    )

    def bounded_range(*args):
        result = range(*args)
        assert len(result) < 128
        return result

    monkeypatch.setattr(guarded, "range", bounded_range, raising=False)
    statistics = {}
    assert dependent_frontiers(records, statistics=statistics)
    assert statistics["visits"] < 512
    monkeypatch.setattr(guarded, "MAX_GUARDED_ACCESS_VISITS", 1)
    with pytest.raises(CompileError, match="guarded scope access visit budget"):
        dependent_frontiers(records)


def test_unsupported_parent_geometry_is_explicit():
    record = effect("reader", "R", 0, Activation(first=Position(False, 2)))
    with pytest.raises(CompileError, match="first/interior/last domains only"):
        dependent_frontiers((record,), initialized=("state",))


GUARDED_SOURCE = """
def guarded_state(
    Scratch:N.Tensor((2,2,4,8),N.f32,access="write"),
    O:N.Tensor((2,4,8),N.f32,access="write"),
    n:N.int64(minimum=1),
):
    with N.Kernel(2) as group:
        state=N.alloc_ub((4,8),N.f32,name="state")
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="initialize"):
                if t == 0:
                    N.fill(state,1.0)
            with N.Stage(1,name="update"):
                for row in N.Pipelined(4,stages=2):
                    with N.Stage(0,name="recurrence"):
                        N.mul(state[row,:],2.0,out=state[row,:])
                    with N.Stage(1,name="export"):
                        N.copy(state[row,:],Scratch[group,t % 2,row,:])
                        if t == n - 1:
                            N.copy(state[row,:],O[group,row,:])
"""


def _guarded_module():
    source = N.from_source(GUARDED_SOURCE, disjoint_external=True).to_ir()
    return lower_scopes(source.body.block.first_op)


def test_actual_case_projection_retains_native_versions_and_ordinary_period_two_root():
    program = extract_scopes(_guarded_module())
    base = analyze_scope_access(program, definitions=True)
    access = analyze_guarded_access(program, base=base)
    assert access.guarded_roots == {"aiv0/state", "O"}
    assert access.loop.storage_periods["Scratch"] == 2
    assert tuple(dep for dep in access.obligations if dep.storage == "Scratch") == tuple(
        dep for dep in base.obligations if dep.storage == "Scratch"
    )
    assert any(value is None for value in access.case_summaries.values())
    assert any(
        span.task[1].endswith("export_2_copy")
        for summary in access.case_summaries.values()
        if summary is not None
        for cell in summary.cells
        for span in cell.pending_reads
    )
    for count in (1, 2, 3, 6):
        expected = {
            edge
            for edge in _literal(access.effects, count)
            if edge[0][1] != edge[1][1]
            or access.endpoints[edge[0][0]][0] != access.endpoints[edge[1][0]][0]
        }
        actual = _expanded(
            [dep for dep in access.obligations if dep.storage in access.guarded_roots], count
        )
        assert actual == expected


def test_actual_last_only_initializer_cannot_initialize_first_recurrence():
    module = _guarded_module()
    guard = next(
        op for op in module.walk() if isinstance(op, ScopeGuardOp) and op.kind.data == "first"
    )
    guard.attributes["kind"] = StringAttr("last")
    program = extract_scopes(module)
    # The full-template descriptor still contains this fill. Only rebuilding
    # the actual cases discovers that its first-parent initialization is absent.
    base = analyze_scope_access(program, definitions=True)
    with pytest.raises(CompileError, match="undefined guarded first read"):
        analyze_guarded_access(program, base=base)


def test_conditional_parent_ring_is_explicitly_rejected_without_affecting_other_roots():
    text = GUARDED_SOURCE.replace(
        'state=N.alloc_ub((4,8),N.f32,name="state")',
        'state=N.alloc_ub((2,4,8),N.f32,name="state")',
    ).replace("state[row,:]", "state[t % 2,row,:]")
    text = text.replace("N.fill(state,1.0)", "N.fill(state[t % 2,:,:],1.0)")
    module = lower_scopes(N.from_source(text, disjoint_external=True).to_ir().body.block.first_op)
    program = extract_scopes(module)
    base = analyze_scope_access(program, definitions=True)
    with pytest.raises(CompileError, match="stationary parent captures"):
        analyze_guarded_access(program, base=base)


def test_unguarded_program_keeps_existing_access_object_and_fast_path():
    module = lower_scopes(nested_cv.to_ir().body.block.first_op)
    program = extract_scopes(module)
    base = analyze_scope_access(program)
    assert analyze_guarded_access(program, base=base) is base

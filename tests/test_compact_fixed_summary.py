"""Compact fixed frontiers preserve byte effects and native reader milestones."""

from dataclasses import replace

import pytest

from tests.test_definition_summary import _check as check_definitions
from tests.test_definition_summary import _ring as definition_ring
from tests.test_fixed_child_summary import _check as check_frontiers
from tests.test_fixed_child_summary import _pitched, _ring
from tests.test_scope_access import program
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.child_completion import project_fixed_child
from xdsl_npu.completion import Position, Site
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.loop_sync import _verify_leaf
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import Ref
from xdsl_npu.region_summary import select_fixed_storage, summarize_fixed_storage
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_model import extract_scopes


@pytest.mark.parametrize("count", (0, 1, 2, 3, 8, 17))
def test_optional_compact_ring_matches_independent_byte_oracle(count):
    loop, state = _ring(count)
    summary = summarize_fixed_storage(loop, {"aiv0/state"}, definitions_needed=())
    check_frontiers(loop, {"aiv0/state"}, summary)
    assert dict(summary.origins)["aiv0/state"] is state
    assert all(not cell.definitions for cell in summary.cells)
    assert summary.definition_visits == 0


def test_large_fixed_ring_scans_only_the_real_storage_period(monkeypatch):
    loops = [_ring(count)[0] for count in (8, 1_000_000)]
    original = Ref.segments
    calls = []

    def segments(ref, *args):
        calls.append(args)
        return original(ref, *args)

    monkeypatch.setattr(Ref, "segments", segments)
    work = []
    for loop in loops:
        calls.clear()
        summary = summarize_fixed_storage(loop, {"aiv0/state"}, definitions_needed=())
        work.append((len(calls), summary.visits, len(summary.cells)))
    assert work[0] == work[1]
    assert work[0][0] == 6  # Three actual state references, two physical slots.


def test_default_and_explicit_definition_roots_keep_all_internal_versions():
    module, loop, _ = definition_ring(8)
    before = dump(module)
    full = summarize_fixed_storage(loop, {"aiv0/state"})
    required = summarize_fixed_storage(loop, {"aiv0/state"}, definitions_needed={"aiv0/state"})
    check_definitions(loop, {"aiv0/state"}, required)
    assert required == full and required.definition_visits > 0
    assert sum(len(cell.definitions) for cell in required.cells) == 8
    assert dump(module) == before


def test_finite_drift_and_stationary_roots_share_one_exact_summary():
    module, loop, _ = _pitched(chunks=4)
    roots = {"Score", "aiv0/temporary"}
    selected = select_fixed_storage(loop, roots, definitions_needed=())
    assert selected.compact == {"aiv0/temporary"}
    before = dump(module)
    summary = summarize_fixed_storage(loop, roots, definitions_needed=(), _selection=selected)
    check_frontiers(loop, roots, summary)
    assert dump(module) == before
    assert sum(isinstance(op, TaskOp) for op in module.walk()) == 2


def test_short_fixed_count_does_not_expand_a_larger_storage_period():
    loop, _ = _ring(1)
    selection = select_fixed_storage(loop, {"aiv0/state"}, definitions_needed=())
    assert not selection.compact and selection.segments == 3
    summary = summarize_fixed_storage(loop, {"aiv0/state"}, definitions_needed=())
    check_frontiers(loop, {"aiv0/state"}, summary)


@pytest.mark.parametrize(
    "native,params", (("mad_first", (16, 16, 16)), ("mad_cycle", (16, 16, 16, 2)))
)
def test_special_mad_modes_keep_their_actual_first_read_semantics(native, params):
    kernel = Kernel("accumulator", core="aic")
    a = kernel.buffer("a", "L0A", 512, "f16", layout="ZZ")
    b = kernel.buffer("b", "L0B", 512, "f16", layout="ZN")
    c = kernel.buffer("c", "L0C", 1024, layout="NZ")
    kernel.loop(8, symbolic=True)
    kernel.task(
        "mad", native, kernel.view(a, 512), kernel.view(b, 512), kernel.view(c, 1024), params=params
    )
    loop = extract_loop(kernel.finish())
    selected = select_fixed_storage(loop, {"aic/c"}, definitions_needed=())
    assert bool(selected.compact) == (native == "mad_first")
    summary = summarize_fixed_storage(loop, {"aic/c"}, definitions_needed=())
    check_frontiers(loop, {"aic/c"}, summary)


def test_query_option_keeps_all_unbounded_readers_and_compacts_bounded_writers():
    kernel = Kernel("incoming_readers")
    source = kernel.buffer("source", "GM", 32, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(8, symbolic=True)
    kernel.copy("read", kernel.view(source, 32), kernel.view(sink, 32))
    loop = extract_loop(kernel.finish())
    roots = {"source", "aiv0/sink"}
    selected = select_fixed_storage(loop, roots, definitions_needed=(), allow_read_spans=False)
    assert selected.compact == {"aiv0/sink"}
    summary = summarize_fixed_storage(loop, roots, definitions_needed=(), allow_read_spans=False)
    check_frontiers(loop, roots, summary)
    (source_cell,) = [cell for cell in summary.cells if cell.storage == "source"]
    assert len(source_cell.pending_reads) == len(source_cell.reads_before_write) == 8
    assert all(span.first == span.last for span in source_cell.pending_reads)


def test_read_only_periodic_spans_keep_every_real_reader():
    kernel = Kernel("periodic_readers")
    source = kernel.buffer("source", "GM", 64, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(9, symbolic=True)
    kernel.copy("read", kernel.view(source, 32, stride=32, slots=2), kernel.view(sink, 32))
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"source"}, definitions_needed=())
    check_frontiers(loop, {"source"}, summary)
    assert len(summary.cells) == 2
    for cell in summary.cells:
        (span,) = cell.pending_reads
        assert span.step == 2 and span.first.evaluate(9) == cell.start // 32
        assert span.last.evaluate(9) == 8 - cell.start // 32


def test_compact_stationary_rows_preserve_holes_and_all_reader_instances():
    kernel = Kernel("pitched_readers")
    source = kernel.buffer("source", "GM", 3 * 96, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 3 * 32)
    kernel.loop(8, symbolic=True)
    kernel.task(
        "read",
        "copy_rows",
        kernel.view(source, 32, rows=3, row_stride=96),
        kernel.view(sink, 32, rows=3, row_stride=32),
    )
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"source"}, definitions_needed=())
    check_frontiers(loop, {"source"}, summary)
    assert [(cell.start, cell.stop) for cell in summary.cells] == [(0, 32), (96, 128), (192, 224)]
    assert all(len(cell.pending_reads) == 1 for cell in summary.cells)


READERS = """
def nested(Y:N.Tensor((n,8),N.f32,access="write"),n:N.int64):
    with N.Kernel(1):
        state=N.alloc_ub((8,),name="state")
        scratch=N.alloc_ub((8,),name="scratch")
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="write"):
                N.fill(state,1.0)
            with N.Stage(1,name="read"):
                for j in N.Pipelined(8,stages=2):
                    N.copy(state,scratch)
                    N.copy(state,Y[t,:])
"""


def test_scope_reader_families_have_first_acquires_and_separate_tail_prefixes():
    scope = program(READERS)
    access = analyze_scope_access(scope)
    tasks = {
        (actor, task.uid): task
        for actor, data in access.loop.actors.items()
        for task in data.program.tasks
    }
    raw = [dep for dep in access.obligations if dep.storage == "aiv0/state" and dep.hazard == "RAW"]
    war = [dep for dep in access.obligations if dep.storage == "aiv0/state" and dep.hazard == "WAR"]
    assert len(raw) == len(war) == 2
    assert {tasks[dep.destination].pipe for dep in raw} == {"V", "MTE3"}
    assert {access.endpoints[dep.destination][1].position.offset for dep in raw} == {0}
    assert {tasks[dep.source].pipe for dep in war} == {"V", "MTE3"}
    assert {access.endpoints[dep.source][1].position.offset for dep in war} == {7}
    assert all(dep.prefix and dep.distance == 1 for dep in war)
    assert all(not dep.prefix for dep in raw)
    for uid, summary in access.summaries.items():
        for cell in summary.cells:
            if access.capture_roots[uid, cell.storage] == "aiv0/state" and cell.pending_reads:
                assert len(cell.pending_reads) == 2
                assert all(
                    span.first.evaluate(8) == 0 and span.last.evaluate(8) == 7
                    for span in cell.pending_reads
                )


def test_scope_periodic_reader_cells_keep_distinct_first_and_tail_positions():
    source = READERS.replace("((n,8)", "((n,2,8)")
    source = source.replace(
        'state=N.alloc_ub((8,),name="state")', 'state=N.alloc_ub((2,8),name="state")'
    )
    source = source.replace("N.copy(state,", "N.copy(state[j%2,:],")
    source = source.replace("Y[t,:]", "Y[t,j%2,:]")
    access = analyze_scope_access(program(source))
    raw = [dep for dep in access.obligations if dep.storage == "aiv0/state" and dep.hazard == "RAW"]
    war = [dep for dep in access.obligations if dep.storage == "aiv0/state" and dep.hazard == "WAR"]
    assert len(raw) == len(war) == 4
    for dep in raw:
        assert access.endpoints[dep.destination][1].position.offset == dep.start // 32
    for dep in war:
        assert dep.prefix
        assert access.endpoints[dep.source][1].position.offset == 6 + dep.start // 32


def test_compact_reader_contract_reaches_normal_and_independent_deep_compilation():
    # One private slot must finish its readers before the next parent write.
    text = READERS.replace("N.Pipelined(n,stages=2)", "N.Pipelined(n,stages=1)")
    text = text.replace('N.Stage(1,name="read")', 'N.Stage(0,name="read")')
    source = N.from_source(text, disjoint_external=True)
    normal = compile(source, timeout_s=5)
    deep = compile(source, debug_verify=True, timeout_s=5)
    assert normal.asc == deep.asc
    assert normal.report["native_templates"] == deep.report["native_templates"] == 3


def test_scope_large_reader_count_charges_actual_static_records(monkeypatch):
    import xdsl_npu.scope_access as analysis

    scopes = [
        program(READERS.replace("N.Pipelined(8,", f"N.Pipelined({count},"))
        for count in (8, 1_000_000)
    ]
    monkeypatch.setattr(analysis, "MAX_SCOPE_ACCESS_SEGMENTS", 32)
    results = [analysis.analyze_scope_access(scope) for scope in scopes]
    assert len(results[0].endpoints) == len(results[1].endpoints)
    assert results[0].visits == results[1].visits


def test_actual_query_path_keeps_full_fixed_readers():
    source = """
def queries(Y:N.Tensor((total_queries,8),N.f32,access="write"),
            total_queries:N.int64,n:N.int64(minimum=1)):
    with N.Kernel(1) as group:
        state=N.alloc_ub((8,),name="state")
        scratch=N.alloc_ub((8,),name="scratch")
        for query in N.serial(N.ceildiv(N.max(0,total_queries-group),1)):
            flat=group+query
            for t in N.Pipelined(n,stages=1):
                with N.Stage(0,name="write"):
                    N.fill(state,1.0)
                with N.Stage(0,name="read"):
                    for j in N.Pipelined(8,stages=2):
                        N.copy(state,scratch)
                        N.copy(state,Y[flat,:])
"""
    scope = extract_scopes(lower_source(N.from_source(source, disjoint_external=True)))
    assert scope.invocations
    access = analyze_scope_access(scope)
    cells = [
        cell
        for uid, summary in access.summaries.items()
        for cell in summary.cells
        if access.capture_roots[uid, cell.storage] == "aiv0/state" and cell.pending_reads
    ]
    (readers,) = cells
    assert len(readers.pending_reads) == 16
    assert all(span.first == span.last for span in readers.pending_reads)


def test_native_reader_tail_end_does_not_retire_the_whole_family():
    kernel = Kernel("read_family")
    source = kernel.buffer("source", "GM", 32, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 8 * 32)
    kernel.loop(8, symbolic=True)
    kernel.copy("read", kernel.view(source, 32), kernel.view(sink, 32, stride=32))
    loop = extract_loop(kernel.finish())
    _, analysis = _verify_leaf(loop, _details=True)
    first = Site(("aiv0", "read"), Position(False, 0), "end", region="reader")
    end = replace(first, position=Position(False, 7))
    prefix = replace(end, point="prefix")
    first_begin = replace(first, point="begin")
    last_begin = replace(end, point="begin")
    projection = project_fixed_child(analysis, (first, end, prefix, first_begin, last_begin))
    assert not projection.reaches(first, end)
    assert projection.reaches(first, prefix)
    assert projection.reaches(first_begin, last_begin)

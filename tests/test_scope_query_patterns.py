"""Compact query summaries retain actual byte hazards and span proof requests."""

from dataclasses import replace

import pytest

from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_scope_query_summary import CARRIED, _expanded, _literal
from xdsl_npu import language as N
from xdsl_npu import region_patterns, region_sequence
from xdsl_npu.budget import limit
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.region_patterns import materialize_hazards
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_checks import check_memory, memory_check_reason, prepare_memory_checks
from xdsl_npu.scope_query_completion import QueryCompletion
from xdsl_npu.scope_query_summary import (
    query_native_sites,
    summarize_queries,
    summarize_query_patterns,
)
from xdsl_npu.scope_sync import _plan_scope_parent, _scope_access, _verify_scope_parent


@pytest.fixture(scope="module", params=("recurrence", "carried"))
def summaries(request):
    source = (
        scoped_query_recurrence
        if request.param == "recurrence"
        else N.from_source(CARRIED, disjoint_external=True)
    )
    program = extract_scopes(lower_source(source))
    access = _scope_access(program)
    return (
        program,
        access,
        summarize_queries(program, access),
        summarize_query_patterns(program, access),
    )


def test_compact_query_materialization_preserves_every_physical_record_and_order(summaries):
    _, access, full, compact = summaries
    expanded = compact.materialize()
    assert expanded.cells == full.cells
    assert tuple(part.summary.cells for part in expanded.parts) == tuple(
        part.summary.cells for part in full.parts
    )
    assert expanded.obligations == full.obligations
    assert query_native_sites(compact, access) == query_native_sites(full, access)
    assert len(compact.parts) == len(full.parts)
    assert compact.statistics["materialized_cells"] == 0
    assert compact.statistics["sequence_materialized_obligations"] == 0
    assert compact.statistics["obligations"] == len(full.obligations)


@pytest.mark.parametrize("count", (0, 1, 2, 5))
def test_compact_query_hazards_equal_independent_actual_native_byte_scan(summaries, count):
    program, access, _, compact = summaries
    expanded = compact.materialize()
    assert _expanded(expanded, program, access, count) == _literal(
        program, count, queries=len(compact.parts)
    )


def test_compact_query_does_not_construct_cells_or_physical_obligations(summaries, monkeypatch):
    program, access, full, _ = summaries

    def forbidden(*args, **kwargs):
        raise AssertionError("compact query path materialized a physical record")

    with monkeypatch.context() as patch:
        patch.setattr(region_patterns, "CellSummary", forbidden)
        patch.setattr(region_sequence, "Cell", forbidden)
        patch.setattr(region_sequence, "Obligation", forbidden)
        compact = summarize_query_patterns(program, access)
        query_native_sites(compact, access)
    assert materialize_hazards(compact.patterns) == full.obligations


@pytest.fixture(scope="module")
def checked_summaries():
    with limit(60):
        module = lower_source(scoped_query_recurrence)
        _plan_scope_parent(extract_scopes(module))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        full = summarize_queries(checked.program, checked.access)
        compact = summarize_query_patterns(checked.program, checked.access)
    return replace(checked, query_summary=full), replace(checked, query_summary=compact), parent


def test_compact_checks_retain_physical_witnesses_and_identical_complete_span_queries(
    checked_summaries, monkeypatch
):
    checked_full, checked_compact, parent = checked_summaries
    anchored = parent.anchored()
    covered_many, batches = anchored.covered_many, []

    def covered(queries):
        queries = tuple(queries)
        batches.append(queries)
        return covered_many(queries)

    monkeypatch.setattr(anchored, "covered_many", covered)
    with limit(60):
        full = prepare_memory_checks(checked_full, parent)
        compact = prepare_memory_checks(checked_compact, parent)
    assert batches[0] == batches[1]
    assert tuple((check.source, check.target, check.domain) for check in compact) == tuple(
        (check.source, check.target, check.domain) for check in full
    )
    for first, second in zip(full, compact, strict=True):
        assert second.patterns and second.obligations == ()
        assert materialize_hazards(second.patterns) == first.obligations
        assert memory_check_reason(second, checked_compact.access) == memory_check_reason(
            first, checked_full.access
        )
    with limit(60):
        completion = QueryCompletion(parent, copies=len(checked_compact.query_summary.parts))
        assert check_memory(compact, completion) == check_memory(full, completion)


def _changed_pattern(summary, *, storage=None, spans=None):
    pattern = summary.patterns[0]
    coverage = pattern.coverage[0]
    ranges = coverage.ranges[0]
    ranges = replace(
        ranges,
        storage=ranges.storage if storage is None else storage,
        spans=ranges.spans if spans is None else spans,
    )
    coverage = replace(coverage, ranges=(ranges,))
    return replace(pattern, coverage=(coverage,))


def test_compact_preparation_validates_every_physical_root_and_range(checked_summaries):
    _, checked, parent = checked_summaries
    summary = checked.query_summary
    bad = _changed_pattern(summary, storage=object())
    with pytest.raises(CompileError, match="actual allocation identity"):
        prepare_memory_checks(
            replace(checked, query_summary=replace(summary, patterns=(bad,))), parent
        )
    bad = _changed_pattern(summary, spans=((-1, 8, 0),))
    with pytest.raises(CompileError, match="invalid physical byte coverage"):
        prepare_memory_checks(
            replace(checked, query_summary=replace(summary, patterns=(bad,))), parent
        )


def test_compact_checks_and_diagnostics_never_expand_obligations(checked_summaries, monkeypatch):
    _, checked, parent = checked_summaries

    def forbidden(*args, **kwargs):
        raise AssertionError("query checks expanded physical obligations")

    monkeypatch.setattr(region_sequence, "Obligation", forbidden)
    with limit(60):
        checks = prepare_memory_checks(checked, parent)
        reasons = tuple(memory_check_reason(check, checked.access) for check in checks)
        completion = QueryCompletion(parent, copies=len(checked.query_summary.parts))
        answers = check_memory(checks, completion)
    assert reasons and len(answers) == len(checks)
    assert all("[" in reason and "unknown allocation" not in reason for reason in reasons)


def test_compact_final_memory_check_rejects_alien_physical_witnesses(checked_summaries):
    _, checked, parent = checked_summaries
    with limit(60):
        checks = prepare_memory_checks(checked, parent)
    bad = _changed_pattern(checked.query_summary, storage=object())
    check = replace(checks[0], patterns=(*checks[0].patterns, bad))
    completion = QueryCompletion(parent, copies=len(checked.query_summary.parts))
    with pytest.raises(CompileError, match="actual allocation identity"):
        check_memory((check,), completion)

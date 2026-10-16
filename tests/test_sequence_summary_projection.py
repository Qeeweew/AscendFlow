"""Output projection preserves full sequence semantics before materialization."""

from dataclasses import replace

import pytest

from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_region_sequence import check, finite, occurrences, part, source
from tests.test_region_sequence_sharing import repeated
from xdsl_npu import region_sequence
from xdsl_npu.completion import CountDomain
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.region_sequence import summarize_sequence
from xdsl_npu.region_summary import ReadSpan
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_summary import summarize_queries
from xdsl_npu.scope_sync import _scope_access


@pytest.fixture(scope="module")
def history():
    loop = source()
    return (
        part(loop, "0", {"init"}, 1),
        part(loop, "1", {"read_v", "read_s", "read_dma"}),
        part(loop, "2", {"overwrite"}, 1),
        part(loop, "3", {"read_dma"}, 1),
    )


@pytest.mark.parametrize("materialize_cells", (True, False))
@pytest.mark.parametrize("sources", (None, frozenset(), frozenset({"0"}), frozenset({"1", "2"})))
def test_projection_equals_full_output_and_preserves_transfer_work(
    history, materialize_cells, sources
):
    parts = tuple(part for part, _ in history)
    full_stats, projected_stats = {}, {}
    full = summarize_sequence(parts, statistics=full_stats)
    projected = summarize_sequence(
        parts,
        materialize_cells=materialize_cells,
        obligation_source_regions=sources,
        statistics=projected_stats,
    )
    expected = tuple(
        obligation
        for obligation in full.obligations
        if sources is None or obligation.source.region in sources
    )
    assert projected.obligations == expected
    assert projected.cells == (full.cells if materialize_cells else ())
    assert projected.bound == full.bound and projected.bound_value is full.bound_value
    for key in (
        "input_cells",
        "input_pattern_visits",
        "physical_intervals",
        "unique_cell_patterns",
        "unique_entry_patterns",
        "reused_intervals",
        "transfer_visits",
    ):
        assert projected_stats[key] == full_stats[key]
    assert full_stats["omitted_cells"] == full_stats["omitted_obligations"] == 0
    assert projected_stats["omitted_cells"] == (0 if materialize_cells else len(full.cells))
    assert projected_stats["omitted_obligations"] == len(full.obligations) - len(expected)
    assert (
        full.visits - projected.visits
        == full_stats["materialized_visits"] - projected_stats["materialized_visits"]
    )
    if materialize_cells and sources is None:
        assert projected == full


@pytest.mark.parametrize("count", (0, 1, 2, 5))
def test_projection_matches_independent_byte_history_and_keeps_reader_families(history, count):
    parts, tasks = zip(*history, strict=True)
    check(history, count)
    _, conflicts = finite(parts, tasks, count)
    projected = summarize_sequence(
        parts, materialize_cells=False, obligation_source_regions={"0", "1"}
    )
    actual = set()
    for obligation in projected.obligations:
        if obligation.domain.contains(count):
            for byte in range(obligation.start, obligation.stop):
                for first in occurrences(obligation.source, count):
                    for last in occurrences(obligation.destination, count):
                        actual.add((byte, obligation.hazard, first, last))
    assert actual == {edge for edge in conflicts if edge[2][0] in {"0", "1"}}
    assert not any(obligation.destination.region == "3" for obligation in projected.obligations)
    if count:
        war = [
            obligation
            for obligation in projected.obligations
            if obligation.hazard == "WAR" and obligation.domain.contains(count)
        ]
        assert {obligation.source.access.task[1] for obligation in war} == {
            "read_v",
            "read_s",
            "read_dma",
        }
        assert all(isinstance(obligation.source.access, ReadSpan) for obligation in war)


def test_projection_avoids_constructors_and_preserves_sparse_physical_obligations(monkeypatch):
    parts = repeated(3)
    stats, full = {}, summarize_sequence(parts)
    constructors = {"cells": region_sequence.Cell, "obligations": region_sequence.Obligation}
    built = {kind: 0 for kind in constructors}

    def count(kind):
        def construct(*args, **kwargs):
            built[kind] += 1
            return constructors[kind](*args, **kwargs)

        return construct

    monkeypatch.setattr(region_sequence, "Cell", count("cells"))
    monkeypatch.setattr(region_sequence, "Obligation", count("obligations"))
    selected = summarize_sequence(
        parts, materialize_cells=False, obligation_source_regions={"read"}, statistics=stats
    )
    expected = tuple(
        obligation for obligation in full.obligations if obligation.source.region == "read"
    )
    assert selected.obligations == expected and not selected.cells
    assert built == {"cells": 0, "obligations": len(expected)}
    assert stats["omitted_cells"] == len(full.cells) > 0
    assert stats["omitted_obligations"] == len(full.obligations) - built["obligations"] > 0
    assert all(
        obligation.start % 16 == 0 and obligation.stop - obligation.start == 8
        for obligation in selected.obligations
    )
    assert all(
        isinstance(obligation.source.access, ReadSpan) for obligation in selected.obligations
    )
    assert {obligation.storage for obligation in selected.obligations} == {
        cell.storage for cell in full.cells
    }


def test_empty_projection_still_rejects_overlapping_domains():
    reader = repeated(1)[1]
    cell = reader.summary.cells[0]
    overlap = replace(cell, domain=CountDomain(2, 0, minimum=2))
    bad = replace(reader, summary=replace(reader.summary, cells=(cell, overlap)))
    with pytest.raises(CompileError, match="overlapping child byte/count domains"):
        summarize_sequence((bad,), materialize_cells=False, obligation_source_regions=())


def test_query_summary_projects_only_unused_sequence_outputs():
    program = extract_scopes(lower_source(scoped_query_recurrence))
    access = _scope_access(program)
    query = summarize_queries(program, access)
    full_stats = {}
    full = summarize_sequence(query.parts, statistics=full_stats)
    assert query.obligations == tuple(
        obligation for obligation in full.obligations if obligation.source.region == "0"
    )
    assert query.cells and all(part.summary.cells for part in query.parts)
    stats = query.statistics
    assert stats["sequence_materialized_cells"] == 0
    assert stats["sequence_materialized_obligations"] == len(query.obligations)
    assert stats["sequence_omitted_cells"] == len(full.cells) > 0
    assert (
        stats["sequence_omitted_obligations"] == len(full.obligations) - len(query.obligations) > 0
    )
    assert stats["sequence_transfer_visits"] == full_stats["transfer_visits"]

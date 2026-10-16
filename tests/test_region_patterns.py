"""Shared words retain exact byte histories without physical semantic objects."""

from dataclasses import replace
from itertools import product
from random import Random

import pytest

from tests.test_region_sequence import finite, occurrences, part, source
from tests.test_region_sequence_sharing import legacy_sequence, repeated
from tests.test_scope_boundary_summary import FIRST, LAST, _expanded, _literal, effect
from xdsl_npu import region_patterns, region_sequence
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.model import CompileError
from xdsl_npu.region_patterns import (
    PatternPart,
    PatternRegion,
    group_frontiers,
    materialize_frontiers,
    materialize_hazards,
)
from xdsl_npu.region_sequence import summarize_sequence_patterns
from xdsl_npu.region_summary import Access, DefinitionRun, ReadBinding
from xdsl_npu.scope_boundary_summary import summarize_frontier_patterns, summarize_frontiers


def compact_part(part):
    summary = part.summary
    return PatternPart(
        part.name,
        PatternRegion(
            summary.roots,
            group_frontiers(summary.cells),
            summary.visits,
            summary.bound,
            summary.constant_count,
            summary.origins,
            summary.bound_value,
        ),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("compact path constructed a physical semantic record")


@pytest.mark.parametrize("pitch", (4, 13))
def test_boundary_words_are_built_before_physical_cases_and_retain_full_order(pitch, monkeypatch):
    word = (effect("write", "W", 0, FIRST), effect("read", "R", 1), effect("tail", "R", 2, LAST))
    records = tuple(
        replace(record, storage=root, start=row * pitch, stop=row * pitch + 4)
        for root in ("a", "b")
        for row in range(11)
        for record in word
    )
    full = summarize_frontiers(records)
    statistics, small_statistics = {}, {}
    with monkeypatch.context() as patch:
        patch.setattr(region_patterns, "CellSummary", forbidden)
        groups = summarize_frontier_patterns(records, statistics=statistics)
        summarize_frontier_patterns(word, statistics=small_statistics)
    assert len(groups) == 1 and len(groups[0].ranges) == 2
    assert all(len(ranges.spans) == 11 for ranges in groups[0].ranges)
    assert statistics["materialized_cells"] == 0
    assert statistics["summaries"] == len(full)
    assert statistics["shared_frontier_cases"] == small_statistics["shared_frontier_cases"]
    assert statistics["transfer_visits"] == small_statistics["transfer_visits"]
    assert materialize_frontiers(groups) == full
    for count in (0, 1, 2, 3, 9):
        assert _expanded(materialize_frontiers(groups), count) == _literal(records, count)


@pytest.mark.parametrize("pieces", (1, 3, 19))
def test_sequence_shares_hazards_and_range_collections_without_obligations(pieces, monkeypatch):
    parts = repeated(pieces)
    full = legacy_sequence(parts)
    statistics, single_statistics = {}, {}
    summarize_sequence_patterns(tuple(map(compact_part, repeated(1))), statistics=single_statistics)
    with monkeypatch.context() as patch:
        patch.setattr(region_sequence, "Cell", forbidden)
        patch.setattr(region_sequence, "Obligation", forbidden)
        compact = summarize_sequence_patterns(
            tuple(map(compact_part, parts)), statistics=statistics
        )
    expanded = compact.materialize()
    assert expanded.cells == full.cells
    assert expanded.obligations == full.obligations
    assert statistics["materialized_cells"] == statistics["materialized_obligations"] == 0
    assert statistics["hazard_patterns"] == single_statistics["hazard_patterns"]
    assert statistics["transfer_visits"] == single_statistics["transfer_visits"]
    assert statistics["input_physical_pieces"] == 6 * pieces
    assert sum(pattern.physical_count for pattern in compact.patterns) == len(full.obligations)
    assert (
        len({id(coverage.ranges) for pattern in compact.patterns for coverage in pattern.coverage})
        == 1
    )
    for pattern in compact.patterns:
        assert all(
            lo % 16 == 0 and hi - lo == 8
            for coverage in pattern.coverage
            for ranges in coverage.ranges
            for lo, hi, _ in ranges.pieces()
        )
    assert len({ranges.storage for group in compact.frontiers for ranges in group.ranges}) == 2


@pytest.mark.parametrize("seed", range(8))
def test_pattern_composition_matches_independent_finite_byte_histories(seed):
    loop, rng = source(), Random(seed)
    names = ("init", "update", "read_v", "read_s", "read_dma", "overwrite")
    history = [
        part(
            loop,
            str(index),
            set(rng.sample(names, rng.randrange(1, 6))),
            rng.choice((None, None, 0, 1, 3)),
        )
        for index in range(4)
    ]
    parts, tasks = zip(*history, strict=True)
    summary = summarize_sequence_patterns(tuple(map(compact_part, parts)))
    expanded = summary.materialize()
    reference = legacy_sequence(parts)
    assert expanded.cells == reference.cells and expanded.obligations == reference.obligations
    for count in (0, 1, 2, 3, 7):
        _, conflicts = finite(parts, tasks, count)
        observed = set()
        for hazard in materialize_hazards(summary.patterns):
            if hazard.domain.contains(count):
                for byte in range(hazard.start, hazard.stop):
                    for first, last in product(
                        occurrences(hazard.source, count), occurrences(hazard.destination, count)
                    ):
                        observed.add((byte, hazard.hazard, first, last))
        assert observed == conflicts


def test_historical_definition_read_bindings_are_part_of_word_identity():
    parts = repeated(1)
    original = parts[0]
    first = original.summary.cells[0]
    writer = first.first_write
    later = Access(writer.task, Position(False, 1))
    reader = Access(("aiv0", "read_old"), Position(False, 0))
    definition = DefinitionRun(writer, later, (), (ReadBinding(reader, writer),))
    changed = replace(definition, read_bindings=(ReadBinding(reader, later),))
    cells = (
        replace(first, definitions=(definition,)),
        replace(first, start=16, stop=24, definitions=(changed,)),
    )
    groups = group_frontiers(cells)
    assert len(groups) == 2
    assert materialize_frontiers(groups) == cells
    region = replace(original, summary=replace(original.summary, cells=cells))
    statistics = {}
    compact = summarize_sequence_patterns((compact_part(region),), statistics=statistics)
    assert statistics["unique_cell_patterns"] == statistics["unique_transfer_words"] == 2
    assert not compact.patterns
    assert groups[0].word[0].definitions[0].read_bindings[0].writer == writer


def test_source_projection_keeps_all_domain_validation_and_real_physical_witnesses():
    full_parts = repeated(3)
    parts = tuple(map(compact_part, full_parts))
    full = legacy_sequence(full_parts)
    summary = summarize_sequence_patterns(parts, obligation_source_regions={"read"})
    assert materialize_hazards(summary.patterns) == tuple(
        hazard for hazard in full.obligations if hazard.source.region == "read"
    )
    for pattern in summary.patterns:
        storage, lo, hi = pattern.first_range()
        assert any(
            o.storage is storage and o.start == lo and o.stop == hi and o.hazard == pattern.hazard
            for o in full.obligations
        )
    reader = full_parts[1]
    first = reader.summary.cells[0]
    overlap = replace(first, domain=CountDomain(2, 0, minimum=2))
    bad = replace(reader, summary=replace(reader.summary, cells=(first, overlap)))
    with pytest.raises(CompileError, match="overlapping child byte/count domains"):
        summarize_sequence_patterns((compact_part(bad),), obligation_source_regions=())
    duplicate = replace(reader, summary=replace(reader.summary, cells=(first, first)))
    with pytest.raises(CompileError, match="duplicate child byte/count cell"):
        summarize_sequence_patterns((compact_part(duplicate),), obligation_source_regions=())


def test_pattern_sweep_validates_original_bound_and_allocation_identities():
    parts = tuple(map(compact_part, repeated(1)))
    bad = replace(parts[1], summary=replace(parts[1].summary, origins=()))
    with pytest.raises(CompileError, match="original SSA allocation"):
        summarize_sequence_patterns((bad,))
    other = compact_part(repeated(1)[1])
    with pytest.raises(CompileError, match="shared immutable bound"):
        summarize_sequence_patterns((parts[1], replace(other, name="other")))

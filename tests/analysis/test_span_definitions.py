"""Incremental defined-byte unions retain holes, activation and real writers."""

from collections import Counter
from itertools import pairwise
from types import SimpleNamespace

from ascendflow import language as N
from ascendflow.analysis.structured_admission import (
    DefinitionSources,
    _covers,
    _DefinitionSpans,
    _Span,
    admit_structured_memory,
)
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import literal
from tests.sync.test_event_frontiers import sequential


def test_sequential_read_definitions_do_not_retransfer_all_prior_writers():
    for count in (16, 64, 256):
        structure = lower_native_structure(sequential(count))
        memory = collect_native_memory(structure.module)
        result = admit_structured_memory(structure, memory, admit_memory_bounds(memory))
        assert result.work["span_region_transfers"] == count
        assert result.work["coverage_checks"] <= 2 * count
        assert len(result.reads) == count
        # Numerical coalescing must retain the original writer identities.
        assert [len(read.writers) for read in result.reads] == list(range(1, count + 1))
        assert result.work["definition_source_members"] == count
        assert result.work["definition_source_nodes"] == count
        prefixes = [read.writers for read in result.reads]
        assert all(isinstance(p, DefinitionSources) and len(p.members) == 1 for p in prefixes)
        assert all(b.previous is a for a, b in pairwise(prefixes))
        assert tuple(prefixes[0]) == prefixes[0].members
        assert tuple(prefixes[-1]) == tuple(p.members[0] for p in prefixes)


def test_incremental_union_matches_independent_exact_byte_set_with_holes():
    frame = SimpleNamespace(domains=lambda: ())
    definitions, initialized = _DefinitionSpans(), set()
    # Out-of-order, overlapping, adjacent and pitched rows. The pitched holes
    # are deliberately interleaved with independent contiguous definitions.
    for base, width, rows, pitch in (
        (24, 8, 1, 0),
        (0, 8, 2, 32),
        (8, 8, 1, 0),
        (16, 8, 1, 0),
        (4, 8, 1, 0),
    ):
        definitions.add(_Span(literal(base), literal(width), rows, pitch), {})
        initialized.update(base + row * pitch + col for row in range(rows) for col in range(width))
        for start in range(0, 48, 4):
            for extent in (4, 8, 12, 32):
                expected = set(range(start, start + extent)) <= initialized
                assert (
                    _covers(
                        definitions.spans(),
                        _Span(literal(start), literal(extent)),
                        frame,
                        {},
                        Counter(),
                    )
                    == expected
                )


def test_complete_branch_sources_share_prior_snapshots_without_recopying_members():
    for count in (8, 32):
        body = "\n".join(
            f"        if flag==0:\n            N.fill(x[{i},:],1.0)\n"
            f"        else:\n            N.fill(x[{i},:],2.0)\n"
            f"        N.load_scalar(x[{i},:1])"
            for i in range(count)
        )
        source = N.from_source(
            f'def branches(flag:N.i64):\n    with N.Kernel(1,kind="vector"):\n'
            f"        x=N.alloc_ub(({count},64),N.f32)\n" + body + "\n",
            disjoint_external=True,
        )
        structure = lower_native_structure(source)
        memory = collect_native_memory(structure.module)
        result = admit_structured_memory(structure, memory, admit_memory_bounds(memory))
        prefixes = [read.writers for read in result.reads]
        assert len(prefixes) == count
        assert result.work["definition_source_members"] == 2 * count
        assert all(isinstance(p, DefinitionSources) and len(p.members) == 2 for p in prefixes)
        assert all(b.previous is a for a, b in pairwise(prefixes))
        assert [len(p) for p in prefixes] == [2 * i for i in range(1, count + 1)]
        assert tuple(prefixes[0]) == prefixes[0].members

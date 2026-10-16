"""Shared transfers retain literal byte histories and the unshared baseline."""

from dataclasses import replace
from itertools import pairwise
from random import Random

import pytest
from xdsl.ir import SSAValue

from tests.test_region_sequence import check, source
from tests.test_region_sequence import part as native_part
from xdsl_npu import region_sequence
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.budget import poll
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.model import CompileError, require
from xdsl_npu.region_sequence import (
    Cell,
    Endpoint,
    Obligation,
    Part,
    SequenceSummary,
    _domains,
    summarize_sequence,
)
from xdsl_npu.region_summary import Access, CellSummary, ReadSpan, RegionSummary
from xdsl_npu.scalar import from_ssa

MAX_SEQUENCE_VISITS = 1_000_000
FIRST, LAST = Position(False, 0), Position(True, 0)
ACTIVE = CountDomain(1, 0, minimum=1)
WRITE = Access(("aiv0", "write"), FIRST)
READ = ReadSpan(("aiv0", "read"), FIRST, LAST, 1)


def fixture():
    kernel = Kernel("sharing")
    roots = tuple(kernel.buffer(f"state{i}", "UB", 4096) for i in range(2))
    value = kernel.runtime_size("n")
    return roots, value, from_ssa(value)


def make_part(name, cells, roots, value, bound, count=None):
    names = tuple(f"state{i}" for i in range(len(roots)))
    return Part(
        name,
        RegionSummary(
            names,
            tuple(cells),
            0,
            bound if count is None else None,
            count,
            tuple(zip(names, roots, strict=True)),
            value if count is None else None,
        ),
    )


def repeated(pieces=4):
    roots, value, bound = fixture()
    cells = tuple(
        CellSummary(f"state{i}", j * 16, j * 16 + 8, ACTIVE, (), WRITE, WRITE, ())
        for i in range(2)
        for j in range(pieces)
    )
    return (
        make_part("init", cells, roots, value, bound, 1),
        make_part(
            "read",
            (
                replace(
                    c,
                    reads_before_write=(READ,),
                    first_write=None,
                    last_write=None,
                    pending_reads=(READ,),
                )
                for c in cells
            ),
            roots,
            value,
            bound,
        ),
        make_part("overwrite", cells, roots, value, bound, 1),
    )


def same_as_baseline(parts):
    actual = summarize_sequence(parts)
    expected = legacy_sequence(parts)
    assert actual.cells == expected.cells
    assert actual.obligations == expected.obligations
    return actual


def test_one_transfer_is_instantiated_on_each_root_and_each_side_of_holes():
    parts, statistics = repeated(), {"caller": "retained"}
    actual = summarize_sequence(parts, statistics=statistics)
    expected = same_as_baseline(parts)
    assert actual == expected
    assert statistics["caller"] == "retained"
    assert statistics["physical_intervals"] == 8
    assert statistics["unique_entry_patterns"] == 1
    assert statistics["reused_intervals"] == 7
    assert statistics["materialized_cells"] == len(actual.cells) == 16
    assert statistics["materialized_obligations"] == len(actual.obligations) == 32
    assert len({id(cell) for cell in actual.cells}) == len(actual.cells)
    assert len({id(o) for o in actual.obligations}) == len(actual.obligations)
    assert all(c.stop - c.start == 8 and c.start % 16 == 0 for c in actual.cells)
    small = {}
    summarize_sequence(repeated(1), statistics=small)
    assert small["transfer_visits"] == statistics["transfer_visits"]
    assert small["materialized_visits"] * 4 == statistics["materialized_visits"]
    assert statistics["visits"] == actual.visits


def test_guard_holes_preserve_complete_count_domains():
    parts = repeated(2)
    reader = parts[1]
    reader = replace(
        reader,
        summary=replace(
            reader.summary,
            cells=tuple(
                replace(c, domain=CountDomain(2, c.start // 16, minimum=2 + c.start // 16))
                for c in reader.summary.cells
            ),
        ),
    )
    actual = same_as_baseline((parts[0], reader, parts[2]))
    for count in range(7):
        raw = [o for o in actual.obligations if o.hazard == "RAW" and o.domain.contains(count)]
        assert {o.start for o in raw} == ({(count % 2) * 16} if count >= 2 else set())


def test_distinct_part_identities_and_reader_spans_never_share_a_frontier():
    roots, value, bound = fixture()
    a = CellSummary("state0", 0, 8, ACTIVE, (), WRITE, WRITE, ())
    b = replace(a, start=16, stop=24)
    short = ReadSpan(READ.task, FIRST, Position(True, 1), 2)
    parts = (
        make_part("first_part", (a,), roots, value, bound, 1),
        make_part("other_part", (b,), roots, value, bound, 1),
        make_part(
            "read",
            (
                replace(
                    a,
                    reads_before_write=(READ,),
                    first_write=None,
                    last_write=None,
                    pending_reads=(READ,),
                ),
                replace(
                    b,
                    reads_before_write=(short,),
                    first_write=None,
                    last_write=None,
                    pending_reads=(short,),
                ),
            ),
            roots,
            value,
            bound,
        ),
        make_part("after", (a, b), roots, value, bound, 1),
    )
    actual = same_as_baseline(parts)
    for obligation in actual.obligations:
        if obligation.hazard == "RAW":
            assert obligation.source.region == (
                "first_part" if obligation.start == 0 else "other_part"
            )
        if obligation.hazard == "WAR":
            assert obligation.source.access == (READ if obligation.start == 0 else short)


@pytest.mark.parametrize("count", [0, 1, 3, 1_000_000])
def test_fixed_children_use_their_own_count_and_zero_is_identity(count):
    parts = repeated(2)
    reader = replace(
        parts[1],
        summary=replace(
            parts[1].summary,
            constant_count=count,
            bound=None,
            bound_value=None,
            cells=tuple(
                replace(c, domain=CountDomain(1, 0, minimum=1, maximum=3))
                for c in parts[1].summary.cells
            ),
        ),
    )
    actual = same_as_baseline((parts[0], reader, parts[2]))
    raw = [o for o in actual.obligations if o.hazard == "RAW"]
    assert bool(raw) == (1 <= count <= 3)
    assert all(o.destination.constant_count == count for o in raw)


def test_pattern_multiplicity_cannot_hide_overlapping_alias_domains():
    parts = repeated(2)
    init = parts[0]
    original = init.summary.cells[:2]
    alias = replace(original[1], storage="alias")
    origins = (*init.summary.origins, ("alias", init.summary.origins[0][1]))
    bad = replace(
        init,
        summary=replace(
            init.summary,
            roots=(*init.summary.roots, "alias"),
            origins=origins,
            cells=(*original, alias),
        ),
    )
    with pytest.raises(CompileError, match="overlapping child byte/count domains"):
        summarize_sequence((bad,))


def test_reused_patterns_still_charge_every_materialized_obligation_and_cell(monkeypatch):
    parts, statistics = repeated(), {}
    expected = summarize_sequence(parts, statistics=statistics)
    assert statistics["materialized_visits"] >= len(expected.cells) + len(expected.obligations)
    with monkeypatch.context() as patch:
        patch.setattr(region_sequence, "MAX_SEQUENCE_VISITS", expected.visits - 1)
        with pytest.raises(CompileError, match="sequence output-piece budget exceeded"):
            summarize_sequence(parts)
    again = {}
    assert summarize_sequence(parts, statistics=again) == expected
    assert again["unique_entry_patterns"] == 1 and again["transfer_visits"] > 0


def test_old_algorithm_differential_and_independent_finite_byte_histories():
    loop = source()
    names = ("init", "update", "read_v", "read_s", "read_dma", "overwrite")
    for seed in range(6):
        random = Random(seed)
        sequence = [
            native_part(
                loop,
                f"part{i}",
                set(random.sample(names, random.randrange(1, 6))),
                random.choice((None, None, 0, 1, 4)),
            )
            for i in range(4)
        ]
        same_as_baseline(tuple(p for p, _ in sequence))
        for count in (0, 1, 2, 3, 7):
            check(sequence, count)


# Frozen unshared algorithm from 7966bce714f8810e3910c733fba035a30cb9b2a4.
# Kept only as a differential oracle; literal byte histories above independently
# check semantics without either production domain/transfer implementation.
def legacy_sequence(parts):
    """Return exact sequence frontiers and unresolved cross-region hazards.

    Empty dynamic children contribute identity; surrounding constant-count
    regions still execute. Read-only children preserve complete reader spans,
    without assuming their last issued read retires the previous ones.
    """
    parts = tuple(parts)
    require(
        len({part.name for part in parts}) == len(parts), "sequence region names must be unique"
    )
    require(all(part.name for part in parts), "sequence regions need nonempty identities")
    bounds = {part.summary.bound for part in parts if part.summary.constant_count is None}
    bound_values = {
        part.summary.bound_value for part in parts if part.summary.constant_count is None
    }
    require(
        None not in bounds
        and len(bounds) <= 1
        and (len(bound_values) == len(bounds))
        and all(isinstance(value, SSAValue) for value in bound_values),
        "sequence needs one shared immutable bound with its actual SSA identity",
    )
    bound = next(iter(bounds)) if bounds else None
    bound_value = next(iter(bound_values)) if bound_values else None
    roots = {}
    input_cells = 0
    for index, part in enumerate(parts):
        require(
            input_cells + len(part.summary.cells) <= MAX_SEQUENCE_VISITS,
            "sequence input-piece budget exceeded",
        )
        origins = dict(part.summary.origins)
        require(
            len(origins) == len(part.summary.origins)
            and set(origins) == set(part.summary.roots)
            and all(isinstance(root, SSAValue) for root in origins.values()),
            "sequence requires original SSA allocation identities",
        )
        require(
            len(set(part.summary.cells)) == len(part.summary.cells),
            "duplicate child byte/count cell",
        )
        for cell in part.summary.cells:
            require(
                cell.storage in origins and 0 <= cell.start < cell.stop, "invalid child byte cell"
            )
            root = origins[cell.storage]
            events = roots.setdefault(root, {})
            events.setdefault(cell.start, []).append((True, index, cell))
            events.setdefault(cell.stop, []).append((False, index, cell))
            input_cells += 1
            require(input_cells <= MAX_SEQUENCE_VISITS, "sequence input-piece budget exceeded")
            if input_cells % 256 == 0:
                poll("sequential access indexing", pieces=input_cells)
    output, obligations, visits = ([], [], 0)
    for root, events in roots.items():
        active = {}
        points = sorted(events)
        for point, stop in pairwise(points):
            for add, index, cell in events[point]:
                if add:
                    active.setdefault(index, set()).add(cell)
                else:
                    active[index].remove(cell)
                    if not active[index]:
                        del active[index]
            if not active:
                continue
            entries = [(parts[i], tuple(active[i])) for i in sorted(active)]
            for domain, count in _domains(entries):
                incoming, pending = ([], [])
                first, last = (None, None)

                def add(
                    hazard, source, destination, *, root=root, point=point, stop=stop, domain=domain
                ):
                    nonlocal visits
                    visits += 1
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence obligation budget exceeded")
                    obligations.append(
                        Obligation(root, point, stop, domain, hazard, source, destination)
                    )

                for part, cells in entries:
                    visits += len(cells)
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence frontier budget exceeded")
                    poll(
                        "sequential access composition", visits=visits, obligations=len(obligations)
                    )
                    local_count = part.summary.constant_count
                    candidates = [
                        cell
                        for cell in cells
                        if cell.domain.contains(count if local_count is None else local_count)
                    ]
                    require(len(candidates) <= 1, "overlapping child byte/count domains")
                    if not candidates:
                        continue
                    cell = candidates[0]
                    visits += len(cell.reads_before_write) + len(cell.pending_reads) + 2
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence endpoint budget exceeded")

                    def endpoint(access, *, part=part, local_count=local_count):
                        return Endpoint(part.name, access, local_count)

                    readers = tuple(endpoint(reader) for reader in cell.reads_before_write)
                    if last is not None:
                        for reader in readers:
                            add("RAW", last, reader)
                    else:
                        incoming.extend(readers)
                    if cell.first_write is not None:
                        writer = endpoint(cell.first_write)
                        if last is not None:
                            add("WAW", last, writer)
                        for reader in pending:
                            add("WAR", reader, writer)
                        require(cell.last_write is not None, "first writer needs an exit writer")
                        first = first or writer
                        last = endpoint(cell.last_write)
                        pending = [endpoint(reader) for reader in cell.pending_reads]
                    else:
                        require(cell.last_write is None, "exit writer needs a first writer")
                        pending.extend(endpoint(reader) for reader in cell.pending_reads)
                if incoming or first is not None or pending:
                    visits += len(incoming) + len(pending) + 1
                    require(visits <= MAX_SEQUENCE_VISITS, "sequence output-piece budget exceeded")
                    output.append(
                        Cell(
                            root, point, stop, domain, tuple(incoming), first, last, tuple(pending)
                        )
                    )
    return SequenceSummary(bound, bound_value, tuple(output), tuple(obligations), visits)

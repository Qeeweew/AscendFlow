"""Conditional memory transfers against complete, independent byte scans."""

from dataclasses import replace
from random import Random

import pytest

from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.model import CompileError
from xdsl_npu.region_summary import Access, CellSummary, ReadSpan
from xdsl_npu.scope_boundary_summary import summarize_frontiers
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_access import BoundaryEffect
from xdsl_npu.scope_predicates import case_activation

FIRST = Activation(CountDomain(1, 0, minimum=1), Position(False, 0), Position(False, 0))
LAST = Activation(CountDomain(1, 0, minimum=1), Position(True, 0), Position(True, 0))
MIDDLE = case_activation("middle")
ALL = Activation()


def effect(name, modes, rank, activation=ALL, *, root="state", start=0, stop=4, actor="aiv0"):
    return BoundaryEffect((actor, name), root, start, stop, modes, activation, (rank,))


def _literal(records, count):
    """Execute all actual active events byte by byte, with no summary helpers."""
    result = {}
    for generation in range(count):
        word = {}
        for record in records:
            active = record.activation
            if not (
                active.count.contains(count)
                and active.first.evaluate(count) <= generation <= active.last.evaluate(count)
                and generation % active.period in active.residues
            ):
                continue
            for byte in range(record.start, record.stop):
                key = record.logical_order, record.endpoint, record.storage, byte
                word.setdefault(key, set()).update(record.modes)
        for (_, endpoint, root, byte), modes in sorted(word.items()):
            incoming, first, last, pending = result.setdefault(
                (root, byte), (set(), None, None, set())
            )
            access = endpoint, generation
            if "R" in modes:
                pending.add(access)
                if first is None:
                    incoming.add(access)
            if "W" in modes:
                first = first or access
                last, pending = access, set()
            result[root, byte] = incoming, first, last, pending
    return result


def _access(access, count):
    return None if access is None else (access.task, access.position.evaluate(count))


def _reads(spans, count):
    result = set()
    for span in spans:
        assert isinstance(span, ReadSpan) and span.step > 0
        first, last = span.first.evaluate(count), span.last.evaluate(count)
        assert 0 <= first <= last < count
        result.update((span.task, generation) for generation in range(first, last + 1, span.step))
    return result


def _expanded(summaries, count):
    result = {}
    for cell in summaries:
        assert isinstance(cell, CellSummary) and not cell.definitions
        assert not cell.domain.contains(0)
        if not cell.domain.contains(count):
            continue
        first, last = _access(cell.first_write, count), _access(cell.last_write, count)
        assert (first is None) == (last is None)
        frontier = (
            _reads(cell.reads_before_write, count),
            first,
            last,
            _reads(cell.pending_reads, count),
        )
        for byte in range(cell.start, cell.stop):
            key = cell.storage, byte
            assert key not in result, "overlapping summary count domains or physical cells"
            result[key] = frontier
    return result


def _check(records, counts=range(20)):
    statistics = {}
    summaries = summarize_frontiers(iter(records), statistics=statistics)
    for count in counts:
        assert _expanded(summaries, count) == _literal(records, count), count
    assert statistics["summaries"] == len(summaries)
    return summaries, statistics


def test_first_qa_write_preserves_every_middle_reader_in_the_pending_frontier():
    records = (
        effect("load_qa", "W", 0, FIRST, root="qa", actor="aic"),
        effect("mad_j0", "R", 1, root="qa", actor="aic"),
        effect("mad_j7", "R", 2, root="qa", actor="aic"),
    )
    summaries, _ = _check(records)
    long = next(cell for cell in summaries if cell.domain.contains(100))
    assert not long.reads_before_write
    assert long.first_write == long.last_write == Access(("aic", "load_qa"), Position(False, 0))
    for endpoint in (("aic", "mad_j0"), ("aic", "mad_j7")):
        assert any(
            span.task == endpoint
            and span.first == Position(False, 1)
            and span.last == Position(True, 1)
            and span.step == 1
            for span in long.pending_reads
        )
    assert _reads(long.pending_reads, 100) == {
        (endpoint, generation)
        for endpoint in (("aic", "mad_j0"), ("aic", "mad_j7"))
        for generation in range(100)
    }


def test_first_clear_every_kv_rw_and_last_store_keep_the_actual_extremal_writers():
    records = (
        effect("clear", "W", 0, FIRST),
        effect("mul", "RW", 1),
        effect("add", "RW", 2),
        effect("store", "R", 3, LAST),
    )
    summaries, _ = _check(records)
    long = next(cell for cell in summaries if cell.domain.contains(100))
    assert not long.reads_before_write
    assert long.first_write == Access(("aiv0", "clear"), Position(False, 0))
    assert long.last_write == Access(("aiv0", "add"), Position(True, 0))
    assert long.pending_reads == (ReadSpan.one(Access(("aiv0", "store"), Position(True, 0))),)


def test_first_only_q1_reader_remains_a_first_reader_not_a_tail_read():
    records = (effect("load_q1", "R", 0, FIRST, root="q1", actor="aic"),)
    summaries, _ = _check(records)
    for cell in summaries:
        assert cell.first_write is cell.last_write is None
        assert cell.reads_before_write == cell.pending_reads
        assert cell.pending_reads == (ReadSpan.one(Access(("aic", "load_q1"), Position(False, 0))),)


def test_missing_initializer_is_an_incoming_requirement_not_an_implicit_write():
    records = (effect("rw", "RW", 0), effect("late_clear", "W", 1, LAST))
    summaries, _ = _check(records)
    for cell in summaries:
        assert _reads(
            cell.reads_before_write, max(1, cell.domain.exact or cell.domain.minimum)
        ) == {(("aiv0", "rw"), 0)}


@pytest.mark.parametrize("period", (1, 2, 3, 5, 7))
def test_periodic_middle_writer_has_exact_first_last_and_finite_read_suffix(period):
    records = (
        effect("first_read", "R", 0, FIRST),
        effect("before", "R", 1),
        effect("update", "RW", 2, Activation(period=period, residues=(period - 1,))),
        effect("after", "R", 3),
        effect("last_read", "R", 4, LAST),
    )
    summaries, stats = _check(records, range(3 * period + 7))
    assert stats["max_period"] == period
    long = next(cell for cell in summaries if cell.domain.contains(100 * period))
    assert len(long.pending_reads) <= 2 * period + 2
    assert _expanded((long,), 100 * period) == _literal(records, 100 * period)


def test_only_middle_writer_uses_short_n_fallback_without_creating_a_writer():
    records = (
        effect("head", "R", 0, FIRST),
        effect("middle_write", "W", 1, replace(MIDDLE, period=3, residues=(0,))),
        effect("tail", "R", 2, LAST),
    )
    summaries, _ = _check(records)
    for count in (1, 2, 3, 4):
        assert all(cell.first_write is None for cell in summaries if cell.domain.contains(count))
    assert any(cell.first_write is not None for cell in summaries if cell.domain.contains(5))


def test_nonwriter_middle_keeps_complete_periodic_spans_until_the_last_overwrite():
    records = (
        effect("head", "R", 0, FIRST),
        effect("even_reader", "R", 1, Activation(period=2, residues=(0,))),
        effect("other_reader", "R", 2, Activation(period=3, residues=(1,))),
        effect("tail_writer", "W", 3, LAST),
    )
    summaries, stats = _check(records, range(27))
    assert stats["max_period"] == 6
    long = next(cell for cell in summaries if cell.domain.contains(600))
    assert not long.pending_reads
    assert any(span.first != span.last and span.step == 6 for span in long.reads_before_write)
    assert _expanded((long,), 600) == _literal(records, 600)


def test_single_and_first_last_case_words_overlap_only_at_their_real_counts():
    records = (
        effect("single_reader", "R", 0, case_activation("single")),
        effect("head_write", "W", 1, case_activation("first")),
        effect("middle_rw", "RW", 2, MIDDLE),
        effect("tail_read", "R", 3, case_activation("last")),
    )
    summaries, _ = _check(records)
    assert _expanded(summaries, 1)["state", 0] == (
        {(("aiv0", "single_reader"), 0)},
        None,
        None,
        {(("aiv0", "single_reader"), 0)},
    )


def test_count_predicates_and_sparse_physical_cells_remain_separate():
    counts = CountDomain(3, 1, minimum=4, maximum=19)
    records = (
        effect("init", "W", 0, replace(FIRST, count=counts), start=0, stop=3),
        effect("rw", "RW", 1, Activation(counts, period=2, residues=(1,)), start=1, stop=3),
        effect("tail", "R", 2, replace(LAST, count=counts), start=0, stop=3),
        effect("hole_reader", "R", 3, root="state", start=7, stop=9),
        effect("other_actor", "R", 0, root="aiv1/state", actor="aiv1", start=0, stop=2),
    )
    summaries, stats = _check(records, range(28))
    assert stats["max_period"] == 6
    assert not any(
        cell.storage == "state" and cell.start < 7 and cell.stop > 3 for cell in summaries
    )
    assert {cell.storage for cell in summaries} == {"state", "aiv1/state"}


@pytest.mark.parametrize("pitch", (4, 13))
def test_repeated_rows_share_transfers_and_preserve_every_root_and_byte_cell(pitch):
    word = (
        effect("init", "W", 0, FIRST),
        effect("periodic_read", "R", 1, Activation(period=3, residues=(1,))),
        effect("tail", "R", 2, LAST),
    )
    _, single_stats = _check(word)
    physical = tuple(
        (root, row * pitch, row * pitch + 4)
        for root in ("state", "second_allocation")
        for row in range(9)
    )
    records = tuple(
        replace(record, storage=root, start=lo, stop=hi)
        for root, lo, hi in physical
        for record in word
    )
    summaries, stats = _check(records)
    assert {(cell.storage, cell.start, cell.stop) for cell in summaries} == set(physical)
    assert stats["cells"] == len(physical)
    assert stats["unique_transfer_words"] == 1
    assert stats["reused_cells"] == len(physical) - 1
    assert stats["transfer_visits"] == single_stats["transfer_visits"] > 0
    assert stats["short_domains"] == single_stats["short_domains"] * len(physical)
    assert stats["long_domains"] == single_stats["long_domains"] * len(physical)
    assert stats["visits"] > single_stats["visits"]
    active = [cell for cell in summaries if cell.domain.contains(17)]
    assert len(active) == len(physical)
    assert active[0].pending_reads
    assert all(cell.pending_reads is active[0].pending_reads for cell in active)
    assert all(cell.first_write is active[0].first_write for cell in active)


@pytest.mark.parametrize(
    "change",
    (
        {"endpoint": ("aiv0", "different_inner_native")},
        {"modes": "W"},
        {"activation": LAST},
        {"activation": Activation(CountDomain(2, 0, minimum=2))},
        {"activation": Activation(period=3, residues=(1,))},
        {"logical_order": (-1,)},
    ),
)
def test_each_semantic_effect_field_separates_transfer_sharing(change):
    word = (effect("init", "W", 0, FIRST), effect("read", "R", 1))
    other = tuple(replace(record, start=8, stop=12) for record in word)
    other = (other[0], replace(other[1], **change))
    summaries, stats = _check((*word, *other))
    assert stats["cells"] == stats["unique_transfer_words"] == 2
    assert stats["reused_cells"] == 0
    assert any(
        _expanded(summaries, count)["state", 0] != _expanded(summaries, count)["state", 8]
        for count in range(1, 10)
    )


def test_transfer_sharing_never_survives_an_actual_effect_mutation_between_calls():
    records = [effect("init", "W", 0, FIRST), effect("read", "R", 1)]
    original = tuple(records)
    before, before_stats = _check(records)
    records[1] = replace(records[1], endpoint=("aiv0", "changed_native"))
    after, after_stats = _check(records)
    assert before_stats["unique_transfer_words"] == after_stats["unique_transfer_words"] == 1
    assert before_stats["transfer_visits"] == after_stats["transfer_visits"] > 0
    assert before_stats["reused_cells"] == after_stats["reused_cells"] == 0
    assert _expanded(before, 7) == _literal(original, 7)
    assert _expanded(before, 7) != _expanded(after, 7)


@pytest.mark.parametrize("seed", range(12))
def test_complete_random_conditional_sequences_match_independent_byte_scan(seed):
    rng = Random(9517 + seed)
    records = []
    for index in range(10):
        geometry = rng.randrange(6)
        first, last, minimum = (
            (Position(False, 0), Position(True, 0), 1),
            (Position(False, 0), Position(False, 0), 1),
            (Position(True, 0), Position(True, 0), 1),
            (Position(False, 1), Position(True, 1), 3),
            (Position(False, 0), Position(True, 1), 2),
            (Position(False, 1), Position(True, 0), 2),
        )[geometry]
        count_period = rng.choice((1, 2, 3))
        minimum += rng.randrange(3)
        domain = CountDomain(
            count_period,
            rng.randrange(count_period),
            minimum=minimum,
            maximum=minimum + 8 if index % 3 else None,
        )
        period = rng.choice((1, 2, 3))
        residues = tuple(residue for residue in range(period) if rng.randrange(2)) or (0,)
        start = rng.randrange(8)
        records.append(
            effect(
                f"native_inner_{index}",
                rng.choice(("R", "W", "RW")),
                rng.randrange(4),
                Activation(domain, first, last, period, residues),
                root=rng.choice(("qa", "state")),
                start=start,
                stop=start + rng.randrange(1, 4),
            )
        )
    _check(records, range(24))


def test_actual_guarded_native_inner_effects_keep_their_endpoint_identity():
    from tests.test_scope_guarded_access import _guarded_module
    from xdsl_npu.scope_access import analyze_scope_access
    from xdsl_npu.scope_guarded_access import analyze_guarded_access
    from xdsl_npu.scope_model import extract_scopes

    program = extract_scopes(_guarded_module())
    access = analyze_guarded_access(program, base=analyze_scope_access(program, definitions=True))
    summaries, _ = _check(access.effects, range(9))
    endpoints = {effect.endpoint for effect in access.effects}
    assert endpoints <= access.endpoints.keys()
    retained = {
        span.task for cell in summaries for span in (*cell.reads_before_write, *cell.pending_reads)
    }
    retained.update(
        point.task
        for cell in summaries
        for point in (cell.first_write, cell.last_write)
        if point is not None
    )
    assert retained <= endpoints
    assert len({access.endpoints[key][1].position for key in retained}) > 1


def test_empty_input_and_zero_count_are_identity_transfers():
    statistics = {}
    assert summarize_frontiers((), statistics=statistics) == ()
    assert statistics["visits"] == statistics["cells"] == statistics["summaries"] == 0
    assert (
        statistics["unique_transfer_words"]
        == statistics["reused_cells"]
        == statistics["transfer_visits"]
        == 0
    )
    summaries, _ = _check((effect("read", "R", 0),))
    assert _expanded(summaries, 0) == {}


def test_first_last_only_readers_keep_both_real_instances_including_n_one():
    records = (effect("first", "R", 0, FIRST), effect("last", "R", 1, LAST))
    summaries, _ = _check(records)
    for count in (1, 2, 17):
        incoming, first, last, pending = _expanded(summaries, count)["state", 0]
        expected = {(("aiv0", "first"), 0), (("aiv0", "last"), count - 1)}
        assert incoming == pending == expected
        assert first is last is None


def test_no_active_middle_events_leave_short_counts_as_identity():
    summaries, _ = _check((effect("middle", "W", 0, MIDDLE),))
    assert not any(cell.domain.contains(1) or cell.domain.contains(2) for cell in summaries)


@pytest.mark.parametrize(
    "words",
    (
        ((), ("R",), ("R",)),
        (("R",), ("W",), ("R", "R")),
        (("W",), ("R", "RW"), ("R",)),
        (("RW",), ("W",), ("R",)),
        (("R",), (), ("RW",)),
    ),
)
def test_transfer_composition_has_identity_and_associates_against_full_scan(words):
    from xdsl_npu.scope_boundary_summary import IDENTITY, _then, _Transfer

    records, transfers = [], []
    for modes in words:
        word = []
        for mode in modes:
            rank = len(records) + len(word)
            word.append(effect(f"event_{rank}", mode, rank))
        records.extend(word)
        cell = next((cell for cell in summarize_frontiers(word) if cell.domain.contains(1)), None)
        transfers.append(
            IDENTITY
            if cell is None
            else _Transfer(
                cell.reads_before_write, cell.first_write, cell.last_write, cell.pending_reads
            )
        )
    a, b, c = transfers
    combined = _then(_then(a, b), c)
    assert combined == _then(a, _then(b, c))
    assert _then(IDENTITY, combined) == _then(combined, IDENTITY) == combined
    assert (
        _reads(combined.reads_before_write, 1),
        _access(combined.first_write, 1),
        _access(combined.last_write, 1),
        _reads(combined.pending_reads, 1),
    ) == _literal(records, 1)["state", 0]


def test_huge_exact_count_never_controls_work_or_expands_reader_spans(monkeypatch):
    import xdsl_npu.scope_boundary_summary as summary
    import xdsl_npu.scope_guarded_access as guarded

    def bounded_range(*args):
        values = range(*args)
        assert len(values) < 128, "expanded the runtime parent count"
        return values

    monkeypatch.setattr(summary, "range", bounded_range, raising=False)
    monkeypatch.setattr(guarded, "range", bounded_range, raising=False)
    records = (
        effect("load", "W", 0, FIRST, root="qa"),
        effect("read", "R", 1, root="qa"),
        effect("update", "RW", 2, Activation(period=3, residues=(1,))),
        effect("tail", "R", 3, LAST),
    )
    sizes = []
    for count in (10**12, 1 << 60):
        exact = CountDomain(1, 0, exact=count)
        actual = tuple(
            replace(record, activation=replace(record.activation, count=exact))
            for record in records
        )
        stats = {}
        cells = summarize_frontiers(actual, statistics=stats)
        assert all(cell.domain.exact == count for cell in cells)
        assert stats["visits"] < 2048
        qa = next(cell for cell in cells if cell.storage == "qa")
        assert qa.last_write == Access(("aiv0", "load"), Position(False, 0))
        assert any(span.first != span.last for span in qa.pending_reads)
        sizes.append((len(cells), stats["visits"]))
    assert sizes[0] == sizes[1]


@pytest.mark.parametrize(
    "activation",
    (
        Activation(CountDomain(1, 0, minimum=3), Position(False, 2), Position(True, 0)),
        Activation(CountDomain(1, 0, minimum=3), Position(True, 1), Position(True, 1)),
    ),
)
def test_unknown_geometry_is_rejected_explicitly(activation):
    with pytest.raises(CompileError, match="first/interior/last domains only"):
        summarize_frontiers((effect("read", "R", 0, activation),))


def test_existing_static_work_budget_remains_enforced(monkeypatch):
    monkeypatch.setattr("xdsl_npu.scope_guarded_access.MAX_GUARDED_ACCESS_VISITS", 1)
    with pytest.raises(CompileError, match="guarded scope access visit budget"):
        summarize_frontiers((effect("read", "R", 0),))

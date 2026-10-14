"""Independent finite byte histories for sequential access-transfer composition.

These tests validate memory obligations, not asynchronous completion or codegen.
The finite oracle does not use the production frontier/domain/sweep algorithms.
"""

from dataclasses import replace

import pytest

from xdsl_npu import Kernel
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import CompileError
from xdsl_npu.region_sequence import Part, summarize_sequence
from xdsl_npu.region_summary import Access, summarize_reused_storage
from xdsl_npu.scalar import expression, literal


def source():
    k = Kernel("sequence", disjoint_external=True)
    state = k.buffer("state", "UB", 128)
    work = k.buffer("work", "UB", 64)
    output = k.buffer("output", "GM", 128, external=True)
    n = k.runtime_size("n")
    k.loop(n)
    full = k.view(state, 128)
    slot = k.view(state, 32, stride=64, slots=2)
    tmp = k.view(work, 32)
    gm = k.view(output, 32)
    word = k.view(state, 4)
    k.task("init", "fill", full, scalar=1)
    k.task("update", "adds", slot, slot, scalar=1)
    k.task("read_v", "muls", slot, tmp, scalar=2)
    k.task("read_s", "scalar_load", word)
    k.task("read_dma", "copy", slot, gm)
    k.task("overwrite", "fill", full, scalar=0)
    return extract_loop(k.finish())


def part(loop, name, selected, count=None):
    """Select physical native templates without changing allocation identity.

    This supplies analysis inputs only. It does not claim these separate regions
    have already been lowered or synchronized by the production compiler.
    """
    actor, data = next(iter(loop.actors.items()))
    tasks = tuple(t for t in data.program.tasks if t.uid in selected)
    data = replace(data, program=replace(data.program, tasks=tasks))
    leaf = replace(
        loop,
        actors={actor: data},
        constant_count=count,
        bound=loop.bound if count is None else None,
    )
    return Part(name, summarize_reused_storage(leaf, {actor + "/state"})), tasks


def occurrences(endpoint, count):
    length = count if endpoint.constant_count is None else endpoint.constant_count
    a = endpoint.access
    if isinstance(a, Access):
        return {(endpoint.region, a.position.evaluate(length), a.task[1])}
    return {
        (endpoint.region, g, a.task[1])
        for g in range(a.first.evaluate(length), a.last.evaluate(length) + 1, a.step)
    }


def finite(parts, tasks, count):
    # A per-byte history of all accesses in source order. Only cross-region
    # conflicts are returned: internal obligations belong to each leaf proof.
    states, conflicts = {}, set()
    for p, native in zip(parts, tasks):
        length = count if p.summary.constant_count is None else p.summary.constant_count
        for g in range(length):
            for t in native:
                if t.last and g != length - 1:
                    continue
                modes = {}
                for ref, mode in zip(t.refs, t.modes):
                    if ref.storage.name != "state":
                        continue
                    origin = sum(
                        o + s * ((g // d) % m if m else g // d) for o, s, m, d in ref.terms
                    )
                    for row in range(ref.rows):
                        for byte in range(
                            origin + row * ref.row_stride,
                            origin + row * ref.row_stride + ref.nbytes,
                        ):
                            modes.setdefault(byte, set()).update(mode)
                key = (p.name, g, t.uid)
                for byte, mode in modes.items():
                    first_reads, first_write, last_write, pending = states.setdefault(
                        byte, [set(), None, None, set()]
                    )
                    if "R" in mode:
                        if first_write is None:
                            first_reads.add(key)
                        if last_write is not None and last_write[0] != p.name:
                            conflicts.add((byte, "RAW", last_write, key))
                        pending.add(key)
                    if "W" in mode:
                        if last_write is not None and last_write[0] != p.name:
                            conflicts.add((byte, "WAW", last_write, key))
                        for reader in pending:
                            if reader[0] != p.name:
                                conflicts.add((byte, "WAR", reader, key))
                        states[byte] = [first_reads, first_write or key, key, set()]
    return states, conflicts


def check(parts_and_tasks, count):
    parts, tasks = zip(*parts_and_tasks)
    summary = summarize_sequence(parts)
    expected, conflicts = finite(parts, tasks, count)
    observed = {}
    for cell in summary.for_count(count):
        first = next(iter(occurrences(cell.first_write, count))) if cell.first_write else None
        last = next(iter(occurrences(cell.last_write, count))) if cell.last_write else None
        incoming = set().union(*(occurrences(r, count) for r in cell.reads_before_write))
        pending = set().union(*(occurrences(r, count) for r in cell.pending_reads))
        for byte in range(cell.start, cell.stop):
            assert byte not in observed
            observed[byte] = [incoming, first, last, pending]
    actual = set()
    for hazard in summary.obligations:
        if hazard.domain.contains(count):
            for byte in range(hazard.start, hazard.stop):
                for src in occurrences(hazard.source, count):
                    for dst in occurrences(hazard.destination, count):
                        actual.add((byte, hazard.hazard, src, dst))
    assert observed == expected
    assert actual == conflicts
    return summary


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 17])
def test_init_child_use_and_zero_identity(count):
    loop = source()
    summary = check(
        [
            part(loop, "init", {"init"}, 1),
            part(loop, "child", {"update"}),
            part(loop, "use", {"read_dma"}, 1),
        ],
        count,
    )
    assert not any(cell.reads_before_write for cell in summary.for_count(count))
    if count == 0:
        assert {o.source.region for o in summary.obligations if o.domain.contains(0)} == {"init"}


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 17])
def test_read_families_from_multiple_pipes_remain_live_until_overwrite(count):
    loop = source()
    summary = check(
        [
            part(loop, "before", {"init"}, 1),
            part(loop, "readers", {"read_v", "read_s", "read_dma"}),
            part(loop, "after", {"overwrite"}, 1),
        ],
        count,
    )
    if count > 1:
        war = [o for o in summary.obligations if o.hazard == "WAR" and o.domain.contains(count)]
        assert {o.source.access.task[1] for o in war} == {"read_v", "read_s", "read_dma"}
        scalar_reads = set().union(
            *(occurrences(o.source, count) for o in war if o.source.access.task[1] == "read_s")
        )
        assert len(scalar_reads) == count


def test_uninitialized_entry_stays_open_without_fabricating_a_writer():
    loop = source()
    summary = check(
        [
            part(loop, "child", {"update"}),
            part(loop, "use", {"read_dma"}, 1),
        ],
        0,
    )
    assert summary.for_count(0)[0].last_write is None
    assert summary.for_count(0)[0].reads_before_write
    summary = check([part(loop, "child", {"update"}), part(loop, "use", {"read_dma"}, 1)], 7)
    assert any(cell.reads_before_write for cell in summary.for_count(7))


@pytest.mark.parametrize("count", [0, 1, 2, 7])
def test_siblings_have_distinct_generations_and_preserve_physical_holes(count):
    loop = source()
    summary = check(
        [
            part(loop, "left", {"update", "read_v"}),
            part(loop, "right", {"update", "read_dma"}),
        ],
        count,
    )
    assert not any(cell.start < 64 and cell.stop > 32 for cell in summary.cells)
    assert not any(cell.start < 128 and cell.stop > 96 for cell in summary.cells)
    assert all(
        o.source.region == "left" and o.destination.region == "right" for o in summary.obligations
    )


def test_allocation_identity_is_not_a_buffer_name():
    a, b = source(), source()
    left, _ = part(a, "left", {"init"}, 1)
    right, _ = part(b, "right", {"read_dma"}, 1)
    summary = summarize_sequence([left, right])
    assert not summary.obligations
    assert len({cell.storage for cell in summary.cells}) == 2
    assert any(cell.reads_before_write for cell in summary.cells)


def test_constant_million_child_is_not_enumerated_and_uses_its_own_bound():
    loop = source()
    left, _ = part(loop, "init", {"init"}, 1)
    middle, _ = part(loop, "child", {"read_dma"}, 1_000_000)
    right, _ = part(loop, "after", {"overwrite"}, 1)
    result = summarize_sequence([left, middle, right])
    assert result.visits < 100
    assert len(result.cells) < 10
    spans = [o.source for o in result.obligations if o.hazard == "WAR"]
    assert spans and all(s.constant_count == 1_000_000 for s in spans)
    assert all(not isinstance(s.access, Access) for s in spans)


def test_identity_and_incompatible_domains_are_explicit():
    assert not summarize_sequence([]).cells
    loop = source()
    a, _ = part(loop, "a", {"read_dma"})
    b, _ = part(loop, "b", {"overwrite"})
    with pytest.raises(CompileError, match="unique"):
        summarize_sequence([a, a])
    b = replace(b, summary=replace(b.summary, bound=expression("add", b.summary.bound, literal(1))))
    with pytest.raises(CompileError, match="shared immutable bound"):
        summarize_sequence([a, b])
    with pytest.raises(CompileError, match="SSA allocation"):
        summarize_sequence([replace(a, summary=replace(a.summary, origins=()))])


def test_composed_frontiers_against_random_finite_region_histories():
    import random

    loop = source()
    names = ("init", "update", "read_v", "read_s", "read_dma", "overwrite")
    for seed in range(16):
        rng = random.Random(seed)
        sequence = [
            part(
                loop,
                f"region_{i}",
                set(rng.sample(names, rng.randrange(1, 6))),
                rng.choice((None, None, 0, 1, 3)),
            )
            for i in range(4)
        ]
        for count in (0, 1, 2, 3, 7):
            check(sequence, count)


def test_terminal_writer_does_not_erase_initialization_or_historical_reads():
    loop = source()
    actor, data = next(iter(loop.actors.items()))
    native = tuple(
        replace(task, last=True) if task.uid == "overwrite" else task for task in data.program.tasks
    )
    data = replace(data, program=replace(data.program, tasks=native))
    loop = replace(loop, actors={actor: data})
    parts = [part(loop, "child", {"read_s", "overwrite"}), part(loop, "use", {"read_dma"}, 1)]
    for count in (0, 1, 2, 7):
        summary = check(parts, count)
        assert any(cell.reads_before_write for cell in summary.for_count(count))


def test_sequence_budget_and_cancellation_do_not_modify_input(monkeypatch):
    from xdsl_npu import budget, region_sequence

    loop = source()
    a, _ = part(loop, "a", {"init"}, 1)
    b, _ = part(loop, "b", {"read_dma"})
    expected = summarize_sequence([a, b])
    with monkeypatch.context() as patch:
        patch.setattr(region_sequence, "MAX_SEQUENCE_VISITS", 8)
        with pytest.raises(CompileError, match="sequence.*budget exceeded"):
            summarize_sequence([a, b])
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 3

    with (
        budget.limit(cancel=cancel),
        pytest.raises(CompileError, match="cancelled at sequential access composition"),
    ):
        summarize_sequence([a, b])
    assert summarize_sequence([a, b]) == expected


def test_duplicate_domains_are_diagnosed_instead_of_collapsing_in_the_sweep():
    p, _ = part(source(), "a", {"read_dma"})
    duplicate = replace(p.summary, cells=(*p.summary.cells, p.summary.cells[0]))
    with pytest.raises(CompileError, match="duplicate child byte/count cell"):
        summarize_sequence([replace(p, summary=duplicate)])


def test_unrelated_allocation_periods_are_not_multiplied():
    kernel = Kernel("periodic", disjoint_external=True)
    states = {p: kernel.buffer(f"state{p}", "UB", 32 * p) for p in (29, 31)}
    out = kernel.buffer("out", "GM", 32, external=True)
    kernel.loop(kernel.runtime_size("n"))
    for period in (29, 31):
        kernel.task(
            f"read{period}",
            "copy",
            kernel.view(states[period], 32, stride=32, slots=period),
            kernel.view(out, 32),
        )
    loop = extract_loop(kernel.finish())
    actor, data = next(iter(loop.actors.items()))
    parts = []
    for period in (29, 31):
        native = tuple(t for t in data.program.tasks if t.uid == f"read{period}")
        leaf = replace(
            loop, actors={actor: replace(data, program=replace(data.program, tasks=native))}
        )
        parts.append(Part(str(period), summarize_reused_storage(leaf, {f"aiv0/state{period}"})))
    result = summarize_sequence(parts)
    assert {cell.domain.period for cell in result.cells} == {1, 29, 31}
    assert not result.obligations
    assert result.visits < 300_000


def test_equal_printed_parameter_names_do_not_identify_the_same_runtime_bound():
    a, _ = part(source(), "a", {"update"})
    b, _ = part(source(), "b", {"read_dma"})
    assert a.summary.bound == b.summary.bound  # Both print 'n'.
    with pytest.raises(CompileError, match="actual SSA identity"):
        summarize_sequence([a, b])

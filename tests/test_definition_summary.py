"""Fixed definition runs checked against independent per-byte version logs."""

from itertools import pairwise

import pytest

from tests.fixtures.nested import shared_l0c_qk_pv
from xdsl_npu import budget
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.completion import CountDomain
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import Ref
from xdsl_npu.region_summary import CellSummary, summarize_fixed_storage, summarize_reused_storage
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes


def _access(access):
    return None if access is None else (access.position.offset, *access.task)


def _readers(spans):
    return tuple((span.first.offset, *span.task) for span in spans)


def _byte_oracle(loop, selected):
    """Build byte logs, then derive versions by indexed predecessor queries.

    This deliberately avoids Ref.segments, shared cell cutting, summary scans,
    and native modes_for. Pure-write indices partition each byte's event log;
    every read independently looks backward for its actual writer version.
    """
    logs = {}
    ordered = sorted(
        (task.logical_phase, owner, index, actor, task)
        for owner, (actor, data) in enumerate(loop.actors.items())
        for index, task in enumerate(data.program.tasks)
    )
    for generation in range(loop.constant_count):
        for _, _, _, actor, task in ordered:
            if task.last and generation != loop.constant_count - 1:
                continue
            touched = {}
            for number, (ref, mode) in enumerate(zip(task.refs, task.modes, strict=True)):
                root = loop.root(actor, ref)
                if root not in selected:
                    continue
                if number == 2 and (
                    task.native == "mad_first"
                    and generation == 0
                    or task.native == "mad_cycle"
                    and generation % task.params[3] == 0
                ):
                    mode = "W"
                start = sum(
                    offset
                    + stride
                    * (generation // divisor % modulus if modulus else generation // divisor)
                    for offset, stride, modulus, divisor in ref.terms
                )
                for row in range(ref.rows):
                    for byte in range(
                        start + row * ref.row_stride, start + row * ref.row_stride + ref.nbytes
                    ):
                        touched.setdefault((root, byte), set()).update(mode)
            for cell, modes in touched.items():
                logs.setdefault(cell, []).append(((generation, actor, task.uid), modes))
    result = {}
    for cell, events in logs.items():
        starts = [index for index, (_, mode) in enumerate(events) if mode == {"W"}]
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        runs = []
        for start, stop in pairwise((*starts, len(events))):
            writes = [index for index in range(start, stop) if "W" in events[index][1]]
            last = writes[-1] if writes else None
            bindings = tuple(
                (
                    events[index][0],
                    next(
                        (key for key, mode in reversed(events[:index]) if "W" in mode),
                        None,
                    ),
                )
                for index in range(start, stop)
                if "R" in events[index][1]
            )
            runs.append(
                (
                    events[start][0] if events[start][1] == {"W"} else None,
                    events[last][0] if last is not None else None,
                    tuple(
                        events[index][0]
                        for index in range(start if last is None else last + 1, stop)
                        if "R" in events[index][1]
                    ),
                    bindings,
                )
            )
        result[cell] = tuple(runs)
    return result


def _check(loop, roots, summary):
    actual = {}
    for cell in summary.cells:
        runs = tuple(
            (
                _access(run.entry),
                _access(run.last_write),
                _readers(run.pending_reads),
                tuple((_access(use.reader), _access(use.writer)) for use in run.read_bindings),
            )
            for run in cell.definitions
        )
        for byte in range(cell.start, cell.stop):
            assert (cell.storage, byte) not in actual
            actual[cell.storage, byte] = runs
    assert actual == _byte_oracle(loop, roots)


def _ring(count=3):
    kernel = Kernel("definition_ring")
    state = kernel.buffer("state", "UB", 64)
    before = kernel.buffer("before", "UB", 32)
    after = kernel.buffer("after", "UB", 32)
    kernel.loop(count, symbolic=True)
    slot = kernel.view(state, 32, stride=32, slots=2)
    kernel.task("clear", "fill", slot)
    kernel.task("read_before", "vector_copy", slot, kernel.view(before, 32))
    kernel.task("update", "vector_copy", slot, slot)
    kernel.task("read_after", "vector_copy", slot, kernel.view(after, 32))
    module = kernel.finish()
    return module, extract_loop(module), state


@pytest.mark.parametrize("count", (0, 1, 2, 3, 5))
def test_independent_clears_keep_each_rw_version_and_middle_reader(count):
    module, loop, state = _ring(count)
    before = dump(module)
    summary = summarize_fixed_storage(loop, {"aiv0/state"})
    _check(loop, {"aiv0/state"}, summary)
    assert dict(summary.origins)["aiv0/state"] is state
    assert dump(module) == before
    assert sum(len(cell.definitions) for cell in summary.cells) == count
    for cell in summary.cells:
        for run in cell.definitions:
            generation = run.entry.position.offset
            assert run.entry.task == ("aiv0", "clear")
            assert _access(run.last_write) == (generation, "aiv0", "update")
            assert tuple(use.reader.task[1] for use in run.read_bindings) == (
                "read_before",
                "update",
                "read_after",
            )
            assert _readers(run.pending_reads) == ((generation, "aiv0", "read_after"),)
    if count == 0:
        assert summary.cells == () and summary.definition_visits == 0


def test_incoming_rw_is_not_a_fresh_definition_and_clear_splits_the_run():
    kernel = Kernel("incoming_then_clear")
    state = kernel.buffer("state", "UB", 32)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(2, symbolic=True)
    value = kernel.view(state, 32)
    kernel.task("incoming_read", "vector_copy", value, kernel.view(sink, 32))
    kernel.task("incoming_update", "vector_copy", value, value)
    kernel.task("clear", "fill", value)
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"aiv0/state"})
    _check(loop, {"aiv0/state"}, summary)
    (cell,) = summary.cells
    incoming, first, second = cell.definitions
    assert incoming.entry is None
    assert _access(incoming.last_write) == (0, "aiv0", "incoming_update")
    assert [use.writer for use in incoming.read_bindings] == [None, None]
    assert incoming.pending_reads == ()
    assert _access(first.entry) == (0, "aiv0", "clear")
    assert [_access(use.writer) for use in first.read_bindings] == [
        (0, "aiv0", "clear"),
        (0, "aiv0", "clear"),
    ]
    assert _access(second.entry) == (1, "aiv0", "clear")


@pytest.mark.parametrize("initialized", (False, True))
def test_read_only_incoming_run_keeps_every_reader_without_initializing(initialized):
    kernel = Kernel("incoming_readers")
    source = kernel.buffer("source", "GM", 64, external=True, initialized=initialized)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(3, symbolic=True)
    kernel.copy("read", kernel.view(source, 32, stride=32, slots=2), kernel.view(sink, 32))
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"source"})
    _check(loop, {"source"}, summary)
    for cell in summary.cells:
        (run,) = cell.definitions
        assert run.entry is run.last_write is None
        assert all(use.writer is None for use in run.read_bindings)
        assert _readers(run.pending_reads) == tuple(
            _access(use.reader) for use in run.read_bindings
        )


def test_actual_qk_pv_clear_and_recurrence_runs_keep_separate_post_fix_incoming():
    module = lower_scopes(shared_l0c_qk_pv.to_ir().body.block.first_op)
    scope = extract_scopes(module)
    snapshots, captures = [], []
    for leaf in scope.leaves.values():
        capture = next(cap for cap in leaf.captures if cap.root.sym_name.data == "c")
        actor = next(iter(leaf.loop.actors))
        root = actor + "/" + capture.op.sym_name.data
        summary = summarize_fixed_storage(leaf.loop, {root})
        _check(leaf.loop, {root}, summary)
        assert dict(summary.origins)[root] is leaf.value_map[capture.op.ref]
        snapshots.append(summary)
        captures.append(capture.root.ref)
    assert captures[0] is captures[1] is captures[2]
    qk, pv, fix = snapshots
    assert sorted(
        run.entry.position.offset for cell in qk.cells for run in cell.definitions
    ) == list(range(8))
    for cell in qk.cells:
        assert len(cell.definitions) == 4
        for run in cell.definitions:
            assert run.entry == run.last_write
            assert len(run.read_bindings) == 1
            assert run.read_bindings[0].writer == run.entry
            assert _readers(run.pending_reads) == (_access(run.read_bindings[0].reader),)
    (pv_cell,) = pv.cells
    (accumulate,) = pv_cell.definitions
    assert accumulate.entry.position.offset == 0
    assert accumulate.last_write.position.offset == 7
    assert [
        (use.reader.position.offset, use.writer.position.offset) for use in accumulate.read_bindings
    ] == [(inner, inner - 1) for inner in range(1, 8)]
    assert not accumulate.pending_reads
    (fix_cell,) = fix.cells
    (post_fix,) = fix_cell.definitions
    assert post_fix.entry is post_fix.last_write is None
    assert len(post_fix.read_bindings) == 1 and post_fix.read_bindings[0].writer is None


def test_definition_records_do_not_rescan_fixed_footprints_or_clone_native_ir(monkeypatch):
    module, loop, _ = _ring(5)
    before = dump(module)
    calls = 0
    original = Ref.segments

    def segments(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Ref, "segments", segments)
    summarize_fixed_storage(loop, {"aiv0/state"})
    assert calls == 5 * 5  # Five selected refs per generation, including both alias operands.
    assert dump(module) == before
    assert sum(isinstance(op, TaskOp) for op in module.walk()) == 4


def test_definition_budget_precedes_cut_and_record_materialization(monkeypatch):
    import xdsl_npu.region_summary as summaries

    _, loop, _ = _ring()
    monkeypatch.setattr(summaries, "MAX_FIXED_DEFINITION_VISITS", 1)
    monkeypatch.setattr(
        summaries, "cut_access_cells", lambda *args: pytest.fail("cut before budget")
    )
    with pytest.raises(CompileError, match="fixed definition summary visit budget"):
        summarize_fixed_storage(loop, {"aiv0/state"})


def test_definition_scan_polls_cooperative_cancellation(monkeypatch):
    import xdsl_npu.region_summary as summaries

    kernel = Kernel("many_incoming_reads")
    source = kernel.buffer("source", "GM", 32, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(300, symbolic=True)
    kernel.copy("read", kernel.view(source, 32), kernel.view(sink, 32))
    loop = extract_loop(kernel.finish())
    observed = []
    original = summaries.poll

    def poll(phase, **counts):
        observed.append(phase)
        original(phase, **counts)

    monkeypatch.setattr(summaries, "poll", poll)
    with (
        pytest.raises(CompileError, match="cancelled at fixed child definitions"),
        budget.limit(cancel=lambda: "fixed child definitions" in observed),
    ):
        summarize_fixed_storage(loop, {"source"})


def test_existing_frontier_visits_and_periodic_defaults_keep_their_contract():
    _, loop, _ = _ring(3)
    fixed = summarize_fixed_storage(loop, {"aiv0/state"})
    # Five state refs per generation, four native events, two cells, 15 cuts.
    assert fixed.visits == 15 + 12 + 3 + 2 * 15
    assert fixed.definition_visits == 3 * 15
    periodic = summarize_reused_storage(loop, {"aiv0/state"})
    assert periodic.definition_visits == 0
    assert all(cell.definitions == () for cell in periodic.cells)
    cell = CellSummary("state", 0, 32, CountDomain(1, 0, exact=1), (), None, None, ())
    assert cell.definitions == ()

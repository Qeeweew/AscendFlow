"""Fixed child frontiers from real row-pitched effects, without native unrolling."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr

from xdsl_npu import budget
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.completion import Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import FrameOp, TaskOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import Ref
from xdsl_npu.region_summary import Access, summarize_fixed_storage, summarize_reused_storage
from xdsl_npu.scalar import literal


def _pitched(*, rows=3, chunks=4, width=32, pitch=160, groups=2):
    kernel = Kernel("pitched_child", blocks=groups)
    output = kernel.buffer("Score", "GM", rows * pitch * groups, external=True)
    temporary = kernel.buffer("temporary", "UB", rows * width)
    kernel.loop(chunks, symbolic=True)
    kernel.task("initialize", "fill", kernel.view(temporary, rows * width))
    kernel.task(
        "write_column",
        "copy_rows",
        kernel.view(temporary, width, rows=rows, row_stride=width),
        kernel.view(
            output,
            width,
            rows=rows,
            row_stride=pitch,
            stride=width,
            core_stride=rows * pitch,
        ),
    )
    module = kernel.finish()
    return module, extract_loop(module), output


def _key(access, count):
    return access.position.evaluate(count), *access.task


def _reads(spans, count):
    return {
        (g, *span.task)
        for span in spans
        for g in range(span.first.evaluate(count), span.last.evaluate(count) + 1, span.step)
    }


def _oracle(loop, roots):
    """Independent byte effects: no Ref.segments, cut iterator, or _scan."""
    count = loop.constant_count
    ordered = sorted(
        (task.logical_phase, owner, index, actor, task)
        for owner, (actor, data) in enumerate(loop.actors.items())
        for index, task in enumerate(data.program.tasks)
    )
    result = {}
    for g in range(count):
        for _, _, _, actor, task in ordered:
            if task.last and g != count - 1:
                continue
            touched = {}
            for number, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                root = loop.root(actor, ref)
                if root not in roots:
                    continue
                if task.native == "mad_first" and number == 2 and g == 0:
                    mode = "W"
                if task.native == "mad_cycle" and number == 2 and g % task.params[3] == 0:
                    mode = "W"
                start = sum(
                    offset + stride * (g // divisor % modulus if modulus else g // divisor)
                    for offset, stride, modulus, divisor in ref.terms
                )
                width = (
                    ref.tail_nbytes if g == ref.tail_generation and ref.tail_nbytes else ref.nbytes
                )
                for row in range(ref.rows):
                    for byte in range(
                        start + row * ref.row_stride, start + row * ref.row_stride + width
                    ):
                        touched.setdefault((root, byte), set()).update(mode)
            key = (g, actor, task.uid)
            for index, mode in touched.items():
                state = result.setdefault(index, [set(), None, None, set()])
                if "R" in mode:
                    state[3].add(key)
                    if state[1] is None:
                        state[0].add(key)
                if "W" in mode:
                    state[1] = state[1] or key
                    state[2], state[3] = key, set()
    return result


def _check(loop, roots, summary):
    count = loop.constant_count
    actual = {}
    for cell in summary.for_count(count):
        assert cell.domain.exact == count
        frontier = [
            _reads(cell.reads_before_write, count),
            _key(cell.first_write, count) if cell.first_write else None,
            _key(cell.last_write, count) if cell.last_write else None,
            _reads(cell.pending_reads, count),
        ]
        for byte in range(cell.start, cell.stop):
            assert (cell.storage, byte) not in actual
            actual[cell.storage, byte] = frontier
    assert actual == _oracle(loop, roots)


def test_pitched_columns_have_actual_writers_and_keep_holes_and_ir():
    module, loop, output = _pitched()
    before = dump(module)
    assert loop.frames == {"Score": 32}
    assert loop.storage_periods["Score"] == 1
    summary = summarize_fixed_storage(loop, {"Score"})
    _check(loop, {"Score"}, summary)
    assert len(summary.cells) == 12
    assert dict(summary.origins)["Score"] is output
    for cell in summary.cells:
        chunk = cell.start % 160 // 32
        assert cell.start // 160 == (cell.stop - 1) // 160
        assert cell.stop - cell.start == 32
        assert cell.start % 160 < 128
        assert (
            cell.first_write
            == cell.last_write
            == Access(("aiv0", "write_column"), Position(False, chunk))
        )
    assert dump(module) == before
    assert sum(isinstance(op, TaskOp) for op in module.walk()) == 2
    # The periodic API deliberately retains its existing frame restriction.
    with pytest.raises(CompileError, match="parent ownership"):
        summarize_reused_storage(loop, {"Score"})


def test_score_sized_column_summary_cost_tracks_intervals():
    module, loop, _ = _pitched(rows=128, chunks=8, width=512, pitch=4096, groups=2)
    before = dump(module)
    summary = summarize_fixed_storage(loop, {"Score"})
    assert len(summary.cells) == 1024
    assert summary.visits < 10_000
    assert summary.visits < 128 * 4096
    assert {cell.last_write.position.offset for cell in summary.cells} == set(range(8))
    assert dump(module) == before
    assert len(next(iter(loop.actors.values())).program.tasks) == 2


def _ring(count=3):
    kernel = Kernel("shared_ring")
    state = kernel.buffer("state", "UB", 64)
    sinks = [kernel.buffer(name, "UB", 32) for name in ("left", "right")]
    kernel.loop(count, symbolic=True)
    slot = kernel.view(state, 32, stride=32, slots=2)
    kernel.task("initialize", "fill", slot)
    for name, sink in zip(("read_left", "read_right"), sinks):
        kernel.task(name, "vector_copy", slot, kernel.view(sink, 32))
    module = kernel.finish()
    return extract_loop(module), state


@pytest.mark.parametrize("count", [0, 1, 2, 3, 8])
def test_fixed_ring_keeps_actual_allocation_and_every_pending_reader(count):
    loop, state = _ring(count)
    summary = summarize_fixed_storage(loop, {"aiv0/state"})
    _check(loop, {"aiv0/state"}, summary)
    assert dict(summary.origins)["aiv0/state"] is state
    for cell in summary.cells:
        slot = cell.start // 32
        last = count - 1 - ((count - 1 - slot) % 2)
        assert cell.first_write.position.evaluate(count) == slot
        assert cell.last_write.position.evaluate(count) == last
        assert _reads(cell.pending_reads, count) == {
            (last, "aiv0", "read_left"),
            (last, "aiv0", "read_right"),
        }
    assert not summary.cells if count == 0 else summary.cells
    with pytest.raises(CompileError, match="different constant child length"):
        summary.for_count(count + 1)


def test_read_only_cell_keeps_the_entire_actual_reader_family():
    kernel = Kernel("read_only", blocks=2)
    source = kernel.buffer("source", "GM", 32, external=True, initialized=True)
    sink = kernel.buffer("sink", "UB", 32)
    kernel.loop(3, symbolic=True)
    kernel.copy("load", kernel.view(source, 32), kernel.view(sink, 32))
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"source"})
    _check(loop, {"source"}, summary)
    (cell,) = summary.cells
    assert cell.first_write is cell.last_write is None
    assert (
        _reads(cell.reads_before_write, 3)
        == _reads(cell.pending_reads, 3)
        == {(g, "aiv0", "load") for g in range(3)}
    )


def test_actual_qk_cube_ring_keeps_fix6_and_fix7_retirements():
    kernel = Kernel("qk_child", core="aic")
    a = kernel.buffer("a", "L0A", 4096, "f16", layout="ZZ")
    b = kernel.buffer("b", "L0B", 4096, "f16", layout="ZN")
    c = kernel.buffer("c", "L0C", 2 * 65536, layout="NZ")
    score = kernel.buffer("Score", "GM", 128 * 4096, external=True)
    kernel.init_fixpipe()
    kernel.loop(8, symbolic=True)
    slot = kernel.view(c, 65536, stride=65536, slots=2)
    kernel.task(
        "mad", "mad", kernel.view(a, 4096), kernel.view(b, 4096), slot, params=(128, 128, 16, 1)
    )
    kernel.task(
        "fix",
        "fix",
        slot,
        kernel.view(score, 512, stride=512, rows=128, row_stride=4096),
        params=(128, 128),
    )
    module = kernel.finish()
    before = dump(module)
    summary = summarize_fixed_storage(extract_loop(module), {"aic/c"})
    assert dict(summary.origins)["aic/c"] is c
    assert len(summary.cells) == 2
    for slot, cell in enumerate(summary.cells):
        last = 6 + slot
        assert cell.first_write == Access(("aic", "mad"), Position(False, slot))
        assert cell.last_write == Access(("aic", "mad"), Position(False, last))
        assert _reads(cell.pending_reads, 8) == {(last, "aic", "fix")}
    assert dump(module) == before


@pytest.mark.parametrize(
    "native,params", [("mad_first", (16, 16, 16)), ("mad_cycle", (16, 16, 16, 2))]
)
def test_mad_clear_semantics_are_evaluated_at_actual_generations(native, params):
    kernel = Kernel("accumulator", core="aic")
    a = kernel.buffer("a", "L0A", 512, "f16", layout="ZZ")
    b = kernel.buffer("b", "L0B", 512, "f16", layout="ZN")
    c = kernel.buffer("c", "L0C", 1024, layout="NZ")
    kernel.loop(3, symbolic=True)
    kernel.task(
        "mad", native, kernel.view(a, 512), kernel.view(b, 512), kernel.view(c, 1024), params=params
    )
    loop = extract_loop(kernel.finish())
    summary = summarize_fixed_storage(loop, {"aic/c"})
    _check(loop, {"aic/c"}, summary)
    (cell,) = summary.cells
    assert not cell.reads_before_write
    assert cell.first_write.position.evaluate(3) == 0
    assert cell.last_write.position.evaluate(3) == 2
    assert not cell.pending_reads


def test_terminal_task_runs_once_at_actual_last_generation():
    loop, _ = _ring(4)
    actor, data = next(iter(loop.actors.items()))
    terminal = tuple(
        replace(t, last=True) if t.uid == "read_right" else t for t in data.program.tasks
    )
    loop = replace(
        loop, actors={actor: replace(data, program=replace(data.program, tasks=terminal))}
    )
    summary = summarize_fixed_storage(loop, {"aiv0/state"})
    _check(loop, {"aiv0/state"}, summary)
    assert _reads(summary.cells[0].pending_reads, 4) == {(2, actor, "read_left")}
    assert _reads(summary.cells[1].pending_reads, 4) == {
        (3, actor, "read_left"),
        (3, actor, "read_right"),
    }


def test_runtime_count_and_forged_static_binding_are_rejected():
    kernel = Kernel("runtime_child")
    count = kernel.runtime_size("count")
    state = kernel.buffer("state", "UB", 32)
    kernel.loop(count)
    kernel.task("fill", "fill", kernel.view(state, 32))
    loop = extract_loop(kernel.finish())
    with pytest.raises(CompileError, match="actual constant count"):
        summarize_fixed_storage(loop, {"aiv0/state"})
    with pytest.raises(CompileError, match="actual constant child stream"):
        summarize_fixed_storage(replace(loop, constant_count=3), {"aiv0/state"})
    fixed, _ = _ring()
    with pytest.raises(CompileError, match="actual constant child stream"):
        summarize_fixed_storage(replace(fixed, constant_count=4), {"aiv0/state"})


def test_mixed_actors_and_unknown_roots_are_explicitly_rejected():
    loop, _ = _ring()
    actor, data = next(iter(loop.actors.items()))
    with pytest.raises(CompileError, match="mixed fixed summaries"):
        summarize_fixed_storage(replace(loop, actors={actor: data, "aiv1": data}), {"aiv0/state"})
    with pytest.raises(CompileError, match="unknown storage root"):
        summarize_fixed_storage(loop, {"missing"})


@pytest.mark.parametrize(
    "field,value", [("count_offset", 32), ("count_core_stride", 32), ("prefix_extent", literal(8))]
)
def test_unresolved_parent_translation_or_clipping_is_rejected(field, value):
    loop, _ = _ring()
    actor, data = next(iter(loop.actors.items()))
    first, *rest = data.program.tasks
    first = replace(first, refs=(replace(first.refs[0], **{field: value}),))
    loop = replace(
        loop, actors={actor: replace(data, program=replace(data.program, tasks=(first, *rest)))}
    )
    with pytest.raises(CompileError, match="parent ownership|unclipped"):
        summarize_fixed_storage(loop, {"aiv0/state"})


def test_actual_invocation_frame_needs_parent_contract():
    kernel = Kernel("framed_child")
    outputs = kernel.runtime_size("outputs")
    score = kernel.buffer("Score", "GM", 0, external=True, extent_count=outputs, count_scale=32)
    temporary = kernel.buffer("temporary", "UB", 32)
    parent = kernel.invocations(outputs)
    frame = FrameOp.create(
        operands=[score, parent.body.block.args[0], parent.step],
        attributes={"scale": IntAttr(32)},
        result_types=[score.type],
    )
    parent.body.block.add_op(frame)
    kernel.loop(3, symbolic=True)
    value = kernel.view(temporary, 32)
    kernel.task("fill", "fill", value)
    kernel.copy("store", value, kernel.view(frame.ref, 32))
    loop = extract_loop(kernel.finish())
    assert loop.invocation.frames[0].op is frame
    with pytest.raises(CompileError, match="invocation-frame.*parent ownership"):
        summarize_fixed_storage(loop, {"npu_frame_Score"})


def test_dynamic_extent_needs_parent_contract():
    _, loop, _ = _pitched()
    actor, data = next(iter(loop.actors.items()))
    init, store = data.program.tasks
    source, destination = store.refs
    destination = replace(destination, storage=replace(destination.storage, count_scale=32))
    store = replace(store, refs=(source, destination))
    loop = replace(
        loop, actors={actor: replace(data, program=replace(data.program, tasks=(init, store)))}
    )
    with pytest.raises(CompileError, match="parent ownership"):
        summarize_fixed_storage(loop, {"Score"})


def test_fixed_summary_checks_actual_cross_core_owner_union():
    _, loop, _ = _pitched()
    actor, data = next(iter(loop.actors.items()))
    initialize, store = data.program.tasks
    source, destination = store.refs
    store = replace(store, refs=(source, replace(destination, core_terms=())))
    loop = replace(
        loop,
        actors={actor: replace(data, program=replace(data.program, tasks=(initialize, store)))},
    )
    with pytest.raises(CompileError, match="fixed child GM owners overlap"):
        summarize_fixed_storage(loop, {"Score"})


def test_segment_budget_precedes_materializing_rows(monkeypatch):
    import xdsl_npu.region_summary as summaries

    _, loop, _ = _pitched()
    monkeypatch.setattr(summaries, "MAX_FIXED_ACCESS_SEGMENTS", 1)
    monkeypatch.setattr(Ref, "segments", lambda *args: pytest.fail("materialized before budget"))
    with pytest.raises(CompileError, match="access-segment budget"):
        summarize_fixed_storage(loop, {"Score"})


def test_cut_budget_precedes_shared_cut_materialization(monkeypatch):
    import xdsl_npu.region_summary as summaries

    loop, _ = _ring()
    monkeypatch.setattr(summaries, "MAX_FIXED_SUMMARY_VISITS", 20)
    monkeypatch.setattr(
        summaries, "cut_access_cells", lambda *args: pytest.fail("cut before budget")
    )
    with pytest.raises(CompileError, match="cut/scan visit budget"):
        summarize_fixed_storage(loop, {"aiv0/state"})


def test_fixed_scan_polls_and_obeys_cooperative_cancellation(monkeypatch):
    import xdsl_npu.region_summary as summaries

    _, loop, _ = _pitched(rows=128, chunks=8, width=512, pitch=4096)
    observed = []
    original = summaries.poll

    def poll(phase, **counts):
        observed.append(phase)
        original(phase, **counts)

    monkeypatch.setattr(summaries, "poll", poll)
    with (
        pytest.raises(CompileError, match="cancelled at fixed child footprints"),
        budget.limit(cancel=lambda: "fixed child footprints" in observed),
    ):
        summarize_fixed_storage(loop, {"Score"})
    assert "fixed child summary" in observed
    assert "fixed child footprints" in observed


def test_empty_root_selection_never_iterates_a_large_fixed_count(monkeypatch):
    import xdsl_npu.region_summary as summaries

    loop, _ = _ring(1 << 40)
    monkeypatch.setattr(Ref, "segments", lambda *args: pytest.fail("scanned unselected storage"))
    summary = summaries.summarize_fixed_storage(loop, ())
    assert summary.cells == summary.origins == ()
    assert summary.visits == 0

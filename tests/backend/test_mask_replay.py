"""mask replay contracts."""

from __future__ import annotations

import subprocess

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.backend.vector_mask import helper_lines
from tests.fixtures.masks import (
    _mask_spec,
    _masked_fill,
    _row_fill_asc,
)
from tests.models.mask_predicates import _expected_word
from tests.support.mask_cpp import (
    HOST_FLOOR,
    HOST_UNIT_SOURCE,
    _run_emitter_cases,
    _run_mask_helper,
)


def test_emitted_mask_word_algorithm_is_exact(tmp_path):
    values = [
        -(1 << 120),
        -(1 << 64),
        -(1 << 63),
        -65,
        -64,
        -63,
        -1,
        0,
        1,
        63,
        64,
        65,
        (1 << 63) - 1,
        (1 << 120),
    ]
    cases = [
        (a, b, kind)
        for a in (-64, -7, -1, 0, 1, 3, 64, (1 << 62))
        for b in values
        for kind in range(6)
    ]
    words = _run_mask_helper(cases, tmp_path)
    assert len(words) == len(cases)
    for (a, b, kind), word in zip(cases, words, strict=True):
        assert word == _expected_word(a, b, kind), (a, b, kind)


def test_unit_coefficient_word_is_exact_without_a_search(tmp_path):
    """The O(1) unit-coefficient word is exact, including int64 extremes."""
    source = tmp_path / "mask_unit.cpp"
    binary = tmp_path / "mask_unit"
    source.write_text(HOST_UNIT_SOURCE.replace("%(helpers)s", "\n".join(helper_lines())))
    subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    extremes = [
        -(1 << 125),
        -(1 << 120),
        -(1 << 64),
        -(1 << 63),
        -(1 << 63) + 1,
        -(1 << 63) + 64,
        (1 << 63) - 65,
        (1 << 63) - 1,
        (1 << 64),
        (1 << 120),
        (1 << 125),
        (1 << 126) - 1,
        -(1 << 126) + 1,
    ]
    values = [*range(-200, 201), *extremes]
    cases = [(b, kind) for b in values for kind in range(6)]
    lines = "\n".join(f"{b} {kind}" for b, kind in cases)
    completed = subprocess.run(
        [str(binary)], input=lines, check=True, capture_output=True, text=True
    )
    words = [int(value) for value in completed.stdout.split()]
    assert len(words) == len(cases)
    for (b, kind), word in zip(cases, words, strict=True):
        assert word == _expected_word(1, b, kind), (b, kind)


def test_unit_lane_coefficients_emit_the_interval_recipe():
    from ascendflow.mask import ITERATION_CHUNKS, ITERATION_ROWS, WORD_AFFINE

    # A single unit comparison selects the selected row traversal; a Boolean
    # composition keeps chunk iteration, where the +1 unit word (and the -1
    # negate-and-flip form) are still emitted per 64-lane block.
    for expression, call, searched in (
        ("((cols + q) > rows) & ((rows * 2) <= 30)", "npu_mask_unit(", True),
        ("((cols + q) > rows) & ((rows - cols) >= q)", "npu_mask_unit(-(", False),
    ):
        result = compile(_masked_fill(expression), timeout_s=120)
        task = next(
            template
            for template in collect_native_memory(result.plan).templates
            if template.spec.native == "mask_fill"
        )
        attributes = task.operation.attributes
        assert attributes["mask_word"].data == WORD_AFFINE
        assert attributes["mask_iteration"].data == ITERATION_CHUNKS
        assert call in result.asc
        # A zero-coefficient comparison keeps the exact point test; unit-only
        # compositions emit no search at all.
        assert ("npu_mask_monotone((__int128)(" in result.asc) == searched
    rows = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    row_task = next(
        template
        for template in collect_native_memory(rows.plan).templates
        if template.spec.native == "mask_fill"
    )
    assert row_task.operation.attributes["mask_iteration"].data == ITERATION_ROWS
    assert "for (int64_t npu_row = 0" in rows.asc
    assert "npu_mask_clamp(" in rows.asc
    # A non-unit coefficient keeps the general constant-coefficient fallback.
    general = compile(_masked_fill("(cols * 3) > rows"), timeout_s=120)
    assert "npu_mask_monotone((__int128)(3LL)" in general.asc


def test_runtime_tile_bounds_replay_is_exact(tmp_path):
    """The emitted replays match independent per-lane integer truth."""
    cases = [
        # single unit comparison: row-interval traversal at several widths
        (_mask_spec("(i1 + q) > i0", (4, 64)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (4, 128)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": 300}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": -1000}),
        (_mask_spec("(i1 + q) > i0", (16, 1024)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (2, 1024)), {"q": 1000}),
        # all six kinds, unit and negated-unit, incl. eq/ne two-span points
        (_mask_spec("(i1 + q) >= i0"), {"q": 3}),
        (_mask_spec("(i1 + q) < i0"), {"q": 3}),
        (_mask_spec("(i1 + q) <= i0"), {"q": 3}),
        (_mask_spec("(i1 + q) == i0"), {"q": 3}),
        (_mask_spec("(i1 + q) == i0"), {"q": 1000}),
        (_mask_spec("(i1 + q) != i0"), {"q": 3}),
        (_mask_spec("(i1 + q) != i0"), {"q": 1000}),
        (_mask_spec("(i0 >= i1 + q)"), {"q": -5}),
        (_mask_spec("(i0 >= i1 + q)"), {"q": 5}),
        # non-unit / zero coefficient / Boolean / periodic keep the chunk replay
        (_mask_spec("((i1 * 3) - q) > (i0 * 2)"), {"q": 2}),
        (_mask_spec("((i1 - i1) + q) > i0"), {"q": 3}),
        (_mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)"), {"q": 300}),
        (_mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)"), {"q": 3}),
        (_mask_spec("~((i1 + q) > i0)"), {"q": 3}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 0, "hi": 190}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 0, "hi": 1000}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 5, "hi": 0}),
        (_mask_spec("((i1 // 64) % 2) == 0"), {}),
        # opposite signed uses of one runtime atom: true difference is 2*q
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": 5}),
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": 0}),
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": -5}),
        # exact int64 extremes through the row threshold
        (_mask_spec("(i1 + q) > (i0 + (-9223372036854775807 - 1))"), {"q": (1 << 63) - 1}),
        (_mask_spec("(i1 + q) > (i0 + (-9223372036854775807 - 1))"), {"q": -(1 << 63)}),
        (_mask_spec("((i1 + q) + (-9223372036854775807 - 1)) == (i0 + q)"), {"q": 0}),
        # Full-run + trailing/leading recomputation: a run that advances past
        # whole blocks must not reuse the previous chunk/from/span.
        *[(_mask_spec("i1 < hi", (4, 256)), {"hi": hi}) for hi in (65, 127, 129, 191, 255)],
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 65}),
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 127}),
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 255}),
        # `ne` point past one whole chunk: first span has a full run + trailing
        # partial, second span has a leading partial + full run.
        (_mask_spec("(i1 + q) != i0", (4, 256)), {"q": -100}),
        # One row longer than 255 whole blocks plus a trailing partial: the
        # bounded 255 split plus one 1-block repeat and a trailing 63-lane
        # bitmap, i.e. exactly three element calls.
        (_mask_spec("i1 < hi", (1, 16448)), {"hi": 16447}),
    ]
    calls = _run_emitter_cases(cases, tmp_path)
    # Row intervals on a fully mixed large tile: 16 rows x <= 3 spans each, far
    # below the 1024/64 = 256 per-chunk visits it replaces.
    large_mixed = calls[5]
    assert large_mixed <= 16 * 4, calls
    # Same-chunk span (C=64) is exactly one bitmap; the empty/full tiles are the
    # retained whole-tile runtime shortcut.
    assert calls[0] <= 4, calls
    assert calls[6] == 1, calls
    assert calls[4] == 0, calls
    tail = calls[len(cases) - 10 :]
    # Every `i1 < hi` tail case is exactly one full-run call plus one trailing
    # partial block per row; the `<=` variants cover the whole row in one run
    # except hi=65/127/191-style tails handled above.
    assert tail[0:5] == [8, 8, 8, 8, 8], tail
    assert tail[5:8] == [8, 4, 1], tail  # hi=255 proves the whole tile true
    ne_point = tail[8]
    assert 8 < ne_point <= 4 * 6, tail
    large = tail[9]
    assert large == 3, tail


def test_runtime_tile_bounds_prescribe_the_affine_dispatch_only():
    from ascendflow.mask import ITERATION_CHUNKS, ITERATION_ROWS, WORD_AFFINE, WORD_LANE

    affine = compile(_masked_fill("(cols + q) > rows"), timeout_s=120).asc
    assert "if (npu_m_all" in affine and "else if (!(npu_m_none" in affine
    # The selected row traversal is the unproven fallback of the same replay.
    assert "for (int64_t npu_row = 0" in affine
    assert "for (int64_t npu_chunk = 0" not in affine
    assert "asc_set_vector_mask(0, 0)" not in affine
    assert "PIPE_ALL" not in affine
    spec = _mask_spec("(i1 + q) > i0")
    assert spec[3] == WORD_AFFINE and spec[5] == ITERATION_ROWS
    # A Boolean composition keeps the tested runtime bounds + chunk replay.
    composed = compile(_masked_fill("((cols + q) > rows) & ((rows * 2) <= 30)"), timeout_s=120).asc
    assert "if (npu_m_all" in composed and "for (int64_t npu_chunk = 0" in composed
    assert "for (int64_t npu_row = 0" not in composed
    assert _mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)")[5] == ITERATION_CHUNKS
    general = compile(_masked_fill("(cols * 3) > rows"), timeout_s=120).asc
    assert "npu_mask_monotone((__int128)(3LL)" in general
    # A periodic predicate is not proven affine: no bounds, no row traversal.
    periodic = compile(_masked_fill("((cols // 64) % 2) == 0"), timeout_s=120).asc
    assert "npu_m_all" not in periodic and "npu_m_lo" not in periodic
    assert "npu_mask_word |= (UINT64_C(1) << npu_lane)" in periodic
    assert _mask_spec("((i1 // 64) % 2) == 0")[3] == WORD_LANE


def test_row_full_run_keeps_the_static_255_repeat_boundary():
    """A row bounds a full run statically: split only above 255 whole blocks."""

    def row_body(asc):
        # The whole-tile all-true shortcut keeps its own bounded run replay;
        # only the per-row spans are under test here.
        at = asc.index("for (int64_t npu_row = 0")
        return asc[at : asc.index("asc_sync_notify(PIPE_V, PIPE_MTE3", at)]

    small = row_body(_row_fill_asc((16, 256), "(i1 + q) > i0"))
    assert "int64_t npu_rep = (npu_end - npu_at) / 64;" in small
    assert "npu_left" not in small and "? 255 :" not in small
    # Exactly 255 whole blocks per row still fits one ISA repeat ...
    edge = row_body(_row_fill_asc((1, 16320), "i1 < hi"))
    assert "int64_t npu_rep = (npu_end - npu_at) / 64;" in edge
    assert "npu_left" not in edge and "? 255 :" not in edge
    # ... while one block more needs the bounded 255 split.
    large = row_body(_row_fill_asc((1, 16384), "i1 < hi"))
    assert "int64_t npu_left = (npu_end - npu_at) / 64;" in large
    assert "npu_rep = npu_left > 255 ? 255 : npu_left;" in large


def test_emitted_floor_division_matches_python_for_negative_numerators(tmp_path):
    source = tmp_path / "floor_helper.cpp"
    binary = tmp_path / "floor_helper"
    source.write_text(HOST_FLOOR.replace("%(helpers)s", "\n".join(helper_lines())))
    subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [-(1 << 100), -(1 << 63), -101, -7, -1, 0, 1, 7, 101, (1 << 63) - 1]
    divisors = [1, 2, 3, 5, 7, 64, (1 << 62)]
    cases = [(a, b, op) for a in values for b in divisors for op in (0, 1)]
    lines = "\n".join(f"{a} {b} {op}" for a, b, op in cases)
    completed = subprocess.run(
        [str(binary)], input=lines, check=True, capture_output=True, text=True
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    assert len(rows) == len(cases)
    for (a, b, op), (hi, lo) in zip(cases, rows, strict=True):
        unsigned = (int(hi) << 64) | int(lo)
        value = unsigned - (1 << 128) if unsigned >= (1 << 127) else unsigned
        expected = a // b if op == 0 else a % b
        assert value == expected, (a, b, op)

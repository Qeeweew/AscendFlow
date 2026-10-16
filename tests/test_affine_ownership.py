"""Finite row-pitched frames, checked against independent concrete byte sets."""

from random import Random

import pytest

from xdsl_npu.affine_ownership import verify_affine_frames
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.model import Ref, Storage


def _refs(width, pitch, rows, stride, offsets, owner_stride):
    storage = Storage("score", "GM", 1 << 40, "f32", "linear", True, False)
    return tuple(
        Ref(
            storage,
            ((offset, stride, 0, 1),),
            width,
            "f32",
            "linear",
            rows=rows,
            row_stride=pitch,
            core_terms=((owner_stride, 1, 0),),
        )
        for offset in offsets
    )


def test_affine_ownership_matches_small_exhaustive_byte_oracle():
    rng = Random(712)
    for _ in range(250):
        width, stride = rng.randrange(1, 12), rng.randrange(1, 12)
        pitch, rows = rng.randrange(width, 48), rng.randrange(1, 5)
        count, groups = rng.randrange(6), rng.randrange(1, 4)
        offsets = (0, rng.randrange(20))
        owner_stride = rng.randrange(100)
        occupied, safe = set(), True
        for core in range(groups):
            for g in range(count):
                current = {
                    offset + core * owner_stride + g * stride + row * pitch + byte
                    for offset in offsets
                    for row in range(rows)
                    for byte in range(width)
                }
                safe &= not bool(occupied & current)
                occupied |= current
        refs = _refs(width, pitch, rows, stride, offsets, owner_stride)
        if safe:
            verify_affine_frames(refs, count, groups, stride, root="score")
        else:
            with pytest.raises(CompileError, match="overlap|crosses its task frame"):
                verify_affine_frames(refs, count, groups, stride, root="score")


def test_column_slices_preserve_holes_and_reject_an_extra_iteration():
    refs = _refs(512, 4096, 128, 512, (0,), 128 * 4096)
    verify_affine_frames(refs, 8, 20, 512, root="score")
    with pytest.raises(CompileError, match="overlapping affine generations"):
        verify_affine_frames(refs, 9, 20, 512, root="score")


def test_cross_owner_overlap_is_not_hidden_by_distant_rows():
    refs = _refs(32, 1024, 2, 32, (0,), 1024)
    with pytest.raises(CompileError, match="cross-group stream ownership overlap"):
        verify_affine_frames(refs, 8, 2, 32, root="score")


def test_trip_count_does_not_control_ownership_work(monkeypatch):
    original, calls = Ref.segments, []

    def segments(ref, g, core=0):
        calls.append((g, core))
        return original(ref, g, core)

    monkeypatch.setattr(Ref, "segments", segments)
    for count in (8, 1 << 20):
        refs = _refs(32, count * 32, 2, 32, (0,), count * 64)
        calls.clear()
        verify_affine_frames(refs, count, 20, 32, root="score")
        assert calls == [(0, 0)]


def test_fragment_budget_precedes_materialization(monkeypatch):
    import xdsl_npu.affine_ownership as ownership

    monkeypatch.setattr(ownership, "MAX_AFFINE_VISITS", 1)
    monkeypatch.setattr(Ref, "segments", lambda *args: pytest.fail("materialized before budget"))
    with pytest.raises(CompileError, match="fragment budget"):
        verify_affine_frames(_refs(32, 1024, 2, 32, (0,), 1024), 8, 2, 32, root="score")

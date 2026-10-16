"""Validation unions preserve bytes, alignment and every real owner translation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import IntAttr

from xdsl_npu.loop_model import (
    Loop,
    _check_accesses,
    _ownership_footprints,
    verify_access_ownership,
)
from xdsl_npu.model import CompileError, Ref, Storage


def storage(size=16384):
    return Storage("X", "GM", size, "f32", "linear", True, True)


def ref(offset, length=4, *, owner_stride=2048):
    return Ref(
        storage(),
        ((offset, 0, 0, 1), (0, 1024, 2, 1)),
        length,
        "f32",
        "linear",
        core_terms=((owner_stride, 1, 0),),
    )


def bytes_at(refs, generation, owner):
    """Literal byte oracle; it never consumes the compact validation result."""
    result = set()
    for view in refs:
        start = sum(
            o + stride * ((generation // divisor) % modulus if modulus else generation // divisor)
            for o, stride, modulus, divisor in view.terms
        )
        start += sum(
            stride * ((owner // divisor) % modulus if modulus else owner // divisor)
            for stride, divisor, modulus in view.core_terms
        )
        length = view.tail_nbytes if generation == view.tail_generation else view.nbytes
        for row in range(view.rows):
            result.update(
                range(start + row * view.row_stride, start + row * view.row_stride + length)
            )
    return result


def loop(refs):
    task = SimpleNamespace(refs=tuple(refs), modes=("RW",) * len(refs))
    actor = SimpleNamespace(program=SimpleNamespace(tasks=(task,)))
    return Loop(
        SimpleNamespace(blocks=IntAttr(4)), {"aiv0": actor}, {}, None, None, {"X": 2}, {}, (), None
    )


@pytest.mark.parametrize(
    "kind", ("adjacent", "overlap", "holes", "different-owner", "pitched", "tail")
)
def test_union_preserves_exact_bytes_for_every_phase_and_owner(kind):
    views = [ref(0, 8), ref(8, 8), ref(64, 8)]
    if kind == "overlap":
        views.append(ref(4, 16))
    elif kind == "holes":
        views.extend((ref(32), ref(44)))
    elif kind == "different-owner":
        views.append(ref(16, owner_stride=4096))
    elif kind == "pitched":
        views.append(replace(ref(16), rows=3, row_stride=12))
    elif kind == "tail":
        views.append(replace(ref(32, 8), tail_generation=1, tail_nbytes=4))
    compact = _ownership_footprints(views)
    for generation in range(6):
        for owner in range(4):
            assert bytes_at(compact, generation, owner) == bytes_at(views, generation, owner)


def test_validation_cost_is_coverage_size_not_native_fragment_count():
    views = tuple(ref(i, 4) for i in range(0, 1024, 4))
    actual = loop(views)
    stats = {}
    verify_access_ownership(actual, statistics=stats)
    assert stats == {"input_refs": 256, "coverage_refs": 1, "input_group_visits": 2048}
    assert actual.actors["aiv0"].program.tasks[0].refs is views
    assert len(_ownership_footprints(views)) == 1


@pytest.mark.parametrize(
    "damage", ("misaligned", "short-size", "bounds", "owner-overlap", "owner-change")
)
def test_invalid_member_cannot_be_hidden_by_an_otherwise_valid_union(damage):
    views = [ref(0, 8), ref(8, 8)]
    if damage == "misaligned":
        views.append(ref(2, 4))
    elif damage == "short-size":
        views.extend((ref(0, 2), ref(2, 2)))
    elif damage == "bounds":
        views.append(ref(16380, 8))
    elif damage == "owner-overlap":
        views = [ref(0, 8, owner_stride=4)]
    else:
        views.append(ref(0, 8, owner_stride=4096))
    with pytest.raises(CompileError):
        _check_accesses(loop(views), {"aiv0": dict(enumerate(views))}, native=False)
    with pytest.raises(CompileError):
        verify_access_ownership(loop(views))

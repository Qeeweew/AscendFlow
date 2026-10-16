"""Pure count arithmetic shares values within one compile, never proof state."""

from dataclasses import asdict

import pytest

from xdsl_npu import completion
from xdsl_npu.completion import (
    CountDomain,
    canonical,
    completion_math,
    completion_math_statistics,
    count_domain,
    includes,
    intersection,
)
from xdsl_npu.model import CompileError
from xdsl_npu.periodic import MAX_PERIOD


def _allows(domain, count):
    if domain.exact is not None:
        return count == domain.exact
    return (
        count >= (domain.period if domain.minimum is None else domain.minimum)
        and count % domain.period == domain.residue
        and (domain.maximum is None or count <= domain.maximum)
    )


def test_canonicalization_is_idempotent_and_hashing_preserves_public_fields():
    raw = CountDomain(3, 2, minimum=0, maximum=20)
    value = canonical(raw)
    assert value == CountDomain(3, 2, minimum=2, maximum=20)
    assert canonical(value) is value
    assert hash(value) == hash(CountDomain(3, 2, minimum=2, maximum=20))
    assert asdict(value) == {
        "period": 3,
        "residue": 2,
        "exact": None,
        "minimum": 2,
        "maximum": 20,
    }


def test_value_arena_is_invocation_local_and_nested_analyses_share_it():
    with completion_math() as first:
        value = count_domain(7, 4, minimum=0)
        assert canonical(CountDomain(7, 4, minimum=3)) is value
        with completion_math() as nested:
            assert nested is first
            assert count_domain(7, 4, minimum=4) is value
        counts = completion_math_statistics()
        assert counts["values"] == counts["value_creations"] == 1
        assert counts["value_reuses"] == 2
    assert completion_math_statistics() == {}
    with completion_math() as second:
        assert second is not first
        assert count_domain(7, 4, minimum=0) == value
        assert count_domain(7, 4, minimum=0) is not value
        assert completion_math_statistics()["value_creations"] == 1
        assert completion_math_statistics()["intersection_pairs"] == 0


def test_repeated_relations_compute_once_and_match_literal_count_membership(monkeypatch):
    original, calls = completion._intersection, 0

    def compute(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(completion, "_intersection", compute)
    domains = (
        CountDomain(2, 0, minimum=0),
        CountDomain(2, 1, minimum=0),
        CountDomain(3, 1, minimum=2),
        CountDomain(2, 1, minimum=3, maximum=9),
        CountDomain(1, 0, exact=4),
    )
    with completion_math():
        for _ in range(20):
            for left in domains:
                for right in domains:
                    result = intersection(left, right)
                    for count in (*range(32), 10**12):
                        assert (result is not None and _allows(result, count)) == (
                            _allows(left, count) and _allows(right, count)
                        )
                    expected = all(
                        not _allows(right, count) or _allows(left, count) for count in range(64)
                    )
                    assert includes(left, right) == expected
        counts = completion_math_statistics()
        assert calls <= len(domains) ** 2
        assert counts["intersection_requests"] == 20 * len(domains) ** 2
        assert counts["intersection_reuses"] >= 19 * len(domains) ** 2
        assert counts["inclusion_reuses"] > 0


def test_top_level_compilation_arena_isolated_from_an_enclosing_analysis():
    with completion_math() as outer:
        value = count_domain(3, 1, minimum=0)
        before = completion_math_statistics()
        with completion_math(isolated=True) as inner:
            assert inner is not outer
            assert completion_math_statistics()["values"] == 0
            assert count_domain(3, 1, minimum=0) == value
            assert count_domain(3, 1, minimum=0) is not value
        assert completion_math_statistics() == before


def test_false_and_empty_relations_are_cached_without_widening_residue_holes():
    even, odd = CountDomain(2, 0, minimum=0), CountDomain(2, 1, minimum=0)
    with completion_math():
        for _ in range(16):
            assert intersection(even, odd) is None
            assert not includes(even, odd)
        counts = completion_math_statistics()
        assert counts["intersection_pairs"] == counts["inclusion_pairs"] == 1
        assert counts["intersection_reuses"] == counts["inclusion_reuses"] == 15


def test_repeated_construction_scales_with_distinct_mathematical_values():
    with completion_math():
        for _ in range(64):
            for lower in range(32):
                value = count_domain(1, 0, minimum=lower)
                assert canonical(value) is value
        counts = completion_math_statistics()
        assert counts["values"] == counts["value_creations"] == 32
        assert counts["value_requests"] == 2 * 64 * 32
        assert counts["value_reuses"] == counts["value_requests"] - 32


def test_arithmetic_scope_is_released_after_a_failed_analysis():
    with pytest.raises(RuntimeError, match="failed analysis"), completion_math():
        count_domain(3, 1, minimum=0)
        raise RuntimeError("failed analysis")
    assert completion_math_statistics() == {}


@pytest.mark.parametrize("shared", (False, True))
def test_equal_period_fast_path_keeps_the_coupled_period_budget(shared):
    value = CountDomain(MAX_PERIOD + 1, 0, minimum=0)
    if shared:
        with completion_math():
            with pytest.raises(CompileError, match="coupled period"):
                intersection(value, value)
            assert completion_math_statistics()["intersection_pairs"] == 0
    else:
        with pytest.raises(CompileError, match="coupled period"):
            intersection(value, value)

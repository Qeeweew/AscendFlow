"""Residue sweeps retain exact holes while avoiding repeated pair searches."""

import random

import pytest

from xdsl_npu import completion
from xdsl_npu.completion import CountDomain, coalesce


def test_singletons_bridge_families_and_extend_them_in_both_directions():
    pieces = (
        CountDomain(3, 1, minimum=4, maximum=7),
        CountDomain(3, 1, minimum=16, maximum=19),
        *(CountDomain(1, 0, exact=value) for value in (1, 10, 13, 22)),
    )
    assert coalesce(pieces) == (CountDomain(3, 1, minimum=1, maximum=22),)


def test_isolated_points_and_different_residues_do_not_fill_a_hole():
    pieces = (
        CountDomain(2, 0, minimum=0, maximum=2),
        CountDomain(2, 0, minimum=6, maximum=8),
        CountDomain(1, 0, exact=3),
        CountDomain(1, 0, exact=11),
    )
    assert coalesce(pieces) == pieces


@pytest.mark.parametrize("size", (16, 64, 512))
def test_sorted_union_computes_each_input_bound_a_constant_number_of_times(monkeypatch, size):
    original, calls = completion._first, 0

    def first(domain):
        nonlocal calls
        calls += 1
        return original(domain)

    monkeypatch.setattr(completion, "_first", first)
    pieces = tuple(CountDomain(1, 0, minimum=3 * i, maximum=3 * i + 1) for i in range(size))
    assert coalesce(reversed(pieces)) == tuple(reversed(pieces))
    assert calls == 2 * size


@pytest.mark.parametrize("seed", range(16))
def test_mixed_residue_unions_equal_independent_integer_membership(seed):
    rng = random.Random(seed)
    pieces = []
    for _ in range(30):
        if rng.randrange(3) == 0:
            pieces.append(CountDomain(1, 0, exact=rng.randrange(40)))
        else:
            period = rng.choice((1, 2, 3, 4, 5))
            lower = rng.randrange(20)
            pieces.append(
                CountDomain(
                    period,
                    rng.randrange(period),
                    minimum=lower,
                    maximum=rng.choice((None, lower + rng.randrange(16))),
                )
            )

    def contains(domain, value):
        return (
            value == domain.exact
            if domain.exact is not None
            else value >= domain.minimum
            and value % domain.period == domain.residue
            and (domain.maximum is None or value <= domain.maximum)
        )

    merged = coalesce(pieces)
    for value in (*range(80), 10**12):
        assert any(contains(domain, value) for domain in merged) == any(
            contains(domain, value) for domain in pieces
        )

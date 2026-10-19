"""Affine bounds retain nested metadata and modulo head coordinates."""

import pytest

from ascendflow.diagnostics import CompileError
from ascendflow.indexing import check_axis, domain_nonnegative
from ascendflow.scalar import Expr, Fact
from ascendflow.scalar import expression as e
from ascendflow.scalar import literal as c


@pytest.mark.parametrize("length_limit", [1, 64, 256])
def test_metadata_length_and_pitched_head_access(length_limit):
    start, length, row, owner, tokens = [
        Expr("argument", (name,)) for name in ("$start", "$length", "$row", "$owner", "TOKENS")
    ]
    domains = (
        (owner, c(0), c(39)),
        (start, c(0), e("sub", tokens, c(1))),
        (length, c(1), e("min", c(length_limit), e("sub", tokens, start))),
        (row, c(0), e("sub", length, c(1))),
    )
    facts = {"TOKENS": Fact(1)}
    token = e("add", start, row)
    check_axis(token, 1, tokens, domains, facts)
    head = e("rem", owner, c(32))
    offset = e("add", e("mul", token, c(8192)), e("mul", head, c(256)))
    remaining = e("sub", e("mul", tokens, c(8192)), e("add", offset, c(256)))
    assert domain_nonnegative(remaining, domains, facts)
    # Taking the row after the final permitted one has a real counterexample.
    with pytest.raises(CompileError, match="tensor extent"):
        check_axis(e("add", start, length), 1, tokens, domains, facts)
    assert not domain_nonnegative(e("sub", remaining, c(1)), domains, facts)


def test_nonlinear_occurrence_does_not_authorize_linear_substitution():
    row = Expr("argument", ("$row",))
    value = e("sub", e("mul", row, row), e("mul", c(2), row))
    assert not domain_nonnegative(value, ((row, c(0), c(3)),), {})

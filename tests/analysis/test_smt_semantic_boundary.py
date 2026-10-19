"""Arithmetic identities do not grant permission to evaluate source code."""

import pytest

from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.smt import IntegerQueryService
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.scalar import INT64_MAX, Expr, Fact, expression, literal

X = Expr("argument", ("x",))
ZERO = literal(0)


def test_math_zero_keeps_the_original_negative_unsigned_operand_illegal():
    queries = IntegerQueryService({"x": Fact(maximum=0)})
    numerator = expression("sub", X, literal(1))
    remainder = Expr("rem", (numerator, literal(1)))
    identity = Comparison(remainder, "==", ZERO)

    assert queries.equivalent(remainder, ZERO).proved
    assert queries.defined((identity,)).refuted
    with pytest.raises(CompileError, match="nonnegative numerator"):
        remainder.evaluate({"x": 0})

    # Consuming a mathematical relation must not silently add Def(remainder)
    # and thereby erase the only input assignment, x=0.
    positive = queries.implies(Comparison(X, ">", ZERO), (identity,))
    assert positive.refuted
    assert dict(positive.instance.values)["x"] == 0


def test_definedness_refuses_zero_divisor_even_for_a_reflexive_relation():
    divisor = Expr("argument", ("divisor",))
    quotient = Expr("floordiv", (X, divisor))
    queries = IntegerQueryService({"x": Fact(maximum=4), "divisor": Fact(maximum=0)})
    assert queries.equivalent(quotient, quotient).proved
    assert queries.defined((Comparison(quotient, "==", quotient),)).refuted


def test_short_circuit_last_requires_its_nonempty_prefix_first():
    queries = IntegerFacts({"x": Fact(maximum=4)})
    last = Expr("rem", (expression("sub", X, literal(1)), literal(2)))
    nonempty = Comparison(X, ">", ZERO)
    selected = Comparison(last, "==", ZERO)

    assert queries.ordered_definedness((nonempty, selected)) is None
    missing = queries.ordered_definedness((selected, nonempty))
    assert missing is not None
    requirement, prefix = missing
    assert prefix == ()
    assert not requirement.evaluate({"x": 0})


def test_integer_ring_identity_does_not_hide_an_overflowing_intermediate():
    increment = Expr("add", (X, literal(1)))
    recovered = Expr("add", (increment, literal(-1)))
    queries = IntegerQueryService({"x": Fact(minimum=INT64_MAX, maximum=INT64_MAX)})

    assert queries.equivalent(recovered, X).proved
    overflow = queries.machine_representable(recovered)
    assert overflow.refuted
    assert dict(overflow.instance.values)["x"] == INT64_MAX


def test_existence_witness_and_impossibility_have_distinct_meanings():
    queries = IntegerQueryService({"x": Fact(maximum=2)})
    witness = queries.satisfies((Comparison(X, ">", ZERO),))
    assert witness.satisfied
    assert 0 < dict(witness.instance.values)["x"] <= 2

    impossible = queries.satisfies((Comparison(X, ">", literal(2)),))
    assert impossible.impossible
    assert not impossible.satisfied


def test_a_query_cannot_borrow_a_different_premise_or_input_environment():
    inputs = {"x": Fact(maximum=4)}
    queries = IntegerQueryService(inputs)
    positive = Comparison(X, ">", ZERO)

    assert queries.implies(positive, (positive,)).proved
    assert queries.implies(positive).refuted
    inputs["x"] = Fact(minimum=1, maximum=4)
    assert queries.implies(positive).refuted
    assert IntegerQueryService(inputs).implies(positive).proved


@pytest.mark.parametrize("width,overlaps", ((4, False), (5, True)))
def test_adjacent_byte_intervals_match_independent_finite_overlap(width, overlaps):
    queries = IntegerQueryService({"x": Fact(maximum=7)})
    left = expression("mul", literal(8), X)
    right = expression("add", left, literal(4))
    actual = [
        bool(set(range(8 * x, 8 * x + width)) & set(range(8 * x + 4, 8 * x + 4 + width)))
        for x in range(8)
    ]
    assert all(actual) if overlaps else not any(actual)
    result = queries.disjoint(left, right, width)
    assert result.refuted if overlaps else result.proved


@pytest.mark.parametrize("power", (1, 7, 31, 63))
def test_unsigned_mask_identity_includes_values_above_signed_int64(power):
    queries = IntegerQueryService({})
    assert queries.unsigned_remainder_mask(power).proved
    divisor = 1 << power
    for bits in (0, 1, INT64_MAX, 1 << 63, (1 << 64) - 1):
        assert bits % divisor == bits & (divisor - 1)


@pytest.mark.parametrize("modulus", (2, 3, 5, 7))
def test_modular_relations_keep_correlated_integer_factors(modulus):
    y = Expr("argument", ("y",))
    queries = IntegerQueryService({"x": Fact(maximum=7), "y": Fact(maximum=7)})
    value = expression("mul", X, y)
    shifted = expression("add", value, expression("mul", literal(modulus), y))
    original_residue = Expr("rem", (value, literal(modulus)))
    shifted_residue = Expr("rem", (shifted, literal(modulus)))
    assert queries.equivalent(original_residue, shifted_residue).proved
    for x_value in range(8):
        for y_value in range(8):
            assert (x_value * y_value + modulus * y_value) % modulus == (
                x_value * y_value
            ) % modulus

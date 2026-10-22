"""Shared raw nodes retain ordered unsigned checks and prefix-local ownership."""

import sys
from types import CodeType

import pytest

from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.scalar import Expr, Fact, expression, literal

X, D, Y = (Expr("argument", (name,)) for name in ("x", "d", "y"))


def checked_calls(queries, tests):
    """Count actual traversal calls, including completed-node lookups."""
    checker = next(
        code
        for code in IntegerFacts.ordered_definedness.__wrapped__.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "check"
    )
    calls = 0
    previous = sys.getprofile()

    def profile(frame, event, argument):
        nonlocal calls
        if event == "call" and frame.f_code is checker:
            calls += 1
        if previous is not None:
            previous(frame, event, argument)

    try:
        sys.setprofile(profile)
        result = queries.ordered_definedness(tests)
    finally:
        sys.setprofile(previous)
    return result, calls


@pytest.mark.parametrize("levels", (0, 4, 8, 10))
def test_shared_raw_dag_work_follows_unique_nodes_across_both_operands(monkeypatch, levels):
    value = Expr("rem", (X, D))
    for _ in range(levels):
        value = Expr("add", (value, value))
    queries = IntegerFacts({"x": Fact(), "d": Fact(minimum=1)})
    original = queries.prove

    def prove(test, assumptions):
        if test.lhs is value and test.rhs is value:
            # The current == comparison cannot short-circuit as false. Keep
            # this traversal count independent of polynomial reconstruction.
            return False
        return original(test, assumptions)

    monkeypatch.setattr(queries, "prove", prove)
    result, calls = checked_calls(queries, (Comparison(value, "==", value),))
    assert result is None
    unique_nodes = levels + 3  # One add per level, then rem and its two arguments.
    assert unique_nodes <= calls <= 2 * unique_nodes


def test_shared_nodes_keep_definedness_through_prefix_extensions_only(monkeypatch):
    queries = IntegerFacts({"y": Fact()})
    value = Expr("rem", (X, D))
    nonnegative = Comparison(X, ">=", literal(0))
    positive = Comparison(D, ">", literal(0))
    first = Comparison(value, ">=", literal(0))
    extra = Comparison(Y, ">=", literal(0))
    second = Comparison(value, "<=", literal(0))
    prefixes = []
    original = queries.prove

    def observed(test, assumptions):
        if test == nonnegative or test == positive:
            prefixes.append(assumptions)
        return original(test, assumptions)

    monkeypatch.setattr(queries, "prove", observed)
    assert queries.ordered_definedness((nonnegative, positive, first, extra, second)) is None
    assert prefixes == [
        (nonnegative, positive),
        (nonnegative, positive),
    ]
    # The earlier checks imply definedness under the extended prefix. A
    # separate call still checks its own first use, without future premises.
    assert queries.ordered_definedness((first, nonnegative, positive)) == (nonnegative, ())
    assert queries.ordered_definedness((nonnegative, first, positive)) == (
        positive,
        (nonnegative,),
    )


def test_shared_dag_completion_never_crosses_fact_environments():
    numerator = expression("sub", X, literal(1))
    remainder = Expr("rem", (numerator, literal(1)))
    shared = Expr("add", (remainder, remainder))
    tests = (Comparison(shared, "==", literal(0)),)
    positive = IntegerFacts({"x": Fact(minimum=1)})
    nonnegative = IntegerFacts({"x": Fact()})
    assert positive.ordered_definedness(tests) is None
    assert nonnegative.ordered_definedness(tests) == (
        Comparison(numerator, ">=", literal(0)),
        (),
    )
    assert positive.ordered_definedness(tests) is None


@pytest.mark.parametrize("invalid_side", ("lhs", "rhs"))
def test_current_false_comparison_checks_each_original_unsigned_operand(invalid_side):
    queries = IntegerFacts({"x": Fact()})
    numerator = expression("sub", X, literal(1))
    safe = Expr("mul", (literal(0), Expr("rem", (X, literal(1)))))
    invalid = Expr("mul", (literal(0), Expr("rem", (numerator, literal(1)))))
    lhs, rhs = (invalid, safe) if invalid_side == "lhs" else (safe, invalid)
    use = Comparison(lhs, "!=", rhs)
    prerequisite = Comparison(X, ">=", literal(1))
    missing = Comparison(numerator, ">=", literal(0))
    false = Comparison(literal(0), "==", literal(1))
    assert queries.ordered_definedness((use,)) == (missing, ())
    assert queries.ordered_definedness((use, prerequisite)) == (missing, ())
    assert queries.ordered_definedness((use, false)) == (missing, ())
    assert queries.ordered_definedness((false, use)) is None
    assert queries.ordered_definedness((prerequisite, use)) is None


def test_shared_nested_divisions_preserve_the_first_missing_operand_and_exact_prefix():
    queries = IntegerFacts({"x": Fact(minimum=1)})
    inner = Expr("rem", (X, D))
    outer = Expr("floordiv", (inner, Y))
    shared = Expr("add", (outer, outer))
    use = Comparison(shared, "==", literal(0))
    positive_inner = Comparison(D, ">", literal(0))
    positive_outer = Comparison(Y, ">", literal(0))
    assert queries.ordered_definedness((use,)) == (positive_inner, ())
    assert queries.ordered_definedness((positive_inner, use)) == (
        positive_outer,
        (positive_inner,),
    )
    assert queries.ordered_definedness((positive_inner, positive_outer, use)) is None


def test_total_or_dominating_guard_values_need_no_guard_truth_search(monkeypatch):
    queries = IntegerFacts({})
    total = expression("add", X, literal(-1))
    partial = Expr("rem", (X, D))

    def no_search(*args):
        raise AssertionError("no new unsigned evaluation needs a proof")

    monkeypatch.setattr(queries, "prove", no_search)
    assert queries.ordered_definedness((Comparison(total, ">", literal(0)),)) is None
    assert (
        queries.ordered_definedness((Comparison(partial, ">", literal(0)),), frozenset((partial,)))
        is None
    )

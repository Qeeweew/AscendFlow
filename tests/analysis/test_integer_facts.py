"""Scoped integer queries checked against independent finite arithmetic.

All enumerated operations are defined throughout their small input domains.
Ordered unsigned-guard definedness belongs to the event-window regressions;
these tests check premise identity, fact isolation and sufficient answers.
"""

from itertools import product
from operator import eq, ge, gt, le, lt, ne

import pytest

from xdsl_npu.analysis import integer_facts
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import Expr, Fact, expression, literal

X, Y, D = (Expr("argument", (name,)) for name in ("x", "y", "d"))
PREDICATES = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}


@pytest.mark.parametrize("period", (2, 3, 7))
def test_whole_period_translation_keeps_the_inner_unsigned_domain(period):
    queries = IntegerFacts({})
    residue = expression("rem", X, literal(period))
    translated = expression("rem", expression("add", literal(3 * period), residue), literal(period))
    result = queries.bounded(translated)
    assert result == residue
    for value in range(3 * period):
        assert translated.evaluate({"x": value}) == result.evaluate({"x": value}) == value % period
    for value in (translated, result):
        with pytest.raises(CompileError, match="nonnegative numerator"):
            value.evaluate({"x": -1})
    # Removing a negative whole-period translation would make an undefined
    # unsigned remainder appear valid. This rule must retain that domain.
    negative = expression("rem", expression("sub", residue, literal(period)), literal(period))
    with pytest.raises(CompileError, match="nonnegative numerator"):
        queries.bounded(negative).evaluate({"x": 0})


def test_the_same_expression_cannot_borrow_another_environments_facts():
    small = IntegerFacts({"x": Fact(maximum=3)})
    large = IntegerFacts({"x": Fact(minimum=5, maximum=8)})
    unknown = IntegerFacts({})
    clipped = expression("min", X, literal(4))
    below_four = Comparison(X, "<", literal(4))

    assert small.bounded(clipped) == X
    assert small.bounds(X) == (0, 3)
    assert small.prove(below_four, ())
    assert large.bounded(clipped) == literal(4)
    assert large.bounds(X) == (5, 8)
    assert not large.prove(below_four, ())
    assert unknown.bounds(X) == (None, None)
    assert not unknown.prove(below_four, ())
    assert small.bounds(X) == (0, 3)
    assert small.prove(below_four, ())


def test_caller_mutation_cannot_change_hot_or_cold_queries_in_the_snapshot():
    inputs = {"x": Fact(maximum=3)}
    queries = IntegerFacts(inputs)
    below_four = Comparison(X, "<", literal(4))
    assert queries.bounds(X) == (0, 3)
    assert queries.prove(below_four, ())

    inputs["x"] = Fact(minimum=5, maximum=8)
    inputs["y"] = Fact(minimum=1, maximum=1)
    assert queries.bounds(X) == (0, 3)
    assert queries.prove(below_four, ())
    assert queries.bounded(expression("min", X, literal(4))) == X
    assert queries.bounds(Y) == (None, None)
    assert "y" not in queries.facts
    with pytest.raises(TypeError):
        queries.facts["x"] = inputs["x"]

    replacement = IntegerFacts(inputs)
    assert replacement.bounds(X) == (5, 8)
    assert replacement.bounds(Y) == (1, 1)
    assert not replacement.prove(below_four, ())


def test_changed_or_reordered_premises_get_their_own_query(monkeypatch):
    queries = IntegerFacts({})
    bounded = queries.bounded
    calls = []

    def observed(value):
        calls.append(value)
        return bounded(value)

    monkeypatch.setattr(queries, "bounded", observed)
    lower = Comparison(X, ">=", literal(2))
    upper = Comparison(X, "<=", literal(4))
    premises = (lower, upper)
    equivalent = tuple(Comparison(test.lhs, test.predicate, test.rhs) for test in premises)
    positive = Comparison(X, ">", literal(0))

    assert queries.bounds(X, premises) == (2, 4)
    count = len(calls)
    assert queries.bounds(X, equivalent) == (2, 4)
    assert len(calls) == count
    assert queries.bounds(X, premises[::-1]) == (2, 4)
    assert len(calls) > count

    assert queries.prove(positive, premises)
    count = len(calls)
    assert queries.prove(positive, equivalent)
    assert len(calls) == count
    assert queries.prove(positive, premises[::-1])
    assert len(calls) > count

    assert queries.bounds(X, (upper,)) == (None, 4)
    assert not queries.prove(positive, (upper,))
    assert queries.bounds(X, ()) == (None, None)
    assert not queries.prove(positive, ())
    assert queries.bounds(X, premises) == (2, 4)
    assert queries.prove(positive, premises)


def test_unknown_relations_and_false_predicates_never_become_proofs():
    queries = IntegerFacts({})
    equality = Comparison(X, "==", Y)
    assert queries.prove(equality, (equality,))
    assert not queries.prove(equality, ())
    assert not queries.prove(Comparison(X, ">=", literal(0)), ())
    assert not queries.prove(Comparison(X, "<", Y), ())
    for predicate in ("<", ">", "!="):
        assert not queries.prove(Comparison(X, predicate, X), ())
    assert not queries.prove(Comparison(X, ">=", expression("add", X, literal(1))), ())
    assert not queries.prove(Comparison(literal(1), "==", literal(2)), ())


def test_exact_repeated_queries_reuse_normalization_including_false_answers(monkeypatch):
    normalize = integer_facts.normalize
    calls = []

    def observed(value, facts):
        calls.append(value)
        return normalize(value, facts)

    monkeypatch.setattr(integer_facts, "normalize", observed)
    queries = IntegerFacts({"x": Fact(maximum=8)})
    value = expression("add", X, literal(3))
    premises = (Comparison(X, ">=", literal(2)), Comparison(X, "<=", literal(5)))
    impossible = Comparison(value, "==", literal(0))
    queries.normalize(value)
    queries.bounded(value)
    expected = queries.bounds(value, premises)
    assert expected[0] is not None and expected[0] <= 5
    assert expected[1] is not None and expected[1] >= 8
    assert not queries.prove(impossible, premises)
    count = len(calls)
    assert count > 0

    for _ in range(3):
        queries.normalize(value)
        queries.bounded(value)
        assert queries.bounds(value, premises) == expected
        assert not queries.prove(Comparison(value, "==", literal(0)), premises)
    assert len(calls) == count

    separate = IntegerFacts({"x": Fact(maximum=8)})
    assert separate.bounds(value, premises) == expected
    assert len(calls) > count


def test_signed_intermediate_clamps_keep_their_integer_bounds():
    extent, offset, step = (Expr("argument", (name,)) for name in ("extent", "offset", "step"))
    queries = IntegerFacts(
        {"extent": Fact(minimum=4, multiple=4, maximum=16), "offset": Fact(maximum=7)}
    )
    remaining = expression(
        "sub", expression("mul", literal(2), expression("floordiv", extent, literal(4))), offset
    )
    clamped = expression("max", remaining, literal(0))
    capped = expression("min", remaining, literal(3))
    premises = (Comparison(expression("sub", step, literal(1)), ">=", literal(0)),)
    original = [2 * (n // 4) - i for n, i in product(range(4, 17, 4), range(8))]
    assert min(original) < 0 < max(original)

    for value, actual, expected_lower, expected_upper in (
        (clamped, [max(n, 0) for n in original], 0, None),
        (capped, [min(n, 3) for n in original], None, 3),
    ):
        lower, upper = queries.bounds(value, premises)
        assert lower is not None and lower <= min(actual)
        assert upper is not None and max(actual) <= upper
        if expected_lower is not None:
            assert lower == expected_lower
        if expected_upper is not None:
            assert upper == expected_upper

    assert not queries.prove(Comparison(remaining, ">=", literal(0)), premises)
    assert queries.prove(Comparison(clamped, ">=", literal(0)), premises)
    assert not queries.prove(Comparison(clamped, ">", literal(0)), premises)
    assert queries.prove(Comparison(capped, "<=", literal(3)), premises)
    assert not queries.prove(Comparison(capped, ">=", literal(3)), premises)


@pytest.mark.parametrize(
    "kind,limit,expected,opposite,strict,evaluate",
    (
        ("max", 0, (0, None), "<=", ">", max),
        ("min", 3, (None, 3), ">=", "<", min),
    ),
)
def test_one_known_minmax_side_does_not_invent_the_opposite_bound(
    kind, limit, expected, opposite, strict, evaluate
):
    queries = IntegerFacts({})
    value = expression(kind, X, literal(limit))
    assert queries.bounds(X) == (None, None)
    assert queries.bounds(value) == expected
    actual = [evaluate(n, limit) for n in range(-4, 9)]
    lower, upper = expected
    assert lower is None or lower <= min(actual)
    assert upper is None or max(actual) <= upper
    assert any(not PREDICATES[opposite](n, limit) for n in actual)
    assert any(not PREDICATES[strict](n, limit) for n in actual)
    assert not queries.prove(Comparison(value, opposite, literal(limit)), ())
    assert not queries.prove(Comparison(value, strict, literal(limit)), ())


def test_nonnegative_product_preserves_the_factor_premise():
    count, extent = (Expr("argument", (name,)) for name in ("count", "extent"))
    queries = IntegerFacts(
        {"count": Fact(maximum=4), "extent": Fact(minimum=8, multiple=8, maximum=32)}
    )
    factor = expression("sub", count, literal(1))
    value = expression("mul", factor, expression("floordiv", extent, literal(8)))
    premises = (Comparison(factor, ">=", literal(0)),)
    actual = [(q, (q - 1) * (n // 8)) for q, n in product(range(5), range(8, 33, 8))]
    active = [result for q, result in actual if q - 1 >= 0]
    assert min(result for _, result in actual) < 0
    assert min(active) == 0 < max(active)
    lower, upper = queries.bounds(value, premises)
    assert lower == 0
    assert upper is not None and max(active) <= upper
    assert queries.prove(Comparison(value, ">=", literal(0)), premises)
    assert not queries.prove(Comparison(value, ">=", literal(0)), ())
    assert not queries.prove(Comparison(value, ">", literal(0)), premises)
    assert not queries.prove(Comparison(value, "<=", literal(0)), premises)


def quotient_value(kind, numerator, denominator):
    assert numerator >= 0 and denominator > 0
    return (
        numerator // denominator
        if kind == "floordiv"
        else (numerator + denominator - 1) // denominator
    )


@pytest.mark.parametrize("kind", ("floordiv", "ceildiv"))
@pytest.mark.parametrize("dynamic", (False, True), ids=("constant", "dynamic"))
def test_unsigned_quotient_of_signed_clamp_has_sound_finite_bounds(kind, dynamic):
    queries = IntegerFacts(
        {"x": Fact(maximum=8), "y": Fact(maximum=6), "d": Fact(minimum=2, maximum=4)}
    )
    numerator = expression("max", expression("sub", X, Y), literal(0))
    value = expression(kind, numerator, D if dynamic else literal(3))
    divisors = range(2, 5) if dynamic else (3,)
    actual = [
        quotient_value(kind, max(x - y, 0), d) for x, y, d in product(range(9), range(7), divisors)
    ]
    assert min(actual) == 0 < max(actual)
    lower, upper = queries.bounds(value)
    assert lower == 0
    assert upper is not None and max(actual) <= upper
    assert queries.prove(Comparison(value, ">=", literal(0)), ())
    assert not queries.prove(Comparison(value, ">", literal(0)), ())
    assert not queries.prove(Comparison(value, "<=", literal(0)), ())
    assert not queries.prove(Comparison(value, "<", literal(max(actual))), ())


@pytest.mark.parametrize("kind", ("floordiv", "ceildiv"))
def test_unbounded_divisor_cannot_supply_a_quotient_lower_bound_from_its_minimum(kind):
    queries = IntegerFacts({"x": Fact(maximum=8), "y": Fact(maximum=6), "d": Fact(minimum=2)})
    numerator = expression("max", expression("sub", X, Y), literal(3))
    value = expression(kind, numerator, D)
    actual = [
        quotient_value(kind, max(x - y, 3), d)
        for x, y, d in product(range(9), range(7), range(2, 17))
    ]
    expected_lower = 0 if kind == "floordiv" else 1
    assert min(actual) == expected_lower
    lower, upper = queries.bounds(value)
    assert lower == expected_lower
    assert upper is not None and max(actual) <= upper
    assert queries.prove(Comparison(value, ">=", literal(expected_lower)), ())
    assert not queries.prove(Comparison(value, ">", literal(expected_lower)), ())


@pytest.mark.parametrize("kind", ("floordiv", "ceildiv"))
def test_unbounded_numerator_does_not_gain_a_finite_quotient_upper_bound(kind):
    queries = IntegerFacts({"x": Fact(), "y": Fact(maximum=6), "d": Fact(minimum=2, maximum=4)})
    numerator = expression("max", expression("sub", X, Y), literal(0))
    value = expression(kind, numerator, D)
    actual = [
        quotient_value(kind, max(x - y, 0), d)
        for x, y, d in product((0, 1, 5, 40), range(7), range(2, 5))
    ]
    assert min(actual) == 0 and max(actual) > 8
    assert queries.bounds(value) == (0, None)
    assert queries.prove(Comparison(value, ">=", literal(0)), ())
    assert not queries.prove(Comparison(value, ">", literal(0)), ())
    assert not queries.prove(Comparison(value, "<=", literal(8)), ())


ARITHMETIC = (
    ("add", expression("add", X, Y), lambda v: v["x"] + v["y"]),
    ("mul", expression("mul", X, Y), lambda v: v["x"] * v["y"]),
    (
        "signed_mul",
        expression("mul", expression("sub", X, literal(2)), expression("sub", Y, literal(1))),
        lambda v: (v["x"] - 2) * (v["y"] - 1),
    ),
    (
        "min",
        expression("min", expression("add", X, literal(2)), expression("mul", literal(2), Y)),
        lambda v: min(v["x"] + 2, 2 * v["y"]),
    ),
    (
        "max",
        expression("max", X, expression("add", Y, literal(1))),
        lambda v: max(v["x"], v["y"] + 1),
    ),
    (
        "remainder",
        expression("rem", expression("add", expression("mul", literal(2), X), Y), literal(3)),
        lambda v: (2 * v["x"] + v["y"]) % 3,
    ),
    (
        "dynamic_remainder",
        expression("rem", expression("add", X, Y), D),
        lambda v: (v["x"] + v["y"]) % v["d"],
    ),
    (
        "mixed",
        expression(
            "max",
            expression("min", expression("add", X, Y), literal(5)),
            expression("rem", Y, literal(3)),
        ),
        lambda v: max(min(v["x"] + v["y"], 5), v["y"] % 3),
    ),
)


@pytest.mark.parametrize(
    "facts,domains",
    (
        (
            {"x": Fact(maximum=4), "y": Fact(maximum=4), "d": Fact(minimum=1, maximum=3)},
            (range(5), range(5), range(1, 4)),
        ),
        (
            {
                "x": Fact(minimum=2, multiple=2, maximum=6),
                "y": Fact(minimum=1, maximum=3),
                "d": Fact(minimum=2, maximum=3),
            },
            (range(2, 7, 2), range(1, 4), range(2, 4)),
        ),
    ),
    ids=("zero_inclusive", "declared_multiple"),
)
@pytest.mark.parametrize("name,value,evaluate", ARITHMETIC, ids=[row[0] for row in ARITHMETIC])
def test_small_integer_enumeration_checks_every_returned_bound_and_proof(
    facts, domains, name, value, evaluate
):
    queries = IntegerFacts(facts)
    contexts = [dict(zip(("x", "y", "d"), values, strict=True)) for values in product(*domains)]
    premise_cases = (
        ((), lambda v: True),
        (
            (Comparison(X, ">=", literal(2)), Comparison(Y, "<", literal(4))),
            lambda v: v["x"] >= 2 and v["y"] < 4,
        ),
        (
            (Comparison(X, "!=", literal(0)), Comparison(D, "==", literal(2))),
            lambda v: v["x"] != 0 and v["d"] == 2,
        ),
    )
    right_sides = (
        *((literal(n), lambda v, n=n: n) for n in (-32, 0, 3, 64)),
        (Y, lambda v: v["y"]),
    )
    proved, counterexamples = 0, 0
    for premises, admitted in premise_cases:
        active = [context for context in contexts if admitted(context)]
        assert active
        actual = [evaluate(context) for context in active]
        lower, upper = queries.bounds(value, premises)
        assert lower is None or lower <= min(actual), (name, premises, lower, actual)
        assert upper is None or max(actual) <= upper, (name, premises, upper, actual)
        for rhs, evaluate_rhs in right_sides:
            for predicate, compare in PREDICATES.items():
                test = Comparison(value, predicate, rhs)
                holds = all(compare(evaluate(context), evaluate_rhs(context)) for context in active)
                if queries.prove(test, premises):
                    assert holds, (name, premises, test, active)
                    proved += 1
                elif not holds:
                    counterexamples += 1
    assert proved and counterexamples

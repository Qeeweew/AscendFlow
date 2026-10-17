"""Scoped integer queries checked against independent finite arithmetic.

Enumerated arithmetic is defined throughout its small input domains. Ordered
guard queries separately check unsigned prerequisites before short-circuiting;
all queries retain premise identity and isolate their input facts.
"""

from itertools import product
from operator import eq, ge, gt, le, lt, ne

import pytest

from ascendflow.analysis import integer_facts
from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.scalar import Expr, Fact, expression, literal

X, Y, D = (Expr("argument", (name,)) for name in ("x", "y", "d"))
PREDICATES = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}


@pytest.mark.parametrize("predicate", ("==", "!="))
def test_relation_premise_and_query_share_clamp_and_division_normalization(predicate):
    queries = IntegerFacts({"y": Fact()})
    count = expression("ceildiv", expression("max", Y, literal(0)), literal(1))
    end = expression("sub", count, literal(1))
    shifted = expression("add", expression("sub", X, literal(1)), literal(1))
    source = Comparison(shifted, predicate, end)
    consumer = Comparison(X, predicate, end)
    assert queries.prove(source, (consumer,))
    assert queries.prove(consumer, (source,))
    assert not queries.prove(Comparison(X, predicate, count), (source,))
    for x, y in product(range(5), repeat=2):
        assert source.evaluate({"x": x, "y": y}) == consumer.evaluate({"x": x, "y": y})


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
    check = queries._check_context
    calls = []

    def observed(context):
        calls.append(context)
        return check(context)

    monkeypatch.setattr(queries, "_check_context", observed)
    lower = Comparison(X, ">=", literal(2))
    upper = Comparison(X, "<=", literal(4))
    premises = (lower, upper)
    equivalent = tuple(Comparison(test.lhs, test.predicate, test.rhs) for test in premises)
    positive = Comparison(X, ">", literal(0))
    context = queries._context(premises)
    reordered = queries._context(premises[::-1])

    assert queries.bounds(X, premises) == (2, 4)
    assert calls[-1] is context and context.tests is premises
    count = len(calls)
    assert queries.bounds(X, equivalent) == (2, 4)
    assert len(calls) == count
    assert queries.bounds(X, premises[::-1]) == (2, 4)
    assert len(calls) > count
    assert all(item is reordered for item in calls[count:])
    assert reordered is not context and reordered.tests == premises[::-1]

    count = len(calls)
    assert queries.prove(positive, premises)
    assert len(calls) > count
    assert all(item is context for item in calls[count:])
    count = len(calls)
    assert queries.prove(positive, equivalent)
    assert len(calls) == count
    assert queries.prove(positive, premises[::-1])
    assert len(calls) > count
    assert all(item is reordered for item in calls[count:])

    assert queries.bounds(X, (upper,)) == (None, 4)
    assert not queries.prove(positive, (upper,))
    assert queries.bounds(X, ()) == (None, None)
    assert not queries.prove(positive, ())
    assert queries.bounds(X, premises) == (2, 4)
    assert queries.prove(positive, premises)


@pytest.mark.parametrize("fixed", (3, 7))
@pytest.mark.parametrize("predicate,offset", (("<", 2), ("<=", 2), (">", -2), (">=", -2)))
def test_an_affine_prior_relation_keeps_its_parameter_premise(fixed, predicate, offset):
    queries = IntegerFacts({"y": Fact()})
    relation = Comparison(expression("add", X, literal(offset)), predicate, Y)
    equality = Comparison(Y, "==", literal(fixed))
    premises = (equality, relation)
    target = Comparison(X, predicate, literal(fixed - offset))
    direction = -1 if predicate in {"<", "<="} else 1
    stronger = Comparison(X, predicate, literal(fixed - offset + direction))
    changed = (Comparison(Y, "==", literal(fixed - direction)), relation)
    assert queries.prove(target, premises)
    assert not queries.prove(target, (relation,))
    assert not queries.prove(target, changed)
    assert not queries.prove(stronger, premises)
    assert queries.prove(target, premises)
    active = [
        {"x": x, "y": fixed} for x in range(-16, 17) if PREDICATES[predicate](x + offset, fixed)
    ]
    assert active and all(target.evaluate(context) for context in active)
    assert any(not stronger.evaluate(context) for context in active)


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
    retained = integer_facts.AtomIdentity.retained
    calls = []

    def observed(value, facts, *, identity=None):
        calls.append(value)
        return normalize(value, facts, identity=identity)

    def observed_retained(identity, value):
        calls.append(value)
        return retained(identity, value)

    monkeypatch.setattr(integer_facts, "normalize", observed)
    monkeypatch.setattr(integer_facts.AtomIdentity, "retained", observed_retained)
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


@pytest.mark.parametrize("kind", ("floordiv", "ceildiv", "rem"))
def test_ordered_definedness_cannot_borrow_a_later_unsigned_prerequisite(kind):
    queries = IntegerFacts({"x": Fact()})
    numerator = expression("sub", X, literal(1))
    value = expression(kind, numerator, literal(2))
    prerequisite = Comparison(X, ">=", literal(1))
    use = Comparison(value, ">=", literal(0))
    missing = Comparison(numerator, ">=", literal(0))
    assert queries.ordered_definedness((use, prerequisite)) == (missing, ())
    assert queries.ordered_definedness((prerequisite, use)) is None


@pytest.mark.parametrize("predicate", ("==", "!="))
def test_mathematical_remainder_by_one_cannot_hide_a_negative_numerator(predicate):
    queries = IntegerFacts({"x": Fact()})
    numerator = expression("sub", X, literal(1))
    remainder = expression("rem", numerator, literal(1))
    use = Comparison(remainder, predicate, literal(0))
    false_tail = Comparison(literal(0), "==", literal(1))
    # Algebraic answers assume defined operands; they cannot authorize
    # evaluating an invalid first expression in an empty conjunction.
    assert queries.prove(Comparison(remainder, "==", literal(0)), ())
    assert queries.ordered_definedness((use, false_tail)) == (
        Comparison(numerator, ">=", literal(0)),
        (),
    )
    with pytest.raises(CompileError, match="nonnegative numerator"):
        use.evaluate({"x": 0})
    assert queries.ordered_definedness((false_tail, use)) is None


def test_ordered_definedness_reports_the_first_nested_operand_and_exact_prefix():
    queries = IntegerFacts({"x": Fact(minimum=1)})
    inner = expression("rem", expression("sub", X, literal(1)), D)
    outer = expression("floordiv", inner, Y)
    use = Comparison(literal(0), "==", outer)
    positive_inner = Comparison(D, ">", literal(0))
    positive_outer = Comparison(Y, ">", literal(0))
    assert queries.ordered_definedness((use,)) == (positive_inner, ())
    assert queries.ordered_definedness((positive_inner, use)) == (
        positive_outer,
        (positive_inner,),
    )
    assert queries.ordered_definedness((positive_inner, positive_outer, use)) is None


def test_a_defined_false_comparison_short_circuits_the_unreachable_tail():
    queries = IntegerFacts({"x": Fact()})
    always_false = Comparison(expression("rem", X, literal(1)), "!=", literal(0))
    numerator = expression("sub", Y, literal(1))
    unreachable = Comparison(expression("rem", numerator, literal(1)), "==", literal(0))
    assert queries.ordered_definedness((always_false, unreachable)) is None
    assert queries.ordered_definedness((unreachable, always_false)) == (
        Comparison(numerator, ">=", literal(0)),
        (),
    )


def test_ordered_definedness_caches_the_complete_ordered_tuple_including_failures(monkeypatch):
    queries = IntegerFacts({})
    original = queries.prove
    calls = []

    def observed(test, assumptions):
        calls.append((test, assumptions))
        return original(test, assumptions)

    monkeypatch.setattr(queries, "prove", observed)
    nonnegative = Comparison(X, ">=", literal(0))
    positive = Comparison(D, ">", literal(0))
    use = Comparison(expression("rem", X, D), "==", literal(0))
    tests = (nonnegative, positive, use)
    assert queries.ordered_definedness(tests) is None
    count = len(calls)
    assert count
    equivalent = tuple(Comparison(test.lhs, test.predicate, test.rhs) for test in tests)
    assert queries.ordered_definedness(equivalent) is None
    assert len(calls) == count
    assert queries.ordered_definedness((positive, nonnegative, use)) is None
    assert len(calls) > count
    failure = queries.ordered_definedness((use, nonnegative, positive))
    assert failure == (nonnegative, ())
    count = len(calls)
    assert queries.ordered_definedness((use, nonnegative, positive)) is failure
    assert len(calls) == count


def test_ordered_definedness_cannot_borrow_another_environments_operand_facts():
    remainder = expression("rem", expression("sub", X, literal(1)), literal(1))
    tests = (Comparison(remainder, "!=", literal(0)),)
    positive = IntegerFacts({"x": Fact(minimum=1)})
    nonnegative = IntegerFacts({"x": Fact()})
    assert positive.ordered_definedness(tests) is None
    assert nonnegative.ordered_definedness(tests) == (
        Comparison(expression("sub", X, literal(1)), ">=", literal(0)),
        (),
    )
    assert positive.ordered_definedness(tests) is None


def test_an_unsupported_required_relation_remains_a_missing_unsigned_fact():
    queries = IntegerFacts({"x": Fact()})
    positive = Comparison(D, ">", literal(0))
    inner = expression("rem", X, D)
    numerator = expression("sub", inner, literal(1))
    use = Comparison(expression("rem", numerator, literal(1)), "!=", literal(0))
    assert queries.ordered_definedness((positive, use)) == (
        Comparison(numerator, ">=", literal(0)),
        (positive,),
    )
    with pytest.raises(CompileError, match="nonnegative numerator"):
        use.evaluate({"x": 0, "d": 1})


def test_ordered_definedness_propagates_analysis_budget_failures(monkeypatch):
    queries = IntegerFacts({})
    failure = CompileError("integer proof budget exhausted", kind=ErrorKind.ANALYSIS_LIMIT)

    def exhausted(*args):
        raise failure

    monkeypatch.setattr(queries, "prove", exhausted)
    use = Comparison(expression("rem", X, literal(1)), "==", literal(0))
    with pytest.raises(CompileError) as raised:
        queries.ordered_definedness((use,))
    assert raised.value is failure


def test_an_inactive_window_does_not_specialize_its_unreachable_count():
    from xdsl.dialects.builtin import ModuleOp

    from ascendflow.analysis.event_windows import build_event_window
    from ascendflow.analysis.structured_protocols import LogicalNotification, NotificationKind
    from ascendflow.target.ascend910b import Engine

    queries = IntegerFacts({"d": Fact()})
    loop = ModuleOp([])
    count = expression("ceildiv", literal(1), D)
    notification = LogicalNotification(
        "inactive_count",
        (),
        (),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        loop,
        Y,
        1,
        0,
    )
    guards = (
        Comparison(D, "==", literal(0)),
        Comparison(D, ">", literal(0)),
        Comparison(Y, ">=", literal(0)),
        Comparison(Y, "<", count),
    )
    window = build_event_window(
        notification,
        loop,
        ordinal=Y,
        count=count,
        guards=guards,
        facts=queries.facts,
        proves=queries.prove,
        bounds=queries.bounds,
        ordered_definedness=queries.ordered_definedness,
        invariant=lambda value: value != Y,
        parameter_bindings={D: literal(0)},
    )
    assert window.count == literal(0) and window.residues == () and window.upper == 0
    assert not window.membership.evaluate({}) and not window.project_exists().guard.evaluate({})


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


@pytest.mark.parametrize("constant_on_left", (True, False), ids=("left", "right"))
@pytest.mark.parametrize(
    "scalar,minimum,maximum,expected",
    (
        (4, 2, None, (8, None)),
        (-4, 2, None, (None, -8)),
        (4, None, -2, (None, -8)),
        (-4, None, -2, (8, None)),
        (4, None, 5, (None, 20)),
        (-4, None, 5, (-20, None)),
        (4, -3, 5, (-12, 20)),
        (-4, -3, 5, (-20, 12)),
        (0, None, None, (0, 0)),
        (0, 2, None, (0, 0)),
        (4, None, None, (None, None)),
        (-4, None, None, (None, None)),
    ),
)
def test_constant_multiplication_preserves_signed_and_unknown_interval_endpoints(
    constant_on_left, scalar, minimum, maximum, expected
):
    queries = IntegerFacts({})
    # Keep both operand orders, including zero, before scalar simplification.
    operands = (literal(scalar), X) if constant_on_left else (X, literal(scalar))
    value = Expr("mul", operands)
    premises = tuple(
        Comparison(X, predicate, literal(bound))
        for predicate, bound in ((">=", minimum), ("<=", maximum))
        if bound is not None
    )
    assert queries.bounds(value, premises) == expected
    actual = [
        scalar * x
        for x in range(-10, 11)
        if (minimum is None or x >= minimum) and (maximum is None or x <= maximum)
    ]
    lower, upper = expected
    assert lower is None or lower == min(actual)
    assert upper is None or upper == max(actual)
    if lower is None:
        assert not queries.prove(Comparison(value, ">=", literal(-100)), premises)
    if upper is None:
        assert not queries.prove(Comparison(value, "<=", literal(100)), premises)


@pytest.mark.parametrize("facts", ({}, {"x": Fact()}), ids=("unknown", "nonnegative"))
def test_a_lower_factor_premise_proves_the_scaled_positive_difference(facts):
    queries = IntegerFacts(facts)
    premises = (Comparison(X, ">=", literal(2)),)
    scaled = expression("mul", literal(4), X)
    difference = expression("sub", scaled, literal(7))
    assert queries.bounds(scaled, premises) == (8, None)
    assert queries.bounds(difference, premises) == (1, None)
    assert queries.prove(Comparison(difference, ">=", literal(0)), premises)
    assert not queries.prove(Comparison(difference, ">", literal(1)), premises)
    assert not queries.prove(Comparison(difference, ">=", literal(0)), ())


def test_remainder_interval_preserves_its_factored_unsigned_domain():
    queries = IntegerFacts({"x": Fact()})
    factor = expression("sub", Y, literal(1))
    scaled_factor = expression("mul", factor, X)
    phase = expression(
        "rem",
        expression("sub", literal(2), expression("rem", scaled_factor, literal(2))),
        literal(2),
    )
    count = expression("ceildiv", expression("max", X, literal(0)), literal(1))
    previous = expression("sub", count, literal(1))
    remainder = expression("rem", expression("sub", previous, phase), literal(2))
    aligned = expression("sub", previous, remainder)
    premises = (Comparison(Y, ">=", literal(1)), Comparison(count, ">", literal(1)))
    test = Comparison(count, ">", aligned)
    difference = expression("sub", count, aligned)
    bounded = queries.bounded(difference)
    assert queries.bounds(remainder, premises) == (0, 1)
    assert queries.bounds(bounded, premises) == (1, 2)
    assert queries.prove(test, premises)
    assert not queries.prove(test, premises[:1])
    assert not queries.prove(test, premises[1:])
    assert not queries.prove(Comparison(difference, ">", literal(2)), premises)
    for x, y in product(range(2, 9), range(1, 6)):
        context = {"x": x, "y": y}
        expected = x - (x - 1 - (x - 1 - (2 - (y - 1) * x % 2) % 2) % 2)
        assert test.evaluate(context)
        assert difference.evaluate(context) == bounded.evaluate(context) == expected


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


def _quotient_atom_arguments():
    p, s = (Expr("argument", (name,)) for name in ("p", "s"))
    quotient = expression("floordiv", s, literal(4))
    first = expression("sub", p, literal(1))
    second = expression("sub", expression("add", first, literal(1)), literal(1))
    return (
        {"s": Fact(minimum=1, multiple=4)},
        (Comparison(p, ">=", literal(1)),),
        expression("mul", first, quotient),
        expression("mul", second, quotient),
        quotient,
    )


def _atom_value(kind, numerator, divisor):
    return {
        "rem": numerator % divisor,
        "floordiv": numerator // divisor,
        "ceildiv": (numerator + divisor - 1) // divisor,
        "min": min(numerator, divisor),
        "max": max(numerator, divisor),
    }[kind]


def _assert_congruent_atom_queries(queries, lhs, rhs, premises):
    assert queries.ordered_definedness((*premises, Comparison(lhs, "==", rhs))) is None
    difference = expression("sub", lhs, rhs)
    assert queries.normalize(difference).constant == 0
    assert queries.bounded(difference).constant == 0
    for predicate in PREDICATES:
        assert queries.prove(Comparison(lhs, predicate, rhs), premises) is (
            predicate in {"==", "<=", ">="}
        )


@pytest.mark.parametrize("kind", ("rem", "floordiv", "ceildiv", "min", "max"))
def test_atom_identity_uses_congruent_arguments_with_a_declared_multiple_quotient(kind):
    facts, premises, first, second, _ = _quotient_atom_arguments()
    lhs, rhs = (expression(kind, value, literal(2)) for value in (first, second))
    for p, s in product(range(1, 9), range(4, 33, 4)):
        context = {"p": p, "s": s}
        expected = _atom_value(kind, (p - 1) * (s // 4), 2)
        assert lhs.evaluate(context) == rhs.evaluate(context) == expected
    _assert_congruent_atom_queries(IntegerFacts(facts), lhs, rhs, premises)


def test_nested_atom_identities_keep_each_unsigned_operation_and_clamp():
    facts, premises, first, second, _ = _quotient_atom_arguments()

    def nested(value):
        capped = expression("min", value, literal(17))
        divided = expression("floordiv", capped, literal(3))
        bounded = expression("max", divided, literal(2))
        rounded = expression("ceildiv", bounded, literal(3))
        return expression("rem", rounded, literal(2))

    lhs, rhs = nested(first), nested(second)
    actual = set()
    for p, s in product(range(1, 9), range(4, 33, 4)):
        value = (p - 1) * (s // 4)
        expected = ((max(min(value, 17) // 3, 2) + 2) // 3) % 2
        assert lhs.evaluate({"p": p, "s": s}) == rhs.evaluate({"p": p, "s": s}) == expected
        actual.add(expected)
    assert actual == {0, 1}
    _assert_congruent_atom_queries(IntegerFacts(facts), lhs, rhs, premises)


def test_atom_identity_uses_both_congruent_dynamic_arguments():
    facts, premises, first, second, quotient = _quotient_atom_arguments()
    facts["d"] = Fact(minimum=1)
    left_divisor = expression("mul", quotient, expression("add", D, literal(1)))
    right_divisor = expression("add", expression("mul", quotient, D), quotient)
    queries = IntegerFacts(facts)
    for kind in ("rem", "floordiv", "ceildiv"):
        lhs = expression(kind, first, left_divisor)
        rhs = expression(kind, second, right_divisor)
        for p, s, d in product(range(1, 5), range(4, 17, 4), range(1, 4)):
            value, divisor = (p - 1) * (s // 4), (s // 4) * (d + 1)
            expected = _atom_value(kind, value, divisor)
            context = {"p": p, "s": s, "d": d}
            assert lhs.evaluate(context) == rhs.evaluate(context) == expected
        _assert_congruent_atom_queries(queries, lhs, rhs, premises)


@pytest.mark.parametrize("difference", ("numerator", "divisor", "kind"))
def test_distinct_atom_arguments_or_kinds_keep_a_finite_counterexample(difference):
    facts, premises, value, _, _ = _quotient_atom_arguments()
    if difference == "kind":
        lhs = expression("floordiv", value, literal(2))
        rhs = expression("ceildiv", value, literal(2))
        p = 2
    else:
        lhs = expression("rem", value, literal(2))
        rhs = expression(
            "rem",
            expression("add", value, literal(1)) if difference == "numerator" else value,
            literal(3 if difference == "divisor" else 2),
        )
        p = 3 if difference == "divisor" else 2
    context = {"p": p, "s": 4}
    assert lhs.evaluate(context) != rhs.evaluate(context)
    queries = IntegerFacts(facts)
    assert queries.ordered_definedness((*premises, Comparison(lhs, "==", rhs))) is None
    for predicate, compare in PREDICATES.items():
        if not compare(lhs.evaluate(context), rhs.evaluate(context)):
            assert not queries.prove(Comparison(lhs, predicate, rhs), premises)


def test_atom_identity_cannot_borrow_another_environments_declared_multiple():
    s = Expr("argument", ("s",))
    rounded = expression("mul", expression("floordiv", s, literal(4)), literal(4))
    lhs, rhs = (expression("rem", value, literal(2)) for value in (rounded, s))
    equality = Comparison(lhs, "==", rhs)
    aligned = IntegerFacts({"s": Fact(minimum=4, multiple=4)})
    unaligned = IntegerFacts({"s": Fact(minimum=4)})
    assert aligned.prove(equality, ())
    assert lhs.evaluate({"s": 5}) != rhs.evaluate({"s": 5})
    assert not unaligned.prove(equality, ())
    assert aligned.prove(equality, ())


@pytest.mark.parametrize("wrapper", ("zero_product", "remainder_one"))
@pytest.mark.parametrize("predicate", ("==", "!="))
def test_raw_atom_domain_survives_mathematical_zero_and_false_comparisons(wrapper, predicate):
    facts, premises, numerator, _, _ = _quotient_atom_arguments()
    facts["p"] = Fact()
    queries = IntegerFacts(facts)
    inner = expression("rem", numerator, literal(2))
    value = (
        Expr("mul", (literal(0), inner))
        if wrapper == "zero_product"
        else expression("rem", inner, literal(1))
    )
    use = Comparison(value, predicate, literal(0))
    missing = Comparison(numerator, ">=", literal(0))
    false = Comparison(literal(0), "==", literal(1))
    assert queries.prove(Comparison(value, "==", literal(0)), ())
    assert queries.ordered_definedness((use,)) == (missing, ())
    assert queries.ordered_definedness((use, *premises)) == (missing, ())
    assert queries.ordered_definedness((use, false)) == (missing, ())
    assert queries.ordered_definedness((false, use)) is None
    assert queries.ordered_definedness((*premises, use)) is None
    with pytest.raises(CompileError, match="nonnegative numerator"):
        use.evaluate({"p": 0, "s": 4})


def test_a_congruent_atom_false_prefix_can_skip_only_its_later_undefined_tail():
    facts, premises, first, second, _ = _quotient_atom_arguments()
    facts["d"] = Fact()
    lhs, rhs = (expression("rem", value, literal(2)) for value in (first, second))
    false = Comparison(lhs, "!=", rhs)
    numerator = expression("sub", D, literal(1))
    tail = Comparison(expression("rem", numerator, literal(1)), "==", literal(0))
    queries = IntegerFacts(facts)
    assert queries.ordered_definedness((*premises, false, tail)) is None
    assert queries.ordered_definedness((*premises, tail, false)) == (
        Comparison(numerator, ">=", literal(0)),
        premises,
    )


def test_shared_function_identity_never_borrows_another_queries_raw_domain():
    numerator = expression("sub", X, literal(1))
    undefined = expression("rem", numerator, literal(1))
    bad = Expr("rem", (Expr("mul", (literal(0), undefined)), literal(2)))
    good = Expr("rem", (literal(0), literal(2)))
    queries = IntegerFacts({"x": Fact()})
    assert queries.bounded(expression("sub", bad, good)).constant == 0
    for original in (bad, good, bad):
        assert queries.bounded(original) is original
    assert queries.ordered_definedness((Comparison(good, "==", literal(0)),)) is None
    assert queries.ordered_definedness((Comparison(bad, "==", literal(0)),)) == (
        Comparison(numerator, ">=", literal(0)),
        (),
    )
    with pytest.raises(CompileError, match="nonnegative numerator"):
        queries.bounded(bad).evaluate({"x": 0})

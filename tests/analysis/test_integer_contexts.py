"""Exact ordered premise ownership and constant-size recursive query keys."""

from dataclasses import FrozenInstanceError

import pytest

from xdsl_npu.analysis import integer_facts
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, Fact, expression, literal

X, Y = (Expr("argument", (name,)) for name in ("x", "y"))


def test_contexts_keep_exact_order_duplicates_and_original_expressions(monkeypatch):
    queries = IntegerFacts({})
    lower = Comparison(X, ">=", literal(1))
    use = Comparison(
        expression("rem", expression("sub", X, literal(1)), literal(1)), ">=", literal(0)
    )
    premises = (lower, use)

    def no_eager_query(*args):
        pytest.fail("premise context construction evaluated an arithmetic rule")

    for method in ("bounded", "normalize", "infer"):
        monkeypatch.setattr(queries, method, no_eager_query)
    context = queries._context(premises)
    equivalent = tuple(Comparison(t.lhs, t.predicate, t.rhs) for t in premises)
    assert queries._context(equivalent) is context
    assert queries._context(premises[::-1]) is not context
    assert queries._context((lower, lower, use)) is not context
    assert context.tests is premises and context.tests[1].lhs is use.lhs
    assert queries.prove(use, premises)
    with pytest.raises(FrozenInstanceError):
        context.tests = ()


def test_false_results_reuse_equal_sequences_but_not_new_or_reordered_premises(monkeypatch):
    queries = IntegerFacts({})
    lower, upper = Comparison(X, ">=", literal(1)), Comparison(X, "<=", literal(4))
    premises = (lower, upper)
    test = Comparison(X, "<", literal(0))
    calls = []
    check = queries._check_context

    def observed(context):
        calls.append(context)
        return check(context)

    monkeypatch.setattr(queries, "_check_context", observed)
    assert not queries.prove(test, premises)
    count = len(calls)
    assert count > 0
    equivalent = tuple(Comparison(t.lhs, t.predicate, t.rhs) for t in premises)
    assert not queries.prove(Comparison(test.lhs, test.predicate, test.rhs), equivalent)
    assert len(calls) == count
    assert not queries.prove(test, premises[::-1])
    assert len(calls) > count
    count = len(calls)
    assert not queries.prove(test, (upper,))
    assert len(calls) > count
    assert queries.bounds(X, premises) == (1, 4)
    assert queries.bounds(X, (upper,)) == (None, 4)
    assert queries.bounds(X, ()) == (None, None)


def test_default_and_explicit_empty_bounds_use_one_core_result(monkeypatch):
    queries = IntegerFacts({})
    calls = []
    infer = queries.infer

    def observed(value):
        calls.append(value)
        return infer(value)

    monkeypatch.setattr(queries, "infer", observed)
    assert queries.bounds(X) == (None, None)
    assert calls == [X]
    assert queries.bounds(X, ()) == (None, None)
    assert calls == [X]
    assert queries._context(()).tests == ()


@pytest.mark.parametrize("query", ("bounds", "prove"))
def test_recursion_reuses_one_owned_context_without_reentering_public_tuple_api(monkeypatch, query):
    queries = IntegerFacts({})
    premises = (Comparison(X, ">=", literal(1)),)
    interned, checked = [], []
    context, check = queries._context, queries._check_context

    def observed_context(tests):
        interned.append(tests)
        return context(tests)

    def observed_check(value):
        checked.append(value)
        return check(value)

    monkeypatch.setattr(queries, "_context", observed_context)
    monkeypatch.setattr(queries, "_check_context", observed_check)
    if query == "bounds":
        assert queries.bounds(expression("add", X, Y), premises) == (None, None)
    else:
        assert not queries.prove(Comparison(X, "==", Y), premises)
    assert interned == [premises]
    assert len(checked) > 1 and all(value is checked[0] for value in checked)
    assert checked[0].tests is premises


@pytest.mark.parametrize("size", (0, 1, 64))
def test_recursive_bounds_keys_do_not_hash_any_premise_comparison(monkeypatch, size):
    queries = IntegerFacts({})
    premises = tuple(
        Comparison(Expr("argument", (f"p{index}",)), ">=", literal(0)) for index in range(size)
    )
    context = queries._context(premises)
    value = expression("add", X, Y)
    hashes = 0
    original = Comparison.__hash__

    def observed(comparison):
        nonlocal hashes
        hashes += 1
        return original(comparison)

    monkeypatch.setattr(Comparison, "__hash__", observed)
    assert queries._bounds(value, context) == (None, None)
    assert queries._bounds(value, context) == (None, None)
    assert hashes == 0


@pytest.mark.parametrize("query,method", (("bounds", "infer"), ("prove", "bounded")))
def test_failed_core_queries_are_retried_and_never_cached_as_unknown(monkeypatch, query, method):
    queries = IntegerFacts({})
    original = getattr(queries, method)
    failure = CompileError("test integer proof budget", kind=ErrorKind.ANALYSIS_LIMIT)
    attempts = []

    def failed(value):
        attempts.append(value)
        raise failure

    monkeypatch.setattr(queries, method, failed)
    argument = X if query == "bounds" else Comparison(X, ">=", literal(0))
    for _ in range(2):
        with pytest.raises(CompileError) as caught:
            getattr(queries, query)(argument, ())
        assert caught.value is failure
    assert len(attempts) == 2
    monkeypatch.setattr(queries, method, original)
    result = getattr(queries, query)(argument, ())
    assert result == ((None, None) if query == "bounds" else False)


def test_foreign_contexts_are_rejected_without_importing_another_environment():
    small = IntegerFacts({"x": Fact(maximum=3)})
    large = IntegerFacts({"x": Fact(minimum=5, maximum=8)})
    unknown = IntegerFacts({})
    context = small._context(())
    test = Comparison(X, "<", literal(4))
    assert small.bounds(X) == (0, 3) and small.prove(test, ())
    for owner in (large, unknown):
        for _ in range(2):
            with pytest.raises(ValueError, match="another owner"):
                owner._bounds(X, context)
            with pytest.raises(ValueError, match="another owner"):
                owner._prove(test, context)
        with pytest.raises(TypeError, match="immutable tuple"):
            owner.bounds(X, context)
        with pytest.raises(TypeError, match="immutable tuple"):
            owner.prove(test, context)
    assert large.bounds(X) == (5, 8) and not large.prove(test, ())
    assert unknown.bounds(X) == (None, None) and not unknown.prove(test, ())
    assert small.bounds(X) == (0, 3) and small.prove(test, ())


def test_ordered_definedness_keeps_public_tuple_prefixes_and_first_missing_fact(monkeypatch):
    queries = IntegerFacts({"x": Fact()})
    numerator = expression("sub", X, literal(1))
    use = Comparison(expression("rem", numerator, literal(1)), ">=", literal(0))
    prerequisite = Comparison(X, ">=", literal(1))
    seen = []
    prove = queries.prove

    def observed(test, assumptions):
        assert isinstance(assumptions, tuple)
        seen.append(assumptions)
        return prove(test, assumptions)

    monkeypatch.setattr(queries, "prove", observed)
    assert queries.ordered_definedness((use, prerequisite)) == (
        Comparison(numerator, ">=", literal(0)),
        (),
    )
    assert seen == [()]
    assert queries.ordered_definedness((prerequisite, use)) is None
    assert (prerequisite,) in seen


def test_derived_constructions_are_reused_across_distinct_ordered_premises(monkeypatch):
    queries = IntegerFacts({})
    threshold = literal(2)
    test = Comparison(X, ">", threshold)
    lower, extra = Comparison(X, ">=", literal(3)), Comparison(Y, ">=", literal(0))
    original_expression = integer_facts.expression
    constructions = []

    def observed(kind, lhs, rhs):
        if kind == "sub" and lhs is X and rhs is threshold:
            constructions.append((lhs, rhs))
        return original_expression(kind, lhs, rhs)

    monkeypatch.setattr(integer_facts, "expression", observed)
    assert queries.prove(test, (lower,))
    constraint = queries._constraint(test)
    difference = queries._difference(X, threshold)
    assert len(constructions) == 2
    for premises in ((lower,), (lower, extra), (extra, lower)):
        assert queries.prove(test, premises)
        assert queries._constraint(Comparison(X, ">", literal(2))) is constraint
        assert queries._difference(X, literal(2)) is difference
    assert len(constructions) == 2


def test_constraint_keeps_the_raw_factor_and_unsigned_domain_without_reduction(monkeypatch):
    queries = IntegerFacts({})
    numerator = expression("sub", X, literal(1))
    remainder = Expr("rem", (numerator, literal(1)))
    factor = Expr("add", (remainder, Y))
    value = expression("mul", numerator, factor)
    test = Comparison(value, ">", literal(0))
    expected = expression("sub", value, literal(1))

    def no_reduction(*args):
        pytest.fail("a raw constraint performed integer reduction")

    monkeypatch.setattr(queries, "bounded", no_reduction)
    monkeypatch.setattr(queries, "normalize", no_reduction)
    constraint = queries._constraint(test)
    assert constraint == expected and value in constraint.args
    assert any(arg is numerator for arg in value.args)
    assert any(arg is factor for arg in value.args)
    assert factor.args[0] is remainder and remainder.args[0] is numerator
    with pytest.raises(CompileError, match="nonnegative numerator"):
        constraint.evaluate({"x": 0, "y": 2})
    assert constraint.evaluate({"x": 3, "y": 2}) == 3


def test_difference_uses_exactly_one_bounded_raw_subtraction_layer(monkeypatch):
    queries = IntegerFacts({})
    factor = expression("mul", expression("sub", X, literal(1)), Y)
    expected = expression("sub", factor, literal(2))
    marker = Expr("argument", ("bounded_result",))
    seen = []

    def bounded(value):
        seen.append(value)
        return marker

    def no_normalization(*args):
        pytest.fail("difference added another integer normalization layer")

    monkeypatch.setattr(queries, "bounded", bounded)
    monkeypatch.setattr(queries, "normalize", no_normalization)
    assert queries._difference(factor, literal(2)) is marker
    assert queries._difference(factor, literal(2)) is marker
    assert seen == [expected] and factor in seen[0].args


def test_derived_queries_keep_raw_structural_keys_and_do_not_cross_fact_owners(monkeypatch):
    small = IntegerFacts({"x": Fact(maximum=3)})
    large = IntegerFacts({"x": Fact(minimum=5, maximum=8)})
    clipped = expression("min", X, literal(4))
    zero = literal(0)
    assert small._difference(clipped, zero) == X
    assert large._difference(clipped, zero) == literal(4)
    assert small._difference(clipped, zero) == X
    safe = Expr("rem", (X, literal(1)))
    unsafe = Expr("rem", (expression("sub", X, literal(1)), literal(1)))
    original_expression = integer_facts.expression
    constructions = []

    def observed(kind, lhs, rhs):
        if kind == "sub" and (lhs is safe or lhs is unsafe) and rhs is zero:
            constructions.append(lhs)
        return original_expression(kind, lhs, rhs)

    monkeypatch.setattr(integer_facts, "expression", observed)
    assert small._difference(safe, zero) == literal(0)
    assert small._difference(unsafe, zero) == literal(0)
    assert constructions == [safe, unsafe]
    assert large._difference(safe, zero) == literal(0)
    assert constructions == [safe, unsafe, safe]
    for value in (safe, unsafe):
        constraint = small._constraint(Comparison(value, ">=", zero))
        assert constraint is value
    assert small.ordered_definedness((Comparison(unsafe, "==", zero),)) == (
        Comparison(unsafe.args[0], ">=", zero),
        (),
    )


@pytest.mark.parametrize("method", ("_constraint", "_difference"))
def test_derived_construction_failures_are_retried_without_a_cached_answer(monkeypatch, method):
    queries = IntegerFacts({})
    zero = literal(0)
    test = Comparison(X, ">=", zero)
    failure = CompileError("test derived-expression budget", kind=ErrorKind.ANALYSIS_LIMIT)
    attempts = []
    original = integer_facts.expression

    def failed(kind, lhs, rhs):
        if kind == "sub" and lhs is X and rhs is zero:
            attempts.append((lhs, rhs))
            raise failure
        return original(kind, lhs, rhs)

    monkeypatch.setattr(integer_facts, "expression", failed)
    arguments = (test,) if method == "_constraint" else (X, zero)
    for _ in range(2):
        with pytest.raises(CompileError) as caught:
            getattr(queries, method)(*arguments)
        assert caught.value is failure
    assert len(attempts) == 2
    monkeypatch.setattr(integer_facts, "expression", original)
    assert getattr(queries, method)(*arguments) == X

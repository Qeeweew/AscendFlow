"""Invocation-local mathematical facts preserve ordered arithmetic witnesses."""

from dataclasses import FrozenInstanceError
from itertools import permutations, product
from operator import ge, gt, le, lt

import pytest

from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.progress_conditions import ProgressConditions
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, Fact, expression, literal

X = Expr("argument", ("x",))


@pytest.mark.parametrize("contradiction", (False, True))
def test_permutations_and_duplicates_share_one_inconsistency_proof(contradiction, monkeypatch):
    arithmetic = IntegerFacts({})
    calls = []
    prove = arithmetic.prove

    def observed(test, premises):
        calls.append((test, premises))
        return prove(test, premises)

    monkeypatch.setattr(arithmetic, "prove", observed)
    owner = ProgressConditions(arithmetic)
    lower = Comparison(X, ">=", literal(3 if contradiction else 0))
    upper = Comparison(X, "<", literal(3 if contradiction else 4))
    tests = (lower, upper)
    condition = owner.intern(tests)
    assert owner.impossible(tests) is contradiction
    proof_calls = len(calls)
    assert proof_calls == owner.proof_queries and proof_calls > 0
    for order in permutations((lower, upper, lower)):
        assert owner.intern(order) is condition
        assert owner.impossible(order) is contradiction
    assert len(calls) == proof_calls == owner.proof_queries
    assert owner.condition_count == 1 and owner.cache_hits == 6
    independent = ProgressConditions(IntegerFacts({})).intern(reversed(tests))
    assert independent is not condition and independent.predicates == condition.predicates


def test_interned_conditions_are_immutable_and_empty_queries_reuse_their_fact():
    owner = ProgressConditions(IntegerFacts({}))
    lower = Comparison(X, ">=", literal(0))
    upper = Comparison(X, "<=", literal(4))
    empty = owner.intern(())
    both = owner.intern((lower, upper))
    assert both.members == frozenset((lower, upper))
    assert owner.intern((upper, lower)) is both
    assert owner.condition_count == 2 and owner.proof_queries == owner.cache_hits == 0
    with pytest.raises(FrozenInstanceError):
        both.predicates = ()
    for name in ("condition_count", "proof_queries", "cache_hits"):
        with pytest.raises(AttributeError):
            setattr(owner, name, 0)
    assert not owner.impossible(empty)
    assert not owner.impossible(())
    assert owner.proof_queries == 0 and owner.cache_hits == 1


def test_conditions_imported_from_another_scope_cannot_borrow_its_facts():
    test = Comparison(X, "<", literal(4))
    small = ProgressConditions(IntegerFacts({"x": Fact(maximum=3)}))
    large = ProgressConditions(IntegerFacts({"x": Fact(minimum=5, maximum=8)}))
    unknown = ProgressConditions(IntegerFacts({}))
    condition = small.intern((test,))
    assert not small.impossible(condition)
    assert large.impossible(condition)
    assert not unknown.impossible(condition)
    assert large.intern(condition) is not condition
    assert small.proof_queries == large.proof_queries == unknown.proof_queries == 1
    assert not small.impossible(condition)
    assert large.impossible(condition)
    assert small.cache_hits == large.cache_hits == 1


def test_mathematical_context_never_rewrites_the_admitted_unsigned_witness(monkeypatch):
    arithmetic = IntegerFacts({"x": Fact(maximum=4)})
    original = expression("floordiv", expression("sub", X, literal(1)), literal(1))
    lower = Comparison(X, ">=", literal(1))
    read = Comparison(original, ">=", literal(0))
    upper = Comparison(X, "<=", literal(4))
    tests = (lower, read, upper)
    guard = ActionGuard(tests)
    assert arithmetic.ordered_definedness(tests) is None
    assert arithmetic.ordered_definedness((read, lower, upper)) is not None

    def no_second_admission(*args):
        pytest.fail("mathematical condition interning repeated endpoint admission")

    monkeypatch.setattr(arithmetic, "ordered_definedness", no_second_admission)
    owner = ProgressConditions(arithmetic)
    condition = owner.intern(guard.comparisons)
    assert not owner.impossible(condition)
    assert condition.predicates != tests
    assert next(test for test in condition.predicates if test is read).lhs is original
    assert original.kind == "floordiv"
    assert guard.comparisons is tests and guard.comparisons[1] is read
    assert not guard.evaluate({"x": 0}) and guard.evaluate({"x": 1})
    with pytest.raises(CompileError, match="nonnegative numerator"):
        original.evaluate({"x": 0})


def test_unknown_stays_possible_and_budget_failures_do_not_become_cached_answers(monkeypatch):
    arithmetic = IntegerFacts({})
    owner = ProgressConditions(arithmetic)
    test = Comparison(X, ">=", literal(0))
    assert not owner.impossible((test,))
    assert not owner.impossible((test,))
    assert owner.proof_queries == 1 and owner.cache_hits == 1
    failure = CompileError("test proof budget exhausted", kind=ErrorKind.ANALYSIS_LIMIT)

    def failed(*args):
        raise failure

    monkeypatch.setattr(arithmetic, "prove", failed)
    other = Comparison(X, "<", literal(0))
    for _ in range(2):
        with pytest.raises(CompileError) as caught:
            owner.impossible((other,))
        assert caught.value is failure
    assert owner.proof_queries == 3 and owner.cache_hits == 1


def test_interval_contradictions_match_an_independent_finite_integer_oracle():
    predicates = {"<": lt, "<=": le, ">": gt, ">=": ge}
    bounds = tuple(product(predicates, (0, 2, 4)))
    owner = ProgressConditions(IntegerFacts({"x": Fact(maximum=4)}))
    answers = set()
    for (left, a), (right, b) in product(bounds, repeat=2):
        expected = not any(
            predicates[left](value, a) and predicates[right](value, b) for value in range(5)
        )
        tests = (Comparison(X, left, literal(a)), Comparison(X, right, literal(b)))
        actual = owner.impossible(tests)
        assert actual == expected, (left, a, right, b)
        answers.add(actual)
    assert answers == {False, True}

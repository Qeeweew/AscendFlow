"""Finite rule implications keep exact evidence and isolated input scopes."""

import pytest

from ascendflow.analysis.smt import IntegerQueryService, all_of, any_of, negate
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.scalar import Expr, literal


def proposition(name):
    return Comparison(Expr("argument", (name,)), "==", literal(0))


def test_supported_implication_is_owned_without_leaking_temporary_axioms():
    p, q, other = map(proposition, ("p", "q", "other"))
    queries = IntegerQueryService({})
    rules = (p, any_of(negate(p), q), other)
    with queries.rule_system(rules).assuming() as context:
        result = context.prove(q)
        assert result.answer.proved
        assert queries.owns_implication(result.answer, result.condition, ())
        assert IntegerQueryService({}).implies(result.condition).proved
        assert not queries.implies(q).proved
        assert context.prove(q) is result
    assert not queries.implies(q).proved
    with pytest.raises(CompileError, match="context is closed"):
        context.prove(q)


def test_rule_cycle_needs_an_actual_seed_and_preserves_complete_premises():
    p, q, active = map(proposition, ("p", "q", "active"))
    rules = (any_of(negate(p), q), any_of(negate(q), p), any_of(negate(active), p))
    queries = IntegerQueryService({})
    with queries.rule_system(rules).assuming((active,)) as context:
        strong = context.prove(q)
        assert strong.answer.proved
        assert queries.owns_implication(strong.answer, strong.condition, (active,))
        assert not queries.owns_implication(strong.answer, strong.condition, ())
    with queries.rule_system(rules).assuming() as context:
        assert not context.prove(q).answer.proved
    with queries.rule_system(rules[:-1]).assuming((active,)) as context:
        assert not context.prove(q).answer.proved


def test_rule_scope_uses_its_own_assertions_even_during_other_query_scopes():
    p, q = map(proposition, ("p", "q"))
    queries = IntegerQueryService({})
    queries._solver.push()
    try:
        queries._solver.add(False)
        with queries.rule_system((p, any_of(negate(p), q))).assuming() as context:
            assert context.prove(q).answer.proved
            assert not context.prove(negate(q)).answer.proved
    finally:
        queries._solver.pop()
    assert not queries.implies(q).proved


@pytest.mark.parametrize(
    "rules,assumptions,conclusion", [((0,), (), True), ((), (1,), True), ((), (), 0)]
)
def test_rule_queries_reject_numeric_booleans(rules, assumptions, conclusion):
    queries = IntegerQueryService({})
    with (
        pytest.raises(CompileError, match="mathematical|immutable"),
        queries.rule_system(rules).assuming(assumptions) as context,
    ):
        context.prove(conclusion)


def test_deep_negation_preserves_shared_formula_and_original_order():
    p, q = map(proposition, ("p", "q"))
    condition = p
    for _ in range(2048):
        condition = all_of(any_of(condition, q), p)
    opposite = negate(condition)
    queries = IntegerQueryService({})
    assert queries.implies(opposite, (negate(p),)).proved
    assert queries.satisfies((condition, opposite)).impossible
    assert queries.implies(negate(opposite), (p,)).proved


def test_relation_is_encoded_once_and_origin_seeds_never_leak_to_other_scopes():
    p, q, r = map(proposition, ("p", "q", "r"))
    queries = IntegerQueryService({})
    system = queries.rule_system((any_of(negate(p), q),))
    with system.assuming(extra=(p,)) as context:
        assert context.prove(q).answer.proved
    with system.assuming() as context:
        failed = context.prove(q)
        assert failed.answer.refuted
        assert failed.condition is None
        assert not failed.support
    # The same seed with its real transfer disabled cannot reach q.
    with system.assuming(extra=(p,), enabled=()) as context:
        assert not context.prove(q).answer.proved
    system.extend((any_of(negate(q), r),))
    with system.assuming(extra=(p,)) as context:
        proof = context.prove(r)
        assert proof.answer.proved
        assert proof.support == (0, 1, 2)
        assert queries.owns_implication(proof.answer, proof.condition, ())
    assert len(system.rules) == 2
    assert queries.statistics.rule_encodings == 2
    assert queries.statistics.rule_assertions == 7
    # Failed scoped deductions store no global implication/model.
    assert all(answer.proved for answer in queries._answers.values())


def test_complete_premises_change_without_reencoding_or_strengthening_the_relation():
    p, q, active = map(proposition, ("p", "q", "active"))
    queries = IntegerQueryService({})
    system = queries.rule_system((any_of(negate(p), q), any_of(negate(active), p)))
    for premises, expected in (((active,), True), ((), False), ((negate(active),), False)):
        with system.assuming(premises) as context:
            assert context.prove(q).answer.proved == expected
    assert queries.statistics.rule_encodings == 2
    assert queries.statistics.rule_assertions == 6

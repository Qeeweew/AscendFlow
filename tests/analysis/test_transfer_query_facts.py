"""Pruning facts must agree with the independent mathematical service."""

from itertools import product

from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.protocol_transfers import _QueryFacts
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.memo import memo_scope
from ascendflow.scalar import Expr, expression, literal


def add(a, number):
    return expression("add", a, literal(number))


def test_difference_bounds_keep_sign_strictness_offsets_and_exact_query_scope():
    x, y = (Expr("argument", (name,)) for name in ("x", "y"))
    arithmetic = IntegerFacts({})
    contexts = (
        (Comparison(add(x, 2), ">=", add(y, 5)), Comparison(x, "<", add(y, 6))),
        (Comparison(add(y, 4), ">", add(x, 1)),),
        (Comparison(x, "==", add(y, 3)),),
        (Comparison(x, ">=", literal(0)), Comparison(x, "<", literal(9))),
        (Comparison(x, "==", literal(3)), Comparison(y, ">", literal(1))),
        (),
    )
    with memo_scope(fresh=True):
        for assumptions in contexts:
            facts = _QueryFacts(arithmetic, assumptions)
            for left, right, offset, predicate in product(
                (x, y), (x, y), range(-7, 8), ("<", "<=", ">", ">=", "==", "!=")
            ):
                condition = Comparison(add(left, offset), predicate, right)
                truth = facts.truth(condition)
                if truth is None:
                    continue
                opposite = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
                expected = (
                    condition
                    if truth
                    else Comparison(condition.lhs, opposite[predicate], condition.rhs)
                )
                assert arithmetic.queries.implies(expected, assumptions).proved
        bounded = _QueryFacts(arithmetic, contexts[0])
        assert bounded.truth(Comparison(x, ">=", add(y, 3))) is True
        assert bounded.truth(Comparison(x, "<", add(y, 3))) is False
        assert bounded.truth(Comparison(x, "==", add(y, 4))) is None
        weak = _QueryFacts(arithmetic, ())
        assert weak.truth(Comparison(x, "<", y)) is None
        assert weak.truth(Comparison(x, ">=", y)) is None


def test_same_nonlinear_term_bounds_do_not_imply_other_terms_or_origin_facts():
    x, y = (Expr("argument", (name,)) for name in ("x", "y"))
    value = expression("mul", x, y)
    arithmetic = IntegerFacts({})
    assumptions = (Comparison(add(value, 2), "<", literal(8)),)
    with memo_scope(fresh=True):
        facts = _QueryFacts(arithmetic, assumptions)
        condition = Comparison(value, "<=", literal(5))
        assert facts.truth(condition) is True
        assert arithmetic.queries.implies(condition, assumptions).proved
        assert facts.truth(Comparison(x, "<=", literal(5))) is None
        assert facts.truth(Comparison(y, "<=", literal(5))) is None

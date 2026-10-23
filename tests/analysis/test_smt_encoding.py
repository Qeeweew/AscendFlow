"""The closed math encoder preserves exact Z3 formulas and their contexts."""

import pytest
from z3 import And, Bool, BoolVal, Context, IntVal, Not, Or

from ascendflow.analysis.smt import (
    Boolean,
    IntegerQueryService,
    _and,
    _boolean,
    _or,
    all_of,
    any_of,
)
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.memo import memo_scope, memo_table
from ascendflow.scalar import Expr, literal


@pytest.mark.parametrize("constructor,reference", [(_and, And), (_or, Or)])
@pytest.mark.parametrize("arity", [0, 1, 4])
def test_boolean_packets_keep_empty_singleton_order_duplicates_and_nested_asts(
    constructor, reference, arity
):
    p, q = Bool("encoding_p"), Bool("encoding_q")
    values = (p, reference(q, Not(p)), p, BoolVal(False))[:arity]
    assert constructor(*values).eq(reference(*values))


def test_nested_math_encoding_keeps_the_entire_context_formula():
    service = IntegerQueryService({})
    x, y = (Expr("argument", (name,)) for name in ("x", "y"))
    left = Boolean("or", (Comparison(x, "<", literal(0)), Comparison(y, "==", literal(1))))
    right = Boolean("and", (True, Comparison(x, ">=", y), left))
    query = Boolean("and", (left, right, False))
    a, b = service._integer(x), service._integer(y)
    expected_left = Or(a < IntVal(0), b == IntVal(1))
    expected_right = And(BoolVal(True), a >= b, expected_left)
    assert service._comparison(query).eq(And(expected_left, expected_right, BoolVal(False)))


@pytest.mark.parametrize("constructor", [_and, _or])
def test_boolean_packet_refuses_untyped_or_mixed_context_operands(constructor):
    p = Bool("encoding_p")
    foreign = Bool("encoding_p", ctx=Context())
    with pytest.raises(CompileError, match="BoolRef"):
        constructor(True)
    with pytest.raises(CompileError, match="one context"):
        constructor(p, foreign)


def test_shared_boolean_diamonds_encode_once_per_dag_node():
    x = Expr("argument", ("diamond",))
    p, q = Comparison(x, "<", literal(0)), Comparison(x, ">", literal(0))

    def diamond():
        node = any_of(p, q)
        for _ in range(32):
            node = all_of(any_of(node, p), any_of(node, q))
        return node

    with memo_scope(fresh=True):
        left, right = diamond(), diamond()
        assert left is right
        queries = IntegerQueryService({})
        queries._comparison(left)
        assert queries._comparison.cache_info().currsize <= 4 * 32 + 4
        assert queries.implies(left, (p,)).proved
        assert queries.implies(left, (q,)).proved
        assert queries.satisfies((left, Comparison(x, "==", literal(0)))).impossible


def test_boolean_syntax_sharing_keeps_scope_lifetime_and_operand_order():
    x = Expr("argument", ("scope",))
    p, q = Comparison(x, "<", literal(0)), Comparison(x, ">", literal(0))
    with memo_scope(fresh=True):
        first = all_of(p, q)
        assert first is all_of(p, q)
        assert first is not all_of(q, p)
        table = memo_table(_boolean)
    assert not table
    with memo_scope(fresh=True):
        assert all_of(p, q) is not first


def test_boolean_interning_cannot_admit_integer_zero_as_a_cached_false_operand():
    p = Comparison(Expr("argument", ("typed",)), "==", literal(1))
    with memo_scope(fresh=True):
        assert _boolean("and", (False, p)).operands == (False, p)
        with pytest.raises(CompileError, match="mathematical predicates"):
            _boolean("and", (0, p))

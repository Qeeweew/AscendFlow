"""The closed math encoder preserves exact Z3 formulas and their contexts."""

import pytest
from z3 import And, Bool, BoolVal, Context, IntVal, Not, Or

from ascendflow.analysis.smt import Boolean, IntegerQueryService, _and, _or
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
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
    left = Boolean(
        "or", (Comparison(x, "<", literal(0)), Comparison(y, "==", literal(1)))
    )
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

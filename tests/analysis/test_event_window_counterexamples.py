"""Pure-expression event-window counterexamples, without source compilation.

Expectations come from the original integer predicates and fixed-width scalar
semantics. The production predicate queries only establish the builder's
premises; no allocator, native body, lifetime or progress result is consulted.
Explicit unsupported arithmetic is allowed instead of an unsafe certificate.
"""

import pytest
from xdsl.dialects.builtin import ModuleOp

from xdsl_npu.analysis.event_windows import build_event_window
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.structured_protocols import LogicalNotification, NotificationKind
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import INT64_MAX, Expr, Fact, expression, literal
from xdsl_npu.target.ascend910b import Engine

G = Expr("argument", ("g",))
PARENT = Expr("argument", ("parent",))
MASK64 = (1 << 64) - 1


def _contains_ordinal(value):
    return value == G or any(
        _contains_ordinal(child) for child in value.args if isinstance(child, Expr)
    )


def _window(count, extra=(), *, facts=None):
    # Operation identity is the only loop property consumed by this stage.
    # No DSL lowering or full operator construction is involved.
    loop = ModuleOp([])
    notification = LogicalNotification(
        "arithmetic_counterexample",
        (),
        (),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        loop,
        G,
        1,
        0,
    )
    guards = (
        Comparison(G, ">=", literal(0)),
        Comparison(G, "<", literal(count)),
        *extra,
    )
    queries = IntegerFacts({} if facts is None else facts)
    return build_event_window(
        notification,
        loop,
        ordinal=G,
        count=literal(count),
        guards=guards,
        facts=queries.facts,
        proves=queries.prove,
        bounds=queries.bounds,
        invariant=lambda value: not _contains_ordinal(value),
    )


def _signed64(value):
    value &= MASK64
    return value - (1 << 64) if value > INT64_MAX else value


def _evaluate(value, context, *, checked):
    if value.kind == "constant":
        result = value.constant
    elif value.kind == "argument":
        result = context[value.args[0]]
    else:
        a, b = (_evaluate(child, context, checked=checked) for child in value.args)
        if value.kind == "add":
            result = a + b
        elif value.kind == "mul":
            result = a * b
        elif value.kind == "min":
            result = min(a, b)
        elif value.kind == "max":
            result = max(a, b)
        else:
            if checked:
                assert a >= 0 and b > 0, "unsigned scalar operation lacks its arithmetic domain"
            a, b = a & MASK64, b & MASK64
            if value.kind == "rem":
                result = a % b
            elif value.kind == "floordiv":
                result = a // b
            else:
                assert value.kind == "ceildiv"
                result = a // b + int(a % b != 0)
    if checked:
        assert -(1 << 63) <= result <= INT64_MAX, "guard evaluated overflowing signed arithmetic"
    return _signed64(result)


def _guard(guard, context, *, checked):
    # Preserve the ordered short-circuit contract of ActionGuard.
    for test in guard.comparisons:
        a = _evaluate(test.lhs, context, checked=checked)
        b = _evaluate(test.rhs, context, checked=checked)
        if not Comparison(literal(a), test.predicate, literal(b)).evaluate({}):
            return False
    return True


def test_remainder_range_tautology_does_not_prove_its_own_numerator_domain():
    remainder = expression("rem", expression("sub", G, literal(2)), literal(3))
    with pytest.raises(CompileError) as failure:
        _window(3, (Comparison(remainder, ">=", literal(0)),))
    assert failure.value.kind is ErrorKind.UNSUPPORTED


def test_a_later_guard_cannot_authorize_an_earlier_unsigned_remainder():
    remainder = expression("rem", expression("sub", G, literal(2)), literal(3))
    with pytest.raises(CompileError) as failure:
        _window(
            3,
            (
                Comparison(remainder, "==", literal(0)),
                Comparison(G, ">=", literal(2)),
            ),
        )
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.witness[1] == f"required: {remainder.args[0].text()} >= 0"
    assert failure.value.witness[2] == "ordered prefix: g >= 0; g < 3"


def test_dynamic_crt_cannot_certify_a_nonmember_after_i64_intermediate_wrap():
    period = 8_000_000_001
    extra = (
        Comparison(expression("rem", G, literal(2)), "==", literal(0)),
        Comparison(
            expression("rem", G, literal(period)),
            "==",
            expression("rem", PARENT, literal(period)),
        ),
    )
    try:
        window = _window(2 * period + 1, extra, facts={"parent": Fact(maximum=period - 1)})
    except CompileError as error:
        assert error.kind is ErrorKind.UNSUPPORTED
        return

    context = {"parent": period - 2}
    expected_first = 2 * period - 2
    assert expected_first < INT64_MAX
    assert all(test.evaluate(context | {"g": expected_first}) for test in extra)
    assert window.nonempty.evaluate(context)
    first, _ = window.first_last()
    assert first.evaluate(context) == expected_first
    actual_first = _evaluate(first, context, checked=False)
    assert actual_first == expected_first
    assert all(test.evaluate(context | {"g": actual_first}) for test in extra)


def test_empty_alignment_cannot_wrap_its_first_into_a_nonempty_certificate():
    period = INT64_MAX - 1
    extra = (
        Comparison(G, ">=", literal(INT64_MAX - 1)),
        Comparison(expression("rem", G, literal(period)), "==", literal(period - 1)),
    )
    try:
        window = _window(INT64_MAX, extra)
    except CompileError as error:
        assert error.kind is ErrorKind.UNSUPPORTED
        return

    # There is only one in-range ordinal, and it has residue zero.
    assert not window.membership.evaluate({"g": INT64_MAX - 1})
    assert not window.nonempty.evaluate({})
    assert not _guard(window.nonempty, {}, checked=False)
    assert not _guard(window.nonempty, {}, checked=True)


@pytest.mark.parametrize("width", [2, 3])
def test_same_id_guard_short_circuits_before_a_nonexistent_successor_overflows(width):
    try:
        window = _window(INT64_MAX)
        successor_guard = window.same_id_guard(width)
    except CompileError as error:
        assert error.kind is ErrorKind.UNSUPPORTED
        return

    assert not _guard(successor_guard, {"g": INT64_MAX - 1}, checked=True)
    context = {"g": INT64_MAX - width - 1}
    assert _guard(successor_guard, context, checked=True)
    assert _evaluate(window.next_same_id(width), context, checked=True) == INT64_MAX - 1

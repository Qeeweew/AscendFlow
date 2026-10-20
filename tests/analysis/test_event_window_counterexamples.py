"""Event-window arithmetic counterexamples, without operator compilation.

Expectations come from the original integer predicates and fixed-width scalar
semantics. The production predicate queries only establish the builder's
premises; no allocator, native body, lifetime or progress result is consulted.
Explicit unsupported arithmetic is allowed instead of an unsafe certificate.
"""

import pytest
from xdsl.dialects.builtin import ModuleOp

from ascendflow import language as N
from ascendflow.analysis.event_windows import build_event_window
from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_control import build_control_facts
from ascendflow.analysis.structured_protocols import LogicalNotification, NotificationKind
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import INT64_MAX, Expr, Fact, expression, literal
from ascendflow.target.ascend910b import Engine

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
        ordered_definedness=queries.ordered_definedness,
        invariant=lambda value: not _contains_ordinal(value),
    )


def _signed64(value):
    value &= MASK64
    return value - (1 << 64) if value > INT64_MAX else value


def _evaluate(value, context, *, checked):
    if value.kind == "constant":
        result = value.constant
    elif value.kind == "argument":
        symbol = value.args[0]
        result = context[symbol] if symbol in context else context[str(symbol)]
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
    assert 0 <= window.first.constant <= INT64_MAX
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


def test_dense_multi_residue_first_last_and_reuse_remain_defined_near_i64_limit():
    # Reducing before multiplication avoids overflowing the source predicate.
    remainder = expression(
        "rem", expression("mul", expression("rem", G, literal(5)), literal(2)), literal(5)
    )
    lower = INT64_MAX - 20
    window = _window(
        INT64_MAX,
        (
            Comparison(G, ">=", literal(lower)),
            Comparison(remainder, "<=", literal(1)),
        ),
    )
    members = [g for g in range(lower, INT64_MAX) if g % 5 in (0, 3)]
    assert _guard(window.nonempty, {}, checked=True)
    first, last = window.first_last()
    assert _evaluate(first, {}, checked=True) == members[0]
    assert _evaluate(last, {}, checked=True) == members[-1]
    for width in (1, 2, 3):
        for index, g in enumerate(members):
            context = {"g": g}
            exists = index + width < len(members)
            assert _guard(window.same_id_guard(width), context, checked=True) == exists
            phases = [
                delta
                for guard, delta in window.reuse_phases(width)
                if _guard(guard, context, checked=True)
            ]
            assert len(phases) == int(exists)
            if exists:
                assert (
                    _evaluate(window.next_same_id(width), context, checked=True)
                    == members[index + width]
                )
                assert g + phases[0] == members[index + width]


@pytest.fixture(scope="module")
def selected_last_identity():
    source = N.from_source("""
    def clocks(extent:N.int64(multiple_of=8)):
        with N.Kernel():
            for p in N.Pipelined(extent//8):
                pass
""")
    control = build_control_facts(lower_native_structure(source).module)
    (loop,) = control.loops.values()
    queries = IntegerFacts(control.environment.facts)
    extent = next(name for name in control.environment.facts if name.name == "extent")
    tiles = expression("floordiv", Expr("argument", (extent,)), literal(8))
    previous = expression("sub", loop.ordinal, literal(1))
    restored = expression("sub", expression("add", previous, literal(1)), literal(1))
    end = expression("sub", loop.count, literal(1))

    def last(index):
        phase = expression(
            "rem",
            expression(
                "sub",
                literal(2),
                expression("rem", expression("mul", index, tiles), literal(2)),
            ),
            literal(2),
        )
        return expression("sub", end, expression("rem", expression("sub", end, phase), literal(2)))

    prefix = (
        Comparison(loop.count, ">", literal(1)),
        Comparison(loop.ordinal, ">=", literal(1)),
        Comparison(loop.ordinal, "<", loop.count),
    )
    return control, loop, queries, prefix, Comparison(last(previous), "==", last(restored))


def _selected_last_window(control, loop, queries, guards):
    notification = LogicalNotification(
        "selected_last_identity",
        (),
        (),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        loop.operation,
        loop.ordinal,
        1,
        0,
    )
    return build_event_window(
        notification,
        loop.operation,
        ordinal=loop.ordinal,
        count=loop.count,
        guards=guards,
        facts=queries.facts,
        proves=queries.prove,
        bounds=queries.bounds,
        ordered_definedness=queries.ordered_definedness,
        invariant=lambda value: control.invariant_over(value, loop.operation),
    )


def test_equivalent_selected_last_coordinates_need_no_parameter_partition(selected_last_identity):
    control, loop, queries, prefix, equality = selected_last_identity
    assert equality.lhs != equality.rhs
    guards = prefix + (equality,)
    assert queries.ordered_definedness(guards) is None
    # A failed identity proof would request a uniform parameter split. The
    # runtime quotient and full raw endpoint expressions remain in this input.
    window = _selected_last_window(control, loop, queries, guards)
    assert window.period == 1
    for extent in (0, 8, 16, 24, 32, 56, 80):
        context = {"extent": extent}
        count = extent // 8
        members = list(range(1, count))
        observed = []
        for p in range(-1, count + 2):
            actual = context | {loop.ordinal.args[0]: p}
            active = _guard(window.membership, actual, checked=True)
            assert window.interval.evaluate(actual) == active
            if active:
                observed.append(p)
        assert observed == members
        assert _guard(window.nonempty, context, checked=True) == bool(members)
        if not members:
            continue
        first, last = window.first_last()
        assert _evaluate(first, context, checked=True) == members[0]
        assert _evaluate(last, context, checked=True) == members[-1]
        for p in members:
            actual = context | {loop.ordinal.args[0]: p}
            phase = (2 - ((p - 1) * count) % 2) % 2
            selected = count - 1 - (count - 1 - phase) % 2
            assert _evaluate(equality.lhs, actual, checked=True) == selected
            assert _evaluate(equality.rhs, actual, checked=True) == selected


def test_selected_last_identity_keeps_its_original_unsigned_prefix(selected_last_identity):
    control, loop, queries, prefix, equality = selected_last_identity
    guards = (prefix[0], prefix[2], equality)
    assert queries.ordered_definedness(guards) is not None
    with pytest.raises(
        CompileError, match="unsigned arithmetic lacks an ordered activation prefix"
    ) as failure:
        _selected_last_window(control, loop, queries, guards)
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.phase == "event-window-membership"

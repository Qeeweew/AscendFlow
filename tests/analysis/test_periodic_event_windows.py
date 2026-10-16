"""Small independent integer oracles for concrete periodic event fibers."""

from collections.abc import Callable

import pytest
from xdsl.dialects.builtin import ModuleOp

from xdsl_npu.analysis import event_windows
from xdsl_npu.analysis.event_windows import WindowPartitionRequired, build_event_window
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.periodic import evaluate_comparison_at, evaluate_phase, period_of
from xdsl_npu.analysis.structured_protocols import LogicalNotification, NotificationKind
from xdsl_npu.analysis.structured_regions import LoopDomain
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.indexing import arguments
from xdsl_npu.scalar import INT64_MAX, Expr, Fact, expression, literal
from xdsl_npu.target.ascend910b import Engine

G = Expr("argument", ("g",))
N = Expr("argument", ("n",))
M = Expr("argument", ("m",))


def _invariant(value):
    return "g" not in arguments(value)


def _window(extra=(), *, count=N, facts=None, bindings=None, domains=lambda _: ()):
    loop = ModuleOp([])
    notification = LogicalNotification(
        "periodic_oracle",
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
        domains(loop),
    )
    queries = IntegerFacts({"n": Fact(maximum=INT64_MAX), **({} if facts is None else facts)})
    return build_event_window(
        notification,
        loop,
        ordinal=G,
        count=count,
        guards=(Comparison(G, ">=", literal(0)), Comparison(G, "<", count), *extra),
        facts=queries.facts,
        proves=queries.prove,
        bounds=queries.bounds,
        ordered_definedness=queries.ordered_definedness,
        invariant=_invariant,
        parameter_bindings=bindings,
    )


def _compare(a, predicate, b):
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[
        predicate
    ]


def _oracle(window, count, active: Callable[[int], bool], *, context=None, periodic=None):
    context = {"n": count, **({} if context is None else context)}
    members = [g for g in range(count) if active(g)]
    assert window.project_exists().guard.evaluate(context) == bool(members)
    if window.upper is not None:
        assert len(members) <= window.upper
    widths = (1, 2, 3, 7)
    same_id = {width: window.same_id_guard(width) for width in widths}
    phases_by_width = {width: window.reuse_phases(width) for width in widths}
    successors = {width: window.next_same_id(width) for width in widths} if members else {}
    for g in range(-1, count + 2):
        values = context | {"g": g}
        expected = g in members
        assert window.membership.evaluate(values) == expected
        assert window.interval.evaluate(values) == expected
        for width in widths:
            successor = expected and members.index(g) + width < len(members)
            assert same_id[width].evaluate(values) == successor
            phases = [delta for guard, delta in phases_by_width[width] if guard.evaluate(values)]
            assert len(phases) == int(successor)
            if successor:
                target = members[members.index(g) + width]
                assert successors[width].evaluate(values) == target
                assert g + phases[0] == target
    if members:
        first, last = window.first_last()
        assert (first.evaluate(context), last.evaluate(context)) == (members[0], members[-1])
        assert [window.rank().evaluate(context | {"g": g}) for g in members] == list(
            range(len(members))
        )
    if periodic is not None:
        for n in range(count + 2):
            assert window.prefix_count(literal(n)).evaluate(context) == sum(
                periodic(g) for g in range(n)
            )


def test_two_residue_dense_rank_uses_alternating_reuse_shifts():
    remainder = expression("rem", expression("mul", literal(2), G), literal(5))
    window = _window((Comparison(remainder, "<=", literal(1)),))
    assert window.period == 5
    assert tuple(r.constant for r in window.residues) == (0, 3)
    assert [delta for _, delta in window.reuse_phases(1)] == [3, 2]
    for count in (0, 1, 2, 4, 5, 8, 19):
        _oracle(window, count, lambda g: g % 5 in (0, 3), periodic=lambda g: g % 5 in (0, 3))


@pytest.mark.parametrize("predicate", ["==", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("reverse", [False, True])
def test_affine_modular_comparisons_have_exact_static_residues(predicate, reverse):
    numerator = expression("sub", expression("mul", literal(4), G), literal(7))
    remainder = expression("rem", numerator, literal(6))
    test = (
        Comparison(literal(2), predicate, remainder)
        if reverse
        else Comparison(remainder, predicate, literal(2))
    )
    window = _window((Comparison(G, ">=", literal(2)), test))
    periodic = lambda g: (
        _compare(2, predicate, (4 * g - 7) % 6)
        if reverse
        else _compare((4 * g - 7) % 6, predicate, 2)
    )
    for count in (0, 2, 3, 8, 17):
        _oracle(window, count, lambda g: g >= 2 and periodic(g), periodic=periodic)


def test_clipped_multi_residue_window_and_empty_full_sets():
    rem = expression("rem", G, literal(5))
    for predicate, threshold in (("<", 0), (">=", 0), ("!=", 2)):
        window = _window(
            (
                Comparison(G, ">=", literal(3)),
                Comparison(G, "<", literal(14)),
                Comparison(rem, predicate, literal(threshold)),
            )
        )
        for count in (0, 2, 5, 13, 20):
            periodic = lambda g, predicate=predicate, threshold=threshold: _compare(
                g % 5, predicate, threshold
            )
            _oracle(
                window,
                count,
                lambda g, periodic=periodic: 3 <= g < 14 and periodic(g),
                periodic=periodic,
            )
        if threshold == 0 and predicate == "<":
            assert window.residues == ()
        if threshold == 0 and predicate == ">=":
            assert window.period == 1 and window.residues == (literal(0),)


def test_real_multi_residue_intersection_and_non_coprime_empty_crt():
    window = _window(
        (
            Comparison(expression("rem", G, literal(4)), "!=", literal(3)),
            Comparison(expression("rem", G, literal(6)), "<", literal(2)),
        )
    )
    assert window.period == 12
    assert tuple(r.constant for r in window.residues) == (0, 1, 6)
    for count in (0, 1, 8, 27):
        active = lambda g: g % 4 != 3 and g % 6 < 2
        _oracle(window, count, active, periodic=active)
    empty = _window(
        (
            Comparison(expression("rem", G, literal(4)), "==", literal(0)),
            Comparison(expression("rem", G, literal(6)), "==", literal(1)),
        )
    )
    assert empty.residues == ()
    _oracle(empty, 30, lambda _: False, periodic=lambda _: False)


def test_uniform_bindings_specialize_original_count_guards_and_domain_hints():
    residue = expression("rem", M, literal(5))
    window = _window(
        (Comparison(expression("rem", G, literal(5)), "==", residue),),
        count=M,
        facts={"m": Fact(maximum=20)},
        bindings={M: literal(8)},
        domains=lambda loop: (LoopDomain(loop, M, 5, residue),),
    )
    assert window.count.constant == 8
    assert window.residues == (literal(3),)
    _oracle(window, 8, lambda g: g % 5 == 3, context={"m": 0}, periodic=lambda g: g % 5 == 3)
    with pytest.raises(CompileError, match="uniform input"):
        _window(facts={"g": Fact(maximum=20)}, bindings={G: literal(1)})


def test_false_specialized_prefix_does_not_construct_nonexistent_unsigned_selection():
    selection = expression("rem", expression("sub", M, literal(1)), literal(2))
    window = _window(
        (Comparison(M, ">", literal(0)), Comparison(selection, "==", literal(0))),
        facts={"m": Fact(maximum=20)},
        bindings={M: literal(0)},
        domains=lambda loop: (LoopDomain(loop, N, 2, selection),),
    )
    assert window.residues == ()
    assert len(window.membership.comparisons) == 3
    for count in (0, 1, 7):
        _oracle(window, count, lambda _: False, context={"m": 0})


def test_group_invocation_parameter_binding_specializes_the_same_concrete_fiber():
    group = Expr("argument", ("$group",))
    residue = expression("rem", group, literal(3))
    window = _window(
        (Comparison(expression("rem", G, literal(3)), "==", residue),),
        facts={"$group": Fact(maximum=19)},
        bindings={group: literal(2)},
        domains=lambda loop: (LoopDomain(loop, N, 3, residue),),
    )
    assert window.residues == (literal(2),)
    _oracle(window, 8, lambda g: g % 3 == 2, context={"$group": 0})
    sid = Expr("argument", ("$sid",))
    with pytest.raises(CompileError, match="uniform input"):
        _window(facts={"$sid": Fact(maximum=1)}, bindings={sid: literal(0)})


def test_negative_static_affine_coefficient_has_exact_residues():
    numerator = expression("sub", literal(17), expression("mul", literal(2), G))
    window = _window(
        (Comparison(expression("rem", numerator, literal(6)), "!=", literal(3)),), count=literal(9)
    )
    assert window.period == 3
    assert window.residues == (literal(0), literal(2))
    _oracle(window, 9, lambda g: (17 - 2 * g) % 6 != 3)


def test_unknown_dynamic_multi_residue_predicate_is_an_explicit_partition_request():
    test = Comparison(expression("rem", G, literal(5)), "<", M)
    with pytest.raises(WindowPartitionRequired) as failure:
        _window((test,), facts={"m": Fact(maximum=4)})
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.tests == (test,)
    with pytest.raises(CompileError, match="conditional multi-residue"):
        _window(
            (
                Comparison(expression("rem", G, literal(5)), "<", literal(2)),
                Comparison(
                    expression("rem", G, literal(3)), "==", expression("rem", M, literal(3))
                ),
            ),
            facts={"m": Fact(maximum=10)},
        )


def test_encoded_ordered_prefix_fact_remains_available_to_pending_proof():
    residue = expression("rem", expression("mul", expression("sub", G, literal(1)), M), literal(2))
    window = _window(
        (Comparison(G, ">=", literal(1)), Comparison(M, ">", residue)),
        facts={"m": Fact(minimum=2, maximum=INT64_MAX)},
    )
    assert window.period == 1
    for count in (0, 1, 2, 7):
        for parameter in (2, 3, 100):
            _oracle(window, count, lambda g: g >= 1, context={"m": parameter})


def test_already_encoded_equivalent_bounds_do_not_repeat_interval_clipping():
    lower = Comparison(G, ">=", literal(2))
    base = _window((lower,))
    refined = _window(
        (
            lower,
            Comparison(G, ">", literal(1)),
            Comparison(expression("add", G, literal(3)), ">=", literal(5)),
            Comparison(expression("mul", G, literal(2)), ">=", literal(4)),
        )
    )
    assert refined.lower == base.lower
    assert refined.count == base.count
    for count in (0, 1, 2, 3, 9):
        _oracle(refined, count, lambda g: g >= 2)


def test_period_and_intersection_budgets_reject_before_large_work(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            event_windows, "range", lambda *_: pytest.fail("enumerated before limit"), raising=False
        )
        with pytest.raises(CompileError) as failure:
            _window(
                (
                    Comparison(
                        expression("rem", expression("mul", literal(2), G), literal(130)),
                        "<",
                        literal(1),
                    ),
                )
            )
        assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
        with pytest.raises(CompileError) as failure:
            _window(
                (
                    Comparison(
                        expression("rem", G, literal(8)), "<", expression("rem", G, literal(9))
                    ),
                )
            )
        assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    with pytest.raises(CompileError, match="residue intersection") as failure:
        _window(
            (
                Comparison(expression("rem", G, literal(11)), "!=", literal(10)),
                Comparison(expression("rem", G, literal(13)), "!=", literal(12)),
            )
        )
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    with pytest.raises(CompileError, match="residue work") as failure:
        _window(
            (
                Comparison(expression("rem", G, literal(64)), "!=", literal(63)),
                Comparison(expression("rem", G, literal(64)), "<", literal(63)),
            )
        )
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT


def test_nested_modulo_uses_mathematical_phase_outside_valid_source_prefix():
    inner = expression("rem", expression("mul", expression("sub", G, literal(1)), M), literal(2))
    residue = expression("rem", expression("sub", literal(2), inner), literal(2))
    first = expression("rem", expression("add", literal(2), residue), literal(2))
    window = _window(
        (Comparison(G, ">=", literal(1)), Comparison(first, "==", literal(0))),
        facts={"m": Fact(maximum=10)},
        bindings={M: literal(1)},
    )
    assert window.period == 2 and window.residues == (literal(1),)
    for count in (0, 1, 2, 6, 13):
        _oracle(
            window,
            count,
            lambda g: g >= 1 and g % 2 == 1,
            context={"m": 1},
            periodic=lambda g: g % 2 == 1,
        )


def test_periodic_arithmetic_closure_uses_actual_lcm():
    a = expression("add", expression("rem", G, literal(3)), literal(1))
    b = expression("add", expression("rem", G, literal(5)), literal(1))
    for kind in ("add", "mul", "min", "max", "floordiv", "ceildiv", "rem"):
        value = expression(kind, a, b)
        assert period_of(value, G, _invariant) == 15
        for phase in range(15):
            assert evaluate_phase(value, G, phase) == value.evaluate({"g": phase})
            assert evaluate_phase(value, G, phase) == value.evaluate({"g": phase + 15})
    test = Comparison(expression("floordiv", b, a), "<", literal(2))
    window = _window((test,))
    for count in (0, 3, 17):
        active = lambda g: (g % 5 + 1) // (g % 3 + 1) < 2
        _oracle(window, count, active, periodic=active)
    assert period_of(expression("rem", expression("mul", G, G), literal(7)), G, _invariant) == 7
    assert (
        period_of(expression("rem", expression("min", G, literal(5)), literal(3)), G, _invariant)
        is None
    )
    assert period_of(expression("floordiv", G, literal(2)), G, _invariant) is None
    assert evaluate_comparison_at(Comparison(M, "==", literal(0)), G, 0) is None
    negative = expression("rem", expression("sub", G, literal(1)), literal(2))
    assert evaluate_phase(negative, G, 0) == 1


def test_large_residue_set_preserves_the_ordinary_expression_budget():
    window = _window((Comparison(expression("rem", G, literal(64)), "!=", literal(63)),))
    assert len(window.residues) == 63
    assert window.rank().weight <= 4096
    assert window.next_same_id(7).weight <= 4096
    members = [g for g in range(70) if g % 64 != 63]
    for g in (0, 1, 55, 60, 62, 64, 69):
        context = {"n": 70, "g": g}
        index = members.index(g)
        assert window.rank().evaluate(context) == index
        for width in (1, 7):
            exists = index + width < len(members)
            assert window.same_id_guard(width).evaluate(context) == exists
            if exists:
                assert window.next_same_id(width).evaluate(context) == members[index + width]


def test_periodic_division_only_evaluates_phases_reachable_after_its_prefix():
    divisor = expression("rem", G, literal(2))
    quotient = expression("floordiv", expression("rem", G, literal(3)), divisor)
    window = _window((Comparison(divisor, ">", literal(0)), Comparison(quotient, "==", literal(1))))
    assert window.period == 6
    for count in (0, 1, 2, 7, 14):
        active = lambda g: g % 2 == 1 and g % 3 == 1
        _oracle(window, count, active, periodic=active)

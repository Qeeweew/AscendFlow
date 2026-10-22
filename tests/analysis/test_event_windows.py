"""Finite membership oracles on actual retained clocks, without allocation."""

import pytest

from ascendflow import language as N
from ascendflow.analysis.event_windows import build_event_window
from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_control import build_control_facts
from ascendflow.analysis.structured_protocols import LogicalNotification, NotificationKind
from ascendflow.analysis.structured_regions import LoopDomain
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr, expression, literal
from ascendflow.target.ascend910b import Engine


@pytest.fixture(scope="module")
def control():
    source = N.from_source("""
    def clocks(n:N.i64,m:N.i64):
        with N.Kernel():
            for p in N.Pipelined(n):
                for g in N.Pipelined(m):
                    pass
""")
    return build_control_facts(lower_native_structure(source).module)


def clock_guard(control, logical):
    tests = []
    for frame in control.frames[logical.operation].ancestors():
        if frame.loop is not None:
            tests.extend(
                (
                    Comparison(frame.loop.ordinal, ">=", literal(0)),
                    Comparison(frame.loop.ordinal, "<", frame.loop.count),
                )
            )
    return tuple(tests)


def window(control, logical, extra=(), domains=()):
    # Only reuse the already defined predicate arithmetic. Constructing a
    # complete EventDomains here would call the window stage recursively.
    predicates = IntegerFacts(control.environment.facts)
    notification = LogicalNotification(
        "test_window",
        (),
        (),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        logical.operation,
        logical.ordinal,
        1,
        0,
        domains,
    )
    return build_event_window(
        notification,
        logical.operation,
        ordinal=logical.ordinal,
        count=logical.count,
        guards=clock_guard(control, logical) + tuple(extra),
        facts=control.environment.facts,
        proves=predicates.prove,
        bounds=predicates.bounds,
        ordered_definedness=predicates.ordered_definedness,
        invariant=lambda value: control.invariant_over(value, logical.operation),
    )


def check_exact_membership(contract, context, count):
    symbol = contract.ordinal.args[0]
    members = []
    for generation in range(-1, count + 2):
        actual = context | {symbol: generation}
        active = contract.membership.evaluate(actual)
        assert contract.interval.evaluate(actual) == active
        if active:
            members.append(generation)
    assert contract.project_exists().membership is contract.membership
    assert contract.project_exists().guard.evaluate(context) == bool(members)
    if contract.upper is not None:
        assert len(members) <= contract.upper
    if members:
        first, last = contract.first_last()
        assert first.evaluate(context) == members[0]
        assert last.evaluate(context) == members[-1]
        assert [contract.rank().evaluate(context | {symbol: g}) for g in members] == list(
            range(len(members))
        )
        for width in (1, 2, 3):
            for index, generation in enumerate(members):
                actual = context | {symbol: generation}
                exists = index + width < len(members)
                assert contract.same_id_guard(width).evaluate(actual) == exists
                if exists:
                    assert contract.next_same_id(width).evaluate(actual) == members[index + width]


def test_runtime_true_count_empty_short_and_nonzero_first(control):
    logical = next(iter(control.loops.values()))
    g = logical.ordinal
    contract = window(
        control,
        logical,
        (Comparison(g, ">=", literal(2)),),
        (LoopDomain(logical.operation, logical.count, first=2),),
    )
    for count in (0, 1, 2, 3, 4, 9):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


def test_clamped_owner_count_keeps_empty_and_nonempty_invocations():
    source = N.from_source("""
    def owners(n:N.i64(minimum=8, multiple_of=8)):
        with N.Kernel(20) as group:
            for query in N.serial(N.ceildiv(N.max(n // 8 - group, 0), 20)):
                pass
    """)
    control = build_control_facts(lower_native_structure(source).module)
    logical = next(iter(control.loops.values()))
    contract = window(control, logical)
    for extent in (8, 16, 160, 168, 320):
        for group in (0, 1, 19):
            remaining = max(extent // 8 - group, 0)
            count = (remaining + 19) // 20
            check_exact_membership(contract, {"n": extent, "$group": group}, count)


def test_first_last_exclusions_preserve_exact_interval(control):
    logical = next(iter(control.loops.values()))
    g, count = logical.ordinal, logical.count
    contract = window(
        control,
        logical,
        (
            Comparison(g, "!=", literal(0)),
            Comparison(g, "!=", expression("sub", count, literal(1))),
        ),
    )
    for count in (0, 1, 2, 3, 8):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


def test_selected_last_is_single_and_guarded_for_empty_count(control):
    logical = next(iter(control.loops.values()))
    contract = window(
        control,
        logical,
        (Comparison(logical.ordinal, "==", expression("sub", logical.count, literal(1))),),
    )
    assert contract.single and contract.upper <= 1
    for count in (0, 1, 2, 7):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


def test_parent_dependent_residue_remains_an_actual_loop_invariant(control):
    parent, child = control.loops.values()
    residue = expression("rem", parent.ordinal, literal(3))
    contract = window(
        control,
        child,
        (
            Comparison(child.ordinal, ">=", literal(2)),
            Comparison(expression("rem", child.ordinal, literal(3)), "==", residue),
        ),
        (LoopDomain(child.operation, child.count, 3, residue, first=2),),
    )
    for p in range(4):
        for count in (0, 1, 2, 3, 8, 10):
            check_exact_membership(contract, {"n": 4, "m": count, parent.ordinal.args[0]: p}, count)


def test_parent_dependent_lower_and_residue_keep_the_same_member_order(control):
    parent, child = control.loops.values()
    contract = window(
        control,
        child,
        (
            Comparison(child.ordinal, ">", parent.ordinal),
            Comparison(
                expression("rem", child.ordinal, literal(3)),
                "==",
                expression("rem", parent.ordinal, literal(3)),
            ),
        ),
    )
    for p in range(5):
        for count in (0, 1, 3, 5, 12):
            check_exact_membership(contract, {"n": 5, "m": count, parent.ordinal.args[0]: p}, count)


@pytest.mark.parametrize("residues", [(1, 3), (0, 1)])
def test_multiple_residue_constraints_have_exact_intersection(control, residues):
    logical = next(iter(control.loops.values()))
    guards = tuple(
        Comparison(expression("rem", logical.ordinal, literal(period)), "==", literal(residue))
        for period, residue in zip((4, 6), residues, strict=True)
    )
    contract = window(control, logical, guards)
    assert contract.period == 12
    for count in (0, 1, 10, 25, 41):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


def test_dynamic_residue_intersection_preserves_parent_correlation(control):
    parent, child = control.loops.values()
    contract = window(
        control,
        child,
        tuple(
            Comparison(
                expression("rem", child.ordinal, literal(period)),
                "==",
                expression("rem", parent.ordinal, literal(period)),
            )
            for period in (4, 6)
        ),
    )
    for p in range(5):
        for count in (0, 1, 4, 13, 26):
            check_exact_membership(contract, {"n": 5, "m": count, parent.ordinal.args[0]: p}, count)


def test_shifted_residue_and_binary_complement_are_exact(control):
    logical = next(iter(control.loops.values()))
    g = logical.ordinal
    contract = window(
        control,
        logical,
        (
            Comparison(g, ">=", literal(2)),
            Comparison(
                expression("rem", expression("sub", g, literal(2)), literal(3)),
                "==",
                literal(1),
            ),
            Comparison(expression("rem", g, literal(2)), "!=", literal(0)),
        ),
    )
    for count in (0, 1, 3, 10, 21):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


def test_unguarded_negative_unsigned_remainder_is_not_a_window(control):
    logical = next(iter(control.loops.values()))
    with pytest.raises(CompileError, match="unsigned arithmetic"):
        window(
            control,
            logical,
            (
                Comparison(
                    expression("rem", expression("sub", logical.ordinal, literal(2)), literal(3)),
                    "==",
                    literal(1),
                ),
            ),
        )


def test_domain_metadata_cannot_hide_a_longer_actual_source_count(control):
    logical = next(iter(control.loops.values()))
    with pytest.raises(CompileError, match="actual endpoint activation") as failure:
        window(control, logical, domains=(LoopDomain(logical.operation, literal(3)),))
    assert failure.value.kind is ErrorKind.UNSUPPORTED


def test_true_source_count_survives_a_looser_metadata_count(control):
    logical = next(iter(control.loops.values()))
    contract = window(
        control,
        logical,
        (Comparison(logical.ordinal, "<", literal(3)),),
        (LoopDomain(logical.operation, literal(8)),),
    )
    for count in (0, 1, 3, 4, 9):
        check_exact_membership(contract, {"n": count, "m": 0}, count)


@pytest.mark.parametrize("kind", ["hole", "nonlinear", "inner_coordinate"])
def test_unproved_sparse_or_noninvariant_activation_is_explicitly_unsupported(control, kind):
    outer, inner = control.loops.values()
    g = outer.ordinal
    test = {
        "hole": Comparison(g, "!=", literal(2)),
        "nonlinear": Comparison(expression("mul", g, g), "<", outer.count),
        "inner_coordinate": Comparison(g, "<", inner.ordinal),
    }[kind]
    assert not control.invariant_over(inner.ordinal, outer.operation)
    with pytest.raises(CompileError, match="loop-varying guard") as failure:
        window(control, outer, (test,))
    assert failure.value.kind is ErrorKind.UNSUPPORTED


def test_parameter_dependent_parent_activation_is_not_assumed_to_be_one_window(control):
    parent = next(iter(control.loops.values()))
    count = Expr("argument", ("m",))
    predicate = Comparison(
        count, ">", expression("rem", expression("mul", parent.ordinal, count), literal(2))
    )
    # A child has a residue member on even parents only when m=1, and on
    # every parent when m>=2. One unconditional parent interval is unsound.
    for child_count in (1, 2, 3):
        active = [
            p for p in range(6) if predicate.evaluate({"m": child_count, parent.ordinal.args[0]: p})
        ]
        assert active == ([0, 2, 4] if child_count == 1 else list(range(6)))
    with pytest.raises(CompileError, match="exact interval/residue") as failure:
        window(control, parent, (Comparison(count, ">=", literal(1)), predicate))
    assert failure.value.phase == "event-window-membership"
    assert any(item.startswith("unresolved:") for item in failure.value.witness)


def test_unknown_scalar_carry_cannot_be_a_window_parameter():
    source = N.from_source("""
    def carry(n:N.i64):
        with N.Kernel():
            count=0
            for i in N.serial(n):
                count=count-1
            for g in N.Pipelined(n):
                pass
""")
    control = build_control_facts(lower_native_structure(source).module)
    serial, pipeline = control.loops.values()
    carry = control.environment.expression(serial.operation.results[0])
    assert not control.invariant_over(carry, pipeline.operation)
    with pytest.raises(CompileError, match="loop-varying guard"):
        window(control, pipeline, (Comparison(pipeline.ordinal, "<", carry),))

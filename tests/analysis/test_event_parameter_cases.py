"""Uniform alternatives preserve exact membership without runtime expansion."""

from dataclasses import replace

import pytest

from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison
from ascendflow.scalar import Expr, expression, literal
from tests.fixtures.event_slots import nested_slot_source
from tests.support.structured import selected


@pytest.fixture(scope="module")
def base():
    return selected("""
    def clocks(X:N.Tensor((n,64),N.f32),n:N.int64,m:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            for p in N.Pipelined(n,stages=2):
                with N.Stage(0):
                    value=N.copy(X[p,:],x[:])
                with N.Stage(1):
                    N.mul(value,2.0,out=y[:])
    """)


def parameter_domain(base, period):
    logical = next(iter(base.dependencies.memory.control.loops.values()))
    symbol = next(
        name for name in base.dependencies.memory.control.environment.facts if name.name == "m"
    )
    m = Expr("argument", (symbol,))
    predicate = Comparison(
        m, ">", expression("rem", expression("mul", logical.ordinal, m), literal(period))
    )
    original = next(
        n for n in base.notifications if n.sources[0].coordinates == n.consumers[0].coordinates
    )
    notification = replace(
        original,
        requirements=(),
        domains=(),
        origins=(),
        sources=tuple(replace(b, guard=ActionGuard((predicate,))) for b in original.sources),
        consumers=tuple(replace(b, guard=ActionGuard((predicate,))) for b in original.consumers),
    )
    # This explicitly supplied event is a window/partition contract fixture,
    # not a claim that notification selection discovers the predicate.
    selection = replace(
        base,
        dependencies=replace(base.dependencies, required=()),
        notifications=(notification,),
        coverage=(),
    )
    return EventDomains(selection, ResourceWork()), notification


@pytest.mark.parametrize("period", (2, 5))
def test_uniform_partition_is_disjoint_cover_and_keeps_interleaved_residues(base, period):
    domains, notification = parameter_domain(base, period)
    family = domains.cases[notification]
    for m in range(2 * period + 1):
        context = {"m": m, "n": 17, "$group": 0, "$sid": 0}
        active = [case for case in family if case.condition.evaluate(context)]
        assert len(active) == 1
        case = active[0]
        expected = [p for p in range(17) if m > (p * m) % period]
        if not expected:
            assert case.empty
            assert case.windows == ()
            continue
        assert not case.empty
        (window,) = case.windows
        actual = [
            p for p in range(17) if window.interval.evaluate(context | {window.ordinal.args[0]: p})
        ]
        assert actual == expected
        assert [
            window.rank().evaluate(context | {window.ordinal.args[0]: p}) for p in actual
        ] == list(range(len(actual)))
        if period == 5 and m == 2:
            assert tuple(r.constant for r in window.residues) == (0, 3)
    assert len(family) <= period + 2


def test_empty_intersection_does_not_hide_mismatched_endpoints(base):
    from ascendflow.diagnostics import CompileError

    original = next(
        n for n in base.notifications if n.sources[0].coordinates == n.consumers[0].coordinates
    )
    m = Expr("argument", ("m",))
    notification = replace(
        original,
        requirements=(),
        domains=(),
        origins=(),
        sources=tuple(
            replace(b, guard=ActionGuard((Comparison(m, "==", literal(0)),)))
            for b in original.sources
        ),
        consumers=tuple(
            replace(b, guard=ActionGuard((Comparison(m, ">", literal(0)),)))
            for b in original.consumers
        ),
    )
    selection = replace(
        base,
        dependencies=replace(base.dependencies, required=()),
        notifications=(notification,),
        coverage=(),
    )
    with pytest.raises(CompileError, match="activation domains are not proved equivalent"):
        EventDomains(selection, ResourceWork())


def test_real_nested_slot_selection_preserves_actual_endpoint_domains():
    slots = 2
    domains = EventDomains(selected(nested_slot_source(slots)), ResourceWork())
    parent, child = domains.control.loops.values()
    for notification, family in domains.cases.items():
        for m in range(slots + 2):
            uniform = {"n": 3, "m": m, "$group": 0, "$sid": 0}
            (case,) = tuple(c for c in family if c.condition.evaluate(uniform))
            for p in range(5):
                for g in range(m + 2):
                    context = uniform | {parent.ordinal.args[0]: p, child.ordinal.args[0]: g}
                    original = ActionGuard(domains.tests[notification]).evaluate(context)
                    actual = (
                        False
                        if case.empty
                        else domains._instance(
                            notification.sources[0], {}, case=case
                        ).guard.evaluate(context)
                    )
                    assert actual == original


def test_more_general_memory_frontier_keeps_source_order_without_exact_residue_claims():
    from ascendflow import compile
    from ascendflow import language as N
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    result = compile(N.from_source(nested_slot_source(5)))
    final = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for n in (0, 1, 3):
        for m in (0, 1, 2, 5, 7):
            check_structured_events(final, {"n": n, "m": m}, logical_reference=original)


def test_endpoint_cannot_borrow_another_endpoints_division_prefix(base):
    from ascendflow.diagnostics import CompileError

    original = next(
        n for n in base.notifications if n.sources[0].coordinates == n.consumers[0].coordinates
    )
    ordinal = next(iter(base.dependencies.memory.control.loops.values())).ordinal
    prerequisite = Comparison(ordinal, ">", literal(0))
    division = Comparison(expression("rem", literal(1), ordinal), "==", literal(0))
    notification = replace(
        original,
        requirements=(),
        domains=(),
        origins=(),
        sources=tuple(
            replace(b, guard=ActionGuard((prerequisite, division))) for b in original.sources
        ),
        consumers=tuple(
            replace(b, guard=ActionGuard((division, prerequisite))) for b in original.consumers
        ),
    )
    selection = replace(
        base,
        dependencies=replace(base.dependencies, required=()),
        notifications=(notification,),
        coverage=(),
    )
    with pytest.raises(CompileError, match="ordered activation prefix"):
        EventDomains(selection, ResourceWork())


def test_mathematically_empty_domain_does_not_hide_undefined_source_remainder(base):
    from ascendflow.diagnostics import CompileError

    original = next(
        n for n in base.notifications if n.sources[0].coordinates == n.consumers[0].coordinates
    )
    ordinal = next(iter(base.dependencies.memory.control.loops.values())).ordinal
    bad = Comparison(
        expression("rem", expression("sub", ordinal, literal(1)), literal(1)), "!=", literal(0)
    )
    notification = replace(
        original,
        requirements=(),
        domains=(),
        origins=(),
        sources=tuple(replace(b, guard=ActionGuard((bad,))) for b in original.sources),
        consumers=tuple(replace(b, guard=ActionGuard((bad,))) for b in original.consumers),
    )
    selection = replace(
        base,
        dependencies=replace(base.dependencies, required=()),
        notifications=(notification,),
        coverage=(),
    )
    with pytest.raises(CompileError, match="ordered activation prefix"):
        EventDomains(selection, ResourceWork())

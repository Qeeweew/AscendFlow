"""Exact child existence is the input of the containing event window."""

from dataclasses import FrozenInstanceError

import pytest

from tests.support.structured import prepared, selected

NESTED = """
def nested(X:N.Tensor((n,2,64),N.f32),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((2,64))
        for outer in N.serial(n):
            for inner in N.Pipelined(2):
                with N.Stage(0):
                    value=N.copy(X[outer,inner,:],x[inner,:])
                with N.Stage(1):
                    N.mul(value,2.0,out=x[inner,:])
"""


@pytest.fixture(scope="module")
def domains():
    from ascendflow.analysis.event_families import build_event_family

    result = prepared(selected(NESTED))
    # Test the window rule explicitly. Stable cell credits in normal
    # generation deliberately need pairing, not dense ordinal windows.
    families = {n: build_event_family(result, n) for n in result.notifications if n in result.cases}
    return result, families


def test_child_membership_is_projected_once_before_parent_window(domains):
    domains, families = domains
    projected = [
        case.projections
        for cases in families.values()
        for case in cases
        if len(case.projections) > 1
    ]
    assert projected
    for case in (case for cases in families.values() for case in cases):
        projections, windows = case.projections, case.windows
        assert tuple(p.loop for p in projections) == tuple(w.loop for w in reversed(windows))
        for index, projection in enumerate(projections[:-1]):
            parent = windows[-index - 2]
            assert parent.membership.comparisons == projection.guard.comparisons


def test_small_exact_existence_matches_actual_member_enumeration(domains):
    domains, families = domains
    for case in (case for cases in families.values() for case in cases):
        windows = case.windows
        for count in (0, 1, 2, 3):
            for outer in range(4):
                context = {"n": count, "$group": 0, "$sid": 0}
                for logical in domains.control.loops.values():
                    context[logical.ordinal.args[0]] = outer
                for window in windows:
                    members = [
                        value
                        for value in range(5)
                        if window.membership.evaluate(context | {window.ordinal.args[0]: value})
                    ]
                    assert bool(members) == window.project_exists().guard.evaluate(context)
                    if members:
                        first, last = window.first, window.last()
                        assert first.evaluate(context) == min(members)
                        assert last.evaluate(context) == max(members)


def test_established_domain_identity_and_maps_are_read_only(domains):
    domains, _ = domains
    with pytest.raises(FrozenInstanceError):
        domains.notifications = ()
    with pytest.raises(TypeError):
        domains.cases[domains.notifications[0]] = ()


def test_nested_ports_match_real_messages_with_empty_children_and_parent_dependent_residues():
    from ascendflow.analysis.event_families import build_event_family
    from ascendflow.target.ascend910b import Engine

    source = """
def dependent(X:N.Tensor((n,5,64),N.f32),n:N.i64):
    with N.Kernel(1):
        x=N.alloc_ub((64,))
        for outer in N.serial(n):
            for inner in N.serial(outer%5):
                if inner%2==outer%2:
                    N.copy(X[outer,inner,:],x)
                    N.load_scalar(x[:1])
"""
    domains = prepared(selected(source))
    notification = next(
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.S
    )
    cases = build_event_family(domains, notification)
    outer, inner = domains.control.loops
    for count in (0, 1, 2, 3, 5, 7, 13):
        context = {"n": count, "$group": 0, "$sid": 0}
        # Direct source interpretation: neither window projection, endpoint
        # selection nor production order supplies these actual messages.
        members = [(a, b) for a in range(count) for b in range(a % 5) if b % 2 == a % 2]
        active_cases = [c for c in cases if c.condition.evaluate(context)]
        assert len(active_cases) == 1
        (case,) = active_cases
        if case.empty:
            assert not members
            continue
        for last in (False, True):
            sources, consumers = domains._boundary(notification, case, None, 0, last=last)
            for point in sources + consumers:
                assert point.guard.evaluate(context) == bool(members)
                if members:
                    coordinates = dict(point.coordinates)
                    assert (
                        coordinates[outer].evaluate(context),
                        coordinates[inner].evaluate(context),
                    ) == members[-1 if last else 0]
        if not members:
            continue
        # The parent window ranks only invocations with actual descendants;
        # it must skip empty children rather than invent a FIRST/LAST for one.
        calls = list(dict.fromkeys(a for a, _ in members))
        for a in range(count):
            current = context | {domains.control.loops[outer].ordinal.args[0]: a}
            assert case.window(outer).membership.evaluate(current) == (a in calls)
            if a not in calls:
                continue
            assert case.window(outer).rank().evaluate(current) == calls.index(a)
            for last in (False, True):
                sources, consumers = domains._boundary(notification, case, outer, 0, last=last)
                expected = [b for p, b in members if p == a]
                for point in sources + consumers:
                    assert point.guard.evaluate(current)
                    assert (
                        dict(point.coordinates)[inner].evaluate(current)
                        == expected[-1 if last else 0]
                    )

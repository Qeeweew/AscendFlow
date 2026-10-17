"""Exact child existence is the input of the containing event window."""

from dataclasses import FrozenInstanceError

import pytest

from tests.support.structured import prepared, selected

NESTED = """
def nested(X:N.Tensor((n,2,64),N.f32),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((2,64))
        for outer in N.serial(n):
            for inner in N.Pipelined(2,stages=2):
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
                        first, last = window.first_last()
                        assert first.evaluate(context) == min(members)
                        assert last.evaluate(context) == max(members)


def test_established_domain_identity_and_maps_are_read_only(domains):
    domains, _ = domains
    with pytest.raises(FrozenInstanceError):
        domains.notifications = ()
    with pytest.raises(TypeError):
        domains.cases[domains.notifications[0]] = ()

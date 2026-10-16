"""Exact endpoint and invocation facts belong to one progress composition."""

from dataclasses import replace

import pytest

from tests.analysis.test_progress_parent_interfaces import PARENT_REUSE
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_progress import ProgressPort, _Composer
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.scalar import expression, literal


@pytest.fixture(scope="module")
def domains():
    return prepared(selected(PARENT_REUSE))


def invocation(domains):
    composer = _Composer(domains)
    parent, child = domains.control.loops
    notification = next(n for n in domains.notifications if composer.owners[n] is parent)
    (case,) = domains.cases[notification]
    source = ProgressPort(notification, notification.sources[0], "set", case)
    wait = ProgressPort(notification, notification.consumers[0], "wait", case)
    return composer, parent, child, source, wait


def count_instances(monkeypatch):
    calls = []
    original = EventDomains._instance

    def instance(self, bound, bindings, *, case, extra_coordinate_guards=()):
        calls.append((bound, bindings, case, extra_coordinate_guards))
        return original(
            self, bound, bindings, case=case, extra_coordinate_guards=extra_coordinate_guards
        )

    monkeypatch.setattr(EventDomains, "_instance", instance)
    return calls


def count_proofs(domains, monkeypatch):
    calls = []
    arithmetic = type(domains.arithmetic)
    original = arithmetic.prove

    def prove(self, test, assumptions):
        calls.append((test, assumptions))
        return original(self, test, assumptions)

    monkeypatch.setattr(arithmetic, "prove", prove)
    return calls


def test_equal_own_and_imported_endpoints_share_one_bound_fact(domains, monkeypatch):
    composer, _, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    bound = composer._instance(source)
    imported = replace(source, bound=replace(source.bound))
    assert composer._instance(imported) is bound
    assert composer._ambient(source) is bound.guard.comparisons
    assert composer._coordinates(imported) is bound.coordinates
    assert len(calls) == 1


def test_guard_and_proof_binding_remain_part_of_the_endpoint_fact(domains, monkeypatch):
    composer, parent, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    impossible = Comparison(literal(0), "==", literal(1))
    inactive = replace(
        source,
        bound=replace(
            source.bound, guard=ActionGuard(source.bound.guard.comparisons + (impossible,))
        ),
    )
    ordinal = domains.control.loops[parent].ordinal
    shifted = replace(source, bindings=((ordinal, expression("add", ordinal, literal(1))),))
    original = composer._instance(source)
    assert impossible not in original.guard.comparisons
    assert impossible in composer._ambient(inactive)
    shifted_coordinates = dict(composer._coordinates(shifted))
    original_coordinates = dict(original.coordinates)
    context = {ordinal.args[0]: 2}
    assert shifted_coordinates[parent].evaluate(context) == 2
    assert original_coordinates[parent].evaluate(context) == 1
    assert source.bound.coordinates == original.coordinates
    assert len(calls) == 3


def test_actual_placement_is_not_aliased_by_notification_and_coordinates(domains, monkeypatch):
    composer, _, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    before = replace(source, bound=replace(source.bound, placement=Placement.BEFORE))
    assert composer._instance(source).placement is Placement.AFTER
    assert composer._instance(before).placement is Placement.BEFORE
    assert len(calls) == 2


@pytest.mark.parametrize("same_invocation", (True, False))
def test_invocation_result_is_established_once_including_impossible(
    domains, monkeypatch, same_invocation
):
    composer, _, child, source, wait = invocation(domains)
    calls = count_proofs(domains, monkeypatch)
    other = source if same_invocation else wait
    result = composer._invocation_guard(child, source, other)
    assert (result is not None) is same_invocation
    first_checks = len(calls)
    assert first_checks
    assert composer._invocation_guard(child, replace(source), replace(other)) is result
    assert len(calls) == first_checks


def test_invocation_fact_includes_the_complete_premise_context(domains, monkeypatch):
    composer, _, child, source, _ = invocation(domains)
    original = composer._ambient
    premises = []
    # Supply two invariant premise contexts for the same actual placement.
    # A stronger, inconsistent context must not reuse the first answer.
    monkeypatch.setattr(composer, "_ambient", lambda port: original(port) + tuple(premises))
    assert composer._invocation_guard(child, source, source) is not None
    premises.append(Comparison(literal(0), "==", literal(1)))
    assert composer._invocation_guard(child, source, source) is None


def test_bound_facts_do_not_outlive_the_composer(domains, monkeypatch):
    first, _, _, source, _ = invocation(domains)
    second = _Composer(domains)
    calls = count_instances(monkeypatch)
    assert first._instance(source) is not second._instance(source)
    assert len(calls) == 2

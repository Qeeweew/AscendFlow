"""Candidate horizons use actual parent steps, before any reuse-path search.

The real nested QK/PV fixture distinguishes a parent's stage from child-local
stages. Only selected domains and their bounded interface candidates are built.
"""

import pytest

from tests.fixtures.nested import shared_l0c_qk_pv
from tests.support.structured import selected
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_lifetimes import _candidates
from xdsl_npu.target.ascend910b import Engine


@pytest.fixture(scope="module")
def shared_candidates():
    domains = EventDomains(selected(shared_l0c_qk_pv), ResourceWork())
    outer, qk, pv = domains.control.loops
    candidates = tuple(_candidates(domains, outer, 0, 2, needs_clock=False))
    assert domains.work.lifetime_interfaces == domains.work.path_states == 0
    return domains, (outer, qk, pv), candidates


def _peer(domains, source, consumer, distance):
    source = domains.memory.templates[source].operation
    consumer = domains.memory.templates[consumer].operation
    (peer,) = tuple(
        peer
        for peer in domains.notifications
        if tuple(bound.operation for bound in peer.sources) == (source,)
        and tuple(bound.operation for bound in peer.consumers) == (consumer,)
        and peer.distance == distance
    )
    return peer


def _values(domains, outer, count, ordinal):
    return {
        "n": count,
        "$group": 0,
        "$sid": 0,
        **{
            logical.ordinal.args[0]: ordinal if loop is outer else 0
            for loop, logical in domains.control.loops.items()
        },
    }


def test_pv_parent_stage_exposes_previous_outer_release_with_real_activation(shared_candidates):
    domains, (outer, _, pv), candidates = shared_candidates
    peer = _peer(domains, 7, 5, 2)
    assert peer.source_engine is Engine.MTE1 and peer.destination_engine is Engine.MTE2
    assert peer.clock is pv
    for bound in peer.sources + peer.consumers:
        task = domains.dispatch.tasks[bound.operation, bound.actor]
        assert task.scope.operation is pv and task.scope.parent.operation is outer
        assert task.scope.parent_stage == 1
        assert dict(bound.coordinates)[outer] == domains.control.loops[outer].ordinal

    interfaces = [(sources, consumers) for item, sources, consumers in candidates if item is peer]
    # Six child uses are represented by their first and last interfaces, at
    # each of two actual parent steps, regardless of the runtime outer count.
    assert len(interfaces) == 4
    samples = _values(domains, outer, 8, 3)
    previous = [
        (sources, consumers)
        for sources, consumers in interfaces
        if dict(sources[0].coordinates)[outer].evaluate(samples) == 2
    ]
    assert len(previous) == 2
    child_pairs = set()
    for sources, consumers in previous:
        (source,), (consumer,) = sources, consumers
        source_coordinates, consumer_coordinates = (
            dict(source.coordinates),
            dict(consumer.coordinates),
        )
        child_pairs.add(
            (
                source_coordinates[pv].evaluate(samples),
                consumer_coordinates[pv].evaluate(samples),
            )
        )
        for count in (0, 1, 3, 1_000_000):
            for ordinal in (0, 1, 2, count, count + 1):
                values = _values(domains, outer, count, ordinal)
                assert source_coordinates[outer].evaluate(values) == ordinal - 1
                assert consumer_coordinates[outer].evaluate(values) == ordinal - 1
                expected = 0 <= ordinal - 1 < count
                assert source.guard.evaluate(values) == expected
                assert consumer.guard.evaluate(values) == expected
    assert child_pairs == {(0, 2), (5, 7)}


@pytest.mark.parametrize("source,consumer", ((0, 1), (3, 4)))
def test_qk_child_stages_do_not_shift_the_parent_candidate_interval(
    shared_candidates, source, consumer
):
    domains, (outer, qk, _), candidates = shared_candidates
    tasks = [task for task in domains.dispatch.tasks.values() if task.scope.operation is qk]
    assert {task.stage for task in tasks} == {0, 1, 2, 3}
    assert all(task.scope.parent_stage == 0 for task in tasks)
    peer = _peer(domains, source, consumer, 0)
    assert peer.clock is qk
    interfaces = [(sources, consumers) for item, sources, consumers in candidates if item is peer]
    assert len(interfaces) == 4
    samples = _values(domains, outer, 8, 3)
    points = set()
    for sources, consumers in interfaces:
        (source,), (consumer,) = sources, consumers
        before, after = dict(source.coordinates), dict(consumer.coordinates)
        assert before[outer].evaluate(samples) == after[outer].evaluate(samples)
        assert before[qk].evaluate(samples) == after[qk].evaluate(samples)
        points.add((before[outer].evaluate(samples) - 3, before[qk].evaluate(samples)))
    assert points == {(0, 0), (0, 7), (1, 0), (1, 7)}

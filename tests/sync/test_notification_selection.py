"""Finite retained-control checks of the one logical notification contract.

These tests check selection/placement and exact notification existence. They
are not hardware retirement, binary-event lifetime or progress proofs.
"""

from collections import Counter
from dataclasses import replace
from importlib import import_module

import pytest

from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.analysis.structured_regions import AccessOccurrence
from ascendflow.diagnostics import CompileError
from ascendflow.indexing import arguments
from ascendflow.passes.notification_binding import bind_notifications
from ascendflow.target.ascend910b import Actor, Engine
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv, partitioned_cv, shared_l0c_qk_pv
from tests.models.structured_domains import actual_dispatch, need_cases, occurrence_key, point_key
from tests.support.structured import selected


@pytest.fixture(
    scope="module",
    params=[
        scale,
        batched_matmul,
        nested_copy,
        nested_cv,
        partitioned_cv,
        shared_l0c_qk_pv,
        attention_target,
    ],
)
def actual_selection(request):
    return selected(request.param)


def test_one_selection_covers_every_actual_required_relation(actual_selection):
    result = actual_selection
    assert not result.unresolved
    assert len(result.coverage) == len(result.dependencies.required)
    assert {id(c.requirement) for c in result.coverage} == {
        id(n) for n in result.dependencies.required
    }
    for notification in result.notifications:
        assert notification.requirements
        assert all(
            point.coordinates == tuple(point.coordinates)
            for point in (
                *notification.sources,
                *notification.consumers,
            )
        )
        if notification.mechanism is NotificationKind.CROSS_EVENT:
            a, b = (
                {p.actor for p in notification.sources},
                {p.actor for p in notification.consumers},
            )
            assert (a, b) in [
                ({Actor.AIC}, {Actor.AIV0, Actor.AIV1}),
                ({Actor.AIV0, Actor.AIV1}, {Actor.AIC}),
            ]
            assert sorted((len(notification.sources), len(notification.consumers))) == [1, 2]
        elif notification.mechanism is NotificationKind.LOCAL_EVENT:
            assert notification.source_engine is not notification.destination_engine
            assert notification.destination_engine is not Engine.S


def test_actual_captures_and_acquires_cover_sources_and_earliest_consumers(actual_selection):
    result, memory = actual_selection, actual_selection.dependencies.memory
    actors = {actor for _, actor in result.dispatch.tasks}
    for count in (0, 1, 2, 3):
        uniform = {
            "tiles": count,
            "n": count,
            "Sq": 512,
            "Sk": max(1, count) * 1024,
            "$group": 0,
            "$sid": 0,
        }
        traces = {actor: actual_dispatch(result, uniform, actor, ports=True) for actor in actors}
        positions = {
            key: index for actor, events in traces.items() for index, (key, _) in enumerate(events)
        }
        for coverage in result.coverage:
            need, notification = coverage.requirement, coverage.notification
            assert isinstance(need.source, AccessOccurrence)
            if coverage.ownership is not None:
                owner = coverage.ownership
                assert owner.cell == need.cell and need in owner.requirements
                assert owner.cell.private is not None
                assert all(
                    point.participant is owner.cell.private
                    for body in owner.rounds
                    for point in (body.writer,) + body.readers
                )
                # This is a complete round obligation, not one relocated
                # capture/acquire pair. Actual tokens/bytes are checked by
                # test_persistent_cell_ownership's independent event model.
                continue
            for context in need_cases(need, uniform, memory):
                source = occurrence_key(need.source, context, memory)
                consumer = occurrence_key(need.consumer, context, memory)
                if source not in positions or consumer not in positions:
                    # Ambient Task regions remain part of occurrence existence;
                    # a same-branch internal need is inert outside that branch.
                    assert source not in positions and consumer not in positions
                    continue
                if notification is None:
                    assert coverage.ordering is not None
                    assert source[1] is consumer[1] and positions[source] < positions[consumer]
                    continue
                capture = next(p for p in notification.sources if p.actor is source[1])
                acquire = next(p for p in notification.consumers if p.actor is consumer[1])
                if notification.mechanism is NotificationKind.PIPE_FENCE:
                    # The real fence captures the already-issued same-pipe
                    # prefix at the acquire. There is no source-side Set.
                    capture = acquire
                a, b = point_key(capture, context, memory), point_key(acquire, context, memory)
                assert a in positions and b in positions
                assert positions[b] <= positions[consumer]
                if notification.mechanism is NotificationKind.PIPE_FENCE:
                    assert positions[source] < positions[b]
                else:
                    assert positions[source] <= positions[a]


def test_selected_set_and_wait_existence_is_exact_for_empty_short_and_reentry(actual_selection):
    result = actual_selection
    actors = {actor for _, actor in result.dispatch.tasks}
    control = result.dependencies.memory.control
    maximum_group = control.environment.facts[control.group.args[0]].maximum
    assert maximum_group is not None
    for count, requested_group in [(0, 0), (1, 0), (2, 0), (3, 0), (2, 19)]:
        group = min(requested_group, maximum_group)
        uniform = {
            "tiles": count,
            "n": count,
            "Sq": 256 if requested_group else 512,
            "Sk": max(1, count) * 1024,
            "$group": group,
            "$sid": 0,
        }
        traces = {actor: actual_dispatch(result, uniform, actor, ports=True) for actor in actors}
        for notification in result.notifications:
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                continue
            points = (*notification.sources, *notification.consumers)
            common = set.intersection(*(set(dict(p.substitutions)) for p in points))
            free = [
                coordinate
                for coordinate in common
                if all(
                    coordinate.args[0] in arguments(dict(p.substitutions)[coordinate])
                    for p in points
                )
            ]
            counters = []
            for point in points:
                counter = Counter()
                for key, context in traces[point.actor]:
                    if key[0] is point.operation and point.guard.evaluate(context):
                        identity = tuple(
                            (e.args[0], point.expression(e).evaluate(context))
                            for e in sorted(free, key=lambda e: e.order_key)
                        )
                        counter[identity] += 1
                counters.append(counter)
            assert all(counter == counters[0] for counter in counters), (
                notification.name,
                count,
                group,
                counters,
            )


def test_first_count_and_dynamic_residues_are_not_replaced_by_unbounded_activation():
    result = selected(attention_target)
    dynamic = [
        domain
        for n in result.dependencies.required
        for domain in n.domains
        if not isinstance(domain.residue, int)
    ]
    assert dynamic
    assert any(d.first > 0 for n in result.dependencies.required for d in n.domains)
    assert any(p.families for n in result.notifications for p in n.sources)
    assert all(
        bool(p.coordinates) == bool(p.occurrence.frame.loops())
        for n in result.notifications
        for p in (*n.sources, *n.consumers)
    )
    assert any(n.distance is None for n in result.notifications)


@pytest.mark.parametrize("source", [nested_cv, partitioned_cv])
def test_incomplete_mode2_quorum_stays_explicitly_unresolved(source):
    result = selected(source)
    broken = replace(
        result.dependencies,
        required=tuple(
            n
            for n in result.dependencies.required
            if not (n.source.participant is Actor.AIV1 and n.consumer.participant is Actor.AIC)
        ),
    )
    rejected = bind_notifications(broken, result.dispatch)
    assert any("quorum" in issue.reason for issue in rejected.unresolved)
    with pytest.raises(CompileError, match="quorum"):
        rejected.require_closed()


def test_unordered_same_engine_dependency_cannot_be_repaired_by_a_fence():
    with pytest.raises(CompileError, match="same-engine source"):
        selected("""
        def bad(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((2,64))
                for t in N.Pipelined(n):
                    with N.Stage(2,name="first"):
                        N.fill(x[t%2,:],1.0)
                    with N.Stage(0,name="later"):
                        N.mul(x[t%2,:],2.0,out=x[t%2,:])
""")


def test_selection_reuses_owned_memory_dependency_and_native_facts(monkeypatch):
    result = selected(nested_cv)

    def forbidden(*args, **kwargs):
        raise AssertionError("selection rebuilt an owned analysis input")

    targets = [
        ("ascendflow.analysis.structured_memory", "collect_native_memory"),
        ("ascendflow.analysis.structured_dependencies", "build_structured_dependencies"),
        ("ascendflow.target.native", "native_contract"),
    ]
    modules = [(import_module(name), attribute) for name, attribute in targets]
    for module, attribute in modules:
        monkeypatch.setattr(module, attribute, forbidden)
    again = bind_notifications(result.dependencies, result.dispatch)
    again.require_closed()
    assert again == result

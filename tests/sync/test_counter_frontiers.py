"""Counter intervals retain every participant and freeze complete-credit rows."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import color_event_resources, counter_group_usage
from ascendflow.analysis.event_contracts import EventDemand, EventIdPools
from ascendflow.analysis.event_counter_groups import plan_counter_time_groups
from ascendflow.analysis.event_frontier_contracts import FrozenEventRelations
from ascendflow.analysis.event_frontiers import EventFrontiers
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.event_sharing import share_counter_time
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.counter_epochs import counter_epochs
from tests.fixtures.fifo_panels import fifo_panels
from tests.models.structured_events import check_compiled_events
from tests.support.structured import frontier_members, prepared, selected
from tests.sync.test_counter_time_reuse import forward_messages, numeric_reuses


def no_pair_projection(self):
    pytest.fail("frozen counter permissions were expanded into all numeric pairs")


def test_counter_enclosure_checks_the_second_vector_receiver():
    domains = prepared(selected(fifo_panels.specialize(PANELS=3)))
    messages = [
        n
        for n in domains.notifications
        if n.mechanism is NotificationKind.CROSS_EVENT and n.source_engine is Engine.FIX
    ]
    oldest, left, right = messages[:3]
    frontiers = EventFrontiers(domains, HandbackRule(domains), (), None)
    joined = frontiers._join(frontiers._leaf(left), frontiers._leaf(right))
    assert joined is not None
    assert {a.actor for a, _ in joined.enclosure} == {Actor.AIC, Actor.AIV0, Actor.AIV1}
    # Real positions from the earlier receive invert only the second AIV.
    # This isolates the enclosure premise, not admission of this mutation.
    crossed = replace(
        right,
        consumers=tuple(
            next(p for p in oldest.consumers if p.actor is Actor.AIV1)
            if p.actor is Actor.AIV1
            else p
            for p in right.consumers
        ),
    )
    assert frontiers._join(frontiers._leaf(left), frontiers._leaf(crossed)) is None


@pytest.mark.parametrize("feedback", [False, True])
def test_counter_rows_retain_all_small_exhaustive_member_permissions(monkeypatch, feedback):
    definition = counter_epochs
    if not feedback:
        definition = replace(
            definition,
            source=definition.source.replace(
                "N.copy(Feedback[phase - 1, :, :], a1)", "N.copy(A, a1)"
            ).replace("N.copy(Feedback[PHASES - 1, :, :], a1)", "N.copy(A, a1)"),
        )
    plan = synchronize_structure(definition.specialize(PHASES=2))
    monkeypatch.setattr(FrozenEventRelations, "__iter__", no_pair_projection)
    # An empty pool forces construction of optional temporal facts, without
    # pretending that zero hardware IDs could execute the program.
    problem = plan_event_resources(plan.generation, id_pools=EventIdPools(cross=()))
    resources = dict(problem.uses)
    pair = tuple(sorted(resources[n] for n in forward_messages(plan.generation)))
    assert len(pair) == 2
    assert (problem.counter_reuses.get(pair) is not None) == feedback
    observed = False
    for a, b in combinations(range(len(problem.demands)), 2):
        left, right = problem.demands[a], problem.demands[b]
        if left.backlog is None or right.backlog is None:
            continue
        expected = all(
            share_counter_time(plan.generation.domains, plan.generation.handbacks, x, y) is not None
            for x in left.members
            for y in right.members
        )
        permission = problem.counter_reuses.get((a, b))
        if expected:
            assert permission is not None
            observed = True
        if permission is not None:
            (certificate,) = permission
            assert all(
                any(
                    x in frontier_members(p.left)
                    and y in frontier_members(p.right)
                    or y in frontier_members(p.left)
                    and x in frontier_members(p.right)
                    for p in certificate.interfaces
                )
                for x in left.members
                for y in right.members
            )
    if feedback:
        assert observed
    for count in (0, 1, 3):
        # The feedback closes repeated controller-body invocations. Its
        # removal tests one kernel invocation, retaining runtime-loop reuse;
        # separate launches obtain their join from the runtime stream.
        check_compiled_events(plan, {"n": count}, invocations=2 if feedback else 1)


def test_counter_numbering_consumes_rows_without_permission_pair_projection(monkeypatch):
    reuses = numeric_reuses(17, combinations(range(17), 2))
    problem = SimpleNamespace(
        demands=tuple(EventDemand((), 1, (), backlog=15) for _ in range(17)),
        reservations=(),
        peers=(),
        counter_groups=(),
        relations=FrozenEventRelations(()),
        counter_reuses=reuses,
        id_pools=EventIdPools(cross=(0, 3)),
    )
    monkeypatch.setattr(FrozenEventRelations, "__iter__", no_pair_projection)
    ids, _ = color_event_resources(problem)
    assert {values[0] for values in ids} <= {0, 3}
    for flag in {values[0] for values in ids}:
        members = {n for n, values in enumerate(ids) if values == (flag,)}
        assert reuses.cover_group(members) is not None
        assert counter_group_usage(problem.demands, members, reuses, ()) == ("time", 15)


def test_fifo_child_time_composition_checks_both_rows_without_pair_projection(monkeypatch):
    demands = tuple(EventDemand((), 1, (), backlog=7) for _ in range(4))
    reuses = numeric_reuses(4, ((0, 2), (0, 3), (1, 2), (1, 3)))
    children = (SimpleNamespace(resources=(0, 1)), SimpleNamespace(resources=(2, 3)))
    monkeypatch.setattr(FrozenEventRelations, "__iter__", no_pair_projection)
    groups = plan_counter_time_groups(SimpleNamespace(order=object()), demands, children, reuses)
    assert groups and all(group.rectangles for group in groups)
    assert counter_group_usage(
        demands,
        range(4),
        reuses,
        tuple(frozenset(c.resources) for c in children),
        time_packs=tuple(c.groups for c in groups),
    ) == ("time_fifo", 14)
    rows = list(reuses._rows)
    rows[3] = tuple((mask & ~1, proof) for mask, proof in rows[3] if mask & ~1)
    incomplete = FrozenEventRelations(rows)
    assert incomplete.cover_groups((frozenset((0, 1)), frozenset((2, 3)))) is None
    remaining = plan_counter_time_groups(
        SimpleNamespace(order=object()), demands, children, incomplete
    )
    assert all(set(group.resources) != set(range(4)) for group in remaining)
    assert (
        counter_group_usage(
            demands,
            range(4),
            incomplete,
            tuple(frozenset(c.resources) for c in children),
            time_packs=tuple(c.groups for c in remaining),
        )
        is None
    )

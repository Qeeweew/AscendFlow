"""Whole-credit consumption frontiers on real native and event programs."""

from dataclasses import replace
from itertools import combinations

import pytest

from ascendflow import language as N
from ascendflow.analysis import event_planning
from ascendflow.analysis.event_frontier_contracts import (
    CreditSeparation,
    FrozenEventRelations,
    LocalEventReuse,
)
from ascendflow.analysis.event_frontiers import LocalEventFrontiers
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events
from tests.support.structured import prepared, selected
from tests.sync.test_stream_event_sharing import conditional_loads, disjoint_loop_episodes


def sequential(count, *, scalar=True):
    body = "\n".join(
        f"        N.fill(x[{i},:],1.0)\n"
        + (f"        N.load_scalar(x[{i},:1])" if scalar else f"        N.copy(x[{i},:],O[{i},:])")
        for i in range(count)
    )
    return N.from_source(
        f'def sequential(O:N.Tensor(({count},64),N.f32,access="write")):\n'
        '    with N.Kernel(1,kind="vector"):\n'
        f"        x=N.alloc_ub(({count},64),N.f32)\n" + body + "\n",
        disjoint_external=True,
    )


def test_real_sequential_frontiers_remove_quadratic_sharing_queries():
    measurements = []
    for count in (16, 64, 256):
        plan = synchronize_structure(sequential(count))
        work = plan.resources.work
        assert work.frontier_rectangles
        assert work.frontier_queries < 4 * count
        assert work.sharing_pairs < 4 * count
        assert work.local_reuse_row_checks <= count
        assert work.local_reuse_rectangles < 2 * count
        assert all(isinstance(w, LocalEventReuse) for w in plan.resources.sharing)
        from ascendflow.analysis.resource_report import event_resource_report

        report = event_resource_report(plan.resources)
        assert len(report["local_event_frontiers"]) < 4 * count
        assert len(report["local_event_certificates"]) < 2 * count
        measurements.append(work.frontier_queries)
        if count == 16:
            # Independent event consumption, byte ownership and repeated-body
            # checks, without assuming completion at a loop/control exit.
            check_compiled_events(plan, invocations=2)
    assert measurements[2] < 5 * measurements[1]
    assert measurements[1] < 5 * measurements[0]


def test_distinct_waits_at_one_anchor_do_not_supply_an_enclosure_order():
    domains = prepared(selected(sequential(2)))
    item = domains.notifications[0]
    duplicate = replace(item, name="same_native_anchor")
    frontiers = LocalEventFrontiers(domains, HandbackRule(domains), (), None)
    # This isolates enclosure, without claiming the duplicate has a paired
    # protocol or owns another physical credit. Real dispatch positions alone
    # cannot order two notification instructions inserted at the same anchor.
    assert frontiers._join(frontiers._leaf(item), frontiers._leaf(duplicate)) is None


@pytest.mark.parametrize(
    "source",
    [
        sequential(12),
        conditional_loads(reuse=True),
        disjoint_loop_episodes(scalar=True),
    ],
)
def test_frontiers_retain_every_exhaustive_complete_credit_permission(monkeypatch, source):
    original = event_planning._Planning.relations
    observed = []

    def compare(self, peers):
        current = original(self, peers)
        for a, b in combinations(range(len(self.demands)), 2):
            left, right = self.demands[a], self.demands[b]
            if left.backlog is not None or right.backlog is not None:
                continue
            if left.persistent is not None or right.persistent is not None:
                continue
            if not set(left.banks).intersection(right.banks).intersection(self.pressured):
                continue
            expected = all(
                self.compatible(x, y) is not None for x in left.members for y in right.members
            )
            permission = current.get((a, b))
            if expected:
                assert permission is not None
            if permission is not None:
                (certificate,) = permission
                assert isinstance(certificate, CreditSeparation)
                assert all(
                    any(
                        x in p.left.members
                        and y in p.right.members
                        or y in p.left.members
                        and x in p.right.members
                        for p in certificate.interfaces
                    )
                    for x in left.members
                    for y in right.members
                )
        observed.append(True)
        return current

    monkeypatch.setattr(event_planning._Planning, "relations", compare)
    plan = synchronize_structure(source)
    assert observed
    if "Meta" not in source.source and "n:" not in source.source:
        check_compiled_events(plan, invocations=2)


def test_group_masks_reject_one_missing_credit_permission(monkeypatch):
    from ascendflow.analysis import structured_resources

    original = structured_resources.allocate_event_ids
    observed = []

    def check(problem):
        allocation = original(problem)
        for group in allocation.sharing:
            if not isinstance(group, LocalEventReuse):
                continue
            members = {r for r, _ in group.credits}
            assert problem.relations.cover_group(members)
            assert all(
                problem.relations.get((a, b)) is not None
                for a, b in combinations(sorted(members), 2)
            )
            a, b = sorted(members)[:2]
            rows = list(problem.relations._rows)
            rows[a] = tuple(
                (mask & ~(1 << b), proof) for mask, proof in rows[a] if mask & ~(1 << b)
            )
            incomplete = FrozenEventRelations(problem.relations._counters, rows)
            assert incomplete.cover_group(members) is None
            observed.append(True)
        return allocation

    monkeypatch.setattr(structured_resources, "allocate_event_ids", check)
    synchronize_structure(sequential(16))
    assert observed

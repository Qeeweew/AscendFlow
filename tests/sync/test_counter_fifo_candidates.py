"""Whole FIFO choices establish their actual word without member-pair queries."""

from dataclasses import replace

import pytest

from ascendflow.analysis import event_planning
from ascendflow.analysis.event_contracts import EventIdPools
from ascendflow.analysis.event_counter_groups import _certificate
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.fifo_panels import fifo_panels
from tests.models.structured_events import check_compiled_events


@pytest.mark.parametrize("legal_ids", [tuple(range(16)), (1, 5, 9, 14)])
def test_nine_panel_word_uses_only_legal_ids_and_preserves_both_readers(monkeypatch, legal_ids):
    # Restrict the target pool to an existing legal subset; no proof rule,
    # candidate generation or capacity check is replaced by this regression.
    if len(legal_ids) == 16:

        def forbidden(*args, **kwargs):
            raise AssertionError("a fitting certified FIFO partition needs no temporal queries")

        monkeypatch.setattr(event_planning, "share_counter_time", forbidden)
    plan = synchronize_structure(fifo_panels, id_pools=EventIdPools(cross=legal_ids))
    assert all(
        set(assignment.ids) <= set(legal_ids)
        for assignment in plan.resources.assignments
        if not assignment.banks
    )
    assert 0 < plan.resources.work.allocated_cross_ids <= len(legal_ids)
    assert 0 < plan.resources.work.maximum_counter_backlog <= 15
    assert plan.resources.counter_groups
    assert plan.resources.work.counter_structure_ids <= len(legal_ids)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count})


def test_complete_credits_retain_distinct_sequential_loop_words():
    plan = synchronize_structure(fifo_panels.specialize(EPOCHS=2))
    assert plan.resources.work.allocated_cross_ids <= 16
    assert any(group.certificate.episodes for group in plan.resources.counter_groups)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, max_commands=10000)


def test_whole_fifo_words_do_not_request_counter_member_pair_permissions(monkeypatch):
    original = event_planning._Planning.compatible

    def local_only(self, left, right):
        assert left.mechanism is right.mechanism is NotificationKind.LOCAL_EVENT
        return original(self, left, right)

    monkeypatch.setattr(event_planning._Planning, "compatible", local_only)
    plan = synchronize_structure(
        fifo_panels.specialize(PANELS=3), id_pools=EventIdPools(cross=(0, 3, 7, 11))
    )
    assert plan.resources.counter_groups
    certificate = next(
        leaf
        for group in plan.resources.counter_groups
        for leaf in (group.certificate.episodes or (group.certificate,))
        if len(leaf.resources) > 1
    )
    assert all(word == certificate.words[0][2] for _, _, word in certificate.words)
    a = min(certificate.resources)
    demands = list(plan.resources.occupancies)
    demands[a] = replace(demands[a], backlog=15)

    # This is an adversarial numeric premise on a real audited word, not a
    # hardware capacity increase or new DSL qualification. Reject it before
    # asking any member relation: another active credit makes the sum >15.
    def forbidden(*args, **kwargs):
        pytest.fail("an over-capacity FIFO word requested a guard proof")

    with monkeypatch.context() as patch:
        patch.setattr(plan.generation.domains.arithmetic.queries, "implies", forbidden)
        assert _certificate(plan.generation, demands, certificate.resources) is None
    for count in (0, 1, 3):
        # This fixture is one kernel invocation. The model's invocations=2
        # concatenates controller bodies without a host launch join; its GM
        # Work is intentionally not an inter-invocation feedback protocol.
        # Real nested reentry remains covered by counter_epochs regressions.
        check_compiled_events(plan, {"n": count})

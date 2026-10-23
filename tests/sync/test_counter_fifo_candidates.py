"""Whole FIFO choices survive earlier disjoint-residue candidate pairing."""

from dataclasses import replace

import pytest

from ascendflow.analysis import event_coloring
from ascendflow.analysis.event_coloring import NoEventAssignment
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.fifo_panels import fifo_feedback, fifo_panels
from tests.models.structured_events import check_compiled_events


@pytest.mark.parametrize("legal_ids", [tuple(range(16)), tuple(range(4))])
def test_nine_panel_word_uses_only_legal_ids_and_preserves_both_readers(monkeypatch, legal_ids):
    # Restrict the target pool to an existing legal subset; no proof rule,
    # candidate generation or capacity check is replaced by this regression.
    monkeypatch.setattr(event_coloring, "SUPPORTED_CROSS_IDS", legal_ids)
    plan = synchronize_structure(fifo_panels)
    assert 0 < plan.resources.work.allocated_cross_ids <= len(legal_ids)
    assert 0 < plan.resources.work.maximum_counter_backlog <= 15
    assert plan.resources.counter_groups
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count})


def test_complete_credits_retain_distinct_sequential_loop_words():
    plan = synchronize_structure(fifo_panels.specialize(EPOCHS=2))
    assert plan.resources.work.allocated_cross_ids <= 16
    assert any(certificate.episodes for _, certificate, _ in plan.resources.counter_groups)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, max_commands=10000)


def test_fifo_children_time_share_only_through_the_real_feedback(monkeypatch):
    monkeypatch.setattr(event_coloring, "SUPPORTED_CROSS_IDS", tuple(range(5)))
    plan = synchronize_structure(fifo_feedback)
    assert plan.resources.counter_compositions
    assert plan.resources.work.maximum_counter_backlog <= 15
    for count in (1, 3):
        check_compiled_events(plan, {"n": count}, max_commands=10000)


def test_absent_feedback_cannot_supply_the_same_temporal_composition(monkeypatch):
    monkeypatch.setattr(event_coloring, "SUPPORTED_CROSS_IDS", tuple(range(5)))
    source = fifo_feedback.source.replace("N.copy(Feedback[epoch - 1, :, :], a1)", "N.copy(A, a1)")
    assert source != fifo_feedback.source
    with pytest.raises(NoEventAssignment):
        synchronize_structure(replace(fifo_feedback, source=source))

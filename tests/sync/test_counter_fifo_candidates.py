"""Whole FIFO choices survive earlier disjoint-residue candidate pairing."""

import pytest

from ascendflow.analysis import event_planning
from ascendflow.analysis.event_contracts import EventIdPools
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
    assert any(certificate.episodes for _, certificate, _ in plan.resources.counter_groups)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, max_commands=10000)

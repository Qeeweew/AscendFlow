"""FIFO child composition keeps a complete real feedback consumption bridge."""

from ascendflow.analysis.event_contracts import EventIdPools
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.fifo_panels import fifo_feedback
from tests.models.structured_events import check_compiled_events


def test_fifo_children_time_share_only_through_the_real_feedback():
    plan = synchronize_structure(fifo_feedback, id_pools=EventIdPools(cross=(0, 3, 7, 11, 15)))
    assert plan.resources.counter_compositions
    assert plan.resources.work.maximum_counter_backlog <= 15
    for count in (1, 3):
        check_compiled_events(plan, {"n": count}, max_commands=10000)

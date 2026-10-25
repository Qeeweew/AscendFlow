"""Removing the cross-controller bridge cannot authorize FIFO time reuse."""

from dataclasses import replace

import pytest

from ascendflow.analysis.event_coloring import NoEventAssignment
from ascendflow.analysis.event_contracts import EventIdPools
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.fifo_panels import fifo_feedback


def test_absent_feedback_cannot_supply_the_same_temporal_composition():
    source = fifo_feedback.source.replace("N.copy(Feedback[epoch - 1, :, :], a1)", "N.copy(A, a1)")
    assert source != fifo_feedback.source
    with pytest.raises(NoEventAssignment):
        synchronize_structure(
            replace(fifo_feedback, source=source), id_pools=EventIdPools(cross=(0, 3, 7, 11, 15))
        )

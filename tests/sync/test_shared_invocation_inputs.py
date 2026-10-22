"""Shared GM inputs checked from current physical SSA and finite byte versions."""

from dataclasses import replace

import pytest

from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.state import shared_state_sum
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module")
def shared_structure():
    return synchronize_structure(shared_state_sum.specialize(B=16, GROUPS=4))


@pytest.mark.parametrize("calls,tiles", [(0, 0), (0, 7), (1, 0), (3, 0), (3, 1), (2, 7)])
def test_actual_shared_input_and_partitioned_output_bytes_in_every_group(
    shared_structure, calls, tiles
):
    check_compiled_events(
        shared_structure,
        {"outputs": calls, "tiles": tiles},
        groups=(0, 1, 2, 3),
    )


def test_all_groups_can_read_the_same_gm_without_inventing_exclusive_frames():
    source = (
        shared_state_sum.source.replace("(GROUPS, B), T.f32)", '(B,), T.f32, access="read_write")')
        .replace("(GROUPS, tiles, B)", "(tiles, B)")
        .replace("Initial[group, :]", "Initial[:]")
        .replace("X[group, t, :]", "X[t, :]")
    )
    plan = synchronize_structure(
        replace(shared_state_sum, source=source).specialize(B=16, GROUPS=4)
    )
    check_compiled_events(plan, {"outputs": 2, "tiles": 3}, groups=(0, 1, 2, 3))

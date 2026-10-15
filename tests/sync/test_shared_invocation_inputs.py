"""Unframed GM sharing needs actual read effects and every physical group bound."""

from dataclasses import replace

import pytest

from examples.state.shared_inputs import shared_state_sum
from tests.models.local_sequence import finite_check
from xdsl_npu import compile
from xdsl_npu.performance import runtime_work


@pytest.mark.parametrize("calls,tiles", [(0, 0), (0, 7), (1, 0), (3, 0), (3, 1), (2, 7)])
def test_actual_shared_input_and_partitioned_output_bytes_in_every_group(
    shared_artifact, calls, tiles
):
    plan = shared_artifact.plan
    for group in range(4):
        finite_check(plan, tiles, invocations=calls, group=group)
    assert shared_artifact.report["invocation_frames"] == 1
    work = runtime_work(plan, {"outputs": calls, "tiles": tiles})
    assert work["vector_requested_elements"]["add"] == 4 * calls * tiles * 16
    assert work["declared_gm_working_set_bytes"] == 4 * (1 + calls + tiles) * 16 * 4


def test_all_groups_can_read_the_same_gm_without_inventing_exclusive_frames():
    source = (
        shared_state_sum.source.replace("(GROUPS, B), T.f32)", '(B,), T.f32, access="read_write")')
        .replace("(GROUPS, tiles, B)", "(tiles, B)")
        .replace("Initial[group, :]", "Initial[:]")
        .replace("X[group, t, :]", "X[t, :]")
    )
    result = compile(replace(shared_state_sum, source=source).specialize(B=16, GROUPS=4))
    for group in range(4):
        finite_check(result.plan, 3, invocations=2, group=group)
    assert result.report["invocation_frames"] == 1
    work = runtime_work(result.plan, {"outputs": 2, "tiles": 3})
    assert work["declared_gm_working_set_bytes"] == (1 + 3 + 4 * 2) * 16 * 4

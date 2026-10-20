"""Qualify seeds, actual rounds, empty prefixes and final credit reclaim."""

import pytest

from ascendflow.analysis.structured_memory import collect_native_memory
from tests.models.structured_events import EventModelError, check_structured_events
from tests.support.credit_primitives import seeded_credit_probe


@pytest.fixture(scope="module", params=(False, True), ids=("binary", "two-aiv"))
def seeded(request):
    return seeded_credit_probe(cross=request.param)


@pytest.mark.parametrize("n", (0, 1, 2, 3, 6, 10, 17))
def test_credit_survives_unvisited_slots_and_consecutive_empty_calls(seeded, n):
    plan, original = seeded
    check_structured_events(
        collect_native_memory(plan.module), {"n": n}, logical_reference=original, invocations=2
    )


@pytest.mark.parametrize("cross", (False, True), ids=("binary", "two-aiv"))
@pytest.mark.parametrize("remove", ("seed", "acquire", "ready", "read", "return", "reclaim"))
def test_persistent_rule_needs_every_real_instruction(cross, remove):
    plan, original = seeded_credit_probe(cross=cross, remove=remove)
    with pytest.raises(EventModelError):
        check_structured_events(
            collect_native_memory(plan.module), {"n": 10}, logical_reference=original, invocations=2
        )


def test_empty_mode_two_initial_credit_needs_both_aiv_contributions():
    plan, original = seeded_credit_probe(cross=True, missing_lane=True)
    with pytest.raises(EventModelError, match="mode-2 quorum"):
        check_structured_events(
            collect_native_memory(plan.module), {"n": 0}, logical_reference=original, invocations=2
        )


def test_local_final_reclaim_must_precede_the_destination_queue_drain():
    plan, original = seeded_credit_probe(late_reclaim=True)
    with pytest.raises(EventModelError, match="binary event overwritten"):
        check_structured_events(
            collect_native_memory(plan.module), {"n": 0}, logical_reference=original, invocations=2
        )

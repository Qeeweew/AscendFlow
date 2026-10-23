"""Mode-2 time reuse retains complete consumption and existing causality."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import color_event_resources, counter_group_usage
from ascendflow.analysis.event_contracts import EventDemand
from ascendflow.analysis.event_sharing import CounterReuse, share_counter_time
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.counter_epochs import counter_epochs
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module", params=[2, 17])
def plan(request):
    return request.param, synchronize_structure(counter_epochs.specialize(PHASES=request.param))


def test_complete_epochs_keep_empty_calls_reentry_and_two_vector_receivers(plan):
    phases, program = plan
    if phases == 2:
        assert not program.resources.counter_reuse_groups
        assert program.resources.work.counter_time_pairs == 0
    else:
        assert program.resources.counter_reuse_groups
        assert any(isinstance(w, CounterReuse) for w in program.resources.sharing)
    for group in program.resources.counter_reuse_groups:
        assert 0 <= group.event < 16
        assert 0 < group.backlog <= 15
        for witness in group.witnesses:
            if witness.handbacks:
                assert {h.consume.actor for h in witness.handbacks} == {
                    p.actor for p in witness.earlier.consumers
                }
                assert {h.next_set.actor for h in witness.handbacks} == {
                    p.actor for p in witness.later.sources
                }
    for count in (0, 1, 2, 3):
        execution = check_compiled_events(program, {"n": count}, invocations=2, max_commands=20000)
        assert count == 0 or execution.bytes_read and execution.bytes_written


def forward_messages(program):
    return [
        n
        for n in program.selection.notifications
        if n.mechanism is NotificationKind.CROSS_EVENT
        and n.source_engine is Engine.FIX
        and n.sources[0].actor is Actor.AIC
    ]


def test_a_required_return_is_needed_even_when_publisher_and_receiver_words_agree():
    source = counter_epochs.source.replace(
        "N.copy(Feedback[phase - 1, :, :], a1)", "N.copy(A, a1)"
    ).replace("N.copy(Feedback[PHASES - 1, :, :], a1)", "N.copy(A, a1)")
    assert source != counter_epochs.source
    for definition, expected in (
        (counter_epochs, True),
        (replace(counter_epochs, source=source), False),
    ):
        program = synchronize_structure(definition.specialize(PHASES=2)).generation
        left, right = forward_messages(program)
        witness = share_counter_time(
            program.domains,
            HandbackRule(program.domains, boundaries=program.boundaries),
            left,
            right,
        )
        assert (witness is not None) == expected
        if witness is not None:
            assert {h.consume.actor for h in witness.handbacks} == {Actor.AIV0, Actor.AIV1}


def test_temporal_complete_credits_use_maximum_instead_of_adding_full_counters():
    demands = tuple(EventDemand((), 1, (), backlog=15) for _ in range(17))
    reuses = {pair: () for pair in combinations(range(17), 2)}
    problem = SimpleNamespace(
        demands=demands,
        reservations=(),
        peers=(),
        counter_groups=(),
        relations={},
        counter_reuses=reuses,
    )
    ids, _ = color_event_resources(problem)
    assert len({group[0] for group in ids}) <= 16
    shared = {i for i, group in enumerate(ids) if sum(group == other for other in ids) > 1}
    assert shared
    assert counter_group_usage(demands, shared, reuses, ()) == ("time", 15)


def test_counter_set_requires_one_complete_temporal_or_fifo_premise():
    demands = tuple(EventDemand((), 1, (), backlog=5) for _ in range(3))
    partial = {(0, 1): (), (1, 2): ()}
    assert counter_group_usage(demands, {0, 1, 2}, partial, (frozenset({0, 2}),)) is None
    assert counter_group_usage(demands, {0, 1, 2}, partial, (frozenset({0, 1, 2}),)) == ("fifo", 15)


def test_frozen_fifo_children_compose_with_a_maximum_only_after_time_proof():
    demands = tuple(EventDemand((), 1, (), backlog=7) for _ in range(4))
    assert counter_group_usage(demands, {0, 1, 2, 3}, {}, ()) is None
    groups = ((frozenset({0, 1}), frozenset({2, 3})),)
    assert counter_group_usage(demands, {0, 1, 2, 3}, {}, (), time_packs=groups) == (
        "time_fifo",
        14,
    )

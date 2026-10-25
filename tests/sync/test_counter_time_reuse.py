"""Mode-2 time reuse retains complete consumption and existing causality."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import color_event_resources, counter_group_usage
from ascendflow.analysis.event_contracts import CounterReuseGroup, EventDemand, EventIdPools
from ascendflow.analysis.event_frontier_contracts import CreditSeparation, FrozenEventRelations
from ascendflow.analysis.event_sharing import share_counter_time
from ascendflow.analysis.resource_report import event_resource_report
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.counter_epochs import counter_epochs
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module", params=[2, 17])
def plan(request):
    # The large case must need temporal reuse after certified FIFO words have
    # been packed. A legal subset makes that need explicit as policy changes.
    # Capture-epoch completion coverage reduces the unshared plan to four
    # counters. Three keeps this a real temporal-reuse test.
    pools = EventIdPools(cross=tuple(range(3 if request.param == 17 else 16)))
    return request.param, synchronize_structure(
        counter_epochs.specialize(PHASES=request.param), id_pools=pools
    )


def test_complete_epochs_keep_empty_calls_reentry_and_two_vector_receivers(plan):
    phases, program = plan
    if phases == 2:
        assert not program.resources.counter_reuse_groups
        assert program.resources.work.counter_time_pairs == 0
    else:
        assert bool(program.resources.counter_reuse_groups), "pressure must require temporal reuse"
        assert any(isinstance(w, CounterReuseGroup) for w in program.resources.sharing)
    for group in program.resources.counter_reuse_groups:
        assert 0 <= group.event < 16
        assert 0 < group.backlog <= 15
        for rectangle in group.rectangles:
            for interface in rectangle.certificate.interfaces:
                if interface.original is not None:
                    obligations = ((w.earlier, w.later, w.handbacks) for w in interface.original)
                elif interface.handbacks:
                    earlier, later = (
                        (interface.left.tail, interface.right.head)
                        if interface.forward
                        else (interface.right.tail, interface.left.head)
                    )
                    obligations = ((earlier, later, interface.handbacks),)
                else:
                    obligations = ()
                for earlier, later, handbacks in obligations:
                    if handbacks:
                        assert {h.consume.actor for h in handbacks} == {
                            p.actor for p in earlier.consumers
                        }
                        assert {h.next_set.actor for h in handbacks} == {
                            p.actor for p in later.sources
                        }
    report = event_resource_report(program.resources)
    assert bool(report["counter_time_reuse"]) == (phases == 17)
    for group in report["counter_time_reuse"]:
        assert group["rectangles"]
        for rectangle in group["rectangles"]:
            assert report["event_certificates"][rectangle["certificate"]]["interfaces"]
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
            program.handbacks,
            left,
            right,
        )
        assert (witness is not None) == expected
        if witness is not None:
            assert {h.consume.actor for h in witness.handbacks} == {Actor.AIV0, Actor.AIV1}


def test_temporal_complete_credits_use_maximum_instead_of_adding_full_counters():
    demands = tuple(EventDemand((), 1, (), backlog=15) for _ in range(17))
    reuses = numeric_reuses(17, combinations(range(17), 2))
    problem = SimpleNamespace(
        demands=demands,
        reservations=(),
        peers=(),
        counter_groups=(),
        relations={},
        counter_reuses=reuses,
        id_pools=EventIdPools(),
    )
    ids, _ = color_event_resources(problem)
    assert len({group[0] for group in ids}) <= 16
    shared = {i for i, group in enumerate(ids) if sum(group == other for other in ids) > 1}
    assert shared
    assert counter_group_usage(demands, shared, reuses, ()) == ("time", 15)


def test_counter_set_requires_one_complete_temporal_or_fifo_premise():
    demands = tuple(EventDemand((), 1, (), backlog=5) for _ in range(3))
    partial = numeric_reuses(3, ((0, 1), (1, 2)))
    assert counter_group_usage(demands, {0, 1, 2}, partial, (frozenset({0, 2}),)) is None
    assert counter_group_usage(demands, {0, 1, 2}, partial, (frozenset({0, 1, 2}),)) == ("fifo", 15)


def test_frozen_fifo_children_compose_with_a_maximum_only_after_time_proof():
    demands = tuple(EventDemand((), 1, (), backlog=7) for _ in range(4))
    empty = FrozenEventRelations(())
    assert counter_group_usage(demands, {0, 1, 2, 3}, empty, ()) is None
    groups = ((frozenset({0, 1}), frozenset({2, 3})),)
    assert counter_group_usage(demands, {0, 1, 2, 3}, empty, (), time_packs=groups) == (
        "time_fifo",
        14,
    )


def numeric_reuses(size, pairs):
    """Explicit numeric constraints only; these do not grant a DSL handback."""
    masks = [0] * size
    for a, b in pairs:
        masks[a] |= 1 << b
        masks[b] |= 1 << a
    return FrozenEventRelations(
        ((mask, CreditSeparation((), 1 << r, mask)),) if mask else ()
        for r, mask in enumerate(masks)
    )

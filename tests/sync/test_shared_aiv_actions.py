"""Participant sharing preserves instruction order, multiplicity and eligibility."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow.analysis.structured_resources import _combine_actions
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.scalar import Expr, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind


@pytest.fixture
def anchor():
    # Sharing only reads insertion identity; no native, event or safety premise
    # is inferred from this standalone operation.
    return arith.ConstantOp.from_int_and_width(0, IndexType())


def action(anchor, actor, event, *, label):
    return SyncAction(
        anchor,
        Placement.BEFORE,
        (actor,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.MTE1 if actor is Actor.AIC else Engine.V,
        literal(event),
        protocols=(label,),
    )


def trace(actions, actor, **values):
    return [
        (a.anchor, a.placement, a.kind, a.source, a.destination, a.event.evaluate(values))
        for a in actions
        if actor in a.participants and a.guard.evaluate(values)
    ]


def test_same_instructions_with_distinct_origins_share_without_losing_repetitions(anchor):
    original = tuple(
        action(anchor, actor, event, label=f"{actor.value}_{i}")
        for actor in (Actor.AIV0, Actor.AIV1)
        for i, event in enumerate((3, 3, 1))
    )
    shared = _combine_actions(original)
    assert len(shared) == 3
    assert all(a.participants == (Actor.AIV0, Actor.AIV1) for a in shared)
    assert shared[0].protocols == ("aiv0_0", "aiv1_0")
    assert shared[1].protocols == ("aiv0_1", "aiv1_1")
    for actor in (Actor.AIV0, Actor.AIV1):
        assert [item[-1] for item in trace(shared, actor)] == [3, 3, 1]
        assert trace(shared, actor) == trace(original, actor)


def test_opposite_orders_cannot_be_merged_by_an_unordered_instruction_key(anchor):
    original = tuple(
        action(anchor, actor, event, label="same diagnostic name")
        for actor, events in ((Actor.AIV0, (0, 1)), (Actor.AIV1, (1, 0)))
        for event in events
    )
    shared = _combine_actions(original)
    assert len(shared) == 4
    assert [item[-1] for item in trace(shared, Actor.AIV0)] == [0, 1]
    assert [item[-1] for item in trace(shared, Actor.AIV1)] == [1, 0]


@pytest.mark.parametrize("difference", ("id", "guard", "direction", "kind"))
def test_real_participant_differences_keep_their_own_execution(anchor, difference):
    left = action(anchor, Actor.AIV0, 0, label="left")
    right = action(anchor, Actor.AIV1, 0, label="right")
    if difference == "id":
        right = replace(right, event=literal(1))
    elif difference == "guard":
        right = replace(
            right, guard=ActionGuard((Comparison(Expr("argument", ("n",)), ">", literal(0)),))
        )
    elif difference == "direction":
        right = replace(right, source=Engine.V, destination=Engine.MTE2)
    else:
        right = replace(right, kind=SyncKind.LOCAL_SET)
    original = (left, right)
    shared = _combine_actions(original)
    assert len(shared) == 2 and all(len(a.participants) == 1 for a in shared)
    for n in (0, 1):
        for actor in (Actor.AIV0, Actor.AIV1):
            assert trace(shared, actor, n=n) == trace(original, actor, n=n)


@pytest.mark.parametrize("boundary", ("anchor", "placement"))
def test_equal_instructions_at_different_sites_cannot_be_combined(anchor, boundary):
    left = action(anchor, Actor.AIV0, 0, label="left")
    right = action(anchor, Actor.AIV1, 0, label="right")
    right = replace(
        right,
        **(
            {"anchor": arith.ConstantOp.from_int_and_width(1, IndexType())}
            if boundary == "anchor"
            else {"placement": Placement.AFTER}
        ),
    )
    shared = _combine_actions((left, right))
    assert len(shared) == 2 and all(len(a.participants) == 1 for a in shared)


def test_an_already_shared_action_keeps_its_position_in_both_actor_sequences(anchor):
    shared = replace(
        action(anchor, Actor.AIV0, 2, label="both"), participants=(Actor.AIV0, Actor.AIV1)
    )
    original = (
        action(anchor, Actor.AIV0, 0, label="first-left"),
        action(anchor, Actor.AIV1, 0, label="first-right"),
        shared,
        action(anchor, Actor.AIV0, 1, label="last-left"),
        action(anchor, Actor.AIV1, 1, label="last-right"),
    )
    combined = _combine_actions(original)
    assert len(combined) == 3
    for actor in (Actor.AIV0, Actor.AIV1):
        assert [item[-1] for item in trace(combined, actor)] == [0, 2, 1]


def test_equal_actions_on_one_actor_are_never_deduplicated(anchor):
    original = (action(anchor, Actor.AIC, 0, label="cube"),) * 2
    assert _combine_actions(original) == original
    vector = (action(anchor, Actor.AIV0, 0, label="vector"),) * 2
    assert _combine_actions(vector) == vector

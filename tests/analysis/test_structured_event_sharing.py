"""Actual finite channel sharing through existing queue return paths."""

from dataclasses import replace
from functools import partial

import pytest

from examples.vector.scale import scale
from tests.models.structured_domains import actual_dispatch
from tests.support.structured import prepared, selection
from xdsl_npu.analysis.event_lifetimes import bridge
from xdsl_npu.analysis.structured_event_sharing import (
    EventSharingContext,
    prove_finite_event_sharing,
)
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.scalar import literal
from xdsl_npu.target.ascend910b import Actor, Engine


@pytest.fixture(scope="module")
def finite():
    source = [
        "def many(X:N.Tensor((9,64),N.f32)):",
        "    with N.Kernel():",
        "        x=N.alloc_ub((64,))",
    ]
    for index in range(9):
        source.extend(
            (
                f"        N.copy(X[{index},:],x[:])",
                "        N.mul(x[:],2.0,out=x[:])",
            )
        )
    return selection("\n".join(source))


def sharing_context(selected):
    # These tests isolate channel reuse. Removing a return edge deliberately
    # invalidates memory coverage, which is tested at the selection boundary.
    selected = replace(
        selected, dependencies=replace(selected.dependencies, required=()), coverage=()
    )
    allocator = prepared(selected)
    return EventSharingContext(
        selected, allocator.windows, allocator._instance, partial(bridge, allocator)
    )


def route(selected, source):
    return tuple(n for n in selected.notifications if n.source_engine is source)


def assert_actual_return_path(witness, selected):
    positions = {
        key: index
        for index, (key, _) in enumerate(
            actual_dispatch(selected, {"$group": 0, "$sid": 0}, witness.bank[0])
        )
    }

    def position(point):
        return positions[
            point.operation,
            point.actor,
            tuple((loop, value.evaluate({})) for loop, value in point.coordinates),
        ]

    current = witness.earlier
    assert witness.steps and witness.steps[-1].notification is None
    for step in witness.steps:
        following = witness.later if step.notification is None else step.notification
        assert not step.controller
        assert current.destination_engine is following.source_engine
        assert step.consume.actor is step.source.actor
        assert position(step.consume) <= position(step.source)
        if step.notification is not None:
            current = step.notification


@pytest.mark.parametrize("source", [Engine.MTE2, Engine.V])
def test_nine_finite_pairs_have_actual_per_bank_sharing_paths(finite, source):
    channels = route(finite, source)
    assert len(channels) == (9 if source is Engine.MTE2 else 8)
    context = sharing_context(finite)
    for left, right in ((channels[0], channels[1]), (channels[0], channels[-1])):
        witness = prove_finite_event_sharing(left, right, context=context)
        assert witness is not None
        assert witness.earlier is left and witness.later is right
        assert witness.bank == (Actor.AIV0, source, left.destination_engine)
        assert_actual_return_path(witness, finite)


def test_assignment_order_does_not_change_actual_sharing_order(finite):
    first, *_, last = route(finite, Engine.MTE2)
    witness = prove_finite_event_sharing(last, first, context=sharing_context(finite))
    assert witness is not None
    assert witness.earlier is first and witness.later is last
    assert_actual_return_path(witness, finite)


def test_missing_return_notifications_do_not_become_controller_gates(finite):
    forwards = route(finite, Engine.MTE2)
    remaining = replace(finite, notifications=forwards)
    assert (
        prove_finite_event_sharing(forwards[0], forwards[1], context=sharing_context(remaining))
        is None
    )


def test_return_path_must_exist_in_the_actual_coexisting_domain(finite):
    inactive = ActionGuard((Comparison(literal(0), "==", literal(1)),))
    changed = replace(
        finite,
        notifications=tuple(
            replace(
                n,
                sources=tuple(replace(p, guard=inactive) for p in n.sources),
                consumers=tuple(replace(p, guard=inactive) for p in n.consumers),
            )
            if n.source_engine is Engine.V
            else n
            for n in finite.notifications
        ),
    )
    forwards = route(changed, Engine.MTE2)
    assert (
        prove_finite_event_sharing(forwards[0], forwards[1], context=sharing_context(changed))
        is None
    )


def test_sharing_proof_is_bound_to_one_actual_directed_bank(finite):
    forward = route(finite, Engine.MTE2)[0]
    reverse = route(finite, Engine.V)[0]
    assert prove_finite_event_sharing(forward, reverse, context=sharing_context(finite)) is None


def test_empty_common_windows_do_not_erase_repeated_actual_scopes():
    selected = selection(scale)
    forward = route(selected, Engine.MTE2)[0]
    context = sharing_context(selected)
    clone = replace(forward, name="another_repeated_channel")

    def forbidden_bridge(*args):
        pytest.fail("a repeated invocation reached the finite sharing bridge")

    context = replace(
        context,
        windows={n: () for n in (*selected.notifications, clone)},
        bridge=forbidden_bridge,
    )
    assert prove_finite_event_sharing(forward, clone, context=context) is None


def test_allocator_reuses_finite_ids_and_retains_each_sharing_witness(finite):
    from xdsl_npu.analysis.structured_resources import allocate_structured_events

    result = allocate_structured_events(finite)
    assert len(result.assignments) == 17
    for source in (Engine.MTE2, Engine.V):
        assignments = [a for a in result.assignments if a.notification.source_engine is source]
        assert {a.ids for a in assignments} == {(0,)}
    assert result.sharing
    for witness in result.sharing:
        assert_actual_return_path(witness, finite)

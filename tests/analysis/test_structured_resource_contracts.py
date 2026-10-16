"""Adversarial actual-domain checks of the resource stage's output contract."""

from dataclasses import replace

import pytest

from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy
from tests.models.structured_domains import actual_dispatch, need_cases
from tests.support.structured import selected
from xdsl_npu.analysis.access_frontier import AccessEndpoint
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding
from xdsl_npu.analysis.structured_resources import allocate_structured_events
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind

POINT = """
def point(X:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
        value=N.copy(X[:],x[:])
        value=N.mul(value,2.0,out=y[:])
        N.copy(value,O[:])
"""

FINITE = """
def finite(X:N.Tensor((3,64),N.f32),O:N.Tensor((3,64),N.f32,access="write")):
    with N.Kernel():
        x,y=N.alloc_ub((3,64)),N.alloc_ub((3,64))
        for j in N.Pipelined(3,stages=3):
            with N.Stage(0,name="load"):
                value=N.copy(X[j,:],x[j,:])
            with N.Stage(1,name="compute"):
                value=N.mul(value,2.0,out=y[j,:])
            with N.Stage(2,name="store"):
                N.copy(value,O[j,:])
"""


def replaced_notification(selection, original, changed):
    return replace(
        selection,
        notifications=tuple(changed if n is original else n for n in selection.notifications),
        coverage=tuple(
            replace(item, notification=changed) if item.notification is original else item
            for item in selection.coverage
        ),
    )


def test_single_use_requires_a_wait_for_every_set():
    selection = selected(POINT)
    original = selection.notifications[0]
    changed = replace(
        original,
        consumers=tuple(
            replace(p, guard=ActionGuard((Comparison(literal(0), "==", literal(1)),)))
            for p in original.consumers
        ),
    )
    with pytest.raises(CompileError):
        allocate_structured_events(replaced_notification(selection, original, changed))


def test_valid_first_and_last_ports_do_not_hide_a_missing_middle_wait():
    selection = selected(FINITE)
    ordinal = next(iter(selection.dependencies.memory.control.loops.values())).ordinal
    original = selection.notifications[0]
    changed = replace(
        original,
        consumers=tuple(
            replace(
                p,
                guard=ActionGuard(p.guard.comparisons + (Comparison(ordinal, "!=", literal(1)),)),
            )
            for p in original.consumers
        ),
    )
    with pytest.raises(CompileError):
        allocate_structured_events(replaced_notification(selection, original, changed))


def test_event_binding_must_preserve_the_same_generation_at_both_ends():
    selection = selected(FINITE)
    ordinal = next(iter(selection.dependencies.memory.control.loops.values())).ordinal
    original = selection.notifications[0]
    changed = replace(
        original,
        consumers=tuple(
            replace(
                p,
                substitutions=((ordinal, expression("add", ordinal, literal(1))),),
            )
            for p in original.consumers
        ),
    )
    with pytest.raises(CompileError):
        allocate_structured_events(replaced_notification(selection, original, changed))


def finite_ports(*, at_most_one=False):
    selection = selected("""
    def anchors(X:N.Tensor((n,64),N.f32),n:N.int64):
        with N.Kernel():
            a,b=N.alloc_ub((2,64)),N.alloc_ub((2,64))
            for j in N.Pipelined(n,stages=1):
                N.fill(a[j%2,:],1.0)
                N.fill(b[j%2,:],1.0)
                N.copy(X[j,:],a[j%2,:])
    """)
    memory = selection.dependencies.memory
    logical = next(iter(memory.control.loops.values()))
    loop, ordinal, count = logical.operation, logical.ordinal, logical.count
    first_vector, later_vector, load = memory.templates

    def port(template, *, first, source):
        coordinate = literal(0) if first else expression("sub", count, literal(1))
        nonempty = Comparison(count, ">", literal(0))
        activation = (nonempty,)
        if at_most_one:
            activation += (Comparison(count, "<=", literal(1)),)
        occurrence = AccessOccurrence(
            AccessEndpoint.from_template(template, Actor.AIV0),
            (LoopBinding(loop, coordinate),),
        )
        return BoundPlacement(
            occurrence=occurrence,
            placement=Placement.AFTER if source else Placement.BEFORE,
            guard=ActionGuard(activation + (Comparison(ordinal, "==", coordinate),)),
            substitutions=((ordinal, coordinate),),
            coordinate_guards=((loop, ActionGuard((nonempty,))),),
            families=(),
            coordinates=((loop, coordinate),),
        )

    notifications = (
        LogicalNotification(
            "last_to_load",
            (port(first_vector, first=False, source=True),),
            (port(load, first=True, source=False),),
            Engine.V,
            Engine.MTE2,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        ),
        LogicalNotification(
            "load_to_earlier_vector",
            (port(load, first=True, source=True),),
            (port(later_vector, first=True, source=False),),
            Engine.MTE2,
            Engine.V,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        ),
    )
    # Isolate the resource contract from memory coverage. Every endpoint still
    # names an actual audited native and its exact retained invocation.
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def actual_port_cycle(selection, count):
    uniform = {"n": count, "$group": 0, "$sid": 0}
    positions = {
        key: position
        for position, (key, _) in enumerate(actual_dispatch(selection, uniform, Actor.AIV0))
    }

    def position(point):
        coordinates = tuple((loop, value.evaluate(uniform)) for loop, value in point.coordinates)
        return positions[point.operation, point.actor, coordinates]

    first, second = selection.notifications
    # Effect1 -> Wait1 -> Effect2 -> Wait2 are selected matching and queue
    # edges. The remaining actual queue edge closes the cycle only for n>1.
    assert first.destination_engine is second.source_engine
    assert second.destination_engine is first.source_engine
    return position(first.consumers[0]) <= position(second.sources[0]) and position(
        second.consumers[0]
    ) <= position(first.sources[0])


def test_parameter_dependent_blocking_order_cannot_be_dropped_from_progress():
    selection = finite_ports()
    assert not actual_port_cycle(selection, 1)
    assert actual_port_cycle(selection, 2)
    with pytest.raises(CompileError):
        allocate_structured_events(selection)


def test_restricting_the_finite_ports_to_one_iteration_removes_the_cycle():
    selection = finite_ports(at_most_one=True)
    assert not actual_port_cycle(selection, 1)
    plan = allocate_structured_events(selection)
    assert plan.progress.points


@pytest.mark.parametrize("source", [scale, nested_copy])
def test_emitted_ring_ids_pair_each_actual_generation_across_clocks(source):
    selection = selected(source)
    plan = allocate_structured_events(selection)
    memory = selection.dependencies.memory
    actors = {actor for _, actor in selection.dispatch.tasks}
    for count in (0, 1, 2, 3):
        uniform = {"tiles": count, "n": count, "$group": 0, "$sid": 0}
        actual = {key for actor in actors for key, _ in actual_dispatch(selection, uniform, actor)}
        for assignment in plan.assignments:
            notification = assignment.notification
            assert notification.mechanism is NotificationKind.LOCAL_EVENT
            for need in notification.requirements:
                for context in need_cases(need, uniform, memory):
                    (case,) = tuple(
                        case for case in assignment.cases if case.case.condition.evaluate(context)
                    )
                    expected = assignment.ids[case.ordinal.evaluate(context) % case.width]
                    for points, kind in (
                        (notification.sources, SyncKind.LOCAL_SET),
                        (notification.consumers, SyncKind.LOCAL_WAIT),
                    ):
                        for point in points:
                            coordinates = tuple(
                                (loop, value.evaluate(context)) for loop, value in point.coordinates
                            )
                            assert (point.operation, point.actor, coordinates) in actual
                            runtime = context | {
                                memory.control.loops[loop].ordinal.args[0]: value
                                for loop, value in coordinates
                            }
                            events = [
                                event
                                for action in plan.actions
                                if action.protocol == notification.name
                                and action.kind is kind
                                and action.anchor is point.operation
                                and point.actor in action.participants
                                and (event := action.active_event(runtime)) is not None
                            ]
                            assert events == [expected]

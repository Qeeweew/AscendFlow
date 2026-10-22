"""Conditional Wait-admission to native-after-Set order on real dispatches.

The finite oracle interprets retained IR, including stages and issue order.
Empty notification selections isolate queue order; they claim no memory
coverage, event pairing, native completion or lifetime proof.
"""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_frontier import AccessEndpoint
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventCase, EventDomains
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from ascendflow.analysis.structured_regions import AccessOccurrence, LoopBinding
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr, expression, literal
from ascendflow.target.ascend910b import Actor
from tests.models.structured_domains import actual_dispatch


def domains_for(source):
    memory = collect_native_memory(lower_native_structure(N.from_source(source)).module)
    dependencies = replace(build_structured_dependencies(memory), required=())
    selection = SynchronizationSelection(dependencies, bind_structured_dispatch(memory), (), ())
    return EventDomains(selection, ResourceWork())


def point(domains, operation, coordinates=(), *, actor=Actor.AIV0, phase=Placement.BEFORE):
    task = domains.dispatch.tasks[operation, actor]
    return BoundPlacement(
        AccessOccurrence(
            AccessEndpoint.from_template(task.template, actor),
            tuple(LoopBinding(loop, value) for loop, value in coordinates),
        ),
        phase,
        ActionGuard(),
        (),
        (),
        (),
        coordinates,
    )


def notification(domains, source, consumer):
    """Route metadata comes from both actual native endpoint engines."""
    return LogicalNotification(
        "queue_order_test",
        (source,),
        (consumer,),
        domains.dispatch.tasks[source.operation, source.actor].engine,
        domains.dispatch.tasks[consumer.operation, consumer.actor].engine,
        NotificationKind.LOCAL_EVENT,
        (),
        None,
        literal(0),
        1,
        0,
    )


def assert_actual_order(domains, uniform, actor):
    events = actual_dispatch(domains.selection, uniform, actor)
    following = {
        loop: Expr("argument", (f"$following_{index}",))
        for index, loop in enumerate(domains.control.loops)
    }
    points = {}

    def symbolic(key, after):
        operation, _, actual = key
        cache_key = operation, after
        if cache_key not in points:
            coordinates = tuple(
                (loop, following[loop] if after else domains.control.loops[loop].ordinal)
                for loop, _ in actual
            )
            points[cache_key] = point(
                domains,
                operation,
                coordinates,
                actor=actor,
                phase=Placement.AFTER if after else Placement.BEFORE,
            )
        return points[cache_key]

    for before_index, (before_key, context) in enumerate(events):
        before = symbolic(before_key, False)
        for after_index, (after_key, _) in enumerate(events):
            after = symbolic(after_key, True)
            values = context | {following[loop].args[0]: value for loop, value in after_key[2]}
            conditions = domains._order_conditions(before, after)
            # Admission precedes its own native; all other native positions
            # follow the retained interpreter's total controller issue order.
            assert sum(condition.evaluate(values) for condition in conditions) == int(
                before_index <= after_index
            ), (before_key, after_key, conditions)


@pytest.fixture(scope="module")
def flat():
    return domains_for("""
    def flat(X:N.Tensor((64,),N.f32),n:N.i64):
        with N.Kernel(kind="mixed"):
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            with N.Vector():
                N.copy(X[:],x[:])
                for i in N.Pipelined(n,issue_order=("second","first")):
                    with N.Stage(0,name="first"):
                        N.fill(x,1.0)
                    with N.Stage(1,name="second"):
                        N.mul(x,2.0,out=y)
                N.copy(X[:],y[:])
""")


@pytest.mark.parametrize("actor", (Actor.AIV0, Actor.AIV1))
@pytest.mark.parametrize("count", (0, 1, 3))
def test_staged_order_matches_retained_issue_order(flat, actor, count):
    assert_actual_order(flat, {"n": count, "$group": 0}, actor)


def test_equal_clock_uses_wait_admission_and_reversed_stage_rank(flat):
    _, first, second, _ = (template.operation for template in flat.memory.templates)
    (loop,) = flat.control.loops.values()
    same = ((loop.operation, loop.ordinal),)
    prior = ((loop.operation, expression("sub", loop.ordinal, literal(1))),)
    consume = point(flat, first, same)
    source = point(flat, first, same, phase=Placement.AFTER)
    assert flat._order_conditions(consume, source) == (ActionGuard(),)

    # first(g) and second(g-1) share a clock. The explicit issue order puts
    # second first, including its native position before first's admission.
    source = point(flat, second, prior, phase=Placement.AFTER)
    assert not flat._order_conditions(consume, source)
    consume, source = (
        replace(source, placement=Placement.BEFORE),
        replace(consume, placement=Placement.AFTER),
    )
    assert flat._order_conditions(consume, source) == (ActionGuard(),)


def test_last_selected_member_keeps_its_residue_order_condition(flat):
    root, first, _, _ = (template.operation for template in flat.memory.templates)
    (loop,) = flat.control.loops.values()
    last = expression("sub", loop.count, literal(1))
    residue = expression("rem", last, literal(2))
    selected = expression("sub", last, residue)
    consume = point(flat, first, ((loop.operation, last),))
    source = point(flat, first, ((loop.operation, selected),), phase=Placement.AFTER)
    incoming = notification(flat, point(flat, root, phase=Placement.AFTER), consume)
    outgoing = notification(flat, source, point(flat, root))
    conditions = flat._following_conditions(incoming, consume, outgoing, source)
    assert conditions
    for count in range(1, 7):
        values = {"n": count, "$group": 0}
        events = actual_dispatch(flat.selection, values, Actor.AIV0)
        positions = {key: index for index, (key, _) in enumerate(events)}
        before_key = first, Actor.AIV0, ((loop.operation, last.evaluate(values)),)
        after_key = first, Actor.AIV0, ((loop.operation, selected.evaluate(values)),)
        expected = positions[before_key] <= positions[after_key]
        assert expected == ((count - 1) % 2 == 0)
        assert any(condition.evaluate(values) for condition in conditions) == expected


@pytest.mark.parametrize("actor", (Actor.AIV0, Actor.AIV1))
@pytest.mark.parametrize("count", (0, 1, 2))
def test_nested_sibling_and_direct_parent_order_matches_retained_ir(actor, count):
    domains = domains_for("""
    def nested(X:N.Tensor((64,),N.f32),n:N.i64):
        with N.Kernel(kind="mixed"):
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            with N.Vector():
                N.copy(X[:],x[:])
                for p in N.Pipelined(n,issue_order=("later","early")):
                    with N.Stage(0,name="early"):
                        N.fill(x,1.0)
                        for i in N.Pipelined(2,issue_order=("read","write")):
                            with N.Stage(0,name="write"):
                                N.fill(x,2.0)
                            with N.Stage(1,name="read"):
                                N.mul(x,2.0,out=y)
                    with N.Stage(1,name="later"):
                        for j in N.serial(2):
                            N.mul(x,3.0,out=y)
                        N.fill(y,4.0)
                        for k in N.Pipelined(2):
                            N.fill(x,5.0)
""")
    assert_actual_order(domains, {"n": count, "$group": 0}, actor)


def test_nested_conditions_grow_with_depth_without_expanding_iterations():
    depth = 4
    lines = [
        "def deep():",
        "    with N.Kernel():",
        "        x,y=N.alloc_ub((64,)),N.alloc_ub((64,))",
    ]
    for index in range(depth):
        lines.append("    " * (index + 2) + f"for i{index} in N.Pipelined(2):")
    lines.extend(
        "    " * (depth + 2) + operation for operation in ("N.fill(x,1.0)", "N.mul(x,2.0,out=y)")
    )
    domains = domains_for("\n".join(lines))
    before_coordinates = tuple(
        (loop.operation, loop.ordinal) for loop in domains.control.loops.values()
    )
    after_coordinates = tuple(
        (loop, Expr("argument", (f"$following_{index}",)))
        for index, (loop, _) in enumerate(before_coordinates)
    )
    first, second = (template.operation for template in domains.memory.templates)
    conditions = domains._order_conditions(
        point(domains, first, before_coordinates),
        point(domains, second, after_coordinates, phase=Placement.AFTER),
    )
    assert len(conditions) == depth
    assert max(len(condition.comparisons) for condition in conditions) == depth
    assert_actual_order(domains, {"$group": 0}, Actor.AIV0)


def test_routes_actors_and_inactive_coordinates_never_invent_order(flat):
    root, first, _, tail = (template.operation for template in flat.memory.templates)
    (loop,) = flat.control.loops.values()
    consume = point(flat, first, ((loop.operation, loop.ordinal),))
    source = point(flat, tail, phase=Placement.AFTER)
    incoming = notification(flat, point(flat, root, phase=Placement.AFTER), consume)
    tail_notification = notification(flat, source, consume)
    assert flat._order_conditions(consume, source) == (ActionGuard(),)
    assert not flat._following_conditions(incoming, consume, tail_notification, source)

    source = point(flat, first, consume.coordinates, phase=Placement.AFTER)
    outgoing = notification(flat, source, point(flat, tail))
    other_actor = point(flat, first, source.coordinates, actor=Actor.AIV1, phase=Placement.AFTER)
    assert not flat._following_conditions(incoming, consume, outgoing, other_actor)
    assert not flat._order_conditions(replace(consume, coordinates=()), source)
    inactive = replace(consume, guard=ActionGuard((Comparison(literal(0), ">", literal(0)),)))
    assert not flat._order_conditions(inactive, source)

    empty = EventCase(ActionGuard(), ((loop.count, literal(0)),), (), ())
    inactive = flat._instance(consume, {loop.ordinal: literal(0)}, case=empty)
    assert not inactive.coordinates
    assert not flat._order_conditions(inactive, source)

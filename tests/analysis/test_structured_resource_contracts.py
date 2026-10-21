"""Adversarial actual-domain checks of the resource stage's output contract."""

from dataclasses import replace

import pytest

from ascendflow.analysis.access_frontier import AccessEndpoint
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from ascendflow.analysis.structured_regions import AccessOccurrence, LoopBinding
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement
from ascendflow.diagnostics import CompileError
from ascendflow.passes.generate_notifications import _complete_bindings
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy
from tests.models.structured_domains import actual_dispatch, need_cases
from tests.support.structured import allocated, native_memory, prepared, selected

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
        for j in N.Pipelined(3):
            with N.Stage(0,name="load"):
                value=N.copy(X[j,:],x[j,:])
            with N.Stage(1,name="compute"):
                value=N.mul(value,2.0,out=y[j,:])
            with N.Stage(2,name="store"):
                N.copy(value,O[j,:])
"""


def test_selected_boundary_facts_keep_case_guards_and_lazy_terminal_selection():
    from ascendflow.analysis.notification_lifetimes import HandbackRule

    domains = prepared(selected(scale))
    notification = next(
        n
        for n in domains.notifications
        if n.mechanism is NotificationKind.LOCAL_EVENT and n.source_engine is Engine.MTE2
    )
    case = next(c for c in HandbackRule(domains)._cases(notification) if c.windows and not c.empty)
    first = next(domains._instances(notification, case, None, 0))
    terminal = tuple(domains._instances(notification, case, None, 0, terminal=True))
    assert len(terminal) == 1 and terminal[0] is first
    count = domains.control.loops[case.windows[0].loop].count
    strict = replace(
        case,
        condition=ActionGuard(case.condition.comparisons + (Comparison(count, ">", literal(1)),)),
    )
    other = next(domains._instances(notification, strict, None, 0))
    assert other is not first
    for tiles in (0, 1, 2, 3):
        parameters = {"tiles": tiles, "$group": 0, "$sid": 0}
        original = first[0][0].guard.evaluate(parameters)
        restricted = other[0][0].guard.evaluate(parameters)
        assert restricted == (original and count.evaluate(parameters) > 1)
    # LAST remains a separate selected port, including after FIRST was reused.
    both = tuple(domains._instances(notification, case, None, 0))
    assert len(both) == 2 and both[0] is first
    assert len(tuple(domains._instances(notification, case, None, 0, terminal=True))) == 1


@pytest.mark.parametrize("scope", ["branch", "loop"])
def test_conditional_input_contract_cannot_prove_the_guard_that_precedes_its_load(scope):
    from ascendflow import language as N

    header, upper = (
        ("if condition==0:", "1-condition")
        if scope == "branch"
        else ("for i in N.serial(condition):", "condition")
    )
    memory = native_memory(
        N.from_source(
            f"""
def conditional(Meta:N.Tensor((1,),N.i64),Inner:N.Tensor((1,),N.i64),
                Out:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        row=N.alloc_ub((64,),N.f32)
        N.fill(row,0.0)
        condition=N.load_index(Meta[:],minimum=0,maximum=1)
        {header}
            dependent=N.load_index(Inner[:],minimum=1,maximum={upper})
            N.fill(row,1.0)
        N.copy(row,Out)
""",
            disjoint_external=True,
        )
    )
    dependencies = build_structured_dependencies(memory)
    writer = [t for t in memory.templates if t.spec.engine is Engine.V][-1]
    reader = next(t for t in memory.templates if t.spec.engine is Engine.MTE3)

    def point(template, placement):
        loops = template.frame.loops()
        return BoundPlacement(
            AccessOccurrence(AccessEndpoint.from_template(template, Actor.AIV0), ()),
            placement,
            ActionGuard(),
            tuple((loop.ordinal, literal(0)) for loop in loops),
            (),
            (),
            tuple((loop.operation, literal(0)) for loop in loops),
        )

    # Isolate endpoint validation on real instructions. The proposed source
    # Set is conditional, while its Wait is unconditional. A contract loaded
    # inside the branch/loop holds only after entering it and cannot prove
    # that entry. This is an adversarial protocol, not a DSL support fixture.
    notification = LogicalNotification(
        "conditional_input",
        (point(writer, Placement.AFTER),),
        (point(reader, Placement.BEFORE),),
        Engine.V,
        Engine.MTE3,
        NotificationKind.LOCAL_EVENT,
        (),
        None,
        literal(0),
        1,
        None,
    )
    selection = SynchronizationSelection(
        replace(dependencies, required=()),
        dependencies.dispatch,
        (notification,),
        (),
    )
    with pytest.raises(CompileError, match="activation domains.*equivalent"):
        prepared(selection)


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
        allocated(_complete_bindings(replaced_notification(selection, original, changed)))


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
        allocated(_complete_bindings(replaced_notification(selection, original, changed)))


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
        allocated(_complete_bindings(replaced_notification(selection, original, changed)))


def finite_ports(*, at_most_one=False):
    selection = selected("""
    def anchors(X:N.Tensor((n,64),N.f32),n:N.int64):
        with N.Kernel():
            a,b=N.alloc_ub((2,64)),N.alloc_ub((2,64))
            for j in N.Pipelined(n):
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
        allocated(_complete_bindings(selection))


def test_restricting_the_finite_ports_to_one_iteration_removes_the_cycle():
    selection = finite_ports(at_most_one=True)
    assert not actual_port_cycle(selection, 1)
    plan = allocated(_complete_bindings(selection))
    assert plan.order.root.points


@pytest.mark.parametrize("source", [scale, nested_copy])
def test_emitted_ring_ids_pair_each_actual_generation_across_clocks(source):
    selection = selected(source)
    plan = allocated(_complete_bindings(selection))
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
                                if notification.name in action.protocols
                                and action.kind is kind
                                and action.anchor is point.operation
                                and point.actor in action.participants
                                and (event := action.active_event(runtime)) is not None
                            ]
                            assert events == [expected]

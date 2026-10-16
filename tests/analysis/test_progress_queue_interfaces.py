"""Actual queue dependencies that must survive region interface elimination.

The endpoint fixtures isolate progress from memory coverage. Finite dispatch
order below comes from retained IR, without production progress/rank queries.
"""

from dataclasses import replace

import pytest

from tests.analysis.test_progress_parent_interfaces import PARENT_REUSE
from tests.models.structured_domains import actual_dispatch
from tests.models.structured_events import Command, EventModelError, Occurrence, _causality
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.access_frontier import AccessEndpoint
from xdsl_npu.analysis.event_progress import prove_progress
from xdsl_npu.analysis.progress_regions import ProgressPort, QueueCut
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind


def endpoint(selection, index, actor, phase, guard=None):
    if guard is None:
        guard = ActionGuard()
    memory = selection.dependencies.memory
    template = memory.templates[index]
    coordinates = tuple(
        (loop.operation, loop.ordinal) for loop in memory.control.frames[template.operation].loops()
    )
    return BoundPlacement(
        AccessOccurrence(
            AccessEndpoint.from_template(template, actor),
            tuple(LoopBinding(*coordinate) for coordinate in coordinates),
        ),
        phase,
        guard,
        (),
        (),
        (),
        coordinates,
    )


def isolate(selection, notifications):
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def local_child_route(*, vector_publication):
    publication = "N.fill(x,2.0)" if vector_publication else "N.copy(y,O[:])"
    selection = selected(f"""
    def anchors(X:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write"),n:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            N.fill(y,0.0)
            N.copy(X[:],x)
            for i in N.Pipelined(n,stages=1):
                N.copy(X[:],y)
                N.fill(x,1.0)
            {publication}
    """)
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))

    def notification(name, source, consumer, guard=None):
        return LogicalNotification(
            name,
            (endpoint(selection, source, Actor.AIV0, Placement.AFTER, guard),),
            (endpoint(selection, consumer, Actor.AIV0, Placement.BEFORE, guard),),
            memory.templates[source].contract.engine,
            memory.templates[consumer].contract.engine,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        )

    return isolate(
        selection,
        (
            notification("outer", 4, 1, ActionGuard((Comparison(loop.count, ">", literal(0)),))),
            notification("inner", 2, 3),
        ),
    )


def positions_at_one(selection):
    trace = actual_dispatch(selection, {"n": 1, "$group": 0, "$sid": 0}, Actor.AIV0)
    positions = {key[0]: index for index, (key, _) in enumerate(trace)}
    assert len(trace) == len(positions) == 5
    return positions


def test_internal_cross_queue_path_still_closes_the_actual_outer_cycle():
    selection = local_child_route(vector_publication=True)
    positions = positions_at_one(selection)
    outer, inner = selection.notifications
    assert outer.destination_engine is inner.source_engine is Engine.MTE2
    assert inner.destination_engine is outer.source_engine is Engine.V
    assert positions[outer.consumers[0].operation] < positions[inner.sources[0].operation]
    assert positions[inner.consumers[0].operation] < positions[outer.sources[0].operation]
    with pytest.raises(CompileError):
        prove_progress(prepared(selection))


def test_ordinary_vector_wait_does_not_block_the_outer_mte3_publication():
    selection = local_child_route(vector_publication=False)
    positions = positions_at_one(selection)
    outer, inner = selection.notifications
    assert inner.destination_engine is Engine.V and outer.source_engine is Engine.MTE3
    assert positions[inner.consumers[0].operation] < positions[outer.sources[0].operation]
    proof = prove_progress(prepared(selection))
    child = proof.region.children[0]
    interface = child.interface
    assert not interface.ports
    assert all(cut.operation is child.operation for cut in interface.cuts)
    assert all(
        endpoint in interface.ports + interface.cuts
        for relation in interface.relations
        for endpoint in (relation.source, relation.destination)
    )
    routes = [
        relation
        for relation in interface.relations
        if isinstance(relation.source, QueueCut)
        and isinstance(relation.destination, QueueCut)
        and (relation.source.phase, relation.source.engine) == ("input", Engine.MTE2)
        and (relation.destination.phase, relation.destination.engine) == ("output", Engine.V)
    ]
    assert routes and all(not relation.exact for relation in routes)
    assert any(
        relation.activation.evaluate({"n": 1, "$group": 0, "$sid": 0}) for relation in routes
    )
    assert all(
        not relation.activation.evaluate({"n": 0, "$group": 0, "$sid": 0}) for relation in routes
    )
    assert all(
        not (
            isinstance(relation.destination, QueueCut)
            and relation.destination.engine is None
            and relation.activation.evaluate({"n": 1, "$group": 0, "$sid": 0})
        )
        for relation in interface.relations
    )


def mode2_participants(*, reverse):
    selection = selected("""
    def participants(X:N.Tensor((16,16),N.f16),n:N.int64):
        with N.Kernel(kind="mixed"):
            a=N.alloc_l1((16,16))
            v=N.alloc_ub((64,))
            for i in N.Pipelined(n,stages=1):
                with N.Cube():
                    N.copy(X[:,:],a)
                with N.Vector() as sid:
                    N.fill(v,1.0)
    """)
    cube = endpoint(selection, 0, Actor.AIC, Placement.BEFORE if reverse else Placement.AFTER)
    vector = tuple(
        endpoint(selection, 1, actor, Placement.AFTER if reverse else Placement.BEFORE)
        for actor in (Actor.AIV0, Actor.AIV1)
    )
    notification = LogicalNotification(
        "complete_mode2",
        vector if reverse else (cube,),
        (cube,) if reverse else vector,
        Engine.V if reverse else Engine.MTE2,
        Engine.MTE2 if reverse else Engine.V,
        NotificationKind.CROSS_EVENT,
        (),
        None,
        literal(0),
        1,
        None,
    )
    return isolate(selection, (notification,))


@pytest.mark.parametrize("reverse", [False, True])
def test_mode2_interface_keeps_each_actual_vector_participant(reverse):
    selection = mode2_participants(reverse=reverse)
    actual = {
        actor: actual_dispatch(selection, {"n": 1, "$group": 0}, actor)
        for actor in (Actor.AIC, Actor.AIV0, Actor.AIV1)
    }
    assert all(len(trace) == 1 for trace in actual.values())
    assert actual[Actor.AIV0][0][0][0] is actual[Actor.AIV1][0][0][0]
    proof = prove_progress(prepared(selection))
    child = proof.region.children[0]
    assert not child.interface.ports
    assert all(cut.operation is child.operation for cut in child.interface.cuts)
    relations = [
        relation
        for relation in child.interface.relations
        if isinstance(relation.source, QueueCut)
        and isinstance(relation.destination, QueueCut)
        and relation.source.phase == "input"
        and relation.destination.phase == "output"
        and relation.destination.engine is None
        and relation.activation.evaluate({"n": 1, "$group": 0, "$sid": 0})
    ]
    actual_routes = {
        (relation.source.actor, relation.source.engine, relation.destination.actor)
        for relation in relations
    }
    expected = (
        {(actor, Engine.V, Actor.AIC) for actor in (Actor.AIV0, Actor.AIV1)}
        if reverse
        else {(Actor.AIC, Engine.MTE2, actor) for actor in (Actor.AIV0, Actor.AIV1)}
    )
    assert expected <= actual_routes
    assert all(not relation.exact for relation in relations)
    assert all(
        not relation.activation.evaluate({"n": 0, "$group": 0, "$sid": 0}) for relation in relations
    )


def test_child_interface_cannot_identify_two_distinct_parent_invocations():
    domains = prepared(selected(PARENT_REUSE))
    proof = prove_progress(domains)
    parent = proof.region.children[0]
    child = parent.children[0]
    assert len(child.interface.ports) == 2
    release = child.interface.ports[0].notification
    assert all(port.notification is release for port in child.interface.ports)
    ordinal = domains.control.loops[parent.operation].ordinal
    source = dict(release.sources[0].coordinates)[parent.operation]
    consumer = dict(release.consumers[0].coordinates)[parent.operation]
    for invocation in (1, 2):
        context = {"n": 3, "$group": 0, "$sid": 0, ordinal.args[0]: invocation}
        assert consumer.evaluate(context) - source.evaluate(context) == 1
        assert all(
            not relation.activation.evaluate(context)
            for relation in child.interface.relations
            if isinstance(relation.source, ProgressPort)
            and isinstance(relation.destination, ProgressPort)
            and relation.source.notification is relation.destination.notification is release
            and relation.source.phase != relation.destination.phase
        )


def translated_child_route(count):
    selection = selected(f"""
    def translated(X:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write")):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            N.copy(X[:],x)
            for p in N.Pipelined({count},stages=1):
                for j in N.Pipelined(1,stages=1):
                    N.copy(X[:],x)
                    N.fill(y,1.0)
                N.copy(y,O[:])
            N.copy(y,O[:])
    """)
    memory = selection.dependencies.memory
    parent, child = memory.control.loops.values()
    previous = endpoint(
        selection,
        2,
        Actor.AIV0,
        Placement.AFTER,
        ActionGuard((Comparison(parent.ordinal, "<", literal(count - 1)),)),
    )
    coordinates = (
        (parent.operation, expression("sub", parent.ordinal, literal(1))),
        (child.operation, literal(0)),
    )
    previous = replace(
        previous,
        occurrence=replace(
            previous.occurrence,
            bindings=tuple(LoopBinding(*coordinate) for coordinate in coordinates),
        ),
        coordinates=coordinates,
    )

    def notification(name, source, destination, source_bound=None, destination_guard=None):
        return LogicalNotification(
            name,
            (source_bound or endpoint(selection, source, Actor.AIV0, Placement.AFTER),),
            (endpoint(selection, destination, Actor.AIV0, Placement.BEFORE, destination_guard),),
            memory.templates[source].contract.engine,
            memory.templates[destination].contract.engine,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        )

    return (
        isolate(
            selection,
            (
                notification("outer", 4, 0),
                notification("ready", 1, 2),
                notification(
                    "previous",
                    2,
                    3,
                    previous,
                    ActionGuard((Comparison(parent.ordinal, ">", literal(0)),)),
                ),
            ),
        ),
        parent,
        child,
    )


def translated_commands(selection, parent, child, count):
    """Instantiate only the tiny fixture, giving each matched pair a fresh ID.

    Actual retained dispatch supplies queue order and endpoint membership. The
    event model supplies target blocking semantics; no production interface,
    window, resource allocation or progress relation supplies an edge.
    """
    trace = actual_dispatch(selection, {"$group": 0}, Actor.AIV0)
    positions = {key: index for index, (key, _) in enumerate(trace)}
    assert len(trace) == len(positions) == 2 + 3 * count
    templates = {item.operation: item for item in selection.dependencies.memory.templates}
    before, after, pairs = {}, {}, {}
    event = 0
    for notification in selection.notifications:
        pairs[notification.name] = []
        seen = set()
        for invocation in range(count):
            context = {
                "$group": 0,
                "$sid": 0,
                parent.ordinal.args[0]: invocation,
                child.ordinal.args[0]: 0,
            }
            bounds = notification.sources + notification.consumers
            keys = tuple(
                (
                    bound.operation,
                    bound.actor,
                    tuple((loop, value.evaluate(context)) for loop, value in bound.coordinates),
                )
                for bound in bounds
            )
            if keys in seen or any(key not in positions for key in keys):
                continue
            seen.add(keys)
            commands = []
            for bound, key, kind, destinations in zip(
                bounds,
                keys,
                (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT),
                (after, before),
                strict=True,
            ):
                actual = context | {
                    selection.dependencies.memory.control.loops[loop].ordinal.args[0]: ordinal
                    for loop, ordinal in key[2]
                }
                assert bound.guard.evaluate(actual)
                command = Command(
                    Occurrence(0, 0, bound.actor, bound.operation, key[2]),
                    kind.value,
                    notification.source_engine,
                    notification.destination_engine,
                    event,
                )
                destinations.setdefault(key, []).append(command)
                commands.append(command)
            pairs[notification.name].append(tuple(commands))
            event += 1
    assert event <= 8
    commands = []
    for key, _ in trace:
        operation, actor, coordinates = key
        contract = templates[operation].contract
        commands.extend(before.get(key, ()))
        commands.append(
            Command(
                Occurrence(0, 0, actor, operation, coordinates),
                "native",
                contract.engine,
                contract.engine,
                implicit=contract.implicit_completion,
            )
        )
        commands.extend(after.get(key, ()))
    return commands, pairs


@pytest.mark.parametrize("count", (1, 2))
def test_parent_composes_a_previous_child_publication_with_the_actual_outer_wait(count):
    selection, parent, child = translated_child_route(count)
    commands, pairs = translated_commands(selection, parent, child, count)
    domains = prepared(selection)
    assert len(pairs["outer"]) == 1
    assert len(pairs["ready"]) == count
    assert len(pairs["previous"]) == count - 1
    if count == 1:
        _causality(commands)
        prove_progress(domains)
        return

    outer_set, outer_wait = pairs["outer"][0]
    ready_set, ready_wait = pairs["ready"][0]
    previous_set, previous_wait = pairs["previous"][0]
    assert dict(ready_wait.occurrence.coordinates)[parent.operation] == 0
    assert dict(previous_set.occurrence.coordinates)[parent.operation] == 0
    assert dict(previous_wait.occurrence.coordinates)[parent.operation] == 1
    positions = {id(command): index for index, command in enumerate(commands)}
    for wait, publication in (
        (outer_wait, ready_set),
        (ready_wait, previous_set),
        (previous_wait, outer_set),
    ):
        assert wait.destination is publication.source
        assert positions[id(wait)] < positions[id(publication)]
    with pytest.raises(EventModelError, match="actual queue/controller command deadlock"):
        _causality(commands)
    with pytest.raises(CompileError) as failure:
        prove_progress(domains)
    assert failure.value.phase == "structured-event-progress"

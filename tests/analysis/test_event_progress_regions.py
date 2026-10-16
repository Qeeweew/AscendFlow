"""Small actual-region progress contracts, without full operator builds."""

from dataclasses import replace

import pytest

from tests.analysis.test_structured_resource_contracts import finite_ports
from tests.models.structured_domains import actual_dispatch
from tests.support.structured import selected
from xdsl_npu.analysis.access_frontier import AccessEndpoint
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_progress import ProgressPort, _Composer, prove_progress
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding, LoopDomain
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine

STREAM = """
def stream(X:N.Tensor((n,64),N.f32),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for i in N.Pipelined(n,stages=2):
            with N.Stage(0,name="load"):
                value=N.copy(X[i,:],x[:])
            with N.Stage(1,name="compute"):
                N.mul(value,2.0,out=x[:])
"""


def progress(selection):
    return prove_progress(EventDomains(selection, ResourceWork()))


def isolated(selection, notifications):
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def port(selection, template, *, source, coordinates, activation):
    memory = selection.dependencies.memory
    bindings = tuple(LoopBinding(loop, value) for loop, value in coordinates)
    substitutions = tuple(
        (memory.control.loops[loop].ordinal, value) for loop, value in coordinates
    )
    selectors = tuple(
        Comparison(memory.control.loops[loop].ordinal, "==", value) for loop, value in coordinates
    )
    return BoundPlacement(
        AccessOccurrence(AccessEndpoint.from_template(template, Actor.AIV0), bindings),
        Placement.AFTER if source else Placement.BEFORE,
        ActionGuard(activation + selectors),
        substitutions,
        coordinates,
        (),
        coordinates,
    )


def notification(name, source, consumer, source_engine, destination_engine):
    return LogicalNotification(
        name,
        (source,),
        (consumer,),
        source_engine,
        destination_engine,
        NotificationKind.LOCAL_EVENT,
        (),
        None,
        literal(0),
        1,
        None,
    )


def sibling_ports(*, at_most_one=False):
    selection = selected("""
    def siblings(X:N.Tensor((64,),N.f32),n:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            for j in N.Pipelined(n,stages=1):
                N.fill(x,1.0)
                N.fill(y,1.0)
            for k in N.Pipelined(1,stages=1):
                N.copy(X[:],x)
    """)
    memory = selection.dependencies.memory
    left, right = tuple(memory.control.loops.values())
    a, b, c = memory.templates
    activation = (Comparison(left.count, ">", literal(0)),)
    if at_most_one:
        activation += (Comparison(left.count, "<=", literal(1)),)
    make = lambda template, source, coordinates: port(
        selection, template, source=source, coordinates=coordinates, activation=activation
    )
    notifications = (
        notification(
            "left_last_to_right_first",
            make(a, True, ((left.operation, expression("sub", left.count, literal(1))),)),
            make(c, False, ((right.operation, literal(0)),)),
            Engine.V,
            Engine.MTE2,
        ),
        notification(
            "right_first_to_left_first",
            make(c, True, ((right.operation, literal(0)),)),
            make(b, False, ((left.operation, literal(0)),)),
            Engine.MTE2,
            Engine.V,
        ),
    )
    return isolated(selection, notifications)


def child_to_parent(*, scalar):
    body = "N.load_scalar(x[:1])" if scalar else "N.copy(X[:],x)"
    selection = selected(f"""
    def parent(X:N.Tensor((64,),N.f32),n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for j in N.Pipelined(n,stages=1):
                {body}
            N.fill(x,1.0)
    """)
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))
    consumer, producer = memory.templates
    activation = (Comparison(loop.count, ">", literal(0)),)
    source = port(selection, producer, source=True, coordinates=(), activation=activation)
    destination = port(
        selection,
        consumer,
        source=False,
        coordinates=((loop.operation, literal(0)),),
        activation=activation,
    )
    return isolated(
        selection,
        (
            notification(
                "after_parent_to_first_child",
                source,
                destination,
                Engine.V,
                Engine.S if scalar else Engine.MTE2,
            ),
        ),
    )


def test_single_loop_ready_release_keeps_one_real_clock_certificate():
    plan = progress(selected(STREAM))
    (root,) = (plan.region,)
    assert not root.ports and not root.relations
    (child,) = root.children
    stream = child.stream
    assert stream is not None and stream.operation is child.operation and stream.rank
    assert not child.ports and not child.relations
    for edge in stream.relations:
        assert (
            stream.rank[edge.source] + 1
            <= stream.stride * edge.steps + stream.rank[edge.destination]
        )


def test_dynamic_last_cycle_retains_its_parameter_condition():
    with pytest.raises(CompileError, match="blocking cycle"):
        progress(finite_ports())


def test_dynamic_last_one_iteration_domain_is_acyclic():
    plan = progress(finite_ports(at_most_one=True))
    assert plan.region.children and plan.points


def test_two_actual_sibling_regions_keep_the_same_count_in_their_cycle():
    with pytest.raises(CompileError, match="blocking cycle"):
        progress(sibling_ports())
    plan = progress(sibling_ports(at_most_one=True))
    assert len(plan.region.children) == 2
    assert all(port.bound.coordinates for child in plan.region.children for port in child.ports)


def test_actual_scalar_wait_in_child_blocks_later_parent_dispatch():
    with pytest.raises(CompileError, match="blocking cycle"):
        progress(child_to_parent(scalar=True))


def test_non_scalar_child_wait_does_not_block_other_parent_queue():
    plan = progress(child_to_parent(scalar=False))
    assert plan.region.children
    assert all("controller" not in edge.reason for edge in plan.region.relations)


def test_inactive_minimum_shift_does_not_drop_later_residue_gates():
    selection = selected("""
    def residue(X:N.Tensor((n,64),N.f32),n:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            for j in N.Pipelined(n,stages=1):
                N.fill(x,1.0)
                N.copy(X[j,:],y)
    """)
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))
    vector, load = memory.templates

    def member(template, *, source, residue):
        guards = ActionGuard(
            (Comparison(expression("rem", loop.ordinal, literal(2)), "==", literal(residue)),)
        )
        return BoundPlacement(
            AccessOccurrence(
                AccessEndpoint.from_template(template, Actor.AIV0),
                (LoopBinding(loop.operation, loop.ordinal),),
            ),
            Placement.AFTER if source else Placement.BEFORE,
            guards,
            (),
            (),
            (),
            ((loop.operation, loop.ordinal),),
        )

    notifications = (
        LogicalNotification(
            "even",
            (member(vector, source=True, residue=0),),
            (member(load, source=False, residue=0),),
            Engine.V,
            Engine.MTE2,
            NotificationKind.LOCAL_EVENT,
            (),
            loop.operation,
            loop.ordinal,
            2,
            0,
            (LoopDomain(loop.operation, loop.count, 2, 0),),
        ),
        LogicalNotification(
            "odd",
            (member(load, source=True, residue=1),),
            (member(vector, source=False, residue=1),),
            Engine.MTE2,
            Engine.V,
            NotificationKind.LOCAL_EVENT,
            (),
            loop.operation,
            loop.ordinal,
            2,
            0,
            (LoopDomain(loop.operation, loop.count, 2, 1),),
        ),
    )
    selection = isolated(selection, notifications)
    domains = EventDomains(selection, ResourceWork())
    composer = _Composer(domains)
    wait = ProgressPort(notifications[0], notifications[0].consumers[0], "wait")
    source = ProgressPort(notifications[1], notifications[1].sources[0], "set")
    invocation = composer._invocation_guard(loop.operation, wait, source)
    assert invocation is not None and not invocation.comparisons
    shift, guard, exact = composer._gate_step(wait, source, loop.operation)
    assert shift == 0 and not exact
    assert guard.evaluate({"n": 3})
    # This conservative envelope cannot yet prove the alternating schedule;
    # accepting after deleting its inactive minimum would omit real gates.
    with pytest.raises(CompileError, match="no increasing blocking-interface rank"):
        prove_progress(domains)


def collective_children(*, reverse_vector):
    order = '("second","first")' if reverse_vector else '("first","second")'
    selection = selected(f"""
    def collective(X:N.Tensor((16,16),N.f16),n:N.int64):
        with N.Kernel(1,kind="mixed"):
            a,b=N.alloc_l1((16,16)),N.alloc_l1((16,16))
            v,w=N.alloc_ub((16,16),N.f16),N.alloc_ub((16,16),N.f16)
            for q in N.Pipelined(1,stages=1,issue_order={{"cube":("first","second"),"vector":{order}}}):
                with N.Stage(0,name="first"):
                    for i in N.Pipelined(n,stages=1):
                        with N.Cube():
                            N.copy(X[:,:],a)
                        with N.Vector():
                            N.copy(X[:,:],v)
                with N.Stage(0,name="second"):
                    for j in N.Pipelined(n,stages=1):
                        with N.Cube():
                            N.copy(X[:,:],b)
                        with N.Vector():
                            N.copy(X[:,:],w)
    """)
    memory = selection.dependencies.memory
    outer, first, second = tuple(memory.control.loops.values())
    a, v, b, w = memory.templates

    def endpoint(template, actor, loop, source):
        coordinates = ((outer.operation, outer.ordinal), (loop.operation, loop.ordinal))
        bindings = tuple(LoopBinding(owner, value) for owner, value in coordinates)
        return BoundPlacement(
            AccessOccurrence(AccessEndpoint.from_template(template, actor), bindings),
            Placement.AFTER if source else Placement.BEFORE,
            ActionGuard(),
            (),
            (),
            (),
            coordinates,
        )

    notifications = (
        LogicalNotification(
            "first_vectors_to_cube",
            tuple(endpoint(v, actor, first, True) for actor in (Actor.AIV0, Actor.AIV1)),
            (endpoint(a, Actor.AIC, first, False),),
            Engine.MTE2,
            Engine.MTE2,
            NotificationKind.CROSS_EVENT,
            (),
            first.operation,
            first.ordinal,
            1,
            0,
        ),
        LogicalNotification(
            "second_cube_to_vectors",
            (endpoint(b, Actor.AIC, second, True),),
            tuple(endpoint(w, actor, second, False) for actor in (Actor.AIV0, Actor.AIV1)),
            Engine.MTE2,
            Engine.MTE2,
            NotificationKind.CROSS_EVENT,
            (),
            second.operation,
            second.ordinal,
            1,
            0,
        ),
    )
    return isolated(selection, notifications)


def test_closed_children_do_not_hide_opposite_controller_entry_orders():
    selection = collective_children(reverse_vector=True)
    _outer, first, second = tuple(selection.dependencies.memory.control.loops)
    cube = tuple(
        selection.dispatch.scopes[loop, Actor.AIC].parent_lexical for loop in (first, second)
    )
    vector = tuple(
        selection.dispatch.scopes[loop, Actor.AIV0].parent_lexical for loop in (first, second)
    )
    assert cube[0] < cube[1] and vector[0] > vector[1]
    with pytest.raises(CompileError, match="controller dispatch order") as failure:
        progress(selection)
    assert all(actor.value in " ".join(failure.value.witness) for actor in Actor)
    accepted = progress(collective_children(reverse_vector=False))
    assert accepted.region.children[0].controller_entries


def prologue_stream_epilogue(*, cyclic=False):
    selection = selected("""
    def boundary(X:N.Tensor((n,64),N.f32),Y:N.Tensor((64,),N.f32,access="write"),n:N.int64):
        with N.Kernel():
            p,a,b=N.alloc_ub((64,)),N.alloc_ub((64,)),N.alloc_ub((64,))
            N.fill(p,1.0)
            for i in N.Pipelined(n,stages=1):
                N.copy(X[i,:],a)
                N.mul(a,2.0,out=b)
            N.copy(p,Y[:])
            N.fill(p,2.0)
    """)
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))
    prologue, load, compute, epilogue, after = memory.templates
    activation = (Comparison(loop.count, ">", literal(0)),)

    def endpoint(template, *, source, coordinate=None):
        return port(
            selection,
            template,
            source=source,
            coordinates=() if coordinate is None else ((loop.operation, coordinate),),
            activation=activation,
        )

    notifications = (
        notification(
            "startup",
            endpoint(after if cyclic else prologue, source=True),
            endpoint(load, source=False, coordinate=literal(0)),
            Engine.V,
            Engine.MTE2,
        ),
        notification(
            "body",
            endpoint(load, source=True, coordinate=loop.ordinal),
            endpoint(compute, source=False, coordinate=loop.ordinal),
            Engine.MTE2,
            Engine.V,
        ),
        notification(
            "finish",
            endpoint(compute, source=True, coordinate=expression("sub", loop.count, literal(1))),
            endpoint(epilogue, source=False),
            Engine.V,
            Engine.MTE3,
        ),
    )
    return isolated(selection, notifications)


def test_prologue_stream_epilogue_keeps_two_separate_progress_certificates():
    plan = progress(prologue_stream_epilogue())
    (child,) = plan.region.children
    stream = child.stream
    assert stream is not None
    assert {point.notification.name for point in stream.ports} == {"body"}
    assert {point.notification.name for point in child.ports} == {"startup", "finish"}
    assert len(stream.ports) == len(child.ports) == 2
    assert child.interface.ports == child.ports
    assert all(cut.operation is child.operation for cut in child.interface.cuts)
    retained = child.interface.ports + child.interface.cuts
    assert all(
        relation.source in retained and relation.destination in retained
        for relation in child.interface.relations
    )
    for edge in stream.relations:
        assert (
            stream.rank[edge.source] + 1
            <= stream.stride * edge.steps + stream.rank[edge.destination]
        )
    (bridge,) = child.relations
    assert child.ports[bridge.source].notification.name == "startup"
    assert child.ports[bridge.destination].notification.name == "finish"
    assert not bridge.guard.evaluate({"n": 0, "$group": 0, "$sid": 0})
    for count in (1, 2, 5):
        assert bridge.guard.evaluate({"n": count, "$group": 0, "$sid": 0})


def test_finite_startup_cannot_wait_for_publication_blocked_by_the_owned_stream():
    selection = prologue_stream_epilogue(cyclic=True)
    loop = next(iter(selection.dependencies.memory.control.loops.values()))
    uniform = {"n": 1, "$group": 0, "$sid": 0, loop.ordinal.args[0]: 0}
    positions = {
        key: position
        for position, (key, _) in enumerate(actual_dispatch(selection, uniform, Actor.AIV0))
    }

    def position(point):
        coordinates = tuple((owner, value.evaluate(uniform)) for owner, value in point.coordinates)
        return positions[point.operation, point.actor, coordinates], int(
            point.placement is Placement.AFTER
        )

    startup, body, _ = selection.notifications
    # Two selected matching arcs and these two actual FIFO gates form a cycle.
    assert startup.destination_engine is body.source_engine is Engine.MTE2
    assert body.destination_engine is startup.source_engine is Engine.V
    assert position(startup.consumers[0]) < position(body.sources[0])
    assert position(body.consumers[0]) < position(startup.sources[0])
    with pytest.raises(CompileError, match="blocking cycle|projected exactly"):
        progress(selection)

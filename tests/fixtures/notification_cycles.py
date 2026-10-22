"""Real retained-IR endpoint fixtures for independent generation counterexamples.

These builders specify actual placements, not a progress solver or an oracle.
"""

from dataclasses import replace

from ascendflow.analysis.access_frontier import AccessEndpoint
from ascendflow.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from ascendflow.analysis.structured_regions import AccessOccurrence, LoopBinding, LoopDomain
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_domains import actual_dispatch
from tests.models.structured_events import Command, Occurrence
from tests.support.structured import selected


def isolated(selection, notifications):
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def port(selection, template, *, source, coordinates, activation):
    memory = selection.dependencies.memory
    bindings = tuple((LoopBinding(loop, value) for loop, value in coordinates))
    substitutions = tuple(
        ((memory.control.loops[loop].ordinal, value) for loop, value in coordinates)
    )
    selectors = tuple(
        (Comparison(memory.control.loops[loop].ordinal, "==", value) for loop, value in coordinates)
    )
    coordinate_guards = tuple(
        (
            (loop, ActionGuard((Comparison(memory.control.loops[loop].count, ">", literal(0)),)))
            for loop, value in coordinates
            if value != memory.control.loops[loop].ordinal
        )
    )
    return BoundPlacement(
        occurrence=AccessOccurrence(AccessEndpoint.from_template(template, Actor.AIV0), bindings),
        placement=Placement.AFTER if source else Placement.BEFORE,
        guard=ActionGuard(activation + selectors),
        substitutions=substitutions,
        coordinate_guards=coordinate_guards,
        families=(),
        coordinates=coordinates,
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
    selection = selected(
        "\n    def siblings(X:N.Tensor((64,),N.f32),n:N.i64):\n        with N.Kernel():\n            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))\n            for j in N.Pipelined(n):\n                N.fill(x,1.0)\n                N.fill(y,1.0)\n            for k in N.Pipelined(1):\n                N.copy(X[:],x)\n"
    )
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
    selection = selected(
        f"\n    def parent(X:N.Tensor((64,),N.f32),n:N.i64):\n        with N.Kernel():\n            x=N.alloc_ub((64,))\n            for j in N.Pipelined(n):\n                {body}\n            N.fill(x,1.0)\n"
    )
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


def alternating_queues():
    selection = selected(
        "\n    def residue(X:N.Tensor((n,64),N.f32),n:N.i64):\n        with N.Kernel():\n            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))\n            for j in N.Pipelined(n):\n                N.fill(x,1.0)\n                N.copy(X[j,:],y)\n"
    )
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
    return selection


def collective_children(*, reverse_vector):
    # This is an explicit physical-input counterexample: each child has a
    # closed C/V transfer, while the parent issues the children in different
    # controller orders. The public DSL exposes only actor-local Stage scopes.
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IndexType, IntAttr, StringAttr
    from xdsl.ir import Block, Region

    from ascendflow import language as N
    from ascendflow.analysis.structured_dependencies import build_structured_dependencies
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.physical import EndOp
    from ascendflow.dialects.structured import PipelineLoopOp, PipelineStageOp
    from ascendflow.passes.lower_native import lower_native_structure
    from ascendflow.passes.notification_binding import bind_notifications

    structure = lower_native_structure(
        N.from_source("""
    def collective(X:N.Tensor((16,16),N.f16),n:N.i64):
        with N.Kernel(1,kind="mixed"):
            a,b=N.alloc_l1((16,16)),N.alloc_l1((16,16))
            v,w=N.alloc_ub((16,16),N.f16),N.alloc_ub((16,16),N.f16)
            for q in N.Dataflow(1):
                for i in N.Dataflow(n):
                    with N.Cube():
                        N.copy(X[:,:],a)
                    with N.Vector():
                        N.copy(X[:,:],v)
                for j in N.Dataflow(n):
                    with N.Cube():
                        N.copy(X[:,:],b)
                    with N.Vector():
                        N.copy(X[:,:],w)
    """)
    )
    outer = next(op for op in structure.module.walk() if isinstance(op, PipelineLoopOp))
    first_stage = next(op for op in outer.body.block.ops if isinstance(op, PipelineStageOp))
    children = [op for op in first_stage.body.block.ops if isinstance(op, PipelineLoopOp)]
    first_stage.label = StringAttr("first")
    first_stage.body.block.detach_op(children[1])
    second_stage = PipelineStageOp.build(
        attributes={"offset": IntAttr(0), "label": StringAttr("second")},
        regions=[Region(Block([children[1], EndOp()], arg_types=[IndexType()]))],
    )
    outer.body.block.insert_op_after(second_stage, first_stage)
    outer.issue_order = DictionaryAttr(
        {
            "cube": ArrayAttr(StringAttr(x) for x in ("first", "second")),
            "vector": ArrayAttr(
                StringAttr(x)
                for x in (("second", "first") if reverse_vector else ("first", "second"))
            ),
        }
    )
    structure.module.verify()
    memory = collect_native_memory(structure.module)
    dependencies = build_structured_dependencies(memory)
    selection = bind_notifications(dependencies, dependencies.dispatch)
    selection.require_closed()
    memory = selection.dependencies.memory
    outer, first, second = tuple(memory.control.loops.values())
    a, v, b, w = memory.templates

    def endpoint(template, actor, loop, source):
        coordinates = ((outer.operation, outer.ordinal), (loop.operation, loop.ordinal))
        bindings = tuple((LoopBinding(owner, value) for owner, value in coordinates))
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


def prologue_stream_epilogue(*, cyclic=False):
    selection = selected(
        '\n    def boundary(X:N.Tensor((n,64),N.f32),Y:N.Tensor((64,),N.f32,access="write"),n:N.i64):\n        with N.Kernel():\n            p,a,b=N.alloc_ub((64,)),N.alloc_ub((64,)),N.alloc_ub((64,))\n            N.fill(p,1.0)\n            for i in N.Pipelined(n):\n                N.copy(X[i,:],a)\n                N.mul(a,2.0,out=b)\n            N.copy(p,Y[:])\n            N.fill(p,2.0)\n'
    )
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
    selection = selected(
        f'\n    def anchors(X:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write"),n:N.i64):\n        with N.Kernel():\n            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))\n            N.fill(y,0.0)\n            N.copy(X[:],x)\n            for i in N.Pipelined(n):\n                N.copy(X[:],y)\n                N.fill(x,1.0)\n            {publication}\n'
    )
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))

    def notification(name, source, consumer, guard=None):
        return LogicalNotification(
            name,
            (endpoint(selection, source, Actor.AIV0, Placement.AFTER, guard),),
            (endpoint(selection, consumer, Actor.AIV0, Placement.BEFORE, guard),),
            memory.templates[source].spec.engine,
            memory.templates[consumer].spec.engine,
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


def mode2_participants(*, reverse):
    selection = selected(
        '\n    def participants(X:N.Tensor((16,16),N.f16),n:N.i64):\n        with N.Kernel(kind="mixed"):\n            a=N.alloc_l1((16,16))\n            v=N.alloc_ub((64,))\n            for i in N.Dataflow(n):\n                with N.Cube():\n                    N.copy(X[:,:],a)\n                with N.Vector() as sid:\n                    N.fill(v,1.0)\n'
    )
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


def translated_child_route(count):
    selection = selected(
        f'\n    def translated(X:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write")):\n        with N.Kernel():\n            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))\n            N.copy(X[:],x)\n            for p in N.Pipelined({count}):\n                for j in N.Pipelined(1):\n                    N.copy(X[:],x)\n                    N.fill(y,1.0)\n                N.copy(y,O[:])\n            N.copy(y,O[:])\n'
    )
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
        substitutions=(
            (parent.ordinal, expression("add", parent.ordinal, literal(1))),
            (child.ordinal, literal(0)),
        ),
    )

    def notification(name, source, destination, source_bound=None, destination_guard=None):
        return LogicalNotification(
            name,
            (source_bound or endpoint(selection, source, Actor.AIV0, Placement.AFTER),),
            (endpoint(selection, destination, Actor.AIV0, Placement.BEFORE, destination_guard),),
            memory.templates[source].spec.engine,
            memory.templates[destination].spec.engine,
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
    before, after, pairs = ({}, {}, {})
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
                    tuple(((loop, value.evaluate(context)) for loop, value in bound.coordinates)),
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
        contract = templates[operation].spec
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
    return (commands, pairs)


def nested_channel_cycle():
    selection = selected(
        '\n    def anchors(X:N.Tensor((16,16),N.f16),O:N.Tensor((16,16),N.f32,access="write"),n:N.i64):\n        with N.Kernel(kind="cube"):\n            x=N.alloc_l1((16,16))\n            a=N.alloc_l0a((16,16))\n            b=N.alloc_l0b((16,16))\n            c=N.alloc_l0c((16,16))\n            N.copy(X[:,:],x)\n            for i in N.Pipelined(n):\n                N.copy(X[:,:],x)\n                N.copy(x,a)\n                N.copy(x,b,transpose=True)\n                for j in N.Pipelined(n):\n                    N.copy(x,a)\n                    N.matmul(a,b,accum=c,initialize=True)\n                N.matmul(a,b,accum=c,initialize=True)\n                N.copy(c,O[:,:])\n            N.copy(c,O[:,:])\n'
    )
    memory = selection.dependencies.memory

    def port(index, source):
        template = memory.templates[index]
        coordinates = tuple(
            (loop.operation, loop.ordinal)
            for loop in memory.control.frames[template.operation].loops()
        )
        return BoundPlacement(
            AccessOccurrence(
                AccessEndpoint.from_template(template, Actor.AIC),
                tuple(LoopBinding(*coordinate) for coordinate in coordinates),
            ),
            Placement.AFTER if source else Placement.BEFORE,
            ActionGuard(()),
            (),
            (),
            (),
            coordinates,
        )

    def notification(name, source, destination):
        return LogicalNotification(
            name,
            (port(source, True),),
            (port(destination, False),),
            memory.templates[source].spec.engine,
            memory.templates[destination].spec.engine,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        )

    notifications = (
        notification("root", 8, 0),
        notification("outer_left", 1, 2),
        notification("inner", 4, 5),
        notification("outer_right", 6, 7),
    )
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def finite_ports(*, at_most_one=False):
    selection = selected(
        "\n    def anchors(X:N.Tensor((n,64),N.f32),n:N.i64):\n        with N.Kernel():\n            a,b=N.alloc_ub((2,64)),N.alloc_ub((2,64))\n            for j in N.Pipelined(n):\n                N.fill(a[j%2,:],1.0)\n                N.fill(b[j%2,:],1.0)\n                N.copy(X[j,:],a[j%2,:])\n"
    )
    memory = selection.dependencies.memory
    logical = next(iter(memory.control.loops.values()))
    loop, ordinal, count = (logical.operation, logical.ordinal, logical.count)
    first_vector, later_vector, load = memory.templates

    def port(template, *, first, source):
        coordinate = literal(0) if first else expression("sub", count, literal(1))
        nonempty = Comparison(count, ">", literal(0))
        activation = (nonempty,)
        if at_most_one:
            activation += (Comparison(count, "<=", literal(1)),)
        occurrence = AccessOccurrence(
            AccessEndpoint.from_template(template, Actor.AIV0), (LoopBinding(loop, coordinate),)
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
        coordinates = tuple(((loop, value.evaluate(uniform)) for loop, value in point.coordinates))
        return positions[point.operation, point.actor, coordinates]

    first, second = selection.notifications
    assert first.destination_engine is second.source_engine
    assert second.destination_engine is first.source_engine
    return position(first.consumers[0]) <= position(second.sources[0]) and position(
        second.consumers[0]
    ) <= position(first.sources[0])


PARENT_REUSE = "\ndef parent_reuse(X:N.Tensor((n,2,64),N.f32),n:N.i64):\n    with N.Kernel():\n        x=N.alloc_ub((2,64))\n        for p in N.Pipelined(n):\n            for j in N.Pipelined(2):\n                with N.Stage(0):\n                    v=N.copy(X[p,j,:],x[j,:])\n                with N.Stage(1):\n                    N.mul(v,2.0,out=x[j,:])\n"

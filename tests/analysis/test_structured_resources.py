"""Finite uses, real reentry paths and target event banks on selected IR."""

from dataclasses import replace

import pytest

from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.event_contracts import CROSS_BACKLOG, LeaseKind, _Order
from xdsl_npu.analysis.event_lifetimes import reentry
from xdsl_npu.analysis.structured_protocols import NotificationKind
from xdsl_npu.analysis.structured_resources import allocate_structured_events
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind

SAME_QUEUE = """
def same_queue(n:N.int64,m:N.int64):
    with N.Kernel():
        x,y=N.alloc_ub((2,8)),N.alloc_ub((2,8))
        for p in N.serial(n):
            for g in N.Pipelined(m,stages=2):
                slot=(p*m+g)%2
                with N.Stage(0):
                    N.fill(x[slot,:],1.0)
                with N.Stage(1):
                    N.mul(x[slot,:],2.0,out=y[slot,:])
"""


@pytest.fixture(scope="module")
def same_queue():
    from xdsl_npu import language as N
    from xdsl_npu.budget import limit
    from xdsl_npu.passes.structured_sync import synchronize_structure

    with limit(30):
        return synchronize_structure(N.from_source(SAME_QUEUE))


@pytest.mark.parametrize("n,m", ((0, 3), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)))
def test_same_queue_fences_keep_byte_hazards_without_binary_event_domains(same_queue, n, m):
    from tests.models.structured_events import check_structured_events

    plan = same_queue.resources
    assert plan.selection.notifications
    assert all(
        notification.mechanism is NotificationKind.PIPE_FENCE
        for notification in plan.selection.notifications
    )
    assert not plan.assignments and not plan.leases
    assert plan.work.parameter_cases == 0 and plan.work.window_constructions == 0
    for action in (action for action in plan.actions if action.kind is SyncKind.FENCE):
        notification = next(n for n in plan.selection.notifications if n.name == action.protocol)
        assert any(action.guard == bound.guard for bound in notification.consumers)
        assert action.event is None
    execution = check_structured_events(same_queue.memory, {"n": n, "m": m}, max_commands=2048)
    assert len(execution.natives) == 2 * n * m
    assert execution.bytes_read == 32 * n * m
    assert execution.bytes_written == 64 * n * m


def test_fence_endpoint_still_requires_its_raw_unsigned_domain():
    from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
    from xdsl_npu.scalar import Expr

    selected_ir = selected(SAME_QUEUE)
    fence = selected_ir.notifications[0]
    assert fence.mechanism is NotificationKind.PIPE_FENCE
    unsafe = expression("rem", expression("sub", literal(0), Expr("argument", ("m",))), literal(2))
    consumer = replace(
        fence.consumers[0], guard=ActionGuard((Comparison(unsafe, "==", literal(0)),))
    )
    forged = replace(fence, sources=(consumer,), consumers=(consumer,), requirements=(), domains=())
    selected_ir = replace(
        selected_ir,
        dependencies=replace(selected_ir.dependencies, required=()),
        notifications=(forged,),
        coverage=(),
    )
    with pytest.raises(CompileError, match="ordered activation prefix proof") as error:
        prepared(selected_ir)
    assert error.value.phase == "structured-event-pairing"


@pytest.fixture(scope="module")
def finite():
    return allocate_structured_events(
        selected("""
        def finite(X:N.Tensor((2,64),N.f32),O:N.Tensor((2,64),N.f32,access="write")):
            with N.Kernel():
                x,y=N.alloc_ub((2,64)),N.alloc_ub((2,64))
                for j in N.Pipelined(2,stages=3):
                    with N.Stage(0,name="load"):
                        value=N.copy(X[j,:],x[j,:])
                    with N.Stage(1,name="compute"):
                        value=N.mul(value,2.0,out=y[j,:])
                    with N.Stage(2,name="store"):
                        N.copy(value,O[j,:])
    """)
    )


@pytest.fixture(scope="module")
def repeated():
    return allocate_structured_events(selected(nested_copy))


@pytest.fixture(scope="module")
def mixed():
    return allocate_structured_events(selected(nested_cv))


def test_two_finite_uses_get_two_ids_without_selecting_reverse_events(finite):
    assert len(finite.assignments) == 2
    assert all(len(item.ids) == 2 for item in finite.assignments)
    assert all(
        case.inside[0].kind is LeaseKind.FINITE for lease in finite.leases for case in lease.cases
    )
    assert all(
        witness.path_cover is None
        for lease in finite.leases
        for case in lease.cases
        for witness in case.inside
    )
    assert all(not case.reentries for lease in finite.leases for case in lease.cases)
    assert {
        (item.notification.source_engine, item.notification.destination_engine)
        for item in finite.assignments
    } == {(Engine.MTE2, Engine.V), (Engine.V, Engine.MTE3)}
    ordinal = next(iter(finite.selection.dependencies.memory.control.loops.values())).ordinal
    for assignment in finite.assignments:
        endpoints = [
            action for action in finite.actions if action.protocol == assignment.notification.name
        ]
        ids = {kind: [] for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT)}
        for action in endpoints:
            for generation in range(2):
                event = action.active_event({ordinal.args[0]: generation})
                if event is not None:
                    ids[action.kind].append(event)
        assert sorted(ids[SyncKind.LOCAL_SET]) == sorted(assignment.ids)
        assert sorted(ids[SyncKind.LOCAL_WAIT]) == sorted(assignment.ids)


def test_real_parent_reentry_has_its_own_consumption_witness(repeated):
    invocations = [
        witness
        for lease in repeated.leases
        for case in lease.cases
        for witness in case.reentries
        if witness.kind is LeaseKind.INVOCATION
    ]
    assert invocations and any(witness.delta == 2 for witness in invocations)
    assert all(witness.path_cover is not None for witness in invocations)
    assert any(witness.path_cover.paths for witness in invocations)
    for lease in repeated.leases:
        for case in lease.cases:
            for witness in case.inside + case.reentries:
                if witness.kind is LeaseKind.FINITE:
                    assert witness.path_cover is None
                    continue
                assert witness.path_cover is not None
                for path in witness.path_cover.paths:
                    current = lease.notification
                    assert path.steps
                    for step in path.steps:
                        assert step.controller == (
                            current.mechanism is NotificationKind.CROSS_EVENT
                            or current.destination_engine is Engine.S
                        )
                        if step.notification is not None:
                            current = step.notification


def test_missing_real_return_path_does_not_become_child_exit_completion(repeated):
    selected = repeated.selection
    allocator = prepared(selected)
    ready = next(
        n
        for n in selected.notifications
        if n.source_engine is Engine.MTE2
        and n.sources[0].operation is selected.dependencies.memory.templates[0].operation
    )
    (case,) = allocator.cases[ready]
    parent = case.windows[0].loop
    remaining = replace(
        selected,
        notifications=tuple(n for n in selected.notifications if n.source_engine is Engine.MTE2),
        dependencies=replace(selected.dependencies, required=()),
        coverage=(),
    )
    allocator = prepared(remaining)
    (case,) = allocator.cases[ready]
    assert reentry(allocator, ready, case, case.window(parent), 2) is None


def test_non_scalar_wait_only_gates_its_real_destination_queue():
    selected_ir = selected(scale)
    allocator = prepared(selected_ir)
    forward = next(n for n in selected_ir.notifications if n.source_engine is Engine.MTE2)
    reverse = next(
        n
        for n in selected_ir.notifications
        if n.source_engine is Engine.V and n.destination_engine is Engine.MTE2
    )
    (forward_case,) = allocator.cases[forward]
    (reverse_case,) = allocator.cases[reverse]
    loop = forward_case.windows[0].loop
    ordinal = allocator.control.loops[loop].ordinal
    wait = allocator._instance(forward.consumers[0], {}, case=forward_case)
    source = allocator._instance(
        forward.sources[0], {ordinal: expression("add", ordinal, literal(4))}, case=forward_case
    )
    assert not allocator._following_conditions(forward, wait, forward, source)
    assert allocator._may_follow(forward, wait, forward, source, ()) is _Order.IMPOSSIBLE
    source = allocator._instance(
        reverse.sources[0], {ordinal: expression("add", ordinal, literal(4))}, case=reverse_case
    )
    assert allocator._following_conditions(forward, wait, reverse, source)
    assert allocator._may_follow(forward, wait, reverse, source, ()) is _Order.PROVED
    assert not allocator._gate(forward)


def test_mode2_keeps_every_real_contributor_receiver_and_bounds_backlog(mixed):
    cross = [
        assignment
        for assignment in mixed.assignments
        if assignment.notification.mechanism is NotificationKind.CROSS_EVENT
    ]
    assert len(cross) == 2
    for assignment in cross:
        assert len(assignment.ids) == 1 and all(case.width == 1 for case in assignment.cases)
        notification = assignment.notification
        assert (
            {b.actor for b in notification.sources},
            {b.actor for b in notification.consumers},
        ) in [({Actor.AIC}, {Actor.AIV0, Actor.AIV1}), ({Actor.AIV0, Actor.AIV1}, {Actor.AIC})]
        actions = [a for a in mixed.actions if a.protocol == notification.name]
        assert {
            actor for a in actions if a.kind is SyncKind.CROSS_SET for actor in a.participants
        } == {b.actor for b in notification.sources}
        assert {
            actor for a in actions if a.kind is SyncKind.CROSS_WAIT for actor in a.participants
        } == {b.actor for b in notification.consumers}
    assert all(
        case.backlog is not None and 1 <= case.backlog <= CROSS_BACKLOG
        for lease in mixed.leases
        if lease.notification.mechanism is NotificationKind.CROSS_EVENT
        for case in lease.cases
    )


def test_configuration_and_final_drain_ids_are_in_the_same_bank_ledger():
    plan = allocate_structured_events(selected(batched_matmul))
    assert any(
        reservation.bank == (Actor.AIC, Engine.S, Engine.FIX)
        and reservation.event == 0
        and "configuration" in reservation.reason
        for reservation in plan.reservations
    )
    drains = [r for r in plan.reservations if "final" in r.reason]
    assert {r.bank[1] for r in drains} == {Engine.MTE2, Engine.MTE1, Engine.M, Engine.FIX}
    assert all(r.event == 7 and r.bank[2] is Engine.S for r in drains)
    for reservation in plan.reservations:
        assert all(
            reservation.event not in assignment.ids
            for assignment in plan.assignments
            if reservation.bank in assignment.banks
        )
    assert len([a for a in plan.actions if a.protocol.startswith("final_")]) == 2 * len(drains)


def test_each_region_certifies_stream_ranks_and_guarded_interfaces(repeated):
    domains = prepared(repeated.selection)
    opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}

    def check(region):
        assert all(port.phase in {"set", "wait"} for port in region.ports)
        stream = region.stream
        if stream is not None:
            assert len(stream.rank) == len(stream.nodes)
            for edge in stream.relations:
                assert (
                    stream.rank[edge.source] + 1
                    <= stream.stride * edge.steps + stream.rank[edge.destination]
                )
        for cycle in region.excluded_cycles:
            tests = cycle.condition.comparisons
            assert any(
                domains.arithmetic.prove(
                    type(test)(test.lhs, opposite[test.predicate], test.rhs),
                    tests[:index] + tests[index + 1 :],
                )
                for index, test in enumerate(tests)
            )
        for child in region.children:
            check(child)

    assert repeated.progress.region is not None
    check(repeated.progress.region)


def test_conditional_lifetime_does_not_certify_a_zero_distance_deadlock():
    selected_ir = selected(scale)
    release = next(
        n
        for n in selected_ir.notifications
        if n.source_engine is Engine.V and n.destination_engine is Engine.MTE2
    )
    ready = next(n for n in selected_ir.notifications if n.source_engine is Engine.MTE2)
    wrong = replace(
        release,
        sources=(
            replace(
                release.sources[0],
                coordinates=ready.sources[0].coordinates,
                guard=ready.sources[0].guard,
                substitutions=(),
            ),
        ),
        consumers=(
            replace(release.consumers[0], guard=ready.consumers[0].guard, substitutions=()),
        ),
        distance=0,
        domains=ready.domains,
    )
    selected_ir = replace(
        selected_ir,
        notifications=tuple(wrong if n is release else n for n in selected_ir.notifications),
        coverage=tuple(
            replace(item, notification=wrong) if item.notification is release else item
            for item in selected_ir.coverage
        ),
    )
    with pytest.raises(CompileError, match="no increasing blocking-interface rank") as failure:
        allocate_structured_events(selected_ir)
    assert failure.value.phase == "structured-event-progress" and failure.value.witness

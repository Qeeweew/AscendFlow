"""Finite uses, real reentry paths and target event banks on selected IR."""

from dataclasses import replace

import pytest

from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv
from tests.support.structured import prepared, selection
from xdsl_npu.analysis.event_contracts import CROSS_BACKLOG, LeaseKind
from xdsl_npu.analysis.event_lifetimes import reentry
from xdsl_npu.analysis.structured_protocols import NotificationKind
from xdsl_npu.analysis.structured_resources import allocate_structured_events
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind


@pytest.fixture(scope="module")
def finite():
    return allocate_structured_events(
        selection("""
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
    return allocate_structured_events(selection(nested_copy))


@pytest.fixture(scope="module")
def mixed():
    return allocate_structured_events(selection(nested_cv))


def test_two_finite_uses_get_two_ids_without_selecting_reverse_events(finite):
    assert len(finite.assignments) == 2
    assert all(len(item.ids) == 2 for item in finite.assignments)
    assert all(lease.inside[0].kind is LeaseKind.FINITE for lease in finite.leases)
    assert all(not lease.reentries for lease in finite.leases)
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
        for witness in lease.reentries
        if witness.kind is LeaseKind.INVOCATION
    ]
    assert invocations and any(witness.delta == 2 for witness in invocations)
    assert all(witness.steps for witness in invocations)
    for lease in repeated.leases:
        for witness in lease.inside + lease.reentries:
            current = lease.notification
            for step in witness.steps:
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
    parent = allocator.windows[ready][0].loop
    remaining = replace(
        selected,
        notifications=tuple(n for n in selected.notifications if n.source_engine is Engine.MTE2),
        dependencies=replace(selected.dependencies, required=()),
        coverage=(),
    )
    allocator = prepared(remaining)
    assert reentry(allocator, ready, parent, 2) is None


def test_non_scalar_wait_only_gates_its_real_destination_queue():
    selected = selection(scale)
    allocator = prepared(selected)
    forward = next(n for n in selected.notifications if n.source_engine is Engine.MTE2)
    reverse = next(
        n
        for n in selected.notifications
        if n.source_engine is Engine.V and n.destination_engine is Engine.MTE2
    )
    loop = allocator.windows[forward][0].loop
    ordinal = allocator.control.loops[loop].ordinal
    wait = allocator._instance(forward.consumers[0], {})
    source = allocator._instance(
        forward.sources[0], {ordinal: expression("add", ordinal, literal(4))}
    )
    assert allocator._order(wait, source, ())
    assert not allocator._follows(forward, wait, forward, source, ())
    source = allocator._instance(
        reverse.sources[0], {ordinal: expression("add", ordinal, literal(4))}
    )
    assert allocator._follows(forward, wait, reverse, source, ())
    assert not allocator._gate(forward)


def test_mode2_keeps_every_real_contributor_receiver_and_bounds_backlog(mixed):
    cross = [
        assignment
        for assignment in mixed.assignments
        if assignment.notification.mechanism is NotificationKind.CROSS_EVENT
    ]
    assert len(cross) == 2
    for assignment in cross:
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
        lease.backlog is not None and 1 <= lease.backlog <= CROSS_BACKLOG
        for lease in mixed.leases
        if lease.notification.mechanism is NotificationKind.CROSS_EVENT
    )


def test_configuration_and_final_drain_ids_are_in_the_same_bank_ledger():
    plan = allocate_structured_events(selection(batched_matmul))
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
        assert all(0 <= index < len(region.ports) for index in region.interface)
        assert all(port.phase in {"set", "wait"} for port in region.ports)
        if region.clock is not None:
            assert len(region.rank) == len(region.ports)
            for edge in region.relations:
                assert (
                    region.rank[edge.source] + 1
                    <= region.stride * edge.steps + region.rank[edge.destination]
                )
        for cycle in region.excluded_cycles:
            tests = cycle.condition.comparisons
            assert any(
                domains._prove(
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
    selected = selection(scale)
    release = next(
        n
        for n in selected.notifications
        if n.source_engine is Engine.V and n.destination_engine is Engine.MTE2
    )
    ready = next(n for n in selected.notifications if n.source_engine is Engine.MTE2)
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
    selected = replace(
        selected,
        notifications=tuple(wrong if n is release else n for n in selected.notifications),
        coverage=tuple(
            replace(item, notification=wrong) if item.notification is release else item
            for item in selected.coverage
        ),
    )
    with pytest.raises(CompileError, match="no increasing blocking-interface rank") as failure:
        allocate_structured_events(selected)
    assert failure.value.phase == "structured-event-progress" and failure.value.witness

"""Historical M2 pipelines and independent checks of their binary-event leases."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.ir import Block, Region

from tests.fixtures.pipelines import cube_pipeline, scalar_pipeline, vector_pipeline
from tests.legacy.analysis.protocols import LocalMechanism, LocalProtocol
from tests.legacy.analysis.resources import (
    ControllerBridge,
    ControllerWitness,
    EventAction,
    LocalResourceWork,
    allocate_local_events,
)
from tests.legacy.passes.insert_sync import synchronize_local
from tests.legacy.passes.local_sync import select_local_protocols
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis.dependencies import analyze_dependencies
from xdsl_npu.analysis.frontier import Clock
from xdsl_npu.analysis.hazards import Activation, bind_hazards
from xdsl_npu.analysis.native import extract_native_tasks
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.physical import BoundaryOp, EndOp, TaskOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.target.ascend910b import Engine, Milestone


def selected(module):
    loop = extract_loop(module)
    tasks = extract_native_tasks(loop, clock=Clock("g"))
    hazards = bind_hazards(loop, tasks, analyze_dependencies(loop).required)
    return select_local_protocols(tasks, hazards).protocols


def find(protocols, source, destination):
    return next(
        p
        for p in protocols
        if p.source.operation.uid.data == source and p.destination.operation.uid.data == destination
    )


def check_witness(witness, generation, count):
    """Interpret actual endpoint guards/order, without production proof helpers."""
    p = witness.protocol
    assert generation >= witness.first
    assert generation % witness.period == witness.residue
    assert p.activation.consumes(generation, count)
    assert p.activation.consumes(generation + witness.successor_delta, count)
    if isinstance(witness, ControllerWitness):
        assert p.destination.engine is Engine.S
        assert witness.consume.action is EventAction.WAIT
        assert witness.consume.anchor.operation is p.destination.operation
        assert witness.blocking_end.operation is p.destination.operation
        assert witness.blocking_end.engine is Engine.S
        assert witness.blocking_end.milestone is Milestone.NATIVE_END
        assert witness.dispatch.operation is p.source.operation
        assert witness.dispatch.engine is None
        assert witness.dispatch.milestone is Milestone.ISSUE
        assert witness.next_set.anchor.operation is p.source.operation
        assert witness.next_set.anchor.milestone is Milestone.NATIVE_END
        assert witness.next_set.generation_offset == witness.successor_delta - p.distance
        before, after = witness.blocking_end.rank, witness.dispatch.rank
        assert (before.offset, before.lexical) < (after.offset, after.lexical)
        return
    q = witness.peer
    source = generation + witness.peer_set.generation_offset
    destination = generation + witness.peer_wait.generation_offset
    assert destination == source + q.distance
    assert q.activation.produces(source, count)
    assert q.activation.consumes(destination, count)
    assert witness.next_set.generation_offset == witness.successor_delta - p.distance
    assert witness.consume.action is EventAction.WAIT
    assert witness.peer_set.action is EventAction.SET
    assert witness.peer_wait.action is EventAction.WAIT
    assert witness.next_set.action is EventAction.SET
    assert witness.peer_set.anchor.operation is q.source.operation
    assert witness.peer_wait.anchor.operation is q.destination.operation
    assert witness.next_set.anchor.operation is p.source.operation
    assert witness.peer_set.anchor.milestone is Milestone.NATIVE_END
    for queue in witness.queues:
        if isinstance(queue, ControllerBridge):
            assert queue.wait is witness.peer_wait
            assert queue.wait.anchor.engine is Engine.S
            assert queue.blocking_end.operation is q.destination.operation
            assert queue.blocking_end.milestone is Milestone.NATIVE_END
            assert queue.blocking_end.actor is queue.dispatch.actor
            assert queue.blocking_end.clock is queue.dispatch.clock
            assert queue.dispatch.operation is p.source.operation
            assert queue.dispatch.engine is None
            assert queue.dispatch.milestone is Milestone.ISSUE
            before, after = queue.blocking_end.rank, queue.dispatch.rank
            assert (before.offset, before.lexical) < (after.offset, after.lexical)
            continue
        before, after = queue.wait.anchor, queue.native
        assert before.actor is after.actor
        assert before.engine is after.engine
        assert before.clock is after.clock
        assert after.milestone is Milestone.ADMISSION
        assert (before.rank.offset, before.rank.lexical) <= (after.rank.offset, after.rank.lexical)


def check_plan(plan):
    for assignment, lease in zip(plan.assignments, plan.leases):
        assert assignment.protocol is lease.protocol
        assert len(assignment.ids) == lease.width
        for count in range(12):
            notifications = [
                g for g in range(count) if lease.protocol.activation.consumes(g, count)
            ]
            assert [lease.ordinal(g) for g in notifications] == list(range(len(notifications)))
            for index, generation in enumerate(notifications[: -lease.width]):
                next_generation = notifications[index + lease.width]
                witnesses = [w for w in lease.witnesses if generation % w.period == w.residue]
                assert len(witnesses) == 1
                assert witnesses[0].successor_delta == next_generation - generation
                check_witness(witnesses[0], generation, count)
        for witness in lease.witnesses:
            generation = witness.first + 10**12 * witness.period
            check_witness(witness, generation, generation + witness.successor_delta + 1)


@pytest.mark.parametrize("slots", [1, 2, 3, 8])
@pytest.mark.parametrize("stages", [(0, 0, 0), (0, 1, 2), (0, 2, 4)])
def test_scale_uses_real_ready_release_and_active_ordinal(slots, stages):
    protocols = selected(vector_pipeline(slots=slots, stages=stages))
    plan = allocate_local_events(protocols)
    assert len(plan.assignments) == 4
    assert len([p for p in protocols if p.mechanism is LocalMechanism.FENCE]) == 1
    assert all(lease.width == slots for lease in plan.leases)
    check_plan(plan)


def test_cube_actual_coalesced_interfaces_have_retirement_roundtrips():
    plan = allocate_local_events(selected(cube_pipeline()))
    assert len(plan.assignments) == 8
    assert all(lease.width == 2 for lease in plan.leases)
    bank_ids = {}
    for assignment in plan.assignments:
        ids = bank_ids.setdefault(assignment.protocol.bank, set())
        assert ids.isdisjoint(assignment.ids)
        ids.update(assignment.ids)
    assert max(map(len, bank_ids.values())) == 4
    ready = next(
        lease
        for lease in plan.leases
        if lease.protocol.source.operation.uid.data == "operand_b"
        and lease.protocol.destination.operation.uid.data == "mad"
    )
    assert ready.roundtrips[0].peer.destination.operation.uid.data == "operand_a"
    # This queue step is operand_a begin -> operand_b begin, not end FIFO.
    queue = ready.roundtrips[0].queues[1]
    assert queue.wait.anchor.operation.uid.data == "operand_a"
    assert queue.native.operation.uid.data == "operand_b"
    assert not ready.protocol.destination.implicit_completion  # actual short MAD
    check_plan(plan)


def test_scalar_pipeline_completes_with_direct_controller_lease_and_no_reverse_event():
    synchronization = synchronize_local(scalar_pipeline())
    plan = synchronization.resources
    lease = next(
        lease for lease in plan.leases if lease.protocol.destination.operation.uid.data == "factor"
    )
    assert lease.protocol.source.engine is Engine.MTE2
    assert lease.protocol.destination.engine is Engine.S
    assert lease.width == 2 and not lease.roundtrips and len(lease.controllers) == 1
    assert not any(
        p.source.engine is Engine.S and p.destination.engine is Engine.MTE2
        for p in synchronization.selection.protocols
    )
    assert not any(
        p.source.operation.uid.data == "load" and p.destination.operation.uid.data == "multiply"
        for p in synchronization.selection.protocols
    )
    release = next(
        lease
        for lease in plan.leases
        if lease.protocol.source.operation.uid.data == "multiply"
        and lease.protocol.destination.operation.uid.data == "load"
    )
    assert release.width == 2
    witness = release.roundtrips[0]
    assert witness.peer.source.operation.uid.data == "load"
    assert witness.peer.destination.operation.uid.data == "factor"
    assert isinstance(witness.queues[1], ControllerBridge)
    assert synchronization.resource_work.controller_comparisons == 3
    check_plan(plan)


@pytest.mark.parametrize("source_after_scalar,expected_width", [(False, 2), (True, 1)])
def test_scalar_gate_respects_same_step_source_dispatch_lexical_order(
    source_after_scalar, expected_width
):
    module = scalar_pipeline()
    if source_after_scalar:
        load = next(o for o in module.walk() if isinstance(o, TaskOp) and o.uid.data == "load")
        load.attributes["issue_order"] = IntAttr(4)
    ready = find(selected(module), "load", "factor")
    # For width 1 the next load and old Scalar Wait/native have the same
    # physical step. A load lexically before that Wait is already dispatched.
    assert ready.source.stage + 1 == ready.destination.stage
    assert (ready.source.lexical > ready.destination.lexical) == source_after_scalar
    plan = allocate_local_events((ready,))
    lease = plan.leases[0]
    assert lease.width == expected_width
    assert not lease.roundtrips and len(lease.controllers) == 1
    witness = lease.controllers[0]
    assert witness.next_set.generation_offset == expected_width
    assert witness.dispatch.rank.offset == ready.source.stage + expected_width
    check_plan(plan)


def test_scalar_controller_lease_also_counts_active_notification_ordinals():
    ready = find(selected(scalar_pipeline()), "load", "factor")
    ready = replace(ready, activation=Activation(0, 2, frozenset({0})))
    plan = allocate_local_events((ready,))
    lease = plan.leases[0]
    assert lease.width == 1
    assert lease.controllers[0].successor_delta == 2
    assert lease.ordinal(2) == 1
    check_plan(plan)


def test_scalar_source_does_not_make_an_ordinary_destination_wait_block_controller():
    loop = extract_loop(scalar_pipeline())
    tasks = extract_native_tasks(loop, clock=Clock("g"))
    hazards = bind_hazards(loop, tasks, analyze_dependencies(loop).required)
    members = tuple(
        h
        for h in hazards
        if h.producer.engine is Engine.S and h.consumer.operation.uid.data == "load"
    )
    protocol = LocalProtocol(
        "scalar_source",
        members[0].producer,
        members[0].consumer,
        Activation(2, 1, frozenset({0})),
        LocalMechanism.EVENT,
        members,
    )
    work = LocalResourceWork()
    with pytest.raises(CompileError, match="no consume-to-next-Set roundtrip"):
        allocate_local_events((protocol,), work=work)
    assert work.controller_comparisons == 0
    readiness = tuple(
        h
        for h in hazards
        if h.producer.operation.uid.data == "load" and h.consumer.operation.uid.data == "factor"
    )
    peer = LocalProtocol(
        "scalar_destination",
        readiness[0].producer,
        readiness[0].consumer,
        Activation(0, 1, frozenset({0})),
        LocalMechanism.EVENT,
        readiness,
    )
    work = LocalResourceWork()
    plan = allocate_local_events((protocol, peer), work=work)
    assert work.pair_candidates == 2  # reverse bank and Scalar bank coincide; visit once
    assert plan.leases[0].width == 2 and plan.leases[0].roundtrips
    check_plan(plan)


def test_ordinary_vector_wait_cannot_bridge_to_another_source_engine():
    kernel = Kernel("no_vector_controller_bridge", disjoint_external=True)
    x = kernel.buffer("x", "GM", 160, external=True, initialized=True)
    y = kernel.buffer("y", "GM", 160, external=True)
    a = kernel.buffer("a", "UB", 64)
    kernel.loop(5)
    local = kernel.view(a, 32, stride=32, slots=2)
    kernel.copy("load", kernel.view(x, 32, stride=32), local)
    kernel.scale("scale", local, local, 2.0, stage=1)
    kernel.copy("store", local, kernel.view(y, 32, stride=32), stage=2)
    protocols = selected(kernel.finish())
    release = find(protocols, "store", "load")
    peer = find(protocols, "load", "scale")
    assert peer.source.engine is release.destination.engine
    assert peer.destination.engine is Engine.V
    assert release.source.engine is Engine.MTE3
    work = LocalResourceWork()
    with pytest.raises(CompileError, match=f"{release.name}: no consume-to-next-Set roundtrip"):
        allocate_local_events((release, peer), work=work)
    assert work.pair_candidates == work.controller_comparisons == 0


@pytest.mark.parametrize(
    "recipe,source,destination",
    [(vector_pipeline, "load", "scale"), (cube_pipeline, "operand_b", "mad")],
)
def test_ordinary_vector_or_cube_wait_never_supplies_a_controller_lease(
    recipe, source, destination
):
    ready = find(selected(recipe()), source, destination)
    work = LocalResourceWork()
    with pytest.raises(CompileError, match="no consume-to-next-Set roundtrip"):
        allocate_local_events((ready,), work=work)
    assert work.controller_comparisons == 0


def test_buffer_slots_without_consumption_chain_are_not_a_lease():
    protocols = selected(vector_pipeline(slots=8))
    ready = find(protocols, "load", "scale")
    with pytest.raises(CompileError, match="no consume-to-next-Set roundtrip") as error:
        allocate_local_events((ready,))
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_sparse_channels_count_notifications_instead_of_generation_distance():
    protocols = selected(vector_pipeline())
    ready = find(protocols, "load", "scale")
    release = find(protocols, "scale", "load")
    sparse = tuple(
        replace(p, activation=Activation(p.distance, 2, frozenset({0}))) for p in (ready, release)
    )
    plan = allocate_local_events(sparse)
    assert [lease.width for lease in plan.leases] == [1, 1]
    assert all(lease.roundtrips[0].successor_delta == 2 for lease in plan.leases)
    check_plan(plan)


def test_coupled_residues_are_proved_at_both_true_endpoint_domains():
    protocols = selected(vector_pipeline())
    ready = replace(find(protocols, "load", "scale"), activation=Activation(0, 2, frozenset({0})))
    release = replace(
        find(protocols, "scale", "load"), activation=Activation(2, 3, frozenset({0, 1, 2}))
    )
    plan = allocate_local_events((ready, release))
    assert [lease.width for lease in plan.leases] == [1, 3]
    assert {w.period for lease in plan.leases for w in lease.roundtrips} == {6}
    check_plan(plan)


def test_earliest_generation_cannot_use_a_fictional_negative_peer_source():
    module = vector_pipeline(late_reader=True)
    for operation in module.walk():
        if isinstance(operation, TaskOp) and operation.uid.data in {
            "last_read",
            "postprocess",
            "store",
        }:
            operation.attributes["stage"] = IntAttr(4 if operation.uid.data == "store" else 3)
    protocols = selected(module)
    ready = find(protocols, "load", "scale")
    release = find(protocols, "last_read", "load")
    plan = allocate_local_events((ready, release))
    # Steady-state rank inequalities allow width 1 with peer source g-1.
    # At g=0 that source does not exist, so the uniform lease needs width 2.
    assert [lease.width for lease in plan.leases] == [2, 1]
    assert plan.leases[0].roundtrips[0].peer_set.generation_offset == 0
    check_plan(plan)


def test_independent_pairs_never_form_a_global_product_period():
    protocols = selected(vector_pipeline())
    names = (("load", "scale", 5), ("postprocess", "store", 7))
    selected_pairs = []
    for source, destination, period in names:
        ready = find(protocols, source, destination)
        release = find(
            protocols,
            "scale" if source == "load" else "store",
            "load" if source == "load" else "scale",
        )
        selected_pairs.extend(
            (
                replace(ready, activation=Activation(0, period, frozenset({0}))),
                replace(release, activation=Activation(2, period, frozenset({2}))),
            )
        )
    work = LocalResourceWork()
    plan = allocate_local_events(selected_pairs, work=work)
    assert {w.period for lease in plan.leases for w in lease.roundtrips} == {5, 7}
    assert work.projected_residues == work.lease_witnesses == 4
    assert work.indexed_protocols == work.pair_candidates == 4
    assert [lease.width for lease in plan.leases] == [1, 1, 1, 1]
    check_plan(plan)


def test_bank_index_visits_only_actual_reverse_interface_candidates():
    protocols = tuple(
        replace(protocol, name=f"{prefix}_{protocol.name}")
        for prefix, module in (("scale", vector_pipeline()), ("cube", cube_pipeline()))
        for protocol in selected(module)
    )
    work = LocalResourceWork()
    plan = allocate_local_events(protocols, work=work)
    assert len(plan.assignments) == work.indexed_protocols == 12
    # Scale has four single reverse interfaces. Cube has four L1 events
    # seeing two real reverse peers each, plus four single reverse peers.
    assert work.pair_candidates == 4 + 4 * 2 + 4
    assert work.pair_candidates < work.indexed_protocols**2
    check_plan(plan)


def test_projection_cost_has_an_explicit_limit_without_runtime_N_expansion():
    protocols = selected(vector_pipeline())
    ready = replace(find(protocols, "load", "scale"), activation=Activation(0, 101, frozenset({0})))
    release = replace(
        find(protocols, "scale", "load"), activation=Activation(2, 103, frozenset({2}))
    )
    work = LocalResourceWork()
    with pytest.raises(CompileError, match="projection budget") as error:
        allocate_local_events((ready, release), work=work, projection_limit=16)
    assert error.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert work.projected_residues == 0


def test_missing_peer_residues_cannot_be_extrapolated_from_one_matching_phase():
    protocols = selected(vector_pipeline())
    ready = find(protocols, "load", "scale")
    release = replace(
        find(protocols, "scale", "load"), activation=Activation(2, 17, frozenset({2}))
    )
    with pytest.raises(CompileError, match="no consume-to-next-Set roundtrip") as error:
        allocate_local_events((ready, release))
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_runtime_trip_count_does_not_change_resource_proof_or_work():
    sizes = []
    for count in (2**10, 2**20):
        work = LocalResourceWork()
        plan = allocate_local_events(selected(vector_pipeline(count=count)), work=work)
        sizes.append((tuple((lease.width, len(lease.roundtrips)) for lease in plan.leases), work))
    assert sizes[0] == sizes[1]


def test_bank_reservations_and_eight_ID_capacity_are_exact():
    protocols = selected(vector_pipeline())
    ready, release = find(protocols, "load", "scale"), find(protocols, "scale", "load")
    plan = allocate_local_events((ready, release), reserved={ready.bank: (0,)})
    assert plan.assignments[0].ids == (1, 2)
    assert plan.assignments[1].ids == (0, 1)
    copies = tuple(replace(ready, name=f"ready_{i}") for i in range(4))
    plan = allocate_local_events((*copies, release))
    assert [a.ids for a in plan.assignments[:4]] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    with pytest.raises(CompileError, match="cross-channel lease sharing is not supported") as error:
        allocate_local_events((*copies, replace(ready, name="ready_4"), release))
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_terminal_event_has_at_most_one_notification_and_no_repeat_lease():
    kernel = Kernel("terminal_event_lease", disjoint_external=True)
    x = kernel.buffer("x", "GM", 320, external=True, initialized=True)
    y = kernel.buffer("y", "GM", 32, external=True)
    a = kernel.buffer("a", "UB", 64)
    kernel.loop(10)
    local = kernel.view(a, 32, stride=32, slots=2)
    kernel.copy("load", kernel.view(x, 32, stride=32), local)
    block = kernel.task_block
    boundary = BoundaryOp.create(
        operands=[block.args[0]], attributes={"kind": StringAttr("last")}, regions=[Region(Block())]
    )
    block.add_op(boundary)
    kernel.task_block = boundary.body.block
    kernel.copy("store", local, kernel.view(y, 32), stage=1)
    kernel.task_block.add_op(EndOp())
    kernel.task_block = block
    loop = extract_loop(kernel.finish())
    tasks = extract_native_tasks(loop, clock=Clock("g"))
    hazards = bind_hazards(loop, tasks, analyze_dependencies(loop).required)
    members = tuple(
        h
        for h in hazards
        if h.producer.operation.uid.data == "load" and h.consumer.operation.uid.data == "store"
    )
    protocol = LocalProtocol(
        "terminal",
        members[0].producer,
        members[0].consumer,
        Activation(0, 1, frozenset({0}), consumer_last=True),
        LocalMechanism.EVENT,
        members,
    )
    lease = allocate_local_events((protocol,)).leases[0]
    assert lease.width == 1 and lease.terminal and not lease.roundtrips
    assert lease.ordinal(10**12) == 0


def test_lifetime_proof_does_not_claim_zero_distance_cycle_can_progress():
    protocols = selected(vector_pipeline())
    ready = find(protocols, "load", "scale")
    release = replace(find(protocols, "scale", "load"), activation=Activation(0, 1, frozenset({0})))
    # Both selected Waits precede their matching opposite Sets at g. This is
    # deadlocked, but the conditional consume->nextSet property still holds.
    plan = allocate_local_events((ready, release))
    assert [lease.width for lease in plan.leases] == [1, 1]
    check_plan(plan)


def test_no_old_completion_graph_or_finite_lifetime_cover_is_called(monkeypatch):
    protocols = selected(cube_pipeline())

    def fail(*args, **kwargs):
        raise AssertionError("legacy completion/lifetime query called")

    monkeypatch.setattr("xdsl_npu.loop_sync.completion_graph", fail)
    monkeypatch.setattr("xdsl_npu.events.lifetime_chains", fail)
    monkeypatch.setattr("xdsl_npu.events.pack_event_lifetimes", fail)
    work = LocalResourceWork()
    plan = allocate_local_events(protocols, work=work)
    assert len(plan.leases) == 8 and work.assigned_ids == 16
    assert all(
        queue.native.engine is not Engine.S
        for lease in plan.leases
        for w in lease.roundtrips
        for queue in w.queues
    )

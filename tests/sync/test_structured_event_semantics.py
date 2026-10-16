"""Finite hardware-semantics counterexamples on actual final physical IR.

Manual instructions establish the oracle independently of production planning.
Generated Scale/GEMM cases subsequently check the same model. Small counts and
body repetitions exercise boundaries; they establish no arbitrary-N proof and
execute no tensor arithmetic, CANN runtime or device operation.
"""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IndexType, StringAttr

from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.models.structured_events import EventModelError, check_structured_events
from xdsl_npu import language as N
from xdsl_npu.analysis.structured_memory import collect_native_memory
from xdsl_npu.dialects.structured import ActorDomainOp, NotificationOp
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind


def memory(source):
    return collect_native_memory(lower_native_structure(source).module)


def notify(anchor, kind, source, destination, *, actors=(Actor.AIV0,), event=0):
    block = anchor.parent
    value = None
    if kind is not SyncKind.FENCE:
        constant = arith.ConstantOp.from_int_and_width(event, IndexType())
        block.insert_op_before(constant, anchor)
        value = constant.result
    operation = NotificationOp.build(
        operands=[value],
        attributes={
            "kind": StringAttr(kind.value),
            "participants": ArrayAttr(StringAttr(actor.value) for actor in actors),
            "source": StringAttr(source.value),
            "destination": StringAttr(destination.value),
            "protocol": StringAttr("diagnostic spelling is not token identity"),
        },
    )
    block.insert_op_before(operation, anchor)
    return operation


def empty(*, mixed=False):
    return memory(
        N.from_source(
            """
        def events():
            with N.Kernel(kind="mixed"):
                with N.Cube():
                    pass
                with N.Vector() as sid:
                    pass
    """
            if mixed
            else """
        def events():
            with N.Kernel():
                pass
    """
        )
    )


def test_source_native_ends_are_not_fifo_and_set_effect_does_not_gate_future_admission():
    facts = memory(
        N.from_source("""
        def independent():
            with N.Kernel():
                x=N.alloc_ub((8,))
                y=N.alloc_ub((8,))
                z=N.alloc_ub((8,))
                N.fill(x,1.0)
                N.fill(y,2.0)
                N.fill(z,3.0)
    """)
    )
    last = facts.templates[-1].operation
    send = notify(last, SyncKind.LOCAL_SET, Engine.V, Engine.MTE3)
    notify(last, SyncKind.LOCAL_WAIT, Engine.V, Engine.MTE3)
    facts.module.verify()
    result = check_structured_events(facts)
    first, second, third = result.natives
    effect = next(command for command in result.commands if command.occurrence.operation is send)
    assert not result.happens_before(first.end, second.end)
    assert not result.happens_before(second.end, first.end)
    assert result.happens_before(first.end, effect.end)
    assert result.happens_before(second.end, effect.end)
    assert not result.happens_before(effect.end, third.begin)


@pytest.mark.parametrize("hazard", ["RAW", "WAR", "WAW"])
def test_each_byte_hazard_requires_retirement_before_admission(hazard):
    statements = {
        "RAW": "N.fill(x,1.0)\n        N.mul(x,2.0,out=y)",
        "WAR": "N.fill(x,1.0)\n        N.mul(x,2.0,out=y)\n        N.fill(x,3.0)",
        "WAW": "N.fill(x,1.0)\n        N.fill(x,3.0)",
    }
    source = N.from_source(
        """
def hazards():
    with N.Kernel():
        x=N.alloc_ub((8,))
        y=N.alloc_ub((8,))
        """
        + statements[hazard]
        + "\n"
    )
    facts = memory(source)
    if hazard == "WAR":
        notify(facts.templates[1].operation, SyncKind.FENCE, Engine.V, Engine.V)
    with pytest.raises(EventModelError, match=hazard):
        check_structured_events(facts)
    notify(facts.templates[-1].operation, SyncKind.FENCE, Engine.V, Engine.V)
    facts.module.verify()
    result = check_structured_events(facts)
    assert hazard in {need.kind for need in result.hazards}


def ready_pipeline():
    facts = memory(
        N.from_source("""
        def ready(X:N.Tensor((8,),N.f32),O:N.Tensor((8,),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((8,))
                y=N.alloc_ub((8,))
                N.copy(X,x)
                N.mul(x,2.0,out=y)
                N.copy(y,O)
    """)
    )
    load, compute, store = (template.operation for template in facts.templates)
    send = notify(load.next_op, SyncKind.LOCAL_SET, Engine.MTE2, Engine.V)
    wait = notify(compute, SyncKind.LOCAL_WAIT, Engine.MTE2, Engine.V)
    notify(compute.next_op, SyncKind.LOCAL_SET, Engine.V, Engine.MTE3)
    notify(store, SyncKind.LOCAL_WAIT, Engine.V, Engine.MTE3)
    return facts, send, wait


@pytest.mark.parametrize("mutation", ["delete_wait", "wrong_id", "early_set"])
def test_actual_final_ir_corruption_has_a_finite_counterexample(mutation):
    facts, send, wait = ready_pipeline()
    valid = check_structured_events(facts)
    assert valid.bytes_read and valid.bytes_written and valid.consumptions
    if mutation == "delete_wait":
        wait.parent.erase_op(wait)
    elif mutation == "wrong_id":
        other = arith.ConstantOp.from_int_and_width(1, IndexType())
        wait.parent.insert_op_before(other, wait)
        wait.operands = (other.result,)
    else:
        load = facts.templates[0].operation
        constant = send.event.owner
        constant.parent.detach_op(constant)
        send.parent.detach_op(send)
        load.parent.insert_ops_before((constant, send), load)
    facts.module.verify()
    with pytest.raises(EventModelError, match="event|RAW"):
        check_structured_events(facts)


def test_oracle_never_calls_production_selection_resource_or_progress(monkeypatch):
    from importlib import import_module

    facts, _, _ = ready_pipeline()

    def forbidden(*args, **kwargs):
        raise AssertionError("finite oracle consulted a production proof")

    for module, function in (
        ("xdsl_npu.passes.select_notifications", "select_notifications"),
        ("xdsl_npu.analysis.structured_resources", "allocate_structured_events"),
    ):
        monkeypatch.setattr(import_module(module), function, forbidden)
    resources = import_module("xdsl_npu.analysis.structured_resources")
    monkeypatch.setattr(resources, "prove_progress", forbidden)
    for operation in facts.module.walk():
        if isinstance(operation, NotificationOp):
            operation.protocol = StringAttr("unrelated metadata")
    assert check_structured_events(facts).hazards


@pytest.mark.parametrize("destination,blocked", [(Engine.V, False), (Engine.S, True)])
def test_only_blocking_scalar_wait_stops_its_controller(destination, blocked):
    facts = empty()
    end = facts.module.body.block.first_op.body.block.last_op
    notify(end, SyncKind.LOCAL_WAIT, Engine.MTE2, destination)
    notify(end, SyncKind.LOCAL_SET, Engine.MTE2, destination)
    facts.module.verify()
    if blocked:
        with pytest.raises(EventModelError, match="controller command deadlock"):
            check_structured_events(facts)
    else:
        assert len(check_structured_events(facts).consumptions) == 1


@pytest.mark.parametrize("destination", [Engine.V, Engine.S])
def test_binary_event_consumption_must_precede_reuse_across_actual_body_invocations(destination):
    facts = empty()
    end = facts.module.body.block.first_op.body.block.last_op
    notify(end, SyncKind.LOCAL_SET, Engine.MTE2, destination)
    notify(end, SyncKind.LOCAL_WAIT, Engine.MTE2, destination)
    assert len(check_structured_events(facts).consumptions) == 1
    if destination is Engine.S:
        assert len(check_structured_events(facts, invocations=3).consumptions) == 3
    else:
        with pytest.raises(EventModelError, match="overwritten before actual Wait consumption"):
            check_structured_events(facts, invocations=2)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7])
def test_finite_zero_single_and_short_actual_pipeline_instances(count):
    facts = memory(
        N.from_source("""
        def short(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((8,))
                for t in N.Pipelined(n,stages=3):
                    with N.Stage(2,name="work"):
                        N.fill(x,1.0)
    """)
    )
    task = facts.templates[0].operation
    notify(task, SyncKind.LOCAL_SET, Engine.MTE2, Engine.S)
    notify(task, SyncKind.LOCAL_WAIT, Engine.MTE2, Engine.S)
    notify(task, SyncKind.FENCE, Engine.V, Engine.V)
    facts.module.verify()
    result = check_structured_events(facts, {"n": count})
    assert len(result.natives) == len(result.consumptions) == count


def quorum():
    facts = memory(
        N.from_source("""
        def quorum(X:N.Tensor((16,16),N.f16),Shared:N.Tensor((16,16),N.f16,access="write")):
            with N.Kernel(kind="mixed"):
                x=N.alloc_ub((8,16),N.f16)
                l1=N.alloc_l1((16,16),N.f16)
                with N.Vector() as sid:
                    N.copy(X[sid*8:(sid+1)*8,:],x)
                    N.copy(x,Shared[sid*8:(sid+1)*8,:])
                with N.Cube():
                    N.copy(Shared[:,:],l1)
    """)
    )
    load, store, consume = (template.operation for template in facts.templates)
    actors = (Actor.AIV0, Actor.AIV1)
    notify(load.next_op, SyncKind.LOCAL_SET, Engine.MTE2, Engine.MTE3, actors=actors)
    notify(store, SyncKind.LOCAL_WAIT, Engine.MTE2, Engine.MTE3, actors=actors)
    send = notify(
        store.next_op, SyncKind.CROSS_SET, Engine.MTE3, Engine.MTE2, actors=actors, event=2
    )
    wait = notify(
        consume, SyncKind.CROSS_WAIT, Engine.MTE3, Engine.MTE2, actors=(Actor.AIC,), event=2
    )
    return facts, send, wait


def test_mode2_acquires_actual_end_contributions_from_both_aiv_owners():
    facts, send, wait = quorum()
    facts.module.verify()
    result = check_structured_events(facts)
    contributors = [pair for pair in result.consumptions if pair.wait.occurrence.operation is wait]
    assert {pair.source.occurrence.actor for pair in contributors} == {Actor.AIV0, Actor.AIV1}
    assert all(pair.source.occurrence.operation is send and pair.counter for pair in contributors)
    assert {
        need.source.actor
        for need in result.hazards
        if need.consumer.operation is facts.templates[-1].operation
    } == {Actor.AIV0, Actor.AIV1}


def test_missing_second_aiv_contribution_cannot_be_replaced_by_a_protocol_name():
    facts, send, _ = quorum()
    send.participants = ArrayAttr([StringAttr(Actor.AIV0.value)])
    facts.module.verify()
    with pytest.raises(EventModelError, match="mode-2 quorum"):
        check_structured_events(facts)


@pytest.mark.parametrize("count", [1, 15, 16])
def test_mode2_counter_capacity_counts_unconsumed_effects_per_actual_recipient(count):
    facts = empty(mixed=True)
    cube, vector = (op for op in facts.module.walk() if isinstance(op, ActorDomainOp))
    for _ in range(count):
        notify(
            cube.body.block.last_op,
            SyncKind.CROSS_SET,
            Engine.FIX,
            Engine.MTE2,
            actors=(Actor.AIC,),
            event=3,
        )
        notify(
            vector.body.block.last_op,
            SyncKind.CROSS_WAIT,
            Engine.FIX,
            Engine.MTE2,
            actors=(Actor.AIV0, Actor.AIV1),
            event=3,
        )
    facts.module.verify()
    if count > 15:
        with pytest.raises(EventModelError, match="backlog can exceed 15"):
            check_structured_events(facts)
    else:
        assert len(check_structured_events(facts).consumptions) == 2 * count


@pytest.mark.parametrize("wait_first", [False, True])
def test_cross_wait_blocks_the_actual_controller_across_different_engines(wait_first):
    facts = empty(mixed=True)
    cube, vector = (op for op in facts.module.walk() if isinstance(op, ActorDomainOp))
    for domain, actors, source, destination, outgoing, incoming in (
        (cube, (Actor.AIC,), Engine.FIX, Engine.MTE2, 0, 1),
        (vector, (Actor.AIV0, Actor.AIV1), Engine.V, Engine.MTE2, 1, 0),
    ):
        for kind in (
            (SyncKind.CROSS_WAIT, SyncKind.CROSS_SET)
            if wait_first
            else (SyncKind.CROSS_SET, SyncKind.CROSS_WAIT)
        ):
            notify(
                domain.body.block.last_op,
                kind,
                source,
                destination,
                actors=actors,
                event=outgoing if kind is SyncKind.CROSS_SET else incoming,
            )
    facts.module.verify()
    if wait_first:
        with pytest.raises(EventModelError, match="controller command deadlock"):
            check_structured_events(facts)
    else:
        assert len(check_structured_events(facts).consumptions) == 4


@pytest.fixture(
    scope="module",
    params=[scale.specialize(B=64), batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1)],
)
def generated(request):
    from xdsl_npu.passes.structured_sync import synchronize_structure

    # One actual compiled IR is interpreted at every small runtime count.
    return synchronize_structure(request.param)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 8])
def test_generated_final_ir_preserves_finite_access_versions_and_event_consumption(
    generated, count
):
    plan = generated
    result = check_structured_events(plan.memory, {"tiles": count}, invocations=2)
    assert len(result.natives) == 2 * count * len(plan.memory.templates)
    assert result.consumptions
    if count:
        assert result.bytes_read and result.bytes_written and result.hazards


@pytest.mark.parametrize("mutation", ["delete_wait", "wrong_id", "early_set"])
def test_generated_actual_waits_and_retirement_captures_are_necessary(mutation):
    from xdsl_npu.passes.structured_sync import synchronize_structure

    plan = synchronize_structure(scale.specialize(B=64))
    check_structured_events(plan.memory, {"tiles": 3})
    wait = next(
        op
        for op in plan.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.LOCAL_WAIT.value
        and op.source.data == Engine.MTE2.value
        and op.destination.data == Engine.V.value
    )
    if mutation == "delete_wait":
        wait.parent.erase_op(wait)
    elif mutation == "wrong_id":
        constant = arith.ConstantOp.from_int_and_width(6, IndexType())
        wait.parent.insert_op_before(constant, wait)
        wait.operands = (constant.result,)
    else:
        send = next(
            op
            for op in plan.module.walk()
            if isinstance(op, NotificationOp)
            and op.kind.data == SyncKind.LOCAL_SET.value
            and op.source.data == Engine.MTE2.value
            and op.destination.data == Engine.V.value
        )
        source = plan.memory.templates[0].operation
        # Preserve the ID's defining SCF/SSA scope and move that complete
        # guarded instruction before the producer's real execution point.
        outer = send
        while outer.parent is not source.parent:
            outer = outer.parent_op()
        block = source.parent
        original = tuple(block.ops)
        positions = {operation: index for index, operation in enumerate(original)}
        moving = {outer}

        def scalar_dependencies(operation):
            for nested in operation.walk():
                for operand in nested.operands:
                    owner = operand.owner
                    if (
                        isinstance(owner, tuple(arith.Arith.operations))
                        and owner.parent is block
                        and positions[owner] > positions[source]
                        and owner not in moving
                    ):
                        moving.add(owner)
                        scalar_dependencies(owner)

        scalar_dependencies(outer)
        ordered = [operation for operation in original if operation in moving]
        for operation in ordered:
            block.detach_op(operation)
        block.insert_ops_before(ordered, source)
    plan.module.verify()
    with pytest.raises(EventModelError, match="event|RAW|WAR|WAW"):
        check_structured_events(plan.memory, {"tiles": 3})

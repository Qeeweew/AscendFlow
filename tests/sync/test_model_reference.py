"""Finite reference checks across native-preserving notification materialization.

The fixtures lower real DSL, clone its unsynchronized native IR, and materialize
explicit hardware instructions. No selector, allocator or progress proof chooses
the expected execution. This remains a bounded model with trusted native rules.
"""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import Placement, SyncAction
from ascendflow.dialects.structured import AddressOp, NotificationOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import EventModelError, check_structured_events


def snapshots(source):
    memory = collect_native_memory(
        lower_native_structure(N.from_source(source, disjoint_external=True)).module
    )
    reference = collect_native_memory(memory.module.clone())
    return memory, reference


def action(anchor, placement, kind, source, destination, participants, event=0):
    return SyncAction(
        anchor,
        placement,
        participants,
        kind,
        source,
        destination,
        literal(event),
        protocols=("manual finite reference fixture",),
    )


def complete_each_native(memory):
    actions = []
    for template in memory.templates:
        # Scalar consumes each completed native before dispatching the next.
        # This manual fixture tests the oracle, not a production schedule.
        for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT):
            actions.append(
                action(
                    template.operation,
                    Placement.AFTER,
                    kind,
                    template.spec.engine,
                    Engine.S,
                    (Actor.AIV0,),
                )
            )
    materialize_notifications(memory, actions)


def test_captured_native_prefix_does_not_invent_fifo_between_distinct_local_set_effects():
    memory, reference = snapshots('''
def independent(X:N.Tensor((2,64),N.f32)):
    with N.Kernel(1,kind="vector"):
        a=N.alloc_ub((64,),N.f32)
        b=N.alloc_ub((64,),N.f32)
        N.copy(X[0,:],a)
        N.copy(X[1,:],b)
''')
    actions = [
        action(t.operation, Placement.AFTER, SyncKind.LOCAL_SET, Engine.MTE2, Engine.V, (Actor.AIV0,), i)
        for i, t in enumerate(memory.templates)
    ]
    actions.extend(
        action(memory.templates[-1].operation, Placement.AFTER, SyncKind.LOCAL_WAIT, Engine.MTE2, Engine.V, (Actor.AIV0,), i)
        for i in range(2)
    )
    materialize_notifications(memory, actions)
    checked = check_structured_events(collect_native_memory(memory.module), logical_reference=reference)
    sends = [c for c in checked.commands if c.kind == SyncKind.LOCAL_SET.value]
    assert len(sends) == 2
    assert all(checked.happens_before(c.end, sends[1].end) for c in checked.natives)
    assert not checked.happens_before(sends[0].end, sends[1].end)
    assert not checked.happens_before(sends[1].end, sends[0].end)


def local_pipeline():
    memory, reference = snapshots("""
        def stream(X:N.Tensor((n,64),N.f32),
                   Y:N.Tensor((n,2,64),N.f32,access="write"),n:N.int64):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for t in N.Pipelined(n):
                    N.copy(X[t,:],x)
                    N.mul(x,2.0,out=y)
                    if t < n:
                        N.copy(y,Y[t,0,:])
""")
    complete_each_native(memory)
    memory.module.verify()
    return memory, reference


@pytest.mark.parametrize("count", [0, 1, 3])
def test_materialized_notifications_preserve_independent_native_reference(count):
    memory, reference = local_pipeline()
    result = check_structured_events(
        memory, {"n": count}, logical_reference=reference, invocations=2
    )
    assert len(result.natives) == 6 * count
    assert len(result.consumptions) == 6 * count
    assert all(
        command.occurrence.operation in {template.operation for template in memory.templates}
        for command in result.natives
    )
    assert all(
        template.operation not in {native.occurrence.operation for native in result.natives}
        for template in reference.templates
    )


@pytest.mark.parametrize("count", [0, 1])
def test_deleted_store_is_rejected_even_when_the_supplied_loop_is_empty(count):
    memory, reference = local_pipeline()
    store = memory.templates[-1].operation
    store.parent.erase_op(store)
    memory.module.verify()
    # A same-IR oracle cannot know that the output store was required.
    check_structured_events(memory, {"n": count})
    with pytest.raises(EventModelError, match="reference native UID set changed"):
        check_structured_events(memory, {"n": count}, logical_reference=reference)


def test_in_bounds_output_address_change_is_rejected_against_reference():
    memory, reference = local_pipeline()
    address = memory.templates[-1].operation.refs[-1].owner
    assert isinstance(address, AddressOp)
    displacement = arith.ConstantOp.from_int_and_width(64 * 4, IndexType())
    shifted = arith.AddiOp(address.offset, displacement.result)
    address.parent.insert_ops_before((displacement, shifted), address)
    address.operands = tuple(
        shifted.result if index == 1 else value for index, value in enumerate(address.operands)
    )
    memory.module.verify()
    check_structured_events(memory, {"n": 1})
    with pytest.raises(EventModelError, match="changed an actual native address"):
        check_structured_events(memory, {"n": 1}, logical_reference=reference)


def test_changed_actual_native_guard_is_rejected_against_reference():
    memory, reference = local_pipeline()
    store = memory.templates[-1].operation
    conditional = store.parent_op()
    assert isinstance(conditional, scf.IfOp)
    false = arith.ConstantOp.from_int_and_width(0, 1)
    conditional.parent.insert_op_before(false, conditional)
    conditional.operands = (false.result,)
    memory.module.verify()
    check_structured_events(memory, {"n": 1})
    with pytest.raises(EventModelError, match="changed native execution instances"):
        check_structured_events(memory, {"n": 1}, logical_reference=reference)


def test_actual_geometry_cannot_borrow_stale_admitted_footprints():
    memory, reference = local_pipeline()
    address = memory.templates[-1].operation.refs[-1].owner
    assert isinstance(address, AddressOp)
    address.nbytes = IntAttr(32 * 4)
    memory.module.verify()
    with pytest.raises(EventModelError, match="changed its admitted native geometry"):
        check_structured_events(memory, {"n": 1}, logical_reference=reference)


def test_reference_requires_unique_native_identity_and_a_separate_snapshot():
    memory, reference = local_pipeline()
    with pytest.raises(EventModelError, match="independent snapshot"):
        check_structured_events(memory, {"n": 0}, logical_reference=memory)
    memory.templates[-1].operation.uid = memory.templates[0].operation.uid
    with pytest.raises(EventModelError, match="ambiguous logical reference identity"):
        check_structured_events(memory, {"n": 0}, logical_reference=reference)


def test_reference_preserves_read_versions_when_final_native_order_changes():
    memory, reference = snapshots("""
        def versions(Y:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,1.0)
                N.mul(x,2.0,out=y)
                N.fill(x,3.0)
                N.copy(y,Y)
    """)
    _, read, overwrite, _ = (template.operation for template in memory.templates)
    block = overwrite.parent
    # Move the pure address definition as well, preserving SSA dominance.
    # The test changes native order, not the validity of address evaluation.
    address = overwrite.refs[0].owner
    offset = address.offset.owner
    assert isinstance(offset, arith.ConstantOp)
    block.detach_op(offset)
    block.insert_op_before(offset, read)
    block.detach_op(address)
    block.insert_op_before(address, read)
    block.detach_op(overwrite)
    block.insert_op_before(overwrite, read)
    # Materialize completion after the reordered tasks. Every byte and UID is
    # still present; a final-IR logical traversal would bind a different value.
    complete_each_native(memory)
    memory.module.verify()
    check_structured_events(memory)
    with pytest.raises(EventModelError, match="unordered physical WAR"):
        check_structured_events(memory, logical_reference=reference)


def mixed_handoff():
    memory, reference = snapshots("""
        def handoff(X:N.Tensor((16,16),N.f16),
                    Shared:N.Tensor((16,16),N.f16,access="write")):
            with N.Kernel(kind="mixed"):
                x=N.alloc_ub((8,16),N.f16)
                l1=N.alloc_l1((16,16),N.f16)
                with N.Vector() as sid:
                    N.copy(X[sid*8:(sid+1)*8,:],x)
                    N.copy(x,Shared[sid*8:(sid+1)*8,:])
                with N.Cube():
                    N.copy(Shared[:,:],l1)
    """)
    load, store, consume = (template.operation for template in memory.templates)
    vector = (Actor.AIV0, Actor.AIV1)
    materialize_notifications(
        memory,
        (
            action(load, Placement.AFTER, SyncKind.LOCAL_SET, Engine.MTE2, Engine.MTE3, vector),
            action(store, Placement.BEFORE, SyncKind.LOCAL_WAIT, Engine.MTE2, Engine.MTE3, vector),
            action(store, Placement.AFTER, SyncKind.CROSS_SET, Engine.MTE3, Engine.MTE2, vector, 2),
            action(
                consume,
                Placement.BEFORE,
                SyncKind.CROSS_WAIT,
                Engine.MTE3,
                Engine.MTE2,
                (Actor.AIC,),
                2,
            ),
        ),
    )
    memory.module.verify()
    return memory, reference


def test_reference_retains_one_aiv_template_two_private_ub_instances_and_quorum():
    memory, reference = mixed_handoff()
    result = check_structured_events(memory, logical_reference=reference)
    load = memory.templates[0].operation
    copies = [command for command in result.natives if command.occurrence.operation is load]
    assert len(copies) == 2
    assert {command.occurrence.actor for command in copies} == {Actor.AIV0, Actor.AIV1}
    assert {command.spans[-1].actor for command in copies} == {Actor.AIV0, Actor.AIV1}
    assert len({command.spans[-1].root for command in copies}) == 1
    assert {pair.source.occurrence.actor for pair in result.consumptions if pair.counter} == {
        Actor.AIV0,
        Actor.AIV1,
    }


def test_reference_does_not_replace_a_missing_aiv_event_contribution():
    memory, reference = mixed_handoff()
    send = next(
        operation
        for operation in memory.module.walk()
        if isinstance(operation, NotificationOp) and operation.kind.data == SyncKind.CROSS_SET.value
    )
    send.participants = ArrayAttr((StringAttr(Actor.AIV0.value),))
    memory.module.verify()
    with pytest.raises(EventModelError, match="mode-2 quorum"):
        check_structured_events(memory, logical_reference=reference)

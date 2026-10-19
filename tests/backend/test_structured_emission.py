"""structured emission contracts."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith

from ascendflow import language as N
from ascendflow.analysis.structured_schedule import (
    Placement,
    SyncAction,
    SyncKind,
)
from ascendflow.backend.structured import EmissionWork, _emit_owned, emit_structure_debug
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump
from ascendflow.dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from ascendflow.passes.executable import NativeExecution
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.control import NESTED
from tests.support.asc_trace import forbid_analysis, memory


@pytest.mark.parametrize("source", [scale, batched_matmul])
def test_owned_emission_only_consumes_final_ir_and_preserves_it(source, monkeypatch):
    plan = synchronize_structure(source)
    before = dump(plan.module)
    logical = dump(plan.structure.module)
    assert plan.module is not plan.structure.module
    forbid_analysis(monkeypatch)
    work = EmissionWork()
    asc = _emit_owned(plan, work=work)
    assert dump(plan.module) == before
    assert dump(plan.structure.module) == logical
    assert work.native_templates == work.native_prints == len(plan.executable.natives)
    assert all(count == 1 for count in work.task_prints.values())
    assert work.actions_printed == sum(isinstance(op, NotificationOp) for op in plan.module.walk())
    assert work.output_bytes == len(asc.encode())
    assert "#error" not in asc and 'extern "C" int launch_' in asc
    assert "asc_sync_notify(" in asc and "asc_sync_wait(" in asc
    assert "PIPE_ALL" not in asc  # Final drains are already explicit root EXIT actions.
    assert "npu_size_$" not in asc


def test_terminal_emission_does_not_borrow_logical_control_addresses_or_native_fields(monkeypatch):
    from xdsl.dialects.builtin import FloatAttr, IndexType, f32

    plan = synchronize_structure(scale)
    expected = _emit_owned(plan)
    logical = plan.admission.memory
    assert any(isinstance(op, PipelineLoopOp) for op in logical.module.walk())
    assert not any(isinstance(op, PipelineLoopOp) for op in plan.module.walk())
    old = next(t.operation for t in logical.templates if t.spec.native == "muls")
    old.scalar = FloatAttr(99.0, f32)
    address = next(op for op in logical.module.walk() if isinstance(op, AddressOp))
    offset = arith.ConstantOp.from_int_and_width(65536, IndexType())
    address.parent.insert_op_before(offset, address)
    address.operands = (address.source, offset.result, *address.origins, *address.extents)

    def forbidden(*args, **kwargs):
        pytest.fail("terminal emission queried a logical coordinate")

    monkeypatch.setattr(logical.control.environment, "expression", forbidden)
    forbid_analysis(monkeypatch)
    assert _emit_owned(plan) == expected


@pytest.mark.parametrize("field", ["native", "params", "address_count"])
def test_terminal_binding_rejects_changes_to_admitted_instruction_geometry(field):
    from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

    _, facts = memory(scale)
    template = facts.templates[0]
    operation = template.operation.clone()
    operation.attributes[field] = {
        "native": StringAttr("exp"),
        "params": ArrayAttr([IntAttr(64)]),
        "address_count": IntAttr(1),
    }[field]
    with pytest.raises(CompileError, match="admitted instruction specification"):
        NativeExecution.from_template(template, operation)


def test_fa_structure_debug_prints_each_template_once_and_is_not_executable(monkeypatch):
    structure, facts = memory(attention_target)
    forbid_analysis(monkeypatch)
    work = EmissionWork()
    asc = emit_structure_debug(facts, work=work)
    assert work.native_templates == work.native_prints == structure.work["native_templates"]
    assert work.native_templates == 52
    assert all(count == 1 for count in work.task_prints.values())
    assert work.output_bytes == len(asc.encode())
    assert asc.count("if ASC_IS_AIV") == 1 and asc.count("if ASC_IS_AIC") == 1
    assert '#error "Unsynchronized structured intermediate: debug output only"' in asc
    assert "launch_" not in asc and "PIPE_ALL" not in asc and "$" not in asc
    assert asc.count("asc_set_l0c_copy_params(") == 1
    assert asc.index("asc_set_l0c_copy_params(") < asc.index("if ASC_IS_AIV")


def test_nested_print_work_depends_on_templates_instead_of_iteration_counts():
    outputs = []
    for count in (1024, 1_048_576):
        structure, facts = memory(N.from_source(NESTED).specialize(PARTS=count))
        work = EmissionWork()
        asc = emit_structure_debug(facts, work=work)
        assert work.native_templates == work.native_prints == 7
        assert len(structure.loops) == 3
        assert asc.count("for (int64_t npu_step_") == sum(
            "npu.pipeline_clock" in op.attributes for op in facts.module.walk()
        )
        assert all(prints == 1 for prints in work.task_prints.values())
        outputs.append(work.output_bytes)
    assert abs(outputs[1] - outputs[0]) < 100


def test_foreign_memory_inconsistent_native_mapping_and_unbound_address_symbols_are_rejected():
    _, facts = memory(scale)
    with pytest.raises(CompileError, match="owned synchronization"):
        _emit_owned(facts)
    plan = synchronize_structure(scale)
    with pytest.raises(CompileError, match="native instruction correspondence"):
        replace(plan.executable, module=plan.module.clone())
    invalid = SyncAction(
        facts.templates[0].operation,
        Placement.ENTRY,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        literal(0),
    )
    with pytest.raises(CompileError, match="invocation boundary placement"):
        materialize_notifications(facts, (invalid,))
    template = facts.templates[0]
    address = replace(
        template.accesses[0].address,
        offset=facts.control.loops[next(iter(facts.control.loops))].ordinal,
    )
    # Retained proof expressions cannot choose the executable address.
    changed = replace(
        template,
        accesses=(replace(template.accesses[0], address=address), *template.accesses[1:]),
    )
    altered = replace(facts, templates=(changed, *facts.templates[1:]))
    assert emit_structure_debug(altered) == emit_structure_debug(facts)
    # A packet moved out of its real SSA scope is rejected at that actual use.
    root = facts.module.body.block.first_op
    template.operation.parent.detach_op(template.operation)
    root.body.block.insert_op_before(template.operation, root.body.block.last_op)
    with pytest.raises(CompileError, match="unbound structured SSA scalar"):
        emit_structure_debug(facts)


def test_final_address_ssa_and_notification_operations_are_the_executable_input(monkeypatch):
    from xdsl.dialects.builtin import FloatAttr, IndexType, StringAttr, f32

    _, facts = memory(scale)
    task = facts.templates[0].operation
    action = SyncAction(
        task,
        Placement.BEFORE,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        literal(3),
    )
    materialize_notifications(facts, (action,))
    notification = next(op for op in facts.module.walk() if isinstance(op, NotificationOp))
    address = task.refs[0].owner
    assert isinstance(address, AddressOp)
    constant = arith.ConstantOp.from_int_and_width(128, IndexType())
    address.parent.insert_op_before(constant, address)
    address.operands = (address.source, constant.result, *address.origins, *address.extents)
    event = arith.ConstantOp.from_int_and_width(6, IndexType())
    notification.parent.insert_op_before(event, notification)
    notification.operands = (event.result,)
    notification.kind = StringAttr(SyncKind.LOCAL_SET.value)
    compute = next(t for t in facts.templates if t.spec.native == "muls")
    compute.operation.scalar = FloatAttr(3.0, f32)
    # The scalar literal is executable IR, not a second value in analysis facts.
    facts.module.verify()

    def forbidden(*args, **kwargs):
        pytest.fail("backend interpreted a retained scalar proof expression")

    monkeypatch.setattr(facts.control.environment, "expression", forbidden)
    asc = emit_structure_debug(facts)
    assert "npu_buffer_0 + (128)" in asc
    assert "float(3.0)" in asc
    assert "asc_sync_notify(PIPE_MTE2, PIPE_V, event_t(6));" in asc
    assert "asc_sync_wait(" not in asc
    notification.parent.erase_op(notification)
    assert "asc_sync_notify(" not in emit_structure_debug(facts)


def test_replaced_native_address_operand_is_read_from_final_ir():
    from xdsl.dialects.builtin import IndexType

    _, facts = memory(scale)
    task = facts.templates[0].operation
    original = task.refs[0].owner
    constant = arith.ConstantOp.from_int_and_width(256, IndexType())
    replacement = original.clone()
    replacement.operands = (
        replacement.source,
        constant.result,
        *replacement.origins,
        *replacement.extents,
    )
    original.parent.insert_ops_before((constant, replacement), original)
    task.operands = (replacement.results[0], *tuple(task.refs)[1:])
    facts.module.verify()
    # This is debug printing of deliberately changed IR, not proof reuse for
    # execution. The backend must not print the detached retained operand.
    asc = emit_structure_debug(facts)
    assert "npu_buffer_0 + (256)" in asc


def test_async_native_and_notification_operations_survive_generic_scalar_passes():
    from xdsl.context import Context
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

    from ascendflow.dialects.physical import TaskOp

    _, facts = memory(
        N.from_source("""
        def effects():
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                N.fill(x,1.0)
    """)
    )
    actions = tuple(
        SyncAction(
            template.operation,
            Placement.BEFORE,
            template.participants,
            SyncKind.FENCE,
            Engine.V,
            Engine.V,
        )
        for template in facts.templates
        for _ in range(2)
    )
    materialize_notifications(facts, actions)
    effects = tuple(op for op in facts.module.walk() if isinstance(op, (TaskOp, NotificationOp)))
    context = Context()
    CanonicalizePass().apply(context, facts.module)
    CommonSubexpressionElimination().apply(context, facts.module)
    CanonicalizePass().apply(context, facts.module)
    facts.module.verify()
    assert (
        tuple(op for op in facts.module.walk() if isinstance(op, (TaskOp, NotificationOp)))
        == effects
    )

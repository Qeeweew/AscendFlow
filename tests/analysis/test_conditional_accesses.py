"""Conditional numerical reads retain complete native effects and real clocks.

These tests stop at native memory/admission. Small integer checks independently
exercise the primitive's reset predicate, without compiling synchronization,
expanding native bodies, or running device code.
"""

from textwrap import indent

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from ascendflow import language as N
from ascendflow.analysis.native import AccessMode, ConditionalRead, NativeContract
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import PipelineLoopOp, PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Engine
from ascendflow.target.native import NativeReadReset, native_read_resets


def _structure(body=None, *, ring=False):
    destination = "c0[k%2,:,:]" if ring else "c0"
    if body is None:
        body = f"N.matmul(aa,bb,accum={destination},initialize=(k==0))"
    shape = "(2,16,16)" if ring else "(16,16)"
    source = f"""
def reduction(A:N.Tensor((n,16,16),N.f16),B:N.Tensor((n,16,16),N.f16),n:N.int64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c({shape},N.f32)
        for k in N.Pipelined(n,stages=3):
            with N.Stage(0):
                av=N.copy(A[k,:,:],a1)
                bv=N.copy(B[k,:,:],b1)
            with N.Stage(1):
                aa=N.copy(av,a0)
                bb=N.copy(bv,b0,transpose=True)
            with N.Stage(2):
{indent(body, " " * 16)}
"""
    return lower_native_structure(N.from_source(source, disjoint_external=True))


def _retarget(structure, period):
    """Construct an actual cycle primitive for its native-contract unit test."""
    task = next(
        op
        for op in structure.module.walk()
        if isinstance(op, TaskOp) and op.native.data in {"mad", "mad_first"}
    )
    task.native = StringAttr("mad_cycle")
    if len(task.refs) == 3:
        stage = task.parent_op()
        while not isinstance(stage, PipelineStageOp):
            stage = stage.parent_op()
        task.operands = (*task.refs, stage.body.block.args[0])
    task.params = ArrayAttr(
        IntAttr(value) for value in (*tuple(p.data for p in task.params)[:3], period)
    )
    return task


def _admit(structure):
    memory = collect_native_memory(structure.module)
    return admit_structured_memory(structure, memory, admit_memory_bounds(memory))


def _accumulator(memory):
    template = next(item for item in memory.templates if item.contract.read_resets)
    return template, template.accesses[2]


@pytest.mark.parametrize(
    "native,params,expected",
    [
        ("mad_first", (16, 16, 16), (NativeReadReset(2, None),)),
        ("mad_cycle", (16, 16, 16, 3), (NativeReadReset(2, 3),)),
        ("mad", (16, 16, 16, 0), ()),
        ("mad", (16, 16, 16, 1), ()),
        ("mad16", (1,), ()),
        ("fix", (16, 16), ()),
    ],
)
def test_target_rules_belong_to_primitives_without_optional_name_markers(native, params, expected):
    assert native_read_resets(native, params) == expected


@pytest.mark.parametrize("period", (None, 1, 2, 3))
def test_conditional_old_value_read_uses_the_real_stage_ordinal_and_keeps_raw_rw(period):
    structure = _structure()
    if period is not None:
        _retarget(structure, period)
    before = dump(structure.module)
    memory = collect_native_memory(structure.module)
    template, accumulator = _accumulator(memory)
    condition = accumulator.read_condition
    assert isinstance(condition, ConditionalRead)
    assert isinstance(condition.loop, PipelineLoopOp)
    assert condition.ordinal is memory.control.loops[condition.loop].ordinal
    assert accumulator.mode is AccessMode.READ_WRITE
    assert template.contract.modes == ("R", "R", "RW")
    assert all(access.read_condition is None for access in template.accesses[:2])
    assert condition.read.lhs is condition.kill.lhs
    assert condition.read.rhs == condition.kill.rhs
    for ordinal in range(7):
        expected_kill = ordinal == 0 if period is None else ordinal % period == 0
        values = {condition.ordinal.args[0]: ordinal}
        assert condition.kill.evaluate(values) == expected_kill
        assert condition.read.evaluate(values) == (not expected_kill)
    assert dump(structure.module) == before
    assert len(memory.templates) == 5


@pytest.mark.parametrize("period", (None, 1, 2, 3))
def test_invariant_accumulator_is_initialized_by_its_own_reset_without_erasing_later_reads(period):
    structure = _structure()
    if period is not None:
        _retarget(structure, period)
    admission = _admit(structure)
    template, accumulator = _accumulator(admission.memory)
    proof = next(
        row for row in admission.reads if row.template is template and row.access is accumulator
    )
    assert proof.initialization is accumulator.read_condition
    assert proof.writers == () and not proof.external
    assert not hasattr(proof, "first_clear")
    assert accumulator.mode.reads and accumulator.mode.writes
    assert admission.work["expanded_iterations"] == 0


@pytest.mark.parametrize("period", (None, 2, 3))
def test_a_reset_on_one_ring_slot_cannot_initialize_other_read_slots(period):
    structure = _structure(ring=True)
    if period is not None:
        _retarget(structure, period)
    with pytest.raises(CompileError, match="initialized writer coverage"):
        _admit(structure)


def test_reset_at_every_ordinal_writes_each_ring_slot_without_an_old_value_read():
    structure = _structure(ring=True)
    _retarget(structure, 1)
    admission = _admit(structure)
    _, accumulator = _accumulator(admission.memory)
    assert any(row.initialization is accumulator.read_condition for row in admission.reads)


@pytest.mark.parametrize(
    "body",
    [
        "if k>0:\n    N.matmul(aa,bb,accum=c0,initialize=(k==0))",
        "for inner in N.serial(k):\n    N.matmul(aa,bb,accum=c0,initialize=True)",
    ],
)
def test_later_activation_cannot_borrow_a_reset_that_never_executed(body):
    structure = _structure(body)
    # Retarget the constant-clear inner body to a first/cycle primitive. Its
    # generation is still the enclosing actual stage, not its serial index.
    if "N.serial" in body:
        _retarget(structure, 2)
    with pytest.raises(CompileError, match="initialized writer coverage"):
        _admit(structure)


def test_uniform_activation_keeps_reset_and_conditional_read_in_the_same_domain():
    admission = _admit(_structure("if n>1:\n    N.matmul(aa,bb,accum=c0,initialize=(k==0))"))
    _, accumulator = _accumulator(admission.memory)
    assert any(row.initialization is accumulator.read_condition for row in admission.reads)


def test_missing_reset_can_instead_read_a_proved_real_prior_writer():
    admission = _admit(
        _structure(
            "N.matmul(aa,bb,accum=c0,initialize=True)\n"
            "if k>0:\n    N.matmul(aa,bb,accum=c0,initialize=(k==0))"
        )
    )
    template, accumulator = _accumulator(admission.memory)
    proof = next(
        row for row in admission.reads if row.template is template and row.access is accumulator
    )
    assert proof.initialization is None and len(proof.writers) == 1


def test_nested_serial_coordinate_cannot_replace_the_actual_pipeline_reset_coordinate():
    structure = _structure(
        "for inner in N.serial(2):\n    N.matmul(aa,bb,accum=c0,initialize=True)"
    )
    task = _retarget(structure, 3)
    memory = collect_native_memory(structure.module)
    _, accumulator = _accumulator(memory)
    condition = accumulator.read_condition
    loops = memory.control.frames[task].loops()
    assert len(loops) == 2 and condition.loop is loops[0].operation
    assert condition.ordinal is loops[0].ordinal and condition.ordinal != loops[1].ordinal
    stage = task.parent_op()
    while not isinstance(stage, PipelineStageOp):
        stage = stage.parent_op()
    assert stage.parent_op() is condition.loop and stage.offset.data == 2


def test_reset_rule_requires_an_actual_pipeline_stage_not_just_a_serial_clock():
    structure = _structure()
    task = next(
        op
        for op in structure.module.walk()
        if isinstance(op, TaskOp) and op.native.data == "mad_first"
    )
    stage = task.parent_op()
    # A private unit fixture can remove the stage association without forging
    # a symbolic loop-name marker. No native body is copied.
    stage.body.block.detach_op(task)
    root = structure.module.body.block.first_op
    root.body.block.insert_op_before(task, root.body.block.last_op)
    task.stage = IntAttr(0)
    with pytest.raises(CompileError, match="actual pipeline stage ordinal"):
        collect_native_memory(structure.module)


def test_static_rmw_has_no_reset_certificate_and_still_needs_defined_input():
    structure = _structure("N.matmul(aa,bb,accum=c0,initialize=False)")
    memory = collect_native_memory(structure.module)
    template = next(item for item in memory.templates if item.contract.native == "mad")
    assert template.accesses[2].read_condition is None
    with pytest.raises(CompileError, match="initialized writer coverage"):
        _admit(structure)


def test_conditional_rule_cannot_be_attached_to_a_non_rw_operand():
    with pytest.raises(CompileError, match="complete read/write operand"):
        NativeContract("mad_first", Engine.M, ("R", "R", "W"), (16, 16, 16), 0.0)

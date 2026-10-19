"""mask packets contracts."""

from __future__ import annotations

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Engine
from ascendflow.target.packets import MaskStep
from tests.fixtures.masks import (
    _data_predicate_program,
    _masked_fill,
    _ordered,
    _task,
)
from tests.models.structured_events import check_structured_events


def test_memory_collection_rejects_a_tampered_iteration_recipe():
    from xdsl.dialects.builtin import StringAttr

    from ascendflow.mask import ITERATION_CHUNKS

    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    plan = parse(result.plan_ir)
    task = next(
        operation
        for operation in plan.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_fill"
    )
    assert task.attributes["mask_iteration"].data == "rows"
    task.attributes["mask_iteration"] = StringAttr(ITERATION_CHUNKS)
    with pytest.raises(CompileError, match="iteration recipe"):
        collect_native_memory(plan)


def test_packed_compare_and_not_are_declared_vector_packets():
    result = compile(
        N.from_source(
            """
            def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
                with N.Kernel(1, kind="vector"):
                    work=N.alloc_ub((8,128),N.f32)
                    temp=N.alloc_ub((8,128),N.f32)
                    bits=N.alloc_mask((8,128))
                    nbits=N.alloc_mask((8,128))
                    N.copy(X,work)
                    p=N.compare(work,0.5,kind="gt",out=bits)
                    n=N.mask_not(p,out=nbits)
                    N.fill(work,-1.0,mask=n,scratch=temp)
                    N.copy(work,Y)
            """,
            disjoint_external=True,
        ),
        timeout_s=120,
    )
    memory = collect_native_memory(result.plan)
    zero = _task(memory, "mask_zero")
    compare = _task(memory, "compare_scalar_gt")
    negative = _task(memory, "mask_not")
    selection = _task(memory, "select_masked_update")
    templates = {t.operation: t for t in memory.templates}
    zero_template, compare_template = templates[zero], templates[compare]
    negative_template = templates[negative]
    # Padding is defined by a real count-mode Vector Dup, not a hidden store.
    assert zero_template.spec.engine is Engine.V
    assert zero_template.spec.modes == ("W",)
    assert zero_template.spec.packet.closed_mask
    assert zero_template.spec.packet.each_count_call
    assert zero_template.spec.packet.mask_steps == (
        MaskStep.SET_COUNTER,
        MaskStep.SET_COUNT,
        MaskStep.VECTOR_READ,
        MaskStep.RESTORE_NORMAL,
    )
    (out,) = zero_template.accesses
    assert out.address.geometry.total_nbytes == 16 * 32
    # Packed compare is still the audited mask-free Vector packet.
    assert compare_template.spec.engine is Engine.V
    assert compare_template.spec.modes == ("R", "RW")
    assert compare_template.spec.packet.mask_steps == ()
    assert compare_template.spec.packet.internal_fences == ()
    # Packed NOT is a real count-mode Vector native, matching its packet.
    assert negative_template.spec.engine is Engine.V
    assert negative_template.spec.modes == ("R", "W")
    assert negative_template.spec.packet.each_count_call
    assert negative_template.spec.packet.mask_steps == zero_template.spec.packet.mask_steps
    for template in (compare_template, negative_template):
        (last,) = template.accesses[-1:]
        assert last.address.geometry.total_nbytes == 16 * 32
    asc = result.asc
    # No scalar UB stores or hidden word building survive in the emitted code.
    assert "*(__ubuf__ uint64_t*)" not in asc
    assert "npu_bits" not in asc
    assert "UINT64_C(0)" not in asc
    zero_call = asc.index("asc_duplicate((__ubuf__ uint16_t*)")
    compare_call = asc.index("asc_gt_scalar(")
    fence = asc.index("asc_sync_pipe(PIPE_V)", zero_call)
    not_call = asc.index("asc_not(")
    assert zero_call < fence < compare_call < not_call
    # An independent finite event model proves the needed orderings.
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    assert ordered(zero, compare)
    assert ordered(compare, negative)
    assert ordered(negative, selection)
    hazards = {
        (hazard.kind, hazard.source.operation, hazard.consumer.operation)
        for hazard in execution.hazards
    }
    assert ("WAW", zero, compare) in hazards
    assert ("RAW", compare, negative) in hazards
    assert ("RAW", negative, selection) in hazards


def test_materialized_not_has_unary_dependency_and_distinct_storage():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                nbits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                n=N.mask_not(p,out=nbits)
                N.fill(work,7.0,mask=n,scratch=temp)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=120)
    memory = collect_native_memory(result.plan)
    negative = _task(memory, "mask_not")
    template = next(t for t in memory.templates if t.operation is negative)
    assert len(template.accesses) == 2
    assert template.accesses[0].mode.reads and template.accesses[1].mode.writes
    result_ir = parse(result.source_ir)
    combine = [op for op in result_ir.walk() if op.name == "npu_source.mask_combine"]
    assert len(combine) == 1 and combine[0].kind.data == "not"
    assert combine[0].rhs is None and combine[0].rhs_storage is None
    assert len(combine[0].dependency_inputs) == 1
    with pytest.raises(CompileError, match="distinct predicate storage"):
        compile(
            N.from_source(
                """
                def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
                    with N.Kernel(1, kind="vector"):
                        work=N.alloc_ub((8,128),N.f32)
                        bits=N.alloc_mask((8,128))
                        N.copy(X,work)
                        p=N.compare(work,0.0,kind="gt",out=bits)
                        N.mask_not(p,out=bits)
                        N.copy(work,Y)
                """,
                disjoint_external=True,
            ),
            timeout_s=60,
        )


def test_selected_word_recipe_and_repeat_policy_are_recorded_before_sync():
    affine = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    lane = compile(_masked_fill("((cols // 8) % 2) == 0"), timeout_s=120)
    for result, word in ((affine, "affine64"), (lane, "lane64")):
        memory = collect_native_memory(result.plan)
        tasks = [t for t in memory.templates if t.spec.native == "mask_fill"]
        assert len(tasks) == 1
        attributes = tasks[0].operation.attributes
        assert attributes["mask_word"].data == word
        assert attributes["mask_repeat"].data == "full_runs_255"
    assert "npu_mask_monotone(" in affine.asc
    assert "npu_mask_word |= (UINT64_C(1) << npu_lane)" in lane.asc


def test_compare_padding_comes_from_a_declared_vector_task_and_pipe_fence():
    result = compile(_data_predicate_program(), timeout_s=180)
    asc = result.asc
    zero = asc.index("asc_duplicate((__ubuf__ uint16_t*)")
    compare = asc.index("asc_gt_scalar(")
    fence = asc.index("asc_sync_pipe(PIPE_V)", zero)
    assert zero < fence < compare
    assert "*(__ubuf__ uint64_t*)" not in asc
    memory = collect_native_memory(result.plan)
    zero_template = next(t for t in memory.templates if t.spec.native == "mask_zero")
    template = next(t for t in memory.templates if t.spec.native == "compare_scalar_gt")
    assert zero_template.spec.engine is Engine.V
    assert zero_template.spec.modes == ("W",)
    assert zero_template.accesses[0].address.geometry.total_nbytes == 16 * 32
    # The padded logical Tile is declared as a complete write by the recipe.
    assert template.accesses[-1].mode.writes
    assert template.accesses[-1].address.geometry.total_nbytes == 16 * 32


def test_standalone_compare_cannot_authorize_uninitialized_packed_reads():
    source = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                N.copy(X,other)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.where(p,work,other,out=temp)
                N.copy(temp,Y)
        """,
        disjoint_external=True,
    )
    structure = lower_native_structure(source)
    # The declared mask_zero writer satisfies ordinary read admission.
    memory = collect_native_memory(structure.module)
    admit_structured_memory(structure, memory, admit_memory_bounds(memory))
    # Remove only that initializer; the partial compare's RW destination read
    # must not become a fresh whole-block initialization fact by itself.
    zero = next(
        operation
        for operation in structure.module.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_zero"
    )
    zero.detach()
    partial = collect_native_memory(structure.module)
    partial_template = next(
        template for template in partial.templates if template.spec.native == "compare_scalar_gt"
    )
    assert partial_template.spec.modes == ("R", "RW")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        admit_structured_memory(structure, partial, admit_memory_bounds(partial))


def test_memory_collection_rejects_a_tampered_word_recipe():
    from xdsl.dialects.builtin import StringAttr

    from ascendflow.mask import WORD_AFFINE, WORD_LANE

    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    plan = parse(result.plan_ir)
    task = next(
        operation
        for operation in plan.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_fill"
    )
    # Tamper with the value the admitted predicate did not select.
    recorded = task.attributes["mask_word"].data
    assert recorded in {WORD_AFFINE, WORD_LANE}
    task.attributes["mask_word"] = StringAttr(WORD_LANE if recorded == WORD_AFFINE else WORD_AFFINE)
    with pytest.raises(CompileError, match="word recipe"):
        collect_native_memory(plan)

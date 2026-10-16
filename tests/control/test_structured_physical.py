"""Retained physical control, SSA addresses, and conservative async effects."""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, StringAttr, i32
from xdsl.ir import Block, Region
from xdsl.traits import Pure, RecursiveMemoryEffect
from xdsl.utils.exceptions import VerifyException

from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.loops import LoopScheduleAttr
from xdsl_npu.dialects.physical import (
    BufferOp,
    EndOp,
    KernelOp,
    RuntimeSizeOp,
    TaskOp,
    task_attrs,
)
from xdsl_npu.dialects.structured import AddressOp, PipelineLoopOp, PipelineStageOp
from xdsl_npu.dialects.types import AsyncEffects, TaskType, ref_type


def index(value):
    return arith.ConstantOp.from_int_and_width(value, IndexType())


def stage(offset=1, *, arg_types=None):
    return PipelineStageOp.build(
        attributes={"offset": IntAttr(offset), "label": StringAttr(f"stage{offset}")},
        regions=[
            Region(
                Block(
                    [EndOp()],
                    arg_types=[IndexType(), IndexType()] if arg_types is None else arg_types,
                )
            )
        ],
    )


def program(*, stages=None, lower=2, upper=11, step=3, horizon=2):
    lower, upper, step = index(lower), index(upper), index(step)
    root = BufferOp.build(
        operands=[None],
        attributes={
            "sym_name": StringAttr("buffer"),
            "nbytes": IntAttr(4096),
            "external": IntAttr(0),
            "initialized": IntAttr(0),
        },
        result_types=[ref_type("UB", "f32")],
    )
    stages = (stage(),) if stages is None else stages
    loop = PipelineLoopOp.build(
        operands=[lower, upper, step],
        attributes={"schedule": LoopScheduleAttr(IntAttr(horizon))},
        regions=[Region(Block([*stages, EndOp()]))],
    )
    kernel = KernelOp.build(
        attributes={
            "sym_name": StringAttr("structured"),
            "target": StringAttr("dav-2201"),
            "core": StringAttr("aiv"),
            "disjoint_external": IntAttr(1),
            "blocks": IntAttr(1),
        },
        regions=[Region(Block([root, lower, upper, step, loop, EndOp()]))],
    )
    return ModuleOp([kernel]), loop, stages[0] if stages else None, root


def address(
    root,
    offset,
    *,
    nbytes=64,
    rows=1,
    row_stride=0,
    result_type=None,
    origins=(),
    extents=(),
    widths=None,
    axes=None,
    displacement=None,
):
    attrs = {"nbytes": IntAttr(nbytes), "rows": IntAttr(rows), "row_stride": IntAttr(row_stride)}
    if widths is not None:
        attrs["widths"] = ArrayAttr(IntAttr(width) for width in widths)
    if axes is not None:
        attrs["axes"] = ArrayAttr(IntAttr(axis) for axis in axes)
    if displacement is not None:
        attrs["displacement"] = IntAttr(displacement)
    return AddressOp.build(
        operands=[root, offset, origins, extents],
        attributes=attrs,
        result_types=[root.type if result_type is None else result_type],
    )


def test_actual_structured_ir_round_trips_without_losing_ssa_or_native_effects():
    module, loop, work, root = program()
    scale = index(64)
    offset = arith.MuliOp(work.body.block.args[1], scale.result)
    view = address(root.ref, offset.result, rows=2, row_stride=96)
    fill = TaskOp.create(
        operands=[view.ref], result_types=[TaskType()], attributes=task_attrs("fill", "fill", 1)
    )
    work.body.block.insert_ops_before([scale, offset, view, fill], work.body.block.last_op)
    before = tuple(module.walk())
    module.verify()
    assert tuple(module.walk()) == before
    assert not loop.results and not work.results
    assert view.source is root.ref and view.offset is offset.result
    assert fill.refs[0] is view.ref
    text = dump(module)
    restored = parse(text)
    assert dump(restored) == text
    restored_loop = next(op for op in restored.walk() if isinstance(op, PipelineLoopOp))
    restored_stage = next(op for op in restored.walk() if isinstance(op, PipelineStageOp))
    restored_address = next(op for op in restored.walk() if isinstance(op, AddressOp))
    assert restored_stage.parent_op() is restored_loop
    assert restored_address.source is restored_address.source.owner.ref
    assert isinstance(restored_address.source.owner, BufferOp)
    assert restored_address.offset.owner.lhs is restored_stage.body.block.args[1]
    assert restored_address.ref.type == restored_address.source.type


@pytest.mark.parametrize("upper", [1, 2, 3, 8, 11])
def test_nonzero_lower_and_step_bind_stage_local_ordinal_and_iv(upper):
    stages = tuple(stage(offset) for offset in (2, 0, 1))
    module, loop, _, _ = program(stages=stages, lower=2, upper=upper, step=3)
    module.verify()
    count = len(range(2, upper, 3))
    actual = []
    for phase in loop.schedule.phases:
        interval = phase.interval(count)
        if interval is None:
            continue
        for physical_step in range(*interval):
            for rank, work in enumerate(stages):
                ordinal = physical_step - work.offset.data
                if 0 <= ordinal < count:
                    actual.append((physical_step, rank, ordinal, 2 + ordinal * 3))
    expected = sorted(
        (ordinal + work.offset.data, rank, ordinal, iv)
        for rank, work in enumerate(stages)
        for ordinal, iv in enumerate(range(2, upper, 3))
    )
    assert actual == expected
    assert len(actual) == count * len(stages)
    assert len(tuple(loop.body.block.ops)) == len(stages) + 1


def test_nested_pipeline_retains_independent_real_parent_and_child_coordinates():
    module, loop, outer, root = program()
    child = stage(0)
    nested = PipelineLoopOp.build(
        operands=[outer.body.block.args[1], loop.upper, loop.step],
        attributes={"schedule": LoopScheduleAttr(IntAttr(0))},
        regions=[Region(Block([child, EndOp()]))],
    )
    combined = arith.AddiOp(outer.body.block.args[1], child.body.block.args[1])
    view = address(root.ref, combined.result)
    child.body.block.insert_ops_before([combined, view], child.body.block.last_op)
    outer.body.block.insert_op_before(nested, outer.body.block.last_op)
    module.verify()
    restored = parse(dump(module))
    assert sum(isinstance(op, PipelineLoopOp) for op in restored.walk()) == 2
    assert nested.lower is outer.body.block.args[1]
    assert combined.lhs is outer.body.block.args[1]
    assert combined.rhs is child.body.block.args[1]
    assert combined.lhs is not combined.rhs


def test_scf_serial_carry_keeps_its_own_yield_and_unmodified_control():
    module, loop, _, _ = program()
    block = loop.parent
    block.detach_op(loop)
    initial = index(0)
    block.insert_op_before(initial, block.last_op)
    body = Block(arg_types=[IndexType(), IndexType()])
    update = arith.AddiOp(body.args[1], body.args[0])
    body.add_ops([loop, update, scf.YieldOp(update.result)])
    serial = scf.ForOp(loop.lower, loop.upper, loop.step, [initial.result], body)
    block.insert_op_before(serial, block.last_op)
    module.verify()
    assert len(serial.iter_args) == len(serial.results) == 1
    assert serial.body.block.last_op.arguments[0] is update.result
    assert "npu.loop_schedule" not in serial.attributes
    assert dump(parse(dump(module))) == dump(module)


@pytest.mark.parametrize("step", [0, -1])
def test_pipeline_step_requires_a_positive_integer_constant(step):
    module, _, _, _ = program(step=step)
    with pytest.raises(VerifyException, match="positive integer constant"):
        module.verify()


def test_positive_runtime_step_cannot_silently_replace_the_constant_contract():
    module, loop, _, _ = program()
    runtime = RuntimeSizeOp.build(
        attributes={"sym_name": StringAttr("step"), "minimum": IntAttr(1), "multiple": IntAttr(1)},
        result_types=[IndexType()],
    )
    loop.parent.insert_op_before(runtime, loop)
    loop.operands = [loop.lower, loop.upper, runtime.value]
    with pytest.raises(VerifyException, match="positive integer constant"):
        module.verify()


@pytest.mark.parametrize("offset", [-1, 3])
def test_stage_offset_must_be_inside_its_actual_loop_schedule(offset):
    module, _, _, _ = program(stages=(stage(offset),))
    with pytest.raises(VerifyException, match="outside its loop schedule"):
        module.verify()


@pytest.mark.parametrize("types", [(), (IndexType(),), (IndexType(), i32), (IndexType(),) * 3])
def test_stage_requires_two_index_block_arguments(types):
    module, _, _, _ = program(stages=(stage(arg_types=types),))
    with pytest.raises(VerifyException, match="index ordinal and iv"):
        module.verify()


def test_stage_must_be_a_direct_loop_child():
    module, _, work, _ = program()
    nested = stage(0)
    work.body.block.insert_op_before(nested, work.body.block.last_op)
    with pytest.raises(VerifyException, match="directly to a pipeline loop"):
        module.verify()


def test_loop_accepts_only_stages_and_an_end_without_its_own_coordinate_argument():
    module, loop, _, _ = program()
    loop.body.block.insert_arg(IndexType(), 0)
    with pytest.raises(VerifyException, match="no block arguments"):
        module.verify()
    module, loop, _, _ = program()
    loop.body.block.insert_op_before(index(1), loop.body.block.last_op)
    with pytest.raises(VerifyException, match="only pipeline stages"):
        module.verify()


@pytest.mark.parametrize("which", ["loop", "stage"])
def test_structured_bodies_must_use_existing_end_terminator(which):
    module, loop, work, _ = program()
    block = loop.body.block if which == "loop" else work.body.block
    block.erase_op(block.last_op)
    block.add_op(scf.YieldOp())
    with pytest.raises(VerifyException):
        module.verify()


@pytest.mark.parametrize(
    "nbytes,rows,stride", [(0, 1, 0), (-1, 1, 0), (64, 0, 0), (64, 1, -1), (64, 2, 32)]
)
def test_invalid_address_geometry_is_rejected(nbytes, rows, stride):
    module, _, work, root = program()
    view = address(root.ref, work.body.block.args[1], nbytes=nbytes, rows=rows, row_stride=stride)
    work.body.block.insert_op_before(view, work.body.block.last_op)
    with pytest.raises(VerifyException, match="positive row width/count and non-overlapping rows"):
        module.verify()


def test_address_must_keep_original_root_and_reference_type():
    module, _, work, root = program()
    view = address(root.ref, work.body.block.args[1])
    alias = address(view.ref, work.body.block.args[1])
    work.body.block.insert_ops_before([view, alias], work.body.block.last_op)
    with pytest.raises(VerifyException, match="original BufferOp allocation root"):
        module.verify()
    module, _, work, root = program()
    changed = address(root.ref, work.body.block.args[1], result_type=ref_type("UB", "f16"))
    work.body.block.insert_op_before(changed, work.body.block.last_op)
    with pytest.raises(VerifyException, match="preserve its allocation reference type"):
        module.verify()


def test_address_well_formedness_does_not_certify_dynamic_allocation_bounds():
    module, _, work, root = program()
    runtime = RuntimeSizeOp.build(
        attributes={
            "sym_name": StringAttr("offset"),
            "minimum": IntAttr(0),
            "multiple": IntAttr(1),
        },
        result_types=[IndexType()],
    )
    module.body.block.first_op.body.block.insert_op_before(runtime, root)
    view = address(root.ref, runtime.value)
    work.body.block.insert_op_before(view, work.body.block.last_op)
    module.verify()  # No upper bound for offset was proved here.
    assert view.offset is runtime.value
    assert view.results[0].type != TaskType()


def test_axis_metadata_retains_real_ssa_and_source_geometry_after_roundtrip():
    module, _, work, root = program()
    extent0, extent1, pitch, itemsize = index(32), index(24), index(96), index(4)
    row = arith.MuliOp(work.body.block.args[0], pitch.result)
    column = arith.MuliOp(work.body.block.args[1], itemsize.result)
    offset = arith.AddiOp(row.result, column.result)
    view = address(
        root.ref,
        offset.result,
        rows=2,
        row_stride=96,
        origins=work.body.block.args,
        extents=(extent0.result, extent1.result),
        widths=(2, 16),
        axes=(0, 1),
        displacement=0,
    )
    work.body.block.insert_ops_before(
        [extent0, extent1, pitch, itemsize, row, column, offset, view], work.body.block.last_op
    )
    module.verify()
    restored = parse(dump(module))
    assert dump(restored) == dump(module)
    restored_view = next(op for op in restored.walk() if isinstance(op, AddressOp))
    restored_stage = next(op for op in restored.walk() if isinstance(op, PipelineStageOp))
    assert tuple(restored_view.origins) == tuple(restored_stage.body.block.args)
    assert [value.owner.value.value.data for value in restored_view.extents] == [32, 24]
    assert [width.data for width in restored_view.widths] == [2, 16]
    assert [axis.data for axis in restored_view.axes] == [0, 1]
    assert restored_view.displacement.data == 0
    assert restored_view.offset.owner.lhs.owner.lhs is restored_view.origins[0]
    assert restored_view.offset.owner.rhs.owner.lhs is restored_view.origins[1]


@pytest.mark.parametrize("missing", ["origins", "extents", "widths", "axes", "displacement"])
def test_address_axis_metadata_must_be_complete_or_entirely_absent(missing):
    module, loop, work, root = program()
    metadata = {
        "origins": work.body.block.args,
        "extents": (loop.upper, loop.upper),
        "widths": (1, 16),
        "axes": (1,),
        "displacement": 0,
    }
    metadata[missing] = () if missing in {"origins", "extents"} else None
    view = address(root.ref, work.body.block.args[1], **metadata)
    work.body.block.insert_op_before(view, work.body.block.last_op)
    with pytest.raises(VerifyException, match="complete or entirely absent"):
        module.verify()


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("extents", (), "complete"),
        ("widths", (16,), "widths"),
        ("widths", (1, 0), "widths"),
        ("axes", (1, 0), "ordered unique"),
        ("axes", (0, 0), "ordered unique"),
        ("axes", (2,), "original rank"),
        ("axes", (-1,), "original rank"),
        ("displacement", -1, "displacement"),
    ],
)
def test_address_axis_metadata_has_consistent_static_rank_and_displacement(
    field, value, diagnostic
):
    module, loop, work, root = program()
    metadata = {
        "origins": work.body.block.args,
        "extents": (loop.upper, loop.upper),
        "widths": (1, 16),
        "axes": (1,),
        "displacement": 0,
    }
    metadata[field] = value
    view = address(root.ref, work.body.block.args[1], **metadata)
    work.body.block.insert_op_before(view, work.body.block.last_op)
    with pytest.raises(VerifyException, match=diagnostic):
        module.verify()


def test_axis_extents_require_one_real_index_ssa_per_original_axis():
    module, loop, work, root = program()
    view = address(
        root.ref,
        work.body.block.args[1],
        origins=work.body.block.args,
        extents=(loop.upper,),
        widths=(1, 16),
        axes=(1,),
        displacement=0,
    )
    work.body.block.insert_op_before(view, work.body.block.last_op)
    with pytest.raises(VerifyException, match="complete or entirely absent"):
        module.verify()


def test_subview_displacement_can_retain_the_original_axis_descriptors():
    module, loop, work, root = program()
    bias = index(32)
    offset = arith.AddiOp(work.body.block.args[1], bias.result)
    view = address(
        root.ref,
        offset.result,
        nbytes=32,
        origins=work.body.block.args,
        extents=(loop.upper, loop.upper),
        widths=(1, 16),
        axes=(1,),
        displacement=32,
    )
    work.body.block.insert_ops_before([bias, offset, view], work.body.block.last_op)
    module.verify()
    assert view.source is root.ref
    assert tuple(view.origins) == tuple(work.body.block.args)
    assert tuple(view.extents) == (loop.upper, loop.upper)
    assert view.displacement.data == 32
    assert view.nbytes.data == 32
    assert dump(parse(dump(module))) == dump(module)


def test_control_never_marks_async_work_pure_or_returns_completion_handles():
    module, loop, work, root = program()
    view = address(root.ref, work.body.block.args[1])
    fill = TaskOp.create(
        operands=[view.ref], result_types=[TaskType()], attributes=task_attrs("fill", "fill", 1)
    )
    work.body.block.insert_ops_before([view, fill], work.body.block.last_op)
    module.verify()
    assert loop.has_trait(RecursiveMemoryEffect)
    assert work.has_trait(RecursiveMemoryEffect)
    assert fill.has_trait(AsyncEffects)
    assert not view.has_trait(AsyncEffects)
    assert all(not op.has_trait(Pure) for op in (loop, work, view, fill))
    assert not loop.results and not work.results

"""AST source semantics and physical mixed-protocol regression boundaries."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from ascendflow import language as N
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.source import AllocationOp, DomainOp, PipelineOp, StageOp, ViewOp
from ascendflow.dialects.types import TileType
from ascendflow.frontend import FrontendError
from tests.fixtures.source_programs import cv_source, vector_source


def test_typed_copy_requires_an_explicit_rounding_conversion():
    definition = N.from_source("""
        def invalid(X:N.Tensor((1,64),N.f32)):
            with N.Kernel():
                output=N.alloc_ub((64,),N.f16)
                for task in N.Pipelined(1,stages=1):
                    N.copy(X[task,:],output)
    """)
    with pytest.raises(FrontendError, match="explicit cast"):
        definition.to_ir()


def test_source_axis_bounds_are_checked_before_flattening_addresses():
    definition = N.from_source("""
        def invalid(X:N.Tensor((2,64),N.f32)):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    N.copy(X[task,64:128],x)
    """)
    with pytest.raises(FrontendError, match="logical axis"):
        definition.to_ir()


def test_incompatible_domain_is_rejected_without_silently_dropping_work():
    definition = N.from_source("""
        def invalid(X:N.Tensor((1,64),N.f32)):
            with N.Kernel(kind="vector"):
                x=N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    with N.Cube():
                        N.copy(X[task,:],x)
    """)
    with pytest.raises(VerifyException, match="kernel kind"):
        definition.to_ir()


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_ast_stages_retain_one_task_coordinate_and_explicit_slot_axes(count):
    source = parse(dump(vector_source(count)))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert pipeline.count.data == count
    assert [op.offset.data for op in stages] == [0, 1, 2]
    assert len(pipeline.body.block.args) == 1
    allocations = [
        op
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "UB"
    ]
    assert all(op.versions.data == 1 and op.ring.data == 0 for op in allocations)
    assert {tuple(n.data for n in op.ref.type.shape) for op in allocations} == {(2, 64)}


def test_ast_cv_regions_share_stage_one_without_merging_their_semantics():
    source = parse(dump(cv_source()))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert [op.offset.data for op in stages] == [0, 1, 1, 2]
    assert [op.label.data for op in stages] == ["qk", "vector", "pv", "output"]
    assert [op.kind.data for op in source.walk() if isinstance(op, DomainOp)] == [
        "cube",
        "vector",
        "cube",
        "vector",
    ]

    def uses_coordinate(value):
        return value is pipeline.body.block.args[0] or any(
            uses_coordinate(operand) for operand in getattr(value.owner, "operands", ())
        )

    assert all(
        any(uses_coordinate(origin) for origin in op.origins)
        for op in source.walk()
        if isinstance(op, ViewOp)
        and op.source.owner.sym_name.data in {"scores", "probabilities", "pv"}
    )
    assert "npu.task" not in dump(source)


def test_stage_depth_does_not_silently_change_explicit_buffer_slots():
    source = vector_source()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    pipeline.attributes["stages"] = IntAttr(2)
    with pytest.raises(VerifyException, match="stage horizon"):
        source.verify()


def test_region_yield_types_are_verified_after_source_roundtrip():
    source = parse(dump(vector_source()))
    stage = next(op for op in source.walk() if isinstance(op, StageOp))
    stage.results[0]._type = TileType(ArrayAttr([IntAttr(32)]), StringAttr("f32"))
    with pytest.raises(VerifyException, match="yield types"):
        source.verify()

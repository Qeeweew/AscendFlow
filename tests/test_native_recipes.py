"""Native recipes bind real SSA and insertion blocks without a stream adapter."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr, f32
from xdsl.ir import Block

from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import BufferOp, TaskOp, ViewOp, task_attrs
from xdsl_npu.dialects.source import ElementOp, ReduceOp, ScalarReadOp, ScaleOp
from xdsl_npu.dialects.types import TaskType, ref_type
from xdsl_npu.lowering import lower_source
from xdsl_npu.passes import native_recipes
from xdsl_npu.passes.native_recipes import (
    NativeRead,
    RecipeContext,
    lower_element,
    lower_reduce,
    lower_scalar_read,
    lower_scale,
)


class Insertion:
    """A test-owned ordinary scf block; no production actor or control builder."""

    def __init__(self):
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        one = arith.ConstantOp.from_int_and_width(1, IndexType())
        self.block = Block(arg_types=[IndexType()])
        self.loop = scf.ForOp(zero, one, one, [], self.block)
        self.module = ModuleOp([zero, one, self.loop])
        self.bindings = {}
        self.tasks = []
        self.context = RecipeContext(self.view, self.subview, self.emit)

    def bind(self, value, nbytes, *, rows=1, row_stride=0):
        buffer = BufferOp.create(
            result_types=[ref_type("UB", "f32")],
            attributes={
                "sym_name": StringAttr(f"buffer{len(self.bindings)}"),
                "nbytes": IntAttr(nbytes if rows == 1 else (rows - 1) * row_stride + nbytes),
                "external": IntAttr(0),
                "initialized": IntAttr(1),
            },
        )
        self.module.body.block.insert_op_before(buffer, self.loop)
        ref = self.subview(buffer.ref, nbytes, rows=rows, row_stride=row_stride)
        self.bindings[value] = ref
        return ref

    def view(self, value, *, matrix=False):
        return self.bindings[value]

    def subview(self, source, nbytes, *, offset=0, rows=1, row_stride=0):
        view = ViewOp.create(
            operands=[source, self.block.args[0]],
            result_types=[source.type],
            attributes={
                "offset": IntAttr(offset),
                "stride": IntAttr(0),
                "modulus": IntAttr(0),
                "nbytes": IntAttr(nbytes),
                "tail_nbytes": IntAttr(0),
                "rows": IntAttr(rows),
                "row_stride": IntAttr(row_stride),
            },
        )
        self.block.add_op(view)
        return view.ref

    def emit(self, suffix, native, *refs, params=(), scalar=0.0, order_delta=0):
        task = TaskOp.create(
            operands=refs,
            result_types=[TaskType(), f32] if native == "scalar_load" else [TaskType()],
            attributes=task_attrs(f"native{len(self.tasks)}{suffix}", native, 0, params, scalar)
            | {"issue_order": IntAttr(len(self.tasks))},
        )
        self.block.add_op(task)
        self.tasks.append(task)
        return task

    def finish(self):
        self.block.add_op(scf.YieldOp())
        self.module.verify()
        assert not any(
            op.name in {"npu.stream", "npu.call", "npu.once"} for op in self.module.walk()
        )


def test_broadcast_result_identifies_statistic_reader_and_destination_operand():
    source = N.from_source("""
        def broadcast():
            with N.Kernel():
                x=N.alloc_ub((8,64))
                statistic=N.alloc_ub((8,))
                scratch=N.alloc_ub((8,8))
                out=N.alloc_ub((8,64))
                for i in N.Pipelined(1,stages=1):
                    N.sub(x,statistic[:,None],out=out,scratch=scratch)
    """).to_ir()
    op = next(op for op in source.walk() if isinstance(op, ElementOp))
    target = Insertion()
    value = target.bind(op.inputs[0], 256, rows=8, row_stride=256)
    statistic = target.bind(op.inputs[1], 32)
    scratch = target.bind(op.scratch, 32, rows=8, row_stride=32)
    destination = target.bind(op.destination, 256, rows=8, row_stride=256)
    result = lower_element(op, target.context)
    broadcast, arithmetic = result.tasks
    assert tuple(broadcast.refs) == (statistic, scratch)
    assert tuple(arithmetic.refs) == (value, scratch, destination)
    assert result.definition is arithmetic and result.definition_ref == 2
    assert [(r.source_input, r.operation, r.ref_index) for r in result.readers] == [
        (0, arithmetic, 0),
        (1, broadcast, 0),
    ]
    assert all(task.parent is target.block for task in result.tasks)
    target.finish()


def test_block8_result_keeps_disjoint_workspace_planes_and_original_input_reader():
    source = N.from_source("""
        def reduction():
            with N.Kernel():
                x=N.alloc_ub((8,1024))
                workspace=N.alloc_ub((1152,))
                out=N.alloc_ub((8,))
                for i in N.Pipelined(1,stages=1):
                    N.reduce_sum(x,axis=1,out=out,scratch=workspace,
                        order="block8_then_packed_merge")
    """).to_ir()
    op = next(op for op in source.walk() if isinstance(op, ReduceOp))
    target = Insertion()
    value = target.bind(op.source, 4096, rows=8, row_stride=4096)
    workspace = target.bind(op.scratch, 4608)
    destination = target.bind(op.destination, 32)
    result = lower_reduce(op, target.context)
    partial, compact, merge = result.tasks
    first, second = partial.refs[1].owner, compact.refs[1].owner
    assert first.source is second.source is workspace
    assert (first.offset.data, first.nbytes.data, first.rows.data) == (0, 512, 8)
    assert (second.offset.data, second.nbytes.data, second.rows.data) == (4096, 64, 8)
    assert partial.refs[0] is value and compact.refs[0] is first.ref
    assert tuple(merge.refs) == (second.ref, destination)
    assert result.definition is merge and result.definition_ref == 1
    assert [(r.source_input, r.operation, r.ref_index) for r in result.readers] == [(0, partial, 0)]
    target.finish()


def test_scalar_result_is_the_actual_ssa_operand_of_its_consumer():
    source = N.from_source("""
        def scalar():
            with N.Kernel():
                statistic=N.alloc_ub((1,))
                x=N.alloc_ub((64,))
                out=N.alloc_ub((64,))
                for i in N.Pipelined(1,stages=1):
                    factor=N.load_scalar(statistic)
                    N.mul(x,factor,out=out)
    """).to_ir()
    read = next(op for op in source.walk() if isinstance(op, ScalarReadOp))
    scale = next(op for op in source.walk() if isinstance(op, ScaleOp))
    target = Insertion()
    statistic = target.bind(read.source, 4)
    value = target.bind(scale.source, 256)
    destination = target.bind(scale.destination, 256)
    captured = lower_scalar_read(read, target.context)
    result = lower_scale(scale, target.context, factor=captured.scalar_value)
    assert captured.definition.value is captured.scalar_value
    assert captured.definition_ref is None
    assert tuple(captured.definition.refs) == (statistic,)
    assert tuple(result.definition.refs) == (value, destination, captured.scalar_value)
    assert result.readers[0].operation is result.definition
    assert result.readers[0].ref_index == 0
    target.finish()


@pytest.mark.parametrize("symbolic", [False, True])
def test_every_native_reader_of_one_logical_input_is_checked(monkeypatch, symbolic):
    source = N.from_source("""
        def readers():
            with N.Kernel():
                x=N.alloc_ub((64,))
                y=N.alloc_ub((64,))
                for i in N.Pipelined(2,stages=1):
                    value=N.fill(x,1.0)
                    N.add(value,1.0,out=y)
    """).to_ir()
    original = native_recipes.lower_element

    def overwritten_second_read(op, context):
        result = original(op, context)
        if op.kind.data != "add_scalar":
            return result
        source, destination = result.definition.refs
        overwrite = context.emit("_overwrite", "fill", source, scalar=0.0, order_delta=1)
        late = context.emit("_late", "adds", source, destination, scalar=1.0, order_delta=2)
        return replace(
            result,
            tasks=(*result.tasks, overwrite, late),
            definition=late,
            readers=(*result.readers, NativeRead(0, late, 0)),
        )

    monkeypatch.setattr(native_recipes, "lower_element", overwritten_second_read)
    with pytest.raises(CompileError, match="overwritten before.*late"):
        lower_source(source, symbolic=symbolic)

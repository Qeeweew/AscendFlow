"""Explicit fixed-issue builder; there are no user event IDs or barriers."""

from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr, f32
from xdsl.ir import Block, Region, SSAValue

from .ir import (
    BufferOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    StreamOp,
    TaskOp,
    TaskType,
    ViewOp,
    ref_type,
    task_attrs,
)


class Kernel:
    def __init__(self, name: str, core="aiv", *, disjoint_external=False, blocks=1):
        self.block = Block()
        self.op = KernelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "core": StringAttr(core),
                "target": StringAttr("dav-2201"),
                "disjoint_external": IntAttr(int(disjoint_external)),
                "blocks": IntAttr(blocks),
            },
            regions=[Region(self.block)],
        )
        self.module = ModuleOp([self.op])
        self.stream: StreamOp | None = None
        self.task_block: Block | None = None
        self.num_tasks = 0

    def buffer(
        self, name, bank, nbytes, dtype="f32", *, layout="linear", external=False, initialized=False
    ) -> SSAValue:
        op = BufferOp.create(
            result_types=[ref_type(bank, dtype, layout)],
            attributes={
                "sym_name": StringAttr(name),
                "nbytes": IntAttr(nbytes),
                "external": IntAttr(int(external)),
                "initialized": IntAttr(int(initialized)),
            },
        )
        self.block.add_op(op)
        op.ref.name_hint = name
        return op.ref

    def init_fixpipe(self):
        self.block.add_op(FixConfigOp())

    def loop(self, count: int):
        if self.stream is not None:
            raise ValueError("v0.1 supports one finite task stream per kernel")
        self.task_block = Block(arg_types=[IndexType()])
        self.task_block.args[0].name_hint = "generation"
        self.stream = StreamOp.create(
            attributes={"count": IntAttr(count)}, regions=[Region(self.task_block)]
        )
        self.block.add_op(self.stream)
        return self

    def view(
        self,
        source: SSAValue,
        nbytes: int,
        *,
        offset=0,
        stride=0,
        slots=0,
        dtype=None,
        layout=None,
        tail_nbytes=0,
        rows=1,
        row_stride=0,
        core_stride=0,
    ) -> SSAValue:
        assert self.task_block is not None
        typ = source.type
        op = ViewOp.create(
            operands=[source, self.task_block.args[0]],
            result_types=[
                ref_type(typ.bank.data, dtype or typ.dtype.data, layout or typ.layout.data)
            ],
            attributes={
                "offset": IntAttr(offset),
                "stride": IntAttr(stride),
                "modulus": IntAttr(slots),
                "nbytes": IntAttr(nbytes),
                "tail_nbytes": IntAttr(tail_nbytes),
                "rows": IntAttr(rows),
                "row_stride": IntAttr(row_stride),
                "core_stride": IntAttr(core_stride),
            },
        )
        self.task_block.add_op(op)
        return op.ref

    def task(
        self, uid, native, *refs, stage=0, params=(), scalar=0.0, active=(0, -1), issue_order=None
    ) -> TaskOp:
        assert self.task_block is not None
        attrs = task_attrs(uid, native, stage, params, scalar)
        attrs.update(
            lower=IntAttr(active[0]),
            upper=IntAttr(active[1]),
            issue_order=IntAttr(self.num_tasks if issue_order is None else issue_order),
        )
        op = TaskOp.create(
            operands=list(refs),
            result_types=[TaskType(), f32] if native == "scalar_load" else [TaskType()],
            attributes=attrs,
        )
        self.num_tasks += 1
        self.task_block.add_op(op)
        op.task.name_hint = uid
        if op.value is not None:
            op.value.name_hint = uid + "_value"
        return op

    def scalar_load(self, uid, source, *, stage=0, issue_order=None):
        op = self.task(uid, "scalar_load", source, stage=stage, issue_order=issue_order)
        assert op.value is not None
        return op.value

    def scale_value(self, uid, source, destination, value, *, stage=0, issue_order=None):
        return self.task(
            uid, "scale_value", source, destination, value, stage=stage, issue_order=issue_order
        )

    def subtract_value(self, uid, source, destination, value, *, stage=0, issue_order=None):
        return self.task(
            uid, "sub_value", source, destination, value, stage=stage, issue_order=issue_order
        )

    def copy(self, uid, source, destination, *, stage=0):
        return self.task(uid, "copy", source, destination, stage=stage)

    def scale(self, uid, source, destination, factor, *, stage=0):
        return self.task(uid, "muls", source, destination, stage=stage, scalar=factor)

    def add_scalar(self, uid, source, destination, scalar, *, stage=0):
        return self.task(uid, "adds", source, destination, stage=stage, scalar=scalar)

    def add(self, uid, lhs, rhs, destination, *, stage=0):
        return self.task(uid, "add", lhs, rhs, destination, stage=stage)

    def mad(self, uid, a, b, c, *, stage=0, clear=True):
        return self.task(uid, "mad16", a, b, c, stage=stage, params=(int(clear),))

    def fixpipe(self, uid, source, destination, *, stage=0):
        return self.task(uid, "fix16", source, destination, stage=stage)

    def finish(self) -> ModuleOp:
        assert self.task_block is not None
        self.task_block.add_op(EndOp())
        self.block.add_op(EndOp())
        self.module.verify()
        return self.module

"""Explicit fixed-issue builder; there are no user event IDs or barriers."""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, StringAttr, f32
from xdsl.ir import Block, Region, SSAValue

from .ir import (
    BufferOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    OnceOp,
    RuntimeSizeOp,
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
        self.invocation = None

    def buffer(
        self,
        name,
        bank,
        nbytes,
        dtype="f32",
        *,
        layout="linear",
        external=False,
        initialized=False,
        extent_count=None,
        count_scale=0,
    ) -> SSAValue:
        op = BufferOp.create(
            operands=[] if extent_count is None else [extent_count],
            result_types=[ref_type(bank, dtype, layout)],
            attributes={
                "sym_name": StringAttr(name),
                "nbytes": IntAttr(nbytes),
                "external": IntAttr(int(external)),
                "initialized": IntAttr(int(initialized)),
                "count_scale": IntAttr(count_scale),
            },
        )
        self.block.add_op(op)
        op.ref.name_hint = name
        return op.ref

    def init_fixpipe(self):
        self.block.add_op(FixConfigOp())

    def runtime_size(self, name, *, multiple=1):
        op = RuntimeSizeOp.create(
            result_types=[IndexType()],
            attributes={"sym_name": StringAttr(name), "multiple": IntAttr(multiple)},
        )
        self.block.add_op(op)
        return op.value

    def loop(self, count: int | SSAValue, *, symbolic=False):
        if self.stream is not None:
            raise ValueError("v0.1 supports one finite task stream per kernel")
        self.task_block = Block(arg_types=[IndexType()])
        self.task_block.args[0].name_hint = "generation"
        self.stream = StreamOp.create(
            operands=[count] if isinstance(count, SSAValue) else [],
            attributes=({} if isinstance(count, SSAValue) else {"count": IntAttr(count)})
            | ({"symbolic": IntAttr(1)} if symbolic or isinstance(count, SSAValue) else {}),
            regions=[Region(self.task_block)],
        )
        (self.block if self.invocation is None else self.invocation.body.block).add_op(self.stream)
        return self

    def invocations(self, bound):
        if self.invocation is not None or self.stream is not None:
            raise ValueError("invocation scope must precede its child stream")
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        one = arith.ConstantOp.from_int_and_width(1, IndexType())
        self.block.add_ops([zero, one])
        self.invocation = scf.ForOp(
            zero.result, bound, one.result, [], Block(arg_types=[IndexType()])
        )
        self.block.add_op(self.invocation)
        return self.invocation

    def once(self, uid):
        if self.stream is not None or self.invocation is not None:
            raise ValueError("ordinary block needs its own lexical region")
        self.task_block = Block(arg_types=[IndexType()])
        self.stream = OnceOp.create(
            attributes={"uid": StringAttr(uid)}, regions=[Region(self.task_block)]
        )
        self.block.add_op(self.stream)
        return self

    def region(self, uid, count=None):
        """Append an ordinary block or retained stream, sharing real allocations."""
        if self.invocation is not None:
            raise ValueError("sequential regions inside an invocation are not yet supported")
        if self.task_block is not None:
            self.task_block.add_op(EndOp())
        self.stream, self.task_block = None, None
        if count is None:
            return self.once(uid)
        self.loop(count, symbolic=True)
        self.stream.attributes["uid"] = StringAttr(uid)
        return self

    def view(
        self,
        source: SSAValue,
        nbytes: int,
        *,
        offset=0,
        stride=0,
        slots=0,
        divisor=1,
        dtype=None,
        layout=None,
        tail_nbytes=0,
        rows=1,
        row_stride=0,
        core_stride=0,
        core_divisor=1,
        core_modulus=0,
        index_terms=(),
        core_terms=(),
        extent_count=None,
        count_offset=0,
        count_core_stride=0,
        prefix_elements=0,
    ) -> SSAValue:
        assert self.task_block is not None
        typ = source.type
        op = ViewOp.create(
            operands=[source, self.stream.body.block.args[0]]
            + ([] if extent_count is None else [extent_count]),
            result_types=[
                ref_type(typ.bank.data, dtype or typ.dtype.data, layout or typ.layout.data)
            ],
            attributes={
                "offset": IntAttr(offset),
                "stride": IntAttr(stride),
                "modulus": IntAttr(slots),
                "divisor": IntAttr(divisor),
                "nbytes": IntAttr(nbytes),
                "tail_nbytes": IntAttr(tail_nbytes),
                "rows": IntAttr(rows),
                "row_stride": IntAttr(row_stride),
                "core_stride": IntAttr(core_stride),
                "core_divisor": IntAttr(core_divisor),
                "core_modulus": IntAttr(core_modulus),
                "count_offset": IntAttr(count_offset),
                "count_core_stride": IntAttr(count_core_stride),
                **({"prefix_elements": IntAttr(prefix_elements)} if prefix_elements else {}),
            },
        )
        if index_terms:
            op.attributes["index_terms"] = ArrayAttr(
                ArrayAttr(IntAttr(n) for n in term) for term in index_terms
            )
        if core_terms:
            op.attributes["core_terms"] = ArrayAttr(
                ArrayAttr(IntAttr(n) for n in term) for term in core_terms
            )
        self.task_block.add_op(op)
        return op.ref

    def task(
        self,
        uid,
        native,
        *refs,
        stage=0,
        params=(),
        scalar=0.0,
        active=(0, -1),
        issue_order=None,
        period=1,
        phase=0,
        phase_width=1,
        logical_phase=0,
    ) -> TaskOp:
        assert self.task_block is not None
        attrs = task_attrs(uid, native, stage, params, scalar)
        attrs.update(
            lower=IntAttr(active[0]),
            upper=IntAttr(active[1]),
            issue_order=IntAttr(self.num_tasks if issue_order is None else issue_order),
            period=IntAttr(period),
            phase=IntAttr(phase),
            phase_width=IntAttr(phase_width),
        )
        if logical_phase:
            attrs["logical_phase"] = IntAttr(logical_phase)
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
        if self.invocation is not None:
            self.invocation.body.block.add_op(scf.YieldOp())
        self.block.add_op(EndOp())
        self.module.verify()
        return self.module

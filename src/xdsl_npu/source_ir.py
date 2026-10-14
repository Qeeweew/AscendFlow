"""Typed source regions for software stages and Cube/Vector domains.

Logical tile results express data dependence. Their SSA identity does not
promise asynchronous completion; lowering supplies actual physical effects.
"""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IndexType, IntAttr, StringAttr, f32, i1
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsTerminator, RecursiveMemoryEffect
from xdsl.utils.exceptions import VerifyException

from .ir import AsyncEffects


def _verify_yield(op):
    terminator = op.body.block.last_op
    if not isinstance(terminator, YieldOp):
        raise VerifyException("source region must end with npu_source.yield")
    if tuple(value.type for value in terminator.values) != tuple(
        value.type for value in op.results
    ):
        raise VerifyException("source yield types must match region result types")


def _verify_result(op, destination):
    typ = destination.type
    if op.value.type.shape != typ.shape or op.value.type.dtype != typ.dtype:
        raise VerifyException("logical result type must match its destination tile")


def _verify_dependencies(op, inputs):
    if len(op.dependencies) != len(op.dependency_inputs):
        raise VerifyException("logical dependencies must name their consumed input")
    for value, index in zip(op.dependencies, op.dependency_inputs):
        if not 0 <= index.data < len(inputs):
            raise VerifyException("invalid logical dependency input index")
        typ = inputs[index.data].type
        if value.type.shape != typ.shape or value.type.dtype != typ.dtype:
            raise VerifyException("logical dependency type must match its consumed input")


@irdl_attr_definition
class TensorType(ParametrizedAttribute, TypeAttribute):
    name = "npu_source.tensor"
    shape: ArrayAttr[IntAttr]
    dtype: StringAttr
    memory: StringAttr
    layout: StringAttr


@irdl_attr_definition
class TileType(ParametrizedAttribute, TypeAttribute):
    name = "npu_source.tile"
    shape: ArrayAttr[IntAttr]
    dtype: StringAttr


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "npu_source.yield"
    values = var_operand_def(TileType)
    traits = traits_def(IsTerminator())


@irdl_op_definition
class ProgramOp(IRDLOperation):
    name = "npu_source.program"
    sym_name = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    groups = attr_def(IntAttr)
    disjoint_external = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        if self.kind.data not in {"vector", "cube", "mixed"} or not 1 <= self.groups.data <= 1024:
            raise VerifyException("source kernel requires vector/cube/mixed and groups in [1,1024]")
        if self.disjoint_external.data not in {0, 1}:
            raise VerifyException("external disjoint contract must be boolean")
        if len(self.body.block.args) != 1 or not isinstance(
            self.body.block.args[0].type, IndexType
        ):
            raise VerifyException("kernel region has one group coordinate")
        for op in self.walk():
            if isinstance(op, DomainOp) and self.kind.data != "mixed" and op.kind != self.kind:
                raise VerifyException("execution domain does not belong to this kernel kind")


@irdl_op_definition
class AllocationOp(IRDLOperation):
    name = "npu_source.alloc"
    sym_name = attr_def(StringAttr)
    versions = attr_def(IntAttr)
    access = attr_def(StringAttr)
    ring = attr_def(IntAttr)
    dimensions = var_operand_def(IndexType)
    dynamic_axes = opt_attr_def(ArrayAttr[IntAttr])
    ref = result_def(TensorType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        typ = self.ref.type
        if (
            not typ.shape
            or any(n.data == 0 or n.data < -1 for n in typ.shape)
            or typ.dtype.data not in {"f16", "bf16", "f32"}
            or typ.memory.data not in {"GM", "UB", "L1", "L0A", "L0B", "L0C"}
            or typ.layout.data not in {"linear", "NZ", "ZZ", "ZN"}
        ):
            raise VerifyException(
                "allocation needs positive typed dimensions and a target memory/layout"
            )
        if not 1 <= self.versions.data <= 8 or self.access.data not in {
            "read",
            "write",
            "read_write",
            "local",
        }:
            raise VerifyException(
                "allocation requires versions in [1,8] and a known access contract"
            )
        if self.ring.data not in {0, 1}:
            raise VerifyException("task ring marker must be boolean")
        axes = [] if self.dynamic_axes is None else [axis.data for axis in self.dynamic_axes]
        if axes != [i for i, n in enumerate(typ.shape) if n.data == -1] or len(axes) != len(
            self.dimensions
        ):
            raise VerifyException("dynamic shape axes must bind actual SSA dimensions")
        if axes and (typ.memory.data != "GM" or self.ring.data or self.versions.data != 1):
            raise VerifyException(
                "runtime dimensions are GM extents; native/local tiles stay static"
            )


@irdl_op_definition
class SizeOp(IRDLOperation):
    name = "npu_source.size"
    sym_name = attr_def(StringAttr)
    multiple = attr_def(IntAttr)
    value = result_def(IndexType)

    def verify_(self):
        if not self.sym_name.data.isidentifier() or self.multiple.data <= 0:
            raise VerifyException(
                "runtime size requires a name and a positive divisibility contract"
            )
        if not isinstance(self.parent_op(), ProgramOp):
            raise VerifyException("runtime size is a uniform program parameter")


@irdl_op_definition
class PipelineOp(IRDLOperation):
    name = "npu_source.pipeline"
    uid = attr_def(StringAttr)
    count = opt_attr_def(IntAttr)
    limit = opt_operand_def(IndexType)
    stages = attr_def(IntAttr)
    order = attr_def(StringAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        if (
            (self.count is None) == (self.limit is None)
            or (self.count is not None and not 0 <= self.count.data <= (1 << 31) - 1)
            or not 1 <= self.stages.data <= 33
            or self.order.data != "source"
        ):
            raise VerifyException(
                "fixed pipeline needs bounded count/stage horizon and source order"
            )
        if self.limit is not None and not isinstance(self.limit.owner, SizeOp):
            raise VerifyException("runtime pipeline length must be a uniform SSA size parameter")
        if len(self.body.block.args) != 1 or not isinstance(
            self.body.block.args[0].type, IndexType
        ):
            raise VerifyException("pipeline region has one logical task coordinate")
        for op in self.body.block.ops:
            if isinstance(op, StageOp) and not 0 <= op.offset.data < self.stages.data:
                raise VerifyException("Stage offset lies outside this pipeline's stage horizon")


@irdl_op_definition
class StageOp(IRDLOperation):
    name = "npu_source.stage"
    offset = attr_def(IntAttr)
    label = attr_def(StringAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        parent = self.parent_op()
        if not isinstance(parent, PipelineOp) or not 0 <= self.offset.data < parent.stages.data:
            raise VerifyException("Stage must be directly inside its pipeline's stage horizon")
        if self.body.block.args:
            raise VerifyException("Stage inherits its pipeline coordinate")


@irdl_op_definition
class DomainOp(IRDLOperation):
    name = "npu_source.domain"
    kind = attr_def(StringAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        if self.kind.data not in {"cube", "vector"}:
            raise VerifyException("execution domain must be Cube or Vector")
        if len(self.body.block.args) != 1 or not isinstance(
            self.body.block.args[0].type, IndexType
        ):
            raise VerifyException("execution domain has one owner coordinate")
        parent = self.parent_op()
        while parent is not None and not isinstance(parent, ProgramOp):
            if isinstance(parent, DomainOp):
                raise VerifyException("Cube/Vector execution domains cannot nest")
            parent = parent.parent_op()


@irdl_op_definition
class ViewOp(IRDLOperation):
    name = "npu_source.view"
    source = operand_def(TensorType)
    indices = var_operand_def(IndexType)
    offset = attr_def(IntAttr)
    terms = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    dynamic_terms = opt_attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    strides = attr_def(ArrayAttr[IntAttr])
    ref = result_def(TensorType)

    def verify_(self):
        if (
            self.ref.type.memory != self.source.type.memory
            or self.ref.type.dtype != self.source.type.dtype
            or self.ref.type.layout != self.source.type.layout
        ):
            raise VerifyException("view cannot change memory, dtype, or layout")
        if not self.ref.type.shape or any(n.data <= 0 for n in self.ref.type.shape):
            raise VerifyException("view dimensions must be positive")
        if len(self.strides) != len(self.ref.type.shape) or any(s.data <= 0 for s in self.strides):
            raise VerifyException("view needs a positive byte stride per axis")
        for term in self.terms:
            if len(term) != 4:
                raise VerifyException("view term requires index/coefficient/divisor/modulus")
            index, _, divisor, modulus = (v.data for v in term)
            if not 0 <= index < len(self.indices) or divisor <= 0 or modulus < 0:
                raise VerifyException("invalid view coordinate term")
        for term in self.dynamic_terms or ():
            if len(term) != 5:
                raise VerifyException(
                    "runtime view term requires coordinate/size/coefficient/divisor/modulus"
                )
            index, size, _, divisor, modulus = (v.data for v in term)
            if (
                not 0 <= index < len(self.indices)
                or not 0 <= size < len(self.indices)
                or not isinstance(self.indices[size].owner, SizeOp)
                or divisor <= 0
                or modulus < 0
            ):
                raise VerifyException("invalid runtime view coordinate/size term")


@irdl_op_definition
class PredicateOp(IRDLOperation):
    """A typed comparison of affine task coordinates, not a completion token."""

    name = "npu_source.predicate"
    indices = var_operand_def(IndexType)
    lhs_terms = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    rhs_terms = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    lhs_offset = attr_def(IntAttr)
    rhs_offset = attr_def(IntAttr)
    comparison = attr_def(StringAttr)
    value = result_def(i1)

    def verify_(self):
        if self.comparison.data not in {"eq", "lt", "le"}:
            raise VerifyException("unsupported task comparison")
        for terms in (self.lhs_terms, self.rhs_terms):
            for term in terms:
                if len(term) != 4:
                    raise VerifyException(
                        "predicate term requires index/coefficient/divisor/modulus"
                    )
                index, _, divisor, modulus = (v.data for v in term)
                if not 0 <= index < len(self.indices) or divisor <= 0 or modulus < 0:
                    raise VerifyException("invalid predicate coordinate term")


@irdl_op_definition
class CopyOp(IRDLOperation):
    name = "npu_source.copy"
    source = operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.source])
        if (
            self.source.type.shape != self.destination.type.shape
            or self.source.type.dtype != self.destination.type.dtype
        ):
            raise VerifyException("copy requires the same typed tile extent; use explicit cast")


@irdl_op_definition
class ElementOp(IRDLOperation):
    name = "npu_source.element"
    inputs = var_operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    kind = attr_def(StringAttr)
    scalar = opt_attr_def(FloatAttr[f32])
    value = result_def(TileType)
    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, self.inputs)
        arity = {
            "fill": 0,
            "mul_scalar": 1,
            "add_scalar": 1,
            "exp": 1,
            "cast": 1,
            "mul": 2,
            "add": 2,
            "sub": 2,
        }
        if self.kind.data not in arity or len(self.inputs) != arity[self.kind.data]:
            raise VerifyException("unknown element operation or incorrect operand count")
        if any(v.type.shape != self.destination.type.shape for v in self.inputs):
            raise VerifyException("element operands require equal tile shape")
        if (self.kind.data in {"fill", "mul_scalar", "add_scalar"}) != (self.scalar is not None):
            raise VerifyException("scalar policy must match element operation")
        if self.kind.data != "cast" and any(
            v.type.dtype.data != "f32" for v in (*self.inputs, self.destination)
        ):
            raise VerifyException("arithmetic currently requires explicit f32 operands")


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "npu_source.matmul"
    a = operand_def(TensorType)
    b = operand_def(TensorType)
    accumulator = operand_def(TensorType)
    initialize = operand_def(i1)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    transpose_b = attr_def(IntAttr)
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.accumulator)
        _verify_dependencies(self, [self.a, self.b, self.accumulator])
        if self.transpose_b.data not in {0, 1}:
            raise VerifyException("matmul transpose policy must be boolean")
        if any(len(v.type.shape) != 2 for v in (self.a, self.b, self.accumulator)):
            raise VerifyException("matmul requires matrix tiles")
        m, k = self.a.type.shape
        b0, b1 = self.b.type.shape
        n, bk = (b0, b1) if self.transpose_b.data else (b1, b0)
        if k != bk or self.accumulator.type.shape != ArrayAttr([m, n]):
            raise VerifyException("matmul extents disagree")
        if (
            self.a.type.dtype != self.b.type.dtype
            or self.a.type.dtype.data not in {"f16", "bf16"}
            or self.accumulator.type.dtype.data != "f32"
        ):
            raise VerifyException("matmul requires b16 operands and f32 accumulator")


SOURCE = Dialect(
    "npu_source",
    [
        ProgramOp,
        AllocationOp,
        SizeOp,
        PipelineOp,
        StageOp,
        DomainOp,
        ViewOp,
        PredicateOp,
        CopyOp,
        ElementOp,
        MatmulOp,
        YieldOp,
    ],
    [TensorType, TileType],
)

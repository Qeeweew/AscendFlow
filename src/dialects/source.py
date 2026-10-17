"""Typed source regions for software stages and Cube/Vector domains.

Logical tile results express data dependence. Their SSA identity does not
promise asynchronous completion; lowering supplies actual physical effects.
"""

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    StringAttr,
    f32,
    i1,
)
from xdsl.ir import Block, Dialect
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
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

from .types import AsyncEffects, TensorType, TileType, verify_runtime_contract


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
    minimum = attr_def(IntAttr)
    multiple = attr_def(IntAttr)
    value = result_def(IndexType)

    def verify_(self):
        verify_runtime_contract(self.sym_name.data, self.minimum.data, self.multiple.data)
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
    issue_order = opt_attr_def(DictionaryAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        if (
            (self.count is None) == (self.limit is None)
            or (self.count is not None and not 0 <= self.count.data <= (1 << 63) - 1)
            or not 1 <= self.stages.data <= 33
            or self.order.data != "source"
        ):
            raise VerifyException(
                "fixed pipeline needs bounded count/stage horizon and source order"
            )
        # Arithmetic bounds stay in source SSA. Target lowering must separately
        # prove uniformity/range and reject unsupported expressions.
        if len(self.body.block.args) != 1 or not isinstance(
            self.body.block.args[0].type, IndexType
        ):
            raise VerifyException("pipeline region has one logical task coordinate")
        for op in self.body.block.ops:
            if isinstance(op, StageOp) and not 0 <= op.offset.data < self.stages.data:
                raise VerifyException("Stage offset lies outside this pipeline's stage horizon")
        if self.issue_order is not None:
            parent = self.parent_op()
            while parent is not None and not isinstance(parent, ProgramOp):
                parent = parent.parent_op()
            allowed = (
                {"cube", "vector"}
                if parent is None or parent.kind.data == "mixed"
                else {parent.kind.data}
            )
            for actor, labels in self.issue_order.data.items():
                if (
                    actor not in allowed
                    or not isinstance(labels, ArrayAttr)
                    or any(not isinstance(label, StringAttr) for label in labels)
                ):
                    raise VerifyException("issue_order needs actor names and stage label arrays")
                if len({label.data for label in labels}) != len(labels):
                    raise VerifyException("issue_order stage labels must be unique")


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


def _dominates(value, use, positions, ancestors):
    """Require a definition before its use along the enclosing source regions."""
    if use not in ancestors:
        anchors = {}
        anchor, block = use, use.parent
        while block is not None:
            anchors[block] = anchor
            anchor = block.parent_op()
            block = None if anchor is None else anchor.parent
        ancestors[use] = anchors
    anchors = ancestors[use]
    owner = value.owner
    if isinstance(owner, Block):
        if owner not in anchors:
            raise VerifyException("view index block argument does not dominate its use")
        return
    block = owner.parent
    if block not in anchors:
        raise VerifyException("view operand definition does not dominate its use")
    if block not in positions:
        positions[block] = {operation: position for position, operation in enumerate(block.ops)}
    if positions[block][owner] < positions[block][anchors[block]]:
        return
    raise VerifyException("view operand definition must precede its use")


def _verify_index_provenance(value, use, seen, active, positions, ancestors, depth=0):
    from xdsl.dialects import arith, scf

    _dominates(value, use, positions, ancestors)
    if value in active:
        raise VerifyException("view index SSA contains a cycle")
    if value in seen:
        return
    if depth > 128 or len(seen) + len(active) >= 4096:
        raise VerifyException("view index SSA exceeds expression budget")
    owner = value.owner
    if isinstance(owner, Block):
        region = owner.parent_op()
        if (
            not isinstance(region, (ProgramOp, PipelineOp, DomainOp, scf.ForOp))
            or value is not owner.args[0]
        ):
            raise VerifyException("view index needs a declared source coordinate")
        seen.add(value)
        return
    if isinstance(owner, SizeOp):
        if not isinstance(owner.parent_op(), ProgramOp):
            raise VerifyException("view runtime size must be a uniform program parameter")
        seen.add(value)
        return
    if not isinstance(
        owner,
        (
            arith.ConstantOp,
            arith.AddiOp,
            arith.SubiOp,
            arith.MuliOp,
            arith.DivUIOp,
            arith.RemUIOp,
            arith.MinSIOp,
            arith.MaxSIOp,
            arith.CeilDivUIOp,
        ),
    ):
        raise VerifyException("view index has unsupported SSA provenance")
    active.add(value)
    for operand in owner.operands:
        if not isinstance(operand.type, IndexType):
            raise VerifyException("view index arithmetic requires index operands")
        _verify_index_provenance(operand, owner, seen, active, positions, ancestors, depth + 1)
    active.remove(value)
    seen.add(value)


def _constant_index(value):
    from xdsl.dialects import arith

    memo = {}

    def visit(current):
        if current in memo:
            return memo[current]
        owner = current.owner
        result = None
        if isinstance(owner, arith.ConstantOp):
            result = owner.value.value.data
        elif hasattr(owner, "operands") and len(owner.operands) == 2:
            lhs, rhs = (visit(operand) for operand in owner.operands)
            if lhs is not None and rhs is not None:
                if isinstance(owner, arith.AddiOp):
                    result = lhs + rhs
                elif isinstance(owner, arith.SubiOp):
                    result = lhs - rhs
                elif isinstance(owner, arith.MuliOp):
                    result = lhs * rhs
                elif isinstance(owner, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
                    if lhs < 0 or rhs <= 0:
                        raise VerifyException(
                            "unsigned index division requires nonnegative numerator and positive divisor"
                        )
                    if isinstance(owner, arith.RemUIOp):
                        result = lhs % rhs
                    else:
                        result = (
                            (lhs + rhs - 1) // rhs
                            if isinstance(owner, arith.CeilDivUIOp)
                            else lhs // rhs
                        )
                elif isinstance(owner, arith.MinSIOp):
                    result = min(lhs, rhs)
                elif isinstance(owner, arith.MaxSIOp):
                    result = max(lhs, rhs)
        if result is not None and not -(1 << 63) <= result <= (1 << 63) - 1:
            raise VerifyException("constant view index arithmetic exceeds int64")
        memo[current] = result
        return result

    # Provenance and DAG budgets have already been checked by the caller.
    return visit(value)


@irdl_op_definition
class ViewOp(IRDLOperation):
    """A native tile with an origin and static width on every allocation axis."""

    name = "npu_source.view"
    source = operand_def(TensorType)
    origins = var_operand_def(IndexType)
    widths = attr_def(ArrayAttr[IntAttr])
    axes = attr_def(ArrayAttr[IntAttr])
    valid = opt_operand_def(IndexType)
    ref = result_def(TensorType)
    irdl_options = (AttrSizedOperandSegments(as_property=False),)

    def verify_(self):
        if not isinstance(self.source.owner, AllocationOp):
            raise VerifyException("view source must be the original allocation")
        positions, ancestors = {}, {}
        _dominates(self.source, self, positions, ancestors)
        source, result = self.source.type, self.ref.type
        if (result.memory, result.dtype, result.layout) != (
            source.memory,
            source.dtype,
            source.layout,
        ):
            raise VerifyException("view cannot change memory, dtype, or layout")
        rank = len(source.shape)
        if len(self.origins) != rank or len(self.widths) != rank:
            raise VerifyException("view needs one origin and width per original axis")
        widths, axes = (
            tuple(width.data for width in self.widths),
            tuple(axis.data for axis in self.axes),
        )
        if any(width <= 0 for width in widths):
            raise VerifyException("view widths must be positive static integers")
        if (
            not axes
            or tuple(sorted(set(axes))) != axes
            or any(not 0 <= axis < rank for axis in axes)
        ):
            raise VerifyException("view axes must retain ordered distinct original axes")
        if any(width != 1 for axis, width in enumerate(widths) if axis not in axes):
            raise VerifyException("integer-indexed view axes must have width one")
        if tuple(dimension.data for dimension in result.shape) != tuple(
            widths[axis] for axis in axes
        ):
            raise VerifyException("view result shape must match its retained original axes")
        seen = set()
        for origin in (*self.origins, *((self.valid,) if self.valid is not None else ())):
            _verify_index_provenance(origin, self, seen, set(), positions, ancestors)
        for origin, width, extent in zip(self.origins, widths, source.shape):
            constant = _constant_index(origin)
            if constant is not None and (
                constant < 0 or extent.data > 0 and constant + width > extent.data
            ):
                raise VerifyException("view origin and width exceed the original axis")


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
    transpose = opt_attr_def(IntAttr)
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.source])
        transpose = 0 if self.transpose is None else self.transpose.data
        if transpose not in {0, 1}:
            raise VerifyException("copy transpose policy must be boolean")
        if transpose and (
            (self.source.type.memory.data, self.destination.type.memory.data) != ("L1", "L0B")
            or len(self.source.type.shape) != 2
        ):
            raise VerifyException("transposed copy requires a complete L1→L0B matrix")
        expected = (
            ArrayAttr(reversed(self.source.type.shape.data))
            if transpose
            else self.source.type.shape
        )
        if (
            expected != self.destination.type.shape
            or self.source.type.dtype != self.destination.type.dtype
        ):
            raise VerifyException("copy requires the same typed tile extent; use explicit cast")


@irdl_op_definition
class ElementOp(IRDLOperation):
    name = "npu_source.element"
    inputs = var_operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    scratch = opt_operand_def(TensorType)
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
            "sqrt": 1,
            "rsqrt": 1,
            "cast": 1,
            "mul": 2,
            "add": 2,
            "sub": 2,
            "div": 2,
            "maximum": 2,
            "mul_rows": 2,
            "sub_rows": 2,
            "div_rows": 2,
        }
        if self.kind.data not in arity or len(self.inputs) != arity[self.kind.data]:
            raise VerifyException("unknown element operation or incorrect operand count")
        if self.kind.data.endswith("_rows"):
            shape = tuple(n.data for n in self.destination.type.shape)
            if (
                len(shape) != 2
                or not 8 <= shape[0] <= 248
                or shape[0] % 8
                or self.inputs[0].type.shape != self.destination.type.shape
                or tuple(n.data for n in self.inputs[1].type.shape) != (shape[0],)
                or self.scratch is None
                or tuple(n.data for n in self.scratch.type.shape) != (shape[0], 8)
            ):
                raise VerifyException(
                    "row broadcast needs a matrix, packed row values and explicit (rows,8) scratch; rows must be a multiple of eight"
                )
            if any(
                v.type.memory.data != "UB" or v.type.dtype.data != "f32"
                for v in (*self.inputs, self.destination, self.scratch)
            ):
                raise VerifyException("row broadcast requires f32 UB views")
        elif self.scratch is not None:
            raise VerifyException("only row broadcast arithmetic uses scratch")
        elif any(v.type.shape != self.destination.type.shape for v in self.inputs):
            raise VerifyException("element operands require equal tile shape")
        if (self.kind.data in {"fill", "mul_scalar", "add_scalar"}) != (self.scalar is not None):
            raise VerifyException("scalar policy must match element operation")
        if self.kind.data != "cast" and any(
            v.type.dtype.data != "f32" for v in (*self.inputs, self.destination)
        ):
            raise VerifyException("arithmetic currently requires explicit f32 operands")


def block8_reduction_widths(columns: int) -> tuple[int, ...]:
    """Exact packed planes for block8 stages followed by one native merge.

    Each existing block8 native consumes complete 64-element repeats. Its
    output rows must remain 32-byte aligned for the next stage and merge;
    padding or a different reduction tree must be selected explicitly.
    """
    widths = []
    while True:
        if columns < 64 or columns > 4096 or columns % 64:
            raise VerifyException(
                "block8 reduction stages require whole 64-element repeats "
                "and 32-byte aligned partial rows"
            )
        columns //= 8
        widths.append(columns)
        if columns <= 64:
            return tuple(widths)


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "npu_source.reduce"
    source = operand_def(TensorType)
    scratch = operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    kind = attr_def(StringAttr)
    order = attr_def(StringAttr)
    axis = attr_def(IntAttr)
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.source])
        shape = tuple(n.data for n in self.source.type.shape)
        if len(shape) not in {1, 2} or self.axis.data != len(shape) - 1:
            raise VerifyException("reduction currently requires the final axis of a vector/matrix")
        cols = shape[-1]
        if cols < 64 or cols > 4096 or cols % 64:
            raise VerifyException("reduction needs 64-aligned columns in [64,4096]")
        if self.kind.data not in {"sum", "max"} or self.order.data not in {
            "r64_then_native_merge",
            "r64_then_packed_merge",
            "block8_then_packed_merge",
        }:
            raise VerifyException("reduction requires an explicit supported order")
        block8 = self.order.data == "block8_then_packed_merge"
        packed = block8 or self.order.data == "r64_then_packed_merge"
        result_shape = shape[:-1] if packed else (*shape[:-1], 1)
        if packed and len(shape) != 2:
            raise VerifyException("packed reduction requires a matrix")
        scratch_shape = (
            (shape[0] * sum(block8_reduction_widths(cols)),)
            if block8
            else (*shape[:-1], cols // 64)
        )
        if (
            tuple(n.data for n in self.scratch.type.shape) != scratch_shape
            or tuple(n.data for n in self.destination.type.shape) != result_shape
        ):
            raise VerifyException(
                "reduction scratch/result must expose exact partial/statistic shapes"
            )
        if any(
            v.type.memory.data != "UB" or v.type.dtype.data != "f32"
            for v in (self.source, self.scratch, self.destination)
        ):
            raise VerifyException("reduction requires f32 UB views")


@irdl_op_definition
class ScalarReadOp(IRDLOperation):
    name = "npu_source.scalar_read"
    source = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    value = result_def(f32)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_dependencies(self, [self.source])
        if (
            self.source.type.memory.data != "UB"
            or self.source.type.dtype.data != "f32"
            or any(n.data != 1 for n in self.source.type.shape)
        ):
            raise VerifyException("Scalar read requires exactly one f32 UB element")


@irdl_op_definition
class ScaleOp(IRDLOperation):
    name = "npu_source.scale_value"
    source = operand_def(TensorType)
    destination = operand_def(TensorType)
    factor = operand_def(f32)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.source])
        if self.source.type.shape != self.destination.type.shape or any(
            v.type.dtype.data != "f32" or v.type.memory.data != "UB"
            for v in (self.source, self.destination)
        ):
            raise VerifyException("scalar multiplication requires equal f32 UB views")


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
        banks = (
            self.a.type.memory.data,
            self.b.type.memory.data,
            self.accumulator.type.memory.data,
        )
        if banks != ("L0A", "L0B", "L0C"):
            raise VerifyException("matmul requires explicit L0A/L0B/L0C operands")
        if self.transpose_b.data:
            raise VerifyException("L0B is already packed for matmul; transpose its L1 copy")
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
        ReduceOp,
        ScalarReadOp,
        ScaleOp,
        MatmulOp,
        YieldOp,
    ],
    [TensorType, TileType],
)

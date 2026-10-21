"""Typed source regions for software stages and Cube/Vector domains.

Logical tile results express data dependence. Their SSA identity does not
promise asynchronous completion; lowering supplies actual physical effects.
"""

from math import prod

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

from .types import AsyncEffects, MaskType, TensorType, TileType, verify_runtime_contract


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


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _shapes(rows):
    return ArrayAttr(_ints(row) for row in rows)


def mask_chunks(shape):
    """One 32-byte packed mask block per 64 f32 lanes."""
    lanes = prod(shape)
    if lanes % 64:
        raise VerifyException("packed predicates need a whole 64-lane block")
    return lanes // 64


def _mask_storage(value):
    """The mask allocation identity behind an explicit packed storage view."""
    owner = value.owner
    if not isinstance(owner, ViewOp):
        raise VerifyException("predicate storage needs an explicit source view")
    allocation = owner.source.owner
    if not isinstance(allocation, AllocationOp):
        raise VerifyException("predicate storage needs one allocation identity")
    return allocation


def _verify_materialized_mask(mask, storage, dependency, *, message="predicate"):
    if isinstance(mask.owner, MaskOp):
        raise VerifyException(f"{message} requires materialized predicate storage")
    chunks = mask_chunks(tuple(size.data for size in mask.type.shape))
    typ = storage.type
    if (
        typ.memory.data != "UB"
        or typ.dtype.data != "u8"
        or prod(size.data for size in typ.shape) != chunks * 32
    ):
        raise VerifyException(f"{message} needs exact packed 32-byte predicate blocks")
    producer = dependency.owner
    if not isinstance(producer, (CompareOp, MaterializeMaskOp, MaskCombineOp)):
        raise VerifyException(f"{message} dependency must name the predicate writer")
    if _mask_storage(storage) is not _mask_storage(producer.out):
        raise VerifyException(f"{message} storage view does not belong to its writer")
    return chunks


def _verify_mask_operand(op, destination_shape):
    """Common masked-update shape and storage contract for one element op."""
    if op.mask is None:
        if op.mask_storage is not None or op.mask_dependency is not None:
            raise VerifyException("predicate storage requires an explicit mask")
        return
    shape = tuple(size.data for size in op.mask.type.shape)
    if shape != tuple(destination_shape):
        raise VerifyException("masked update needs a predicate of the destination shape")
    materialized = not isinstance(op.mask.owner, MaskOp)
    if materialized != (op.mask_storage is not None and op.mask_dependency is not None):
        raise VerifyException("materialized predicates need explicit storage and dependence")
    if not materialized:
        return
    _verify_materialized_mask(op.mask, op.mask_storage, op.mask_dependency)


def copy_compatible(source, destination, *, transpose=False, zero_pad=False):
    """Source copy contract, including explicit NZ padding and FIX conversion."""
    route = source.memory.data, destination.memory.data
    shape = tuple(n.data for n in source.shape)
    target = tuple(n.data for n in destination.shape)
    if zero_pad:
        return (
            not transpose
            and route == ("GM", "L1")
            and source.dtype == destination.dtype
            and source.dtype.data in {"f16", "bf16"}
            and len(shape) == len(target) == 2
            and all(0 < a <= b and b % 16 == 0 for a, b in zip(shape, target, strict=True))
        )
    if route == ("L0C", "GM"):
        return (
            not transpose
            and source.dtype.data == "f32"
            and destination.dtype.data in {"f32", "f16", "bf16"}
            and len(shape) == len(target) == 2
            and all(0 < b <= a for a, b in zip(shape, target, strict=True))
        )
    return (shape[::-1] if transpose else shape) == target and source.dtype == destination.dtype


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "npu_source.yield"
    values = var_operand_def(TileType)
    traits = traits_def(IsTerminator())


@irdl_op_definition
class ProgramOp(IRDLOperation):
    """SPMD groups with source-defined, independent non-atomic GM accesses.

    The program supplies the inter-group independence precondition. Lowering
    preserves its addresses; AutoSync orders work within each actual group.
    Vector peers likewise execute one shared body with private UB. Shared GM
    handoffs between Cube and Vector remain explicit memory effects.
    """

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
    cache = opt_attr_def(StringAttr)
    write_disjoint = opt_attr_def(IntAttr)
    ref = result_def(TensorType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        if self.write_disjoint is not None and (
            self.write_disjoint.data != 1
            or self.ref.type.memory.data != "GM"
            or self.access.data not in {"write", "read_write"}
        ):
            raise VerifyException("write_disjoint requires a writable external tensor")
        if self.cache is not None and (
            self.cache.data not in {"normal", "bypass"}
            or self.cache.data == "bypass"
            and self.ref.type.memory.data != "GM"
        ):
            raise VerifyException("cache policy must be normal or GM bypass")
        typ = self.ref.type
        if (
            not typ.shape
            or any(n.data == 0 or n.data < -1 for n in typ.shape)
            or typ.dtype.data not in {"f16", "bf16", "f32", "u8", "i32", "i64"}
            or typ.dtype.data in {"i32", "i64"}
            and typ.memory.data != "GM"
            or typ.dtype.data == "u8"
            and typ.memory.data != "UB"
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
class IterationOp(IRDLOperation):
    """A logical iteration domain, independent of any actor's issue clock."""

    name = "npu_source.iteration"
    uid = attr_def(StringAttr)
    count = opt_attr_def(IntAttr)
    limit = opt_operand_def(IndexType)
    issue_order = opt_attr_def(DictionaryAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        if (self.count is None) == (self.limit is None) or (
            self.count is not None and not 0 <= self.count.data <= (1 << 63) - 1
        ):
            raise VerifyException("iteration needs one bounded static or runtime count")
        # Arithmetic bounds stay in source SSA. Target lowering must separately
        # prove uniformity/range and reject unsupported expressions.
        if len(self.body.block.args) != 1 or not isinstance(
            self.body.block.args[0].type, IndexType
        ):
            raise VerifyException("iteration region has one logical task coordinate")
        if self.issue_order is not None:
            parent = self.parent_op()
            while parent is not None and not isinstance(parent, (DomainOp, ProgramOp)):
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
                    or any(not isinstance(label, StringAttr) or not label.data for label in labels)
                ):
                    raise VerifyException("issue_order needs actor names and stage label arrays")
                if len({label.data for label in labels}) != len(labels):
                    raise VerifyException("issue_order stage labels must be unique")


@irdl_op_definition
class StageOp(IRDLOperation):
    name = "npu_source.stage"
    offset = attr_def(IntAttr)
    label = attr_def(StringAttr)
    family = attr_def(StringAttr)
    body = region_def("single_block")
    values = var_result_def(TileType)
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        _verify_yield(self)
        parent = self.parent_op()
        family = None
        while isinstance(parent, DomainOp):
            family = parent.kind.data
            parent = parent.parent_op()
        if not isinstance(parent, IterationOp) or not 0 <= self.offset.data <= 32:
            raise VerifyException(
                "Stage needs its iteration through actor scopes and offset in [0,32]"
            )
        if family is None:
            context = self.parent_op()
            while context is not None:
                if isinstance(context, DomainOp):
                    family = context.kind.data
                    break
                if isinstance(context, ProgramOp):
                    family = context.kind.data
                    break
                context = context.parent_op()
        if family not in {"cube", "vector"} or self.family.data != family:
            raise VerifyException("Stage must belong to its actual Cube or Vector actor")
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


def _verify_index_provenance(value, use, seen, positions, ancestors):
    """Check executable SSA structure, independently of proof expression syntax."""
    from xdsl.dialects import arith, scf

    pending = [(value, use)]
    while pending:
        value, use = pending.pop()
        _dominates(value, use, positions, ancestors)
        if value in seen:
            continue
        seen.add(value)
        owner = value.owner
        if isinstance(owner, Block):
            region = owner.parent_op()
            if not isinstance(region, (ProgramOp, IterationOp, DomainOp, scf.ForOp)) or (
                not isinstance(region, scf.ForOp) and value is not owner.args[0]
            ):
                raise VerifyException("view index needs a structured scalar binding")
            continue
        if isinstance(owner, SizeOp):
            if not isinstance(owner.parent_op(), ProgramOp):
                raise VerifyException("view runtime size must be a uniform program parameter")
            continue
        if isinstance(owner, IndexReadOp):
            continue
        if isinstance(owner, (scf.IfOp, scf.ForOp)):
            pending.extend((operand, owner) for operand in owner.operands)
            for region in owner.regions:
                terminator = region.block.last_op
                if not isinstance(terminator, scf.YieldOp):
                    raise VerifyException("structured scalar index needs yielded results")
                pending.extend((operand, terminator) for operand in terminator.arguments)
            continue
        if type(owner) not in arith.Arith.operations:
            raise VerifyException("view index needs an executable integer operation")
        pending.extend((operand, owner) for operand in owner.operands)


def _constant_index(value, memo=None):
    from xdsl.dialects import arith

    memo = {} if memo is None else memo

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
            _verify_index_provenance(origin, self, seen, positions, ancestors)
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
class MaskOp(IRDLOperation):
    """An exact boolean predicate over logical lane coordinates.

    The program's leaves are static constants, uniform index SSA operands and
    lane coordinates of the predicate's own static lane domain. It is a
    symbolic value: materialization needs explicit packed UB storage.
    """

    name = "npu_source.mask"
    scalars = var_operand_def(IndexType)
    shape = attr_def(ArrayAttr[IntAttr])
    node_shapes = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    nodes = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    value = result_def(MaskType)

    def program(self):
        from ..mask import MaskError, Node, Program, scalar_operand_count

        nodes = tuple(
            Node(row.data[0].data, tuple(value.data for value in row.data[1:]))
            for row in self.nodes
        )
        shape = tuple(value.data for value in self.shape)
        shapes = tuple(tuple(value.data for value in row) for row in self.node_shapes)
        try:
            program = Program(shape, shapes, nodes)
            count = scalar_operand_count(nodes)
        except MaskError as error:
            raise VerifyException(str(error)) from error
        if count != len(self.scalars):
            raise VerifyException("predicate scalar atom count must equal its operand list")
        return program

    def verify_(self):
        program = self.program()
        if tuple(size.data for size in self.value.type.shape) != program.shape:
            raise VerifyException("predicate result shape must match its lane domain")


@irdl_op_definition
class CompareOp(IRDLOperation):
    """An actual UB data comparison packed into explicit predicate storage.

    The comparison is one real read/compute; its result binds this write and
    is never deferred or recomputed after the source tile changes.
    """

    name = "npu_source.compare"
    source = operand_def(TensorType)
    other = opt_operand_def(TensorType)
    out = operand_def(TensorType)
    kind = attr_def(StringAttr)
    scalar = opt_attr_def(FloatAttr[f32])
    value = result_def(TileType)
    mask = result_def(MaskType)
    irdl_options = (AttrSizedOperandSegments(),)

    def verify_(self):
        from ..mask import CMP_KINDS

        if self.kind.data not in CMP_KINDS:
            raise VerifyException("compare requires a supported f32 comparison kind")
        if (self.other is not None) == (self.scalar is not None):
            raise VerifyException("compare needs exactly one tensor or scalar right operand")
        shape = tuple(size.data for size in self.source.type.shape)
        if self.other is not None and (tuple(size.data for size in self.other.type.shape) != shape):
            raise VerifyException("data comparison operands require equal tile shapes")
        if any(
            value.type.memory.data != "UB" or value.type.dtype.data != "f32"
            for value in (self.source, *(() if self.other is None else (self.other,)))
        ):
            raise VerifyException("packed data comparison currently requires f32 UB sources")
        chunks = mask_chunks(shape)
        if (
            self.out.type.memory.data != "UB"
            or self.out.type.dtype.data != "u8"
            or tuple(size.data for size in self.out.type.shape) != (chunks * 32,)
            or tuple(size.data for size in self.value.type.shape) != (chunks * 32,)
            or self.value.type.dtype.data != "u8"
        ):
            raise VerifyException("data comparison needs explicit 32-byte packed predicate blocks")
        if tuple(size.data for size in self.mask.type.shape) != shape:
            raise VerifyException("predicate shape must match the compared tile")


@irdl_op_definition
class MaterializeMaskOp(IRDLOperation):
    """Write a symbolic coordinate predicate into explicit packed storage."""

    name = "npu_source.materialize_mask"
    mask = operand_def(MaskType)
    out = operand_def(TensorType)
    value = result_def(TileType)
    result = result_def(MaskType)

    def verify_(self):
        if not isinstance(self.mask.owner, MaskOp):
            raise VerifyException("only symbolic coordinate predicates need materialization")
        shape = tuple(size.data for size in self.mask.type.shape)
        chunks = mask_chunks(shape)
        if (
            self.out.type.memory.data != "UB"
            or self.out.type.dtype.data != "u8"
            or tuple(size.data for size in self.out.type.shape) != (chunks * 32,)
            or tuple(size.data for size in self.value.type.shape) != (chunks * 32,)
            or self.value.type.dtype.data != "u8"
            or tuple(size.data for size in self.result.type.shape) != shape
        ):
            raise VerifyException("predicate materialization needs explicit 32-byte packed blocks")


@irdl_op_definition
class MaskCombineOp(IRDLOperation):
    """A bitwise boolean combination of materialized packed predicates."""

    name = "npu_source.mask_combine"
    kind = attr_def(StringAttr)
    lhs = operand_def(MaskType)
    rhs = opt_operand_def(MaskType)
    lhs_storage = operand_def(TensorType)
    rhs_storage = opt_operand_def(TensorType)
    out = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    value = result_def(TileType)
    result = result_def(MaskType)
    irdl_options = (AttrSizedOperandSegments(),)

    def verify_(self):
        if self.kind.data not in {"and", "or", "not"}:
            raise VerifyException("mask combination requires and/or/not")
        binary = self.kind.data != "not"
        if binary != (self.rhs is not None and self.rhs_storage is not None):
            raise VerifyException("mask combination operand pack disagrees with its kind")
        operands = [self.lhs_storage]
        masks = [self.lhs]
        if binary:
            operands.append(self.rhs_storage)
            masks.append(self.rhs)
        _verify_dependencies(self, operands)
        shape = tuple(size.data for size in self.result.type.shape)
        chunks = mask_chunks(shape)
        for position, (mask, storage) in enumerate(zip(masks, operands, strict=True)):
            _verify_materialized_mask(mask, storage, self.dependencies[position])
            if tuple(size.data for size in mask.type.shape) != shape:
                raise VerifyException("combined predicates require equal lane shapes")
        if (
            self.out.type.memory.data != "UB"
            or self.out.type.dtype.data != "u8"
            or tuple(size.data for size in self.out.type.shape) != (chunks * 32,)
            or tuple(size.data for size in self.value.type.shape) != (chunks * 32,)
            or self.value.type.dtype.data != "u8"
            or tuple(size.data for size in self.result.type.shape) != shape
        ):
            raise VerifyException("mask combination needs explicit packed result blocks")
        if _mask_storage(self.out) in {_mask_storage(storage) for storage in operands}:
            raise VerifyException("mask combination output needs distinct predicate storage")


@irdl_op_definition
class WhereOp(IRDLOperation):
    """Select two legal input branches per element using packed predicate bits."""

    name = "npu_source.where"
    mask = operand_def(MaskType)
    mask_storage = operand_def(TensorType)
    mask_dependency = operand_def(TileType)
    x = operand_def(TensorType)
    y = operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    value = result_def(TileType)

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.mask_storage])
        if self.mask_dependency not in self.dependencies:
            raise VerifyException("where needs its actual predicate writer dependence")
        lanes = _verify_materialized_mask(
            self.mask, self.mask_storage, self.mask_dependency, message="where"
        )
        if tuple(size.data for size in self.mask.type.shape) != tuple(
            size.data for size in self.destination.type.shape
        ):
            raise VerifyException("where predicate must match its output shape")
        if not self.destination.type.shape:
            raise VerifyException("where requires a non-empty tile shape")
        for value in (self.x, self.y, self.destination):
            if value.type.shape != self.destination.type.shape:
                raise VerifyException("where requires equal tile shapes")
            if value.type.memory.data != "UB" or value.type.dtype.data != "f32":
                raise VerifyException("where currently requires f32 UB operands")
        if lanes != mask_chunks(tuple(size.data for size in self.destination.type.shape)):
            raise VerifyException("where predicate block count disagrees with its output")


@irdl_op_definition
class CopyOp(IRDLOperation):
    name = "npu_source.copy"
    source = operand_def(TensorType)
    destination = operand_def(TensorType)
    dependencies = var_operand_def(TileType)
    dependency_inputs = attr_def(ArrayAttr[IntAttr])
    transpose = opt_attr_def(IntAttr)
    zero_pad = opt_attr_def(IntAttr)
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        _verify_dependencies(self, [self.source])
        transpose = 0 if self.transpose is None else self.transpose.data
        if transpose not in {0, 1}:
            raise VerifyException("copy transpose policy must be boolean")
        if transpose and (
            self.source.type.memory.data != "L1"
            or self.destination.type.memory.data not in {"L0A", "L0B"}
            or len(self.source.type.shape) != 2
        ):
            raise VerifyException("transposed copy requires a complete L1→L0A/L0B matrix")
        if self.zero_pad is not None and self.zero_pad.data != 1:
            raise VerifyException("copy zero padding must be explicitly enabled")
        if not copy_compatible(
            self.source.type,
            self.destination.type,
            transpose=bool(transpose),
            zero_pad=self.zero_pad is not None,
        ):
            raise VerifyException(
                "copy requires compatible extents/types; use explicit cast or pad_value=0"
            )


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
    mask = opt_operand_def(MaskType)
    mask_storage = opt_operand_def(TensorType)
    mask_dependency = opt_operand_def(TileType)
    value = result_def(TileType)
    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        storage_inputs = () if self.mask_storage is None else (self.mask_storage,)
        _verify_dependencies(self, [*self.inputs, *storage_inputs])
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
            "mul_columns": 2,
            "mul_outer": 2,
            "add_columns": 2,
            "broadcast_columns": 1,
            "broadcast_rows": 1,
        }
        if self.kind.data not in arity or len(self.inputs) != arity[self.kind.data]:
            raise VerifyException("unknown element operation or incorrect operand count")
        if self.kind.data.endswith("_columns"):
            shape = tuple(n.data for n in self.destination.type.shape)
            vector = self.inputs[-1].type
            if (
                len(shape) != 2
                or not 1 <= shape[0] <= 255
                or shape[1] % 64
                or not 64 <= shape[1] <= 2040
                or tuple(n.data for n in vector.shape) != (shape[1],)
                or len(self.inputs) == 2
                and self.inputs[0].type.shape != self.destination.type.shape
                or self.scratch is not None
                or any(v.type.memory.data != "UB" for v in (*self.inputs, self.destination))
            ):
                raise VerifyException("column broadcast needs an f32 UB matrix and matching vector")
        elif self.kind.data == "mul_outer":
            shape = tuple(n.data for n in self.destination.type.shape)
            if (
                len(shape) != 2
                or not 8 <= shape[0] <= 248
                or shape[0] % 8
                or not 64 <= shape[1] <= 1984
                or shape[1] % 64
                or tuple(n.data for n in self.inputs[0].type.shape) != (shape[1],)
                or tuple(n.data for n in self.inputs[1].type.shape) != (shape[0],)
                or self.scratch is None
                or tuple(n.data for n in self.scratch.type.shape) != (shape[0], 8)
                or any(
                    v.type.memory.data != "UB" or v.type.dtype.data != "f32"
                    for v in (*self.inputs, self.destination, self.scratch)
                )
            ):
                raise VerifyException(
                    "outer product needs f32 UB vectors, complete rows/columns and (rows,8) scratch"
                )
        elif self.kind.data == "broadcast_rows":
            shape = tuple(n.data for n in self.destination.type.shape)
            if (
                len(shape) != 2
                or not 8 <= shape[0] <= 248
                or shape[0] % 8
                or shape[1] % 64
                or not 64 <= shape[1] <= 1024
                or tuple(n.data for n in self.inputs[0].type.shape) != (shape[0],)
                or self.scratch is None
                or tuple(n.data for n in self.scratch.type.shape) != (shape[0], 8)
                or any(
                    v.type.memory.data != "UB"
                    for v in (*self.inputs, self.destination, self.scratch)
                )
            ):
                raise VerifyException(
                    "row broadcast needs packed rows and explicit (rows,8) scratch"
                )
        elif self.kind.data.endswith("_rows"):
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
        elif self.scratch is not None and self.mask is None:
            raise VerifyException("only row broadcast arithmetic uses scratch")
        elif any(v.type.shape != self.destination.type.shape for v in self.inputs):
            raise VerifyException("element operands require equal tile shape")
        if (self.kind.data in {"fill", "mul_scalar", "add_scalar"}) != (self.scalar is not None):
            raise VerifyException("scalar policy must match element operation")
        if self.kind.data == "fill" and self.mask is None:
            if self.destination.type.dtype.data not in {"f32", "f16", "bf16"}:
                raise VerifyException("fill requires f32, f16 or bf16 operands")
        elif self.kind.data != "cast" and any(
            v.type.dtype.data != "f32" for v in (*self.inputs, self.destination)
        ):
            raise VerifyException("arithmetic currently requires explicit f32 operands")
        if self.mask is None:
            if self.mask_storage is not None or self.mask_dependency is not None:
                raise VerifyException("predicate storage requires an explicit mask")
            return
        if self.kind.data.endswith(("_rows", "_columns", "_outer")) or self.kind.data == "cast":
            raise VerifyException("masked row broadcast and conversion are not implemented")
        materialized = not isinstance(self.mask.owner, MaskOp)
        if self.scratch is not None and not materialized:
            raise VerifyException("coordinate masks need no temporary result tile")
        if materialized:
            if self.scratch is None:
                raise VerifyException(
                    "data-predicate updates need an explicit same-shape f32 scratch tile"
                )
            if (
                self.scratch.type.shape != self.destination.type.shape
                or self.scratch.type.memory.data != "UB"
                or self.scratch.type.dtype.data != "f32"
                or self.mask_dependency not in self.dependencies
            ):
                raise VerifyException(
                    "data-predicate updates need a same-shape f32 scratch and predicate dependence"
                )
            if self.mask_dependency not in self.dependencies:
                raise VerifyException("masked update lost its predicate writer dependence")
            carried = [
                index.data
                for dependency, index in zip(self.dependencies, self.dependency_inputs, strict=True)
                if dependency is self.mask_dependency
            ]
            if carried != [len(self.inputs)]:
                raise VerifyException("predicate dependence must name the packed storage operand")
        _verify_mask_operand(self, tuple(size.data for size in self.destination.type.shape))


def block8_reduction_widths(columns: int) -> tuple[int, ...]:
    """Exact packed planes for block8 stages followed by one native merge.

    Each existing block8 native consumes complete 64-element repeats. Its
    output rows must remain 32-byte aligned for the next stage and merge;
    padding or a different reduction tree must be selected explicitly.
    """
    if columns == 8:
        # The terminal datablock reduction packs its results directly. Its
        # whole-repeat row extent is checked by ReduceOp below.
        return (1,)
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
        block8 = self.order.data == "block8_then_packed_merge"
        narrow = block8 and cols == 8 and len(shape) == 2
        if narrow and (shape[0] % 8 or not 8 <= shape[0] <= 248):
            raise VerifyException("eight-column reduction needs 8-aligned rows in [8,248]")
        if not narrow and (cols < 64 or cols > 4096 or cols % 64):
            raise VerifyException("reduction needs 64-aligned columns in [64,4096]")
        if self.kind.data not in {"sum", "max"} or self.order.data not in {
            "r64_then_native_merge",
            "r64_then_packed_merge",
            "block8_then_packed_merge",
        }:
            raise VerifyException("reduction requires an explicit supported order")
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
class ScalarWriteOp(IRDLOperation):
    name = "npu_source.scalar_write"
    source = operand_def(f32)
    destination = operand_def(TensorType)
    value = result_def(TileType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        _verify_result(self, self.destination)
        if (
            self.destination.type.memory.data != "UB"
            or self.destination.type.dtype.data != "f32"
            or any(n.data != 1 for n in self.destination.type.shape)
        ):
            raise VerifyException("Scalar store requires one f32 UB element")


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
            self.source.type.memory.data not in {"UB", "GM"}
            or self.source.type.dtype.data not in {"f32", "f16", "bf16"}
            or self.source.type.memory.data == "UB"
            and self.source.type.dtype.data != "f32"
            or any(n.data != 1 for n in self.source.type.shape)
        ):
            raise VerifyException("Scalar read requires one f32 UB or floating GM element")


@irdl_op_definition
class IndexReadOp(IRDLOperation):
    name = "npu_source.index_read"
    source = operand_def(TensorType)
    minimum = operand_def(IndexType)
    maximum = operand_def(IndexType)
    value = result_def(IndexType)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        if (
            self.source.type.memory.data != "GM"
            or self.source.type.dtype.data not in {"i32", "i64"}
            or any(n.data != 1 for n in self.source.type.shape)
        ):
            raise VerifyException(
                "index read requires exactly one immutable GM int32/int64 element"
            )


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
            or self.a.type.dtype.data not in {"f16", "bf16", "f32"}
            or self.accumulator.type.dtype.data != "f32"
        ):
            raise VerifyException("matmul requires equal floating operands and f32 accumulator")


SOURCE = Dialect(
    "npu_source",
    [
        ProgramOp,
        AllocationOp,
        SizeOp,
        IterationOp,
        StageOp,
        DomainOp,
        ViewOp,
        PredicateOp,
        MaskOp,
        CopyOp,
        ElementOp,
        ReduceOp,
        CompareOp,
        MaterializeMaskOp,
        MaskCombineOp,
        WhereOp,
        ScalarReadOp,
        ScalarWriteOp,
        IndexReadOp,
        ScaleOp,
        MatmulOp,
        YieldOp,
    ],
    [TensorType, TileType, MaskType],
)

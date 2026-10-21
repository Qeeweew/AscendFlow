"""Typed source primitives to native tasks, independent of loop construction.

The caller supplies address/SSA bindings and the real insertion point through
three narrow capabilities. Recipes create no Stream, Call, Once, or synchronization
operations. Results retain actual native operations and operand positions rather
than encoding numerical provenance in generated names.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.source import (
    AllocationOp,
    CompareOp,
    CopyOp,
    ElementOp,
    IndexReadOp,
    MaskCombineOp,
    MaskOp,
    MaterializeMaskOp,
    MatmulOp,
    ReduceOp,
    ScalarReadOp,
    ScalarWriteOp,
    ScaleOp,
    ViewOp,
    WhereOp,
    block8_reduction_widths,
    copy_compatible,
    mask_chunks,
)
from ..mask import (
    MASK_REPEAT_POLICY,
    Program,
    iteration_strategy,
    scalar_operand_count,
    word_strategy,
)


class SourceView(Protocol):
    def __call__(self, value: SSAValue, *, matrix: bool = False) -> SSAValue: ...


class ScalarOperand(Protocol):
    def __call__(self, value: SSAValue) -> SSAValue: ...


class PhysicalSubview(Protocol):
    def __call__(
        self,
        source: SSAValue,
        nbytes: int,
        *,
        offset: int = 0,
        rows: int = 1,
        row_stride: int = 0,
    ) -> SSAValue: ...


class NativeEmitter(Protocol):
    def __call__(
        self,
        suffix: str,
        native: str,
        *refs: SSAValue,
        params: tuple[int, ...] = (),
        scalar: float = 0.0,
        order_delta: int = 0,
        operands: tuple[SSAValue, ...] = (),
        mask: Program | None = None,
        mask_word: str | None = None,
        mask_repeat: str | None = None,
        mask_layout: str | None = None,
    ) -> TaskOp: ...


@dataclass(frozen=True)
class RecipeContext:
    view: SourceView
    subview: PhysicalSubview
    emit: NativeEmitter
    operand: "ScalarOperand | None" = None
    load3d_b_width: int | None = None
    load3d_a_width: int | None = None
    aligned: Callable[[SSAValue, int], bool] | None = None


@dataclass(frozen=True)
class NativeRead:
    """A source input's actual native reader and physical operand position."""

    source_input: int
    operation: TaskOp
    ref_index: int


@dataclass(frozen=True)
class RecipeResult:
    tasks: tuple[TaskOp, ...]
    definition: TaskOp
    definition_ref: int | None
    readers: tuple[NativeRead, ...]
    scalar_value: SSAValue | None = None


def source_shape(value):
    return tuple(n.data for n in value.type.shape)


def source_allocation(value):
    owner = value.owner
    require(isinstance(owner, ViewOp), "source work requires explicit typed views")
    root = owner.source.owner
    require(isinstance(root, AllocationOp), "view must retain one allocation identity")
    return root


def strided_b_width(op):
    if (
        not isinstance(op, CopyOp)
        or (op.source.type.memory.data, op.destination.type.memory.data) != ("L1", "L0B")
        or op.transpose is not None
        and op.transpose.data
        or len(op.source.type.shape) != 2
    ):
        return None
    rows, cols = source_shape(op.source)
    width = source_allocation(op.source).ref.type.shape.data[-2].data
    # Load3D handles this strided zN rectangle in one instruction. Complete
    # planes and one-fractal dimensions keep their existing Load2D recipe.
    return (
        width if (16 < rows < width and cols > 16) or op.source.type.dtype.data == "f32" else None
    )


def compact_a_width(op):
    if (
        not isinstance(op, CopyOp)
        or (op.source.type.memory.data, op.destination.type.memory.data) != ("L1", "L0A")
        or len(op.source.type.shape) != 2
    ):
        return None
    rows, _ = source_shape(op.source)
    if op.source.type.dtype.data == "f32":
        return source_allocation(op.source).ref.type.shape.data[-2].data
    return (
        rows
        if rows > 16 and source_allocation(op.source).ref.type.shape.data[-2].data == rows
        else None
    )


def lower_copy(op: CopyOp, context: RecipeContext) -> RecipeResult:
    require(
        source_allocation(op.destination).access.data != "read", "copy writes a read-only tensor"
    )
    require(
        copy_compatible(
            op.source.type,
            op.destination.type,
            transpose=op.transpose is not None and bool(op.transpose.data),
            zero_pad=op.zero_pad is not None,
        ),
        "copy needs compatible extents/types, explicit padding or cast",
    )
    banks = op.source.type.memory.data, op.destination.type.memory.data
    prefix = op.source.owner.valid
    if prefix is not None and len(source_shape(op.source)) == 2:
        require(context.operand is not None, "runtime row copies need their actual scalar operand")
        if banks == ("GM", "L1"):
            if op.zero_pad is not None:
                require(
                    op.destination.owner.valid is None,
                    "padded ND2NZ defines the complete destination",
                )
                native = "nd2nz_rows"
                destination = context.view(op.destination)
                params = source_shape(op.source) + source_shape(op.destination)
            else:
                require(
                    op.destination.owner.valid is prefix,
                    "ND2NZ overlay requires the same runtime prefix",
                )
                native = "nd2nz_overlay"
                destination = context.view(op.destination, matrix=True)
                params = source_shape(op.source) + (
                    source_allocation(op.destination).ref.type.shape.data[-2].data,
                )
        else:
            require(
                banks == ("UB", "GM")
                and op.destination.owner.valid is prefix
                and op.zero_pad is None,
                "runtime output rows require the same UB→GM prefix",
            )
            native = "copy_rows_prefix"
            destination = context.view(op.destination, matrix=True)
            params = ()
        task = context.emit(
            "_copy",
            native,
            context.view(op.source, matrix=True),
            destination,
            params=params,
            operands=(context.operand(prefix),),
        )
        return RecipeResult((task,), task, 1, (NativeRead(0, task, 0),))
    if banks == ("GM", "L1"):
        require(len(source_shape(op.source)) == 2, "ND→NZ copy requires a matrix")
        padded = op.zero_pad is not None and source_shape(op.source) != source_shape(op.destination)
        task = context.emit(
            "_copy",
            "nd2nz_pad" if padded else "nd2nz",
            context.view(op.source, matrix=True),
            context.view(op.destination),
            params=source_shape(op.source) + (source_shape(op.destination) if padded else ()),
        )
    elif banks in {("L1", "L0A"), ("L1", "L0B")}:
        transpose = op.transpose is not None and op.transpose.data
        native = (
            ("load2d_a_t" if transpose else "load2d_a")
            if banks[1] == "L0A"
            else "load2d_b_nt"
            if transpose
            else "load2d_b_nn"
        )
        if context.load3d_b_width is not None and strided_b_width(op) == context.load3d_b_width:
            native = "load3d_b_nn"
        if (
            not transpose
            and context.load3d_a_width is not None
            and compact_a_width(op) == context.load3d_a_width
        ):
            native = "load3d_a"
        task = context.emit(
            "_copy",
            native,
            context.view(op.source),
            context.view(op.destination),
            params=source_shape(op.source),
        )
    elif banks == ("L0C", "GM"):
        source = context.view(op.source)
        params = source_shape(op.source)
        if source_shape(op.source) != source_shape(op.destination):
            m, n = source_shape(op.destination)
            stride = source_shape(op.source)[0]
            source = context.subview(
                source, m * 16 * 4, rows=(n + 15) // 16, row_stride=stride * 16 * 4
            )
            params = (m, n, stride)
        task = context.emit(
            "_export",
            "fix",
            source,
            context.view(op.destination, matrix=True),
            params=params,
        )
    else:
        source, destination = context.view(op.source), context.view(op.destination)
        dimensions = source_shape(op.source)
        padded = len(dimensions) == 2 and any(
            ref.owner.rows.data > 1 for ref in (source, destination)
        )
        if padded:
            # Keep precise rows even when the other side is compact.
            source, destination = (
                context.view(op.source, matrix=True),
                context.view(op.destination, matrix=True),
            )
        native = (
            "vector_copy_rows"
            if padded and banks == ("UB", "UB")
            else "copy_rows"
            if padded
            else "copy"
        )
        if native == "copy" and banks in {("GM", "UB"), ("UB", "GM")}:
            gm = source if banks[0] == "GM" else destination
            if context.aligned is None or not context.aligned(gm, 32):
                native = "copy_bytes"
        task = context.emit("_copy", native, source, destination)
    return RecipeResult((task,), task, 1, (NativeRead(0, task, 0),))


def lower_element(op: ElementOp, context: RecipeContext) -> RecipeResult:
    require(
        source_allocation(op.destination).access.data != "read",
        "arithmetic writes a read-only tensor",
    )
    native = {
        "mul_scalar": "muls",
        "add_scalar": "adds",
        "fill": "fill",
        "mul": "mul",
        "add": "add",
        "sub": "sub",
        "exp": "exp",
        "sqrt": "sqrt",
        "rsqrt": "rsqrt",
        "div": "div",
        "maximum": "max",
    }.get(op.kind.data)
    if op.kind.data.endswith("_columns"):
        rows, cols = source_shape(op.destination)
        task = context.emit(
            "_columns",
            op.kind.data,
            *(context.view(value) for value in (*op.inputs, op.destination)),
            params=(rows, cols),
        )
        return RecipeResult(
            (task,),
            task,
            len(op.inputs),
            tuple(NativeRead(i, task, i) for i in range(len(op.inputs))),
        )
    if op.kind.data == "broadcast_rows":
        rows, cols = source_shape(op.destination)
        source, temporary, destination = (
            context.view(value) for value in (op.inputs[0], op.scratch, op.destination)
        )
        first = context.emit("_expand", "brcb8", source, temporary, params=(rows,))
        last = context.emit(
            "_rows", "broadcast_rows", temporary, destination, params=(rows, cols), order_delta=1
        )
        return RecipeResult((first, last), last, 1, (NativeRead(0, first, 0),))
    if op.kind.data.endswith("_rows") or op.kind.data == "mul_outer":
        require(op.scratch is not None, "row broadcast requires scratch")
        rows, cols = source_shape(op.destination)
        source, statistic, destination = (
            context.view(value) for value in (*op.inputs, op.destination)
        )
        temporary = context.view(op.scratch)
        broadcast = context.emit("_broadcast", "brcb8", statistic, temporary, params=(rows,))
        task = context.emit(
            "_" + op.kind.data,
            "mul_outer"
            if op.kind.data == "mul_outer"
            else op.kind.data.removesuffix("_rows") + "_row_broadcast",
            source,
            temporary,
            destination,
            params=(rows, cols),
            order_delta=1,
        )
        return RecipeResult(
            (broadcast, task), task, 2, (NativeRead(0, task, 0), NativeRead(1, broadcast, 0))
        )
    if op.kind.data == "cast":
        require(len(op.inputs) == 1, "cast takes one source tile")
        source_dtype, destination_dtype = (
            op.inputs[0].type.dtype.data,
            op.destination.type.dtype.data,
        )
        require(source_dtype != destination_dtype, "identity cast needs no conversion operation")
        native = "convert_f32" if destination_dtype == "f32" else "convert_rn"
    require(native is not None, f"unsupported typed element operation {op.kind.data}")
    if op.mask is not None:
        return lower_masked_element(op, context, native)
    require(op.scratch is None, "scratch requires a data predicate or row broadcast")
    if native == "fill" and op.destination.type.memory.data == "L1":
        require(op.scalar.value.data == 0.0, "L1 fill currently requires zero")
        native = "zero_l1"
    refs = [context.view(value) for value in (*op.inputs, op.destination)]
    if native == "add" and any(ref.owner.rows.data > 1 for ref in refs):
        native = "add_strided"
        refs = [context.view(value, matrix=True) for value in (*op.inputs, op.destination)]
    task = context.emit(
        "_" + op.kind.data,
        native,
        *refs,
        scalar=0.0 if op.scalar is None else op.scalar.value.data,
    )
    return RecipeResult(
        (task,),
        task,
        len(op.inputs),
        tuple(NativeRead(i, task, i) for i in range(len(op.inputs))),
    )


@dataclass(frozen=True)
class MaskInfo:
    """One resolved predicate operand, before native selection."""

    materialized: bool
    shape: tuple[int, ...]
    chunks: int
    program: Program | None = None
    scalars: tuple[SSAValue, ...] = ()


def resolve_mask(value, context):
    require(context.operand is not None, "predicate lowering needs actual scalar bindings")
    producer = value.owner
    if isinstance(producer, MaskOp):
        program = producer.program()
        count = scalar_operand_count(program.nodes)
        require(count == len(producer.scalars), "predicate scalar atoms lost their operands")
        scalars = tuple(context.operand(item) for item in producer.scalars)
        return MaskInfo(False, program.shape, program.lanes // 64, program, scalars)
    if isinstance(producer, (CompareOp, MaterializeMaskOp, MaskCombineOp)):
        shape = source_shape(value)
        return MaskInfo(True, shape, mask_chunks(shape))
    require(False, "mask operand lost its symbolic or materialized producer")
    raise AssertionError("unreachable")


def lower_masked_element(op: ElementOp, context: RecipeContext, native: str) -> RecipeResult:
    info = resolve_mask(op.mask, context)
    scalar = 0.0 if op.scalar is None else op.scalar.value.data
    inputs = tuple(context.view(value) for value in op.inputs)
    destination = context.view(op.destination)
    if not info.materialized:
        task = context.emit(
            "_" + op.kind.data,
            "mask_" + native,
            *inputs,
            destination,
            scalar=scalar,
            operands=info.scalars,
            mask=info.program,
            mask_word=word_strategy(info.program),
            mask_repeat=MASK_REPEAT_POLICY,
            mask_iteration=iteration_strategy(info.program),
        )
        return RecipeResult(
            (task,),
            task,
            len(inputs),
            tuple(NativeRead(i, task, i) for i in range(len(inputs))),
        )
    require(op.scratch is not None, "data-predicate update needs an explicit scratch tile")
    scratch = context.view(op.scratch)
    compute = context.emit(
        "_" + op.kind.data,
        native,
        *inputs,
        scratch,
        scalar=scalar,
    )
    select = context.emit(
        "_select",
        "select_masked_update",
        context.view(op.mask_storage),
        scratch,
        destination,
        params=(info.chunks,),
        order_delta=1,
        mask_layout="packed32_block_read",
    )
    readers = tuple(NativeRead(i, compute, i) for i in range(len(inputs))) + (
        NativeRead(len(inputs), select, 0),
    )
    return RecipeResult((compute, select), select, 2, readers)


def lower_compare(op: CompareOp, context: RecipeContext) -> RecipeResult:
    shape = source_shape(op.source)
    chunks = mask_chunks(shape)
    scalar = 0.0 if op.scalar is None else op.scalar.value.data
    native = ("compare_" if op.other is not None else "compare_scalar_") + op.kind.data
    refs = [context.view(op.source)]
    if op.other is not None:
        refs.append(context.view(op.other))
    out_index = len(refs)
    block = context.view(op.out)
    # Packed padding is initialized by real declared work: one count-mode
    # Vector Dup over the whole 32-byte block. The compare then writes only
    # the 64-bit lane word, so the same-Vector WAW between the two actual
    # tasks is a required dependency of ordinary analysis, not a hidden store
    # or an emitter barrier.
    zero = context.emit("_mask_zero", "mask_zero", block, params=(chunks * 16,))
    refs.append(block)
    task = context.emit(
        "_compare",
        native,
        *refs,
        params=(chunks,),
        scalar=scalar,
        order_delta=1,
        mask_layout="packed32_zeroed",
    )
    return RecipeResult(
        (zero, task),
        task,
        out_index,
        tuple(NativeRead(i, task, i) for i in range(out_index)),
    )


def lower_materialize_mask(op: MaterializeMaskOp, context: RecipeContext) -> RecipeResult:
    producer = op.mask.owner
    require(isinstance(producer, MaskOp), "materialization lost its symbolic program")
    program = producer.program()
    require(scalar_operand_count(program.nodes) == len(producer.scalars), "lost predicate scalars")
    scalars = tuple(context.operand(item) for item in producer.scalars)
    chunks = mask_chunks(program.shape)
    task = context.emit(
        "_mask_bits",
        "mask_bits",
        context.view(op.out),
        params=(chunks,),
        operands=scalars,
        mask=program,
        mask_word=word_strategy(program),
    )
    return RecipeResult((task,), task, 0, ())


def lower_mask_combine(op: MaskCombineOp, context: RecipeContext) -> RecipeResult:
    shape = source_shape(op.result)
    chunks = mask_chunks(shape)
    refs = [context.view(op.lhs_storage)]
    if op.rhs_storage is not None:
        refs.append(context.view(op.rhs_storage))
    out_index = len(refs)
    refs.append(context.view(op.out))
    task = context.emit(
        "_mask_" + op.kind.data,
        "mask_" + op.kind.data,
        *refs,
        params=(chunks * 16,),
    )
    return RecipeResult(
        (task,),
        task,
        out_index,
        tuple(NativeRead(i, task, i) for i in range(out_index)),
    )


def lower_where(op: WhereOp, context: RecipeContext) -> RecipeResult:
    shape = source_shape(op.destination)
    chunks = mask_chunks(shape)
    task = context.emit(
        "_select",
        "select_masked",
        context.view(op.mask_storage),
        context.view(op.x),
        context.view(op.y),
        context.view(op.destination),
        params=(chunks,),
        mask_layout="packed32_block_read",
    )
    return RecipeResult((task,), task, 3, (NativeRead(0, task, 0),))


def lower_reduce(op: ReduceOp, context: RecipeContext) -> RecipeResult:
    require(
        source_allocation(op.scratch).access.data != "read"
        and source_allocation(op.destination).access.data != "read",
        "reduction writes a read-only buffer",
    )
    rows, cols = (
        (1, source_shape(op.source)[0])
        if len(source_shape(op.source)) == 1
        else source_shape(op.source)
    )
    source, temporary, destination = (
        context.view(value, matrix=True) for value in (op.source, op.scratch, op.destination)
    )
    tasks = []
    block8 = op.order.data == "block8_then_packed_merge"
    if block8 and cols == 8:
        task = context.emit("_block8", "reduce_" + op.kind.data + "8", source, destination)
        return RecipeResult((task,), task, 1, (NativeRead(0, task, 0),))
    if block8:
        workspace, byte_offset = temporary, 0
        widths = block8_reduction_widths(cols)
        for level, columns in enumerate(widths):
            temporary = context.subview(
                workspace,
                columns * 4,
                offset=byte_offset,
                rows=rows,
                row_stride=columns * 4,
            )
            tasks.append(
                context.emit(
                    "_partial" if level == 0 else f"_block8_{level}",
                    "reduce_" + op.kind.data + "8",
                    source,
                    temporary,
                    order_delta=level,
                )
            )
            source = temporary
            byte_offset += rows * columns * 4
        partials, merge_order = widths[-1], len(widths)
    else:
        tasks.append(
            context.emit(
                "_partial",
                "reduce_" + op.kind.data + "64",
                source,
                temporary,
                params=(rows, cols),
            )
        )
        partials, merge_order = cols // 64, 1
    merge = context.emit(
        "_merge",
        "merge_"
        + op.kind.data
        + ("_packed" if block8 or op.order.data == "r64_then_packed_merge" else ""),
        temporary,
        destination,
        params=(rows, partials),
        order_delta=merge_order,
    )
    tasks.append(merge)
    return RecipeResult(tuple(tasks), merge, 1, (NativeRead(0, tasks[0], 0),))


def lower_scalar_read(op: ScalarReadOp, context: RecipeContext) -> RecipeResult:
    task = context.emit("_capture", "scalar_load", context.view(op.source))
    require(task.value is not None, "native Scalar reader must produce its actual f32 SSA value")
    return RecipeResult((task,), task, None, (NativeRead(0, task, 0),), task.value)


def lower_scalar_write(
    op: ScalarWriteOp, context: RecipeContext, *, value: SSAValue
) -> RecipeResult:
    task = context.emit("_store", "scalar_store", context.view(op.destination), operands=(value,))
    return RecipeResult((task,), task, 0, ())


def lower_index_read(op: IndexReadOp, context: RecipeContext, *, minimum, maximum) -> RecipeResult:
    root = source_allocation(op.source)
    require(
        root.ref.type.memory.data == "UB" or root.access.data == "read",
        "GM index metadata must be immutable",
    )
    task = context.emit(
        "_index", "index_load", context.view(op.source), operands=(minimum, maximum)
    )
    return RecipeResult((task,), task, None, (NativeRead(0, task, 0),), task.value)


def lower_scale(op: ScaleOp, context: RecipeContext, *, factor: SSAValue) -> RecipeResult:
    require(
        source_allocation(op.destination).access.data != "read", "scale writes a read-only buffer"
    )
    task = context.emit(
        "_scale",
        "scale_value",
        context.view(op.source),
        context.view(op.destination),
        operands=(factor,),
    )
    return RecipeResult((task,), task, 1, (NativeRead(0, task, 0),))


Initialization = int | Literal["first"]


def matmul_dimensions(op: MatmulOp) -> tuple[int, int, int]:
    m, k = source_shape(op.a)
    n, bk = (
        source_shape(op.b)
        if op.transpose_b.data
        else (source_shape(op.b)[1], source_shape(op.b)[0])
    )
    require(k == bk and source_shape(op.accumulator) == (m, n), "gemm shapes disagree")
    return m, n, k


def lower_matmul(
    op: MatmulOp, context: RecipeContext, *, initialization: Initialization
) -> RecipeResult:
    """The explicit L0 recipe; operand packing is expressed by separate copies."""
    require(
        (op.a.type.memory.data, op.b.type.memory.data, op.accumulator.type.memory.data)
        == ("L0A", "L0B", "L0C"),
        "native matmul recipe requires explicit L0A/L0B/L0C operands",
    )
    m, n, k = matmul_dimensions(op)
    task = context.emit(
        "_mad",
        "mad_first" if initialization == "first" else "mad",
        context.view(op.a),
        context.view(op.b),
        context.view(op.accumulator),
        params=(m, n, k) if initialization == "first" else (m, n, k, initialization),
    )
    readers = [NativeRead(0, task, 0), NativeRead(1, task, 1)]
    if initialization != 1:
        readers.append(NativeRead(2, task, 2))
    return RecipeResult((task,), task, 2, tuple(readers))

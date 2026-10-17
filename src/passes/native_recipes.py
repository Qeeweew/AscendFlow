"""Typed source primitives to native tasks, independent of loop construction.

The caller supplies address/SSA bindings and the real insertion point through
three narrow capabilities. Recipes create no Stream, Call, Once, or synchronization
operations. Results retain actual native operations and operand positions rather
than encoding numerical provenance in generated names.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.source import (
    AllocationOp,
    CopyOp,
    ElementOp,
    MatmulOp,
    ReduceOp,
    ScalarReadOp,
    ScaleOp,
    ViewOp,
    block8_reduction_widths,
    copy_compatible,
)


class SourceView(Protocol):
    def __call__(self, value: SSAValue, *, matrix: bool = False) -> SSAValue: ...


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
    ) -> TaskOp: ...


@dataclass(frozen=True)
class RecipeContext:
    view: SourceView
    subview: PhysicalSubview
    emit: NativeEmitter
    load3d_b_width: int | None = None
    load3d_a_width: int | None = None


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
    return width if 16 < rows < width and cols > 16 else None


def compact_a_width(op):
    if (
        not isinstance(op, CopyOp)
        or (op.source.type.memory.data, op.destination.type.memory.data) != ("L1", "L0A")
        or len(op.source.type.shape) != 2
    ):
        return None
    rows, _ = source_shape(op.source)
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
        native = "load2d_a" if banks[1] == "L0A" else "load2d_b_nt" if transpose else "load2d_b_nn"
        if context.load3d_b_width is not None and strided_b_width(op) == context.load3d_b_width:
            native = "load3d_b_nn"
        if context.load3d_a_width is not None and compact_a_width(op) == context.load3d_a_width:
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
        task = context.emit("_copy", "copy_rows" if padded else "copy", source, destination)
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
    if op.kind.data.endswith("_rows"):
        require(op.scratch is not None, "row broadcast requires scratch")
        rows, cols = source_shape(op.destination)
        source, statistic, destination = (
            context.view(value) for value in (*op.inputs, op.destination)
        )
        temporary = context.view(op.scratch)
        broadcast = context.emit("_broadcast", "brcb8", statistic, temporary, params=(rows,))
        task = context.emit(
            "_" + op.kind.data,
            op.kind.data.removesuffix("_rows") + "_row_broadcast",
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
    refs = [context.view(value) for value in (*op.inputs, op.destination)]
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


def lower_scale(op: ScaleOp, context: RecipeContext, *, factor: SSAValue) -> RecipeResult:
    require(
        source_allocation(op.destination).access.data != "read", "scale writes a read-only buffer"
    )
    task = context.emit(
        "_scale", "scale_value", context.view(op.source), context.view(op.destination), factor
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

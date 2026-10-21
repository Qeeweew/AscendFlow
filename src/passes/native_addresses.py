"""Native row geometry at a real, scoped SSA byte offset.

Only static representability and obvious constant errors are checked here.
Original axes survive as SSA metadata for subsequent bounds/ownership
admission. Dynamic addresses are neither split into parent captures nor
reinterpreted in a synthetic single-generation coordinate.
"""

from collections.abc import Callable, Mapping
from math import prod

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr
from xdsl.ir import Operation, SSAValue

from ..diagnostics import ErrorKind, require
from ..dialects.physical import BufferOp
from ..dialects.source import AllocationOp, ViewOp, _constant_index
from ..dialects.structured import AddressOp
from ..target.native import ITEMSIZE, nz_columns

Index = int | SSAValue
Append = Callable[[Operation], None]
_MAX_INDEX = (1 << 63) - 1


def _require(condition, message, *, unsupported=False):
    require(
        condition,
        message,
        kind=ErrorKind.UNSUPPORTED if unsupported else ErrorKind.INVALID,
        phase="native-address",
    )


def _constant(value: Index) -> int | None:
    return value if type(value) is int else _constant_index(value)


class _Indices:
    """Small local arithmetic builder; caches only this invocation's constants."""

    def __init__(self, append: Append):
        self.append = append
        self.constants = {}

    def value(self, value: Index) -> SSAValue:
        if isinstance(value, SSAValue):
            _require(isinstance(value.type, IndexType), "native byte addresses require index SSA")
            return value
        _require(
            type(value) is int and 0 <= value <= _MAX_INDEX, "static address exceeds index range"
        )
        if value not in self.constants:
            operation = arith.ConstantOp.from_int_and_width(value, IndexType())
            self.append(operation)
            self.constants[value] = operation.result
        return self.constants[value]

    def binary(self, kind, lhs: Index, rhs: Index) -> Index:
        left, right = _constant(lhs), _constant(rhs)
        if left is not None and right is not None:
            result = left + right if kind is arith.AddiOp else left * right
            _require(0 <= result <= _MAX_INDEX, "static address arithmetic exceeds index range")
            return result
        if kind is arith.AddiOp:
            if left == 0:
                return rhs
            if right == 0:
                return lhs
        else:
            if left == 0 or right == 0:
                return 0
            if left == 1:
                return rhs
            if right == 1:
                return lhs
        operation = kind(self.value(lhs), self.value(rhs))
        self.append(operation)
        return operation.result


def _geometry(nbytes, rows, row_stride):
    _require(
        all(type(value) is int for value in (nbytes, rows, row_stride)),
        "native row geometry must be static integers",
        unsupported=True,
    )
    _require(nbytes > 0 and rows > 0 and row_stride >= 0, "native rows need positive width/count")
    _require(rows == 1 or row_stride >= nbytes, "native address rows must not overlap")
    span = nbytes + (rows - 1) * row_stride
    _require(span <= _MAX_INDEX, "static native footprint exceeds index range")
    return span


def _root(value):
    owner = value.owner
    _require(
        isinstance(owner, BufferOp) and value is owner.ref,
        "native address needs its original BufferOp allocation root",
    )
    return owner


def lower_address(
    value: SSAValue,
    *,
    materialize: Callable[[SSAValue], SSAValue],
    roots: Mapping[AllocationOp, SSAValue],
    append: Append,
    matrix: bool = False,
) -> SSAValue:
    """Lower one source view with its caller's actual lexical SSA bindings.

    Static strides determine native row geometry. Dynamic leading dimensions
    remain ordinary SSA products in the full byte offset and axis metadata.
    The caller still owes dynamic bounds, alignment, and ownership admission.
    """
    view = value.owner
    _require(isinstance(view, ViewOp) and value is view.ref, "native work needs a source ViewOp")
    allocation = view.source.owner
    _require(isinstance(allocation, AllocationOp), "source view needs its original allocation")
    _require(
        view.valid is None or matrix and len(value.type.shape) == 2,
        "valid prefix native addresses are not supported for this operation",
        unsupported=True,
    )
    typ, original = value.type, allocation.ref.type
    _require(
        (typ.memory, typ.dtype, typ.layout) == (original.memory, original.dtype, original.layout),
        "source view cannot change memory, dtype, or layout",
    )
    _require(typ.dtype.data in ITEMSIZE, "native address has an unknown element type")
    itemsize = ITEMSIZE[typ.dtype.data]
    shape = tuple(size.data for size in original.shape)
    widths = tuple(width.data for width in view.widths)
    axes = tuple(axis.data for axis in view.axes)
    dimensions = tuple(size.data for size in typ.shape)
    _require(
        len(shape) == len(view.origins) == len(widths)
        and all(size == -1 or size > 0 for size in shape)
        and all(width > 0 for width in widths)
        and axes == tuple(sorted(set(axes)))
        and all(0 <= axis < len(shape) for axis in axes)
        and dimensions == tuple(widths[axis] for axis in axes)
        and all(width == 1 for axis, width in enumerate(widths) if axis not in axes),
        "native source view must retain its original axis geometry",
    )
    _require(len(dimensions) in {1, 2}, "native tile rank must be one or two", unsupported=True)
    dynamic_axes = (
        () if allocation.dynamic_axes is None else tuple(a.data for a in allocation.dynamic_axes)
    )
    _require(
        dynamic_axes == tuple(axis for axis, size in enumerate(shape) if size == -1)
        and len(dynamic_axes) == len(allocation.dimensions),
        "dynamic allocation axes need their actual SSA extents",
    )
    dynamic = dict(zip(dynamic_axes, allocation.dimensions, strict=True))
    extents = tuple(dynamic.get(axis, size) for axis, size in enumerate(shape))
    static_extents = tuple(_constant(extent) for extent in extents)
    for axis, (origin, width, extent) in enumerate(
        zip(view.origins, widths, static_extents, strict=True)
    ):
        start = _constant(origin)
        _require(start is None or start >= 0, "static view origin must be nonnegative")
        _require(extent is None or extent > 0, "static allocation extent must be positive")
        _require(
            view.valid is not None
            and axis == axes[0]
            or extent is None
            or width <= extent
            and (start is None or start + width <= extent),
            "static view origin and width exceed their original axis",
        )
    strides, stride = [], itemsize
    for extent in reversed(static_extents):
        strides.insert(0, stride)
        stride = None if stride is None or extent is None else stride * extent
    selected = tuple(strides[axis] for axis in axes)
    _require(
        all(s is not None for s in selected), "native row pitch must be static", unsupported=True
    )
    _require(selected[-1] == itemsize, "native tile columns must be contiguous", unsupported=True)
    contiguous = len(dimensions) == 1 or selected[0] == dimensions[1] * itemsize
    if typ.layout.data != "linear":
        column_slice = typ.memory.data == "L1" and typ.layout.data == "NZ"
        # Full M retains the NZ column-block pitch: an aligned L0C column
        # interval is itself compact and needs no strided MAD addressing.
        accumulator_columns = typ.memory.data == "L0C" and typ.layout.data == "NZ"
        _require(
            len(shape) >= 2
            and len(dimensions) == 2
            and axes == (len(shape) - 2, len(shape) - 1)
            and (column_slice or dimensions[0] == shape[-2])
            and (column_slice or accumulator_columns or dimensions[1] == shape[-1])
            and dimensions[0] > 0
            and (column_slice or dimensions[0] % 16 == 0)
            and dimensions[1] > 0
            and dimensions[1] % 16 == 0
            and (column_slice or _constant(view.origins[-2]) == 0)
            and (column_slice or accumulator_columns or _constant(view.origins[-1]) == 0),
            "fractal view needs a complete matrix plane, aligned L1 NZ rectangle or full-row L0C columns",
            unsupported=True,
        )
    if typ.layout.data == "NZ" and (dimensions[0] != shape[-2] or view.valid is not None):
        c0 = nz_columns(typ.memory.data, typ.dtype.data)
        nbytes, rows, row_stride = (
            dimensions[0] * c0 * itemsize,
            dimensions[1] // c0,
            shape[-2] * c0 * itemsize,
        )
    elif len(dimensions) == 2 and typ.layout.data == "linear" and (matrix or not contiguous):
        rows, columns = dimensions
        nbytes, row_stride = columns * itemsize, selected[0]
    else:
        nbytes, rows, row_stride = prod(dimensions) * itemsize, 1, 0
    _geometry(nbytes, rows, row_stride)
    _require(allocation in roots, "source allocation has no physical root binding")
    root = roots[allocation]
    _root(root)
    _require(
        (root.type.bank, root.type.dtype, root.type.layout) == (typ.memory, typ.dtype, typ.layout),
        "physical root cannot change source memory, dtype, or layout",
    )
    indices = _Indices(append)
    origins = tuple(indices.value(materialize(origin)) for origin in view.origins)
    actual_extents = tuple(
        indices.value(materialize(extent) if isinstance(extent, SSAValue) else extent)
        for extent in extents
    )
    offset, stride = 0, itemsize
    pairs = tuple(zip(origins, actual_extents, strict=True))
    for axis in range(len(pairs) - 1, -1, -1):
        origin, extent = pairs[axis]
        physical_stride = stride
        if typ.layout.data == "NZ" and axis == len(pairs) - 1:
            physical_stride = indices.binary(arith.MuliOp, stride, shape[-2])
        elif typ.layout.data == "NZ" and axis == len(pairs) - 2:
            physical_stride = nz_columns(typ.memory.data, typ.dtype.data) * itemsize
        offset = indices.binary(
            arith.AddiOp, offset, indices.binary(arith.MuliOp, origin, physical_stride)
        )
        if axis:
            stride = indices.binary(arith.MuliOp, extent, stride)
    address = AddressOp.build(
        operands=[
            root,
            indices.value(offset),
            origins,
            actual_extents,
            None if view.valid is None else materialize(view.valid),
        ],
        attributes={
            "nbytes": IntAttr(nbytes),
            "rows": IntAttr(rows),
            "row_stride": IntAttr(row_stride),
            "widths": view.widths,
            "axes": view.axes,
            "displacement": IntAttr(0),
        },
        result_types=[root.type],
    )
    address.location = view.location
    append(address)
    return address.ref


def subview(
    source: SSAValue,
    nbytes: int,
    *,
    offset: int = 0,
    rows: int = 1,
    row_stride: int = 0,
    append: Append,
) -> SSAValue:
    """A static recipe slice, still rooted directly in the original BufferOp."""
    _require(type(offset) is int and offset >= 0, "native subview offset must be nonnegative")
    span = _geometry(nbytes, rows, row_stride)
    owner = source.owner
    indices = _Indices(append)
    origins, extents = (), ()
    widths = axes = displacement = None
    if isinstance(owner, AddressOp):
        _require(
            owner.valid_rows is None, "runtime row-prefix addresses cannot be statically sliced"
        )
        _require(source is owner.ref, "native subview needs the actual address result")
        root, base = owner.source, owner.offset
        _root(root)
        parent_span = _geometry(owner.nbytes.data, owner.rows.data, owner.row_stride.data)
        _require(offset + span <= parent_span, "static native subview exceeds its parent footprint")
        pitch, width = owner.row_stride.data, owner.nbytes.data
        if owner.rows.data > 1 and pitch > width:
            first, within = divmod(offset, pitch)
            same_row = within + span <= width
            _require(
                same_row or rows == 1 or row_stride % pitch == 0,
                "subview across pitched rows needs an explicit physical row mapping",
                unsupported=True,
            )
            _require(
                within + nbytes <= width
                and (same_row or first + (rows - 1) * row_stride // pitch < owner.rows.data),
                "static native subview crosses a parent row gap",
            )
        if root.type.layout.data != "linear" and not (
            root.type.bank.data == "L0C" and root.type.layout.data == "NZ"
        ):
            _require(
                offset == 0 and rows == 1 and nbytes == owner.nbytes.data,
                "fractal subview must retain the complete matrix plane",
                unsupported=True,
            )
        origins, extents = tuple(owner.origins), tuple(owner.extents)
        if origins:
            widths, axes = owner.widths, owner.axes
            displacement = owner.displacement.data + offset
            _require(displacement <= _MAX_INDEX, "static subview displacement exceeds index range")
            displacement = IntAttr(displacement)
    else:
        buffer = _root(source)
        root, base = source, 0
        if buffer.extent_count is None:
            _require(
                offset + span <= buffer.nbytes.data, "static native subview exceeds its allocation"
            )
        if root.type.layout.data != "linear":
            _require(
                offset == 0 and rows == 1 and nbytes == buffer.nbytes.data,
                "fractal subview needs a complete matrix plane address",
                unsupported=True,
            )
    address = AddressOp.build(
        operands=[
            root,
            indices.value(indices.binary(arith.AddiOp, base, offset)),
            origins,
            extents,
            None,
        ],
        attributes={
            "nbytes": IntAttr(nbytes),
            "rows": IntAttr(rows),
            "row_stride": IntAttr(row_stride),
            "widths": widths,
            "axes": axes,
            "displacement": displacement,
        },
        result_types=[root.type],
    )
    address.location = owner.location
    append(address)
    return address.ref

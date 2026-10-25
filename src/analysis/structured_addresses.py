"""Decode retained native addresses without flattening their SSA coordinates.

The caller supplies one canonical scalar environment for the actual structured
control. Decoding preserves allocation identity, static native geometry, and
original axes; it proves neither bounds nor ownership, values, or completion.
Static axis diagnostics and native correspondence are separate obligations.
Dynamic source bounds are input contracts. UB-copy disjointness applies only
to operands of the same actual task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.ir import SSAValue
from xdsl.utils.exceptions import VerifyException

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.structured import AddressOp
from ..indexing import equivalent, exact_constant, normalize
from ..scalar import Expr, Fact, expression, from_ssa, literal
from ..symbols import Symbol
from ..target.layout import NZLayout
from ..target.native import ITEMSIZE, NativeGeometry, nz_columns
from .matrix_regions import MatrixRegion


@dataclass(frozen=True, slots=True)
class AxisFacts:
    """Original axes before linearization or a native subview displacement."""

    origins: tuple[Expr, ...]
    extents: tuple[Expr, ...]
    widths: tuple[int, ...]
    axes: tuple[int, ...]
    displacement: int


@dataclass(frozen=True, slots=True)
class NDRectangle:
    extents: tuple[int, int]
    widths: tuple[int, int]
    origins: tuple[Expr, Expr]


def nd_rectangle(offset, geometry, axes, facts):
    """Retain an actual ND box; never derive a tile traversal or its inverse."""
    if (
        geometry.bank != "GM"
        or geometry.layout != "linear"
        or axes is None
        or len(axes.origins) != 2
        or axes.axes != (0, 1)
        or axes.displacement != 0
    ):
        return None
    extents = tuple(exact_constant(value, facts) for value in axes.extents)
    widths = axes.widths
    if any(
        width <= 0 or size is None or size < width
        for size, width in zip(extents, widths, strict=True)
    ):
        return None
    _, columns = extents
    itemsize = ITEMSIZE[geometry.dtype]
    if (
        geometry.rows != widths[0]
        or geometry.nbytes != widths[1] * itemsize
        or geometry.row_stride != columns * itemsize
    ):
        return None
    row, column = axes.origins
    actual = expression(
        "mul",
        expression("add", expression("mul", row, literal(columns)), column),
        literal(itemsize),
    )
    if not equivalent(offset, actual, facts):
        return None
    return NDRectangle(extents, widths, axes.origins)


@dataclass(frozen=True, slots=True)
class AddressFacts:
    """An actual retained address, without any admission certificate."""

    operation: AddressOp
    root: SSAValue
    offset: Expr
    geometry: NativeGeometry
    axes: AxisFacts | None
    rectangle: NDRectangle | None
    valid_rows: Expr | None = None
    matrix: MatrixRegion | None = None

    @property
    def row_capacity(self):
        # Prefix counts are logical matrix rows. NZ represents those rows
        # as a byte prefix in each physical column-block segment.
        if self.geometry.layout == "NZ":
            return self.geometry.nbytes // self.row_bytes
        return self.geometry.rows

    @property
    def row_bytes(self):
        return nz_columns(self.geometry.bank, self.geometry.dtype) * ITEMSIZE[self.geometry.dtype]

    def prefix_span(self):
        if self.geometry.layout == "NZ":
            return expression(
                "add",
                expression("mul", self.valid_rows, literal(self.row_bytes)),
                literal((self.geometry.rows - 1) * self.geometry.row_stride),
            )
        return expression(
            "add",
            literal(self.geometry.nbytes),
            expression(
                "mul",
                expression("sub", self.valid_rows, literal(1)),
                literal(self.geometry.row_stride),
            ),
        )


class ScalarEnvironment:
    """Shared SSA expressions under immutable uniform-input contracts.

    Bindings are supplied by the control traversal. In particular, all stages
    of one pipeline share its logical ordinal, and all vector domains share
    the kernel's runtime sid. This decoder invents no clocks or owner symbols.
    Facts declare uniform inputs only; coordinate ranges belong to the active
    domain supplied to a proof, never to this uniform-input environment.
    """

    def __init__(self, bindings: Mapping[SSAValue, Expr], facts: Mapping[str | Symbol, Fact]):
        require(
            all(
                isinstance(value, SSAValue) and isinstance(expr, Expr)
                for value, expr in bindings.items()
            ),
            "structured scalar bindings need actual SSA values and expressions",
        )
        require(
            all(
                isinstance(name, (str, Symbol)) and isinstance(fact, Fact)
                for name, fact in facts.items()
            ),
            "structured uniform scalar facts need names and Fact values",
        )
        self._memo = dict(bindings)
        self._facts: Mapping[str | Symbol, Fact] = MappingProxyType(dict(facts))

    @property
    def facts(self) -> Mapping[str | Symbol, Fact]:
        return self._facts

    def expression(self, value: SSAValue) -> Expr:
        """Parse each actual SSA expression through the shared scalar memo.

        This does not validate unsigned dividends against an active domain.
        Source dynamic definedness is an input contract, not a decoder query.
        Newly generated evaluations retain their separate lowering obligations.
        """
        if value not in self._memo:
            self._memo[value] = from_ssa(value, memo=self._memo)
        return self._memo[value]


class StructuredAddresses:
    """One decoder/cache for actual AddressOps in the caller's IR snapshot."""

    def __init__(self, environment: ScalarEnvironment):
        self.environment = environment
        self._addresses: dict[SSAValue, AddressFacts] = {}
        self._layouts: dict[SSAValue, NZLayout] = {}

    @property
    def decode_count(self) -> int:
        return len(self._addresses)

    def decode(self, ref: SSAValue) -> AddressFacts:
        if ref in self._addresses:
            return self._addresses[ref]
        operation = ref.owner
        require(
            isinstance(operation, AddressOp) and ref is operation.ref,
            "structured native operand needs an actual AddressOp result",
            kind=ErrorKind.UNSUPPORTED,
            phase="structured-addresses",
        )
        assert isinstance(operation, AddressOp)
        try:
            operation.verify()
        except VerifyException as error:
            raise CompileError(
                str(error),
                phase="structured-addresses",
                location=str(operation.location),
            ) from error

        typ = operation.source.type
        layout = None
        if typ.bank.data in {"GM", "L1"} and typ.layout.data == "NZ":
            if operation.source not in self._layouts:
                root = operation.source.owner
                require(
                    root.matrix_shape is not None, "NZ allocation lost its physical matrix shape"
                )
                layout = NZLayout(
                    tuple(v.data for v in root.matrix_shape), ITEMSIZE[typ.dtype.data]
                )
                require(layout.nbytes == root.nbytes.data, "NZ layout differs from allocation size")
                self._layouts[operation.source] = layout
            layout = self._layouts[operation.source]
        geometry = NativeGeometry(
            typ.bank.data,
            typ.dtype.data,
            typ.layout.data,
            operation.nbytes.data,
            operation.rows.data,
            operation.row_stride.data,
            None if layout is None else layout.shape[-2:],
        )
        axes = None
        if operation.widths is not None:
            assert operation.axes is not None and operation.displacement is not None
            axes = AxisFacts(
                tuple(self.environment.expression(origin) for origin in operation.origins),
                tuple(self.environment.expression(extent) for extent in operation.extents),
                tuple(width.data for width in operation.widths),
                tuple(axis.data for axis in operation.axes),
                operation.displacement.data,
            )
        offset = self.environment.expression(operation.offset)
        valid_rows = (
            None
            if operation.valid_rows is None
            else self.environment.expression(operation.valid_rows)
        )
        rectangle = (
            None
            if valid_rows is not None
            else nd_rectangle(offset, geometry, axes, self.environment.facts)
        )
        matrix = None
        if layout is not None:
            require(
                axes is not None
                and len(axes.extents) == len(layout.shape)
                and axes.displacement == 0
                and tuple(v.constant for v in axes.extents) == layout.shape,
                "NZ view must retain its allocation layout and original coordinates",
            )
            widths = list(map(literal, axes.widths))
            if valid_rows is not None:
                widths[-2] = valid_rows
            matrix = MatrixRegion(
                layout,
                axes.origins,
                tuple(
                    expression("add", start, width)
                    for start, width in zip(axes.origins, widths, strict=True)
                ),
            )
        result = AddressFacts(
            operation,
            operation.source,
            offset,
            geometry,
            axes,
            rectangle,
            valid_rows,
            matrix,
        )
        self._addresses[ref] = result
        return result

    def disjoint(self, left: AddressFacts, right: AddressFacts) -> bool:
        """Prove UB-copy operands disjoint within one actual native task.

        Distinct real BufferOp roots are separate private UB allocations. For
        one root, only contiguous views with a sufficiently large constant
        relative displacement are proved. Shared names and different offset
        SSA values supply no alias premise. This proves no temporal ownership.
        """
        if left.geometry.bank != "UB" or right.geometry.bank != "UB":
            return False
        if left.root is not right.root:
            return True
        if not left.geometry.contiguous or not right.geometry.contiguous:
            return False
        delta = normalize(
            expression("sub", right.offset, left.offset), self.environment.facts
        ).constant
        return delta is not None and abs(delta) >= max(
            left.geometry.total_nbytes, right.geometry.total_nbytes
        )


def check_static_axes(address):
    """Diagnose explicit invalid axes; dynamic accesses are source contracts."""
    axes = address.axes
    require(
        axes is not None,
        "native address has no original-axis metadata",
        kind=ErrorKind.UNSUPPORTED,
        phase="structured-addresses",
    )
    count = address.valid_rows
    if count is not None:
        require(
            count.constant is None or 0 <= count.constant <= address.row_capacity,
            "static row count exceeds its declared capacity",
        )
        if count.constant == 0:
            return
    for axis, (origin, width, extent) in enumerate(
        zip(axes.origins, axes.widths, axes.extents, strict=True)
    ):
        if count is not None and axis == axes.axes[0]:
            width = count.constant
        start, size = origin.constant, extent.constant
        require(
            (start is None or start >= 0)
            and (size is None or width is None or size >= width)
            and (start is None or size is None or width is None or start + width <= size),
            "static logical axis exceeds its tensor extent",
            phase="structured-addresses",
            location=str(address.operation.location),
            witness=(
                f"axis={axis}",
                f"origin={origin.text()}",
                f"width={width}",
                f"extent={extent.text()}",
            ),
        )

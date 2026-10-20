"""Decode retained native addresses without flattening their SSA coordinates.

The caller supplies one canonical scalar environment for the actual structured
control. Decoding preserves allocation identity, static native geometry, and
original axes; it proves neither bounds nor ownership, values, or completion.
Axis checking is a separate obligation over an explicitly supplied active
domain. UB-copy disjointness applies only to operands of the same actual task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.ir import SSAValue
from xdsl.utils.exceptions import VerifyException

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.structured import AddressOp
from ..indexing import check_axis, equivalent, exact_constant, normalize
from ..scalar import Expr, Fact, expression, from_ssa, literal
from ..symbols import Symbol
from ..target.native import ITEMSIZE, NativeGeometry


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
        The consuming traversal must check the original arithmetic operations,
        including expressions removed by scalar simplification.
        """
        return from_ssa(value, memo=self._memo, facts=self.facts)


class StructuredAddresses:
    """One decoder/cache for actual AddressOps in the caller's IR snapshot."""

    def __init__(self, environment: ScalarEnvironment):
        self.environment = environment
        self._addresses: dict[SSAValue, AddressFacts] = {}

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
        geometry = NativeGeometry(
            typ.bank.data,
            typ.dtype.data,
            typ.layout.data,
            operation.nbytes.data,
            operation.rows.data,
            operation.row_stride.data,
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
        result = AddressFacts(
            operation,
            operation.source,
            offset,
            geometry,
            axes,
            rectangle,
            valid_rows,
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

    def check_axes(
        self,
        address: AddressFacts,
        domains: Sequence[tuple[Expr, Expr, Expr]],
    ) -> None:
        """Check every original axis under the caller's actual active domain.

        This supplies no allocation-byte, native-alignment, subview-displacement,
        ownership, initialization, or asynchronous completion proof. A caller
        must exclude inactive iterations before requesting an access proof.
        A failed dynamic range proof is unsupported; only an explicit constant
        axis violations use the invalid-input diagnostic kind.
        """
        require(
            address.axes is not None,
            "structured address has no original-axis metadata for bounds admission",
            kind=ErrorKind.UNSUPPORTED,
            phase="structured-addresses",
            location=str(address.operation.location),
        )
        assert address.axes is not None
        valid_rows = address.valid_rows
        if valid_rows is not None:
            from ..indexing import domain_nonnegative

            require(
                domain_nonnegative(valid_rows, domains, self.environment.facts)
                and domain_nonnegative(
                    expression("sub", literal(address.geometry.rows), valid_rows),
                    domains,
                    self.environment.facts,
                ),
                "runtime row count must stay within its declared capacity",
            )
            # Empty prefixes perform no access. This predicate does not
            # authorize the original scalar arithmetic outside that access.
            domains = (*domains, (valid_rows, literal(1), literal(address.geometry.rows)))
        for axis, (origin, width, extent) in enumerate(
            zip(address.axes.origins, address.axes.widths, address.axes.extents, strict=True)
        ):
            if valid_rows is not None and axis == address.axes.axes[0]:
                width = valid_rows
            try:
                check_axis(origin, width, extent, domains, self.environment.facts)
            except CompileError as error:
                start, size = origin.constant, extent.constant
                width_value = width if type(width) is int else width.constant
                constant_violation = (
                    start is not None
                    and start < 0
                    or size is not None
                    and width_value is not None
                    and size < width_value
                    or start is not None
                    and size is not None
                    and width_value is not None
                    and start + width_value > size
                )
                kind = error.kind
                if kind is ErrorKind.INVALID and not constant_violation:
                    kind = ErrorKind.UNSUPPORTED
                raise CompileError(
                    error.message,
                    kind=kind,
                    phase="structured-addresses",
                    location=str(address.operation.location),
                    witness=(
                        *error.witness,
                        f"axis={axis}",
                        f"origin={origin.text()}",
                        f"width={width}",
                        f"extent={extent.text()}",
                    ),
                ) from error

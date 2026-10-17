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
from ..indexing import check_axis, normalize
from ..scalar import Expr, Fact, expression, from_ssa
from ..target.native import NativeGeometry


@dataclass(frozen=True, slots=True)
class AxisFacts:
    """Original axes before linearization or a native subview displacement."""

    origins: tuple[Expr, ...]
    extents: tuple[Expr, ...]
    widths: tuple[int, ...]
    axes: tuple[int, ...]
    displacement: int


@dataclass(frozen=True, slots=True)
class AddressFacts:
    """An actual retained address, without any admission certificate."""

    operation: AddressOp
    root: SSAValue
    offset: Expr
    geometry: NativeGeometry
    axes: AxisFacts | None


class ScalarEnvironment:
    """Shared SSA expressions under immutable uniform-input contracts.

    Bindings are supplied by the control traversal. In particular, all stages
    of one pipeline share its logical ordinal, and all vector domains share
    the kernel's runtime sid. This decoder invents no clocks or owner symbols.
    Facts declare uniform inputs only; coordinate ranges belong to the active
    domain supplied to a proof, never to this uniform-input environment.
    """

    def __init__(self, bindings: Mapping[SSAValue, Expr], facts: Mapping[str, Fact]):
        require(
            all(
                isinstance(value, SSAValue) and isinstance(expr, Expr)
                for value, expr in bindings.items()
            ),
            "structured scalar bindings need actual SSA values and expressions",
        )
        require(
            all(isinstance(name, str) and isinstance(fact, Fact) for name, fact in facts.items()),
            "structured uniform scalar facts need names and Fact values",
        )
        self._memo = dict(bindings)
        self._facts: Mapping[str, Fact] = MappingProxyType(dict(facts))

    @property
    def facts(self) -> Mapping[str, Fact]:
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
        result = AddressFacts(
            operation,
            operation.source,
            self.environment.expression(operation.offset),
            geometry,
            axes,
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
        for axis, (origin, width, extent) in enumerate(
            zip(address.axes.origins, address.axes.widths, address.axes.extents, strict=True)
        ):
            try:
                check_axis(origin, width, extent, domains, self.environment.facts)
            except CompileError as error:
                start, size = origin.constant, extent.constant
                constant_violation = (
                    start is not None
                    and start < 0
                    or size is not None
                    and size < width
                    or start is not None
                    and size is not None
                    and start + width > size
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

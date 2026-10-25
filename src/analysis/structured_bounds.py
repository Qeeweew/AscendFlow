"""Physical capacity and source-to-native access correspondence.

Dynamic source accesses are caller contracts. This stage diagnoses static
illegality and checks actual allocation/layout/native requirements; optional
spatial enclosures never supply initialization, ownership, or completion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import prod
from types import MappingProxyType

from xdsl.ir import SSAValue

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import BufferOp, MixedOp
from ..dialects.source import _constant_index
from ..indexing import equivalent, exact_constant, multiple_of
from ..scalar import Expr, expression, from_ssa, infer, literal
from ..symbols import Symbol, SymbolKind
from ..target.ascend910b import Actor, Engine
from ..target.native import CAPACITY, ITEMSIZE, nz_columns
from .structured_addresses import AddressFacts, check_static_axes
from .structured_control import ControlFrame
from .structured_memory import NativeMemoryFacts


@dataclass(frozen=True, slots=True)
class AddressEnclosure:
    """An enclosing slice from a legal source view and checked native mapping.

    Its interval can contain untouched holes. It supplies only spatial
    separation, never actual effects, initialization, or reader retirement.
    """

    origin: Expr
    nbytes: int


@dataclass(frozen=True, slots=True)
class AddressBounds:
    """Required native alignment and original nonempty use scopes."""

    address: AddressFacts
    alignment: int
    frames: tuple[ControlFrame, ...]
    enclosures: tuple[AddressEnclosure, ...] = ()


@dataclass(frozen=True)
class MemoryBoundsFacts:
    memory: NativeMemoryFacts
    allocations: Mapping[SSAValue, Expr]
    addresses: Mapping[SSAValue, AddressBounds]
    local_bytes: Mapping[tuple[Actor, str], int]
    checked_accesses: int
    view_contracts: int


def _require(condition, message, operation, *, kind=ErrorKind.INVALID, witness=()):
    require(
        condition,
        message,
        kind=kind,
        phase="structured-bounds",
        location=str(operation.location),
        witness=witness,
    )


def _allocations(memory):
    environment = memory.control.environment
    facts = environment.facts
    kernel = memory.module.body.block.first_op
    mixed = isinstance(kernel, MixedOp)
    allocations, local_bytes = {}, {}
    for operation in memory.control.frames:
        if not isinstance(operation, BufferOp):
            continue
        bank, dtype = operation.ref.type.bank.data, operation.ref.type.dtype.data
        _require(
            operation.cache is None
            or operation.cache.data in {"normal", "bypass"}
            and (operation.cache.data == "normal" or bank == "GM"),
            "cache policy must be normal or GM bypass",
            operation,
        )
        _require(
            bank in {*CAPACITY, "GM"} and dtype in ITEMSIZE,
            "allocation has an unknown physical bank or dtype",
            operation,
        )
        _require(
            operation.external.data in {0, 1}
            and (not operation.external.data or bank == "GM")
            and (
                bank != "GM"
                or operation.external.data
                or operation.access is not None
                and operation.access.data == "local"
                and operation.initialized.data == 0
            ),
            "GM storage is external or uninitialized kernel workspace; local banks are private",
            operation,
        )
        _require(
            operation.nbytes.data >= 0 and (bank == "GM" or operation.nbytes.data % 32 == 0),
            "allocation byte size must be nonnegative and local sizes 32-byte aligned",
            operation,
        )
        scale = 0 if operation.count_scale is None else operation.count_scale.data
        _require(
            scale >= 0 and (scale > 0) == (operation.extent_count is not None),
            "runtime allocation extent must retain its actual SSA count and byte scale",
            operation,
        )
        size = literal(operation.nbytes.data)
        if operation.extent_count is not None:
            size = expression("mul", environment.expression(operation.extent_count), literal(scale))
        _require(
            size.constant is None or size.constant >= 0,
            "negative allocation byte extent",
            operation,
        )
        _require(
            size.constant is None or size.constant < 1 << 63,
            "static allocation byte extent exceeds int64",
            operation,
        )
        allocations[operation.ref] = size
        if bank == "GM":
            continue
        if mixed:
            participants = (Actor.AIV0, Actor.AIV1) if bank == "UB" else (Actor.AIC,)
        else:
            owner = kernel.core.data
            _require(
                owner in {"aic", "aiv"} and ((owner == "aiv") == (bank == "UB")),
                "local allocation bank belongs to another kernel participant",
                operation,
            )
            participants = (Actor.AIV0,) if owner == "aiv" else (Actor.AIC,)
        size_fact = infer(size, facts)
        _require(
            size_fact is not None and size_fact.maximum is not None,
            "local allocation capacity needs a finite byte upper bound",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        _require(
            multiple_of(size, 32, facts),
            "runtime local allocation size must remain 32-byte aligned",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        assert size_fact is not None and size_fact.maximum is not None
        base_alignment = 32 if bank == "UB" else 512
        for actor in participants:
            key = actor, bank
            previous = local_bytes.get(key, 0)
            offset = (previous + base_alignment - 1) // base_alignment * base_alignment
            used = offset + size_fact.maximum
            _require(
                used <= CAPACITY[bank],
                f"{actor.value} {bank} capacity exceeded: {used} > {CAPACITY[bank]} bytes",
                operation,
                kind=ErrorKind.RESOURCE,
            )
            local_bytes[key] = used
    return allocations, local_bytes


def _original_view(address: AddressFacts, facts):
    """Reconstruct the supported original view, independently of its offset."""
    axes, operation = address.axes, address.operation
    _require(
        axes is not None,
        "address bounds need original-axis metadata",
        operation,
        kind=ErrorKind.UNSUPPORTED,
    )
    assert axes is not None
    geometry = address.geometry
    itemsize = ITEMSIZE[geometry.dtype]
    # Verify only the address recipe. Source indices are SSA atoms here, not
    # proof expressions to expand or bound. Their execution remains in the IR.
    anchors, constants = {}, {}
    for value in (*operation.origins, *operation.extents):
        constant = _constant_index(value, constants)
        anchors[value] = (
            literal(constant)
            if constant is not None
            else Expr("argument", (Symbol("$axis", SymbolKind.SSA, value),))
        )
    actual = from_ssa(operation.offset, memo=dict(anchors))
    if address.matrix is not None:
        layout = address.matrix.layout
        expected = literal(0)
        for origin, stride in zip(operation.origins, layout.strides, strict=True):
            expected = expression(
                "add", expected, expression("mul", anchors[origin], literal(stride))
            )
        _require(
            equivalent(actual, expected, {}),
            "address offset does not match its NZ allocation layout",
            operation,
        )
        rank = len(layout.shape)
        _require(
            axes.axes == (rank - 2, rank - 1)
            and all(width == 1 for width in axes.widths[:-2])
            and axes.widths[-1] % layout.c0 == 0
            and multiple_of(axes.origins[-1], layout.c0, facts),
            "native access needs an aligned NZ rectangle",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        expected_geometry = layout.geometry(
            *axes.widths[-2:], prefix=address.valid_rows is not None or geometry.bank == "GM"
        )
        _require(
            (geometry.nbytes, geometry.rows, geometry.row_stride) == expected_geometry,
            "NZ native footprint differs from its allocation layout and view",
            operation,
        )
        return expected_geometry
    stride = literal(itemsize)
    base = literal(0)
    pairs = tuple(
        (anchors[o], anchors[e]) for o, e in zip(operation.origins, operation.extents, strict=True)
    )
    for axis in range(len(pairs) - 1, -1, -1):
        origin, extent = pairs[axis]
        physical_stride = stride
        if geometry.layout == "NZ" and axis == len(pairs) - 1:
            physical_stride = expression("mul", stride, anchors[operation.extents[-2]])
        elif geometry.layout == "NZ" and axis == len(pairs) - 2:
            physical_stride = literal(nz_columns(geometry.bank, geometry.dtype) * itemsize)
        base = expression("add", base, expression("mul", origin, physical_stride))
        stride = expression("mul", stride, extent)
    expected = expression("add", base, literal(axes.displacement))
    strides, stride = [], literal(itemsize)
    for extent in reversed(axes.extents):
        strides.insert(0, stride)
        stride = expression("mul", stride, extent)
    if not equivalent(actual, expected, {}):
        difference = exact_constant(expression("sub", actual, expected), {})
        _require(
            False,
            "address offset does not match original-axis linearization and displacement",
            operation,
            kind=ErrorKind.INVALID if difference is not None else ErrorKind.UNSUPPORTED,
            witness=(f"offset={address.offset.text()}", f"expected={expected.text()}"),
        )
    selected = axes.axes
    _require(
        len(selected) in {1, 2}
        and all(width == 1 for axis, width in enumerate(axes.widths) if axis not in selected),
        "original view needs one or two retained axes and unit projected widths",
        operation,
        kind=ErrorKind.UNSUPPORTED,
    )
    _require(
        equivalent(strides[selected[-1]], literal(itemsize), facts),
        "original view columns need contiguous physical elements",
        operation,
        kind=ErrorKind.UNSUPPORTED,
    )
    width = axes.widths[selected[-1]] * itemsize
    rows, pitch = 1, 0
    if len(selected) == 2:
        rows = axes.widths[selected[0]]
        pitch = exact_constant(strides[selected[0]], facts)
        _require(
            pitch is not None and pitch >= width,
            "original view row pitch needs a static non-overlapping byte stride",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        assert pitch is not None
    if geometry.layout != "linear":
        rank = len(axes.origins)
        extents = tuple(exact_constant(extent, facts) for extent in axes.extents[-2:])
        accumulator_columns = geometry.bank == "L0C" and geometry.layout == "NZ"
        _require(
            rank >= 2
            and selected == (rank - 2, rank - 1)
            and equivalent(axes.origins[-2], literal(0), facts)
            and (accumulator_columns or equivalent(axes.origins[-1], literal(0), facts))
            and all(extent is not None and extent > 0 and extent % 16 == 0 for extent in extents)
            and axes.widths[-2] == extents[0]
            and axes.widths[-2] % 16 == 0
            and multiple_of(axes.origins[-2], 16, facts)
            and axes.widths[-1] % 16 == 0
            and multiple_of(axes.origins[-1], 16, facts)
            and (accumulator_columns or axes.widths[-1] == extents[1]),
            "fractal address needs a complete matrix plane, aligned L1 NZ rectangle or full-row L0C columns",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        _require(
            geometry.layout == "NZ"
            or axes.displacement == 0
            and geometry.contiguous
            and geometry.total_nbytes == width * rows,
            "fractal native subview must retain the complete selected physical plane",
            operation,
            kind=ErrorKind.UNSUPPORTED,
        )
        return width * rows, 1, 0
    return width, rows, pitch


def _subview(address, parent):
    width, rows, pitch = parent
    geometry, operation = address.geometry, address.operation
    displacement = address.axes.displacement
    span = geometry.nbytes + (geometry.rows - 1) * geometry.row_stride
    parent_span = width + (rows - 1) * pitch
    _require(
        displacement + span <= parent_span,
        "native subview exceeds its original view footprint",
        operation,
        witness=(f"displacement={displacement}", f"span={span}", f"parent_span={parent_span}"),
    )
    if rows > 1 and pitch > width:
        first, within = divmod(displacement, pitch)
        _require(
            within + geometry.nbytes <= width,
            "native subview crosses an original-view row gap",
            operation,
        )
        same_row = within + span <= width
        if not same_row:
            _require(
                geometry.rows > 1 and geometry.row_stride % pitch == 0,
                "native subview across pitched rows needs an explicit physical row mapping",
                operation,
                kind=ErrorKind.UNSUPPORTED,
            )
            _require(
                first + (geometry.rows - 1) * geometry.row_stride // pitch < rows,
                "native subview exceeds original-view rows",
                operation,
            )
    return span


def _enclosing_slices(address, facts):
    """Compose legal source axes into enclosing allocation slices.

    Linear storage permits any fixed leading prefix. Fractal matrices retain
    their final two axes together; the already checked layout maps every
    such matrix into its contiguous plane. No native footprint is enlarged.
    Admission may use a complete real definition of this enclosure to cover
    a legal indirect read; the enclosure by itself defines no byte or value.
    """
    axes = address.axes
    if address.matrix is not None:
        result, origin = [], literal(0)
        for coordinate, width, stride in zip(
            address.matrix.starts[:-2],
            axes.widths[:-2],
            address.matrix.layout.strides[:-2],
            strict=True,
        ):
            if width != 1:
                break
            origin = expression("add", origin, expression("mul", coordinate, literal(stride)))
            result.append(AddressEnclosure(origin, stride))
        return tuple(result)
    extents = tuple(exact_constant(value, facts) for value in axes.extents)
    if any(size is None or size <= 0 for size in extents):
        return ()
    limit = len(extents) - (1 if address.geometry.layout == "linear" else 2)
    result, origin = [], literal(0)
    for depth in range(1, limit + 1):
        axis = depth - 1
        if axes.widths[axis] != 1 or axis in axes.axes:
            break
        stride = prod(extents[depth:]) * ITEMSIZE[address.geometry.dtype]
        origin = expression("add", origin, expression("mul", axes.origins[axis], literal(stride)))
        result.append(AddressEnclosure(origin, stride))
    return tuple(result)


def admit_memory_bounds(memory: NativeMemoryFacts) -> MemoryBoundsFacts:
    """Check physical mappings once; retain source domains without proving data bounds."""
    environment = memory.control.environment
    facts = environment.facts
    allocations, local_bytes = _allocations(memory)
    requirements = {value: 1 for value in memory.addresses}
    frames = {value: [] for value in memory.addresses}
    checked_accesses = 0
    for template in memory.templates:
        if template.frame.statically_empty:
            continue
        for access in template.accesses:
            bank, engine = access.address.geometry.bank, template.spec.engine
            alignment = (
                512
                if bank != "GM" and engine in {Engine.M, Engine.MTE1, Engine.FIX}
                else ITEMSIZE[access.address.geometry.dtype]
                if bank == "GM"
                and template.operation.native.data
                in {
                    "nd2nz",
                    "nd2nz_pad",
                    "nd2nz_rows",
                    "nd2nz_overlay",
                    "fix",
                    "fix_rows",
                    "copy_bytes",
                }
                else ITEMSIZE[access.address.geometry.dtype]
                if engine is Engine.S
                else 32
            )
            requirements[access.operand] = max(requirements[access.operand], alignment)
            checked_accesses += 1
            if template.frame not in frames[access.operand]:
                frames[access.operand].append(template.frame)
    admitted, views = {}, {}
    for value, address in memory.addresses.items():
        active_frames, alignment = tuple(frames[value]), requirements[value]
        admitted[value] = AddressBounds(address, alignment, active_frames)
        if not active_frames:
            continue
        # Equal physical views have one bounds/alignment contract, even when
        # lowering produced several SSA AddressOps. Include original axes and
        # raw offset DAG, not just the flattened footprint. Actual use scopes
        # are retained without re-proving their dynamic source bounds.
        key = address.root, address.offset, address.geometry, address.axes, address.valid_rows
        views.setdefault(key, []).append(admitted[value])
    for members in views.values():
        address = members[0].address
        alignment = max(item.alignment for item in members)
        operation = address.operation
        _require(
            address.root in allocations, "address has no actual declared allocation", operation
        )
        try:
            original = _original_view(address, facts)
            span = _subview(address, original)
            capacity = allocations[address.root]
            aligned = multiple_of(address.offset, alignment, facts)
            aligned = aligned and (
                address.geometry.rows == 1 or address.geometry.row_stride % alignment == 0
            )
            _require(
                aligned,
                f"native operand origin requires {alignment}-byte alignment",
                operation,
                kind=(
                    ErrorKind.INVALID
                    if address.offset.constant is not None
                    else ErrorKind.UNSUPPORTED
                ),
                witness=(
                    f"offset={address.offset.text()}",
                    f"row_stride={address.geometry.row_stride}",
                ),
            )
            check_static_axes(address)
            # Source accesses must be legal when executed. Do not re-prove the
            # caller's data contract. Still reject explicit physical capacity
            # mismatches and native overreads introduced by lowering.
            origin = address.offset.constant
            limit = capacity.constant
            actual_span = span if address.valid_rows is None else address.prefix_span().constant
            if origin is not None and limit is not None and actual_span is not None:
                _require(
                    origin >= 0 and origin + actual_span <= limit,
                    "static native byte footprint exceeds its allocation",
                    operation,
                    witness=(f"span={actual_span}", f"allocation={limit}"),
                )
        except CompileError as error:
            if error.phase == "structured-bounds":
                raise
            raise CompileError(
                error.message,
                kind=error.kind,
                phase="structured-bounds",
                location=str(operation.location),
                witness=error.witness,
            ) from error
        enclosures = _enclosing_slices(address, facts)
        for member in members:
            admitted[member.address.operation.ref] = replace(member, enclosures=enclosures)
    return MemoryBoundsFacts(
        memory,
        MappingProxyType(allocations),
        MappingProxyType(admitted),
        MappingProxyType(local_bytes),
        checked_accesses,
        len(views),
    )

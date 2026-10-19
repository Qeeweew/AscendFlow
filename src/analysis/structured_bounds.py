"""Axis, allocation-byte, subview, and alignment proofs on retained natives.

These facts consume already decoded memory/control facts and never rerun a
native rule. They establish mathematical address bounds in actual active
scopes, plus declared local-memory capacity. Integer/launcher overflow, group
ownership, initialized values, registers, and completion remain separate
obligations before executable admission.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from xdsl.ir import SSAValue

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import BufferOp, MixedOp
from ..indexing import domain_nonnegative, equivalent, exact_constant, polynomial
from ..scalar import Expr, Fact, expression, infer, literal
from ..target.ascend910b import Actor, Engine
from ..target.native import CAPACITY, ITEMSIZE, nz_columns
from .structured_addresses import AddressFacts, StructuredAddresses
from .structured_control import ControlFrame
from .structured_memory import NativeMemoryFacts


@dataclass(frozen=True, slots=True)
class AddressBounds:
    """Bounds and maximum native alignment over listed nonempty use scopes."""

    address: AddressFacts
    alignment: int
    frames: tuple[ControlFrame, ...]


@dataclass(frozen=True)
class MemoryBoundsFacts:
    memory: NativeMemoryFacts
    allocations: Mapping[SSAValue, Expr]
    addresses: Mapping[SSAValue, AddressBounds]
    local_bytes: Mapping[tuple[Actor, str], int]
    checked_accesses: int
    view_contracts: int
    domain_contracts: int


def _require(condition, message, operation, *, kind=ErrorKind.INVALID, witness=()):
    require(
        condition,
        message,
        kind=kind,
        phase="structured-bounds",
        location=str(operation.location),
        witness=witness,
    )


def _multiple_of(value: Expr, alignment: int, facts: Mapping[str, Fact]) -> bool:
    """Prove each exact polynomial term a multiple, retaining all coordinates."""
    for atoms, coefficient in polynomial(value, facts).items():
        multiple = Fraction(coefficient)
        for atom in atoms:
            fact = infer(atom, facts)
            multiple *= 1 if fact is None else fact.multiple
        if multiple.denominator != 1 or multiple.numerator % alignment:
            return False
    return True


def _nonnegative(value, facts, operation, message, *, witness=(), domains=()):
    if domain_nonnegative(value, domains, facts):
        return
    constant = exact_constant(value, facts)
    _require(
        False,
        message,
        operation,
        kind=ErrorKind.INVALID if constant is not None and constant < 0 else ErrorKind.UNSUPPORTED,
        witness=(*witness, f"remaining={value.text()}"),
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
            operation.external.data in {0, 1} and bool(operation.external.data) == (bank == "GM"),
            "GM allocations are external; local banks are private",
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
        _nonnegative(size, facts, operation, "allocation byte extent is not provably nonnegative")
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
            _multiple_of(size, 32, facts),
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
    strides, stride = [], literal(itemsize)
    base = literal(0)
    pairs = tuple(zip(axes.origins, axes.extents, strict=True))
    for axis in range(len(pairs) - 1, -1, -1):
        origin, extent = pairs[axis]
        strides.insert(0, stride)
        physical_stride = stride
        if geometry.layout == "NZ" and axis == len(pairs) - 1:
            physical_stride = expression("mul", stride, axes.extents[-2])
        elif geometry.layout == "NZ" and axis == len(pairs) - 2:
            physical_stride = literal(nz_columns(geometry.bank, geometry.dtype) * itemsize)
        base = expression("add", base, expression("mul", origin, physical_stride))
        stride = expression("mul", stride, extent)
    expected = expression("add", base, literal(axes.displacement))
    if not equivalent(address.offset, expected, facts):
        difference = exact_constant(expression("sub", address.offset, expected), facts)
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
        column_slice = geometry.bank == "L1" and geometry.layout == "NZ"
        _require(
            rank >= 2
            and selected == (rank - 2, rank - 1)
            and (column_slice or equivalent(axes.origins[-2], literal(0), facts))
            and (column_slice or equivalent(axes.origins[-1], literal(0), facts))
            and all(extent is not None and extent > 0 and extent % 16 == 0 for extent in extents)
            and (column_slice or axes.widths[-2] == extents[0])
            and axes.widths[-2] % 16 == 0
            and _multiple_of(axes.origins[-2], 16, facts)
            and axes.widths[-1] % 16 == 0
            and _multiple_of(axes.origins[-1], 16, facts)
            and (column_slice or axes.widths[-1] == extents[1]),
            "fractal address needs its complete matrix plane or an aligned L1 NZ rectangle",
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
        if geometry.layout == "NZ" and axes.widths[-2] != extents[0]:
            c0 = nz_columns(geometry.bank, geometry.dtype)
            return (
                axes.widths[-2] * c0 * itemsize,
                axes.widths[-1] // c0,
                extents[0] * c0 * itemsize,
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


def admit_memory_bounds(memory: NativeMemoryFacts) -> MemoryBoundsFacts:
    """Prove each actual address once, then check its real nonempty use domains."""
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
                if bank == "GM" and template.operation.native.data in {"nd2nz", "nd2nz_pad", "fix"}
                else ITEMSIZE[access.address.geometry.dtype]
                if engine is Engine.S
                else 32
            )
            requirements[access.operand] = max(requirements[access.operand], alignment)
            checked_accesses += 1
            if template.frame not in frames[access.operand]:
                frames[access.operand].append(template.frame)
    checker, admitted, views = StructuredAddresses(environment), {}, {}
    for value, address in memory.addresses.items():
        active_frames, alignment = tuple(frames[value]), requirements[value]
        admitted[value] = AddressBounds(address, alignment, active_frames)
        if not active_frames:
            continue
        # Equal physical views have one bounds/alignment contract, even when
        # lowering produced several SSA AddressOps. Include original axes and
        # raw offset DAG, not just the flattened footprint. Each distinct
        # active domain still adds its own obligation below.
        key = address.root, address.offset, address.geometry, address.axes
        views.setdefault(key, []).append(admitted[value])
    domain_contracts = 0
    for members in views.values():
        address = members[0].address
        alignment = max(item.alignment for item in members)
        domains = dict.fromkeys(frame.domains() for item in members for frame in item.frames)
        operation = address.operation
        _require(
            address.root in allocations, "address has no actual declared allocation", operation
        )
        try:
            original = _original_view(address, facts)
            span = _subview(address, original)
            tensor_bytes = literal(ITEMSIZE[address.geometry.dtype])
            for extent in address.axes.extents:
                tensor_bytes = expression("mul", tensor_bytes, extent)
            capacity = allocations[address.root]
            full_tensor = equivalent(capacity, tensor_bytes, facts)
            aligned = _multiple_of(address.offset, alignment, facts)
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
            for domain in domains:
                domain_contracts += 1
                checker.check_axes(address, domain)
                if full_tensor or domain_nonnegative(
                    expression("sub", capacity, tensor_bytes), domain, facts
                ):
                    # The checked linearization maps the bounded original
                    # axes into this tensor; _subview keeps every native byte
                    # in that view. Compose those facts with real capacity,
                    # instead of re-proving the flattened nonlinear address.
                    continue
                _nonnegative(
                    address.offset,
                    facts,
                    operation,
                    "native byte origin may be negative",
                    domains=domain,
                )
                remaining = expression(
                    "sub",
                    allocations[address.root],
                    expression("add", address.offset, literal(span)),
                )
                _nonnegative(
                    remaining,
                    facts,
                    operation,
                    "native byte footprint is not bounded by its allocation",
                    domains=domain,
                    witness=(f"span={span}", f"allocation={allocations[address.root].text()}"),
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
    return MemoryBoundsFacts(
        memory,
        MappingProxyType(allocations),
        MappingProxyType(admitted),
        MappingProxyType(local_bytes),
        checked_accesses,
        len(views),
        domain_contracts,
    )

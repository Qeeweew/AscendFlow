"""Written GM ownership across physical groups and shared AIV participants.

Ownership supplies disjoint group/sid partitions. It does not supply a memory
version or asynchronous completion, which remain dependency/sync duties.
"""

from collections import Counter
from dataclasses import dataclass

from xdsl.ir import SSAValue

from ..diagnostics import ErrorKind, require
from ..indexing import _ratio, arguments, equivalent, interval, nonnegative, normalize, polynomial
from ..scalar import Expr, expression, infer, literal
from ..target.ascend910b import Actor
from .structured_memory import MemoryAccess


def _require(condition, message, operation, *, kind=ErrorKind.UNSUPPORTED, witness=()):
    require(
        condition,
        message,
        kind=kind,
        phase="structured-ownership",
        location=str(operation.location),
        witness=witness,
    )


@dataclass(frozen=True, slots=True)
class OwnerProof:
    root: SSAValue
    coordinate: Expr
    count: int
    stride: Expr
    accesses: tuple[MemoryAccess, ...]
    # None: stride is in bytes. Otherwise it is in cells of this ND tile grid.
    tile_shape: tuple[int, int] | None = None


def _affine(value, coordinate):
    """Split an actual expression without expanding unrelated div/rem atoms."""
    if value == coordinate:
        return literal(0), literal(1)
    if not arguments(value) & arguments(coordinate):
        return value, literal(0)
    if value.kind == "add":
        left, right = (_affine(child, coordinate) for child in value.args)
        if left is not None and right is not None:
            return tuple(expression("add", a, b) for a, b in zip(left, right, strict=True))
    if value.kind == "mul":
        for independent, varying in (value.args, value.args[::-1]):
            if not arguments(independent) & arguments(coordinate):
                split = _affine(varying, coordinate)
                if split is not None:
                    return tuple(expression("mul", independent, part) for part in split)
    return None


def _divisible(value, divisor, facts):
    bottom = polynomial(divisor, facts)
    if len(bottom) != 1:
        return False
    ((required, scale),) = bottom.items()
    if scale <= 0:
        return False
    for atoms, coefficient in polynomial(value, facts).items():
        have, need = Counter(atoms), Counter(required)
        if not need <= have:
            return False
        ratio = _ratio(coefficient, scale)
        for atom in (have - need).elements():
            fact = infer(atom, facts)
            ratio *= 1 if fact is None else fact.multiple
        if ratio.denominator != 1:
            return False
    return True


def _residual(value, cycle, facts):
    if _divisible(value, cycle, facts):
        return literal(0)
    if value.kind == "add":
        return expression("add", *(_residual(child, cycle, facts) for child in value.args))
    if value.kind == "mul":
        for other, addition in (value.args, value.args[::-1]):
            if addition.kind == "add":
                return expression(
                    "add",
                    *(
                        _residual(expression("mul", other, child), cycle, facts)
                        for child in addition.args
                    ),
                )
    return value


def _tiled_partition(uses, coordinate, count, facts):
    """Disjoint aligned rectangles through an injective ND tile-grid index.

    Different grid cells cannot share bytes, even when their bounding byte
    intervals overlap. Every access must use the same grid, exactly match its
    original two-dimensional geometry, and retain a distinct owner residue.
    No temporal/value or asynchronous-completion fact follows from this rule.
    """
    grid, stride, checked = None, None, []
    for template, access in uses:
        tile = access.address.tile
        if tile is None or grid is not None and grid != tile.grid:
            return None
        split = _affine(tile.index, coordinate)
        if split is None:
            return None
        base, coefficient = split
        if (
            not nonnegative(expression("sub", coefficient, literal(1)), facts)
            or stride is not None
            and not equivalent(stride, coefficient, facts)
        ):
            return None
        residual = _residual(base, expression("mul", coefficient, literal(count)), facts)
        low, high = interval(residual, template.frame.domains(), facts)
        if not nonnegative(low, facts) or not nonnegative(
            expression("sub", coefficient, expression("add", high, literal(1))), facts
        ):
            return None
        grid, stride = tile.grid, coefficient
        checked.append(access)
    if grid is None:
        return None
    return OwnerProof(checked[0].address.root, coordinate, count, stride, tuple(checked), grid[1])


def admit_owners(memory):
    environment, control = memory.control.environment, memory.control
    facts, kernel = environment.facts, memory.module.body.block.first_op
    groups = 1 if kernel.blocks is None else kernel.blocks.data
    accesses, writes = {}, {}
    for template in memory.templates:
        if template.frame.statically_empty:
            continue
        for access in template.accesses:
            if access.address.geometry.bank == "GM":
                accesses.setdefault(access.address.root, []).append((template, access))
                if access.mode.writes:
                    writes.setdefault(access.address.root, []).append((template, access))
    _require(
        not writes or len(accesses) <= 1 or kernel.disjoint_external.data == 1,
        "distinct external roots need the launch's disjoint pointer contract",
        kernel,
    )
    proofs = []
    for root, writers in writes.items():
        owners = [(control.group, groups, accesses[root])]
        vector_writes = [(t, a) for t, a in writers if t.participants == (Actor.AIV0, Actor.AIV1)]
        if vector_writes:
            owners.append((control.sid, 2, vector_writes))
        for coordinate, count, uses in owners:
            if count == 1:
                continue
            tiled = _tiled_partition(uses, coordinate, count, facts)
            if tiled is not None:
                proofs.append(tiled)
                continue
            stride = None
            checked = []
            for template, access in uses:
                address = access.address
                split = _affine(address.offset, coordinate)
                if split is None:
                    split = _affine(normalize(address.offset, facts), coordinate)
                _require(
                    split is not None,
                    "written GM owner coordinate is not provably separable",
                    template.operation,
                    witness=(root.owner.sym_name.data, coordinate.text()),
                )
                base, coefficient = split
                _require(
                    nonnegative(coefficient, facts)
                    and not equivalent(coefficient, literal(0), facts),
                    "written GM access lacks a positive owner stride",
                    template.operation,
                    witness=(root.owner.sym_name.data, coordinate.text(), coefficient.text()),
                )
                if stride is None:
                    stride = coefficient
                _require(
                    equivalent(stride, coefficient, facts),
                    "written GM aliases change their owner partition",
                    template.operation,
                    witness=(root.owner.sym_name.data, coordinate.text()),
                )
                residual = _residual(base, expression("mul", stride, literal(count)), facts)
                low, high = interval(residual, template.frame.domains(), facts)
                span = (
                    address.geometry.nbytes
                    + (address.geometry.rows - 1) * address.geometry.row_stride
                )
                _require(
                    nonnegative(low, facts)
                    and nonnegative(
                        expression("sub", stride, expression("add", high, literal(span))), facts
                    ),
                    "written GM owner partitions may overlap; no disjoint residue range was proved",
                    template.operation,
                    witness=(
                        root.owner.sym_name.data,
                        coordinate.text(),
                        f"residual={residual.text()}",
                        f"stride={stride.text()}",
                    ),
                )
                checked.append(access)
            assert stride is not None
            proofs.append(OwnerProof(root, coordinate, count, stride, tuple(checked)))
    return tuple(proofs)

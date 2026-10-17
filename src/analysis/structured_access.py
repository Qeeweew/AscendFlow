"""Project owned physical access geometry into exact storage fragments.

This stage owns slot/coordinate interpretation and rectangle partitioning.
Its immutable result is consumed by recursive dependence composition; it
creates neither RAW/WAR/WAW relations nor event/completion state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise, product
from math import gcd, lcm
from types import MappingProxyType

from xdsl.ir import Operation

from ..diagnostics import CompileError, require
from ..dialects.physical import TaskOp
from ..indexing import arguments, equivalent, nonnegative, normalize, polynomial
from ..scalar import Expr, expression, literal, substitute
from ..target.ascend910b import Actor
from .native import AccessMode
from .structured_memory import MemoryAccess, MemoryTemplate, NativeMemoryFacts
from .structured_regions import (
    CountRelation,
    DependencyIssue,
    DependencyIssueKind,
    DependencyWork,
    Guard,
    LoopCountGuard,
    LoopDomain,
    StorageCell,
)


@dataclass(frozen=True)
class CellEffect:
    """One actual instruction's operands covering an exact storage cell.

    The union is a physical footprint. Conditional numerical reads remain on
    each operand, so an initialization cannot acquire a fictitious old value.
    """

    template: MemoryTemplate
    participant: Actor
    accesses: tuple[MemoryAccess, ...]

    @property
    def mode(self):
        reads = any(access.mode.reads for access in self.accesses)
        writes = any(access.mode.writes for access in self.accesses)
        return (
            AccessMode.READ_WRITE
            if reads and writes
            else AccessMode.READ
            if reads
            else AccessMode.WRITE
        )


@dataclass(frozen=True)
class PartitionedAccesses:
    memory: NativeMemoryFacts
    effects: Mapping[TaskOp, Mapping[StorageCell, tuple[CellEffect, ...]]]
    patterns: Mapping[tuple[StorageCell, Operation], frozenset[tuple[int, int | Expr]]]
    issues: tuple[DependencyIssue, ...]
    work: DependencyWork


def partition_accesses(memory, admission=None) -> PartitionedAccesses:
    return _Partitioner(memory, admission).build()


@dataclass(frozen=True, slots=True)
class _Projection:
    template: MemoryTemplate
    participant: Actor
    access: MemoryAccess
    origin: Expr
    start: int
    stop: int
    slots: tuple[LoopDomain, ...]
    streaming: tuple[Operation, ...]
    rows: int = 1
    row_stride: int = 0
    conditions: tuple[Guard, ...] = ()
    contracted: tuple[Operation, ...] = ()
    coordinates: tuple[tuple[Operation, Expr], ...] = ()


def _normalize(value, facts):
    # Declared uniform multiples can give the proof polynomial rational
    # coefficients (e.g. Sq/128). Keep its original exact quotient DAG when
    # that polynomial cannot be reconstructed with integer coefficients.
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


def _count_core(value, facts):
    if value.kind in {"ceildiv", "floordiv"} and value.args[1].constant == 1:
        return _count_core(value.args[0], facts)
    if value.kind == "max":
        a, b = value.args
        if a.constant == 0 and nonnegative(b, facts):
            return _count_core(b, facts)
        if b.constant == 0 and nonnegative(a, facts):
            return _count_core(a, facts)
    return value


def _depends(value, coordinate):
    return coordinate.args[0] in arguments(value)


def _integer_shift(value, coordinate):
    """A shift preserving every constant-divisor congruence in an integer DAG."""
    if not _depends(value, coordinate) or value.kind in {"argument", "constant"}:
        return 1
    a, b = value.args
    if value.kind in {"floordiv", "ceildiv", "rem"}:
        if b.constant is None:
            return None
        shift = _integer_shift(a, coordinate)
        return None if shift is None else shift * b.constant
    if value.kind not in {"add", "mul"}:
        return None
    a, b = _integer_shift(a, coordinate), _integer_shift(b, coordinate)
    return None if a is None or b is None else lcm(a, b)


def _period(value, coordinate):
    if not _depends(value, coordinate):
        return 1
    if value.kind in {"constant", "argument"}:
        return None
    a, b = value.args
    if value.kind == "rem" and b.constant is not None:
        shift = _integer_shift(a, coordinate)
        return None if shift is None else shift * b.constant
    if value.kind not in {"add", "mul"}:
        return None
    a, b = _period(a, coordinate), _period(b, coordinate)
    return None if a is None or b is None else lcm(a, b)


class _Partitioner:
    MAX_PROJECTIONS = 16384

    def __init__(self, memory, admission):
        self.memory, self.admission = memory, admission
        self.facts = memory.control.environment.facts
        self.work = DependencyWork(native_templates=len(memory.templates))
        self.issues, self.effects, self.patterns = [], {}, {}
        self._origin_pool, self._issue_keys = {}, set()

    def issue(self, kind, operation, message, cell=None):
        key = (kind, operation, cell)
        if key not in self._issue_keys:
            self._issue_keys.add(key)
            self.issues.append(DependencyIssue(kind, operation, message, cell))

    def _separate_tiles(self, projections, loop):
        """A single streaming clock advances exact rectangles on one ND grid.

        Slot substitutions, contracted families and additional varying clocks
        need their own relational rules. They cannot borrow this identity.
        """
        grid, index = None, None
        low = high = 0
        for projection in projections:
            address = projection.access.address
            tile, geometry = address.tile, address.geometry
            if (
                tile is None
                or projection.slots
                or projection.contracted
                or projection.stop - projection.start != geometry.nbytes
                or projection.rows != geometry.rows
                or projection.row_stride != geometry.row_stride
                or grid is not None
                and grid != tile.grid
            ):
                return False
            bindings = {
                self.memory.control.sid: literal(1 if projection.participant is Actor.AIV1 else 0)
            }
            actual = expression("add", projection.origin, literal(projection.start))
            if not equivalent(actual, substitute(address.offset, bindings), self.facts):
                return False
            value = substitute(tile.index, bindings)
            if any(
                other.operation is not loop.operation and _depends(value, other.ordinal)
                for other in projection.template.frame.loops()
            ):
                return False
            if index is not None:
                displacement = _normalize(expression("sub", value, index), self.facts).constant
                if displacement is None:
                    return False
                low, high = min(low, displacement), max(high, displacement)
            else:
                index = value
            grid = tile.grid
        if index is None:
            return False
        shifted = substitute(index, {loop.ordinal: expression("add", loop.ordinal, literal(1))})
        delta = _normalize(expression("sub", shifted, index), self.facts)
        return nonnegative(expression("sub", delta, literal(high - low + 1)), self.facts)

    def _slots(self, template, access, participant):
        address = access.address
        bindings = {self.memory.control.sid: literal(1 if participant is Actor.AIV1 else 0)}
        # Keep remainder nodes until their finite image has been selected.
        # Polynomial normalization expands rem into x-d*floor(x/d), which
        # would hide exactly the storage-period contract needed here.
        offset = substitute(address.offset, bindings)
        variants = self._compound_slots(template, offset)
        return tuple(
            projection
            for value, slots, conditions in variants
            for projection in self._slots_plain(
                template, access, participant, value, slots, conditions
            )
        )

    def _compound_slots(self, template, offset):
        loops = template.frame.loops()

        def find(value):
            if value.kind == "rem" and value.args[1].constant is not None:
                active = [loop for loop in loops if _depends(value.args[0], loop.ordinal)]
                if len(active) > 1:
                    return value, active
            for arg in value.args:
                if isinstance(arg, Expr) and (found := find(arg)) is not None:
                    return found
            return None

        compound = find(offset)
        if compound is None:
            return ((offset, (), ()),)
        atom, active = compound
        inner, outer = active[-1], active[-2]
        numerator, modulus = atom.args
        period = modulus.constant
        if period > self.MAX_PROJECTIONS:
            self.issue(
                DependencyIssueKind.BUDGET,
                template.operation,
                "compound storage-slot period exceeds dependency proof budget",
            )
            return ()
        shifted = substitute(
            numerator, {inner.ordinal: expression("add", inner.ordinal, literal(1))}
        )
        delta = _normalize(expression("sub", shifted, numerator), self.facts).constant
        if delta != 1 or len(active) != 2:
            self.issue(
                DependencyIssueKind.SYMBOLIC_ALIAS,
                template.operation,
                "compound modulo needs a unit-stride innermost coordinate map",
            )
            return ((offset, (), ()),)
        base = substitute(numerator, {inner.ordinal: literal(0)})
        variants = []
        for slot in range(period):
            value = substitute(offset, {atom: literal(slot)})
            phase = expression(
                "rem",
                expression("sub", literal(slot + period), expression("rem", base, modulus)),
                modulus,
            )
            long = LoopCountGuard(inner.operation, inner.count, CountRelation.MORE, period - 1)
            variants.append(
                (value, (LoopDomain(inner.operation, inner.count, period, phase),), (long,))
            )
            # A short child can leave a physical slot unvisited. Partition the
            # uniform short extent, then derive its parent's real residues.
            for count in range(1, period):
                conditions = (
                    LoopCountGuard(inner.operation, inner.count, CountRelation.MORE, count - 1),
                    LoopCountGuard(inner.operation, inner.count, CountRelation.AT_MOST, count),
                )
                for parent_residue in range(period):
                    concrete = substitute(
                        base,
                        {
                            inner.count: literal(count),
                            _normalize(inner.count, self.facts): literal(count),
                            _count_core(inner.count, self.facts): literal(count),
                            outer.ordinal: literal(parent_residue),
                        },
                    )
                    concrete = _normalize(concrete, self.facts)
                    if concrete.constant is None:
                        if self._guards(conditions) is not None:
                            self.issue(
                                DependencyIssueKind.SYMBOLIC_ALIAS,
                                template.operation,
                                "short child slot visits need an exact parent-coordinate mapping",
                            )
                        continue
                    visits = [j for j in range(count) if (concrete.constant + j) % period == slot]
                    if len(visits) == 1:
                        variants.append(
                            (
                                value,
                                (
                                    LoopDomain(
                                        outer.operation, outer.count, period, parent_residue
                                    ),
                                    LoopDomain(inner.operation, inner.count, count, visits[0]),
                                ),
                                conditions,
                            )
                        )
        return tuple(variants)

    def _slots_plain(self, template, access, participant, offset, preset, conditions):
        address = access.address
        original_offset = offset
        geometry = address.geometry
        choices, streaming, contracted = [], [], []
        # Contract a dense address family using its actual native stride.
        # One strided rectangle remains one rectangle regardless of rows or
        # child trip count; original per-occurrence anchors remain unchanged.
        for loop in reversed(template.frame.loops()):
            if not _depends(offset, loop.ordinal) or _period(offset, loop.ordinal) is not None:
                continue
            count = _normalize(loop.count, self.facts).constant
            if count is None or count <= 0:
                continue
            shifted = substitute(
                offset, {loop.ordinal: expression("add", loop.ordinal, literal(1))}
            )
            delta = _normalize(expression("sub", shifted, offset), self.facts).constant
            if delta == geometry.nbytes and (
                geometry.rows == 1 or geometry.nbytes * count <= geometry.row_stride
            ):
                geometry = replace(geometry, nbytes=geometry.nbytes * count)
            elif geometry.contiguous and delta == geometry.total_nbytes:
                geometry = replace(
                    geometry, nbytes=geometry.total_nbytes * count, rows=1, row_stride=0
                )
            elif geometry.rows > 1 and delta == geometry.rows * geometry.row_stride:
                geometry = replace(geometry, rows=geometry.rows * count)
            else:
                continue
            offset = substitute(offset, {loop.ordinal: literal(0)})
            streaming.append(loop.operation)
            contracted.append(loop.operation)
        for loop in template.frame.loops():
            if not _depends(offset, loop.ordinal):
                continue
            period = _period(offset, loop.ordinal)
            if period is None:
                count = _normalize(loop.count, self.facts).constant
                capacity = address.root.owner.nbytes.data
                # This is the image of a physical address coordinate, bounded
                # by declared storage, not enumeration of native loop tasks.
                if count is not None and 0 < count <= capacity // address.geometry.total_nbytes:
                    period = count
                else:
                    streaming.append(loop.operation)
                    continue
            if period > self.MAX_PROJECTIONS:
                self.issue(
                    DependencyIssueKind.BUDGET,
                    template.operation,
                    "storage-slot period exceeds dependency proof budget",
                )
                return ()
            choices.append((loop, period))
        combinations = 1
        for _, period in choices:
            combinations *= period
        if (
            combinations * self.work.projected_accesses > self.MAX_PROJECTIONS**2
            or combinations > self.MAX_PROJECTIONS
        ):
            self.issue(
                DependencyIssueKind.BUDGET,
                template.operation,
                "storage-coordinate image exceeds dependency proof budget",
            )
            return ()
        if geometry.contiguous:
            geometry = replace(geometry, nbytes=geometry.total_nbytes, rows=1, row_stride=0)
        projected = []
        for residues in product(*(range(period) for _, period in choices)):
            substitution = {
                loop.ordinal: literal(residue)
                for (loop, _), residue in zip(choices, residues, strict=True)
            }
            value = _normalize(substitute(offset, substitution), self.facts)
            constant = polynomial(value, self.facts).get((), 0)
            require(
                constant.denominator == 1 if hasattr(constant, "denominator") else True,
                "cell displacement must be integral",
            )
            constant = int(constant)
            origin = _normalize(expression("sub", value, literal(constant)), self.facts)
            key = tuple(sorted(polynomial(origin, self.facts).items(), key=repr))
            origin = self._origin_pool.setdefault(key, origin)
            domains = (
                tuple(
                    LoopDomain(loop.operation, loop.count, period, residue)
                    for (loop, period), residue in zip(choices, residues, strict=True)
                )
                + preset
            )
            projected.append(
                _Projection(
                    template,
                    participant,
                    access,
                    origin,
                    constant,
                    constant + geometry.nbytes,
                    domains,
                    tuple(streaming),
                    geometry.rows,
                    geometry.row_stride,
                    conditions,
                    tuple(contracted),
                    tuple(
                        (loop.operation, substitute(original_offset, substitution))
                        for loop in template.frame.loops()
                    ),
                )
            )
        self.work.projected_accesses += len(projected)
        return tuple(projected)

    def _project(self):
        groups = {}
        for template in self.memory.templates:
            if template.frame.statically_empty:
                continue
            for participant in template.participants:
                for access in template.accesses:
                    private = participant if access.address.geometry.bank != "GM" else None
                    if self.admission is not None:
                        partition = self.admission.partitions[
                            (template.operation, access.operand_index, participant)
                        ]
                        require(
                            partition.root is access.address.root
                            and partition.participant is private,
                            "dependency partition disagrees with actual access",
                        )
                    for projected in self._slots(template, access, participant):
                        key = (access.address.root, private, projected.origin, projected.conditions)
                        groups.setdefault(key, []).append(projected)
        origins = {}
        for (root, private, origin, conditions), projections in groups.items():
            origins.setdefault((root, private, conditions), []).append((origin, projections))
        for items in origins.values():
            if len(items) > 1 and any(p.access.mode.writes for _, ps in items for p in ps):
                # Different symbolic translations are not a non-alias proof.
                self.issue(
                    DependencyIssueKind.SYMBOLIC_ALIAS,
                    items[0][1][0].template.operation,
                    "different symbolic coordinate maps of one writable root need an explicit alias/coordinate proof",
                )
        for (root, private, origin, conditions), projections in groups.items():
            streaming = tuple(dict.fromkeys(loop for p in projections for loop in p.streaming))
            for loop_op in streaming:
                loop = self.memory.control.loops[loop_op]
                if not _depends(origin, loop.ordinal):
                    continue
                shifted = substitute(
                    origin, {loop.ordinal: expression("add", loop.ordinal, literal(1))}
                )
                delta = _normalize(expression("sub", shifted, origin), self.facts)
                span = max(p.stop + (p.rows - 1) * p.row_stride for p in projections) - min(
                    p.start for p in projections
                )
                if (
                    not nonnegative(expression("sub", delta, literal(span)), self.facts)
                    and any(p.access.mode.writes for p in projections)
                    and not self._separate_tiles(projections, loop)
                ):
                    self.issue(
                        DependencyIssueKind.NONPERIODIC_REUSE,
                        loop_op,
                        "streaming coordinate is not proved to separate successive full storage fragments",
                    )
            for start, stop, rows, stride, covered in self._cell_regions(projections):
                cell = StorageCell(
                    root, private, origin, start, stop, streaming, rows, stride, conditions
                )
                self.work.cells += 1
                # Dense family unions do not identify per-iteration cells.
                # A shifted/invariant access inside the same loop needs a
                # relational coordinate map; never turn it into same-j RAW
                # or suppress its real cross-iteration WAR.
                contracted = {loop for p in covered for loop in p.contracted}
                for loop in contracted:
                    coordinates = [
                        (dict(p.coordinates)[loop], p.access.mode)
                        for p in covered
                        if loop in dict(p.coordinates)
                    ]
                    if any(mode.writes for _, mode in coordinates) and any(
                        not equivalent(coordinates[0][0], coordinate, self.facts)
                        for coordinate, _ in coordinates[1:]
                    ):
                        self.issue(
                            DependencyIssueKind.NONPERIODIC_REUSE,
                            loop,
                            "contracted families have different per-occurrence coordinates; relational overlap is not closed",
                            cell,
                        )
                natives = {}
                for projection in covered:
                    key = (projection.template.operation, projection.participant)
                    template, accesses = natives.setdefault(key, (projection.template, {}))
                    accesses[projection.access.operand_index] = projection.access
                    for logical in projection.template.frame.loops():
                        domain = next(
                            (d for d in projection.slots if d.loop is logical.operation),
                            LoopDomain(logical.operation, logical.count),
                        )
                        self.patterns.setdefault((cell, logical.operation), set()).add(
                            (domain.period, domain.residue)
                        )
                for (operation, participant), (template, accesses) in natives.items():
                    self.effects.setdefault(operation, {}).setdefault(cell, []).append(
                        CellEffect(template, participant, tuple(accesses.values()))
                    )

    def _cell_regions(self, projections):
        """Sweep actual rectangle boundaries, never native rows or loop trips."""
        strided = [p for p in projections if p.rows > 1]
        pitch = 0
        for projection in strided:
            pitch = gcd(pitch, projection.row_stride)
        rectangles = []
        if pitch:
            for index, projection in enumerate(projections):
                row, column = divmod(projection.start, pitch)
                width = projection.stop - projection.start
                if projection.rows > 1:
                    if projection.row_stride != pitch or column + width > pitch:
                        self.issue(
                            DependencyIssueKind.FRAGMENT_MAPPING,
                            projection.template.operation,
                            "strided native families need a common exact row-coordinate map",
                        )
                        return
                    rectangles.append((row, row + projection.rows, column, column + width, index))
                    continue
                end_row, end_column = divmod(projection.stop, pitch)
                if row == end_row:
                    rectangles.append((row, row + 1, column, end_column, index))
                    continue
                if column:
                    rectangles.append((row, row + 1, column, pitch, index))
                    row += 1
                if row < end_row:
                    rectangles.append((row, end_row, 0, pitch, index))
                if end_column:
                    rectangles.append((end_row, end_row + 1, 0, end_column, index))
        else:
            rectangles = [(0, 1, p.start, p.stop, i) for i, p in enumerate(projections)]
        row_cuts = sorted({point for r in rectangles for point in r[:2]})
        for row_start, row_stop in pairwise(row_cuts):
            active = [r for r in rectangles if r[0] <= row_start and row_stop <= r[1]]
            starts, stops, coverage = {}, {}, {}
            for rectangle in active:
                starts.setdefault(rectangle[2], []).append(rectangle[4])
                stops.setdefault(rectangle[3], []).append(rectangle[4])
            columns = sorted(starts.keys() | stops.keys())
            for start, stop in pairwise(columns):
                for index in stops.get(start, ()):
                    del coverage[index]
                for index in starts.get(start, ()):
                    coverage[index] = projections[index]
                if coverage:
                    yield (
                        row_start * pitch + start,
                        row_start * pitch + stop,
                        row_stop - row_start,
                        pitch,
                        tuple(coverage.values()),
                    )

    def build(self):
        self._project()
        return PartitionedAccesses(
            self.memory,
            MappingProxyType(
                {
                    operation: MappingProxyType({cell: tuple(uses) for cell, uses in cells.items()})
                    for operation, cells in self.effects.items()
                }
            ),
            MappingProxyType({key: frozenset(value) for key, value in self.patterns.items()}),
            tuple(self.issues),
            self.work,
        )

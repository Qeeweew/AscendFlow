"""Keep exact storage fragments where available, otherwise source memory order.

Slot/coordinate and rectangle analyses are optional precision. An uncertain
mapping retains a source-order AliasRegion and the original native accesses.
The result creates neither dependencies nor event/completion state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations, pairwise, product
from math import gcd, lcm
from types import MappingProxyType

from xdsl.ir import Operation

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..indexing import (
    arguments,
    equivalent,
    exact_constant,
    interval,
    nonnegative,
    normalize_or_identity,
    polynomial,
)
from ..scalar import Expr, expression, infer, literal, substitute
from ..target.ascend910b import Actor
from .native import AccessMode
from .structured_memory import MemoryAccess, MemoryTemplate, NativeMemoryFacts
from .structured_regions import (
    AliasRegion,
    CountRelation,
    DependencyWork,
    Guard,
    LoopCountGuard,
    LoopDomain,
    StorageCell,
)


@dataclass(frozen=True)
class CellEffect:
    """One actual instruction's operands in a cell or may-alias ordering domain.

    Physical footprints and conditional numerical reads stay on each operand.
    An ordering domain never turns a partial write into an initialization.
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
    effects: Mapping[TaskOp, Mapping[StorageCell | AliasRegion, tuple[CellEffect, ...]]]
    patterns: Mapping[tuple[StorageCell | AliasRegion, Operation], tuple[int, int | Expr]]
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
        # Only the loaded value is an indirect index. Its capture dependencies
        # describe which invocation read it; they do not turn every ordinary
        # group/ordinal address in that invocation into data-dependent access.
        self.indirect_values = frozenset(
            memory.control.environment.expression(template.operation.value).args[0]
            for template in memory.templates
            if template.operation.native.data == "index_load" and template.frame.loops()
        )
        self.work = DependencyWork(native_templates=len(memory.templates))
        self.effects, self.patterns, self._origin_pool = {}, {}, {}

    def _separate_rectangles(self, projections, loop):
        """A monotone row/column coordinate separates exact ND rectangles.

        Ancestor coordinates are fixed during this loop. A chosen separating
        axis may not depend on any descendant clock. Other axes can vary: a
        disjoint row interval already separates every byte of those rows.
        """
        for axis in (0, 1):
            anchor, extents = None, None
            low = high = 0
            valid = True
            for projection in projections:
                address = projection.access.address
                rectangle, geometry = address.rectangle, address.geometry
                if (
                    rectangle is None
                    or projection.slots
                    or projection.contracted
                    or projection.stop - projection.start != geometry.nbytes
                    or projection.rows != geometry.rows
                    or projection.row_stride != geometry.row_stride
                    or extents is not None
                    and extents != rectangle.extents
                ):
                    valid = False
                    break
                bindings = {
                    self.memory.control.sid: literal(
                        1 if projection.participant is Actor.AIV1 else 0
                    )
                }
                actual = expression("add", projection.origin, literal(projection.start))
                if not equivalent(actual, substitute(address.offset, bindings), self.facts):
                    valid = False
                    break
                value = substitute(rectangle.origins[axis], bindings)
                inside = False
                for other in projection.template.frame.loops():
                    if other.operation is loop.operation:
                        inside = True
                    elif inside and _depends(value, other.ordinal):
                        valid = False
                        break
                if not inside or not valid:
                    valid = False
                    break
                if anchor is None:
                    anchor = value
                    high = rectangle.widths[axis]
                else:
                    displacement = exact_constant(expression("sub", value, anchor), self.facts)
                    if displacement is None:
                        valid = False
                        break
                    low = min(low, displacement)
                    high = max(high, displacement + rectangle.widths[axis])
                extents = rectangle.extents
            if not valid or anchor is None:
                continue
            shifted = substitute(
                anchor, {loop.ordinal: expression("add", loop.ordinal, literal(1))}
            )
            delta = expression("sub", shifted, anchor)
            if nonnegative(expression("sub", delta, literal(high - low)), self.facts):
                return True
        return False

    def _disjoint_origin_maps(self, left, right):
        """Whole active-domain ND boxes separate these symbolic families."""
        for a, b in product(left, right):
            if not (a.access.mode.writes or b.access.mode.writes):
                continue
            ar, br = a.access.address.rectangle, b.access.address.rectangle
            if ar is None or br is None or ar.extents != br.extents:
                return False
            separated = False
            for axis in (0, 1):
                al, ah = interval(ar.origins[axis], a.template.frame.domains(), self.facts)
                bl, bh = interval(br.origins[axis], b.template.frame.domains(), self.facts)
                if nonnegative(
                    expression("sub", bl, expression("add", ah, literal(ar.widths[axis]))),
                    self.facts,
                ) or nonnegative(
                    expression("sub", al, expression("add", bh, literal(br.widths[axis]))),
                    self.facts,
                ):
                    separated = True
                    break
            if not separated:
                return False
        return True

    def _slots(self, template, access, participant):
        address = access.address
        if arguments(address.offset) & self.indirect_values:
            # Loaded indices describe actual addresses, not affine functions
            # of a loop ordinal. Keep unknown overlap in the existing alias
            # order domain; a later reload may return a different value.
            return None
        bindings = {self.memory.control.sid: literal(1 if participant is Actor.AIV1 else 0)}
        # Keep remainder nodes until their finite image has been selected.
        # Polynomial normalization expands rem into x-d*floor(x/d), which
        # would hide exactly the storage-period contract needed here.
        offset = substitute(address.offset, bindings)
        variants = self._compound_slots(template, offset)
        if variants is None:
            return None
        result = []
        for value, slots, conditions in variants:
            projected = self._slots_plain(template, access, participant, value, slots, conditions)
            if projected is None:
                return None
            result.extend(projected)
        return tuple(result)

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
            return None
        shifted = substitute(
            numerator, {inner.ordinal: expression("add", inner.ordinal, literal(1))}
        )
        delta = exact_constant(expression("sub", shifted, numerator), self.facts)
        if delta != 1 or len(active) != 2:
            return None
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
                            normalize_or_identity(inner.count, self.facts): literal(count),
                            _count_core(inner.count, self.facts): literal(count),
                            outer.ordinal: literal(parent_residue),
                        },
                    )
                    concrete = normalize_or_identity(concrete, self.facts)
                    if concrete.constant is None:
                        return None
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
            count = exact_constant(loop.count, self.facts)
            if count is None and access.mode is AccessMode.READ:
                # A bounded dense reader prefix has a fixed spatial union,
                # but retirement still ranges over its actual loop.count.
                # Do not apply this union to conditional writes or invent a
                # maximum-count LAST occurrence.
                bound = infer(loop.count, self.facts)
                count = None if bound is None else bound.maximum
            if count is None or count <= 0:
                continue
            shifted = substitute(
                offset, {loop.ordinal: expression("add", loop.ordinal, literal(1))}
            )
            delta = exact_constant(expression("sub", shifted, offset), self.facts)
            if delta == geometry.nbytes and (
                geometry.rows == 1 or geometry.nbytes * count <= geometry.row_stride
            ):
                geometry = replace(geometry, nbytes=geometry.nbytes * count)
            elif geometry.contiguous and delta == geometry.total_nbytes:
                geometry = replace(
                    geometry, nbytes=geometry.total_nbytes * count, rows=1, row_stride=0
                )
            elif geometry.contiguous and delta is not None and delta > geometry.total_nbytes:
                # A dense loop can visit a strided rectangle with holes.
                # Keep the holes instead of allocating one cell/event per
                # physical image of its finite iteration coordinate.
                geometry = replace(
                    geometry, nbytes=geometry.total_nbytes, rows=count, row_stride=delta
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
                count = exact_constant(loop.count, self.facts)
                capacity = address.root.owner.nbytes.data
                # This is the image of a physical address coordinate, bounded
                # by declared storage, not enumeration of native loop tasks.
                if count is not None and 0 < count <= capacity // address.geometry.total_nbytes:
                    period = count
                else:
                    streaming.append(loop.operation)
                    continue
            if period > self.MAX_PROJECTIONS:
                return None
            choices.append((loop, period))
        combinations = 1
        for _, period in choices:
            combinations *= period
        if (
            combinations * self.work.projected_accesses > self.MAX_PROJECTIONS**2
            or combinations > self.MAX_PROJECTIONS
        ):
            return None
        if geometry.contiguous:
            geometry = replace(geometry, nbytes=geometry.total_nbytes, rows=1, row_stride=0)
        projected = []
        for residues in product(*(range(period) for _, period in choices)):
            substitution = {
                loop.ordinal: literal(residue)
                for (loop, _), residue in zip(choices, residues, strict=True)
            }
            value = normalize_or_identity(substitute(offset, substitution), self.facts)
            constant = polynomial(value, self.facts).get((), 0)
            require(
                constant.denominator == 1 if hasattr(constant, "denominator") else True,
                "cell displacement must be integral",
            )
            constant = int(constant)
            origin = normalize_or_identity(expression("sub", value, literal(constant)), self.facts)
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
        groups, instances, ordered = {}, {}, set()
        readers = {
            access.address.root
            for template in self.memory.templates
            if not template.frame.statically_empty
            for access in template.accesses
            if access.mode.reads
        }
        for template in self.memory.templates:
            if template.frame.statically_empty:
                continue
            for participant in template.participants:
                for access in template.accesses:
                    # The source contract relates external writes only. If
                    # this kernel also reads the tensor, retain its memory
                    # order: a preceding write is not necessarily its producer.
                    # Real native effects, output completion and all UB input
                    # retirements remain in the owned memory snapshot.
                    if (
                        access.mode.writes
                        and access.address.root not in readers
                        and access.address.root.owner.write_disjoint is not None
                    ):
                        self.work.disjoint_writes += 1
                        continue
                    private = participant if access.address.geometry.bank != "GM" else None
                    if self.admission is not None:
                        instance = self.admission.instances[
                            (template.operation, access.operand_index, participant)
                        ]
                        require(
                            instance.root is access.address.root
                            and instance.participant is private,
                            "dependency instance disagrees with actual access",
                        )
                    identity = access.address.root, private
                    instances.setdefault(identity, {}).setdefault(
                        (template.operation, participant), (template, [])
                    )[1].append(access)
                    projections = self._slots(template, access, participant)
                    if projections is None:
                        ordered.add(identity)
                        continue
                    for projected in projections:
                        key = (access.address.root, private, projected.origin, projected.conditions)
                        groups.setdefault(key, []).append(projected)
        origins = {}
        for (root, private, origin, conditions), projections in groups.items():
            origins.setdefault((root, private, conditions), []).append((origin, projections))
        for (root, private, _), items in origins.items():
            for (_, left), (_, right) in combinations(items, 2):
                if any(
                    p.access.mode.writes for p in (*left, *right)
                ) and not self._disjoint_origin_maps(left, right):
                    ordered.add((root, private))
        fragments = {}
        for (root, private, origin, conditions), projections in groups.items():
            if (root, private) in ordered:
                continue
            streaming = tuple(dict.fromkeys(loop for p in projections for loop in p.streaming))
            for loop_op in streaming:
                loop = self.memory.control.loops[loop_op]
                if not _depends(origin, loop.ordinal):
                    continue
                shifted = substitute(
                    origin, {loop.ordinal: expression("add", loop.ordinal, literal(1))}
                )
                delta = normalize_or_identity(expression("sub", shifted, origin), self.facts)
                span = max(p.stop + (p.rows - 1) * p.row_stride for p in projections) - min(
                    p.start for p in projections
                )
                if (
                    not nonnegative(expression("sub", delta, literal(span)), self.facts)
                    and any(p.access.mode.writes for p in projections)
                    and not self._separate_rectangles(projections, loop)
                ):
                    ordered.add((root, private))
            regions = self._cell_regions(projections)
            if regions is None:
                ordered.add((root, private))
                continue
            fragments[root, private, origin, conditions] = regions
            for _, _, _, _, covered in regions:
                patterns = {}
                for p in covered:
                    for loop in p.template.frame.loops():
                        slot = next(
                            (d for d in p.slots if d.loop is loop.operation),
                            LoopDomain(loop.operation, loop.count),
                        )
                        patterns.setdefault(loop.operation, set()).add((slot.period, slot.residue))
                if any(len(choices) > 1 for choices in patterns.values()):
                    ordered.add((root, private))
                actors = {p.participant for p in covered}
                if {Actor.AIV0, Actor.AIV1} <= actors and any(
                    p.participant in {Actor.AIV0, Actor.AIV1} and p.access.mode.writes
                    for p in covered
                ):
                    ordered.add((root, private))
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
                        ordered.add((root, private))
        # A may-alias domain has no invented byte envelope or logical KILL.
        # All native operands remain exact; only their source memory order is
        # conservatively composed. This uses the same loop and event path.
        for (root, private), natives in instances.items():
            if (root, private) not in ordered:
                continue
            actors = {participant for _, participant in natives}
            peers = (
                (Actor.AIV0, Actor.AIV1)
                if private is None and {Actor.AIV0, Actor.AIV1} <= actors
                else (None,)
            )
            for peer in peers:
                cell = AliasRegion(root, private, peer)
                self.work.alias_regions += 1
                for (operation, participant), (template, accesses) in natives.items():
                    if peer is not None and participant not in {Actor.AIC, peer}:
                        continue
                    self.effects.setdefault(operation, {}).setdefault(cell, []).append(
                        CellEffect(template, participant, tuple(accesses))
                    )
        for (root, private, origin, conditions), regions in fragments.items():
            if (root, private) in ordered:
                continue
            streaming = tuple(
                dict.fromkeys(
                    loop for p in groups[root, private, origin, conditions] for loop in p.streaming
                )
            )
            for start, stop, rows, stride, covered in regions:
                cell = StorageCell(
                    root, private, origin, start, stop, streaming, rows, stride, conditions
                )
                self.work.cells += 1
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
                        return None
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
        regions = []
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
                    regions.append(
                        (
                            row_start * pitch + start,
                            row_start * pitch + stop,
                            row_stop - row_start,
                            pitch,
                            tuple(coverage.values()),
                        )
                    )
        return tuple(regions)

    def build(self):
        self._project()
        # Unresolved/multiple-residue partitions were already retained as
        # AliasRegion above. Consumers receive one optional precision fact,
        # never a new frontier-closure obligation.
        require(
            all(len(value) == 1 for value in self.patterns.values()),
            "access partition retained ambiguous precision instead of an alias region",
        )
        return PartitionedAccesses(
            self.memory,
            MappingProxyType(
                {
                    operation: MappingProxyType({cell: tuple(uses) for cell, uses in cells.items()})
                    for operation, cells in self.effects.items()
                }
            ),
            MappingProxyType({key: next(iter(value)) for key, value in self.patterns.items()}),
            self.work,
        )

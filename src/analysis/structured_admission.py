"""Storage instances, defined reads, and borrowed Tile admission.

All inputs are owned facts from the same recursive lowering invocation. This
pass uses symbolic byte regions and finite native row geometry, never runtime
iteration instances or a completion graph. Initialized storage and a selected
logical writer establish numerical definedness, not asynchronous readiness.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from xdsl.dialects import arith
from xdsl.ir import SSAValue

from ..diagnostics import ErrorKind, require
from ..dialects.physical import FixConfigOp, TaskOp
from ..indexing import (
    arguments,
    domain_nonnegative,
    equivalent,
    exact_constant,
    interval,
    nonnegative,
    normalize,
    polynomial,
)
from ..scalar import Expr, expression, literal, substitute
from ..target.ascend910b import Actor
from ..target.native import ITEMSIZE
from .matrix_regions import RegionDefinitions
from .native import ConditionalRead
from .structured_bounds import MemoryBoundsFacts
from .structured_control import LogicalLoop
from .structured_memory import MemoryAccess, MemoryTemplate, NativeMemoryFacts
from .value_bindings import LastTile, TileDefinition, TileUse

if TYPE_CHECKING:
    from ..passes.lower_native import NativeStructure


@dataclass(frozen=True, slots=True)
class StorageInstance:
    """A real allocation in one group, private to an actor unless it is GM.

    Group independence is a source parallel-program precondition. It is not
    an address-injectivity theorem reconstructed by memory admission.
    """

    root: SSAValue
    participant: Actor | None


@dataclass(frozen=True, slots=True)
class ReadAdmission:
    template: MemoryTemplate
    access: MemoryAccess
    participant: Actor
    writers: tuple[TaskOp, ...] | DefinitionSources
    external: bool = False
    initialization: ConditionalRead | None = None


@dataclass(frozen=True, eq=False, slots=True)
class DefinitionSources(Sequence):
    """Immutable writer prefix; reads retain a snapshot without copying it.

    Iteration is an explicit diagnostic query, not an admission operation.
    A later extension cannot change a previously admitted read's sources.
    """

    previous: DefinitionSources | None = field(default=None, repr=False)
    members: tuple[TaskOp, ...] = ()
    count: int = 0

    def extend(self, members):
        members = tuple(members)
        return DefinitionSources(self, members, self.count + len(members)) if members else self

    def __len__(self):
        return self.count

    def __iter__(self):
        nodes, node = [], self
        while node is not None:
            if node.members:
                nodes.append(node.members)
            node = node.previous
        for members in reversed(nodes):
            yield from members

    def __getitem__(self, index):
        return tuple(self)[index]

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

    def __hash__(self):
        return hash(tuple(self))


@dataclass(frozen=True, slots=True)
class ValueSnapshot:
    """Defined entry/update sources frozen at an actual borrowed-value cut.

    The shared source prefix is numerical state, never completion or reader
    retirement. Later writes cannot change this snapshot's source membership.
    """

    definition: TileDefinition
    instance: StorageInstance
    writers: tuple[TaskOp, ...] | DefinitionSources
    external: bool = False


@dataclass(frozen=True, slots=True)
class TileAdmission:
    use: TileUse
    definition: TileDefinition
    selections: tuple[tuple[LogicalLoop, Expr], ...]
    snapshots: tuple[ValueSnapshot, ...] = ()


@dataclass(frozen=True)
class StructuredAdmission:
    structure: NativeStructure
    memory: NativeMemoryFacts
    bounds: MemoryBoundsFacts
    instances: Mapping[tuple[TaskOp, int, Actor], StorageInstance]
    reads: tuple[ReadAdmission, ...]
    values: tuple[TileAdmission, ...]
    configurations: tuple[FixConfigOp | TaskOp, ...]
    work: Mapping[str, int]


def _require(condition, message, operation, *, kind=ErrorKind.UNSUPPORTED, witness=()):
    require(
        condition,
        message,
        kind=kind,
        phase="structured-admission",
        location=str(operation.location),
        witness=witness,
    )


def _participant(actor, sid):
    return {sid: literal(1 if actor is Actor.AIV1 else 0)}


@dataclass(frozen=True, slots=True)
class _Span:
    base: Expr
    width: Expr
    rows: int = 1
    pitch: int = 0


def _span(access, bindings):
    geometry = access.address.geometry
    result = _Span(
        substitute(access.address.offset, bindings),
        literal(geometry.nbytes),
        geometry.rows,
        geometry.row_stride,
    )
    return _collapse(result)


def _collapse(span):
    if span.rows > 1 and span.width.constant is not None and span.width.constant >= span.pitch:
        return _Span(
            span.base, expression("add", span.width, literal((span.rows - 1) * span.pitch))
        )
    return span


def _guards(frame, environment):
    result = []
    for scope in frame.ancestors():
        if scope.branch is None:
            continue
        if len(scope.comparisons) != 1:
            result.append((scope.operation.cond, scope.branch))
            continue
        (test,) = scope.comparisons
        predicate = test.predicate
        if scope.branch:
            predicate = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[
                predicate
            ]
        result.append(
            (
                test.lhs,
                test.rhs,
                ("==", "!=", "<", "<=", ">", ">=").index(predicate),
                scope.branch,
            )
        )
    return tuple(result)


def _same_guard(left, right, facts):
    if left == right:
        return True
    if len(left) != len(right):
        return False
    if len(left) == 2:
        return left == right
    a, b, predicate, branch = left
    c, d, other_predicate, other_branch = right
    return (
        predicate == other_predicate
        and branch == other_branch
        and (
            equivalent(a, c, facts)
            and equivalent(b, d, facts)
            or predicate in {0, 1}
            and equivalent(a, d, facts)
            and equivalent(b, c, facts)
        )
    )


def _nonempty_count(loop, facts, bindings):
    """A writer loop must run in its actual enclosing invocation.

    Ancestor ranges are established by control analysis. Never borrow this
    loop's own body ranges: that would discard an empty writer's path.
    """
    count = substitute(loop.count, bindings)
    domains = tuple(tuple(substitute(v, bindings) for v in domain) for domain in loop.entry_domains)
    return (
        count if domain_nonnegative(expression("sub", count, literal(1)), domains, facts) else None
    )


def _summarize(span, loop, facts, bindings):
    count = _nonempty_count(loop, facts, bindings)
    if count is None:
        return None
    if loop.ordinal.args[0] not in arguments(span.base):
        return span
    base = substitute(span.base, {loop.ordinal: literal(0)})
    difference = expression("sub", span.base, base)
    varying = polynomial(difference, facts, periodic=True)
    if len(varying) != 1:
        return None
    ((atoms, coefficient),) = varying.items()
    if len(atoms) != 1 or coefficient.denominator != 1:
        return None
    atom, stride = atoms[0], int(coefficient)
    if stride < 0:
        return None
    if atom == loop.ordinal:
        iterations = count
    elif atom.kind == "rem" and atom.args[0] == loop.ordinal and atom.args[1].constant is not None:
        iterations = expression("min", count, atom.args[1])
    else:
        return None
    if span.width.constant is not None and stride <= span.width.constant:
        width = expression(
            "add",
            span.width,
            expression("mul", literal(stride), expression("sub", iterations, literal(1))),
        )
        return _collapse(_Span(base, width, span.rows, span.pitch))
    if span.rows == 1 and iterations.constant is not None:
        return _Span(base, span.width, iterations.constant, stride)
    if span.rows > 1 and stride == span.rows * span.pitch and iterations.constant is not None:
        # Successive complete row groups keep precisely the original holes.
        # This is one strided rectangle, not the contiguous bounding box of
        # the writes. No runtime loop instances or completion facts expand.
        return _Span(base, span.width, span.rows * iterations.constant, span.pitch)
    return None


def _writer_coordinates(
    writer,
    access,
    actor,
    reader,
    memory,
    *,
    selections=(),
    ignore_guards=False,
    logical_value=False,
    rectangular=False,
):
    environment, facts = memory.control.environment, memory.control.environment.facts
    bindings = _participant(actor, memory.control.sid) | dict(selections)
    windows = {}
    writer_loops = writer.frame.loops()
    reader_guards = _guards(reader.frame, environment)
    for guard in _guards(writer.frame, environment):
        if ignore_guards:
            continue
        if any(_same_guard(guard, other, facts) for other in reader_guards):
            continue
        first = None
        if len(guard) == 4:
            left, right, predicate, branch = guard
            if (predicate == 0 and branch == 0) or (predicate == 1 and branch == 1):
                for loop in writer_loops:
                    for a, b in ((left, right), (right, left)):
                        if equivalent(a, loop.induction, facts) and equivalent(
                            b, loop.lower, facts
                        ):
                            first = loop
        if first is None:
            if not rectangular or len(guard) != 4:
                return None
            left, right, predicate, branch = guard
            predicate = {0: 1, 1: 0, 2: 5, 3: 4, 4: 3, 5: 2}[predicate] if branch else predicate
            for loop in writer_loops:
                if (
                    loop in reader.frame.loops()
                    or loop.count.constant is None
                    or loop.lower.constant is None
                ):
                    continue
                for a, b, p in (
                    (left, right, predicate),
                    (right, left, {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3}[predicate]),
                ):
                    if b.constant is None or not equivalent(a, loop.induction, facts):
                        continue
                    low, high = windows.get(loop, (0, loop.count.constant))
                    difference = b.constant - loop.lower.constant
                    if p == 2:
                        high = min(high, -(-difference // loop.step))
                    elif p == 3:
                        high = min(high, difference // loop.step + 1)
                    elif p == 4:
                        low = max(low, difference // loop.step + 1)
                    elif p == 5:
                        low = max(low, -(-difference // loop.step))
                    else:
                        continue
                    if low >= high:
                        return None
                    if logical_value and loop.ordinal in bindings:
                        selected = bindings[loop.ordinal].constant
                        if selected is None or not low <= selected < high:
                            return None
                    windows[loop] = (low, high)
                    break
                else:
                    continue
                break
            else:
                return None
            continue
        if logical_value and (
            first.ordinal not in bindings
            or not equivalent(bindings[first.ordinal], literal(0), facts)
        ):
            return None
        bindings[first.ordinal] = literal(0)
    return bindings, windows


def _value_payload(access):
    """The declared result view, separately from a prefix's actual writes.

    This is only an enclosure for a borrowed value. It cannot establish any
    initialized byte; that needs the frozen definition state at the update.
    """
    address = access.address
    if address.valid_rows is None:
        return access
    matrix = address.matrix
    if matrix is not None:
        matrix = replace(
            matrix,
            stops=tuple(
                expression("add", start, literal(width))
                for start, width in zip(matrix.starts, address.axes.widths, strict=True)
            ),
        )
    return replace(access, address=replace(address, valid_rows=None, matrix=matrix))


def _writer_span(writer, access, actor, reader, memory, **options):
    if access.address.valid_rows is not None:
        return None  # Byte envelopes cannot describe a partial definition.
    scope = _writer_coordinates(writer, access, actor, reader, memory, **options)
    if scope is None:
        return None
    bindings, _ = scope
    facts = memory.control.environment.facts
    writer_loops, reader_loops = writer.frame.loops(), reader.frame.loops()
    result = _span(access, bindings)
    for loop in reversed(writer_loops):
        if loop not in reader_loops:
            if loop.ordinal not in bindings:
                result = _summarize(result, loop, facts, bindings)
                if result is None:
                    return None
            elif _nonempty_count(loop, facts, bindings) is None:
                return None
    return result


def _writer_matrix(writer, access, actor, reader, memory, **options):
    scope = _writer_coordinates(writer, access, actor, reader, memory, rectangular=True, **options)
    if scope is None:
        return None
    bindings, windows = scope
    region = access.address.matrix.bind(bindings)
    facts = memory.control.environment.facts
    for loop in reversed(writer.frame.loops()):
        if loop in reader.frame.loops():
            continue
        if loop.ordinal in bindings:
            if _nonempty_count(loop, facts, bindings) is None:
                return None
            continue
        if loop in windows:
            first, stop = windows[loop]
            region = region.bind({loop.ordinal: expression("add", loop.ordinal, literal(first))})
            count = literal(stop - first)
        else:
            count = _nonempty_count(loop, facts, bindings)
        if count is None:
            return None
        region = region.sweep(loop.ordinal, count, facts)
        if region is None:
            return None
    return region


def _conditional_matrices(alternatives, reader, memory):
    control, groups = memory.control, {}
    facts, reader_loops = control.environment.facts, reader.frame.loops()
    for writer, output, actor, path in alternatives:
        region = output.address.matrix.bind(_participant(actor, control.sid))
        loops = writer.frame.loops()
        for loop in reversed(loops):
            if loop in reader_loops or any(
                len(guard) != 4
                or not all(control.invariant_over(value, loop.operation) for value in guard[:2])
                for guard in path
            ):
                break
            count = _nonempty_count(loop, facts, _participant(actor, control.sid))
            summary = None if count is None else region.sweep(loop.ordinal, count, facts)
            if summary is None:
                break
            region, loops = summary, loops[:-1]
        key = actor, loops, region.canonical(facts)
        paths, writers = groups.setdefault(key, ([], []))
        paths.append(path)
        writers.append(writer.operation)
    for (actor, loops, region), (paths, writers) in groups.items():
        if not _complete_branch_paths(paths, facts):
            continue
        for loop in reversed(loops):
            if loop not in reader_loops:
                count = _nonempty_count(loop, facts, _participant(actor, control.sid))
                region = None if count is None else region.sweep(loop.ordinal, count, facts)
                if region is None:
                    break
        if region is not None:
            yield region, tuple(writers)


class _MatrixDefinitions:
    """Incremental region transfers per real reader invocation/activation.

    This is numerical definedness only. Storage ownership and stale Tile
    checks continue to use the actual accesses and selected writer versions.
    """

    def __init__(self, memory, work):
        self.memory, self.work = memory, work
        self.writes, self.contexts = {}, {}

    def write(self, instance, template, access, actor):
        self.writes.setdefault(instance, []).append((template, access, actor))

    def before(self, instance, reader):
        memory, facts = self.memory, self.memory.control.environment.facts
        guards = _guards(reader.frame, memory.control.environment)
        key = instance, reader.frame.loops(), guards
        cursor, definitions, guarded = self.contexts.setdefault(key, (0, RegionDefinitions(), []))
        writes = self.writes.get(instance, ())
        for writer, output, actor in writes[cursor:]:
            self.work["l1_region_transfers"] += 1
            region = _writer_matrix(writer, output, actor, reader, memory)
            if region is not None:
                definitions.add(region, (writer.operation,), facts)
            missing = tuple(
                guard
                for guard in _guards(writer.frame, memory.control.environment)
                if not any(_same_guard(guard, other, facts) for other in guards)
            )
            if missing:
                guarded.append((writer, output, actor, missing))
        if cursor != len(writes):
            for region, writers in _conditional_matrices(guarded, reader, memory):
                definitions.add(region, writers, facts)
        self.contexts[key] = len(writes), definitions, guarded
        return definitions


class _DefinitionSpans:
    """Exact monotone byte union; constant adjacent intervals are one fact.

    Pitched holes and symbolic coordinates retain their original spans. This
    union provides numerical definedness only, never ownership or retirement.
    """

    def __init__(self):
        self.intervals = []
        self.symbolic = {}

    def add(self, span, facts):
        base, width = exact_constant(span.base, facts), exact_constant(span.width, facts)
        if span.rows != 1 or base is None or width is None:
            self.symbolic.setdefault(span, None)
            return
        if width <= 0:
            return
        end = base + width
        intervals = self.intervals
        first = bisect_left(intervals, (base,))
        if first and intervals[first - 1][1] >= base:
            first -= 1
            base, end = intervals[first][0], max(end, intervals[first][1])
        stop = first
        while stop < len(intervals) and intervals[stop][0] <= end:
            end = max(end, intervals[stop][1])
            stop += 1
        intervals[first:stop] = [(base, end)]

    def spans(self):
        return tuple(_Span(literal(a), literal(b - a)) for a, b in self.intervals) + tuple(
            self.symbolic
        )


class _SpanDefinitions:
    """Transfer each new real writer once per reader loop/activation context."""

    def __init__(self, memory, work):
        self.memory, self.work = memory, work
        self.writes, self.contexts = {}, {}

    def write(self, instance, template, access, actor):
        self.writes.setdefault(instance, []).append((template, access, actor))

    def before(self, instance, reader):
        memory, facts = self.memory, self.memory.control.environment.facts
        guards = _guards(reader.frame, memory.control.environment)
        key = instance, reader.frame.loops(), guards
        cursor, definitions, guarded, writers, known = self.contexts.setdefault(
            key, (0, _DefinitionSpans(), [], DefinitionSources(), set())
        )
        writes = self.writes.get(instance, ())
        for writer, output, actor in writes[cursor:]:
            self.work["span_region_transfers"] += 1
            span = _writer_span(writer, output, actor, reader, memory)
            if span is not None:
                definitions.add(span, facts)
                if writer.operation not in known:
                    known.add(writer.operation)
                    writers = writers.extend((writer.operation,))
                    self.work["definition_source_nodes"] += 1
                    self.work["definition_source_members"] += 1
            missing = tuple(
                guard
                for guard in _guards(writer.frame, memory.control.environment)
                if not any(_same_guard(guard, other, facts) for other in guards)
            )
            if missing:
                guarded.append((writer, output, actor, missing))
        if cursor != len(writes):
            conditional = []
            for span, members in _conditional_spans(guarded, reader, memory):
                definitions.add(span, facts)
                for member in members:
                    if member not in known:
                        known.add(member)
                        conditional.append(member)
            writers = writers.extend(conditional)
            self.work["definition_source_nodes"] += bool(conditional)
            self.work["definition_source_members"] += len(conditional)
        self.contexts[key] = len(writes), definitions, guarded, writers, known
        return definitions.spans(), writers


def _complete_branch_paths(paths, facts):
    """Cover a full ordered branch tree without assuming any input values.

    Branch identities retain their actual SSA and captured coordinates. A
    missing leaf stays missing; unrelated conditions are not combined into a
    Boolean-product search. No guard is evaluated or moved by this proof.
    """
    if any(not path for path in paths):
        return True
    roots = []
    for head, *tail in paths:
        condition = (*head[:-1], 0)
        for prior, branches in roots:
            if _same_guard(condition, prior, facts):
                break
        else:
            branches = ([], [])
            roots.append((condition, branches))
        branches[head[-1]].append(tuple(tail))
    return any(
        left
        and right
        and _complete_branch_paths(left, facts)
        and _complete_branch_paths(right, facts)
        for _, (left, right) in roots
    )


def _conditional_spans(alternatives, reader, memory):
    """Join complete branch definitions within one actual participant.

    A condition invariant over a child loop can guard its complete summary.
    Otherwise a common footprint and remaining loop nest make the branch proof
    pointwise, before eliminating those loops. Equal summary envelopes alone
    cannot join different per-row conditions or independent participants.
    Existing nonempty/count/stride checks still apply at every elimination.
    """
    control, groups = memory.control, []
    facts, reader_loops = control.environment.facts, reader.frame.loops()
    for writer, output, actor, path in alternatives:
        if output.address.valid_rows is not None:
            continue
        span = _span(output, _participant(actor, memory.control.sid))
        loops = writer.frame.loops()
        for loop in reversed(loops):
            if loop in reader_loops or any(
                len(guard) != 4
                or not all(control.invariant_over(value, loop.operation) for value in guard[:2])
                for guard in path
            ):
                break
            summarized = _summarize(span, loop, facts, _participant(actor, control.sid))
            if summarized is None:
                break
            span, loops = summarized, loops[:-1]
        for prior, prior_loops, paths, members in groups:
            if (
                actor is members[0][2]
                and loops == prior_loops
                and (span.width, span.rows, span.pitch) == (prior.width, prior.rows, prior.pitch)
                and equivalent(span.base, prior.base, facts)
            ):
                break
        else:
            paths, members = [], []
            groups.append((span, loops, paths, members))
        paths.append(path)
        members.append((writer, output, actor))
    for span, loops, paths, members in groups:
        if not _complete_branch_paths(paths, facts):
            continue
        for loop in reversed(loops):
            if loop not in reader_loops:
                span = _summarize(span, loop, facts, _participant(members[0][2], control.sid))
                if span is None:
                    break
        if span is not None:
            yield span, tuple(item[0].operation for item in members)


def _exclusive(writer, reader, environment):
    for guard in _guards(writer.frame, environment):
        opposite = (*guard[:-1], 1 - guard[-1])
        if any(
            _same_guard(opposite, other, environment.facts)
            for other in _guards(reader.frame, environment)
        ):
            return True
    return False


def _native_aliases(memory):
    facts = memory.control.environment.facts
    for template in memory.templates:
        if template.frame.statically_empty:
            continue
        for source in template.accesses:
            if not source.mode.reads:
                continue
            for destination in template.accesses:
                if (
                    source is destination
                    or not destination.mode.writes
                    or source.address.root is not destination.address.root
                ):
                    continue
                a, b = source.address, destination.address
                if a.geometry == b.geometry and (
                    a.offset == b.offset or equivalent(a.offset, b.offset, facts)
                ):
                    continue
                delta = normalize(expression("sub", b.offset, a.offset), facts)
                low, high = interval(delta, template.frame.domains(), facts)
                source_span = a.geometry.nbytes + (a.geometry.rows - 1) * a.geometry.row_stride
                destination_span = b.geometry.nbytes + (b.geometry.rows - 1) * b.geometry.row_stride
                _require(
                    nonnegative(expression("sub", low, literal(source_span)), facts)
                    or nonnegative(expression("sub", literal(-destination_span), high), facts),
                    "native read/write aliases need identical views or proved disjoint byte ranges",
                    template.operation,
                    witness=(
                        a.root.owner.sym_name.data,
                        f"operands={source.operand_index},{destination.operand_index}",
                    ),
                )


def _covers(spans, read, frame, facts, work):
    """Unite exact static row intervals only after proving their translations."""
    # Join adjacent row groups with identical column footprints. This keeps
    # their pitched holes, unlike merging their contiguous bounding boxes.
    # The row translation is exact and static; no loop instances expand.
    joined = []
    for anchor in spans:
        if anchor.pitch <= 0 or anchor.width.constant is None:
            continue
        intervals = []
        for candidate in spans:
            if candidate.pitch != anchor.pitch or candidate.width != anchor.width:
                continue
            delta = exact_constant(expression("sub", candidate.base, anchor.base), facts)
            if delta is not None and delta % anchor.pitch == 0:
                first = delta // anchor.pitch
                intervals.append((first, first + candidate.rows))
        merged = []
        for start, stop in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(stop, merged[-1][1])
            else:
                merged.append((start, stop))
        for start, stop in merged:
            candidate = _Span(
                expression("add", anchor.base, literal(start * anchor.pitch)),
                anchor.width,
                stop - start,
                anchor.pitch,
            )
            if candidate not in spans and candidate not in joined:
                joined.append(candidate)
    spans = (*spans, *joined)
    needed_span = expression("add", read.width, literal((read.rows - 1) * read.pitch))

    def contained(base, start, stop):
        delta = expression("sub", read.base, base)
        reduced = normalize(delta, facts)
        for form in dict.fromkeys((reduced, delta)):
            # Cancellation exposes common translations; the original form
            # also retains rem ranges that polynomial expansion can obscure.
            if domain_nonnegative(
                expression("sub", form, start), frame.domains(), facts
            ) and domain_nonnegative(
                expression("sub", stop, expression("add", form, needed_span)),
                frame.domains(),
                facts,
            ):
                return True
        return False

    for candidate in spans:
        work["coverage_checks"] += 1
        if candidate.pitch == read.pitch and read.pitch > 0:
            # A smaller rectangle may select some rows of a strided writer.
            # Its holes are not reads and need no initializer. Only accept
            # an exact, constant translation after common origins cancel;
            # every selected row and column must stay inside the writer.
            shift = exact_constant(expression("sub", read.base, candidate.base), facts)
            width = candidate.width.constant
            needed_width = read.width.constant
            if shift is not None and width is not None and needed_width is not None and shift >= 0:
                row, column = divmod(shift, read.pitch)
                if row + read.rows <= candidate.rows and column + needed_width <= width:
                    return True
            # A runtime row tile may move inside the same pitched rectangle.
            # Prove an exact multiple of the row pitch and both row limits;
            # bounding boxes alone would incorrectly define the row holes.
            if width is not None and needed_width is not None and needed_width <= width:
                delta = normalize(expression("sub", read.base, candidate.base), facts)
                if (
                    read.rows <= candidate.rows
                    and domain_nonnegative(delta, frame.domains(), facts)
                    and domain_nonnegative(
                        expression("sub", literal(width - needed_width), delta),
                        frame.domains(),
                        facts,
                    )
                ):
                    return True
                terms = polynomial(delta, facts)
                if (
                    all(
                        c.denominator == 1 and c.numerator % read.pitch == 0 for c in terms.values()
                    )
                    and domain_nonnegative(delta, frame.domains(), facts)
                    and domain_nonnegative(
                        expression(
                            "sub", literal((candidate.rows - read.rows) * read.pitch), delta
                        ),
                        frame.domains(),
                        facts,
                    )
                ):
                    return True
        if (
            candidate.rows == read.rows
            and candidate.pitch == read.pitch
            and (candidate.base == read.base or equivalent(candidate.base, read.base, facts))
            and nonnegative(expression("sub", candidate.width, read.width), facts)
        ):
            return True
        if candidate.rows == 1 and contained(candidate.base, literal(0), candidate.width):
            return True
    for anchor in spans:
        intervals = []
        for candidate in spans:
            if candidate.width.constant is None:
                continue
            delta = exact_constant(expression("sub", candidate.base, anchor.base), facts)
            if delta is None:
                continue
            intervals.extend(
                (
                    delta + row * candidate.pitch,
                    delta + row * candidate.pitch + candidate.width.constant,
                )
                for row in range(candidate.rows)
            )
            work["row_intervals"] += candidate.rows
        merged = []
        for start, stop in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(stop, merged[-1][1])
            else:
                merged.append((start, stop))
        for start, stop in merged:
            if contained(anchor.base, literal(start), literal(stop)):
                return True
    return False


def _configurations(memory, work):
    from .registers import admit_register_definitions

    root = memory.module.body.block.first_op
    positions = {operation: index for index, operation in enumerate(memory.control.frames)}
    configs = tuple(op for op in positions if isinstance(op, FixConfigOp))
    _require(
        len(configs) <= 1, "native register configuration must have one invariant writer", root
    )
    for operation in configs:
        _require(
            operation.parent is root.body.block,
            "native register configuration must execute at kernel entry",
            operation,
        )
    for template in memory.templates:
        if template.spec.native in {"fix", "fix16"}:
            _require(
                bool(configs) and positions[configs[0]] < positions[template.operation],
                "native requires a preceding invariant register configuration",
                template.operation,
            )
    return configs + admit_register_definitions(memory, work=work)


def _selected_values(structure, memory, templates, facts):
    result = []
    for use in structure.uses:
        reader = templates[use.reader]
        if reader.frame.statically_empty:
            continue
        selected, binding, substitutions = [], use.expected, {}
        while isinstance(binding, LastTile):
            loop = memory.control.loops[binding.loop]
            count = substitute(loop.count, substitutions)
            low, _ = interval(count, reader.frame.domains(), facts)
            _require(
                nonnegative(expression("sub", low, literal(1)), facts),
                "logical Tile last selection is undefined when its loop may be empty",
                use.reader,
                kind=ErrorKind.INVALID if count.constant == 0 else ErrorKind.UNSUPPORTED,
            )
            ordinal = expression("sub", count, literal(1))
            selected.append((loop, ordinal))
            substitutions[loop.ordinal] = ordinal
            binding = binding.value
        result.append((use, reader, tuple(selected), binding, substitutions))
    return result


def _conditional_initialization(access, template, memory):
    """Prove that this instruction's reset initializes every conditional read.

    For an invariant footprint and activation, an executing later ordinal has
    the same instruction at ordinal zero, and each intervening instance writes
    that complete footprint. Nested loop counts must be invariant too: a child
    that is empty at zero cannot establish initialization for a later read.
    This is numerical induction, never an asynchronous retirement statement.
    """
    condition = access.read_condition
    if condition is None:
        return None
    control, facts = memory.control, memory.control.environment.facts
    if equivalent(condition.kill.lhs, condition.kill.rhs, facts):
        # A reset at every ordinal has no old-value read, even for a ring view.
        return condition
    if not control.invariant_over(access.address.offset, condition.loop):
        return None
    inside = False
    for scope in template.frame.ancestors():
        if scope.operation is condition.loop:
            inside = True
            continue
        if not inside:
            continue
        if scope.loop is not None and not control.invariant_over(scope.loop.count, condition.loop):
            return None
        if scope.branch is not None:
            comparison = scope.operation.cond.owner
            if not isinstance(comparison, arith.CmpiOp):
                return None
            if not all(
                control.invariant_over(control.environment.expression(value), condition.loop)
                for value in comparison.operands
            ):
                return None
    return condition if inside else None


def admit_structured_memory(
    structure: NativeStructure, memory: NativeMemoryFacts, bounds: MemoryBoundsFacts
) -> StructuredAdmission:
    """Admit actual memory/value facts; no completion or synchronization inference."""
    _require(
        structure.module is memory.module and bounds.memory is memory,
        "structured admission inputs belong to different compilation invocations",
        memory.module,
    )
    environment, facts = memory.control.environment, memory.control.environment.facts
    work = {
        "native_templates": len(memory.templates),
        "accesses": 0,
        "read_proofs": 0,
        "tile_uses": 0,
        "coverage_checks": 0,
        "row_intervals": 0,
        "expanded_iterations": 0,
        "l1_region_transfers": 0,
        "span_region_transfers": 0,
        "definition_source_nodes": 0,
        "definition_source_members": 0,
        "value_snapshot_queries": 0,
    }
    for root in bounds.allocations:
        initialized = root.owner.initialized.data
        _require(
            initialized in {0, 1} and (not initialized or root.type.bank.data == "GM"),
            "local initialization needs an actual native writer",
            root.owner,
            kind=ErrorKind.INVALID,
        )
    external = {
        access.address.root
        for template in memory.templates
        if not template.frame.statically_empty
        for access in template.accesses
        if access.address.geometry.bank == "GM"
    }
    writes_external = any(
        access.mode.writes and access.address.geometry.bank == "GM"
        for template in memory.templates
        if not template.frame.statically_empty
        for access in template.accesses
    )
    _require(
        not writes_external
        or len(external) <= 1
        or memory.module.body.block.first_op.disjoint_external.data == 1,
        "distinct external roots need the launch's disjoint pointer contract",
        memory.module.body.block.first_op,
    )
    configurations = _configurations(memory, work)
    _native_aliases(memory)
    instances, reads = {}, []
    matrices = _MatrixDefinitions(memory, work)
    spans = _SpanDefinitions(memory, work)
    defined_writes = set()
    templates = {item.operation: item for item in memory.templates}
    positions = {item.operation: index for index, item in enumerate(memory.templates)}
    selected_values = _selected_values(structure, memory, templates, facts)
    partial_values, snapshots = {}, {}
    for selected in selected_values:
        use, reader, _, binding, _ = selected
        writer = templates[binding.operation]
        output = writer.accesses[binding.operand]
        if output.address.valid_rows is not None:
            partial_values.setdefault(binding.operation, []).append(selected)
    write_buckets = {}
    for template in memory.templates:
        if template.frame.statically_empty:
            continue
        for access in template.accesses:
            if access.mode.writes:
                write_buckets.setdefault(access.address.root, []).append((template, access))
    for template in memory.templates:
        if template.frame.statically_empty:
            continue
        for actor in template.participants:
            for access in template.accesses:
                instance = StorageInstance(
                    access.address.root, None if access.address.geometry.bank == "GM" else actor
                )
                instances[template.operation, access.operand_index, actor] = instance
                work["accesses"] += 1
                if not access.mode.reads:
                    continue
                if access.address.root.owner.initialized.data:
                    reads.append(ReadAdmission(template, access, actor, (), external=True))
                    continue
                if (
                    initialization := _conditional_initialization(access, template, memory)
                ) is not None:
                    reads.append(
                        ReadAdmission(template, access, actor, (), initialization=initialization)
                    )
                    continue
                if access.address.matrix is not None:
                    definitions = matrices.before(instance, template)
                    needed = access.address.matrix.bind(_participant(actor, memory.control.sid))
                    writers = definitions.covering(needed, template.frame.domains(), facts)
                    _require(
                        writers is not None,
                        "native reads storage without a proved initialized writer coverage",
                        template.operation,
                        witness=(
                            access.address.root.owner.sym_name.data,
                            f"operand={access.operand_index}",
                            actor.value,
                        ),
                    )
                    reads.append(ReadAdmission(template, access, actor, writers))
                    continue
                candidates, writers = spans.before(instance, template)
                needed = _span(access, _participant(actor, memory.control.sid))
                # The checked source/native mapping keeps every legal read
                # inside the logical allocation. Once real writers define all
                # of it, querying the dynamic read's bounds would duplicate the
                # caller's access contract. Do not include allocation padding,
                # or turn this numerical fact into readiness/ownership/KILL.
                size = literal(ITEMSIZE[access.address.geometry.dtype])
                for extent in access.address.axes.extents:
                    size = expression("mul", size, extent)
                whole = _Span(literal(0), size)
                _require(
                    _covers(candidates, whole, template.frame, facts, work)
                    or needed != whole
                    and _covers(candidates, needed, template.frame, facts, work),
                    "native reads storage without a proved initialized writer coverage",
                    template.operation,
                    witness=(
                        access.address.root.owner.sym_name.data,
                        f"operand={access.operand_index}",
                        actor.value,
                    ),
                )
                reads.append(ReadAdmission(template, access, actor, writers))
            # Reads of all participants precede this instruction's writes below.
        for actor in template.participants:
            for access in template.accesses:
                if access.mode.writes and access.address.matrix is not None:
                    key = instances[template.operation, access.operand_index, actor]
                    matrices.write(key, template, access, actor)
                    continue
                # This frontier supplies complete numerical definitions only.
                # Runtime prefixes supply none under _writer_span's contract;
                # recording one here could suppress a later complete writer
                # with the same static geometry. All real writes remain in
                # memory.templates and write_buckets for ordering/overwrites.
                if access.mode.writes and access.address.valid_rows is None:
                    key = instances[template.operation, access.operand_index, actor]
                    # Numerical definedness is monotone. Repeating the exact
                    # write in the same activation frame adds no initialized
                    # bytes. Sibling loops/branches may be empty independently;
                    # their common parent is not an initialization witness.
                    # Logical-value analysis keeps the precise later versions.
                    signature = (
                        key,
                        actor,
                        template.frame,
                        access.address.offset,
                        access.address.geometry,
                    )
                    if signature not in defined_writes:
                        spans.write(key, template, access, actor)
                        defined_writes.add(signature)
        # Query the existing incremental definition transfers at the real
        # update cut. Querying after all templates would let a later writer
        # initialize preserved bytes retroactively. Each context still moves
        # its prefix cursor only forward, once per newly encountered writer.
        for use, reader, _, binding, _ in partial_values.get(template.operation, ()):
            access = reader.accesses[use.operand]
            for actor in reader.participants:
                participant = actor if actor in template.participants else template.participants[0]
                instance = StorageInstance(
                    access.address.root,
                    None if access.address.geometry.bank == "GM" else participant,
                )
                work["value_snapshot_queries"] += 1
                origins = ()
                external = bool(instance.root.owner.initialized.data)
                if external:
                    covered = True
                elif access.address.matrix is not None:
                    definitions = matrices.before(instance, reader)
                    needed = access.address.matrix.bind(_participant(actor, memory.control.sid))
                    origins = definitions.covering(needed, reader.frame.domains(), facts)
                    covered = origins is not None
                else:
                    candidates, origins = spans.before(instance, reader)
                    needed = _span(access, _participant(actor, memory.control.sid))
                    covered = _covers(candidates, needed, reader.frame, facts, work)
                if covered:
                    snapshots[use, actor] = ValueSnapshot(binding, instance, origins, external)
    values = []
    for use, reader, selected, binding, substitutions in selected_values:
        assert isinstance(binding, TileDefinition)
        writer = templates[binding.operation]
        output, access = writer.accesses[binding.operand], reader.accesses[use.operand]
        _require(
            output.mode.writes and output.address.root is access.address.root,
            "logical Tile writer does not define the consumed actual storage",
            use.reader,
        )
        for actor in reader.participants:
            participant = actor if actor in writer.participants else writer.participants[0]
            _require(
                actor in writer.participants or len(writer.participants) == 1,
                "logical Tile has ambiguous producer participant ownership",
                use.reader,
            )
            snapshot = snapshots.get((use, actor))
            if output.address.valid_rows is not None:
                _require(
                    snapshot is not None,
                    "logical Tile partial update has no initialized entry/update snapshot",
                    use.reader,
                )
            matrix = access.address.matrix is not None
            expected = (_writer_matrix if matrix else _writer_span)(
                writer,
                _value_payload(output) if snapshot is not None else output,
                participant,
                reader,
                memory,
                selections=tuple(substitutions.items()),
                logical_value=True,
            )
            needed = (
                access.address.matrix.bind(_participant(actor, memory.control.sid))
                if matrix
                else _span(access, _participant(actor, memory.control.sid))
            )
            if matrix:
                definitions = RegionDefinitions()
                if expected is not None:
                    definitions.add(expected, (writer.operation,), facts)
                covered = definitions.covering(needed, reader.frame.domains(), facts) is not None
            else:
                covered = expected is not None and _covers(
                    [expected], needed, reader.frame, facts, work
                )
            _require(
                covered,
                "logical Tile selection does not cover the consumed physical view",
                use.reader,
            )
            borrowed_loops = tuple(
                loop
                for loop in reader.frame.loops()
                if loop not in writer.frame.loops()
                and not nonnegative(expression("sub", literal(1), loop.count), facts)
            )
            for other, modified in write_buckets.get(access.address.root, ()):
                if other.operation is writer.operation:
                    continue
                between = (
                    positions[writer.operation] < positions[other.operation] < positions[use.reader]
                )
                previous_iteration = any(loop in other.frame.loops() for loop in borrowed_loops)
                if not between and not previous_iteration:
                    continue
                if access.address.geometry.bank != "GM" and actor not in other.participants:
                    continue
                if not previous_iteration and _exclusive(other, reader, environment):
                    continue
                if previous_iteration:
                    coordinates = (*needed.starts, *needed.stops) if matrix else (needed.base,)
                    _require(
                        not any(
                            loop.ordinal.args[0] in arguments(value)
                            for loop in borrowed_loops
                            for value in coordinates
                        ),
                        "borrowed Tile alias needs independent cross-iteration coordinates",
                        use.reader,
                    )
                candidate = (_writer_matrix if matrix else _writer_span)(
                    other, modified, actor, reader, memory, ignore_guards=True
                )
                _require(
                    candidate is not None,
                    "logical Tile has a possible conditional or cross-iteration overwrite",
                    use.reader,
                    witness=(binding.operation.uid.data, other.operation.uid.data),
                )
                assert candidate is not None
                if matrix:
                    _require(
                        candidate.disjoint(needed, reader.frame.domains(), facts),
                        "logical Tile has a possible overwrite before its actual native reader",
                        use.reader,
                        witness=(binding.operation.uid.data, other.operation.uid.data),
                    )
                    continue
                low, high = interval(
                    normalize(expression("sub", candidate.base, needed.base), facts),
                    reader.frame.domains(),
                    facts,
                )
                candidate_span = expression(
                    "add", candidate.width, literal((candidate.rows - 1) * candidate.pitch)
                )
                needed_span = expression(
                    "add", needed.width, literal((needed.rows - 1) * needed.pitch)
                )
                disjoint = nonnegative(expression("sub", low, needed_span), facts) or nonnegative(
                    expression("mul", literal(-1), expression("add", high, candidate_span)), facts
                )
                _require(
                    disjoint,
                    "logical Tile has a possible overwrite before its actual native reader",
                    use.reader,
                    witness=(binding.operation.uid.data, other.operation.uid.data),
                )
        values.append(
            TileAdmission(
                use,
                binding,
                tuple(selected),
                tuple(
                    snapshots[use, actor]
                    for actor in reader.participants
                    if (use, actor) in snapshots
                ),
            )
        )
    work["read_proofs"], work["tile_uses"] = len(reads), len(values)
    work["storage_instances"] = len(instances)
    work["configuration_writers"] = len(configurations)
    work["defined_write_frontier"] = len(defined_writes)
    work["l1_definition_contexts"] = len(matrices.contexts)
    work["span_definition_contexts"] = len(spans.contexts)
    return StructuredAdmission(
        structure,
        memory,
        bounds,
        MappingProxyType(instances),
        tuple(reads),
        tuple(values),
        configurations,
        MappingProxyType(work),
    )

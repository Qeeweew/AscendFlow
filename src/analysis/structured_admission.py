"""Storage instances, defined reads, and borrowed Tile admission.

All inputs are owned facts from the same recursive lowering invocation. This
pass uses symbolic byte regions and finite native row geometry, never runtime
iteration instances or a completion graph. Initialized storage and a selected
logical writer establish numerical definedness, not asynchronous readiness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from xdsl.dialects import arith
from xdsl.ir import SSAValue

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import FixConfigOp, Load3DConfigOp, TaskOp
from ..indexing import (
    arguments,
    domain_nonnegative,
    equivalent,
    interval,
    nonnegative,
    normalize,
    polynomial,
)
from ..scalar import Expr, expression, literal, substitute
from ..target.ascend910b import Actor
from .native import ConditionalRead
from .structured_bounds import MemoryBoundsFacts
from .structured_control import LogicalLoop
from .structured_host import build_host_expressions
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
    writers: tuple[TaskOp, ...]
    external: bool = False
    initialization: ConditionalRead | None = None


@dataclass(frozen=True, slots=True)
class TileAdmission:
    use: TileUse
    definition: TileDefinition
    selections: tuple[tuple[LogicalLoop, Expr], ...]


@dataclass(frozen=True)
class StructuredAdmission:
    structure: NativeStructure
    memory: NativeMemoryFacts
    bounds: MemoryBoundsFacts
    instances: Mapping[tuple[TaskOp, int, Actor], StorageInstance]
    reads: tuple[ReadAdmission, ...]
    values: tuple[TileAdmission, ...]
    configurations: tuple[FixConfigOp | Load3DConfigOp, ...]
    host_expressions: tuple[Expr, ...]
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
        condition = scope.operation.cond.owner
        if not isinstance(condition, arith.CmpiOp):
            result.append((scope.operation.cond, scope.branch))
            continue
        result.append(
            (
                environment.expression(condition.lhs),
                environment.expression(condition.rhs),
                condition.predicate.value.data,
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


def _summarize(span, loop, facts, bindings):
    count = substitute(loop.count, bindings)
    if not nonnegative(expression("sub", count, literal(1)), facts):
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


def _writer_span(
    writer,
    access,
    actor,
    reader,
    memory,
    *,
    selections=(),
    ignore_guards=False,
    logical_value=False,
):
    environment, facts = memory.control.environment, memory.control.environment.facts
    bindings = _participant(actor, memory.control.sid) | dict(selections)
    writer_loops, reader_loops = writer.frame.loops(), reader.frame.loops()
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
            return None
        if logical_value and (
            first.ordinal not in bindings
            or not equivalent(bindings[first.ordinal], literal(0), facts)
        ):
            return None
        bindings[first.ordinal] = literal(0)
    result = _span(access, bindings)
    for loop in reversed(writer_loops):
        if loop not in reader_loops:
            count = substitute(loop.count, bindings)
            if not nonnegative(expression("sub", count, literal(1)), facts):
                return None
            if loop.ordinal not in bindings:
                result = _summarize(result, loop, facts, bindings)
                if result is None:
                    return None
    return result


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
    needed_span = expression("add", read.width, literal((read.rows - 1) * read.pitch))

    def contained(base, start, stop):
        delta = expression("sub", read.base, base)
        reduced = normalize(delta, facts)
        for form in dict.fromkeys((reduced, delta)):
            # Cancellation exposes common translations; the original form
            # also retains rem ranges that polynomial expansion can obscure.
            try:
                if domain_nonnegative(
                    expression("sub", form, start), frame.domains(), facts
                ) and domain_nonnegative(
                    expression("sub", stop, expression("add", form, needed_span)),
                    frame.domains(),
                    facts,
                ):
                    return True
            except CompileError as error:
                # A candidate translation is a mathematical comparison, not
                # a source address. A normalized division may lack a sign
                # outside this candidate's region; it supplies no coverage.
                # Required original-address definedness was checked by bounds.
                if error.message != "unsigned index division has an unproven nonnegative dividend":
                    raise
        return False

    for candidate in spans:
        work["coverage_checks"] += 1
        if candidate.pitch == read.pitch and read.pitch > 0:
            # A smaller rectangle may select some rows of a strided writer.
            # Its holes are not reads and need no initializer. Only accept
            # an exact, constant translation after common origins cancel;
            # every selected row and column must stay inside the writer.
            shift = normalize(expression("sub", read.base, candidate.base), facts).constant
            width = candidate.width.constant
            needed_width = read.width.constant
            if shift is not None and width is not None and needed_width is not None and shift >= 0:
                row, column = divmod(shift, read.pitch)
                if row + read.rows <= candidate.rows and column + needed_width <= width:
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
            delta = normalize(expression("sub", candidate.base, anchor.base), facts).constant
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


def _configurations(memory):
    root = memory.module.body.block.first_op
    positions = {operation: index for index, operation in enumerate(memory.control.frames)}
    configs = tuple(op for op in positions if isinstance(op, (FixConfigOp, Load3DConfigOp)))
    for kind in (FixConfigOp, Load3DConfigOp):
        matching = tuple(op for op in configs if isinstance(op, kind))
        _require(
            len(matching) <= 1, "native register configuration must have one invariant writer", root
        )
        for operation in matching:
            _require(
                operation.parent is root.body.block,
                "native register configuration must execute at kernel entry",
                operation,
            )
            if isinstance(operation, Load3DConfigOp):
                _require(
                    operation.fp32 is None or operation.fp32.data == 1,
                    "only IEEE FP32 Cube mode is supported",
                    operation,
                )
        for template in memory.templates:
            float_mad = (
                kind is Load3DConfigOp
                and template.contract.native in {"mad", "mad_first", "mad_cycle"}
                and template.accesses[0].address.geometry.dtype == "f32"
            )
            if float_mad:
                _require(
                    bool(matching)
                    and matching[0].fp32 is not None
                    and positions[matching[0]] < positions[template.operation],
                    "FP32 MAD requires a preceding IEEE mode configuration",
                    template.operation,
                )
            if template.contract.native not in (
                {"fix", "fix16"} if kind is FixConfigOp else {"load3d_a", "load3d_b_nn"}
            ):
                continue
            _require(
                bool(matching) and positions[matching[0]] < positions[template.operation],
                "native requires a preceding invariant register configuration",
                template.operation,
            )
            if kind is Load3DConfigOp:
                width = (
                    matching[0].a_width
                    if template.contract.native == "load3d_a"
                    else matching[0].b_width
                )
                source = template.accesses[0].address.geometry
                actual_width = (
                    source.row_stride // 32
                    if not source.contiguous
                    else template.contract.params[0]
                )
                _require(
                    width.data == actual_width,
                    "Load3D invariant FMATRIX width does not match the actual native",
                    template.operation,
                )
    return configs


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
    configurations = _configurations(memory)
    _native_aliases(memory)
    instances, reads, prior = {}, [], {}
    defined_writes = set()
    templates = {item.operation: item for item in memory.templates}
    positions = {item.operation: index for index, item in enumerate(memory.templates)}
    selected_values = _selected_values(structure, memory, templates, facts)
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
                candidates = []
                writers = []
                for writer, output, participant in prior.get(instance, ()):
                    span = _writer_span(writer, output, participant, template, memory)
                    if span is not None:
                        candidates.append(span)
                        writers.append(writer.operation)
                needed = _span(access, _participant(actor, memory.control.sid))
                _require(
                    _covers(candidates, needed, template.frame, facts, work),
                    "native reads storage without a proved initialized writer coverage",
                    template.operation,
                    witness=(
                        access.address.root.owner.sym_name.data,
                        f"operand={access.operand_index}",
                        actor.value,
                    ),
                )
                reads.append(ReadAdmission(template, access, actor, tuple(dict.fromkeys(writers))))
            # Reads of all participants precede this instruction's writes below.
        for actor in template.participants:
            for access in template.accesses:
                if access.mode.writes:
                    key = instances[template.operation, access.operand_index, actor]
                    # Numerical definedness is monotone. Repeating the exact
                    # write in the same control block adds no initialized
                    # bytes. Keep its first defining witness, independently
                    # of the logical-value analysis's precise later versions.
                    signature = (
                        key,
                        actor,
                        template.frame.parent,
                        access.address.offset,
                        access.address.geometry,
                    )
                    if signature not in defined_writes:
                        prior.setdefault(key, []).append((template, access, actor))
                        defined_writes.add(signature)
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
            expected = _writer_span(
                writer,
                output,
                participant,
                reader,
                memory,
                selections=tuple(substitutions.items()),
                logical_value=True,
            )
            needed = _span(access, _participant(actor, memory.control.sid))
            _require(
                expected is not None and _covers([expected], needed, reader.frame, facts, work),
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
                    _require(
                        not any(
                            loop.ordinal.args[0] in arguments(needed.base)
                            for loop in borrowed_loops
                        ),
                        "borrowed Tile alias needs independent cross-iteration coordinates",
                        use.reader,
                    )
                candidate = _writer_span(other, modified, actor, reader, memory, ignore_guards=True)
                _require(
                    candidate is not None,
                    "logical Tile has a possible conditional or cross-iteration overwrite",
                    use.reader,
                    witness=(binding.operation.uid.data, other.operation.uid.data),
                )
                assert candidate is not None
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
        values.append(TileAdmission(use, binding, tuple(selected)))
    work["read_proofs"], work["tile_uses"] = len(reads), len(values)
    work["storage_instances"] = len(instances)
    work["configuration_writers"] = len(configurations)
    host_expressions = build_host_expressions(memory, bounds)
    work["host_expressions"] = len(host_expressions)
    work["defined_write_frontier"] = len(defined_writes)
    return StructuredAdmission(
        structure,
        memory,
        bounds,
        MappingProxyType(instances),
        tuple(reads),
        tuple(values),
        configurations,
        host_expressions,
        MappingProxyType(work),
    )

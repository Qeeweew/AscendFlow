"""Canonical logical clocks and participant scopes of actual physical IR.

All stages of one loop use one logical ordinal. Their physical dispatch steps
remain distinct and belong to independent actor controllers. All Vector domains
of a mixed invocation use the same runtime sid. Names below are fresh internal
symbols mapped to actual operations, never source spelling or alias evidence.

This traversal establishes scalar bindings and explicit range premises only.
It does not prove address bounds, ownership, native effects, or completion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects import scf
from xdsl.dialects.builtin import IndexType, ModuleOp, i1
from xdsl.ir import Operation, SSAValue

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import CoreIdOp, KernelOp, MixedOp, RuntimeSizeOp, TaskOp
from ..dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp
from ..indexing import (
    arguments,
    branch_comparison,
    domain_nonnegative,
    equivalent,
    exact_constant,
    interval,
    merge_scalar_bounds,
    nonnegative,
    pure_scalar_selection,
)
from ..memo import memoized_method
from ..scalar import Expr, Fact, expression, from_ssa, literal, positive, substitute
from ..symbols import Symbol, SymbolKind, is_coordinate
from ..target.ascend910b import Actor
from .structured_addresses import ScalarEnvironment
from .structured_schedule import Comparison

Range = tuple[Expr, Expr, Expr]


def _uniform_parts(value, memo):
    """Exact uniform+bias split, retaining integer division expressions."""
    if value in memo:
        return memo[value]
    zero = literal(0)
    if value.kind in {"constant", "argument"}:
        varying = value.kind == "argument" and is_coordinate(value.args[0])
        result = (zero, value) if varying else (value, zero)
    else:
        (au, av), (bu, bv) = (_uniform_parts(child, memo) for child in value.args)
        if av == bv == zero:
            result = value, zero
        elif value.kind == "add":
            result = expression("add", au, bu), expression("add", av, bv)
        elif value.kind == "mul" and av == zero:
            result = expression("mul", au, bu), expression("mul", au, bv)
        elif value.kind == "mul" and bv == zero:
            result = expression("mul", au, bu), expression("mul", av, bu)
        else:
            # Dividing a sum termwise is not an integer identity. Retain it
            # whole in the bias; this may be less precise, but is exact.
            result = zero, value
    memo[value] = result
    return result


def _division_iteration_range(lower, upper, induction, domains, facts):
    """An active nonnegative i < ceildiv(max(0,x),d) implies d*i < x.

    For floordiv the stronger d*i <= x-d holds. Split x=uniform-bias to retain
    bias+d*i as one joint coordinate, instead of independently bounding the
    ancestor bias and losing their correlation. This is an integer loop rule,
    independent of task names, tensor layouts, group counts, or operators.
    """
    if upper.kind == "min":
        # Active i < min(a,b) entails both upper bounds. Retain the individual
        # division relations as well as their joint min; no bound is assumed
        # on the zero-iteration path or outside this invocation.
        return tuple(
            item
            for bound in upper.args
            for item in (
                (induction, lower, expression("sub", bound, literal(1))),
                *_division_iteration_range(lower, bound, induction, domains, facts),
            )
        )
    if upper.kind not in {"ceildiv", "floordiv"}:
        return ()
    numerator, divisor = upper.args
    if not positive(divisor, facts):
        return ()
    low, _ = interval(lower, domains, facts)
    if not nonnegative(low, facts):
        return ()
    if numerator.kind == "max" and literal(0) in numerator.args:
        numerator = next(value for value in numerator.args if value != literal(0))
    uniform, varying = _uniform_parts(numerator, {})
    bias = expression("mul", literal(-1), varying)
    bias_lower, _ = interval(bias, domains, facts)
    coordinate = expression("add", bias, expression("mul", divisor, induction))
    coordinate_lower = expression("add", bias_lower, expression("mul", divisor, low))
    if lower.kind == "ceildiv" and equivalent(lower.args[1], divisor, facts):
        start = lower.args[0]
        if start.kind == "max" and literal(0) in start.args:
            start = next(value for value in start.args if value != literal(0))
        start_uniform, start_varying = _uniform_parts(start, {})
        if equivalent(start_varying, varying, facts):
            # i >= ceildiv(max(0, begin-bias), d) gives bias+d*i >= begin.
            # Retain the joint lower bound instead of independently losing
            # the correlation between the group and its first iteration.
            coordinate_lower = start_uniform
    gap = divisor if upper.kind == "floordiv" else literal(1)
    return ((coordinate, coordinate_lower, expression("sub", uniform, gap)),)


@dataclass(frozen=True, slots=True)
class LogicalLoop:
    operation: scf.ForOp | PipelineLoopOp
    ordinal: Expr
    induction: Expr
    lower: Expr
    upper: Expr
    step: int
    count: Expr
    # Ranges of the enclosing invocation, before this loop is entered.
    # Its own ordinal domain cannot establish that the loop is nonempty.
    entry_domains: tuple[Range, ...]


@dataclass(frozen=True, slots=True, eq=False)
class ControlFrame:
    parent: ControlFrame | None
    operation: Operation
    participants: tuple[Actor, ...]
    ranges: tuple[Range, ...] = ()
    loop: LogicalLoop | None = None
    branch: int | None = None
    comparisons: tuple[Comparison, ...] = ()

    def ancestors(self) -> tuple[ControlFrame, ...]:
        """The one retained lexical scope chain, including this frame."""
        frames, current = [], self
        while current is not None:
            frames.append(current)
            current = current.parent
        return tuple(reversed(frames))

    def loops(self) -> tuple[LogicalLoop, ...]:
        return tuple(frame.loop for frame in self.ancestors() if frame.loop is not None)

    def domains(self) -> tuple[Range, ...]:
        """Only lexical depth, with no replay of executed iterations."""
        return tuple(domain for frame in self.ancestors() for domain in frame.ranges)

    def predicates(self):
        """Relations of already evaluated branches in this lexical scope."""
        return tuple(test for frame in self.ancestors() for test in frame.comparisons)

    @property
    def statically_empty(self):
        current = self
        while current is not None:
            if current.loop is not None and current.loop.count.constant == 0:
                return True
            current = current.parent
        return False


class _ScalarPrefix:
    """One lexical SSA prefix index and its incrementally decoded atoms."""

    def __init__(self, values, positions):
        self.values, self.positions = tuple(values), positions
        self.symbols = [frozenset()]

    def arguments(self, count, environment):
        while len(self.symbols) <= count:
            names = arguments(environment.expression(self.values[len(self.symbols) - 1]))
            previous = self.symbols[-1]
            self.symbols.append(previous if names <= previous else previous | names)
        return self.symbols[count]


@dataclass(frozen=True)
class ControlFacts:
    module: ModuleOp
    environment: ScalarEnvironment
    loops: Mapping[Operation, LogicalLoop]
    frames: Mapping[Operation, ControlFrame]
    result_frames: Mapping[SSAValue, ControlFrame]
    branches: Mapping[tuple[Operation, int], tuple[Comparison, ...]]
    group: Expr
    sid: Expr
    operations_visited: int

    def participant_comparisons(self, tests, actor):
        """Bind SID to the actual participant, retaining every unknown test.

        This projects activation only; it supplies no completion or ownership.
        """
        bindings = {self.sid: literal(1 if actor is Actor.AIV1 else 0)}
        for test in tests:
            lhs, rhs = (substitute(value, bindings) for value in (test.lhs, test.rhs))
            a, b = (exact_constant(value, self.environment.facts) for value in (lhs, rhs))
            if a is None or b is None:
                yield test
            else:
                bound = Comparison(literal(a), test.predicate, literal(b))
                if not bound.evaluate({}):
                    yield bound

    @memoized_method
    def _scalar_prefix(self, block):
        def scalar(value):
            return isinstance(value.type, IndexType) or value.type == i1

        values = []
        owner = block.parent_op()
        if owner is not None and owner.parent is not None:
            values.extend(self.dominating_scalars(owner.parent, owner))
        values.extend(v for v in block.args if scalar(v))
        positions = {}
        for operation in block.ops:
            if operation in self.frames:
                positions[operation] = len(values)
                values.extend(v for v in operation.results if scalar(v))
        positions[None] = len(values)
        return _ScalarPrefix(values, positions)

    def _scalar_position(self, block, before):
        prefix = self._scalar_prefix(block)
        # Inserted notifications/scalars have no source definitions in this
        # snapshot. Their position is the next retained original operation.
        while before not in prefix.positions:
            before = before.next_op
        return prefix, prefix.positions[before]

    @memoized_method
    def dominating_scalars(self, block, before):
        """Actual scalar definitions at this owned lexical insertion point.

        Each block's prefix is indexed once, including its true ancestor
        prefix. Newly inserted operations do not grant source availability.
        No range query or cross-invocation equality claim is involved.
        """
        prefix, count = self._scalar_position(block, before)
        return prefix.values[:count]

    @memoized_method
    def dominating_arguments(self, block, before):
        """Analysis atoms of the same actual dominating SSA definitions."""
        prefix, count = self._scalar_position(block, before)
        return prefix.arguments(count, self.environment)

    @memoized_method
    def invariant_over(self, value: Expr, operation: Operation) -> bool:
        """Whether an expression is fixed throughout this actual invocation.

        Uniform inputs/group/sid and enclosing loop coordinates are stable.
        Unknown scalar carry/results have no invariance contract merely
        because their expression lacks this loop's ordinal symbol.
        """
        names, metadata = self._invariant_input(operation)
        return all(
            name in names
            or isinstance(name, Symbol)
            and name.kind in {SymbolKind.PREDICATE, SymbolKind.SELECTION}
            or isinstance(name, Symbol)
            and name.kind is SymbolKind.METADATA
            and name._identity in metadata
            for name in value.variables
        )

    @memoized_method
    def _invariant_input(self, operation):
        names = set(self.environment.facts)
        frame = self.frames.get(operation)
        if frame is not None and frame.parent is not None:
            names.update(loop.ordinal.args[0] for loop in frame.parent.loops())
            names.update(
                coordinate.args[0]
                for coordinate, _, _ in frame.parent.domains()
                if coordinate.kind == "argument"
            )
        if frame is not None:
            for ancestor in frame.ancestors():
                operation = ancestor.operation
                if isinstance(operation, TaskOp) and operation.native.data == "index_load":
                    symbol = self.environment.expression(operation.value).args[0]
                    if all(arguments(value) <= names for value in symbol.captures):
                        names.add(symbol)
        metadata = {
            name._identity
            for name in names
            if isinstance(name, Symbol)
            and name.kind is SymbolKind.METADATA
            and name.origin is not None
        }
        # Actor/ancestor selection specializes the capture coordinates, not
        # the load's definition. Expr.variables already includes every capture
        # dependency recursively: all must still be fixed in this invocation.
        # A load defined inside the queried loop has no visible origin here,
        # even if someone substitutes a constant for its captured ordinal.
        return frozenset(names), frozenset(metadata)


def build_control_facts(module: ModuleOp) -> ControlFacts:
    """Bind one owned, structurally verified module without cloning its IR.

    Unknown SCF scalar carries/results retain distinct symbols with no assumed
    range. They remain executable without an explicit range contract. Optional
    range enrichment never changes the accepted source language.
    """
    root = module.body.block.first_op
    require(
        len(module.body.block.ops) == 1 and isinstance(root, (KernelOp, MixedOp)),
        "structured control requires one physical kernel or mixed tree",
    )
    groups = 1 if root.blocks is None else root.blocks.data
    require(1 <= groups <= 1024, "invalid physical group count")
    mixed = isinstance(root, MixedOp)
    require(mixed or root.core.data in {"aic", "aiv"}, "unsupported physical actor family")
    group = Expr("argument", (Symbol("$group", SymbolKind.GROUP, root),))
    sid = Expr("argument", (Symbol("$sid", SymbolKind.PARTICIPANT, root),))
    facts = {
        group.args[0]: Fact(maximum=groups - 1),
        sid.args[0]: Fact(maximum=1 if mixed else 0),
    }
    bindings, loops, frames, branches, result_frames = {}, {}, {}, {}, {}
    runtime_names = set()
    visited = unknown = 0

    def scalar(value):
        return from_ssa(value, memo=bindings)

    def unknown_scalar(value, frame, kind=SymbolKind.CARRY):
        nonlocal unknown
        if isinstance(value.type, IndexType) or value.type == i1:
            if pure_scalar_selection(value.owner):
                bindings[value] = from_ssa(value, memo=bindings)
                return
            captures = (group, sid, *(loop.ordinal for loop in frame.loops()))
            bindings[value] = Expr(
                "argument", (Symbol(f"$scalar{unknown}", kind, value, captures),)
            )
            unknown += 1

    def visit(operation, frame):
        nonlocal visited
        visited += 1
        if isinstance(operation, RuntimeSizeOp):
            name = operation.sym_name.data
            require(name not in runtime_names, "duplicate runtime parameter identity")
            runtime_names.add(name)
            symbol = Symbol(name, SymbolKind.PARAMETER, operation.value)
            facts[symbol] = Fact(operation.minimum.data, operation.multiple.data)
            bindings[operation.value] = Expr("argument", (symbol,))
        elif isinstance(operation, CoreIdOp):
            bindings[operation.value] = group
        elif isinstance(operation, TaskOp) and operation.native.data == "index_load":
            count = operation.attributes.get("address_count")
            require(
                count is not None
                and count.data == 1
                and len(operation.refs) == 3
                and operation.value is not None
                and isinstance(operation.value.type, IndexType),
                "index metadata load lost its signed value domain",
            )
            # A local read snapshots its actual storage generation. It is not
            # immutable input metadata and cannot acquire that invariance.
            bank = operation.refs[0].type.bank.data
            unknown_scalar(
                operation.value, frame, SymbolKind.METADATA if bank == "GM" else SymbolKind.CARRY
            )
            coordinate = bindings[operation.value]
            lower, upper = scalar(operation.refs[1]), scalar(operation.refs[2])
            frame = ControlFrame(
                frame,
                operation,
                frame.participants,
                (
                    (coordinate, lower, upper),
                    (
                        expression("sub", coordinate, lower),
                        literal(0),
                        expression("sub", upper, lower),
                    ),
                ),
            )
        elif isinstance(operation, ActorDomainOp):
            participants = (
                (Actor.AIC,) if operation.kind.data == "cube" else (Actor.AIV0, Actor.AIV1)
            )
            bindings[operation.body.block.args[0]] = (
                literal(0) if operation.kind.data == "cube" else sid
            )
            frame = ControlFrame(frame, operation, participants)
        elif isinstance(operation, (PipelineLoopOp, scf.ForOp)):
            if isinstance(operation, PipelineLoopOp):
                lower, upper, step = literal(0), scalar(operation.count), 1
            else:
                lower, upper = scalar(operation.lb), scalar(operation.ub)
                step = scalar(operation.step).constant
            require(
                step is not None and step > 0,
                "structured loop needs a positive constant step",
                kind=ErrorKind.UNSUPPORTED,
                phase="structured control",
                location=str(operation.location),
            )
            ordinal = Expr(
                "argument", (Symbol(f"$loop{len(loops)}", SymbolKind.ORDINAL, operation),)
            )
            difference = expression("sub", upper, lower)
            # This count is a derived mathematical coordinate, not a rewrite
            # of the original loop header. Preserve that header/SSA for its
            # separate definedness checks; do not manufacture max/div-by-one
            # wrappers around every already nonnegative unit-step extent.
            extent = (
                difference
                if domain_nonnegative(difference, frame.domains(), facts)
                else expression("max", literal(0), difference)
            )
            count = extent if step == 1 else expression("ceildiv", extent, literal(step))
            induction = expression("add", lower, expression("mul", ordinal, literal(step)))
            loop = LogicalLoop(
                operation, ordinal, induction, lower, upper, step, count, frame.domains()
            )
            loops[operation] = loop
            try:
                joint = _division_iteration_range(lower, upper, induction, frame.domains(), facts)
            except CompileError as error:
                if error.kind is not ErrorKind.ANALYSIS_LIMIT:
                    raise
                joint = ()
            frame = ControlFrame(
                frame,
                operation,
                frame.participants,
                (
                    *joint,
                    (induction, lower, expression("sub", upper, literal(1))),
                    (ordinal, literal(0), expression("sub", count, literal(1))),
                ),
                loop,
            )
            if isinstance(operation, scf.ForOp):
                bindings[operation.body.block.args[0]] = induction
                for value in operation.body.block.args[1:]:
                    unknown_scalar(value, frame)
                for value in operation.results:
                    unknown_scalar(value, frame.parent)
        elif isinstance(operation, PipelineStageOp):
            loop = loops[operation.parent_op()]
            bindings[operation.body.block.args[0]] = loop.ordinal
            frame = ControlFrame(frame, operation, frame.participants)
        elif isinstance(operation, scf.IfOp):
            for value in operation.results:
                unknown_scalar(value, frame)
        frames[operation] = frame
        for branch, region in enumerate(operation.regions):
            child_frame = (
                ControlFrame(
                    frame,
                    operation,
                    frame.participants,
                    branch=branch,
                    comparisons=branch_comparison(operation.cond, scalar, branch),
                )
                if isinstance(operation, scf.IfOp)
                else frame
            )
            if isinstance(operation, scf.IfOp):
                branches[operation, branch] = child_frame.comparisons
            for block in region.blocks:
                local = child_frame
                for child in block.ops:
                    visited_frame = visit(child, local)
                    if (isinstance(child, TaskOp) and child.native.data == "index_load") or (
                        isinstance(child, scf.IfOp) and child.results
                    ):
                        local = visited_frame
        if isinstance(operation, scf.IfOp):
            ranges = []
            for i, value in enumerate(operation.results):
                if not isinstance(value.type, IndexType):
                    continue
                exits = [region.block.last_op for region in operation.regions]
                bounds = merge_scalar_bounds(
                    [scalar(exit_.arguments[i]) for exit_ in exits],
                    [frames[exit_].domains() for exit_ in exits],
                    frame.domains(),
                    facts,
                )
                if bounds is not None:
                    ranges.append((bindings[value], *bounds))
            if ranges:
                exit_frame = ControlFrame(frame, operation, frame.participants, tuple(ranges))
                result_frames.update((value, exit_frame) for value in operation.results)
                return exit_frame
        return frame

    participants = () if mixed else (Actor.AIC,) if root.core.data == "aic" else (Actor.AIV0,)
    root_frame = ControlFrame(
        None,
        root,
        participants,
        ((group, literal(0), literal(groups - 1)), (sid, literal(0), literal(1 if mixed else 0))),
    )
    visit(root, root_frame)
    return ControlFacts(
        module,
        ScalarEnvironment(bindings, facts),
        MappingProxyType(loops),
        MappingProxyType(frames),
        MappingProxyType(result_frames),
        MappingProxyType(branches),
        group,
        sid,
        visited,
    )

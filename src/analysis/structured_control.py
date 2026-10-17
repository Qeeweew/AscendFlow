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
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Operation

from ..diagnostics import ErrorKind, require
from ..dialects.physical import CoreIdOp, KernelOp, MixedOp, RuntimeSizeOp
from ..dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp
from ..indexing import arguments, equivalent, interval, nonnegative
from ..scalar import Expr, Fact, expression, from_ssa, literal, positive
from ..target.ascend910b import Actor
from .structured_addresses import ScalarEnvironment

Range = tuple[Expr, Expr, Expr]


def _uniform_parts(value, memo):
    """Exact uniform+bias split, retaining integer division expressions."""
    if value in memo:
        return memo[value]
    zero = literal(0)
    if value.kind in {"constant", "argument"}:
        varying = value.kind == "argument" and value.args[0].startswith("$")
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


@dataclass(frozen=True, slots=True, eq=False)
class ControlFrame:
    parent: ControlFrame | None
    operation: Operation
    participants: tuple[Actor, ...]
    ranges: tuple[Range, ...] = ()
    loop: LogicalLoop | None = None
    branch: int | None = None

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

    @property
    def statically_empty(self):
        current = self
        while current is not None:
            if current.loop is not None and current.loop.count.constant == 0:
                return True
            current = current.parent
        return False


@dataclass(frozen=True)
class ControlFacts:
    module: ModuleOp
    environment: ScalarEnvironment
    loops: Mapping[Operation, LogicalLoop]
    frames: Mapping[Operation, ControlFrame]
    group: Expr
    sid: Expr
    operations_visited: int

    def invariant_over(self, value: Expr, operation: Operation) -> bool:
        """Whether an expression is fixed throughout this actual invocation.

        Uniform inputs/group/sid and enclosing loop coordinates are stable.
        Unknown scalar carry/results have no invariance contract merely
        because their expression lacks this loop's ordinal symbol.
        """
        names = set(self.environment.facts)
        frame = self.frames.get(operation)
        if frame is not None and frame.parent is not None:
            names.update(loop.ordinal.args[0] for loop in frame.parent.loops())
        return arguments(value) <= names


def build_control_facts(module: ModuleOp) -> ControlFacts:
    """Bind one owned, structurally verified module without cloning its IR.

    Unknown SCF scalar carries/results retain distinct symbols with no assumed
    range. Their use in an address must obtain a range contract later; an empty
    or unrelated scalar loop is not rejected merely for containing such carry.
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
    group, sid = Expr("argument", ("$group",)), Expr("argument", ("$sid",))
    facts = {"$group": Fact(maximum=groups - 1), "$sid": Fact(maximum=1 if mixed else 0)}
    bindings, loops, frames = {}, {}, {}
    runtime_names = set()
    visited = unknown = 0

    def scalar(value):
        return from_ssa(value, memo=bindings, facts=facts)

    def unknown_scalar(value):
        nonlocal unknown
        if isinstance(value.type, IndexType):
            bindings[value] = Expr("argument", (f"$scalar{unknown}",))
            unknown += 1

    def visit(operation, frame):
        nonlocal visited
        visited += 1
        if isinstance(operation, RuntimeSizeOp):
            name = operation.sym_name.data
            require(name not in runtime_names, "duplicate runtime parameter identity")
            runtime_names.add(name)
            facts[name] = Fact(operation.minimum.data, operation.multiple.data)
            bindings[operation.value] = Expr("argument", (name,))
        elif isinstance(operation, CoreIdOp):
            bindings[operation.value] = group
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
            ordinal = Expr("argument", (f"$loop{len(loops)}",))
            difference = expression("sub", upper, lower)
            # This count is a derived mathematical coordinate, not a rewrite
            # of the original loop header. Preserve that header/SSA for its
            # separate definedness checks; do not manufacture max/div-by-one
            # wrappers around every already nonnegative unit-step extent.
            extent = (
                difference
                if nonnegative(difference, facts)
                else expression("max", literal(0), difference)
            )
            count = extent if step == 1 else expression("ceildiv", extent, literal(step))
            induction = expression("add", lower, expression("mul", ordinal, literal(step)))
            loop = LogicalLoop(operation, ordinal, induction, lower, upper, step, count)
            loops[operation] = loop
            joint = _division_iteration_range(lower, upper, induction, frame.domains(), facts)
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
                for value in (*operation.body.block.args[1:], *operation.results):
                    unknown_scalar(value)
        elif isinstance(operation, PipelineStageOp):
            loop = loops[operation.parent_op()]
            bindings[operation.body.block.args[0]] = loop.ordinal
            frame = ControlFrame(frame, operation, frame.participants)
        elif isinstance(operation, scf.IfOp):
            for value in operation.results:
                unknown_scalar(value)
        frames[operation] = frame
        for branch, region in enumerate(operation.regions):
            child_frame = (
                ControlFrame(frame, operation, frame.participants, branch=branch)
                if isinstance(operation, scf.IfOp)
                else frame
            )
            for block in region.blocks:
                for child in block.ops:
                    visit(child, child_frame)

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
        group,
        sid,
        visited,
    )

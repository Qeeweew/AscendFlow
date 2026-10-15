"""Source loop scopes and effect order for hierarchical physical lowering.

These immutable handles refer to actual source IR. A child is collected once,
regardless of its trip count. Execution clocks come from lexical regions, even
when every address is constant. Collection supplies neither a schedule nor a
completion proof; the lowerer must admit the bounds, guards and actor protocols.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from xdsl.dialects import arith, scf
from xdsl.ir import Block, SSAValue

from .model import require
from .scalar import ARITHMETIC
from .source_ir import (
    AllocationOp,
    CopyOp,
    DomainOp,
    ElementOp,
    MatmulOp,
    PipelineOp,
    PredicateOp,
    ProgramOp,
    ReduceOp,
    ScalarReadOp,
    ScaleOp,
    SizeOp,
    StageOp,
    ViewOp,
    YieldOp,
)

SourceEffect = CopyOp | ElementOp | MatmulOp | ReduceOp | ScalarReadOp | ScaleOp
Loop = PipelineOp | scf.ForOp


@dataclass(frozen=True)
class ClockFrame:
    """One actual loop clock and the Stage selected on that clock alone."""

    op: Loop
    stage: StageOp | None = None

    @property
    def coordinate(self) -> SSAValue:
        return self.op.body.block.args[0]

    @property
    def bound(self) -> SSAValue | int:
        if isinstance(self.op, scf.ForOp):
            return self.op.ub
        if self.op.limit is not None:
            return self.op.limit
        assert self.op.count is not None
        return self.op.count.data

    @property
    def offset(self) -> int:
        return 0 if self.stage is None else self.stage.offset.data


@dataclass(frozen=True)
class Guard:
    """An actual source if and the branch containing this effect or child."""

    op: scf.IfOp
    taken: bool

    @property
    def condition(self) -> SSAValue:
        return self.op.cond


@dataclass(frozen=True)
class Context:
    """Captured lexical activation, independent of address dependencies.

    ``actor`` is the source family (cube/vector). ``domain`` retains its actual
    owner SSA for physical AIC/AIV actor selection. ``path`` records structural
    source order, including if branch indices; it is not a physical issue key.
    """

    clocks: tuple[ClockFrame, ...]
    guards: tuple[Guard, ...]
    actor: str | None
    domain: DomainOp | None
    path: tuple[int, ...]

    @property
    def clock(self) -> ClockFrame | None:
        return self.clocks[-1] if self.clocks else None

    @property
    def owner(self) -> SSAValue | None:
        return None if self.domain is None else self.domain.body.block.args[0]


@dataclass(frozen=True)
class Work:
    op: SourceEffect
    context: Context

    @property
    def actor(self) -> str:
        assert self.context.actor is not None
        return self.context.actor

    @property
    def clock(self) -> ClockFrame | None:
        return self.context.clock

    @property
    def parent_clocks(self) -> tuple[ClockFrame, ...]:
        return self.context.clocks[:-1]


@dataclass(frozen=True)
class Scope:
    """A program/loop with ordinary work and child calls in source order.

    The entry context captures only ancestor clocks. Work inside this scope has
    its own clock appended. Stages and domains remain actual ops in those
    contexts, rather than becoming independent loops or copied native work.
    """

    op: ProgramOp | Loop
    context: Context
    items: tuple[Scope | Work, ...]

    @property
    def clock(self) -> ClockFrame | None:
        return None if isinstance(self.op, ProgramOp) else ClockFrame(self.op)

    @property
    def parent_clocks(self) -> tuple[ClockFrame, ...]:
        return self.context.clocks

    @property
    def actors(self) -> frozenset[str]:
        return frozenset(work.actor for work in self.walk_work())

    def items_for(self, actor: str) -> tuple[Scope | Work, ...]:
        """Retain one actor's ordinary work and participating child calls."""
        require(actor in {"cube", "vector"}, "source actor must be cube or vector")
        return tuple(
            item
            for item in self.items
            if (actor in item.actors if isinstance(item, Scope) else item.actor == actor)
        )

    def walk_work(self, actor: str | None = None) -> Iterator[Work]:
        for item in self.items:
            if isinstance(item, Scope):
                yield from item.walk_work(actor)
            elif actor is None or item.actor == actor:
                yield item

    def walk_scopes(self) -> Iterator[Scope]:
        yield self
        for item in self.items:
            if isinstance(item, Scope):
                yield from item.walk_scopes()


def collect_source_scopes(program: ProgramOp) -> Scope:
    """Collect verified source IR without expanding clocks or proving effects."""
    effect_types = (CopyOp, ElementOp, MatmulOp, ReduceOp, ScalarReadOp, ScaleOp)
    descriptors = (AllocationOp, SizeOp, ViewOp, PredicateOp, YieldOp, scf.YieldOp)

    def scope(op: ProgramOp | Loop, context: Context) -> Scope:
        body_context = context
        if not isinstance(op, ProgramOp):
            body_context = replace(context, clocks=(*context.clocks, ClockFrame(op)))
        return Scope(op, context, tuple(block(op.body.block, body_context)))

    def block(body: Block, context: Context) -> Iterator[Scope | Work]:
        for position, op in enumerate(body.ops):
            here = replace(context, path=(*context.path, position))
            if isinstance(op, (PipelineOp, scf.ForOp)):
                yield scope(op, here)
            elif isinstance(op, StageOp):
                clock = context.clock
                require(
                    clock is not None and op.parent_op() is clock.op,
                    "source Stage must select its actual enclosing pipeline clock",
                )
                frames = (*context.clocks[:-1], ClockFrame(clock.op, op))
                yield from block(op.body.block, replace(here, clocks=frames))
            elif isinstance(op, DomainOp):
                yield from block(op.body.block, replace(here, actor=op.kind.data, domain=op))
            elif isinstance(op, scf.IfOp):
                for branch, region in enumerate((op.true_region, op.false_region)):
                    guarded = replace(
                        here,
                        guards=(*context.guards, Guard(op, branch == 0)),
                        path=(*here.path, branch),
                    )
                    yield from block(region.block, guarded)
            elif isinstance(op, effect_types):
                require(
                    here.actor in {"cube", "vector"},
                    "mixed source work needs an explicit Cube or Vector execution domain",
                )
                yield Work(op, here)
            elif (
                isinstance(op, (*descriptors, arith.ConstantOp, arith.CmpiOp))
                or type(op) in ARITHMETIC
            ):
                continue
            else:
                require(False, f"unsupported source operation {op.name}; effects cannot be ignored")

    actor = None if program.kind.data == "mixed" else program.kind.data
    return scope(program, Context((), (), actor, None, ()))

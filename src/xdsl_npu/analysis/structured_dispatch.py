"""Bind real native templates to independent actor controller schedules.

Source order remains in memory facts. This one structural projection visits
each participant's code, applies stage issue_order, and indexes direct native
anchors within each actual loop. A child occupies one lexical entry in its
parent; its natives keep their own clock. No execution iterations are expanded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects import scf
from xdsl.ir import Operation

from ..diagnostics import require
from ..dialects.physical import KernelOp, MixedOp, TaskOp
from ..dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp
from ..target.ascend910b import Actor, Milestone
from .frontier import Clock, PrefixRank
from .native import NativeCut
from .structured_memory import MemoryTemplate, NativeMemoryFacts


@dataclass(frozen=True, slots=True)
class DispatchScope:
    operation: Operation
    actor: Actor
    clock: Clock
    parent: DispatchScope | None
    parent_stage: int = 0
    parent_lexical: int = 0


@dataclass(frozen=True, slots=True)
class StructuredTask:
    template: MemoryTemplate
    scope: DispatchScope
    stage: int
    lexical: int

    @property
    def operation(self):
        return self.template.operation

    @property
    def actor(self):
        return self.scope.actor

    @property
    def clock(self):
        return self.scope.clock

    @property
    def engine(self):
        return self.template.contract.engine

    @property
    def implicit_completion(self):
        return self.template.contract.implicit_completion

    @property
    def packet(self):
        return self.template.contract.packet

    @property
    def native(self):
        return self.template.contract.native

    @property
    def scalar_source(self):
        return self.template.scalar_source

    @property
    def registers(self):
        # Native packets are closed. Invariant FIX/Load3D configuration is
        # admitted once on the owned structure, separately from local edges.
        return ()

    @property
    def last(self):
        # FIRST/LAST are actual boundary occurrences, never a Task attribute.
        return False

    @property
    def rank(self):
        return PrefixRank(self.clock, self.stage, self.lexical)

    def cut(self, milestone, *, generation_offset=0):
        return NativeCut(
            self.operation,
            self.actor,
            None if milestone is Milestone.ISSUE else self.engine,
            PrefixRank(self.clock, self.stage + generation_offset, self.lexical),
            milestone,
        )


@dataclass(frozen=True)
class StructuredDispatch:
    memory: NativeMemoryFacts
    scopes: Mapping[tuple[Operation, Actor], DispatchScope]
    tasks: Mapping[tuple[TaskOp, Actor], StructuredTask]
    bodies: Mapping[DispatchScope, tuple[StructuredTask, ...]]
    projected_operations: int


def bind_structured_dispatch(memory: NativeMemoryFacts) -> StructuredDispatch:
    root = memory.module.body.block.first_op
    templates = {template.operation: template for template in memory.templates}
    scopes, tasks, bodies = {}, {}, {}
    projected = 0
    actors = (
        (Actor.AIC, Actor.AIV0, Actor.AIV1)
        if isinstance(root, MixedOp)
        else (Actor.AIC,)
        if isinstance(root, KernelOp) and root.core.data == "aic"
        else (Actor.AIV0,)
    )

    def scope_for(op, actor, parent=None, stage=0, lexical=0):
        scope = DispatchScope(op, actor, Clock(f"scope-{len(scopes)}"), parent, stage, lexical)
        scopes[op, actor], bodies[scope] = scope, []
        return scope

    for actor in actors:
        family = "cube" if actor is Actor.AIC else "vector"
        positions = {}

        def visit(operation, scope, stage=0, *, actor=actor, family=family, positions=positions):
            nonlocal projected
            projected += 1
            if isinstance(operation, ActorDomainOp) and operation.kind.data != family:
                return
            if isinstance(operation, TaskOp):
                template = templates[operation]
                if actor not in template.participants:
                    return
                lexical = positions.get(scope, 0)
                positions[scope] = lexical + 1
                task = StructuredTask(template, scope, stage, lexical)
                tasks[operation, actor] = task
                bodies[scope].append(task)
                return
            if isinstance(operation, (PipelineLoopOp, scf.ForOp)):
                lexical = positions.get(scope, 0)
                positions[scope] = lexical + 1
                scope = scope_for(operation, actor, scope, stage, lexical)
                stage = 0
            elif isinstance(operation, PipelineStageOp):
                stage = operation.offset.data
            if isinstance(operation, PipelineLoopOp):
                children = tuple(
                    child
                    for child in operation.body.block.ops
                    if isinstance(child, PipelineStageOp)
                )
                policy = (
                    None
                    if operation.issue_order is None
                    else operation.issue_order.data.get(family)
                )
                if policy is not None:
                    by_name = {child.label.data: child for child in children}
                    require(
                        len(by_name) == len(children),
                        "physical stage order requires unambiguous labels",
                    )
                    selected = tuple(by_name[label.data] for label in policy)
                    omitted = set(children) - set(selected)
                    require(
                        not any(
                            actor in templates[op].participants
                            for child in omitted
                            for op in child.walk()
                            if isinstance(op, TaskOp)
                        ),
                        "physical stage order omitted work for its participant",
                    )
                    children = selected
                for child in children:
                    visit(child, scope, stage)
            else:
                for region in operation.regions:
                    for block in region.blocks:
                        for child in block.ops:
                            visit(child, scope, stage)

        visit(root, scope_for(root, actor))
    require(
        len(tasks) == sum(len(template.participants) for template in memory.templates),
        "physical dispatch projection lost an actual participant",
    )
    return StructuredDispatch(
        memory,
        MappingProxyType(scopes),
        MappingProxyType(tasks),
        MappingProxyType({scope: tuple(body) for scope, body in bodies.items()}),
        projected,
    )

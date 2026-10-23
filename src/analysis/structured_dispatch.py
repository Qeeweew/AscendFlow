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
from ..memo import memoized_method
from ..scalar import expression, literal
from ..target.ascend910b import Actor, Engine
from .access_frontier import QueueEndpoint
from .local_schedule import LocalSchedule, bind_local_schedules
from .structured_memory import MemoryTemplate, NativeMemoryFacts
from .structured_schedule import Placement


@dataclass(frozen=True, slots=True, eq=False)
class DispatchScope:
    """One actual actor invocation scope, compared by identity."""

    operation: Operation
    actor: Actor
    name: str
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
    def engine(self):
        return self.template.spec.engine

    @property
    def implicit_completion(self):
        return self.template.spec.implicit_completion

    @property
    def native(self):
        return self.template.spec.native


@dataclass(frozen=True, slots=True)
class IssueSite:
    """Controller position; its existence supplies no queue completion."""

    scope: DispatchScope
    stage: int
    lexical: int
    engine: Engine
    implicit_completion: bool = False

    @property
    def actor(self):
        return self.scope.actor


@dataclass(frozen=True)
class StructuredDispatch:
    memory: NativeMemoryFacts
    scopes: Mapping[tuple[Operation, Actor], DispatchScope]
    tasks: Mapping[tuple[TaskOp, Actor], StructuredTask]
    bodies: Mapping[DispatchScope, tuple[StructuredTask, ...]]
    projected_operations: int
    boundaries: Mapping[tuple[Operation, Actor], tuple[DispatchScope, int, int, int]]
    schedules: Mapping[tuple[PipelineLoopOp, str], LocalSchedule]

    def site(self, point):
        occurrence = getattr(point, "occurrence", point)
        return self._site(occurrence.endpoint, getattr(point, "placement", Placement.BEFORE))

    @memoized_method
    def _site(self, endpoint, placement):
        if isinstance(endpoint, QueueEndpoint):
            scope, stage, entry, exit = self.boundaries[endpoint.operation, endpoint.participant]
            lexical = entry * 3 if endpoint.placement is Placement.ENTRY else exit * 3 + 2
            for origin in endpoint.retirements:
                task = self.tasks[origin, endpoint.participant]
                require(
                    task.engine is Engine.S and task.implicit_completion,
                    "queue prefix retirement lost its admitted Scalar source contract",
                )
            return IssueSite(scope, stage, lexical, endpoint.engine, bool(endpoint.retirements))
        task = self.tasks[endpoint.operation, endpoint.participant]
        return IssueSite(
            task.scope,
            task.stage,
            task.lexical * 3 + int(placement is Placement.AFTER),
            task.engine,
            task.implicit_completion,
        )

    def position_path(self, point):
        """Hierarchical issue coordinates; no activation or completion claim."""
        return self._position_path(self.site(point), point.coordinates)

    @memoized_method
    def _position_path(self, site, coordinates):
        coordinates = dict(coordinates)
        children, scope = [], site.scope
        while scope.parent is not None:
            children.append(scope)
            scope = scope.parent
        result = []
        for child in reversed(children):
            parent = child.parent
            result.append(
                (
                    parent,
                    expression(
                        "add",
                        coordinates.get(parent.operation, literal(0)),
                        literal(child.parent_stage),
                    ),
                    child.parent_lexical * 3 + 1,
                )
            )
        result.append(
            (
                site.scope,
                expression(
                    "add", coordinates.get(site.scope.operation, literal(0)), literal(site.stage)
                ),
                site.lexical,
            )
        )
        return tuple(result)


def bind_structured_dispatch(memory: NativeMemoryFacts) -> StructuredDispatch:
    root = memory.module.body.block.first_op
    templates = {template.operation: template for template in memory.templates}
    scopes, tasks, bodies, boundaries = {}, {}, {}, {}
    schedules = bind_local_schedules(memory)
    projected = 0
    actors = (
        (Actor.AIC, Actor.AIV0, Actor.AIV1)
        if isinstance(root, MixedOp)
        else (Actor.AIC,)
        if isinstance(root, KernelOp) and root.core.data == "aic"
        else (Actor.AIV0,)
    )

    def scope_for(op, actor, parent=None, stage=0, lexical=0):
        scope = DispatchScope(op, actor, f"scope-{len(scopes)}", parent, stage, lexical)
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
                boundaries[operation, actor] = (scope, stage, lexical, lexical)
                scope = scope_for(operation, actor, scope, stage, lexical)
                stage = 0
            elif isinstance(operation, scf.IfOp):
                entry = positions.get(scope, 0)
                positions[scope] = entry + 1
            elif isinstance(operation, PipelineStageOp):
                stage = operation.offset.data
            if isinstance(operation, PipelineLoopOp):
                schedule = schedules.get((operation, family))
                for child in () if schedule is None else schedule.stages:
                    visit(child, scope, stage)
            else:
                for region in operation.regions:
                    for block in region.blocks:
                        for child in block.ops:
                            visit(child, scope, stage)
            if isinstance(operation, scf.IfOp):
                exit = positions.get(scope, 0)
                positions[scope] = exit + 1
                boundaries[operation, actor] = (scope, stage, entry, exit)

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
        MappingProxyType(boundaries),
        schedules,
    )

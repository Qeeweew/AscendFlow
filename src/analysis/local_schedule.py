"""Actor-local issue policies on retained logical iteration templates.

Each policy references the existing loop/stage operations. It supplies no
completion or event-consumption evidence and never changes logical source order.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.ir import Operation

from ..diagnostics import require
from ..dialects.structured import PipelineLoopOp, PipelineStageOp
from ..target.ascend910b import Actor


@dataclass(frozen=True, slots=True)
class LocalSchedule:
    loop: PipelineLoopOp
    family: str
    stages: tuple[PipelineStageOp, ...]

    @property
    def horizon(self):
        return max((stage.offset.data for stage in self.stages), default=0)

    def project(self, operations: Mapping[Operation, Operation]):
        """Carry the unchanged policy through an owned one-to-one IR clone."""
        return LocalSchedule(
            operations[self.loop],
            self.family,
            tuple(operations[stage] for stage in self.stages),
        )


def bind_local_schedules(memory):
    """Establish one issue policy per actual loop and participating family."""
    families = {}
    for template in memory.templates:
        families[template.operation] = {
            "cube" if actor is Actor.AIC else "vector" for actor in template.participants
        }
    operations = tuple(memory.module.walk())
    for operation in reversed(operations):
        if operation.parent_op() is not None:
            families.setdefault(operation.parent_op(), set()).update(families.get(operation, ()))
    result = {}
    for loop in operations:
        if not isinstance(loop, PipelineLoopOp):
            continue
        for family in ("cube", "vector"):
            stages = tuple(
                stage
                for stage in loop.body.block.ops
                if isinstance(stage, PipelineStageOp) and family in families.get(stage, ())
            )
            if not stages:
                continue
            policy = None if loop.issue_order is None else loop.issue_order.data.get(family)
            if policy is not None:
                by_name = {stage.label.data: stage for stage in stages}
                names = tuple(label.data for label in policy)
                require(
                    len(by_name) == len(stages)
                    and len(set(names)) == len(names)
                    and set(names) == set(by_name),
                    f"{family}: issue_order must name every participating Stage exactly once",
                )
                stages = tuple(by_name[name] for name in names)
            result[loop, family] = LocalSchedule(loop, family, stages)
    return MappingProxyType(result)

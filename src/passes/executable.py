"""Final SSA and the static native geometry preserved by owned lowering.

These records contain no logical loop, source address expression, dependency,
or allocation query. They are not synchronization certificates: executable
emission is authorized only by the owned source compilation that builds them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.structured import PipelineLoopOp, PipelineStageOp
from ..target.ascend910b import Actor
from ..target.native import NativeSpec


@dataclass(frozen=True, slots=True)
class NativeExecution:
    operation: TaskOp
    participants: tuple[Actor, ...]
    spec: NativeSpec

    def __post_init__(self):
        count = self.operation.attributes.get("address_count")
        require(
            self.operation.native.data == self.spec.native
            and (0 if self.operation.unit_flag is None else self.operation.unit_flag.data)
            == self.spec.unit_flag
            and tuple(value.data for value in self.operation.params) == self.spec.params
            and (len(self.operation.refs) if count is None else count.data)
            == len(self.spec.operands),
            "terminal native differs from its admitted instruction specification",
        )

    @classmethod
    def from_template(cls, template, operation):
        return cls(operation, template.participants, template.spec)


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    module: ModuleOp
    natives: tuple[NativeExecution, ...]
    scalar_ranges: Mapping[SSAValue, tuple[int, int]]

    def __post_init__(self):
        operations, values = set(), set()
        for operation in self.module.walk():
            require(
                not isinstance(operation, (PipelineLoopOp, PipelineStageOp)),
                "terminal program retains a logical pipeline controller",
            )
            if isinstance(operation, TaskOp):
                operations.add(operation)
            values.update(operation.results)
            for region in operation.regions:
                for block in region.blocks:
                    values.update(block.args)
        require(
            len(self.natives) == len(operations)
            and {native.operation for native in self.natives} == operations,
            "terminal lowering changed the native instruction correspondence",
        )
        object.__setattr__(
            self,
            "scalar_ranges",
            MappingProxyType(
                {v: bounds for v, bounds in self.scalar_ranges.items() if v in values}
            ),
        )

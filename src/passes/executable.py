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
from ..target.native import NativeGeometry
from ..target.packets import NativePacket


@dataclass(frozen=True, slots=True)
class NativeOperand:
    operand_index: int
    geometry: NativeGeometry


@dataclass(frozen=True, slots=True)
class NativeExecution:
    operation: TaskOp
    participants: tuple[Actor, ...]
    operands: tuple[NativeOperand, ...]
    packet: NativePacket

    @classmethod
    def from_template(cls, template, operation):
        require(template.contract.packet is not None, "native execution needs an audited packet")
        return cls(
            operation,
            template.participants,
            tuple(NativeOperand(a.operand_index, a.address.geometry) for a in template.accesses),
            template.contract.packet,
        )


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

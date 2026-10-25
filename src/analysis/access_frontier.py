"""Native access identities, milestone cuts and explicit loop selections.

Structured dependency composition lives in access_order. Endpoints
retain their actual task, participant and control frame; generation bindings
belong to structured_regions.AccessOccurrence. Selecting FIRST/LAST ordinals
establishes no native retirement or completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.dialects import scf
from xdsl.ir import Operation

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.structured import PipelineLoopOp
from ..scalar import Expr
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine
from .structured_control import ControlFrame
from .structured_schedule import Placement

if TYPE_CHECKING:
    from .structured_memory import MemoryTemplate


@dataclass(frozen=True, slots=True)
class AccessEndpoint:
    """One admitted native identity with its actual actor and control frame."""

    operation: TaskOp
    participant: Actor
    frame: ControlFrame

    def __post_init__(self):
        require(
            isinstance(self.operation, TaskOp)
            and isinstance(self.participant, Actor)
            and isinstance(self.frame, ControlFrame)
            and self.participant in self.frame.participants,
            "access endpoint needs an actual TaskOp, participant and owned control frame",
        )

    @classmethod
    def from_template(cls, template: MemoryTemplate, participant: Actor):
        return cls(template.operation, participant, template.frame)


@dataclass(frozen=True, slots=True)
class QueueEndpoint:
    """A real invocation insertion position on one participant's queue.

    This is an issue site, never a native effect or a completed access. Only
    a materialized Set captures the queue prefix and only its Wait acquires
    it. The frame is the surrounding invocation, including on an empty path.
    """

    operation: Operation
    participant: Actor
    frame: ControlFrame
    engine: Engine
    placement: Placement
    # Source identities covered by synchronous Scalar retirement at EXIT.
    # A plain queue cut carries no such completion fact. ENTRY, Vector/Cube
    # work and Scalar stores never gain this controller-completion contract.
    retirements: tuple[TaskOp, ...] = ()

    def __post_init__(self):
        require(
            isinstance(self.operation, (PipelineLoopOp, scf.ForOp, scf.IfOp))
            and self.placement in (Placement.ENTRY, Placement.EXIT)
            and supports_engine(self.participant, self.engine),
            "queue endpoint needs a real region boundary and participant queue",
        )
        require(
            isinstance(self.retirements, tuple)
            and (
                not self.retirements
                or self.engine is Engine.S
                and self.placement is Placement.EXIT
                and all(isinstance(op, TaskOp) for op in self.retirements)
            ),
            "only a Scalar EXIT prefix can retain synchronous source retirements",
        )


@dataclass(frozen=True, slots=True)
class AccessCut:
    endpoint: AccessEndpoint | QueueEndpoint
    milestone: Milestone


class ResiduePosition(str, Enum):
    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class LoopResidueSelection:
    """An actual loop's selected logical ordinal, not a closed loop summary."""

    loop: PipelineLoopOp | scf.ForOp
    period: int
    residue: int | Expr
    position: ResiduePosition

    def __post_init__(self):
        require(
            isinstance(self.loop, (PipelineLoopOp, scf.ForOp))
            and type(self.period) is int
            and self.period > 0
            and (
                type(self.residue) is int
                and 0 <= self.residue < self.period
                or isinstance(self.residue, Expr)
                and self.residue.kind == "rem"
                and self.residue.args[1].constant == self.period
            )
            and isinstance(self.position, ResiduePosition),
            "loop residue selection needs an actual loop, period/residue and FIRST/LAST",
        )

    def evaluate(self, count: int, values: dict[str, int] | None = None) -> int | None:
        """Bounded test/materialization interpreter; never enumerate N.

        The guard precedes N-1 arithmetic. Induction remains lb+ordinal*step;
        selecting a task's last occurrence proves no native retirement.
        """
        require(type(count) is int and count >= 0, "selection count must be nonnegative")
        require(
            values is not None or type(self.residue) is int,
            "symbolic residue needs coordinate values",
        )
        residue = self.residue if type(self.residue) is int else self.residue.evaluate(values)
        if count <= residue:
            return None
        if self.position is ResiduePosition.FIRST:
            return residue
        return count - 1 - ((count - 1 - residue) % self.period)

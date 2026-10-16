"""Shared native modes, instruction contracts, and physical milestone cuts.

These types retain actual operation identities and admitted native geometry.
Memory hazards and event lifetimes are established by their own analysis stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..diagnostics import ErrorKind, require
from ..dialects.physical import TaskOp
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine
from ..target.native import implicit_completion as native_implicit_completion
from ..target.packets import NativePacket, audited_packet
from .frontier import PrefixRank


class AccessMode(str, Enum):
    READ = "R"
    WRITE = "W"
    READ_WRITE = "RW"

    @property
    def reads(self):
        return "R" in self.value

    @property
    def writes(self):
        return "W" in self.value


@dataclass(frozen=True, slots=True)
class NativeContract:
    """One actual instruction's audited route, operands, and scalar literal."""

    native: str
    engine: Engine
    modes: tuple[str, ...]
    params: tuple[int, ...]
    scalar: float
    packet: NativePacket | None = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "packet", audited_packet(self.native))

    @property
    def pipe(self):
        return self.engine.value

    @property
    def implicit_completion(self):
        return native_implicit_completion(self)


@dataclass(frozen=True, slots=True)
class NativeCut:
    operation: TaskOp
    actor: Actor
    engine: Engine | None
    rank: PrefixRank
    milestone: Milestone

    def __post_init__(self):
        require(isinstance(self.operation, TaskOp), "native cut needs an actual TaskOp")
        require(
            isinstance(self.actor, Actor) and isinstance(self.rank, PrefixRank),
            "native cut needs a declared actor and clock rank",
        )
        require(isinstance(self.milestone, Milestone), "native cut needs a typed milestone")
        require(
            self.milestone in {Milestone.ISSUE, Milestone.ADMISSION, Milestone.NATIVE_END},
            "a native end anchor cannot establish a retirement capture",
            kind=ErrorKind.UNSUPPORTED,
            phase="native effects",
        )
        require(
            self.engine is None
            if self.milestone is Milestone.ISSUE
            else supports_engine(self.actor, self.engine),
            "native cut has an invalid engine",
        )

    @property
    def clock(self):
        return self.rank.clock

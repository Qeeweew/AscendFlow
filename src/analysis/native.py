"""Native access modes and instruction contracts.

These types retain actual operation identities and admitted native geometry.
Memory hazards and event lifetimes are established by their own analysis stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from xdsl.ir import Operation

from ..diagnostics import require
from ..scalar import Expr
from ..target.ascend910b import Engine
from ..target.native import NativeReadReset, native_read_resets
from ..target.native import implicit_completion as native_implicit_completion
from ..target.packets import NativePacket, audited_packet
from .structured_schedule import Comparison


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
class ConditionalRead:
    """An old-value read and its complementary complete-kill condition.

    The ordinal belongs to the actual stage's pipeline, before its dispatch
    offset. These predicates describe numerical dependence only. They supply
    no initialization, completion, or permission to remove physical effects.
    """

    loop: Operation
    ordinal: Expr
    read: Comparison
    kill: Comparison


@dataclass(frozen=True, slots=True)
class NativeContract:
    """One actual instruction's audited route, operands, and scalar literal."""

    native: str
    engine: Engine
    modes: tuple[str, ...]
    params: tuple[int, ...]
    scalar: float
    packet: NativePacket | None = field(init=False)
    read_resets: tuple[NativeReadReset, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "packet", audited_packet(self.native))
        resets = native_read_resets(self.native, self.params)
        require(
            all(
                reset.operand < len(self.modes) and self.modes[reset.operand] == "RW"
                for reset in resets
            ),
            "conditional native reset needs a complete read/write operand",
        )
        object.__setattr__(self, "read_resets", resets)

    @property
    def pipe(self):
        return self.engine.value

    @property
    def implicit_completion(self):
        return native_implicit_completion(self)

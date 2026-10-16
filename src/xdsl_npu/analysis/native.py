"""Native access modes and instruction contracts.

These types retain actual operation identities and admitted native geometry.
Memory hazards and event lifetimes are established by their own analysis stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..target.ascend910b import Engine
from ..target.native import implicit_completion as native_implicit_completion
from ..target.packets import NativePacket, audited_packet


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

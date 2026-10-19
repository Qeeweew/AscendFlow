"""Physical access modes and loop-bound numerical read conditions.

These types retain actual operation identities and admitted native geometry.
Memory hazards and event lifetimes are established by their own analysis stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdsl.ir import Operation

from ..scalar import Expr
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

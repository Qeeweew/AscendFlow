"""A constant-size fill/body/drain schedule for one retained loop template.

The attribute is a control schedule, never an asynchronous completion token.
Every phase refers to the same IR body. Neither nesting nor a large trip count
copies that body. Release/credit guards come from dependencies, not this depth.
"""

from dataclasses import dataclass
from enum import StrEnum

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import VerifyException


class PhaseKind(StrEnum):
    SHORT = "short"
    PROLOGUE = "prologue"
    LOOP = "loop"
    EPILOGUE = "epilogue"


@dataclass(frozen=True, slots=True)
class CountBound:
    """a * trip_count + b; constructing a bound never reads a runtime count."""

    coefficient: int
    offset: int

    def evaluate(self, count: int) -> int:
        return self.coefficient * count + self.offset


@dataclass(frozen=True, slots=True)
class Phase:
    kind: PhaseKind
    lower: CountBound
    upper: CountBound
    minimum_count: int
    maximum_count: int | None = None

    def interval(self, count: int) -> tuple[int, int] | None:
        """Bind for a finite interpreter/debugger; production keeps the bounds."""
        if type(count) is not int or count < 0:
            raise ValueError("loop trip count must be a nonnegative integer")
        if count < self.minimum_count:
            return None
        if self.maximum_count is not None and count > self.maximum_count:
            return None
        return self.lower.evaluate(count), self.upper.evaluate(count)


@irdl_attr_definition
class LoopScheduleAttr(ParametrizedAttribute):
    """Maximum logical stage offset, not a buffer count or completion depth."""

    name = "npu.loop_schedule"
    horizon: IntAttr

    def verify(self):
        if not 0 <= self.horizon.data <= 32:
            raise VerifyException("pipeline stage offset must be in [0,32]")

    @property
    def phases(self) -> tuple[Phase, ...]:
        depth = self.horizon.data
        self.verify()
        zero, count = CountBound(0, 0), CountBound(1, 0)
        if depth == 0:
            return (Phase(PhaseKind.LOOP, zero, count, 1),)
        end = CountBound(1, depth)
        boundary = CountBound(0, depth)
        short = (Phase(PhaseKind.SHORT, zero, end, 1, depth - 1),) if depth > 1 else ()
        return short + (
            Phase(PhaseKind.PROLOGUE, zero, boundary, depth),
            Phase(PhaseKind.LOOP, boundary, count, depth),
            Phase(PhaseKind.EPILOGUE, count, end, depth),
        )

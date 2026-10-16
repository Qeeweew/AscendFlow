"""Print a fixed prologue/steady/epilogue split in actual dispatch order.

The input command body remains single and ordered. At most four fixed phase
bodies are rendered, regardless of the runtime count. This may duplicate ASC
text by a constant factor; it does not by itself establish a speedup.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby
from typing import Literal, Protocol, TypeVar

from xdsl.dialects.builtin import IntAttr

from ..diagnostics import require
from ..dialects.loops import LoopScheduleAttr, PhaseKind
from ..dialects.loops import Phase as SchedulePhase

Phase = Literal["ordinary", "prologue", "steady", "epilogue", "short"]


@dataclass(frozen=True, slots=True)
class CountBound:
    """Only a constant or the existing count plus a constant offset."""

    coefficient: int = 0
    offset: int = 0

    def __post_init__(self):
        require(
            type(self.coefficient) is int and self.coefficient in (0, 1),
            "pipeline bound needs a count coefficient of zero or one",
        )
        require(type(self.offset) is int, "pipeline bound needs an integer offset")

    def c(self, count: str = "n") -> str:
        if not self.coefficient:
            return str(self.offset)
        if not self.offset:
            return count
        return f"{count} {'+' if self.offset > 0 else '-'} {abs(self.offset)}"

    def evaluate(self, count: int) -> int:
        require(type(count) is int and count >= 0, "pipeline count must be nonnegative")
        return self.coefficient * count + self.offset


@dataclass(frozen=True, slots=True)
class PipelineDomain:
    """Bounds at one group's body, after its remaining stage validity guard.

    ``step_upper`` and ``count_limit`` are exclusive. The two generation
    bounds certify ``g >= generation_minimum`` and
    ``g < count - generation_upper_gap`` only within this phase/group body.
    They do not imply any more distant protocol's startup or release guard.
    """

    phase: Phase
    stage: int
    maximum: int
    count: str
    step_lower: CountBound
    step_upper: CountBound
    count_minimum: int
    count_limit: int | None = None

    @property
    def generation_minimum(self) -> int:
        return self.maximum - self.stage if self.phase in ("steady", "epilogue") else 0

    @property
    def generation_upper_gap(self) -> int:
        return self.stage if self.phase in ("prologue", "steady") else 0

    def implies_lower(self, lower: int) -> bool:
        require(type(lower) is int and lower >= 0, "pipeline lower guard must be nonnegative")
        return lower <= self.generation_minimum

    def implies_upper(self, gap: int) -> bool:
        require(type(gap) is int and gap >= 0, "pipeline upper gap must be nonnegative")
        return gap <= self.generation_upper_gap

    def permits(self, lower: int = 0, upper_gap: int = 0) -> bool:
        """Whether some integer (g, N) satisfies this phase and both guards.

        A generation exists iff every affine lower bound is strictly below
        every affine upper bound. Four comparisons intersect the resulting
        integer N bounds with this phase's count interval. Coefficients are
        only zero/one, so no count, step, or constant-sized range is scanned.
        This does not decide residue or other independent protocol guards.
        """
        require(type(lower) is int and lower >= 0, "pipeline lower guard must be nonnegative")
        require(type(upper_gap) is int and upper_gap >= 0, "pipeline upper gap must be nonnegative")
        lower_bounds = (
            CountBound(self.step_lower.coefficient, self.step_lower.offset - self.stage),
            CountBound(0, lower),  # Also enforces the native's g >= 0.
        )
        upper_bounds = (
            CountBound(self.step_upper.coefficient, self.step_upper.offset - self.stage),
            CountBound(1, -upper_gap),  # Also enforces the native's g < N.
        )
        minimum, limit = max(0, self.count_minimum), self.count_limit
        for begin in lower_bounds:
            for end in upper_bounds:
                coefficient = begin.coefficient - end.coefficient
                gap = end.offset - begin.offset
                if coefficient == 0:
                    if gap <= 0:
                        return False
                elif coefficient == -1:
                    minimum = max(minimum, 1 - gap)
                else:
                    limit = gap if limit is None else min(limit, gap)
        return limit is None or minimum < limit

    @property
    def stage_lower_guard(self) -> bool:
        return self.stage > 0 and self.phase in ("prologue", "short")

    @property
    def stage_upper_guard(self) -> bool:
        return self.phase in ("epilogue", "short") and self.stage < self.maximum

    def active(self, count: int) -> bool:
        require(type(count) is int and count >= 0, "pipeline count must be nonnegative")
        return self.count_minimum <= count and (
            self.count_limit is None or count < self.count_limit
        )

    def contains(self, step: int, count: int) -> bool:
        """Finite evaluator of the emitted phase and stage guards."""
        require(type(step) is int, "pipeline step must be an integer")
        if not self.active(count):
            return False
        generation = step - self.stage
        return (
            self.step_lower.evaluate(count) <= step < self.step_upper.evaluate(count)
            and (not self.stage_lower_guard or generation >= 0)
            and (not self.stage_upper_guard or generation < count)
        )


class _StagedTask(Protocol):
    stage: int


class _Command(Protocol):
    task: _StagedTask


Command = TypeVar("Command", bound=_Command)
Renderer = Callable[[tuple[Command, ...], PipelineDomain], Iterable[str]]


def _phase_lines(
    groups: tuple[tuple[int, tuple[Command, ...]], ...],
    render_group: Renderer[Command],
    *,
    phase: SchedulePhase,
    maximum: int,
    count: str,
) -> list[str]:
    name: Phase = {
        PhaseKind.SHORT: "short",
        PhaseKind.PROLOGUE: "prologue",
        PhaseKind.LOOP: "ordinary" if maximum == 0 else "steady",
        PhaseKind.EPILOGUE: "epilogue",
    }[phase.kind]
    lower = CountBound(phase.lower.coefficient, phase.lower.offset)
    upper = CountBound(phase.upper.coefficient, phase.upper.offset)
    count_limit = None if phase.maximum_count is None else phase.maximum_count + 1
    lines = [f"for (int64_t step = {lower.c(count)}; step < {upper.c(count)}; ++step) {{ // {name}"]
    for stage, group in groups:
        if (phase.kind is PhaseKind.PROLOGUE and stage == maximum) or (
            phase.kind is PhaseKind.EPILOGUE and stage == 0
        ):
            continue
        domain = PipelineDomain(
            name, stage, maximum, count, lower, upper, phase.minimum_count, count_limit
        )
        generation = "step" if stage == 0 else f"step - {stage}"
        lines.extend([f"  {{ // stage {stage}", f"    const int64_t g = {generation};"])
        guards = []
        if domain.stage_lower_guard:
            guards.append("g >= 0")
        if domain.stage_upper_guard:
            guards.append(f"g < {count}")
        if guards:
            lines.append(f"    if ({' && '.join(guards)}) {{")
        indent = "      " if guards else "    "
        for statement in render_group(group, domain):
            lines.extend(f"{indent}{line}" for line in statement.splitlines())
        if guards:
            lines.append("    }")
        lines.append("  }")
    lines.append("}")
    return lines


def emit_pipeline_body(
    commands: Iterable[Command],
    render_group: Renderer[Command],
    *,
    count: str = "n",
    schedule: LoopScheduleAttr | None = None,
) -> list[str]:
    """Render consecutive equal-stage groups, without sorting or expansion.

    The renderer owns native/sync formatting and a fresh scalar scope for the
    supplied group. Its domain can simplify only guards it actually implies.
    All phase bounds come from the canonical LoopScheduleAttr. A supplied
    schedule must describe the actual maximum stage, rather than overriding
    it. Returned lines have no outer indentation and assume a nonnegative
    count.
    """
    require(
        type(count) is str and count.isascii() and count.isidentifier(), "invalid pipeline count"
    )
    commands = tuple(commands)
    for command in commands:
        stage = getattr(getattr(command, "task", None), "stage", None)
        require(type(stage) is int and stage >= 0, "pipeline command needs a nonnegative stage")
    maximum = max((command.task.stage for command in commands), default=0)
    if schedule is None:
        schedule = LoopScheduleAttr(IntAttr(maximum))
    require(isinstance(schedule, LoopScheduleAttr), "pipeline emission needs a typed loop schedule")
    require(
        schedule.horizon.data == maximum,
        "pipeline schedule horizon does not match the actual maximum stage",
    )
    phases = schedule.phases
    if not commands:
        return []
    groups = tuple(
        (stage, tuple(group))
        for stage, group in groupby(commands, key=lambda command: command.task.stage)
    )
    if maximum == 0:
        return _phase_lines(
            groups,
            render_group,
            phase=phases[0],
            maximum=0,
            count=count,
        )
    long = tuple(phase for phase in phases if phase.kind is not PhaseKind.SHORT)
    short = next((phase for phase in phases if phase.kind is PhaseKind.SHORT), None)
    lines = [f"if ({count} >= {long[0].minimum_count}) {{"]
    for phase in long:
        lines.extend(
            f"  {line}"
            for line in _phase_lines(
                groups,
                render_group,
                phase=phase,
                maximum=maximum,
                count=count,
            )
        )
    if short is not None:
        lines.append(f"}} else if ({count} > {short.minimum_count - 1}) {{")
        lines.extend(
            f"  {line}"
            for line in _phase_lines(
                groups,
                render_group,
                phase=short,
                maximum=maximum,
                count=count,
            )
        )
    lines.append("}")
    return lines

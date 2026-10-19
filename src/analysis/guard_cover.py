"""Ordered guard admission followed by one shared SMT coverage query.

Coverage describes a mathematical union, never a runtime schedule or an event
lifetime. The original guards are admitted in their actual short-circuit order
before the query consumes their relations. Unknown remains uncovered.
"""

from dataclasses import dataclass

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from .integer_facts import IntegerFacts
from .smt import Answer
from .structured_schedule import ActionGuard, Comparison

_OPPOSITE = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def opposite(test):
    return Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs)


@dataclass(frozen=True, slots=True)
class GuardCover:
    """A query result and its inputs, not an executable proof tree."""

    assumptions: tuple[Comparison, ...]
    alternatives: tuple[ActionGuard, ...]
    answer: Answer


def prove_guard_cover(
    arithmetic: IntegerFacts,
    assumptions: tuple[Comparison, ...],
    alternatives: tuple[ActionGuard, ...],
) -> GuardCover | None:
    poll("event guard cover")
    for tests in (assumptions, *(assumptions + guard.comparisons for guard in alternatives)):
        missing = arithmetic.ordered_definedness(tests)
        if missing is not None:
            required, prefix = missing
            raise CompileError(
                "guard coverage lacks an ordered arithmetic prefix proof",
                kind=ErrorKind.UNSUPPORTED,
                phase="event-guard-cover",
                witness=(
                    f"required: {required.lhs.text()} {required.predicate} {required.rhs.text()}",
                    "prefix: "
                    + "; ".join(f"{t.lhs.text()} {t.predicate} {t.rhs.text()}" for t in prefix),
                ),
            )
    answer = arithmetic.queries.covers(alternatives, assumptions)
    return GuardCover(assumptions, alternatives, answer) if answer.proved else None

"""Cooperative limits for one Python compilation, including its final proof."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from .diagnostics import CompileError, ErrorKind, require


@dataclass
class Budget:
    deadline: float
    cancel: Callable[[], bool] | None


_active = ContextVar("ascendflow_analysis_budget", default=None)


@contextmanager
def limit(seconds=60.0, cancel=None):
    require(
        type(seconds) in {int, float} and 0 < seconds < 1e9 and isfinite(seconds),
        "compilation timeout_s must be finite, positive and below 1e9 seconds",
    )
    require(cancel is None or callable(cancel), "cancel must be a callable predicate")
    parent = _active.get()
    deadline = perf_counter() + seconds
    if parent is not None:
        deadline = min(deadline, parent.deadline)
        requested = cancel
        cancel = lambda: (
            (parent.cancel is not None and parent.cancel())
            or (requested is not None and requested())
        )
    token = _active.set(Budget(deadline, cancel))
    try:
        poll("start")
        yield
    finally:
        _active.reset(token)


def poll(phase, **counts):
    budget = _active.get()
    if budget is None:
        return
    cancelled = budget.cancel is not None and budget.cancel()
    if cancelled or perf_counter() >= budget.deadline:
        detail = ", ".join(f"{name}={value}" for name, value in counts.items())
        reason = "cancelled" if cancelled else "deadline exceeded"
        raise CompileError(
            f"compilation {reason} at {phase}" + (f" ({detail})" if detail else ""),
            kind=ErrorKind.ANALYSIS_LIMIT,
        )


def remaining(phase):
    """Seconds left in the enclosing compilation before its hard deadline.

    ``None`` when no limit is active. Query services use this to bound a single
    solver call by the same deadline as the operator build; a local time limit
    must never outlive the process group's overall budget.
    """
    budget = _active.get()
    if budget is None:
        return None
    return max(0.0, budget.deadline - perf_counter())

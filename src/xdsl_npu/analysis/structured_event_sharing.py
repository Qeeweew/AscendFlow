"""Share finite channels only through existing consume-to-next-Set evidence.

This module consumes the resource stage's canonical placements and bridge
proofs. It introduces no synchronization, schedule or memory analysis. A
repeated invocation requires a per-ID proof beyond this finite contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from .structured_schedule import Comparison


class SharingStep(Protocol):
    """The existing resource bridge step; no reverse resource import."""

    consume: BoundPlacement
    source: BoundPlacement
    notification: LogicalNotification | None
    controller: bool


SharingCandidate = tuple[
    LogicalNotification, tuple[BoundPlacement, ...], tuple[BoundPlacement, ...]
]


class SharingBridge(Protocol):
    def __call__(
        self,
        notification: LogicalNotification,
        consume: BoundPlacement,
        goal: BoundPlacement,
        loop: Operation | None,
        delta: int,
        assumptions: tuple[Comparison, ...],
        candidates: tuple[SharingCandidate, ...],
        /,
    ) -> tuple[SharingStep, ...] | None: ...


@dataclass(frozen=True)
class EventSharingContext:
    selection: SynchronizationSelection
    windows: Mapping[LogicalNotification, tuple[object, ...]]
    canonicalize: Callable[[BoundPlacement, Mapping[Expr, Expr]], BoundPlacement]
    bridge: SharingBridge


@dataclass(frozen=True, slots=True)
class EventSharingWitness:
    bank: tuple[Actor, Engine, Engine]
    earlier: LogicalNotification
    later: LogicalNotification
    consume: BoundPlacement
    next_set: BoundPlacement
    steps: tuple[SharingStep, ...]
    assumptions: tuple[Comparison, ...]


def _single_occurrence(notification, context):
    if (
        notification.mechanism is NotificationKind.PIPE_FENCE
        or context.windows.get(notification) != ()
    ):
        return False
    control = context.selection.dependencies.memory.control
    root = control.module.body.block.first_op
    for placement in notification.sources + notification.consumers:
        frame = control.frames.get(placement.operation)
        if frame is None:
            return False
        coordinates = dict(placement.coordinates)
        if any(
            loop.operation not in coordinates
            or not control.invariant_over(coordinates[loop.operation], root)
            for loop in frame.loops()
        ):
            return False
        canonical = context.canonicalize(placement, {})
        if any(
            not control.invariant_over(test.lhs, root)
            or not control.invariant_over(test.rhs, root)
            for test in canonical.guard.comparisons
        ):
            return False
    return True


def _local_bank(notification):
    if (
        notification.mechanism is not NotificationKind.LOCAL_EVENT
        or len(notification.sources) != 1
        or len(notification.consumers) != 1
        or notification.sources[0].actor is not notification.consumers[0].actor
    ):
        return None
    return (
        notification.sources[0].actor,
        notification.source_engine,
        notification.destination_engine,
    )


def prove_finite_event_sharing(
    left: LogicalNotification,
    right: LogicalNotification,
    *,
    context: EventSharingContext,
) -> EventSharingWitness | None:
    """Prove two one-use local channels never hold the same binary flag.

    Empty common clocks alone do not establish one use: every real ancestor
    coordinate must also select one occurrence for the whole invocation.
    Only existing selected Set/Wait interfaces can establish the return path.
    """
    bank = _local_bank(left)
    if (
        left is right
        or bank is None
        or bank != _local_bank(right)
        or not _single_occurrence(left, context)
        or not _single_occurrence(right, context)
    ):
        return None
    candidates = tuple(
        (
            notification,
            tuple(context.canonicalize(p, {}) for p in notification.sources),
            tuple(context.canonicalize(p, {}) for p in notification.consumers),
        )
        for notification in context.selection.notifications
        if _single_occurrence(notification, context)
    )
    assumptions = tuple(
        test
        for notification in (left, right)
        for placement in notification.sources + notification.consumers
        for test in context.canonicalize(placement, {}).guard.comparisons
    )
    for earlier, later in ((left, right), (right, left)):
        consume = context.canonicalize(earlier.consumers[0], {})
        next_set = context.canonicalize(later.sources[0], {})
        # Both channels use the same actual bank, so the resource bridge's
        # goal source route is also the later channel's actual source route.
        steps = context.bridge(earlier, consume, next_set, None, 0, assumptions, candidates)
        if steps is not None:
            return EventSharingWitness(
                bank, earlier, later, consume, next_set, steps, assumptions
            )
    return None

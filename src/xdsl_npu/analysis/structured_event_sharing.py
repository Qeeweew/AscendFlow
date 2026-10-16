"""Share finite channels only through existing consume-to-next-Set evidence.

This module consumes canonical placements and complete path-cover proofs.
It introduces no synchronization, schedule or memory analysis. A
repeated invocation requires a per-ID proof beyond this finite contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .event_paths import EventPathCandidate, EventPathCover
from .structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from .structured_schedule import Comparison

if TYPE_CHECKING:
    from .event_domains import EventCase

PathCoverProof = Callable[
    [
        LogicalNotification,
        BoundPlacement,
        BoundPlacement,
        tuple[Comparison, ...],
        tuple[EventPathCandidate, ...],
    ],
    EventPathCover | None,
]


class CaseCanonicalizer(Protocol):
    def __call__(
        self,
        placement: BoundPlacement,
        bindings: Mapping[Expr, Expr],
        *,
        case: EventCase,
    ) -> BoundPlacement: ...


@dataclass(frozen=True)
class EventSharingContext:
    selection: SynchronizationSelection
    cases: Mapping[LogicalNotification, tuple[EventCase, ...]]
    canonicalize: CaseCanonicalizer
    prove_path_cover: PathCoverProof


@dataclass(frozen=True, slots=True)
class EventSharingWitness:
    bank: tuple[Actor, Engine, Engine]
    earlier: LogicalNotification
    later: LogicalNotification
    consume: BoundPlacement
    next_set: BoundPlacement
    path_cover: EventPathCover


def _single_occurrence(notification, context):
    cases = context.cases.get(notification, ())
    if notification.mechanism is NotificationKind.PIPE_FENCE or len(cases) != 1:
        return None
    case = cases[0]
    if case.empty or case.windows:
        return None
    control = context.selection.dependencies.memory.control
    root = control.module.body.block.first_op
    for placement in notification.sources + notification.consumers:
        frame = control.frames.get(placement.operation)
        if frame is None:
            return None
        coordinates = dict(placement.coordinates)
        if any(
            loop.operation not in coordinates
            or not control.invariant_over(coordinates[loop.operation], root)
            for loop in frame.loops()
        ):
            return None
        canonical = context.canonicalize(placement, {}, case=case)
        if any(
            not control.invariant_over(test.lhs, root) or not control.invariant_over(test.rhs, root)
            for test in canonical.guard.comparisons
        ):
            return None
    return case


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
    if left is right or bank is None or bank != _local_bank(right):
        return None
    left_case, right_case = _single_occurrence(left, context), _single_occurrence(right, context)
    if left_case is None or right_case is None:
        return None
    finite = {
        notification: case
        for notification in context.selection.notifications
        if (case := _single_occurrence(notification, context)) is not None
    }
    candidates = tuple(
        (
            notification,
            tuple(context.canonicalize(p, {}, case=case) for p in notification.sources),
            tuple(context.canonicalize(p, {}, case=case) for p in notification.consumers),
        )
        for notification, case in finite.items()
    )
    assumptions = tuple(
        test
        for notification, case in ((left, left_case), (right, right_case))
        for placement in notification.sources + notification.consumers
        for test in context.canonicalize(placement, {}, case=case).guard.comparisons
    )
    for earlier, earlier_case, later, later_case in (
        (left, left_case, right, right_case),
        (right, right_case, left, left_case),
    ):
        consume = context.canonicalize(earlier.consumers[0], {}, case=earlier_case)
        next_set = context.canonicalize(later.sources[0], {}, case=later_case)
        # Both channels use the same actual bank, so the path proof's
        # goal source route is also the later channel's actual source route.
        path_cover = context.prove_path_cover(earlier, consume, next_set, assumptions, candidates)
        if path_cover is not None:
            return EventSharingWitness(bank, earlier, later, consume, next_set, path_cover)
    return None

"""One logical notification contract for loop-internal and boundary needs.

Occurrences retain their actual scopes and existence domains. Selection never
assigns event IDs; allocation never rediscovers memory dependences. A finite
child invocation is not reinterpreted as an infinite single-clock stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind, require
from ..scalar import Expr, substitute
from ..target.ascend910b import Engine
from .structured_dependencies import StructuredDependencies
from .structured_dispatch import StructuredDispatch
from .structured_regions import (
    AccessOccurrence,
    LoopDomain,
    StructuredNeed,
)
from .structured_schedule import ActionGuard, Placement


@dataclass(frozen=True, slots=True)
class FamilyCapture:
    loop: Operation
    period: int
    residue: Expr
    first: Expr
    last: Expr
    engine: Engine


@dataclass(frozen=True, slots=True)
class SelectedOrdinal:
    """FIRST/LAST produced by the finite residue selection rule.

    Binding establishes count <= the actual loop count. Under count > first,
    value is a real member, not an arbitrary coordinate carrying a Boolean
    verification tag. Ordered raw guard definedness is still checked later.
    """

    loop: Operation
    count: Expr
    first: Expr
    value: Expr


@dataclass(frozen=True, slots=True)
class BoundPlacement:
    occurrence: AccessOccurrence
    placement: Placement
    guard: ActionGuard
    substitutions: tuple[tuple[Expr, Expr], ...]
    # Prerequisites in the requirement's canonical coordinates. They precede
    # evaluation of a selected FIRST/LAST in the named actual loop's guards.
    coordinate_guards: tuple[tuple[Operation, ActionGuard], ...]
    families: tuple[FamilyCapture, ...]
    # Actual ordinals expressed in the requirement's canonical coordinates.
    # Selection happens exactly once. Resource proofs consume these values.
    coordinates: tuple[tuple[Operation, Expr], ...]
    selected: tuple[SelectedOrdinal, ...] = ()

    @property
    def operation(self):
        return self.occurrence.operation

    @property
    def actor(self):
        return self.occurrence.participant

    def expression(self, value):
        return substitute(value, dict(self.substitutions))


class NotificationKind(str, Enum):
    LOCAL_EVENT = "local-event"
    PIPE_FENCE = "pipe-fence"
    CROSS_EVENT = "cross-event"


@dataclass(frozen=True, slots=True)
class LogicalNotification:
    name: str
    sources: tuple[BoundPlacement, ...]
    consumers: tuple[BoundPlacement, ...]
    source_engine: Engine
    destination_engine: Engine
    mechanism: NotificationKind
    requirements: tuple[StructuredNeed, ...]
    clock: Operation | None
    ordinal: Expr
    period: int
    distance: int | None
    domains: tuple[LoopDomain, ...] = ()
    # The original adjacent lifetime/value edges of the selected real capture
    # and acquire. Coverage can grow; these ownership edges do not change.
    origins: tuple[StructuredNeed, ...] = ()


@dataclass(frozen=True, slots=True)
class OrderingWitness:
    rule: str
    source: BoundPlacement
    consumer: BoundPlacement
    source_engine: Engine
    destination_engine: Engine
    premises: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NotificationCoverage:
    requirement: StructuredNeed
    rule: str
    notification: LogicalNotification | None = None
    ordering: OrderingWitness | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedSynchronization:
    requirement: StructuredNeed
    reason: str


@dataclass(frozen=True)
class SynchronizationSelection:
    dependencies: StructuredDependencies
    dispatch: StructuredDispatch
    notifications: tuple[LogicalNotification, ...]
    coverage: tuple[NotificationCoverage, ...]
    unresolved: tuple[UnresolvedSynchronization, ...] = ()

    @property
    def closed(self):
        return not self.unresolved

    def require_closed(self):
        if self.unresolved:
            issue = self.unresolved[0]
            raise CompileError(
                issue.reason,
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-selection",
                location=str(issue.requirement.consumer.operation.location),
            )
        require(
            len(self.coverage) == len(self.dependencies.required)
            and {id(item.requirement) for item in self.coverage}
            == {id(item) for item in self.dependencies.required},
            "notification selection lost an actual memory requirement",
        )

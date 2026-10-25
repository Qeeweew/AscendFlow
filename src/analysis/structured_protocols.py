"""One logical notification contract for loop-internal and boundary needs.

Occurrences retain their actual scopes and existence domains. Selection never
assigns event IDs; allocation never rediscovers memory dependences. A finite
child invocation is not reinterpreted as an infinite single-clock stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind, require
from ..scalar import Expr, substitute
from ..target.ascend910b import Engine
from .access_order import AccessOrder
from .structured_regions import (
    AccessOccurrence,
    LoopDomain,
    StructuredNeed,
)
from .structured_schedule import ActionGuard, Comparison, Placement

if TYPE_CHECKING:
    from .cell_ownership import CellLifetime
    from .completion_prefixes import CompletionCoverage
    from .region_coordinates import ScalarScope
    from .smt import Exists
    from .structured_dispatch import StructuredDispatch


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
    scope: ScalarScope | None = None
    _hash: int | None = field(default=None, init=False, compare=False, repr=False)

    def __hash__(self):
        value = self._hash
        if value is None:
            value = hash(
                (
                    self.occurrence,
                    self.placement,
                    self.guard,
                    self.substitutions,
                    self.coordinate_guards,
                    self.families,
                    self.coordinates,
                    self.selected,
                    self.scope,
                )
            )
            object.__setattr__(self, "_hash", value)
        return value

    @property
    def operation(self):
        return self.occurrence.operation

    @property
    def actor(self):
        return self.occurrence.participant

    def expression(self, value):
        result = substitute(value, dict(self.substitutions))
        return result if self.scope is None else self.scope.resolve(result)


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
    # Original bindings that own this actual capture/acquire. Coverage can
    # grow without changing the anchors or recovering them from later actions.
    anchors: tuple[RequirementBinding, ...] = ()
    _hash: int | None = field(default=None, init=False, compare=False, repr=False)

    def __hash__(self):
        value = self._hash
        if value is None:
            value = hash(
                (
                    self.name,
                    self.sources,
                    self.consumers,
                    self.source_engine,
                    self.destination_engine,
                    self.mechanism,
                    self.requirements,
                    self.clock,
                    self.ordinal,
                    self.period,
                    self.distance,
                    self.domains,
                    self.anchors,
                )
            )
            object.__setattr__(self, "_hash", value)
        return value


@dataclass(frozen=True, slots=True)
class OrderingWitness:
    rule: str
    source: BoundPlacement
    consumer: BoundPlacement
    source_engine: Engine
    destination_engine: Engine
    premises: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmptyOrdinalDomain:
    """Owned UNSAT query from the two occurrences' actual loop headers.

    This is an input-domain fact, never a completion or a verified label.
    Source arithmetic was already admitted. No source LAST is constructed for
    an occurrence which has no legal consumer ordinal.
    """

    requirement: StructuredNeed
    dispatch: StructuredDispatch
    premises: tuple[Comparison, ...]
    answer: Exists


@dataclass(frozen=True, slots=True)
class RequirementBinding:
    """Actual demand endpoints, independent of a selected notification.

    CopyGroup's required completion prefix may follow the actual writer.
    Storage admission uses ``source``; completion uses ``completion_source``.
    Action coalescing changes neither endpoint nor the original requirement.
    """

    requirement: StructuredNeed
    source: BoundPlacement
    consumer: BoundPlacement
    ready_source: BoundPlacement | None = None

    @property
    def completion_source(self):
        return self.source if self.ready_source is None else self.ready_source


@dataclass(frozen=True, slots=True)
class NotificationCoverage:
    requirement: StructuredNeed
    rule: str
    notification: LogicalNotification | None = None
    ordering: OrderingWitness | None = None
    ownership: CellLifetime | None = None
    absence: EmptyOrdinalDomain | None = None
    completion: CompletionCoverage | None = None
    binding: RequirementBinding | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedSynchronization:
    requirement: StructuredNeed
    reason: str


@dataclass
class BindingWork:
    """Diagnostic counts only; no coverage or lifetime permission."""

    acquisition_words: int = 0
    retained_acquires: int = 0
    word_queries: int = 0
    order_bands: int = 0
    range_vertices: int = 0


@dataclass(frozen=True)
class SynchronizationSelection:
    dependencies: AccessOrder
    notifications: tuple[LogicalNotification, ...]
    coverage: tuple[NotificationCoverage, ...]
    unresolved: tuple[UnresolvedSynchronization, ...] = ()
    binding_work: BindingWork = field(default_factory=BindingWork, compare=False)
    bindings: tuple[RequirementBinding, ...] = ()

    @property
    def dispatch(self):
        return self.dependencies.dispatch

    @property
    def closed(self):
        return not self.unresolved

    def require_closed(self):
        from .completion_prefixes import CompletionCoverage

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
        bindings = {id(item.requirement): item for item in self.bindings}
        required = {id(item) for item in self.dependencies.required}
        require(
            len(bindings) == len(self.bindings)
            and bindings.keys() <= required
            and all(
                (item.binding is not None or item.ownership is not None or item.absence is not None)
                and (item.binding is None or item.binding is bindings.get(id(item.requirement)))
                for item in self.coverage
            ),
            "coverage lost its original owned requirement binding",
        )
        for notification in self.notifications:
            requirements = {id(need) for need in notification.requirements}
            require(
                all(
                    anchor is bindings.get(id(anchor.requirement))
                    and id(anchor.requirement) in requirements
                    for anchor in notification.anchors
                ),
                "notification anchor lost its original owned requirement binding",
            )
        owners = {
            need: lifetime
            for lifetime in self.dependencies.cell_lifetimes
            for need in lifetime.requirements
        }
        require(
            all(
                item.ownership is None
                or item.ownership is owners.get(item.requirement)
                and item.notification is None
                and item.ordering is None
                for item in self.coverage
            ),
            "ownership coverage does not belong to its actual cell lifetime",
        )
        require(
            all(
                sum(
                    item is not None
                    for item in (c.notification, c.ordering, c.ownership, c.absence, c.completion)
                )
                == 1
                and (
                    c.absence is None
                    or c.absence.requirement is c.requirement
                    and c.absence.dispatch is self.dispatch
                    and c.absence.answer.impossible
                )
                for c in self.coverage
            ),
            "coverage lacks exactly one owned notification, ordering, ownership, absence, or completion fact",
        )
        retained = {id(item) for item in self.notifications}
        completions = {
            id(c.completion): c.completion for c in self.coverage if c.completion is not None
        }
        for item in self.coverage:
            proof = item.completion
            require(
                proof is None
                or isinstance(proof, CompletionCoverage)
                and proof.dispatch is self.dispatch
                and item.binding is not None
                and item.binding.requirement is item.requirement
                and proof.covers(item.binding),
                "completion coverage belongs to another requirement or dispatch",
            )
        for proof in completions.values():
            require(
                proof.obligations
                and proof.cases
                and all(item.answer.proved for item in proof.obligations)
                and all(id(item) in retained for item in proof.supports()),
                "completion coverage does not terminate in retained actual actions",
            )

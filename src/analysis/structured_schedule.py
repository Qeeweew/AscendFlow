"""Final instruction placements on the retained physical control tree.

These are output records, not a synchronization solver or a verified marker.
The owned synchronization pass supplies them after proving coverage, pairing,
progress and event leases. Materialization writes them into actual IR; the backend reads that IR.
Guards are ordered conjunctions: a count test must precede any FIRST/LAST
arithmetic that is undefined for an empty loop. Event expressions are evaluated
only inside the guard. Participant identities never clone a Vector body.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdsl.ir import Operation

from ..diagnostics import require
from ..scalar import Expr
from ..target.ascend910b import Actor, Engine, SyncKind, supports_engine


class Placement(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    ENTRY = "entry"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class Comparison:
    lhs: Expr
    predicate: str
    rhs: Expr

    def __post_init__(self):
        require(
            isinstance(self.lhs, Expr)
            and isinstance(self.rhs, Expr)
            and self.predicate in {"==", "!=", "<", "<=", ">", ">="},
            "synchronization guard needs a typed integer comparison",
        )

    def evaluate(self, values):
        lhs, rhs = self.lhs.evaluate(values), self.rhs.evaluate(values)
        return {
            "==": lhs == rhs,
            "!=": lhs != rhs,
            "<": lhs < rhs,
            "<=": lhs <= rhs,
            ">": lhs > rhs,
            ">=": lhs >= rhs,
        }[self.predicate]


@dataclass(frozen=True, slots=True)
class ActionGuard:
    comparisons: tuple[Comparison, ...] = ()

    def __post_init__(self):
        require(
            isinstance(self.comparisons, tuple)
            and all(isinstance(test, Comparison) for test in self.comparisons),
            "synchronization guards must preserve their ordered conjunction",
        )

    def evaluate(self, values):
        return all(test.evaluate(values) for test in self.comparisons)


@dataclass(frozen=True, slots=True)
class SyncAction:
    """One actual target instruction, not an implied completion relation.

    ENTRY/EXIT surround a loop invocation outside its body; for a kernel they
    denote its body start/end. BEFORE/AFTER surround an actual native packet.
    The guard may use only symbols available at that insertion point.
    A cross Set's source and a cross Wait's destination are its actual queue.
    Cross actor visibility/quorum belongs to the selected protocol contract.
    """

    anchor: Operation
    placement: Placement
    participants: tuple[Actor, ...]
    kind: SyncKind
    source: Engine
    destination: Engine
    event: Expr | None = None
    guard: ActionGuard = ActionGuard()
    protocols: tuple[str, ...] = ()

    def __post_init__(self):
        require(isinstance(self.anchor, Operation), "sync placement needs its actual IR anchor")
        require(
            isinstance(self.placement, Placement) and isinstance(self.kind, SyncKind),
            "sync placement and instruction kind must be explicit",
        )
        require(
            isinstance(self.participants, tuple)
            and bool(self.participants)
            and all(isinstance(actor, Actor) for actor in self.participants)
            and len(set(self.participants)) == len(self.participants),
            "sync action needs distinct actual participants",
        )
        require(
            Actor.AIC not in self.participants or len(self.participants) == 1,
            "one sync instruction cannot be both Cube and Vector code",
        )
        require(
            isinstance(self.source, Engine) and isinstance(self.destination, Engine),
            "sync route requires target engines",
        )
        local = self.kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT}
        if local or self.kind is SyncKind.FENCE:
            require(
                all(
                    supports_engine(actor, self.source) and supports_engine(actor, self.destination)
                    for actor in self.participants
                ),
                "local sync route belongs to another participant",
            )
        else:
            engine = self.source if self.kind is SyncKind.CROSS_SET else self.destination
            require(
                all(supports_engine(actor, engine) for actor in self.participants),
                "cross notification uses an engine outside its participant",
            )
        require(not local or self.source is not self.destination, "same-pipe sync needs a fence")
        require(
            (self.event is None and self.source is self.destination)
            if self.kind is SyncKind.FENCE
            else isinstance(self.event, Expr),
            "a fence has one pipe and no ID; notifications require an event expression",
        )
        require(isinstance(self.guard, ActionGuard), "sync action needs a typed guard")
        require(
            isinstance(self.protocols, tuple)
            and all(isinstance(name, str) for name in self.protocols),
            "sync action protocol names are diagnostic origins, not instruction identity",
        )

    def active_event(self, values):
        """Finite test helper; never evaluate guarded IDs on inactive paths."""
        if not self.guard.evaluate(values):
            return None
        return None if self.event is None else self.event.evaluate(values)

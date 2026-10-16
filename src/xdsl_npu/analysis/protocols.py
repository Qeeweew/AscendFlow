"""Selected local capture/acquire relations, before event ID assignment.

The endpoints are actual native insertion anchors. Activation is expressed in
the destination's logical generation, as for physical memory requirements.
Neither a relation nor its selected flag IDs is a completed notification.
"""

from dataclasses import dataclass
from enum import Enum

from ..diagnostics import require
from .hazards import Activation, NativeHazard
from .native import NativeTask


class LocalMechanism(str, Enum):
    EVENT = "event"
    FENCE = "fence"


@dataclass(frozen=True, slots=True)
class LocalProtocol:
    name: str
    source: NativeTask
    destination: NativeTask
    activation: Activation
    mechanism: LocalMechanism
    obligations: tuple[NativeHazard, ...]

    def __post_init__(self):
        require(bool(self.name), "selected protocol needs a local identity")
        require(
            self.source.actor is self.destination.actor,
            "local protocol cannot replace cross-actor visibility",
        )
        require(
            self.source.clock is self.destination.clock,
            "local protocol endpoints need an explicitly shared clock",
        )
        require(isinstance(self.mechanism, LocalMechanism), "invalid local mechanism")
        require(
            (self.source.engine is self.destination.engine)
            == (self.mechanism is LocalMechanism.FENCE),
            "local event needs different engines; same-pipe order needs a fence",
        )
        require(bool(self.obligations), "selected protocol needs actual memory obligations")

    @property
    def distance(self):
        return self.activation.distance

    @property
    def bank(self):
        return self.source.actor, self.source.engine, self.destination.engine


@dataclass(frozen=True, slots=True)
class EventAssignment:
    protocol: LocalProtocol
    ids: tuple[int, ...]

    def __post_init__(self):
        require(
            self.protocol.mechanism is LocalMechanism.EVENT,
            "pipe fences do not use binary event IDs",
        )
        require(
            bool(self.ids)
            and len(set(self.ids)) == len(self.ids)
            and all(type(flag) is int and 0 <= flag < 8 for flag in self.ids),
            "local binary event IDs must be distinct members of the hardware bank",
        )


@dataclass(frozen=True, slots=True)
class LocalCommand:
    index: int
    task: NativeTask
    kind: str
    protocol: LocalProtocol | None = None


def local_commands(tasks, protocols) -> tuple[LocalCommand, ...]:
    """The selected physical dispatch order, with actual before/after anchors."""
    before, after = {}, {}
    for protocol in protocols:
        before.setdefault(protocol.destination.operation, []).append(protocol)
        if protocol.mechanism is LocalMechanism.EVENT:
            after.setdefault(protocol.source.operation, []).append(protocol)
    result = []
    for task in sorted(tasks, key=lambda task: task.lexical):
        for protocol in before.get(task.operation, ()):
            kind = "wait" if protocol.mechanism is LocalMechanism.EVENT else "barrier"
            result.append(LocalCommand(len(result), task, kind, protocol))
        result.append(LocalCommand(len(result), task, "task"))
        for protocol in after.get(task.operation, ()):
            result.append(LocalCommand(len(result), task, "set", protocol))
    return tuple(result)

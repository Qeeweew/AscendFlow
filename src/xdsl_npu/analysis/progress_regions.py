"""Shared immutable contracts for progress on actual invocation regions.

Ports retain their selected notification, case and complete placement binding.
Queue cuts describe admission and blocking, never native completion. These
records and identity helpers perform no composition, rank or condition proof.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Operation

from ..scalar import Expr
from ..target.ascend910b import Actor, Engine
from .event_domains import EventCase
from .structured_protocols import BoundPlacement, LogicalNotification
from .structured_schedule import ActionGuard


@dataclass(frozen=True, slots=True)
class ProgressPort:
    notification: LogicalNotification
    bound: BoundPlacement
    phase: str
    case: EventCase
    bindings: tuple[tuple[Expr, Expr], ...] = ()

    @property
    def actor(self):
        return self.bound.actor

    @property
    def engine(self):
        return (
            self.notification.source_engine
            if self.phase == "set"
            else self.notification.destination_engine
        )


@dataclass(frozen=True, slots=True)
class GuardedTransfer:
    source: int
    destination: int
    steps: int
    guard: ActionGuard
    reason: str
    exact: bool = True
    projections: tuple = ()


@dataclass(frozen=True, slots=True)
class ExcludedCycle:
    relations: tuple[int, ...]
    condition: ActionGuard


@dataclass(frozen=True, slots=True)
class QueueCut:
    """A region admission/blocking interface, never a completion milestone."""

    operation: Operation
    actor: Actor
    engine: Engine | None
    phase: str
    coordinates: tuple[tuple[Operation, Expr], ...]


@dataclass(frozen=True, slots=True)
class InterfaceTransfer:
    source: ProgressPort | QueueCut
    destination: ProgressPort | QueueCut
    guard: ActionGuard
    exact: bool
    projections: tuple


@dataclass(frozen=True, slots=True)
class ChannelProjection:
    operation: Operation
    ports: tuple[ProgressPort | QueueCut, ...]
    relations: tuple[GuardedTransfer, ...]
    windows: tuple


@dataclass(frozen=True, slots=True)
class InvocationBinding:
    operation: Operation
    coordinates: tuple[tuple[Operation, Expr], ...]
    substitutions: tuple[tuple[Expr, Expr], ...]
    condition: ActionGuard


@dataclass(frozen=True, slots=True)
class GuardedRank:
    """A local SCC rank under a whole-kernel uniform parameter condition."""

    condition: ActionGuard
    nodes: tuple[int, ...]
    relations: tuple[int, ...]
    stride: int
    rank: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StreamProgress:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    cuts: tuple[QueueCut, ...]
    relations: tuple[GuardedTransfer, ...]
    stride: int
    rank: tuple[int, ...]
    conditional_ranks: tuple[GuardedRank, ...] = ()

    @property
    def nodes(self):
        return self.ports + self.cuts


@dataclass(frozen=True, slots=True)
class RegionInterface:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    cuts: tuple[QueueCut, ...]
    relations: tuple[InterfaceTransfer, ...]


@dataclass(frozen=True, slots=True)
class ControllerEntries:
    actor: Actor
    calls: tuple[tuple[Operation, int, int], ...]


@dataclass(frozen=True, slots=True)
class RegionProgress:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    relations: tuple[GuardedTransfer, ...]
    interface: RegionInterface
    children: tuple[RegionProgress, ...]
    stream: StreamProgress | None
    excluded_cycles: tuple[ExcludedCycle, ...] = ()
    controller_entries: tuple[ControllerEntries, ...] = ()
    conditional_ranks: tuple[GuardedRank, ...] = ()


def port_key(port):
    if isinstance(port, QueueCut):
        return (port.operation, port.actor, port.engine, port.phase, port.coordinates)
    return (
        id(port.notification),
        id(port.case),
        port.phase,
        port.actor,
        port.bound.operation,
        port.bound.coordinates,
        port.bindings,
    )


def binding_key(port):
    """The complete actual placement and its containing proof binding."""
    if isinstance(port, QueueCut):
        return (QueueCut, port)
    return (ProgressPort, id(port.notification), port.case, port.phase, port.bound, port.bindings)


def is_wait(node):
    return node.phase in {"wait", "output"}


def is_source(node):
    return node.phase in {"set", "input"}


def port_label(node):
    if isinstance(node, ProgressPort):
        return f"{node.notification.name}:{node.phase}"
    engine = "controller" if node.engine is None else node.engine.value
    return f"{node.actor.value}:{engine}:{node.phase}"

"""Bind periodic memory obligations to actual native cuts and existence domains.

This adapter consumes one admitted loop and its already computed dependencies.
It neither reruns access analysis nor invokes a completion planner. Runtime N
is an input of the activation predicate, never an analysis iteration bound.
The existing periodic access scan is transitional; it is not a nested-region
summary and must not be used to expand a child completion graph.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.ir import SSAValue

from ..diagnostics import require
from ..target.ascend910b import Milestone
from .frontier import Frontier, KnownPrefix, Partition, PrefixDomain
from .native import NativeCut, NativeTask

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..loop_model import Dependence, Loop


class HazardKind(str, Enum):
    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"


@dataclass(frozen=True, slots=True)
class Activation:
    """The exact domain of both endpoints, in consumer logical generation g.

    A Set exists precisely when its matching consumer exists. In particular a
    distance-d release has no fictional negative source and no trailing token
    for a consumer beyond N. These predicates do not establish event capacity
    or make a notification's completion payload available.
    """

    distance: int
    period: int
    residues: frozenset[int]
    consumer_last: bool = False
    source_last: bool = False

    def __post_init__(self):
        require(
            type(self.distance) is int and self.distance >= 0,
            "hazard distance must be a nonnegative integer",
        )
        require(
            type(self.period) is int and self.period > 0, "hazard needs a positive storage period"
        )
        residues = frozenset(self.residues)
        require(
            bool(residues) and all(type(r) is int and 0 <= r < self.period for r in residues),
            "hazard residues must belong to their actual storage period",
        )
        require(
            type(self.consumer_last) is bool and type(self.source_last) is bool,
            "hazard endpoint existence must be explicit",
        )
        object.__setattr__(self, "residues", residues)

    def consumes(self, generation: int, count: int) -> bool:
        """Finite evaluator for tests/materialization, not a planning loop."""
        require(
            type(generation) is int and type(count) is int and count >= 0,
            "activation needs an integer generation and nonnegative count",
        )
        return (
            self.distance <= generation < count
            and generation % self.period in self.residues
            and (not self.consumer_last or generation == count - 1)
            and (not self.source_last or generation - self.distance == count - 1)
        )

    def produces(self, generation: int, count: int) -> bool:
        """Same pair's domain, expressed in producer logical generation."""
        require(type(generation) is int, "activation needs an integer generation")
        return self.consumes(generation + self.distance, count)


@dataclass(frozen=True, slots=True)
class MemoryWitness:
    root: SSAValue
    name: str
    start: int
    stop: int
    frame_stride: int

    def __post_init__(self):
        require(isinstance(self.root, SSAValue), "memory witness needs the actual root SSA")
        require(
            0 <= self.start < self.stop and self.frame_stride >= 0,
            "invalid periodic memory witness",
        )


@dataclass(frozen=True, slots=True)
class NativeHazard:
    producer: NativeTask
    consumer: NativeTask
    kind: HazardKind
    activation: Activation
    memory: MemoryWitness
    reader_prefix: bool = False

    @property
    def source_cut(self) -> NativeCut:
        """Producer end at g-distance, expressed in consumer logical g."""
        return self.producer.cut(Milestone.NATIVE_END, generation_offset=-self.activation.distance)

    @property
    def consumer_cut(self) -> NativeCut:
        return self.consumer.consumer_cut

    def physical_cuts(self) -> tuple[NativeCut, NativeCut]:
        """Bind g=j-consumer.stage once, for a scan in physical step j."""
        shift = -self.consumer.stage
        return (
            self.producer.cut(
                Milestone.NATIVE_END, generation_offset=shift - self.activation.distance
            ),
            self.consumer.cut(Milestone.ADMISSION, generation_offset=shift),
        )

    def completion_prefixes(self, partition: Partition, *, physical_step=False):
        """Alternative sufficient witnesses, conditional on this activation.

        An ordinary producer end is not a cumulative prefix. A retirement
        capture may discharge any memory hazard. A filtered implicit prefix
        may discharge only a producer admitted into that same filtered chain;
        a later large MAD cannot thereby discharge an earlier short MAD.
        The caller must establish the activation/partition before querying.
        """
        cut = self.physical_cuts()[0] if physical_step else self.source_cut
        kinds = (Milestone.RETIREMENT_CAPTURE,)
        if self.producer.implicit_completion:
            kinds += (Milestone.IMPLICIT_RETIREMENT,)
        return tuple(
            KnownPrefix(
                PrefixDomain(cut.actor, cut.engine, kind, cut.rank.clock, partition), cut.rank
            )
            for kind in kinds
        )

    def covered_by(self, state: Frontier, partition: Partition, *, physical_step=False):
        """One typed lookup per admissible witness; no hidden graph traversal."""
        return any(
            state.covers(prefix)
            for prefix in self.completion_prefixes(partition, physical_step=physical_step)
        )


def bind_hazards(
    loop: "Loop",
    tasks: tuple[NativeTask, ...],
    dependencies: "Iterable[Dependence]",
) -> tuple[NativeHazard, ...]:
    """Bind each existing access obligation once, retaining consumer identity.

    Work is O(A+H), excluding the original access scan and stored residue
    lists. Names only join the legacy scan's keys to already-owned actual SSA
    roots. They never establish non-aliasing. Separate physical fragments and
    consumers are retained; a later reader cannot erase an earlier Wait cut.
    """
    clock = tasks[0].clock if tasks else None
    native, roots = {}, {}
    for task in tasks:
        require(task.clock is clock, "hazard endpoints need one explicitly bound loop clock")
        key = task.actor.value, task.operation.uid.data
        require(key not in native, "hazard binding repeats a native task identity")
        native[key] = task
        for access in task.accesses:
            name = loop.root(task.actor.value, access.ref)
            old = roots.setdefault((key, name), access.root)
            require(old is access.root, "physical storage name aliases distinct root SSA values")
    result = []
    for dependence in dependencies:
        require(
            dependence.source in native and dependence.destination in native,
            "hazard endpoint is not one of the admitted native tasks",
        )
        producer, consumer = native[dependence.source], native[dependence.destination]
        source_root = roots.get((dependence.source, dependence.storage))
        consumer_root = roots.get((dependence.destination, dependence.storage))
        require(
            source_root is not None and source_root is consumer_root,
            "hazard endpoints do not access the same actual allocation",
        )
        require(
            dependence.last == consumer.last, "hazard activation lost the actual terminal consumer"
        )
        result.append(
            NativeHazard(
                producer,
                consumer,
                HazardKind(dependence.hazard),
                Activation(
                    dependence.distance,
                    dependence.period,
                    frozenset(dependence.residues),
                    consumer.last,
                    producer.last,
                ),
                MemoryWitness(
                    source_root,
                    dependence.storage,
                    dependence.start,
                    dependence.stop,
                    loop.frames.get(dependence.storage, 0),
                ),
                dependence.prefix,
            )
        )
    return tuple(result)

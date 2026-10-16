"""Direct dav-2201 instruction rules on typed completion-state ports.

Ports describe facts at distinct hardware milestones. In particular the
retirement port after a task is a conditional capture state: its own native
end AND the previous retirement state. Dispatching that task does not make
this state available at the controller or the next native's admission.

The caller admits the native footprint/engine and supplies an already-bound
physical-order rank. These rules do not add the TaskOp's stage a second time,
bind another clock, infer an event match, or search a completion graph.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from ..diagnostics import ErrorKind, require
from ..dialects.physical import TaskOp
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine
from ..target.native import implicit_completion
from .frontier import (
    Capture,
    Clock,
    FrontierLayout,
    Handoff,
    KnownPrefix,
    Partition,
    PrefixDomain,
    PrefixRank,
)
from .regions import BoundaryLayout, BoundaryTransfer, Output, Port

DEFAULT_PARTITION = Partition()


class InstructionKind(str, Enum):
    NATIVE = "native"
    SET = "set"
    WAIT = "wait"
    CROSS_WAIT = "cross_wait"
    BARRIER = "barrier"


@dataclass(frozen=True, slots=True)
class InstructionSite:
    """An actual native anchor with an explicitly substituted clock/rank.

    A native uses this anchor itself; Set/barrier are after this source native
    and Wait is before this consumer native. The anchor's end is not supplied
    as a completion fact. Event placement and activation remain obligations of
    the owner that materializes this selected instruction.
    """

    operation: TaskOp
    actor: Actor
    engine: Engine
    rank: PrefixRank
    partition: Partition = DEFAULT_PARTITION

    def __post_init__(self):
        require(isinstance(self.operation, TaskOp), "instruction site needs an actual TaskOp")
        require(supports_engine(self.actor, self.engine), "instruction site has an invalid engine")
        require(isinstance(self.rank, PrefixRank), "instruction site needs a bound prefix rank")
        require(isinstance(self.partition, Partition), "instruction site needs a typed partition")

    @property
    def clock(self):
        return self.rank.clock


def transition_layout(
    engines: Mapping[Actor, Iterable[Engine]],
    clock: Clock,
    *,
    partition: Partition = DEFAULT_PARTITION,
) -> BoundaryLayout:
    """Declare only the actual actors/pipelines, without period products."""
    require(isinstance(clock, Clock), "native layout needs a declared clock")
    require(isinstance(partition, Partition), "native layout needs a typed partition")
    ports, domains = set(), []
    for actor, members in engines.items():
        require(isinstance(actor, Actor), "native layout needs typed actors")
        members = tuple(members)
        require(len(set(members)) == len(members), "native layout repeats an engine")
        actor_ports = {Port(actor, None, Milestone.ISSUE)}
        for engine in members:
            require(supports_engine(actor, engine), "native layout has an invalid actor engine")
            actor_ports.update(
                Port(actor, engine, kind)
                for kind in (
                    Milestone.ADMISSION,
                    Milestone.RETIREMENT_CAPTURE,
                    Milestone.IMPLICIT_RETIREMENT,
                )
            )
        for port in sorted(
            actor_ports,
            key=lambda item: (
                "" if item.engine is None else item.engine.value,
                item.milestone.value,
            ),
        ):
            domains.append(PrefixDomain(actor, port.engine, port.milestone, clock, partition))
        ports.update(actor_ports)
    return BoundaryLayout(frozenset(ports), FrontierLayout(tuple(domains)))


def _context(layout, site):
    require(isinstance(layout, BoundaryLayout), "instruction needs a fixed boundary layout")
    require(isinstance(site, InstructionSite), "instruction needs a typed actual site")
    ports, prefixes = {}, {}
    for kind in (
        Milestone.ISSUE,
        Milestone.ADMISSION,
        Milestone.RETIREMENT_CAPTURE,
        Milestone.IMPLICIT_RETIREMENT,
    ):
        engine = None if kind is Milestone.ISSUE else site.engine
        port = Port(site.actor, engine, kind)
        domain = PrefixDomain(site.actor, engine, kind, site.clock, site.partition)
        require(
            port in layout.ports and domain in layout.frontier._keys,
            "instruction uses an undeclared port/clock/partition",
            kind=ErrorKind.UNSUPPORTED,
            phase="native-transition",
        )
        ports[kind] = port
        prefixes[kind] = KnownPrefix(domain, site.rank)
    return ports, prefixes


def _handoff(site, handoff):
    require(isinstance(handoff, Handoff), "notification needs an explicit handoff identity")
    require(
        handoff.clock is site.clock,
        "notification clock is not bound to its actual instruction site",
        kind=ErrorKind.UNSUPPORTED,
        phase="native-transition",
    )
    require(
        handoff.partition.guard == site.partition.guard,
        "notification and native site need the same bound guard partition",
        kind=ErrorKind.UNSUPPORTED,
        phase="native-transition",
    )


@dataclass(frozen=True, slots=True)
class InstructionTransition:
    site: InstructionSite
    kind: InstructionKind
    boundary: BoundaryTransfer
    admission: Output
    retirement_capture: Output | None = None
    effect: Output | None = None
    handoff: Handoff | None = None

    def _at(self, point, inputs):
        self.boundary._inputs(inputs)
        return point.guarantees.acquire(*(inputs[source] for source in point.sources))

    def admission_state(self, inputs):
        """Facts at this actual begin, rather than every future output port."""
        return self._at(self.admission, inputs)

    def notification(self, inputs) -> Capture:
        """A real selected Set effect captures the prior retirement state."""
        require(
            self.kind is InstructionKind.SET and self.effect is not None,
            "only a real Set transition supplies a notification capture",
        )
        return self._at(self.effect, inputs).capture(
            self.site.actor, self.site.engine, handoff=self.handoff
        )


@dataclass(frozen=True, slots=True)
class _NativeAttrs:
    """Only the attributes consumed by the shared target implicit rule."""

    pipe: str
    native: str
    params: tuple[int, ...]


def native_transfer(layout: BoundaryLayout, site: InstructionSite) -> InstructionTransition:
    """Apply the owned native rule, preserving end/capture distinctions."""
    ports, prefixes = _context(layout, site)
    issue, queue, retire, implicit = (
        ports[kind]
        for kind in (
            Milestone.ISSUE,
            Milestone.ADMISSION,
            Milestone.RETIREMENT_CAPTURE,
            Milestone.IMPLICIT_RETIREMENT,
        )
    )
    empty = layout.frontier.empty()
    rows = dict(BoundaryTransfer.identity(layout).outputs)
    begin = Output(frozenset({issue, queue, implicit}), empty.record(prefixes[Milestone.ADMISSION]))
    # This is the future native-end + prior-retirement conjunction, not an
    # assertion that ordinary native end is FIFO or already observed on issue.
    capture = Output(
        begin.sources | {retire}, begin.guarantees.record(prefixes[Milestone.RETIREMENT_CAPTURE])
    )
    rows[queue], rows[retire] = begin, capture
    native = site.operation.native.data
    params = tuple(value.data for value in site.operation.params)
    require(
        site.engine is not Engine.M or native == "mad16" or len(params) >= 2,
        "Cube transition needs admitted native shape parameters",
    )
    if implicit_completion(_NativeAttrs(site.engine.value, native, params)):
        # Filtered qualified-native ends only. A large MAD end cannot absorb a
        # previous short MAD's retirement state by virtue of sharing PIPE_M.
        completed = Output(
            begin.sources, begin.guarantees.record(prefixes[Milestone.IMPLICIT_RETIREMENT])
        )
        rows[implicit] = completed
        if site.engine is Engine.S:
            rows[issue] = Output(
                completed.sources, completed.guarantees.record(prefixes[Milestone.ISSUE])
            )
    return InstructionTransition(
        site,
        InstructionKind.NATIVE,
        BoundaryTransfer(layout, rows),
        begin,
        retirement_capture=capture,
    )


def set_transfer(
    layout: BoundaryLayout,
    site: InstructionSite,
    handoff: Handoff,
) -> InstructionTransition:
    """Select Set after a real source native; its effect never feeds a prefix."""
    ports, prefixes = _context(layout, site)
    _handoff(site, handoff)
    issue, queue, retire = (
        ports[kind]
        for kind in (
            Milestone.ISSUE,
            Milestone.ADMISSION,
            Milestone.RETIREMENT_CAPTURE,
        )
    )
    empty = layout.frontier.empty()
    # The anchor native has begun before this Set. The Set's effect has not.
    begin = Output(frozenset({issue, queue}), empty.record(prefixes[Milestone.ADMISSION]))
    effect = Output(
        begin.sources | {retire}, begin.guarantees.record(prefixes[Milestone.RETIREMENT_CAPTURE])
    )
    rows = dict(BoundaryTransfer.identity(layout).outputs)
    rows[queue] = begin
    return InstructionTransition(
        site,
        InstructionKind.SET,
        BoundaryTransfer(layout, rows),
        begin,
        effect=effect,
        handoff=handoff,
    )


def wait_transfer(
    layout: BoundaryLayout,
    site: InstructionSite,
    capture: Capture,
    expected: Handoff,
    *,
    cross: bool = False,
) -> InstructionTransition:
    """Acquire at Wait admission, before the consumer native actually begins."""
    ports, _ = _context(layout, site)
    _handoff(site, expected)
    require(type(cross) is bool, "cross-wait selection must be a boolean")
    require(isinstance(capture, Capture), "Wait needs an actual notification capture")
    require(
        capture.producer is not site.actor if cross else capture.producer is site.actor,
        "Wait mechanism does not match the actual source and destination actors",
    )
    require(
        cross or capture.engine is not site.engine,
        "local Set/Wait needs different engines; use the pipe-fence contract",
    )
    acquired = layout.frontier.empty().acquire(capture, expected=expected)
    issue, queue = ports[Milestone.ISSUE], ports[Milestone.ADMISSION]
    begin = Output(frozenset({issue, queue}), acquired)
    rows = dict(BoundaryTransfer.identity(layout).outputs)
    rows[queue] = begin
    if cross or site.engine is Engine.S:
        rows[issue] = begin
    # No current-consumer ADMISSION/ISSUE/RETIREMENT rank is invented here:
    # only this Wait has admitted. The consumer's own rule establishes begin.
    return InstructionTransition(
        site,
        InstructionKind.CROSS_WAIT if cross else InstructionKind.WAIT,
        BoundaryTransfer(layout, rows),
        begin,
    )


def barrier_transfer(layout: BoundaryLayout, site: InstructionSite) -> InstructionTransition:
    """A pipe fence after a source native acquires its cumulative retirement."""
    ports, prefixes = _context(layout, site)
    issue, queue, retire = (
        ports[kind]
        for kind in (
            Milestone.ISSUE,
            Milestone.ADMISSION,
            Milestone.RETIREMENT_CAPTURE,
        )
    )
    begin = Output(
        frozenset({issue, queue, retire}),
        layout.frontier.empty().record(
            prefixes[Milestone.ADMISSION], prefixes[Milestone.RETIREMENT_CAPTURE]
        ),
    )
    rows = dict(BoundaryTransfer.identity(layout).outputs)
    rows[queue] = begin
    return InstructionTransition(
        site, InstructionKind.BARRIER, BoundaryTransfer(layout, rows), begin
    )

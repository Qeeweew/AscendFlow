"""Bounded knowledge of certified cumulative completion prefixes.

M1 supports unconditional facts, or facts in exactly the same declared
slot/owner/guard partition. A rank is ``(g + offset, lexical)`` in one declared
prefix-order clock. The native rule must certify that this clock orders the
particular cumulative prefix; a task's logical generation alone does not do
so. Different clocks need an explicit binding outside this module.

For K declared domains the state is a product of K chains with a bottom
element. Acquire is pointwise max; branch must-merge is pointwise min with a
missing entry equal to bottom. Thus each state holds at most K ranks, joins
are idempotent, and no operation creates domains, guards, paths or origins.
Individual ends remain local premises of a real capture rule. Neither those
ends nor the notification's own effect enter this state.
Handoff identities check actual notification pairing, but never become state
keys or retained notification history.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ..diagnostics import CompileError, ErrorKind
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine


class FrontierError(CompileError):
    """An unsupported or inconsistent completion-frontier contract."""


def _require(condition, message, *, kind=ErrorKind.INVALID):
    if not condition:
        raise FrontierError(message, kind=kind, phase="completion frontier")


@dataclass(frozen=True, slots=True, eq=False)
class Clock:
    """A declared coordinate with object identity; its name is diagnostic only.

    Reusing a spelling does not bind two loop/SSA declarations. Every user of
    one clock must retain the same object until an explicit binding is made.
    """

    name: str

    def __post_init__(self):
        _require(type(self.name) is str and bool(self.name), "frontier clock needs an identity")


@dataclass(frozen=True, slots=True)
class Partition:
    """Opaque, predeclared owner/slot/guard identity; no predicate inference."""

    owner: str = "local"
    slot: str | None = None
    guard: str = "all"

    def __post_init__(self):
        _require(type(self.owner) is str and bool(self.owner), "invalid frontier owner partition")
        _require(
            self.slot is None or type(self.slot) is str and bool(self.slot),
            "invalid frontier slot partition",
        )
        _require(type(self.guard) is str and bool(self.guard), "invalid frontier guard partition")


@dataclass(frozen=True, slots=True)
class Handoff:
    """One protocol token family in an explicitly bound clock/owner partition.

    ``ordinal`` means ``g + ordinal`` in this clock. M1 neither equates
    different clocks nor computes a modulo, guard implication or token match.
    The partition names the shared logical handoff, not the producing actor.
    """

    protocol: str
    clock: Clock
    ordinal: int = 0
    partition: Partition = Partition()

    def __post_init__(self):
        _require(
            type(self.protocol) is str and bool(self.protocol),
            "handoff needs an actual protocol identity",
        )
        _require(isinstance(self.clock, Clock), "handoff needs a declared clock")
        _require(type(self.ordinal) is int, "handoff ordinal offset must be an integer")
        _require(isinstance(self.partition, Partition), "handoff needs a typed owner partition")


@dataclass(frozen=True, slots=True)
class PrefixDomain:
    actor: Actor
    engine: Engine | None
    milestone: Milestone
    clock: Clock
    partition: Partition = Partition()

    def __post_init__(self):
        _require(isinstance(self.actor, Actor), "frontier domain needs a typed actor")
        _require(isinstance(self.milestone, Milestone), "frontier domain needs a typed milestone")
        _require(
            self.milestone.cumulative,
            "native_end and set_effect are not cumulative prefixes; "
            "establish a real retirement capture explicitly",
            kind=ErrorKind.UNSUPPORTED,
        )
        if self.milestone is Milestone.ISSUE:
            _require(self.engine is None, "actor issue prefix has no pipeline engine")
        else:
            _require(
                supports_engine(self.actor, self.engine),
                "frontier engine does not belong to the declared actor",
            )
        _require(isinstance(self.clock, Clock), "frontier domain needs a declared clock")
        _require(isinstance(self.partition, Partition), "frontier domain needs a typed partition")


@dataclass(frozen=True, slots=True)
class PrefixRank:
    clock: Clock
    offset: int = 0
    lexical: int = 0

    def __post_init__(self):
        _require(isinstance(self.clock, Clock), "frontier rank needs a declared clock")
        _require(type(self.offset) is int, "frontier generation offset must be an integer")
        _require(
            type(self.lexical) is int and self.lexical >= 0,
            "frontier lexical rank must be a nonnegative integer",
        )

    def compare(self, other: "PrefixRank") -> int:
        _require(isinstance(other, PrefixRank), "frontier comparison needs a prefix rank")
        _require(
            self.clock == other.clock,
            "unsupported comparison across frontier clocks; bind clocks explicitly",
            kind=ErrorKind.UNSUPPORTED,
        )
        first, last = (self.offset, self.lexical), (other.offset, other.lexical)
        return (first > last) - (first < last)


@dataclass(frozen=True, slots=True)
class KnownPrefix:
    """A cumulative fact already established by a trusted native/protocol rule."""

    domain: PrefixDomain
    rank: PrefixRank

    def __post_init__(self):
        _require(
            isinstance(self.domain, PrefixDomain) and isinstance(self.rank, PrefixRank),
            "known prefix needs a typed domain and rank",
        )
        _require(
            self.domain.clock == self.rank.clock,
            "unsupported prefix clock binding; bind clocks explicitly",
            kind=ErrorKind.UNSUPPORTED,
        )


@dataclass(frozen=True, slots=True)
class FrontierLayout:
    """The fixed key universe, derived from effects and boundary partitions."""

    domains: tuple[PrefixDomain, ...] = field(compare=False)
    _keys: frozenset[PrefixDomain] = field(init=False, repr=False)

    def __post_init__(self):
        domains = tuple(self.domains)
        _require(
            all(isinstance(domain, PrefixDomain) for domain in domains),
            "frontier layout needs typed prefix domains",
        )
        keys = frozenset(domains)
        _require(len(domains) == len(keys), "frontier layout contains a duplicate domain")
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "_keys", keys)

    @property
    def capacity(self):
        return len(self.domains)

    def empty(self):
        return Frontier(self)


@dataclass(slots=True)
class FrontierWork:
    """Optional per-operation work accounting, never part of semantic state."""

    entries: int = 0
    comparisons: int = 0
    updates: int = 0
    copied_entries: int = 0


@dataclass(frozen=True, slots=True, init=False)
class Frontier:
    layout: FrontierLayout
    known: Mapping[PrefixDomain, PrefixRank]

    def __init__(self, layout: FrontierLayout, prefixes: Iterable[KnownPrefix] = ()):
        _require(isinstance(layout, FrontierLayout), "frontier needs a fixed declared layout")
        known = {}
        for prefix in prefixes:
            self._check_prefix(layout, prefix)
            previous = known.get(prefix.domain)
            if previous is None or prefix.rank.compare(previous) > 0:
                known[prefix.domain] = prefix.rank
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "known", MappingProxyType(known))

    @staticmethod
    def _check_prefix(layout, prefix):
        _require(isinstance(prefix, KnownPrefix), "frontier only accepts certified prefixes")
        _require(
            prefix.domain in layout._keys,
            "undeclared frontier domain/clock/partition; extend the input layout explicitly",
            kind=ErrorKind.UNSUPPORTED,
        )

    @classmethod
    def _from_known(cls, layout, known):
        value = object.__new__(cls)
        object.__setattr__(value, "layout", layout)
        object.__setattr__(value, "known", MappingProxyType(known))
        return value

    def __len__(self):
        return len(self.known)

    def _same_layout(self, other):
        _require(isinstance(other, Frontier), "acquire needs a frontier or capture")
        _require(
            self.layout is other.layout or self.layout == other.layout,
            "frontier composition needs the same fixed layout",
            kind=ErrorKind.UNSUPPORTED,
        )

    def covers(self, required: KnownPrefix) -> bool:
        self._check_prefix(self.layout, required)
        known = self.known.get(required.domain)
        return known is not None and known.compare(required.rank) >= 0

    def record(self, *prefixes: KnownPrefix, work: FrontierWork | None = None) -> "Frontier":
        """Record witnessed prefixes; this does not infer a fact from native end."""
        for prefix in prefixes:
            self._check_prefix(self.layout, prefix)
        return self._acquire_entries(
            ((prefix.domain, prefix.rank) for prefix in prefixes), work=work
        )

    def rebase(self, clock: Clock, offset: int, *, work: FrontierWork | None = None) -> "Frontier":
        """Substitute ``g_old = g_new + offset`` in one declared clock.

        This is coordinate substitution, not execution progress. Domain keys,
        owner/slot/guard partitions, lexical ranks and other clocks are kept
        exactly. No historical snapshot is reconstructed or token inferred.
        """
        _require(isinstance(clock, Clock), "frontier rebasing needs a declared clock")
        _require(type(offset) is int, "frontier rebasing offset must be an integer")
        if offset == 0:
            return self
        known = self.known
        changed = False
        for domain, rank in self.known.items():
            if work is not None:
                work.entries += 1
            if domain.clock is not clock:
                continue
            if not changed:
                if work is not None:
                    work.copied_entries += len(known)
                known = dict(known)
                changed = True
            known[domain] = PrefixRank(clock, rank.offset + offset, rank.lexical)
            if work is not None:
                work.updates += 1
        return self._from_known(self.layout, known) if changed else self

    def _acquire_entries(self, entries, *, work):
        known = self.known
        changed = False
        for domain, rank in entries:
            if work is not None:
                work.entries += 1
            previous = known.get(domain)
            if previous is not None:
                if work is not None:
                    work.comparisons += 1
                if rank.compare(previous) <= 0:
                    continue
            if not changed:
                if work is not None:
                    work.copied_entries += len(known)
                known = dict(known)
                changed = True
            known[domain] = rank
            if work is not None:
                work.updates += 1
        return self._from_known(self.layout, known) if changed else self

    def acquire(
        self,
        *sources: "Frontier | Capture",
        expected: Handoff | None = None,
        work: FrontierWork | None = None,
    ):
        """Join facts actually acquired on this path, with pointwise max.

        Batch independent contributions in one call: each incoming active key
        is read once and the receiver is copied at most once. Alternative
        witnesses for one prefix do not create another semantic state entry.
        Protocol captures require an explicit matching handoff. A local native
        snapshot without a handoff is not a protocol token.
        """
        _require(
            expected is None or isinstance(expected, Handoff),
            "acquire needs a typed expected handoff",
        )
        for source in sources:
            if isinstance(source, Capture):
                if source.handoff is not None:
                    _require(expected is not None, "paired capture requires an expected handoff")
                _require(source.handoff == expected, "capture does not match the expected handoff")
        frontiers = tuple(
            source.frontier if isinstance(source, Capture) else source for source in sources
        )
        for frontier in frontiers:
            self._same_layout(frontier)
        return self._acquire_entries(
            (item for frontier in frontiers for item in frontier.known.items()), work=work
        )

    @classmethod
    def must_merge(cls, branches: Iterable["Frontier"], *, work: FrontierWork | None = None):
        """Keep exactly the cumulative facts guaranteed on every actual path."""
        branches = tuple(branches)
        _require(bool(branches), "branch must-merge needs at least one actual path")
        first = branches[0]
        _require(isinstance(first, Frontier), "branch must-merge needs frontiers")
        for branch in branches[1:]:
            first._same_layout(branch)
        smallest = min(branches, key=len)
        known = {}
        for domain, rank in smallest.known.items():
            for branch in branches:
                if work is not None:
                    work.entries += 1
                value = branch.known.get(domain)
                if value is None:
                    break
                if work is not None:
                    work.comparisons += 1
                if value.compare(rank) < 0:
                    rank = value
            else:
                known[domain] = rank
        if known == first.known:
            return first
        if work is not None:
            work.updates += len(known)
        return cls._from_known(first.layout, known)

    def transparent(self):
        """A skipped/empty region inherits knowledge without completing work."""
        return self

    def capture(
        self,
        producer: Actor,
        engine: Engine,
        *,
        handoff: Handoff | None = None,
    ) -> "Capture":
        """Snapshot already established facts; Set adds no prefix by itself."""
        return Capture(producer, engine, self, handoff)

    def acquire_quorum(
        self,
        required: Iterable[Actor],
        captures: Iterable["Capture"],
        *,
        expected: Handoff,
        work: FrontierWork | None = None,
    ):
        """Acquire a complete AND quorum for exactly one expected handoff."""
        _require(isinstance(expected, Handoff), "quorum needs an explicit expected handoff")
        required = tuple(required)
        _require(
            bool(required) and all(isinstance(actor, Actor) for actor in required),
            "quorum needs declared target actors",
        )
        _require(len(set(required)) == len(required), "quorum repeats a required contributor")
        received = {}
        for capture in captures:
            _require(isinstance(capture, Capture), "quorum needs actual capture contributions")
            _require(
                capture.handoff == expected, "quorum capture does not match the expected handoff"
            )
            _require(capture.producer in required, "quorum has an unexpected contributor")
            _require(capture.producer not in received, "quorum repeats a contributor")
            received[capture.producer] = capture
        missing = set(required) - received.keys()
        _require(
            not missing,
            "incomplete AND quorum; missing " + ", ".join(sorted(actor.value for actor in missing)),
        )
        return self.acquire(*(received[actor] for actor in required), expected=expected, work=work)


@dataclass(frozen=True, slots=True)
class Capture:
    """An immutable native snapshot or an explicitly identified notification.

    ``handoff=None`` is a local snapshot, usable as prefix evidence but not as
    a quorum contribution. Neither form stores individual native endpoints.
    """

    producer: Actor
    engine: Engine
    frontier: Frontier
    handoff: Handoff | None = None

    def __post_init__(self):
        _require(
            supports_engine(self.producer, self.engine),
            "capture needs a typed actor and its actual source engine",
        )
        _require(isinstance(self.frontier, Frontier), "capture needs established frontier facts")
        _require(
            self.handoff is None or isinstance(self.handoff, Handoff),
            "capture needs a typed handoff identity",
        )

    def rebase(self, clock: Clock, offset: int, *, work: FrontierWork | None = None) -> "Capture":
        """Re-express this existing snapshot and its ordinal in one clock.

        Producer, engine and handoff partition retain their identities. A
        subsequent acquire must still supply the exact rebased handoff; this
        operation neither finds a matching Wait nor replays the producer.
        """
        frontier = self.frontier.rebase(clock, offset, work=work)
        handoff = self.handoff
        if offset != 0 and handoff is not None and handoff.clock is clock:
            handoff = replace(handoff, ordinal=handoff.ordinal + offset)
        if frontier is self.frontier and handoff is self.handoff:
            return self
        return Capture(self.producer, self.engine, frontier, handoff)

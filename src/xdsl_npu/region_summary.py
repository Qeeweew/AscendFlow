"""Exact access frontiers at the boundary of a repeated leaf pipeline.

These are memory obligations, not completion certificates or event plans. The
caller selects stationary storage roots that survive a child invocation. GM
frames require a parent ownership proof and are deliberately not accepted here.

For period P, enumerate N=1..P-1 and P long domains N>=P, N%P=r. A periodic
writer leaves a suffix of at most P generations. With no periodic writer, a
read family remains a span: its final member is not a whole-prefix retirement.
No domain enumerates runtime N, nor is a finite sample used as an induction.
"""

from bisect import bisect_left
from dataclasses import dataclass
from math import gcd

from xdsl.ir import SSAValue

from .ir import BufferOp
from .loop_model import access_cells, slope
from .model import require
from .scalar import Expr

MAX_SUMMARY_VISITS = 1_000_000


@dataclass(frozen=True)
class CountDomain:
    period: int
    residue: int
    exact: int | None = None
    minimum: int | None = None

    def contains(self, count):
        return (
            count == self.exact
            if self.exact is not None
            else (
                count >= (self.period if self.minimum is None else self.minimum)
                and count % self.period == self.residue
            )
        )


@dataclass(frozen=True)
class Position:
    """A generation relative to this child invocation, never to its parent."""

    end: bool
    offset: int

    def evaluate(self, count):
        return count - 1 - self.offset if self.end else self.offset


@dataclass(frozen=True)
class Access:
    task: tuple[str, str]
    position: Position


@dataclass(frozen=True)
class ReadSpan:
    task: tuple[str, str]
    first: Position
    last: Position
    step: int

    @classmethod
    def one(cls, access):
        return cls(access.task, access.position, access.position, 1)


@dataclass(frozen=True)
class CellSummary:
    storage: str
    start: int
    stop: int
    domain: CountDomain
    reads_before_write: tuple[ReadSpan, ...]
    first_write: Access | None
    last_write: Access | None
    pending_reads: tuple[ReadSpan, ...]


@dataclass(frozen=True)
class InvocationDependence:
    """All source accesses in invocation p precede consumers in p+1.

    A ReadSpan source requires every member's retirement. It must not be
    rewritten to just last.position without a separate completion proof.
    """

    storage: str
    start: int
    stop: int
    domain: CountDomain
    hazard: str
    source: Access | ReadSpan
    destination: Access | ReadSpan


@dataclass(frozen=True)
class EventReuse:
    """Old invocation's Wait consumption precedes new invocation's Set effect.

    These endpoint tasks identify sync insertion sites, not native completion.
    In particular the old Wait does not retire its consumer's buffer reads.
    """

    channel: str
    event_id: int
    bank: tuple[str, str, str]
    domain: CountDomain
    consumed: Access
    republished: Access


def _event_reuse(channel, bank):
    require(channel.kind == "local", "hierarchical counters require participant analysis")
    require(channel.ids, "child event IDs must be assigned before resource summarization")
    (source,) = channel.sources
    (destination,) = channel.destinations
    if channel.last:
        require(len(channel.ids) == 1, "a terminal child channel needs one event ID")
        for residue in channel.residues:
            yield EventReuse(
                channel.name,
                channel.ids[0],
                bank,
                CountDomain(
                    channel.period,
                    (residue + channel.distance + 1) % channel.period,
                    minimum=channel.distance + 1,
                ),
                Access(destination, Position(True, 0)),
                Access(source, Position(True, channel.distance)),
            )
        return
    # Event-ID rotation may have a longer clock than the channel's guard.
    period = channel.period * len(channel.ids) // gcd(len(channel.residues), len(channel.ids))
    require(period * len(channel.ids) <= MAX_SUMMARY_VISITS, "child event boundary budget exceeded")
    epochs = {i: [] for i in channel.ids}
    ordinal = 0
    for g in range(period):
        if g % channel.period in channel.residues:
            epochs[channel.ids[ordinal % len(channel.ids)]].append(g)
            ordinal += 1
    for event_id, active in epochs.items():
        for residue in range(period):
            gap = min((residue - 1 - channel.distance - g) % period for g in active)
            yield EventReuse(
                channel.name,
                event_id,
                bank,
                CountDomain(period, residue, minimum=active[0] + channel.distance + 1),
                Access(destination, Position(True, gap)),
                Access(source, Position(False, active[0])),
            )


def summarize_local_events(loop):
    """Read actual child Set/Wait operations and retain cross-call obligations."""
    from .loop_graph import channels, commands

    require(len(loop.actors) == 1, "mixed child resources need collective region analysis")
    channel_map = channels(loop)
    commands(loop, channel_map)  # Validate endpoint positions, guards and pairing.
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    result = []
    for channel in channel_map.values():
        bank = (
            channel.sources[0][0],
            tasks[channel.sources[0]].pipe,
            tasks[channel.destinations[0]].pipe,
        )
        result.extend(
            item
            for item in _event_reuse(channel, bank)
            if loop.constant_count is None or item.domain.contains(loop.constant_count)
        )
        require(len(result) <= MAX_SUMMARY_VISITS, "child event summary budget exceeded")
    return tuple(result)


@dataclass(frozen=True)
class RegionSummary:
    roots: tuple[str, ...]
    cells: tuple[CellSummary, ...]
    visits: int
    bound: Expr | None
    constant_count: int | None
    # Actual allocation identities, not textual buffer names. Region composition
    # must not merge two private buffers just because their printed names match.
    origins: tuple[tuple[str, SSAValue], ...] = ()
    bound_value: SSAValue | None = None

    def for_count(self, count):
        require(count >= 0, "negative invocation length")
        require(
            self.constant_count is None or count == self.constant_count,
            "summary is bound to a different constant child length",
        )
        # The empty child is the identity transfer, not an initialized output.
        return tuple(cell for cell in self.cells if cell.domain.contains(count))

    def invocation_dependencies(self):
        result = []
        for cell in self.cells:

            def add(hazard, source, destination, *, cell=cell):
                result.append(
                    InvocationDependence(
                        cell.storage,
                        cell.start,
                        cell.stop,
                        cell.domain,
                        hazard,
                        source,
                        destination,
                    )
                )

            if cell.last_write is not None:
                for reader in cell.reads_before_write:
                    add("RAW", cell.last_write, reader)
                if cell.first_write is not None:
                    add("WAW", cell.last_write, cell.first_write)
            if cell.first_write is not None:
                for reader in cell.pending_reads:
                    add("WAR", reader, cell.first_write)
        return tuple(result)


def _scan(events):
    first, last = None, None
    live_in, pending = [], []
    for access, read, write in events:
        if read:
            span = ReadSpan.one(access)
            pending.append(span)
            if first is None:
                live_in.append(span)
        if write:
            first = first or access
            last, pending = access, []
    return tuple(live_in), first, last, tuple(pending)


def summarize_reused_storage(loop, roots):
    """Summarize explicitly selected, stationary roots of one local child.

    Internal loop synchronization and the parent completion/resource proof are
    separate consumers. No entry/exit fence is manufactured by this analysis.
    Repeated invocations share this immutable bound. A variable per-parent
    child length needs two distinct domain parameters, not this transfer.
    """
    roots = tuple(sorted(set(roots)))
    require(len(loop.actors) == 1, "mixed region summaries require participant analysis")
    known = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
    }
    require(set(roots) <= known, "region summary names an unknown storage root")
    require(not set(roots) & loop.frames.keys(), "stream-frame summary needs parent ownership")
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            for ref in task.refs:
                if loop.root(actor, ref) in roots:
                    require(
                        not slope(ref) and not ref.count_offset and not ref.count_core_stride,
                        "translated region storage needs parent ownership",
                    )
                    require(
                        ref.prefix_extent is None,
                        "region summary needs unclipped physical storage",
                    )
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    summaries, visits = [], 0
    for root, lo, hi, events in access_cells(loop, roots=set(roots), include_readonly=True):
        period = loop.storage_periods[root]
        # Account before allocating domains. This is static-period cost, not N.
        visits += 2 * period * len(events)
        require(visits <= MAX_SUMMARY_VISITS, "region boundary summary exceeds static visit budget")
        steady, terminal = [], {}
        for (g, actor, uid), read, write, _ in events:
            event = (g, (actor, uid), read, write)
            if tasks[actor, uid].last:
                terminal.setdefault(g, []).append(event)
            else:
                steady.append(event)

        def at(event, position, *, first=False, root=root):
            _, key, read, write = event
            task = tasks[key]
            if first and task.native == "mad_first" and root == loop.root(key[0], task.refs[2]):
                read = False
            return Access(key, position), read, write

        def save(domain, frontier, *, root=root, lo=lo, hi=hi):
            if loop.constant_count is None or domain.contains(loop.constant_count):
                summaries.append(CellSummary(root, lo, hi, domain, *frontier))

        for count in range(1, period):
            sequence = [
                at(event, Position(False, event[0]), first=event[0] == 0)
                for event in steady
                if event[0] < count
            ]
            sequence += [
                at(event, Position(False, count - 1)) for event in terminal.get(count - 1, ())
            ]
            save(CountDomain(period, count, count), _scan(sequence))

        head = [at(event, Position(False, event[0]), first=event[0] == 0) for event in steady]
        incoming, first_write, _, _ = _scan(head)
        generations = [event[0] for event in steady]
        for residue in range(period):
            domain = CountDomain(period, residue)
            split = bisect_left(generations, residue)
            # Rotate the nonempty access word; do not visit P empty generations
            # for each of P boundary residues of a sparsely touched cell.
            tail = [
                at(event, Position(True, (residue - 1 - event[0]) % period))
                for event in steady[split:] + steady[:split]
            ]
            final = [
                at(event, Position(True, 0)) for event in terminal.get((residue - 1) % period, ())
            ]
            if first_write is not None:
                _, _, last_write, pending = _scan(tail + final)
                save(domain, (incoming, first_write, last_write, pending))
            else:
                # No periodic writer bounds these reads. Keep their whole
                # families; a terminal writer only changes the exit frontier.
                spans = tuple(
                    ReadSpan(
                        key, Position(False, g), Position(True, (residue - 1 - g) % period), period
                    )
                    for g, key, read, _ in steady
                    if read
                )
                final_in, final_first, final_last, final_pending = _scan(final)
                save(
                    domain,
                    (
                        spans + final_in,
                        final_first,
                        final_last,
                        final_pending if final_last is not None else spans + final_pending,
                    ),
                )
    origins = {}
    for actor, data in loop.actors.items():
        for op in data.kernel.body.block.ops:
            if isinstance(op, BufferOp):
                root = op.sym_name.data if op.external.data else actor + "/" + op.sym_name.data
                if root in roots:
                    origins[root] = op.ref
    require(set(origins) == set(roots), "region summary needs actual stationary allocation roots")
    return RegionSummary(
        roots,
        tuple(summaries),
        visits,
        loop.bound,
        loop.constant_count,
        tuple(origins.items()),
        next(iter(loop.actors.values())).stream.limit if loop.constant_count is None else None,
    )

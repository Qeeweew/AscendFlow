"""Exact access frontiers at the boundary of a repeated leaf pipeline.

These are memory obligations, not completion certificates or event plans. The
caller selects storage roots. Periodic summaries require stationary storage;
fixed summaries inspect the real constant child domain. Checked invocation
frames retain their actual FrameOp identity, rather than losing the owning call
in a root name.

For period P, enumerate N=1..P-1 and P long domains N>=P, N%P=r. A periodic
writer leaves a suffix of at most P generations. With no periodic writer, a
read family remains a span: its final member is not a whole-prefix retirement.
No domain enumerates runtime N, nor is a finite sample used as an induction.
"""

from bisect import bisect_left
from dataclasses import dataclass, replace
from math import gcd

from xdsl.ir import SSAValue

from .budget import poll
from .completion import CountDomain, Position
from .ir import BufferOp, FrameOp
from .loop_model import access_cells, cut_access_cells, slope
from .model import require, verify_translated_owners
from .scalar import Expr

MAX_SUMMARY_VISITS = 1_000_000
MAX_FIXED_ACCESS_SEGMENTS = 65_536
MAX_FIXED_SUMMARY_VISITS = 262_144
MAX_FIXED_DEFINITION_VISITS = 262_144


def allocation(storage):
    """Physical allocation behind a stationary root or checked call-local frame."""
    op = storage.owner
    if isinstance(op, FrameOp):
        op = op.source.owner
    require(isinstance(op, BufferOp), "region summary lost allocation provenance")
    return op


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
class ReadBinding:
    """The exact logical version read by one cell access.

    None defers the producer to this invocation's incoming value. It neither
    initializes storage nor permits rebinding the read to a later overwrite.
    """

    reader: Access
    writer: Access | None


@dataclass(frozen=True)
class DefinitionRun:
    """One independent W-only definition and its R/W continuation versions.

    An incoming run has entry=None, including a run beginning with RW. Every
    read keeps its writer binding even when a later RW retires its frontier.
    pending_reads is only the frontier after last_write, not all run readers.
    These are memory facts; they authorize neither retirement nor GM reordering.
    """

    entry: Access | None
    last_write: Access | None
    pending_reads: tuple[ReadSpan, ...]
    read_bindings: tuple[ReadBinding, ...] = ()


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
    definitions: tuple[DefinitionRun, ...] = ()


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


def fixed_event_reuse(channel, bank: tuple[str, str, str], count: int) -> tuple[EventReuse, ...]:
    """Return exact event reuse endpoints without enumerating the child clock.

    The caller validates the actual Set/Wait operations separately. Source
    epochs are ordered by ``(period quotient, residue rank)``; event IDs rotate
    over that ordinal, including when their values are not an arithmetic list.
    Only IDs with a real notification in this fixed invocation get a boundary.
    """
    require(type(count) is int and count >= 0, "fixed event count must be nonnegative")
    require(channel.kind == "local", "hierarchical counters require participant analysis")
    require(not channel.carry, "fixed child events cannot carry invocation credits")
    require(channel.ids, "child event IDs must be assigned before resource summarization")
    require(
        len(set(channel.ids)) == len(channel.ids),
        "child event IDs must be distinct",
    )
    require(
        len(channel.sources) == len(channel.destinations) == 1
        and channel.sources[0][0] == channel.destinations[0][0],
        "child local events need one source and destination in one actor",
    )
    require(
        type(channel.period) is int
        and channel.period > 0
        and type(channel.distance) is int
        and channel.distance >= 0
        and channel.residues,
        "invalid fixed child event activation",
    )
    previous = -1
    for residue in channel.residues:
        require(
            type(residue) is int and previous < residue < channel.period,
            "fixed child event residues must be sorted, distinct and in range",
        )
        previous = residue
    (source,), (destination,) = channel.sources, channel.destinations
    width = len(channel.ids)
    if channel.last:
        require(width == 1, "a terminal child channel needs one event ID")
        first = count - channel.distance - 1
        if first < 0 or first % channel.period not in channel.residues:
            return ()
        boundaries = ((channel.ids[0], first, count - 1),)
    else:
        stop = count - channel.distance
        if stop <= 0:
            return ()
        cycles, tail = divmod(stop, channel.period)
        ranks = len(channel.residues)
        notifications = cycles * ranks + bisect_left(channel.residues, tail)
        if not notifications:
            return ()

        def epoch(ordinal):
            cycle, rank = divmod(ordinal, ranks)
            return cycle * channel.period + channel.residues[rank]

        boundaries = (
            (
                event_id,
                epoch(index),
                epoch(index + (notifications - 1 - index) // width * width) + channel.distance,
            )
            for index, event_id in enumerate(channel.ids[: min(width, notifications)])
        )
    domain = CountDomain(1, 0, exact=count)
    return tuple(
        EventReuse(
            channel.name,
            event_id,
            bank,
            domain,
            Access(destination, Position(False, last)),
            Access(source, Position(False, first)),
        )
        for event_id, first, last in boundaries
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
        if loop.constant_count is not None and not channel.carry:
            result.extend(fixed_event_reuse(channel, bank, loop.constant_count))
        else:
            # Carried invocation credits have a different cross-call contract.
            # Keep their established domains even for a fixed-length child;
            # retained scope children reject carry before resource analysis.
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
    # Separately budgeted fixed-scan/record work; the established frontier
    # visits retain their original definition for existing callers.
    definition_visits: int = 0

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


def _scan(events, *, definitions: list[DefinitionRun] | None = None):
    first, last = None, None
    live_in, pending = [], []
    entry, bindings, active, count = None, [], False, 0
    for access, read, write in events:
        if definitions is not None:
            if write and not read:
                if active:
                    definitions.append(DefinitionRun(entry, last, tuple(pending), tuple(bindings)))
                entry, bindings = access, []
            active = True
            if read:
                bindings.append(ReadBinding(access, last))
            count += 1
            if count % 256 == 0:
                poll("fixed child definitions", accesses=count, runs=len(definitions))
        if read:
            span = ReadSpan.one(access)
            pending.append(span)
            if first is None:
                live_in.append(span)
        if write:
            first = first or access
            last, pending = access, []
    if definitions is not None and active:
        definitions.append(DefinitionRun(entry, last, tuple(pending), tuple(bindings)))
    return tuple(live_in), first, last, tuple(pending)


def _fixed_roots(loop, roots):
    roots = tuple(sorted(set(roots)))
    require(len(loop.actors) == 1, "mixed fixed summaries require participant analysis")
    count = loop.constant_count
    require(type(count) is int and count >= 0, "fixed summary requires an actual constant count")
    require(
        all(
            data.stream.limit is None
            and data.stream.count is not None
            and data.stream.count.data == count
            for data in loop.actors.values()
        ),
        "fixed summary count must match the actual constant child stream",
    )
    selected = set(roots)
    known = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
    }
    require(selected <= known, "fixed summary names an unknown storage root")
    require(
        loop.invocation is None
        or not selected.intersection(frame.name for frame in loop.invocation.frames),
        "fixed invocation-frame summary needs parent ownership",
    )
    return roots


@dataclass(frozen=True)
class FixedStorageSelection:
    """A scan strategy, not a validation or completion certificate."""

    roots: tuple[str, ...]
    definitions: frozenset[str]
    compact: frozenset[str]
    segments: int


def select_fixed_storage(loop, roots, *, definitions_needed=None, allow_read_spans=True):
    """Select cheaper periodic scans without excluding irregular fixed roots.

    The default retains every definition, as the original fixed API does.
    Query consumers that cannot yet carry a native retirement span may opt
    into only roots whose every reader has a matching periodic writer.
    Selection uses descriptors only, before allocating physical row tuples.
    """
    roots = _fixed_roots(loop, roots)
    definitions = frozenset(roots if definitions_needed is None else definitions_needed)
    require(definitions <= set(roots), "fixed definitions name an unselected storage root")
    refs = {root: [] for root in roots}
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            for number, (ref, mode) in enumerate(zip(task.refs, task.modes, strict=True)):
                root = loop.root(actor, ref)
                if root in refs:
                    refs[root].append((task, number, ref, mode))
    compact = set()
    for root, parts in refs.items():
        period = loop.storage_periods[root]
        # Two scans of one period replace a full fixed frontier scan. A short
        # fixed domain never acquires a larger period or an extra budget cost.
        if root in definitions or root in loop.frames or loop.constant_count <= 2 * period:
            continue
        if any(
            task.last
            or slope(ref)
            or ref.count_offset
            or ref.count_core_stride
            or ref.storage.count_scale
            or ref.prefix_extent is not None
            or ref.tail_nbytes
            or task.native == "mad_cycle"
            and number == 2
            for task, number, ref, _ in parts
        ):
            continue
        if not allow_read_spans:
            # Exact descriptor equality is a sufficient coverage proof. More
            # complicated writer unions keep their established fixed scan.
            writes = {ref for _, _, ref, mode in parts if "W" in mode}
            if any("R" in mode and ref not in writes for _, _, ref, mode in parts):
                continue
        compact.add(root)
    segments = sum(
        (
            loop.storage_periods[root]
            if root in compact
            else int(loop.constant_count > 0)
            if task.last
            else loop.constant_count
        )
        * ref.rows
        * (loop.access_groups(ref.storage) if ref.storage.external else 1)
        for root, parts in refs.items()
        for task, _, ref, _ in parts
    )
    return FixedStorageSelection(roots, definitions, frozenset(compact), segments)


def summarize_fixed_storage(
    loop, roots, *, definitions_needed=None, allow_read_spans=True, _selection=None
):
    """Exact boundary transfer over one genuinely fixed child domain.

    Required definition roots retain every internal version and read binding.
    Other stationary roots may use one exact periodic word; finite drifting
    footprints keep their absolute byte coordinates and original fixed scan.
    Compact ReadSpans describe all readers and grant no completion fact.
    """
    roots = _fixed_roots(loop, roots)
    selection = _selection or select_fixed_storage(
        loop, roots, definitions_needed=definitions_needed, allow_read_spans=allow_read_spans
    )
    require(selection.roots == roots, "fixed scan selection names different storage roots")
    count, selected = loop.constant_count, set(roots)
    work = []
    segments = ownership_segments = events = 0
    physical_refs = []
    for _, _, _, actor, task in sorted(
        (task.logical_phase, owner, index, actor, task)
        for owner, (actor, data) in enumerate(loop.actors.items())
        for index, task in enumerate(data.program.tasks)
    ):
        parts = []
        generations = 0
        for number, ref in enumerate(task.refs):
            root = loop.root(actor, ref)
            if root not in selected:
                continue
            require(
                not ref.count_offset and not ref.count_core_stride and not ref.storage.count_scale,
                "fixed translated storage needs parent ownership",
            )
            require(ref.prefix_extent is None, "fixed summary needs unclipped physical storage")
            require(ref.rows > 0, "fixed summary needs nonempty physical row declarations")
            scan_count = loop.storage_periods[root] if root in selection.compact else count
            generations = max(generations, int(count > 0) if task.last else scan_count)
            cores = loop.access_groups(ref.storage) if ref.storage.external else 1
            fragments = (int(count > 0) if task.last else scan_count) * ref.rows * cores
            segments += fragments
            if ref.storage.external:
                ownership_segments += fragments
                physical_refs.append(ref)
            parts.append((root, number, ref, cores, scan_count))
        if parts:
            events += generations
            work.append((actor, task, parts, generations))
    # Reject before asking Ref.segments to allocate any row tuple. Counts are
    # analysis bounds, never periods or permission to materialize native tasks.
    require(
        segments <= MAX_FIXED_ACCESS_SEGMENTS,
        "fixed summary physical access-segment budget exceeded",
    )
    visits = segments + events + ownership_segments + len(physical_refs) * loop.groups
    require(visits <= MAX_FIXED_SUMMARY_VISITS, "fixed summary visit budget exceeded")
    poll("fixed child summary", count=count, segments=segments, visits=visits)
    cells, effects = {}, []
    order = {(actor, task.uid): i for i, (actor, task, _, _) in enumerate(work)}
    iterations = 0
    for actor, task, parts, scan_count in work:
        generations = (
            (count - 1,) if task.last and count else () if task.last else range(scan_count)
        )
        for g in generations:
            modes = task.modes_for(g)
            touched = {}
            for root, number, ref, cores, root_count in parts:
                if not task.last and g >= root_count:
                    continue
                mode = task.modes[number] if root in selection.compact else modes[number]
                for core in range(cores):
                    for lo, hi in ref.segments(g, core):
                        require(
                            0 <= lo < hi <= ref.storage.nbytes,
                            f"{root}: fixed child footprint exceeds actual allocation",
                        )
                        if core == 0:
                            touched.setdefault(root, []).append(
                                (lo, hi, mode, ref.storage.initialized)
                            )
                            if ref.storage.external:
                                effects.append((ref.storage.name, lo, hi, mode))
                        iterations += 1
                        if iterations % 256 == 0:
                            poll("fixed child footprints", segments=iterations, visits=visits)
            for root, intervals in touched.items():
                cells.setdefault(root, []).append((g, actor, task.uid, intervals))
    if count and physical_refs:
        poll("fixed child owners", effects=len(effects), groups=loop.groups)
        verify_translated_owners(
            physical_refs,
            effects,
            loop.groups,
            translation_error="fixed child views disagree on GM ownership",
            overlap_error="fixed child GM owners overlap",
        )
    # Charge cuts AND frontier visits before the shared sweep allocates its
    # per-cell event lists. Interval count alone does not bound intersecting cuts.
    definition_visits = 0
    for root, accesses in cells.items():
        accesses.sort(key=lambda access: (access[0], order[access[1], access[2]]))
        points = sorted(
            {p for _, _, _, parts in accesses for lo, hi, _, _ in parts for p in (lo, hi)}
        )
        cuts = {point: i for i, point in enumerate(points)}
        cut_visits = sum(
            cuts[hi] - cuts[lo] for _, _, _, parts in accesses for lo, hi, _, _ in parts
        )
        visits += len(points) + 2 * cut_visits
        require(visits <= MAX_FIXED_SUMMARY_VISITS, "fixed summary cut/scan visit budget exceeded")
        # Each interval/cell incidence bounds one extra scan action, plus
        # a run and a read binding (the first incoming RW/R may need both).
        # Estimate before allocating cell-event lists or definition records.
        definition_visits += 3 * cut_visits if root in selection.definitions else 0
        if root in selection.compact:
            visits += 2 * cut_visits
            require(
                visits <= MAX_FIXED_SUMMARY_VISITS, "fixed summary transfer visit budget exceeded"
            )
        require(
            definition_visits <= MAX_FIXED_DEFINITION_VISITS,
            "fixed definition summary visit budget exceeded",
        )
        poll("fixed child cuts", root=root, points=len(points), visits=visits)
    summaries = []
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    for root, lo, hi, accesses in cut_access_cells(cells):
        if root in selection.compact:
            summaries.extend(
                replace(cell, domain=CountDomain(1, 0, exact=count))
                for cell in _periodic_cell(loop, tasks, root, lo, hi, accesses)
            )
            if len(summaries) % 256 == 0:
                poll("fixed child frontiers", cells=len(summaries), visits=visits)
            continue
        definitions = [] if root in selection.definitions else None
        frontier = _scan(
            (
                (Access((actor, uid), Position(False, g)), read, write)
                for (g, actor, uid), read, write, _ in accesses
            ),
            definitions=definitions,
        )
        summaries.append(
            CellSummary(
                root, lo, hi, CountDomain(1, 0, exact=count), *frontier, tuple(definitions or ())
            )
        )
        if len(summaries) % 256 == 0:
            poll("fixed child frontiers", cells=len(summaries), visits=visits)
    origins = {root: op.ref for root, op in loop.roots.items() if root in selected}
    for actor, data in loop.actors.items():
        for op in data.kernel.body.block.ops:
            if isinstance(op, BufferOp):
                root = op.sym_name.data if op.external.data else actor + "/" + op.sym_name.data
                if root in selected:
                    origins[root] = op.ref
    require(set(origins) == selected, "fixed summary needs actual allocation roots")
    return RegionSummary(
        roots,
        tuple(summaries),
        visits,
        loop.bound,
        count,
        tuple(origins.items()),
        definition_visits=definition_visits,
    )


def _periodic_cell(loop, tasks, root, lo, hi, events):
    """The exact periodic transfer, restricted to the actual count if fixed."""
    period = loop.storage_periods[root]
    steady, terminal = [], {}
    for (g, actor, uid), read, write, _ in events:
        event = (g, (actor, uid), read, write)
        if tasks[actor, uid].last:
            terminal.setdefault(g, []).append(event)
        else:
            steady.append(event)

    def at(event, position, *, first=False):
        _, key, read, write = event
        task = tasks[key]
        if first and task.native == "mad_first" and root == loop.root(key[0], task.refs[2]):
            read = False
        return Access(key, position), read, write

    fixed = loop.constant_count
    counts = range(1, period) if fixed is None else (fixed,) if 0 < fixed < period else ()
    for count in counts:
        sequence = [
            at(event, Position(False, event[0]), first=event[0] == 0)
            for event in steady
            if event[0] < count
        ]
        sequence += [at(event, Position(False, count - 1)) for event in terminal.get(count - 1, ())]
        yield CellSummary(root, lo, hi, CountDomain(period, count, count), *_scan(sequence))
    if fixed is not None and fixed < period:
        return

    head = [at(event, Position(False, event[0]), first=event[0] == 0) for event in steady]
    incoming, first_write, _, _ = _scan(head)
    generations = [event[0] for event in steady]
    for residue in range(period) if fixed is None else (fixed % period,):
        domain = CountDomain(period, residue)
        split = bisect_left(generations, residue)
        # Rotate the nonempty access word; do not visit P empty generations
        # for each of P boundary residues of a sparsely touched cell.
        tail = [
            at(event, Position(True, (residue - 1 - event[0]) % period))
            for event in steady[split:] + steady[:split]
        ]
        final = [at(event, Position(True, 0)) for event in terminal.get((residue - 1) % period, ())]
        if first_write is not None:
            _, _, last_write, pending = _scan(tail + final)
            yield CellSummary(root, lo, hi, domain, incoming, first_write, last_write, pending)
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
            yield CellSummary(
                root,
                lo,
                hi,
                domain,
                spans + final_in,
                final_first,
                final_last,
                final_pending if final_last is not None else spans + final_pending,
            )


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
        visits += 2 * (period if loop.constant_count is None else 1) * len(events)
        require(visits <= MAX_SUMMARY_VISITS, "region boundary summary exceeds static visit budget")
        summaries.extend(_periodic_cell(loop, tasks, root, lo, hi, events))
    origins = {}
    for actor, data in loop.actors.items():
        for op in data.kernel.body.block.ops:
            if isinstance(op, BufferOp):
                root = op.sym_name.data if op.external.data else actor + "/" + op.sym_name.data
                if root in roots:
                    origins[root] = op.ref
    if loop.invocation is not None:
        for frame in loop.invocation.frames:
            if frame.name in roots:
                origins[frame.name] = frame.op.ref
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

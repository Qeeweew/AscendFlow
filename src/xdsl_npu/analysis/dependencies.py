"""One periodic access frontier, with direct read-carry witnesses.

The scan is bounded by physical storage periods, never by a loop trip count.
Read carries are recorded while a cell's pending readers are discharged. They
therefore retain the actual intervening reader and do not require a later join
between WAW hazards and all possible readers.
"""

from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

from ..diagnostics import require

if TYPE_CHECKING:
    from ..loop_model import Loop


@dataclass(frozen=True, slots=True)
class Dependence:
    source: tuple[str, str]
    destination: tuple[str, str]
    distance: int
    period: int
    residues: tuple[int, ...]
    hazard: str
    storage: str
    start: int
    stop: int
    last: bool = False
    prefix: bool = False


@dataclass(frozen=True, slots=True)
class GenerationDomain:
    """An explicit read predicate in a carry's WAW-consumer generation.

    This predicate is intersected with the WAW's endpoint existence, storage
    residues, and terminal activation. Its period is a native clear period;
    it is never multiplied into the storage period or runtime count.
    """

    minimum: int = 0
    period: int = 1
    residues: tuple[int, ...] = (0,)

    def __post_init__(self):
        require(type(self.minimum) is int, "generation minimum must be an integer")
        require(type(self.period) is int and self.period > 0, "invalid read-domain period")
        require(
            tuple(sorted(set(self.residues))) == self.residues
            and all(type(r) is int and 0 <= r < self.period for r in self.residues),
            "invalid read-domain residues",
        )

    def contains(self, generation: int) -> bool:
        return generation >= self.minimum and generation % self.period in self.residues

    def first(self, period: int, residue: int, minimum: int = 0) -> int | None:
        """First generation in two residue families, using CRT rather than expansion."""
        lower = max(minimum, self.minimum)
        common = gcd(period, self.period)
        quotient = self.period // common
        inverse = pow(period // common, -1, quotient) if quotient > 1 else 0
        repeat = period * quotient
        result = None
        for allowed in self.residues:
            if (allowed - residue) % common:
                continue
            generation = residue + period * (((allowed - residue) // common * inverse) % quotient)
            generation += max(0, (lower - generation + repeat - 1) // repeat) * repeat
            result = generation if result is None else min(result, generation)
        return result

    def covers(self, period: int, residue: int, minimum: int) -> bool:
        """Whether an entire real WAW residue family satisfies this read predicate."""
        first = residue + max(0, (minimum - residue + period - 1) // period) * period
        if first < self.minimum:
            return False
        common = gcd(period, self.period)
        return sum(r % common == residue % common for r in self.residues) == self.period // common


@dataclass(frozen=True, slots=True)
class ReadCarry:
    """RAW(W,R) and WAR(R,Wnew) directly cover this exact WAW fragment.

    For a read-modify-write instruction R is Wnew itself. Its real RAW then
    suffices and ``war`` is None; no fictional same-instruction WAR is added.
    ``domain`` always uses the WAW consumer's logical generation.
    """

    waw: Dependence
    raw: Dependence
    war: Dependence | None
    domain: GenerationDomain


@dataclass(frozen=True, slots=True)
class DependencyAnalysis:
    obligations: tuple[Dependence, ...]
    required: tuple[Dependence, ...]
    read_carries: tuple[ReadCarry, ...]


def cut_access_cells(cells):
    """Exact interval cuts shared by periodic and fixed-domain access scans.

    Aliased operands of one actual instruction contribute one read-before-write
    event per cell. Holes and distinct physical fragments remain separate.
    """
    for root, work in cells.items():
        points = sorted({p for _, _, _, parts in work for lo, hi, _, _ in parts for p in (lo, hi)})
        cuts = {point: i for i, point in enumerate(points)}
        per_cell = [[] for _ in points[:-1]]
        for g, actor, uid, parts in work:
            touched = {}
            for start, stop, mode, init in parts:
                for cell in range(cuts[start], cuts[stop]):
                    old_read, old_write, old_init = touched.get(cell, (False, False, False))
                    touched[cell] = (
                        old_read or "R" in mode,
                        old_write or "W" in mode,
                        old_init or init,
                    )
            for cell, (read, write, init) in touched.items():
                per_cell[cell].append(((g, actor, uid), read, write, init))
        for lo, hi, cell_events in zip(points, points[1:], per_cell):
            if cell_events:
                yield root, lo, hi, cell_events


def _read_domain(loop, actor, task, root):
    # Native validation fixes the three MAD operands to distinct L0 banks.
    # Thus only the accumulator's exact root has a generation-dependent read;
    # reads of the two multiplicands remain unconditional.
    if task.native in {"mad_first", "mad_cycle"} and root == loop.root(actor, task.refs[2]):
        if task.native == "mad_first":
            return GenerationDomain(minimum=1)
        period = task.params[3]
        return GenerationDomain(period=period, residues=tuple(range(1, period)))
    return GenerationDomain()


@dataclass(frozen=True, slots=True)
class _Reader:
    occurrence: tuple[int, str, str]
    writer: tuple[int, str, str] | None
    raw: tuple | None
    domain: GenerationDomain


def analyze_dependencies(loop: "Loop", *, live_in=()) -> DependencyAnalysis:
    """Scan each exact periodic cell once, retaining all and required obligations.

    The legacy obligations remain conservative for conditional native reads.
    A WAW leaves ``required`` only when direct real-read certificates cover its
    entire original activation. Partial read domains never silently drop WAW.
    """
    from ..loop_model import access_cells

    edges, carries, covered = {}, {}, {}
    task_map = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    domains = {
        (actor, task.uid, loop.root(actor, ref)): _read_domain(
            loop, actor, task, loop.root(actor, ref)
        )
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
    }

    def edge(
        source, destination, hazard, root, lo, hi, terminal=False, prefix=False, *, record=True
    ):
        if source is None or source == destination:
            return None
        sg, sa, su = source
        dg, da, du = destination
        distance = dg - sg
        require(distance >= 0, "logical effect order is inconsistent")
        key = (sa, su, da, du, distance, hazard, root, lo, hi, terminal, prefix)
        if record:
            edges.setdefault(key, set()).add(dg % loop.storage_periods[root])
        return key

    def carry(writer, reader, destination, waw, war, period):
        if waw is None or reader.raw is None or reader.writer != writer:
            return
        offset = destination[0] - reader.occurrence[0]
        read = reader.domain
        domain = GenerationDomain(
            minimum=max(waw[4], read.minimum + offset),
            period=read.period,
            residues=tuple(sorted((r + offset) % read.period for r in read.residues)),
        )
        residue = destination[0] % period
        if domain.first(period, residue, waw[4]) is None:
            return
        carries.setdefault((waw, reader.raw, war, domain), set()).add(residue)
        if domain.covers(period, residue, waw[4]):
            covered.setdefault(waw, set()).add(residue)

    for root, lo, hi, cell_events in access_cells(loop):
        period = loop.storage_periods[root]
        initialized = cell_events[-1][3]
        steady = [
            (event, read, write)
            for event, read, write, _ in cell_events
            if not task_map[event[1:]].last
        ]
        terminal = [
            (event, read, write)
            for event, read, write, _ in cell_events
            if task_map[event[1:]].last
        ]
        writes = [i for i, (_, _, write) in enumerate(steady) if write]
        last = writes[-1] if writes and root not in loop.frames else None
        writer = None if last is None else (steady[last][0][0] - period, *steady[last][0][1:])

        def reader_at(event, writer, *, root=root, lo=lo, hi=hi):
            return _Reader(
                event,
                writer,
                edge(writer, event, "RAW", root, lo, hi, record=False),
                domains[*event[1:], root],
            )

        pending = (
            []
            if last is None
            else [
                reader_at((event[0][0] - period, *event[0][1:]), writer)
                for event in steady[last + 1 :]
                if event[1]
            ]
        )
        unbounded_readers = not writes and root not in loop.frames
        if unbounded_readers:
            pending = [
                reader_at((event[0] - period, *event[1:]), None)
                for event, read, _ in steady
                if read
            ]

        def visit(
            events,
            writer,
            pending,
            *,
            terminal=False,
            root=root,
            period=period,
            initialized=initialized,
            lo=lo,
            hi=hi,
            unbounded_readers=unbounded_readers,
        ):
            for destination, read, write in events:
                g, actor, uid = destination
                domain = domains[actor, uid, root]
                if read:
                    # A representative clear iteration is not a read. For a
                    # terminal template a fixed N selects its actual final g;
                    # otherwise CRT finds the first actual read in this family.
                    if terminal and loop.constant_count is not None:
                        actual = loop.constant_count - 1
                        first = (
                            actual
                            if actual >= 0 and actual % period == g and domain.contains(actual)
                            else None
                        )
                    else:
                        first = domain.first(period, g)
                        if (
                            first is not None
                            and loop.constant_count is not None
                            and first >= loop.constant_count
                        ):
                            first = None
                    require(
                        initialized
                        or root in live_in
                        or first is None
                        or (writer is not None and first - (g - writer[0]) >= 0),
                        f"{actor}/{uid} has an undefined {'terminal' if terminal else 'first'} read on {root}[{lo}:{hi}]",
                    )
                    raw = edge(writer, destination, "RAW", root, lo, hi, terminal)
                    pending.append(_Reader(destination, writer, raw, domain))
                if write:
                    waw = edge(writer, destination, "WAW", root, lo, hi, terminal)
                    # A stream read with only a terminal overwrite has no
                    # finite writer window. Retire each cumulative reader
                    # family, preserving the old reader-prefix obligation.
                    frontier = pending
                    if terminal and unbounded_readers:
                        latest = {}
                        for reader in pending:
                            key = reader.occurrence[1:]
                            previous = latest.get(key)
                            if previous is None or reader.occurrence > previous.occurrence:
                                latest[key] = reader
                        frontier = latest.values()
                    for reader in frontier:
                        prefix = (
                            terminal
                            and unbounded_readers
                            and not task_map[reader.occurrence[1:]].last
                        )
                        war = edge(
                            reader.occurrence, destination, "WAR", root, lo, hi, terminal, prefix
                        )
                        carry(writer, reader, destination, waw, war, period)
                    writer, pending = destination, []
            return writer, pending

        # Terminal effects run on a copy and never shadow the next steady
        # generation's writer or pending-reader frontier.
        by_generation, tails = {}, {}
        for event in steady:
            by_generation.setdefault(event[0][0], []).append(event)
        for event in terminal:
            tails.setdefault(event[0][0], []).append(event)
        for g in sorted(by_generation.keys() | tails.keys()):
            if root in loop.frames:
                writer, pending = None, []
            writer, pending = visit(by_generation.get(g, ()), writer, pending)
            visit(tails.get(g, ()), writer, list(pending), terminal=True)

    def dependence(key, residues):
        sa, su, da, du, distance, hazard, root, lo, hi, terminal, prefix = key
        return Dependence(
            (sa, su),
            (da, du),
            distance,
            loop.storage_periods[root],
            tuple(sorted(residues)),
            hazard,
            root,
            lo,
            hi,
            terminal,
            prefix,
        )

    obligations, required = [], []
    for key, residues in sorted(edges.items()):
        obligation = dependence(key, residues)
        obligations.append(obligation)
        if key[5] != "WAW" or not residues <= covered.get(key, set()):
            required.append(obligation)
    read_carries = []
    for (waw, raw, war, domain), residues in sorted(
        carries.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or (),
            item[0][3].minimum,
            item[0][3].period,
            item[0][3].residues,
        ),
    ):
        require(raw in edges, "read-carry RAW is missing from the periodic access scan")
        offset = 0 if war is None else war[4]
        raw_residues = {(r - offset) % loop.storage_periods[raw[6]] for r in residues}
        read_carries.append(
            ReadCarry(
                dependence(waw, residues),
                dependence(raw, raw_residues),
                None if war is None else dependence(war, residues),
                domain,
            )
        )
    return DependencyAnalysis(tuple(obligations), tuple(required), tuple(read_carries))

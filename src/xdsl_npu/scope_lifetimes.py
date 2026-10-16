"""Reorder independent private definitions without changing logical producers.

The first contract handles stationary captures of actor-private local storage.
Each fixed child keeps its exact internal definition runs and leaf proof; a
fresh boundary occupancy exports its first clear, last version and pending
native readers. Incoming read-only calls remain attached to their logical
writer. Incoming RW and unsupported addresses retain the old logical contract.

Finite parent fill/drain is derived from inactive intervening writer families,
never from samples of N. This proposes memory obligations, not completion.
"""

from dataclasses import dataclass, replace

from .budget import poll
from .completion import CountDomain, Position, _first, coalesce
from .completion_flow import ALL
from .diagnostics import require
from .loop_model import Dependence
from .region_summary import Access, CellSummary, ReadSpan
from .scope_domains import Activation, ScopedDependence, affine, position
from .symbolic_completion import choice, holds, le, meet

MAX_LIFETIME_VISITS = 262_144
MAX_LIFETIME_RELATIONS = 65_536


@dataclass(frozen=True)
class LifetimePlan:
    dependencies: tuple[Dependence | ScopedDependence, ...]
    reordered_roots: tuple[str, ...]
    fallbacks: tuple[tuple[str, str], ...]
    visits: int
    # Physical occupancy at invocation boundaries, after logical RAW binding.
    # Only roots whose WAW/WAR contract was actually reordered contribute.
    boundaries: tuple[CellSummary, ...] = ()


@dataclass
class _Budget:
    visits: int = 0

    def charge(self, amount):
        self.visits += amount
        require(self.visits <= MAX_LIFETIME_VISITS, "scope lifetime analysis visit budget exceeded")
        poll("scope definition lifetimes", visits=self.visits)


class _Unsupported(Exception):
    pass


@dataclass(frozen=True)
class _Use:
    uid: str
    first: tuple[str, str] | None
    last: tuple[str, str] | None
    readers: tuple[tuple[str, str], ...]
    incoming: tuple[tuple[str, str], ...]


@dataclass
class _Writer:
    use: _Use
    stage: int
    rank: tuple
    readers: list[tuple[tuple[str, str], int]]


def _single_count(domain):
    first = _first(domain)
    if (
        domain.exact is not None
        or domain.maximum is not None
        and first + domain.period > domain.maximum
    ):
        return CountDomain(1, 0, exact=first)
    return domain


def _ranges(lower, upper, guard=ALL):
    guard = meet(guard, le(lower, upper))
    if guard is not None:
        yield Activation(_single_count(guard), position(lower), position(upper))


def _bounded_ranges(lower, upper, guard=ALL, *, floor=None, ceiling=None):
    lowers = ((lower, guard),) if floor is None else choice(lower, floor, guard, maximum=True)
    for left, domain in lowers:
        uppers = (
            ((upper, domain),) if ceiling is None else choice(upper, ceiling, domain, maximum=False)
        )
        for right, active in uppers:
            yield from _ranges(left, right, active)


def _adjacent(writers, source, destination, distance, count):
    """Exact source-generation domain with no active intervening writer."""
    first = source.stage, source.rank
    last = distance + destination.stage, destination.rank
    if first >= last:
        return ()
    head, tail = None, None
    for middle in writers:
        # Every possible intermediate generation is g+e for an integer e in
        # this closed interval. The nearest e to zero gives the strongest
        # inactivity constraint. No parent generation is instantiated.
        low = source.stage - middle.stage + int(middle.rank <= source.rank)
        high = distance + destination.stage - middle.stage - int(middle.rank >= destination.rank)
        if low > high:
            continue
        if low <= 0 <= high:
            return ()  # middle(g) is active whenever source(g) is active.
        if high < 0:
            bound = -high - 1
            head = bound if head is None else min(head, bound)
        else:
            bound = -low  # g >= N-low.
            tail = bound if tail is None else max(tail, bound)
    guard = CountDomain(1, 0, exact=count) if count is not None else ALL
    return tuple(
        _bounded_ranges(
            (0, max(0, -distance)),
            (1, -1 - max(0, distance)),
            guard,
            floor=None if tail is None else (1, tail),
            ceiling=None if head is None else (0, head),
        )
    )


def _reader_activation(activation, shift):
    # Clip to the reader's own real call domain, then change source epoch from
    # writer g to reader h=g+shift. Set and Wait must use the same source h.
    for active in _bounded_ranges(
        affine(activation.first),
        affine(activation.last),
        activation.count,
        floor=(0, -shift),
        ceiling=(1, -1 - shift),
    ):
        lower, upper = affine(active.first), affine(active.last)
        yield replace(
            active,
            first=position((lower[0], lower[1] + shift)),
            last=position((upper[0], upper[1] + shift)),
        )


def _coalesce(relations):
    """Fold short-count endpoint coincidences into their general templates."""
    families = {}
    for relation in relations:
        key = (
            relation.source,
            relation.destination,
            relation.distance,
            relation.hazard,
            relation.storage,
            relation.start,
            relation.stop,
            relation.prefix,
        )
        families.setdefault(key, []).append(relation.activation)
    result = []
    for key, activations in families.items():
        templates = [active for active in activations if active.count.exact is None]
        geometries = {}
        for active in activations:
            if active.count.exact is not None:
                for template in templates:
                    if all(
                        holds(affine(a), affine(b), active.count)
                        and holds(affine(b), affine(a), active.count)
                        for a, b in ((active.first, template.first), (active.last, template.last))
                    ):
                        active = replace(active, first=template.first, last=template.last)
                        break
            geometry = active.first, active.last, active.period, active.residues
            geometries.setdefault(geometry, []).append(active.count)
        source, target, distance, hazard, storage, start, stop, prefix = key
        for (first, last, period, residues), domains in geometries.items():
            for domain in coalesce(domains):
                result.append(
                    ScopedDependence(
                        source,
                        target,
                        distance,
                        Activation(domain, first, last, period, residues),
                        hazard,
                        storage,
                        start,
                        stop,
                        prefix,
                    )
                )
    return tuple(sorted(set(result), key=repr))


def _root_parts(program, access, root, endpoints, budget):
    parts = []
    captures = access.capture_roots
    for uid in program.leaves:
        summary = access.summaries[uid]
        for cell in summary.cells:
            if captures.get((uid, cell.storage)) != root:
                continue
            offset = access.capture_offsets[uid, root]
            bias, terms, cores, count_offset, count_core = offset
            if cores or count_offset or count_core or any(stride for _, stride, _, _ in terms):
                raise _Unsupported("capture is outside the stationary private lifetime contract")
            if not cell.definitions:
                raise _Unsupported("fixed definition runs were not projected")

            def endpoint(at: Access, *, owner=uid, count=summary.constant_count):
                key = owner, at.task[1], at.position.evaluate(count)
                if key not in endpoints:
                    raise _Unsupported("a definition native endpoint was not projected")
                return endpoints[key]

            incoming, fresh = [], []
            for run in cell.definitions:
                budget.charge(1 + len(run.read_bindings) + len(run.pending_reads))
                if run.entry is None:
                    if run.last_write is not None:
                        raise _Unsupported("incoming RW is a continuation, not a fresh definition")
                    if any(binding.writer is not None for binding in run.read_bindings):
                        raise _Unsupported("incoming read has an unsupported internal producer")
                    incoming.extend(endpoint(binding.reader) for binding in run.read_bindings)
                else:
                    if any(binding.writer is None for binding in run.read_bindings):
                        raise _Unsupported("fresh definition has an unresolved incoming read")
                    fresh.append(run)
            first = endpoint(fresh[0].entry) if fresh else None
            last = endpoint(fresh[-1].last_write) if fresh else None
            readers = []
            if fresh:
                for span in fresh[-1].pending_reads:
                    if span.first != span.last:
                        raise _Unsupported("a pending reader needs an exact fixed native position")
                    readers.append(endpoint(Access(span.task, span.first)))
            use = _Use(uid, first, last, tuple(readers), tuple(incoming))
            bias += sum(offset for offset, _, _, _ in terms)
            parts.append((bias + cell.start, bias + cell.stop, use))
    return parts


def _plan_root(program, access, root, endpoints, budget):
    origin = access.origins[root].owner
    if origin.external.data or "/" not in root:
        raise _Unsupported("only actor-private local storage may reorder definitions")
    if access.loop.storage_periods[root] != 1:
        raise _Unsupported("periodic captures retain the logical lifetime contract")
    old = [dep for dep in access.obligations if dep.storage == root]
    if any(dep.last or dep.prefix for dep in old):
        raise _Unsupported("terminal or prefix memory families retain the logical contract")
    parts = _root_parts(program, access, root, endpoints, budget)
    points = sorted({point for lo, hi, _ in parts for point in (lo, hi)})
    cuts = {point: index for index, point in enumerate(points)}
    incidences = sum(cuts[hi] - cuts[lo] for lo, hi, _ in parts)
    budget.charge(len(points) + incidences)
    cells = [[] for _ in points[:-1]]
    for lo, hi, use in parts:
        for index in range(cuts[lo], cuts[hi]):
            cells[index].append(use)
    lexical = {
        op: index
        for parent in program.parents.values()
        for index, op in enumerate(parent.body.block.ops)
    }
    raw = {}
    for dep in old:
        if dep.hazard == "RAW":
            raw.setdefault(dep.destination, []).append(dep)
    relations, boundaries, changed = [], [], False
    for lo, hi, uses in zip(points[:-1], points[1:], cells, strict=True):
        writers = []
        for use in uses:
            if use.first is None:
                continue
            call = program.leaves[use.uid].call
            _, entry = access.endpoints[use.first]
            leaf = program.leaves[use.uid]
            data = next(iter(leaf.loop.actors.values()))
            native_index, native = next(
                (index, task)
                for index, task in enumerate(data.program.tasks)
                if task.uid == entry.task[1]
            )
            rank = (
                call.issue_order.data,
                lexical[call],
                entry.position.evaluate(leaf.loop.constant_count) + native.stage,
                native.issue_order,
                native_index,
            )
            writers.append(
                _Writer(use, call.stage.data, rank, [(reader, 0) for reader in use.readers])
            )
        if not writers:
            if any(use.incoming for use in uses):
                raise _Unsupported("incoming private reads have no fresh boundary occupancy")
            continue
        by_last = {writer.use.last: writer for writer in writers}
        for use in uses:
            for reader in use.incoming:
                bindings = [
                    dep for dep in raw.get(reader, ()) if dep.start <= lo and hi <= dep.stop
                ]
                budget.charge(1 + len(bindings))
                if len(bindings) != 1 or bindings[0].source not in by_last:
                    raise _Unsupported("incoming read does not bind one exported logical version")
                dep = bindings[0]
                by_last[dep.source].readers.append((reader, dep.distance))
        # Every writer in this contract exists at every parent generation.
        # Their physical (stage, rank) order therefore has these extremal
        # occupants for every positive N, including N=1. Readers stay bound
        # to their logical writer even when their call is a separate child.
        head = min(writers, key=lambda writer: (writer.stage, writer.rank))
        tail = max(writers, key=lambda writer: (writer.stage, writer.rank))
        if any(shift < 0 for _, shift in tail.readers):
            raise _Unsupported("negative logical reader shift needs a boundary activation")
        domain = (
            CountDomain(1, 0, minimum=1)
            if program.count is None
            else CountDomain(1, 0, exact=program.count)
        )
        boundaries.append(
            CellSummary(
                root,
                lo,
                hi,
                domain,
                (),
                Access(head.use.first, Position(False, 0)),
                Access(tail.use.last, Position(True, 0)),
                tuple(
                    ReadSpan.one(Access(reader, Position(True, 0)))
                    for reader, shift in tail.readers
                    if shift == 0
                ),
            )
        )
        stages = [writer.stage for writer in writers]
        span = max(stages) - min(stages)
        width = 2 * span + 2
        budget.charge(len(writers) ** 3 * width)
        logical_pairs = {
            (dep.source, dep.destination, dep.distance)
            for dep in old
            if dep.hazard == "WAW" and dep.start <= lo and hi <= dep.stop
        }
        for source in writers:
            for target in writers:
                for distance in range(-span, span + 2):
                    domains = _adjacent(writers, source, target, distance, program.count)
                    if not domains:
                        continue
                    changed |= (source.use.last, target.use.first, distance) not in logical_pairs
                    for active in domains:
                        relations.append(
                            ScopedDependence(
                                source.use.last,
                                target.use.first,
                                distance,
                                active,
                                "WAW",
                                root,
                                lo,
                                hi,
                            )
                        )
                        for reader, shift in source.readers:
                            for reader_active in _reader_activation(active, shift):
                                relations.append(
                                    ScopedDependence(
                                        reader,
                                        target.use.first,
                                        distance - shift,
                                        reader_active,
                                        "WAR",
                                        root,
                                        lo,
                                        hi,
                                    )
                                )
                    require(
                        len(relations) <= MAX_LIFETIME_RELATIONS,
                        "scope lifetime relation budget exceeded",
                    )
    return (_coalesce(relations), tuple(boundaries)) if changed else None


def plan_lifetimes(program, access) -> LifetimePlan:
    """Return complete dependencies and explicit root-level fallback reasons.

    Unaffected roots preserve their original Dependence objects and fast path.
    Logical RAW edges are always retained. Only an opted-in private root's
    independent occupancy WAW/WAR families can change.
    """
    roots = sorted(getattr(access, "definition_roots", ()))
    if not roots or program.count == 0:
        return LifetimePlan(tuple(access.obligations), (), (), 0)
    endpoints = {
        (uid, site.task[1], site.position.evaluate(access.summaries[uid].constant_count)): key
        for key, (uid, site) in access.endpoints.items()
    }
    budget, replacement, fallbacks, boundaries = _Budget(), {}, [], []
    for root in roots:
        try:
            planned = _plan_root(program, access, root, endpoints, budget)
        except _Unsupported as error:
            fallbacks.append((root, str(error)))
            continue
        if planned is not None:
            replacement[root] = planned[0]
            boundaries.extend(planned[1])
    dependencies = tuple(
        dep
        for dep in access.obligations
        if dep.storage not in replacement or dep.hazard not in {"WAW", "WAR"}
    ) + tuple(dep for root in sorted(replacement) for dep in replacement[root])
    return LifetimePlan(
        dependencies, tuple(sorted(replacement)), tuple(fallbacks), budget.visits, tuple(boundaries)
    )


def propose_lifetimes(program, access) -> tuple[Dependence | ScopedDependence, ...]:
    """Convenience adapter; plan_lifetimes also exposes fallback diagnostics."""
    return plan_lifetimes(program, access).dependencies

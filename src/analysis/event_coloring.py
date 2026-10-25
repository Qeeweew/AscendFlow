"""Finite event mappings from frozen consumption and whole-word certificates.

Direct exclusive/word constructions need no solver. All residual protocol
options and IDs form one existential problem; no Python numbering search or
source/lifetime query participates in allocation.
"""

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import combinations
from time import perf_counter

from z3 import And, Distinct, If, Implies, Int, Or, Sum

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind, require
from .event_contracts import CROSS_BACKLOG, EventDemand, EventIdPools
from .event_frontier_contracts import FrozenEventRelations
from .smt import finite_model


class NoEventAssignment(CompileError):
    """The complete frozen choice problem has no finite mapping."""


def namespaces(demand):
    return (None,) if demand.backlog is not None else demand.banks


def relation(problem, a, b):
    return problem.relations.get((min(a, b), max(a, b)))


@dataclass(frozen=True)
class _ColoringProblem:
    demands: tuple
    reservations: tuple
    relations: object
    id_pools: EventIdPools
    counter_packs: tuple[frozenset[int], ...] = ()
    counter_reuses: object = FrozenEventRelations(())
    counter_time_packs: tuple[tuple[frozenset[int], ...], ...] = ()


def _peer_demands(demands, peers):
    """Quotient only allocation variables; never merge physical credits.

    Peers occupy separate private banks. A unit's number must be available
    in every one of them; all original per-bank reuse constraints survive.
    """
    roots = list(range(len(demands)))
    for a, b in peers:
        left, right = demands[a], demands[b]
        require(
            left.width == right.width
            and left.backlog is right.backlog is None
            and not set(left.banks).intersection(right.banks),
            "peer equality must retain distinct private event namespaces",
        )
        roots[b] = roots[a]
    groups = {}
    for number, root in enumerate(roots):
        groups.setdefault(root, []).append(number)
    groups = tuple(tuple(g) for g in groups.values())
    merged = tuple(
        EventDemand(
            tuple(m for n in group for m in demands[n].members),
            demands[group[0]].width,
            tuple(dict.fromkeys(b for n in group for b in demands[n].banks)),
            demands[group[0]].backlog,
            demands[group[0]].persistent,
        )
        for group in groups
    )
    return merged, groups


@dataclass(frozen=True)
class _PeerRelations:
    """Numeric projection queried only for actual residual conflicts.

    Do not materialize all quotient pairs before the exclusive/word fast
    paths. The original frozen permissions and every private bank survive.
    """

    original: object
    groups: tuple
    banks: tuple

    def local_conflicts(self, group):
        """Project missing row permissions through every original private bank.

        Expand actual conflicts from the frozen masks. A peer equality changes
        only the numeric variable; no original bank or required permission is
        dropped. Ordinary numeric mappings retain the finite fallback below.
        """
        row = getattr(self.original, "row_mask", None)
        if row is None:
            return None
        owners = {old: number for number in group for old in self.groups[number]}
        banks = {}
        for old in owners:
            for bank in self.banks[old]:
                banks[bank] = banks.get(bank, 0) | (1 << old)
        result = set()
        for old, number in owners.items():
            poll("event conflict projection", credits=len(owners), row=old)
            common = 0
            for bank in self.banks[old]:
                common |= banks[bank]
            missing = common & ~row(old) & ~(1 << old)
            while missing:
                bit = missing & -missing
                other = bit.bit_length() - 1
                missing ^= bit
                target = owners[other]
                key = min(number, target), max(number, target)
                if number == target or key in result:
                    continue
                # Keep ordinary canonical lookup semantics even for an
                # explicitly supplied asymmetric numeric row collection.
                if self.original.get((min(old, other), max(old, other))) is None:
                    result.add(key)
        return result

    def get(self, key, default=None):
        a, b = key
        common = False
        for x in self.groups[a]:
            for y in self.groups[b]:
                if self.banks[x].isdisjoint(self.banks[y]):
                    continue
                common = True
                if self.original.get((min(x, y), max(x, y))) is None:
                    return default
        return () if common else default


@dataclass(frozen=True)
class _CounterRelations:
    """Keep counter credit rows in their original numeric identity space."""

    original: FrozenEventRelations
    groups: tuple

    def __bool__(self):
        return bool(self.original)

    def conflicts(self, members):
        owners = {self.groups[n][0]: n for n in members}
        mask = sum(1 << old for old in owners)
        missing = set()
        for old, number in owners.items():
            blocked = mask & ~self.original.row_mask(old) & ~(1 << old)
            while blocked:
                bit = blocked & -blocked
                blocked ^= bit
                other = owners[bit.bit_length() - 1]
                missing.add((min(number, other), max(number, other)))
        yield from sorted(missing)


def _equal_numbers(problem):
    demands, groups = _peer_demands(problem.demands, problem.peers)
    owners = {n: i for i, group in enumerate(groups) for n in group}
    packs = tuple(frozenset(owners[n] for n in c.resources) for c in problem.counter_groups)
    time_packs = tuple(
        tuple(frozenset(owners[n] for n in group) for group in c.groups)
        for c in getattr(problem, "counter_compositions", ())
    )
    # Mode-2 has no peer quotient. Keep its original rows, including when
    # local peer equalities shift the surrounding numeric variable indices.
    time = _CounterRelations(problem.counter_reuses, groups)
    relations = _PeerRelations(
        problem.relations,
        groups,
        tuple(frozenset(namespaces(d)) for d in problem.demands),
    )
    return _ColoringProblem(
        demands, problem.reservations, relations, problem.id_pools, packs, time, time_packs
    ), groups


def counter_group_usage(demands, members, reuses, packs, *, packing=True, time_packs=()):
    """Check a whole counter set using frozen numeric permissions only.

    Pairwise noninterference composes for time reuse: at most one complete
    credit has outstanding contributions. FIFO packing instead needs one
    whole-word certificate and the sum of credit bounds. Frozen temporal
    compositions take the maximum of certified FIFO-child sums; pairwise
    permissions alone do not authorize an arbitrary mixed set.
    """
    members = tuple(sorted(members))
    if reuses.permits_group(members):
        return "time", max(demands[n].backlog for n in members)
    if packing and any(set(members) <= group for group in packs):
        return "fifo", sum(demands[n].backlog for n in members)
    if packing:
        for groups in time_packs:
            if set(members) <= frozenset().union(*groups):
                return "time_fifo", max(
                    sum(demands[n].backlog for n in members if n in g) for g in groups
                )
    return None


def counter_fifo_partition(demands, packs, members=None):
    """A concrete finite upper bound using already certified whole words.

    Removing messages preserves a CounterGroup certificate. This greedy
    partition is a sufficient mapping proposal, never a minimum-ID claim.
    It neither queries a domain nor derives an event consumption relation.
    """
    remaining = {
        n
        for n, demand in enumerate(demands)
        if demand.backlog is not None and (members is None or n in members)
    }
    if any(demands[n].width != 1 or demands[n].backlog > CROSS_BACKLOG for n in remaining):
        return None
    result = []
    pending = [(-len(pack), tuple(sorted(pack)), pack) for pack in packs]
    heapify(pending)
    while pending:
        size, key, pack = heappop(pending)
        selected = frozenset(remaining.intersection(pack))
        if len(selected) < 2:
            continue
        if len(selected) != -size:
            # Old gains are upper bounds. Refresh them until the largest
            # remaining word is current; a large consumed word must not
            # outrank a smaller word that covers more still-live credits.
            heappush(pending, (-len(selected), key, pack))
            continue
        if sum(demands[n].backlog for n in selected) > CROSS_BACKLOG:
            continue
        result.append(selected)
        remaining.difference_update(selected)
    result.extend(frozenset((n,)) for n in sorted(remaining))
    return tuple(result)


def _structured_counter_colors(problem, group, reserved):
    partition = counter_fifo_partition(problem.demands, problem.counter_packs, group)
    available = tuple(flag for flag in problem.id_pools.cross if flag not in reserved.get(None, ()))
    if partition is None or len(partition) > len(available):
        return None
    return {
        number: (flag,) for pack, flag in zip(partition, available, strict=False) for number in pack
    }


def _components(demands):
    roots, banks = list(range(len(demands))), {}

    def root(number):
        while roots[number] != number:
            roots[number] = roots[roots[number]]
            number = roots[number]
        return number

    for number, demand in enumerate(demands):
        for bank in namespaces(demand):
            prior = banks.setdefault(bank, number)
            a, b = root(prior), root(number)
            roots[max(a, b)] = min(a, b)
    groups = {}
    for number in range(len(demands)):
        groups.setdefault(root(number), []).append(number)
    yield from (tuple(group) for group in groups.values())


def _available(pools, banks, occupied):
    return tuple(
        sorted(
            set.intersection(
                *(set(pools.ids(bank)).difference(occupied.get(bank, ())) for bank in banks)
            )
        )
    )


def _exclusive(demands, group, reserved, pools):
    occupied = {b: set(ids) for b, ids in reserved.items()}
    result = {}
    for number in group:
        d = demands[number]
        choices = _available(pools, namespaces(d), occupied)
        if len(choices) < d.width:
            return None
        ids = next(
            (
                tuple(range(base, base + d.width))
                for base in choices
                if all(flag in choices for flag in range(base, base + d.width))
            ),
            tuple(choices[: d.width]),
        )
        result[number] = ids
        for bank in namespaces(d):
            occupied.setdefault(bank, set()).update(ids)
    return result


def exclusive_pressure_banks(demands, reservations, peers, id_pools):
    """Banks whose joint exclusive placement fails the finite fast path.

    Equal-number peers couple otherwise private banks. Per-bank counts can
    all fit while their available-number intersections do not. This is a
    request for reuse premises, never a claim of physical infeasibility.
    """
    demands, _ = _peer_demands(demands, peers)
    reserved = {}
    for r in reservations:
        reserved.setdefault(r.bank, set()).add(r.event)
    pressured = set()
    for group in _components(demands):
        if _exclusive(demands, group, reserved, id_pools) is None:
            pressured.update(bank for n in group for bank in namespaces(demands[n]))
    return pressured


def fixed_credit_pressure_banks(problem):
    """A finite clique of whole-credit constraints requests phase precision.

    This is a sufficient failure of the frozen coarse mapping, never a
    physical-occupancy minimum. Do not enumerate compatible member subsets.
    """
    groups, reserved = {}, {}
    for r in problem.reservations:
        reserved.setdefault(r.bank, set()).add(r.event)
    for number, demand in enumerate(problem.demands):
        if demand.backlog is None:
            for bank in demand.banks:
                groups.setdefault(bank, []).append(number)
    result = set()
    for bank, members in groups.items():
        clique, width = [], 0
        capacity = len(set(problem.id_pools.ids(bank)).difference(reserved.get(bank, ())))
        for number in members:
            if all(relation(problem, number, prior) is None for prior in clique):
                clique.append(number)
                width += problem.demands[number].width
                if width > capacity:
                    if any(len(problem.demands[n].members) > 1 for n in members):
                        result.add(bank)
                    break
    return frozenset(result)


def _exclusive_overfull(problem, group, reserved):
    """A real exclusive bank count avoids constructing a dense clique."""
    counts = {}
    for number in group:
        for bank in namespaces(problem.demands[number]):
            counts[bank] = counts.get(bank, 0) + problem.demands[number].width
    return any(
        count > len(set(problem.id_pools.ids(bank)).difference(reserved.get(bank, ())))
        for bank, count in counts.items()
    )


def _local_conflicts(problem, group):
    sparse = problem.relations.local_conflicts(group)
    if sparse is not None:
        yield from sorted(sparse)
        return
    banks = {}
    for number in group:
        for bank in namespaces(problem.demands[number]):
            banks.setdefault(bank, []).append(number)
    seen = set()
    for members in banks.values():
        for a, b in combinations(members, 2):
            pair = min(a, b), max(a, b)
            if pair not in seen:
                seen.add(pair)
                if relation(problem, *pair) is None:
                    yield pair


def _local_clique_overfull(problem, group, reserved, *, reuse):
    """Reject a bounded numeric witness before expanding a dense graph.

    All vertices in each row share this bank's finite palette. Peer coupling
    may supply another reason they need different numbers; this is still a
    constraint-graph witness, not a physical-occupancy minimum.
    """
    banks = {}
    for number in group:
        for bank in namespaces(problem.demands[number]):
            banks.setdefault(bank, []).append(number)
    for bank, members in banks.items():
        capacity = len(set(problem.id_pools.ids(bank)).difference(reserved.get(bank, ())))
        clique, width = [], 0
        for number in members:
            if all(not reuse or relation(problem, number, other) is None for other in clique):
                clique.append(number)
                width += problem.demands[number].width
                if width > capacity:
                    return True
    return False


def _prepare(problem):
    numeric, peers = _equal_numbers(problem)
    reserved = {}
    for item in numeric.reservations:
        reserved.setdefault(item.bank, set()).add(item.event)
    direct, residual = {}, []
    for group in _components(numeric.demands):
        choices = {
            n: _available(numeric.id_pools, namespaces(numeric.demands[n]), reserved) for n in group
        }
        if any(numeric.demands[n].width > len(choices[n]) for n in group):
            return numeric, peers, None, ()
        if any(
            numeric.demands[n].backlog is not None and numeric.demands[n].backlog > CROSS_BACKLOG
            for n in group
        ):
            return numeric, peers, None, ()
        colors = _exclusive(numeric.demands, group, reserved, numeric.id_pools)
        cross = numeric.demands[group[0]].backlog is not None
        if colors is None and cross and not numeric.counter_reuses:
            colors = _structured_counter_colors(numeric, group, reserved)
        if colors is not None:
            direct.update(colors)
        elif not cross and _local_clique_overfull(numeric, group, reserved, reuse=True):
            return numeric, peers, None, ()
        else:
            residual.append((group, choices, not _exclusive_overfull(numeric, group, reserved)))
    return numeric, peers, direct, tuple(residual)


def _counter_constraints(problem, group, lanes, prefix, add):
    # Each certificate admits every filtered subset of its actual word(s).
    # A mode selects an existing certificate, never a completion truth value.
    certificates = tuple((pack,) for pack in problem.counter_packs) + problem.counter_time_packs
    owners = {n: [] for n in group}
    for number, groups in enumerate(certificates, 1):
        for n in frozenset().union(*groups).intersection(group):
            owners[n].append(number)
    conflicts = tuple(problem.counter_reuses.conflicts(group))
    if not certificates:
        # With time separation alone, every conflicting pair must use distinct
        # numbers. Pool membership is already asserted by the caller, so these
        # lane inequalities are exactly the per-ID exclusions below, without
        # repeating the same graph for every hardware counter.
        for a, b in conflicts:
            add(
                f"counter.time.{a}.{b}",
                And(*(left != right for left in lanes[a] for right in lanes[b])),
            )
        return 0
    flags = problem.id_pools.cross
    for flag in flags:
        poll("counter certificate constraints", members=len(group), event=flag)
        members = {n: Or(*(value == flag for value in lanes[n])) for n in group}
        mode = Int(f"{prefix}_counter_{flag}")
        add(f"counter.{flag}.mode", And(mode >= 0, mode <= len(certificates)))
        for n in group:
            add(
                f"counter.{flag}.member.{n}",
                Implies(members[n], Or(mode == 0, *(mode == c for c in owners[n]))),
            )
            if problem.demands[n].backlog > CROSS_BACKLOG:
                add(f"counter.{flag}.backlog.{n}", ~members[n])
        for a, b in conflicts:
            add(f"counter.{flag}.time.{a}.{b}", Implies(mode == 0, Or(~members[a], ~members[b])))
        for number, groups in enumerate(certificates, 1):
            for child, subset in enumerate(groups):
                terms = [
                    If(members[n], problem.demands[n].backlog, 0) for n in subset if n in members
                ]
                add(
                    f"counter.{flag}.word.{number}.{child}",
                    Implies(mode == number, Sum(terms) <= CROSS_BACKLOG),
                )
    return len(flags)


def _constraints(prepared, option, selected, constraints, objectives):
    problem, _peers, direct, residual = prepared
    active = selected == option
    prefix = f"event_{option}"

    def add(name, condition):
        constraints.append((f"option.{option}.{name}", Implies(active, condition)))

    if direct is None:
        add("capacity", False)
        return {}, 0
    lanes, variables = {}, 0
    for component, (group, choices, exclusive_possible) in enumerate(residual):
        poll("event ID constraints", component=component, credits=len(group))
        current = {
            n: tuple(Int(f"{prefix}_{n}_{lane}") for lane in range(problem.demands[n].width))
            for n in group
        }
        lanes.update(current)
        variables += sum(map(len, current.values()))
        for n in group:
            for lane, value in enumerate(current[n]):
                add(f"pool.{n}.{lane}", Or(*(value == flag for flag in choices[n])))
            add(f"lanes.{n}", Distinct(*current[n]))
        if exclusive_possible:
            banks = {}
            for n in group:
                for bank in namespaces(problem.demands[n]):
                    banks.setdefault(bank, []).extend(current[n])
            exclusive = And(*(Distinct(*values) for values in banks.values()))
            objectives.append(If(active, If(exclusive, 0, 1), 0))
        if problem.demands[group[0]].backlog is not None:
            variables += _counter_constraints(problem, group, current, f"{prefix}_{component}", add)
        else:
            conflicts = set(_local_conflicts(problem, group))
            for a, b in sorted(conflicts):
                add(
                    f"separation.{a}.{b}",
                    And(*(left != right for left in current[a] for right in current[b])),
                )
            palette = choices[group[0]]
            if all(choices[n] == palette for n in group):
                # All numbers in this component have identical permissions.
                # Fix one actual conflict clique by a palette permutation;
                # every legal mapping has such an equivalent representative.
                # This removes color-name symmetry, without finding a coloring
                # or changing any frozen lifetime/phase constraint.
                clique = []
                for n in group:
                    if all((min(n, m), max(n, m)) in conflicts for m in clique):
                        clique.append(n)
                anchor = tuple(value for n in clique for value in current[n])
                if len(anchor) <= len(palette):
                    for lane, (value, flag) in enumerate(zip(anchor, palette)):
                        add(f"palette.{component}.{lane}", value == flag)
    return lanes, variables


def color_event_resources(problem, *, alternatives=()):
    """Select a frozen protocol and all residual IDs in one finite problem.

    Return the actual lane mappings and selected option. The first option is
    preferred, then exclusive mappings in its independent bank components.
    SAT provides only a mapping under the frozen premises, never those premises.
    """
    started = perf_counter()
    candidates = (problem, *alternatives)
    preferred = _prepare(problem)
    prepared = [preferred]
    checks = variable_count = constraint_count = 0
    if preferred[2] is not None and not preferred[3]:
        option, direct, model, symbolic = 0, preferred[2], None, {}
    else:
        prepared.extend(_prepare(candidate) for candidate in alternatives)
        selected = Int("event_protocol_option")
        constraints = [("protocol.option", And(selected >= 0, selected < len(candidates)))]
        objectives = [selected]
        symbols = []
        variable_count = 1
        for number, item in enumerate(prepared):
            lanes, count = _constraints(item, number, selected, constraints, objectives)
            symbols.append(lanes)
            variable_count += count
        checks, constraint_count = 1, len(constraints)
        work = getattr(problem, "work", None)
        if work is not None:
            work.allocation_queries = checks
            work.allocation_variables = variable_count
            work.allocation_constraints = constraint_count
        try:
            model, core = finite_model(constraints, objectives)
        finally:
            if work is not None:
                work.allocation_seconds = perf_counter() - started
        if model is None:
            capacity = max(
                (
                    len(problem.id_pools.ids(bank))
                    for demand in problem.demands
                    for bank in namespaces(demand)
                ),
                default=0,
            )
            raise NoEventAssignment(
                f"no assignment within capacity {capacity} under the frozen protocol options and "
                "established reuse constraints; this is not a hardware minimum",
                kind=ErrorKind.UNSUPPORTED,
                phase="event-id-allocation",
                witness=core,
            )
        option = model.eval(selected).as_long()
        direct, symbolic = prepared[option][2], symbols[option]
    result = dict(direct)
    for n, lanes in symbolic.items():
        result[n] = tuple(model.eval(value).as_long() for value in lanes)
    expanded = {}
    for unit, peers in enumerate(prepared[option][1]):
        for number in peers:
            expanded[number] = result[unit]
    work = getattr(problem, "work", None)
    if work is not None:
        work.allocation_queries = checks
        work.allocation_variables = variable_count
        work.allocation_constraints = constraint_count
        work.allocation_seconds = perf_counter() - started
    return tuple(expanded[n] for n in range(len(expanded))), option

"""Deterministic finite event coloring over frozen reuse constraints.

Distinct namespaces are independent. Prefer exclusive contiguous mappings;
reuse is considered only in a component whose exclusive placement fails.
Search never queries lifetime facts or changes an instruction's position.
"""

from dataclasses import dataclass
from itertools import combinations

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind, require
from .event_contracts import CROSS_BACKLOG, EventDemand, EventIdPools

MAX_COLORING_NODES = 16384


class NoEventAssignment(CompileError):
    """Fixed protocol constraints exhausted; another frozen option may apply."""

    def __init__(self, *args, nodes=0, **kwargs):
        self.nodes = nodes
        super().__init__(*args, **kwargs)


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
    counter_reuses: frozenset[tuple[int, int]] = frozenset()
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
    demands: tuple
    packs: tuple
    reuses: frozenset
    time_packs: tuple

    def get(self, key, default=None):
        a, b = key
        if (a, b) in self.reuses:
            return ()
        if self.demands[a].backlog is not None:
            return (
                ()
                if (
                    any({a, b} <= pack for pack in self.packs)
                    or any({a, b} <= frozenset().union(*groups) for groups in self.time_packs)
                )
                else default
            )
        common = False
        for x in self.groups[a]:
            for y in self.groups[b]:
                if self.banks[x].isdisjoint(self.banks[y]):
                    continue
                common = True
                if self.original.get((min(x, y), max(x, y))) is None:
                    return default
        return () if common else default


def _equal_numbers(problem):
    demands, groups = _peer_demands(problem.demands, problem.peers)
    owners = {n: i for i, group in enumerate(groups) for n in group}
    packs = tuple(frozenset(owners[n] for n in c.resources) for c in problem.counter_groups)
    time_packs = tuple(
        tuple(frozenset(owners[n] for n in group) for group in c.groups)
        for c in getattr(problem, "counter_compositions", ())
    )
    # Mode-2 has no peer quotient: each original temporal pair remains one
    # numeric pair. Local peers must satisfy all original private-bank rows.
    time = frozenset(
        (min(owners[a], owners[b]), max(owners[a], owners[b])) for a, b in problem.counter_reuses
    )
    relations = _PeerRelations(
        problem.relations,
        groups,
        tuple(frozenset(namespaces(d)) for d in problem.demands),
        demands,
        packs,
        time,
        time_packs,
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
    if all((a, b) in reuses for a, b in combinations(members, 2)):
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
    for pack in sorted(packs, key=lambda values: (-len(values), tuple(sorted(values)))):
        selected = frozenset(remaining.intersection(pack))
        if len(selected) < 2 or sum(demands[n].backlog for n in selected) > CROSS_BACKLOG:
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


def _color(problem, group, reserved, capacity, node_limit, *, reuse, counter_packing=False):
    demands = problem.demands
    cross = demands[group[0]].backlog is not None
    vertices = tuple((number, lane) for number in group for lane in range(demands[number].width))
    adjacency = [set() for _ in vertices]
    allowed = []
    for number, _ in vertices:
        allowed.append(_available(problem.id_pools, namespaces(demands[number]), reserved))
    # Unused IDs with the same admissible vertices are interchangeable.
    # Reservations can distinguish numbers across joint banks, so retain the
    # whole allowed-column signature rather than just a count of used IDs.
    signatures = {
        flag: tuple(flag in choices for choices in allowed) for flag in set().union(*allowed)
    }
    for i, j in combinations(range(len(vertices)), 2):
        a, b = vertices[i][0], vertices[j][0]
        common = set(namespaces(demands[a])).intersection(namespaces(demands[b]))
        permitted = (
            reuse
            and relation(problem, a, b) is not None
            and (not cross or counter_packing or (min(a, b), max(a, b)) in problem.counter_reuses)
        )
        if a == b or common and (not reuse or not permitted):
            adjacency[i].add(j)
            adjacency[j].add(i)
    # This is a lower bound for the *supported constraint graph*, not a
    # witness of joint physical overlap in some real execution.
    clique = []
    for v in sorted(range(len(vertices)), key=lambda v: (-len(adjacency[v]), v)):
        if all(other in adjacency[v] for other in clique):
            clique.append(v)
    if len(clique) > capacity:
        return None, 0
    colors, nodes = {}, 0

    def visit():
        nonlocal nodes
        nodes += 1
        poll("event coloring", nodes=nodes, vertices=len(vertices))
        if nodes > node_limit:
            raise CompileError(
                f"event coloring reached its {MAX_COLORING_NODES}-node work limit within capacity {capacity}; "
                "no infeasibility claim or schedule fallback",
                kind=ErrorKind.ANALYSIS_LIMIT,
                phase="event-id-coloring",
            )
        if len(colors) == len(vertices):
            return True
        v = max(
            (v for v in range(len(vertices)) if v not in colors),
            key=lambda v: (
                len({colors[n] for n in adjacency[v] if n in colors}),
                len(adjacency[v]),
                -v,
            ),
        )
        forbidden = {colors[n] for n in adjacency[v] if n in colors}
        number, lane = vertices[v]
        bank_set = set(namespaces(demands[number]))
        used = {
            color
            for n, color in colors.items()
            if bank_set.intersection(namespaces(demands[vertices[n][0]]))
        }
        choices = [c for c in allowed[v] if c not in forbidden]
        # A contiguous ring needs no general selector. Otherwise prefer a
        # free physical slot before aliasing another complete credit.
        previous = next(
            (
                colors[n] + 1
                for n, item in enumerate(vertices)
                if item == (number, lane - 1) and n in colors
            ),
            None,
        )
        choices.sort(key=lambda c: (c != previous if previous is not None else False, c in used, c))
        occupied = set(colors.values())
        tried_unused = set()
        for color in choices:
            if color not in occupied:
                signature = signatures[color]
                if signature in tried_unused:
                    continue
                tried_unused.add(signature)
            if cross:
                members = {vertices[n][0] for n, c in colors.items() if c == color} | {number}
                usage = counter_group_usage(
                    demands,
                    members,
                    problem.counter_reuses,
                    problem.counter_packs,
                    packing=counter_packing,
                    time_packs=problem.counter_time_packs,
                )
                if usage is None or usage[1] > CROSS_BACKLOG:
                    continue
            colors[v] = color
            if visit():
                return True
            del colors[v]
        return False

    if not visit():
        return None, nodes
    return {
        n: tuple(colors[i] for i, (number, _) in enumerate(vertices) if number == n) for n in group
    }, nodes


def color_event_resources(problem, *, node_limit=None):
    node_limit = MAX_COLORING_NODES if node_limit is None else node_limit
    require(0 <= node_limit <= MAX_COLORING_NODES, "invalid remaining event search work")
    problem, peers = _equal_numbers(problem)
    reserved = {}
    for r in problem.reservations:
        reserved.setdefault(r.bank, set()).add(r.event)
    result = {}
    nodes = 0
    for group in _components(problem.demands):
        cross = problem.demands[group[0]].backlog is not None
        capacity = len(
            set().union(
                *(
                    problem.id_pools.ids(bank)
                    for number in group
                    for bank in namespaces(problem.demands[number])
                )
            )
        )
        colors = _exclusive(problem.demands, group, reserved, problem.id_pools)
        if colors is None and not _exclusive_overfull(problem, group, reserved):
            # Greedy placement can fail even when a different exclusive
            # mapping fits all peer banks. Exhaust that finite constraint
            # graph before enabling any optional reuse relation.
            colors, work = _color(
                problem, group, reserved, capacity, node_limit - nodes, reuse=False
            )
            nodes += work
        if colors is None and cross and not problem.counter_reuses:
            colors = _structured_counter_colors(problem, group, reserved)
        if colors is None:
            colors, work = _color(
                problem, group, reserved, capacity, node_limit - nodes, reuse=True
            )
            nodes += work
        if colors is None and cross:
            colors = _structured_counter_colors(problem, group, reserved)
            if colors is None:
                colors, work = _color(
                    problem,
                    group,
                    reserved,
                    capacity,
                    node_limit - nodes,
                    reuse=True,
                    counter_packing=True,
                )
                nodes += work
        if colors is None:
            first = problem.demands[group[0]]
            names = [m.name for n in group for m in problem.demands[n].members]
            names += [
                problem.demands[n].persistent.name
                for n in group
                if problem.demands[n].persistent is not None
            ]
            raise NoEventAssignment(
                f"no assignment within capacity {capacity} under the selected virtual protocol and "
                "established reuse constraints; this is not a hardware minimum",
                kind=ErrorKind.UNSUPPORTED,
                phase="event-id-coloring",
                location=str(first.members[0].sources[0].operation.location)
                if first.members
                else None,
                witness=(
                    f"credits={names}",
                    f"reservations={reserved}",
                    "missing reuse proof, protocol demand, or joint-bank constraint; no new wait",
                ),
                nodes=nodes,
            )
        result.update(colors)
    expanded = {}
    for unit, group in enumerate(peers):
        for number in group:
            expanded[number] = result[unit]
    return tuple(expanded[n] for n in range(len(expanded))), nodes

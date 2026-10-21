"""Deterministic finite event coloring over frozen reuse constraints.

Distinct namespaces are independent. Prefer exclusive contiguous mappings;
reuse is considered only in a component whose exclusive placement fails.
Search never queries lifetime facts or changes an instruction's position.
"""

from dataclasses import dataclass
from itertools import combinations

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind, require
from .event_contracts import CROSS_BACKLOG, LOCAL_CAPACITY, SUPPORTED_CROSS_IDS, EventDemand

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
    relations: dict
    counter_packs: tuple[frozenset[int], ...] = ()
    counter_reuses: frozenset[tuple[int, int]] = frozenset()


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


def _equal_numbers(problem):
    demands, groups = _peer_demands(problem.demands, problem.peers)
    owners = {n: i for i, group in enumerate(groups) for n in group}
    packs = tuple(frozenset(owners[n] for n in c.resources) for c in problem.counter_groups)
    relations = {}
    time = set()
    for a, b in combinations(range(len(groups)), 2):
        common = set(namespaces(demands[a])).intersection(namespaces(demands[b]))
        if not common:
            continue
        permissions = [
            relation(problem, x, y)
            for x in groups[a]
            for y in groups[b]
            if set(namespaces(problem.demands[x])).intersection(namespaces(problem.demands[y]))
        ]
        temporal = demands[a].backlog is not None and all(
            (min(x, y), max(x, y)) in problem.counter_reuses for x in groups[a] for y in groups[b]
        )
        packing = demands[a].backlog is None or any({a, b} <= pack for pack in packs)
        if temporal:
            time.add((a, b))
        if temporal or packing and all(permission is not None for permission in permissions):
            relations[a, b] = ()
    return _ColoringProblem(
        demands, problem.reservations, relations, packs, frozenset(time)
    ), groups


def counter_group_usage(demands, members, reuses, packs, *, packing=True):
    """Check a whole counter set using frozen numeric permissions only.

    Pairwise noninterference composes for time reuse: at most one complete
    credit has outstanding contributions. FIFO packing instead needs one
    whole-word certificate and the sum of credit bounds. Mixed sets without
    either complete premise are unsupported.
    """
    members = tuple(sorted(members))
    if all((a, b) in reuses for a, b in combinations(members, 2)):
        return "time", max(demands[n].backlog for n in members)
    if packing and any(set(members) <= group for group in packs):
        return "fifo", sum(demands[n].backlog for n in members)
    return None


def _components(demands):
    pending = set(range(len(demands)))
    while pending:
        first = min(pending)
        pending.remove(first)
        group, banks = [first], set(namespaces(demands[first]))
        changed = True
        while changed:
            changed = False
            for number in sorted(pending):
                own = set(namespaces(demands[number]))
                if banks.intersection(own):
                    pending.remove(number)
                    group.append(number)
                    banks.update(own)
                    changed = True
        yield tuple(sorted(group))


def _exclusive(demands, group, reserved, capacity):
    occupied = {b: set(ids) for b, ids in reserved.items()}
    result = {}
    for number in group:
        d = demands[number]
        blocked = set().union(*(occupied.get(b, set()) for b in namespaces(d)))
        choices = [flag for flag in range(capacity) if flag not in blocked]
        if len(choices) < d.width:
            return None
        ids = next(
            (
                tuple(range(base, base + d.width))
                for base in range(capacity - d.width + 1)
                if all(flag not in blocked for flag in range(base, base + d.width))
            ),
            tuple(choices[: d.width]),
        )
        result[number] = ids
        for bank in namespaces(d):
            occupied.setdefault(bank, set()).update(ids)
    return result


def exclusive_pressure_banks(demands, reservations, peers):
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
        cross = demands[group[0]].backlog is not None
        capacity = len(SUPPORTED_CROSS_IDS) if cross else LOCAL_CAPACITY
        if _exclusive(demands, group, reserved, capacity) is None:
            pressured.update(bank for n in group for bank in namespaces(demands[n]))
    return pressured


def _color(problem, group, reserved, capacity, node_limit, *, reuse, counter_packing=False):
    demands = problem.demands
    cross = demands[group[0]].backlog is not None
    vertices = tuple((number, lane) for number in group for lane in range(demands[number].width))
    adjacency = [set() for _ in vertices]
    allowed = []
    for number, _ in vertices:
        blocked = set().union(*(reserved.get(b, set()) for b in namespaces(demands[number])))
        allowed.append(tuple(flag for flag in range(capacity) if flag not in blocked))
    for i, j in combinations(range(len(vertices)), 2):
        a, b = vertices[i][0], vertices[j][0]
        common = set(namespaces(demands[a])).intersection(namespaces(demands[b]))
        permitted = relation(problem, a, b) is not None and (
            not cross or counter_packing or (min(a, b), max(a, b)) in problem.counter_reuses
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
        for color in choices:
            if cross:
                members = {vertices[n][0] for n, c in colors.items() if c == color} | {number}
                usage = counter_group_usage(
                    demands,
                    members,
                    problem.counter_reuses,
                    problem.counter_packs,
                    packing=counter_packing,
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
        capacity = len(SUPPORTED_CROSS_IDS) if cross else LOCAL_CAPACITY
        colors = _exclusive(problem.demands, group, reserved, capacity)
        if colors is None:
            # Greedy placement can fail even when a different exclusive
            # mapping fits all peer banks. Exhaust that finite constraint
            # graph before enabling any optional reuse relation.
            colors, work = _color(
                problem, group, reserved, capacity, node_limit - nodes, reuse=False
            )
            nodes += work
        if colors is None:
            colors, work = _color(
                problem, group, reserved, capacity, node_limit - nodes, reuse=True
            )
            nodes += work
        if colors is None and cross:
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

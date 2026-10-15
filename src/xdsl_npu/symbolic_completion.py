"""Completion facts over symbolic generation intervals, without time expansion.

A fact states that one source precedes every instance of a node in [lo, hi]
under a count guard. Endpoints are a*N+b, a in {0,1}. Only native monotone
prefixes may extend an interval to their physical-loop horizon: the transparent
prefix cycle supplies that induction. Ordinary task/Set completion does not.
"""

from dataclasses import dataclass
from heapq import heappop, heappush

from .budget import poll
from .completion import CountDomain, _first, canonical, intersection
from .completion_flow import ALL, _includes
from .model import require


def add(value, delta):
    return value[0], value[1] + delta


def meet(a, b):
    if a is None or b is None:
        return None
    if a == ALL:
        return b
    if b == ALL or a == b:
        return a
    return intersection(a, b)


def ge(a, b):
    if a == 0:
        return ALL if b >= 0 else None
    if a == 1:
        return CountDomain(1, 0, minimum=max(0, -b))
    require(a == -1, "symbolic completion needs a single uniform count")
    return CountDomain(1, 0, minimum=0, maximum=b) if b >= 0 else None


def le(a, b):
    return ge(b[0] - a[0], b[1] - a[1])


def holds(a, b, guard):
    """a <= b throughout an arithmetic count domain."""
    if a[0] == b[0]:
        return a[1] <= b[1]
    slope, offset = b[0] - a[0], b[1] - a[1]
    if slope * _first(guard) + offset < 0:
        return False
    if guard.exact is not None or slope >= 0:
        return True
    if guard.maximum is None:
        return False
    last = guard.maximum - (guard.maximum - guard.residue) % guard.period
    return slope * last + offset >= 0


def choice(a, b, guard, *, maximum):
    """Disjoint domains for max/min of two affine endpoints."""
    if holds(a, b, guard):
        yield b if maximum else a, guard
        return
    if holds(b, a, guard):
        yield a if maximum else b, guard
        return
    left, right = (b, a) if maximum else (a, b)
    selected = meet(guard, le(left, right))
    if selected is not None:
        yield a, selected
    other = meet(guard, le(add(right, 1), left))
    if other is not None:
        yield b, other


@dataclass(frozen=True)
class Family:
    lower: int
    horizon: tuple[int, int]
    period: int = 1
    residue: int = 0
    last: tuple[int, int] | None = None
    monotone: bool = False

    def residue_guard(self, point):
        a, b = point
        if not a:
            return ALL if b % self.period == self.residue else None
        require(a == 1, "unsupported symbolic generation expression")
        return CountDomain(self.period, (self.residue - b) % self.period, minimum=0)


def native_family(meta, period, count=(1, 0)):
    """The native builder certifies these per-period monotone carry cycles."""
    return Family(
        meta.lower,
        add(count, -1 - meta.upper),
        period,
        meta.residue,
        None if meta.last_gap is None else add(count, -1 - meta.last_gap),
        (meta.guard is not None and meta.point in {"issue", "queue", "retire", "implicit"})
        or (meta.command.kind == "cross_set" and meta.point == "end"),
    )


class SymbolicCompletion:
    def __init__(self):
        self.families, self.edges, self.facts = [], [], []
        self.pending, self.delta = [], {}
        self.edge_count = self.transfers = 0

    def node(self, family):
        require(len(self.edges) < 100_000, "symbolic completion node budget exceeded")
        index = len(self.edges)
        self.families.append(family)
        self.edges.append([])
        self.facts.append({})
        return index

    def _remember(self, node, lo, hi, guard, bits):
        guard = ALL if guard == ALL else canonical(guard)
        if guard.exact is not None:
            lo = 0, lo[0] * guard.exact + lo[1]
            hi = 0, hi[0] * guard.exact + hi[1]
        known = self.facts[node]
        bits &= ~known.get((lo, hi, guard), 0)
        if not bits:
            return
        for (left, right, prior), mask in known.items():
            if (
                bits & mask
                and (prior == guard or _includes(prior, guard))
                and holds(left, lo, guard)
                and holds(hi, right, guard)
            ):
                bits &= ~mask
                if not bits:
                    return
        for key, mask in tuple(known.items()):
            left, right, prior = key
            if (
                bits & mask
                and (guard == prior or _includes(guard, prior))
                and holds(lo, left, prior)
                and holds(right, hi, prior)
            ):
                if remaining := mask & ~bits:
                    known[key] = remaining
                else:
                    del known[key]
        key = lo, hi, guard
        known[key] = known.get(key, 0) | bits
        if node not in self.delta:
            self.delta[node] = {}
            heappush(self.pending, node)
        pending = self.delta[node]
        pending[key] = pending.get(key, 0) | bits

    def add_fact(self, node, lo, hi, guard, bits):
        if guard is None or not bits:
            return
        family = self.families[node]
        lower = family.lower + (family.residue - family.lower) % family.period
        if not (lo[0] or hi[0] or family.horizon[0]) and (
            family.last is None or not family.last[0]
        ):
            left, right = max(lo[1], lower), min(hi[1], family.horizon[1])
            if family.last is not None:
                last = family.last[1]
                if last % family.period != family.residue or not left <= last <= right:
                    return
                left = right = last
            if left <= right:
                self._remember(
                    node,
                    (0, left),
                    (0, family.horizon[1] if family.monotone else right),
                    guard,
                    bits,
                )
            return
        for left, lower_domain in choice(lo, (0, lower), guard, maximum=True):
            for right, domain in choice(hi, family.horizon, lower_domain, maximum=False):
                domain = meet(domain, le(left, right))
                if domain is None:
                    continue
                if family.last is not None:
                    domain = meet(domain, meet(le(left, family.last), le(family.last, right)))
                    domain = meet(domain, family.residue_guard(family.last))
                    if domain is None:
                        continue
                    left = right = family.last
                if family.monotone:
                    # Reach the real prefix once, then use its transparent
                    # carry cycle. Never extend a native end or Set effect.
                    right = family.horizon
                self._remember(node, left, right, domain, bits)

    def seed(self, node, point, guard, bits):
        self.add_fact(
            node, point, point, meet(guard, self.families[node].residue_guard(point)), bits
        )

    def _transfer(self, target, shift, source_point, target_point, guard, lo, hi, domain, bits):
        domain = meet(domain, guard)
        if domain is None:
            return
        if source_point is None:
            self.add_fact(target, add(lo, shift), add(hi, shift), domain, bits)
        else:
            active = meet(domain, meet(le(lo, source_point), le(source_point, hi)))
            self.seed(target, target_point, active, bits)

    def edge(self, source, target, shift=0, *, source_point=None, target_point=None, guard=ALL):
        if guard is None:
            return
        require((source_point is None) == (target_point is None), "partial boundary binding")
        if source_point is not None:
            guard = meet(guard, self.families[source].residue_guard(source_point))
        if guard is None:
            return
        edge = target, shift, source_point, target_point, canonical(guard)
        if edge in self.edges[source]:
            return
        self.edges[source].append(edge)
        self.edge_count += 1
        for (lo, hi, domain), bits in tuple(self.facts[source].items()):
            self._transfer(*edge, lo, hi, domain, bits)

    def propagate(self):
        while self.pending:
            source = heappop(self.pending)
            for (lo, hi, domain), bits in self.delta.pop(source).items():
                bits &= self.facts[source].get((lo, hi, domain), 0)
                if not bits:
                    continue
                for edge in self.edges[source]:
                    self.transfers += 1
                    if self.transfers % 1024 == 0:
                        poll(
                            "symbolic completion propagation",
                            transfers=self.transfers,
                            nodes=len(self.edges),
                        )
                    require(
                        self.transfers <= 2_000_000,
                        "symbolic completion propagation budget exceeded",
                    )
                    self._transfer(*edge, lo, hi, domain, bits)

    def coverage(self, node, point, bit, guard):
        result = []
        guard = meet(guard, self.families[node].residue_guard(point))
        for (lo, hi, domain), bits in self.facts[node].items():
            if bits & bit:
                active = meet(meet(domain, guard), meet(le(lo, point), le(point, hi)))
                if active is not None:
                    result.append(active)
        return result


class NativeCompletion:
    """Local translation domains use the same symbolic-interval transfer rules.

    Each domain has the smallest legal extent, represented as an integer
    bound, never an expanded instance graph. A path in this domain lifts to
    every permitted translation of the endpoint pair and extent. Last-only
    work is admitted only when that domain fixes the distance to the exit.
    """

    def __init__(self, graph):
        self.graph = graph
        self.flows, self.origins, self.queries = {}, {}, {}

    @property
    def transfers(self):
        return sum(flow.transfers for flow in self.flows.values())

    def family(self, node, guard):
        _, first, gap, terminal = guard
        meta = self.graph.nodes[node]
        if meta.last_gap is not None and not terminal:
            return Family(0, (0, -1))
        return native_family(meta, self.graph.period, (0, first + gap + 1))

    def node(self, node):
        for guard, flow in self.flows.items():
            require(flow.node(self.family(node, guard)) == node, "symbolic family identity changed")

    def edge(self, source, target, distance):
        shift = (
            distance * self.graph.period
            + self.graph.nodes[target].residue
            - self.graph.nodes[source].residue
        )
        for flow in self.flows.values():
            flow.edge(source, target, shift)

    def domain(self, guard):
        if guard not in self.flows:
            flow = self.flows[guard] = SymbolicCompletion()
            self.origins[guard] = {}
            for node in range(len(self.graph.nodes)):
                flow.node(self.family(node, guard))
            for a, b, delta in self.graph.edges:
                shift = (
                    delta * self.graph.period
                    + self.graph.nodes[b].residue
                    - self.graph.nodes[a].residue
                )
                flow.edge(a, b, shift)
        return self.flows[guard], self.origins[guard]

    @staticmethod
    def key(query):
        source, sg, destination, dg, *rest = query
        return source.index, sg, destination.index, dg, tuple(rest)

    def register(self, queries):
        for query in queries:
            key = self.key(query)
            if key in self.queries:
                continue
            normalized = self.graph._query(*query)
            self.queries[key] = None
            if normalized is None:
                continue
            guard, (source, cycle), (target, _) = normalized
            residue, first, _, _ = guard
            generation = (
                first - residue + self.graph.nodes[source].residue + cycle * self.graph.period
            )
            flow, origins = self.domain(guard)
            origin = source, generation
            if origin not in origins:
                bit = origins[origin] = 1 << len(origins)
                flow.seed(source, (0, generation), ALL, bit)
            self.queries[key] = flow, target, (0, first), origins[origin]

    def covered(self, query):
        item = self.queries[self.key(query)]
        if item is None:
            return True
        flow, node, point, bit = item
        flow.propagate()
        return bool(flow.coverage(node, point, bit, ALL))

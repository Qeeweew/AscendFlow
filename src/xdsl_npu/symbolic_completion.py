"""Completion facts over symbolic generation intervals, without time expansion.

A fact states that one source precedes every instance of a node in [lo, hi]
under a count guard. Endpoints are a*N+b, a in {0,1}. Only native monotone
prefixes may extend an interval to their physical-loop horizon: the transparent
prefix cycle supplies that induction. Ordinary task/Set completion does not.
"""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

from .budget import poll
from .completion import CountDomain, _first, canonical, intersection
from .completion_flow import ALL, _includes
from .model import require

FACT_STATISTICS = (
    "fact_nodes",
    "condition_facts",
    "peak_condition_facts",
    "fact_attempts",
    "fact_comparisons",
    "fact_updates",
    "fact_transfers",
)


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


class CompletionTopology:
    """One owner appends immutable family/arc records; domains only read them."""

    def __init__(self):
        self.families, self.edges = [], []
        self.edge_count = 0

    def node(self, family):
        require(len(self.edges) < 100_000, "symbolic completion node budget exceeded")
        index = len(self.edges)
        self.families.append(family)
        self.edges.append(())
        return index

    def edge(self, source, target, shift=0, *, source_point=None, target_point=None, guard=ALL):
        if guard is None:
            return None
        require((source_point is None) == (target_point is None), "partial boundary binding")
        if source_point is not None:
            guard = meet(guard, self.families[source].residue_guard(source_point))
        if guard is None:
            return None
        edge = target, shift, source_point, target_point, canonical(guard)
        if edge in self.edges[source]:
            return None
        self.edges[source] += (edge,)
        self.edge_count += 1
        return edge


class SymbolicCompletion:
    def __init__(self, *, topology=None, count=None, terminal=True):
        self._owns_topology = topology is None
        self.topology = CompletionTopology() if topology is None else topology
        self.families, self.edges = self.topology.families, self.topology.edges
        # A domain allocates facts only at reached nodes, not V empty maps.
        self.facts = defaultdict(dict)
        self.count, self.terminal = count, terminal
        # Bind immutable family bounds once, only when a fact touches a node.
        # This is domain-local activation, not another copy of the topology.
        self.activation = {}
        self.pending, self.delta = [], {}
        self.transfers = 0
        self.fact_attempts = self.fact_comparisons = self.fact_updates = 0
        self.fact_records = self.peak_fact_records = 0

    @property
    def edge_count(self):
        return self.topology.edge_count

    def node(self, family):
        require(self._owns_topology, "a query domain cannot mutate its shared topology")
        return self.topology.node(family)

    def _remember(self, node, lo, hi, guard, bits):
        self.fact_attempts += 1
        guard = ALL if guard == ALL else canonical(guard)
        if guard.exact is not None:
            lo = 0, lo[0] * guard.exact + lo[1]
            hi = 0, hi[0] * guard.exact + hi[1]
        known = self.facts[node]
        key = lo, hi, guard
        present = known.get(key, 0)
        bits &= ~present
        if not bits:
            return
        comparisons = 0
        for (left, right, prior), mask in known.items():
            comparisons += 1
            if (
                bits & mask
                and (prior == guard or _includes(prior, guard))
                and holds(left, lo, guard)
                and holds(hi, right, guard)
            ):
                bits &= ~mask
                if not bits:
                    self.fact_comparisons += comparisons
                    return
        records = tuple(known.items())
        self.fact_comparisons += comparisons + len(records)
        removals, updates = 0, 1
        for prior_key, mask in records:
            left, right, prior = prior_key
            if (
                bits & mask
                and (guard == prior or _includes(guard, prior))
                and holds(lo, left, prior)
                and holds(right, hi, prior)
            ):
                if remaining := mask & ~bits:
                    known[prior_key] = remaining
                else:
                    del known[prior_key]
                    removals += 1
                updates += 1
        # Incoming bits exclude `present`, so supersession cannot change this
        # exact key. Reuse its first lookup instead of hashing the guard again.
        self.fact_records += (not present) - removals
        known[key] = present | bits
        self.fact_updates += updates
        if self.fact_records > self.peak_fact_records:  # noqa: PLR1730 - avoid a call per fact update
            self.peak_fact_records = self.fact_records
        if node not in self.delta:
            self.delta[node] = {}
            heappush(self.pending, node)
        pending = self.delta[node]
        pending[key] = pending.get(key, 0) | bits

    def add_fact(self, node, lo, hi, guard, bits):
        if guard is None or not bits:
            return
        family = self.families[node]
        activation = self.activation.get(node)
        if activation is None:
            horizon, last = family.horizon, family.last
            if last is not None and self.count is not None and not self.terminal:
                self.activation[node] = ()
                return
            if self.count is not None:
                # The smallest legal count only changes activation bounds.
                horizon = (0, horizon[0] * self.count + horizon[1])
                if last is not None:
                    last = (0, last[0] * self.count + last[1])
            if last is not None and not last[0] and last[1] % family.period != family.residue:
                self.activation[node] = ()
                return
            lower = family.lower + (family.residue - family.lower) % family.period
            activation = self.activation[node] = lower, horizon, last
        if not activation:
            return
        lower, horizon, last = activation
        if not (lo[0] or hi[0] or horizon[0]) and (last is None or not last[0]):
            left, right = max(lo[1], lower), min(hi[1], horizon[1])
            if last is not None:
                if not left <= last[1] <= right:
                    return
                left = right = last[1]
            if left <= right:
                self._remember(
                    node,
                    (0, left),
                    (0, horizon[1] if family.monotone else right),
                    guard,
                    bits,
                )
            return
        for left, lower_domain in choice(lo, (0, lower), guard, maximum=True):
            for right, domain in choice(hi, horizon, lower_domain, maximum=False):
                domain = meet(domain, le(left, right))
                if domain is None:
                    continue
                if last is not None:
                    domain = meet(domain, meet(le(left, last), le(last, right)))
                    domain = meet(domain, family.residue_guard(last))
                    if domain is None:
                        continue
                    left = right = last
                if family.monotone:
                    # Reach the real prefix once, then use its transparent
                    # carry cycle. Never extend a native end or Set effect.
                    right = horizon
                self._remember(node, left, right, domain, bits)

    def seed(self, node, point, guard, bits):
        self.add_fact(
            node, point, point, meet(guard, self.families[node].residue_guard(point)), bits
        )

    def _transfer(self, target, shift, source_point, target_point, guard, lo, hi, domain, bits):
        self.transfers += 1
        if self.transfers % 1024 == 0:
            poll("symbolic completion propagation", transfers=self.transfers, nodes=len(self.edges))
        require(self.transfers <= 2_000_000, "symbolic completion propagation budget exceeded")
        domain = meet(domain, guard)
        if domain is None:
            return
        if source_point is None:
            self.add_fact(target, add(lo, shift), add(hi, shift), domain, bits)
        else:
            active = meet(domain, meet(le(lo, source_point), le(source_point, hi)))
            self.seed(target, target_point, active, bits)

    def edge(self, source, target, shift=0, *, source_point=None, target_point=None, guard=ALL):
        require(self._owns_topology, "a query domain cannot mutate its shared topology")
        edge = self.topology.edge(
            source, target, shift, source_point=source_point, target_point=target_point, guard=guard
        )
        if edge is not None:
            self.replay(source, edge)

    def replay(self, source, edge):
        """An accepted new arc receives this domain's existing source facts."""
        for (lo, hi, domain), bits in tuple(self.facts.get(source, {}).items()):
            self._transfer(*edge, lo, hi, domain, bits)

    def propagate(self):
        while self.pending:
            source = heappop(self.pending)
            for (lo, hi, domain), bits in self.delta.pop(source).items():
                bits &= self.facts[source].get((lo, hi, domain), 0)
                if not bits:
                    continue
                for edge in self.edges[source]:
                    self._transfer(*edge, lo, hi, domain, bits)

    def coverage(self, node, point, bit, guard):
        result = []
        guard = meet(guard, self.families[node].residue_guard(point))
        for (lo, hi, domain), bits in self.facts.get(node, {}).items():
            if bits & bit:
                active = meet(meet(domain, guard), meet(le(lo, point), le(point, hi)))
                if active is not None:
                    result.append(active)
        return result

    def statistics(self):
        """Stored condition records and actual work, not translated instances."""
        return {
            "fact_nodes": sum(bool(facts) for facts in self.facts.values()),
            "condition_facts": self.fact_records,
            "peak_condition_facts": self.peak_fact_records,
            "fact_attempts": self.fact_attempts,
            "fact_comparisons": self.fact_comparisons,
            "fact_updates": self.fact_updates,
            "fact_transfers": self.transfers,
        }


class NativeCompletion:
    """Local translation domains use the same symbolic-interval transfer rules.

    Each domain has the smallest legal extent, represented as an integer
    bound, never an expanded instance graph. A path in this domain lifts to
    every permitted translation of the endpoint pair and extent. Last-only
    work is admitted only when that domain fixes the distance to the exit.
    """

    def __init__(self, graph):
        self.graph = graph
        self.topology = CompletionTopology()
        self.flows, self.origins, self.queries = {}, {}, {}
        for node in range(len(graph.nodes)):
            self.node(node)
        for arc in graph.edges:
            self.edge(*arc)

    @property
    def transfers(self):
        return sum(flow.transfers for flow in self.flows.values())

    def node(self, node):
        require(
            self.topology.node(native_family(self.graph.nodes[node], self.graph.period)) == node,
            "symbolic family identity changed",
        )

    def edge(self, source, target, distance):
        shift = (
            distance * self.graph.period
            + self.graph.nodes[target].residue
            - self.graph.nodes[source].residue
        )
        edge = self.topology.edge(source, target, shift)
        if edge is not None:
            for flow in self.flows.values():
                flow.replay(source, edge)

    def domain(self, guard):
        if guard not in self.flows:
            _, first, gap, terminal = guard
            self.flows[guard] = SymbolicCompletion(
                topology=self.topology, count=first + gap + 1, terminal=terminal
            )
            self.origins[guard] = {}
        return self.flows[guard], self.origins[guard]

    def statistics(self):
        counts = [flow.statistics() for flow in self.flows.values()]
        return {
            "query_domains": len(self.flows),
            "topology_instances": 1,
            "topology_nodes": len(self.topology.families),
            "topology_edges": self.topology.edge_count,
            "queries": len(self.queries),
            "vacuous_queries": sum(item is None for item in self.queries.values()),
            **{key: sum(c[key] for c in counts) for key in FACT_STATISTICS},
        }

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

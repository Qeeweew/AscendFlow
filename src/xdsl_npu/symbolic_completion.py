"""Completion facts over symbolic generation intervals, without time expansion.

A fact states that one source precedes every instance of a node in [lo, hi]
under a count guard. Endpoints are a*N+b, a in {0,1}. Only native monotone
prefixes may extend an interval to their physical-loop horizon: the transparent
prefix cycle supplies that induction. Ordinary task/Set completion does not.
"""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import islice

from .budget import poll
from .completion import _first, canonical, count_domain, intersection
from .completion_flow import ALL, _includes
from .model import require

FACT_STATISTICS = (
    "fact_nodes",
    "condition_facts",
    "peak_condition_facts",
    "fact_attempts",
    "fact_comparisons",
    "fact_index_steps",
    "fact_interval_joins",
    "fact_updates",
    "fact_transfers",
    "pending_frontier_steps",
    "pending_interval_joins",
    "coverage_batches",
    "coverage_requests",
    "coverage_fact_scans",
    "coverage_result_items",
)


def add(value, delta):
    return value[0], value[1] + delta


def meet(a, b):
    if a is b:
        return a
    if a is None or b is None:
        return None
    if a is ALL or a == ALL:
        return b
    if b is ALL or b == ALL or a == b:
        return a
    return intersection(a, b)


def ge(a, b):
    if a == 0:
        return ALL if b >= 0 else None
    if a == 1:
        return ALL if b >= 0 else count_domain(1, 0, minimum=-b)
    require(a == -1, "symbolic completion needs a single uniform count")
    return count_domain(1, 0, minimum=0, maximum=b) if b >= 0 else None


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


class _OriginBranch:
    """A dyadic range of origin positions; a record occupies one branch only."""

    __slots__ = ("bits", "children", "level", "lower", "records")

    def __init__(self, lower, level):
        self.bits, self.lower, self.level = 0, lower, level
        self.records, self.children = {}, {}


class _OriginIndex:
    """Find records sharing origins without expanding masks into single bits.

    Records crossing a split stay at its parent. Dense masks therefore use
    one entry, while a sparse single-origin query skips unrelated branches.
    Branch unions are monotone hints; actual record masks decide membership.
    """

    def __init__(self, records):
        self.root, self.locations = None, {}
        self.steps = 0
        for key, bits in records.items():
            self.put(key, bits)

    def put(self, key, bits):
        if previous := self.locations.pop(key, None):
            del previous.records[key]
        if not bits:
            return
        first, last = (bits & -bits).bit_length() - 1, bits.bit_length() - 1
        level = last.bit_length()
        if self.root is None:
            self.root = _OriginBranch(0, level)
        while self.root.level < level:
            self.steps += 1
            parent = _OriginBranch(0, self.root.level + 1)
            parent.bits = self.root.bits
            parent.children[0] = self.root
            self.root = parent
        branch = self.root
        while True:
            self.steps += 1
            branch.bits |= bits
            if branch.level == 0:
                break
            middle = branch.lower + (1 << (branch.level - 1))
            if first < middle <= last:
                break
            side = int(first >= middle)
            if side not in branch.children:
                branch.children[side] = _OriginBranch(
                    middle if side else branch.lower, branch.level - 1
                )
            branch = branch.children[side]
        branch.records[key] = bits
        self.locations[key] = branch

    def candidates(self, bits):
        pending = [self.root] if self.root is not None else []
        while pending:
            branch = pending.pop()
            self.steps += 1
            if not branch.bits & bits:
                continue
            for key, mask in branch.records.items():
                self.steps += 1
                if mask & bits:
                    yield key, mask
            pending.extend(branch.children.values())


def _joined_interval(lo, hi, left, right, guard):
    """One exact union on the entire guard, never a hull across a hole."""
    if not (holds(lo, add(right, 1), guard) and holds(left, add(hi, 1), guard)):
        return None
    if holds(lo, left, guard):
        lower = lo
    elif holds(left, lo, guard):
        lower = left
    else:
        return None
    if holds(hi, right, guard):
        upper = right
    elif holds(right, hi, guard):
        upper = hi
    else:
        return None
    return lower, upper


class _ConditionalFrontier:
    """Origin-labelled interval unions with a count-conditioned antichain.

    A join moves only the shared origin bits. The other bits keep their old
    interval and guard. It creates no new count predicates and never splits a
    mask into individual origins. Exact edits also normalize pending deltas.
    """

    def __init__(self, records):
        self.records, self.bits, self.index = records, 0, None
        self.origin_count, self.index_probe_size = 0, 0
        self.retired_index_steps = 0
        self.joins = 0

    def _candidates(self, bits, *, snapshot=True):
        # Python's direct scan is cheaper for a small bucket or a mask naming
        # much of its origin set. An index cannot skip that dense work.
        sparse = bits.bit_count() * 4 < self.origin_count
        if (
            sparse
            and self.index is None
            and len(self.records) >= max(64, 2 * self.index_probe_size)
        ):
            self.index_probe_size = len(self.records)
            middle = 1 << ((self.bits.bit_length() - 1).bit_length() - 1)
            sample = tuple(islice(self.records.values(), 8))
            self.retired_index_steps += len(sample)
            spanning = sum(
                (mask & -mask).bit_length() - 1 < middle <= mask.bit_length() - 1 for mask in sample
            )
            if spanning * 2 < len(sample):
                self.index = _OriginIndex(self.records)
        candidates = (
            self.index.candidates(bits)
            if sparse and self.index is not None
            else self.records.items()
        )
        return tuple(candidates) if snapshot else candidates

    def _put(self, key, bits):
        if bits:
            self.records[key] = bits
        else:
            del self.records[key]
        if self.index is not None:
            self.index.put(key, bits)
            if len(self.index.root.records) * 2 >= len(self.records):
                # Masks growing across the root split make this index dense.
                # Reconsider only after the frontier grows, amortizing probes.
                self.index_probe_size = len(self.records)
                self.retired_index_steps += self.index.steps
                self.index = None

    def insert(self, lo, hi, guard, bits):
        key = lo, hi, guard
        bits &= ~self.records.get(key, 0)
        if not bits:
            return 0, 0, 0, ()
        comparisons = updates = 0
        candidates = self._candidates(bits, snapshot=False) if bits & self.bits else ()
        for (left, right, prior), mask in candidates:
            comparisons += 1
            if (
                bits & mask
                and (prior == guard or _includes(prior, guard))
                and holds(left, lo, guard)
                and holds(hi, right, guard)
            ):
                bits &= ~mask
                if not bits:
                    return 0, comparisons, 0, ()
        fresh = bits
        edits = []
        work = [(lo, hi, bits)]
        while work:
            lo, hi, bits = work.pop()
            key = lo, hi, guard
            bits &= ~self.records.get(key, 0)
            if not bits:
                continue
            joined = False
            candidates = self._candidates(bits) if bits & self.bits else ()
            for prior_key, mask in candidates:
                comparisons += 1
                left, right, prior = prior_key
                shared = bits & mask
                if not shared:
                    continue
                if (
                    (guard == prior or _includes(guard, prior))
                    and holds(lo, left, prior)
                    and holds(right, hi, prior)
                ):
                    self._put(prior_key, mask & ~shared)
                    updates += 1
                    edits.append((key, prior_key, key, shared))
                    continue
                interval = _joined_interval(lo, hi, left, right, guard) if prior == guard else None
                # Two partial masks would turn two compact rectangles into
                # three. Keep that already compact representation instead.
                if interval is None or shared != bits and shared != mask:
                    continue
                self._put(prior_key, mask & ~shared)
                updates += 1
                self.joins += 1
                edits.append((key, prior_key, (*interval, guard), shared))
                if remaining := bits & ~shared:
                    work.append((lo, hi, remaining))
                work.append((*interval, shared))
                joined = True
                break
            if not joined:
                self._put(key, self.records.get(key, 0) | bits)
                if novel := bits & ~self.bits:
                    self.bits |= novel
                    self.origin_count += novel.bit_count()
                updates += 1
        return fresh, comparisons, updates, edits


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
        if self.period == 1:
            return ALL
        return count_domain(self.period, (self.residue - b) % self.period, minimum=0)


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
        self._node_bits = {}
        self._frontiers = {}
        self._pending_frontier_steps = self._pending_interval_joins = 0
        self.count, self.terminal = count, terminal
        # Bind immutable family bounds once, only when a fact touches a node.
        # This is domain-local activation, not another copy of the topology.
        self.activation = {}
        self.pending, self.delta = [], {}
        self.transfers = 0
        self.fact_attempts = self.fact_comparisons = self.fact_updates = 0
        self.fact_records = self.peak_fact_records = 0
        self.coverage_batches = self.coverage_requests = 0
        self.coverage_fact_scans = self.coverage_result_items = 0

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
        if node not in self._frontiers:
            self._frontiers[node] = _ConditionalFrontier(known)
        frontier = self._frontiers[node]
        frontier.records = known
        before = len(known)
        bits, comparisons, updates, edits = frontier.insert(lo, hi, guard, bits)
        self.fact_comparisons += comparisons
        if not bits:
            return
        self.fact_records += len(known) - before
        self._node_bits[node] = frontier.bits
        self.fact_updates += updates
        if self.fact_records > self.peak_fact_records:  # noqa: PLR1730 - avoid a call per fact update
            self.peak_fact_records = self.fact_records
        if node not in self.delta:
            self.delta[node] = {}
            heappush(self.pending, node)
        # Enqueue the incoming support, not its potentially much wider joined
        # frontier. Existing coverage that already travelled is not replayed.
        pending = self.delta[node]
        key = lo, hi, guard
        pending[key] = pending.get(key, 0) | bits
        for incoming, prior, result, mask in edits:
            self._pending_frontier_steps += 1
            if incoming == result:
                # A dominating interval is already represented by sent or
                # pending incoming support (possibly several joined pieces).
                # The dominated record contributes no unique queued coverage.
                selected = mask & pending.get(prior, 0)
                if selected:
                    if remaining := pending[prior] & ~selected:
                        pending[prior] = remaining
                    else:
                        del pending[prior]
                continue
            # The retained frontier can include work already propagated. An
            # edit applies to the delta only for origins with BOTH supports
            # still pending at these exact intervals and full count guards.
            selected = mask & pending.get(incoming, 0) & pending.get(prior, 0)
            if not selected:
                continue
            self._pending_interval_joins += 1
            if remaining := pending[incoming] & ~selected:
                pending[incoming] = remaining
            else:
                del pending[incoming]
            if remaining := pending[prior] & ~selected:
                pending[prior] = remaining
            else:
                del pending[prior]
            pending[result] = pending.get(result, 0) | selected

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
                active_left, active_right = left, right
                domain = meet(domain, le(active_left, active_right))
                if domain is None:
                    continue
                if last is not None:
                    domain = meet(domain, meet(le(active_left, last), le(last, active_right)))
                    domain = meet(domain, family.residue_guard(last))
                    if domain is None:
                        continue
                    active_left = active_right = last
                if family.monotone:
                    # Reach the real prefix once, then use its transparent
                    # carry cycle. Never extend a native end or Set effect.
                    active_right = horizon
                self._remember(node, active_left, active_right, domain, bits)

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
                # A delta remains valid when its key disappears into a wider
                # retained interval. Both frontiers represent monotone unions.
                for edge in self.edges[source]:
                    self._transfer(*edge, lo, hi, domain, bits)

    def coverage(self, node, point, bit, guard):
        self.coverage_batches += 1
        self.coverage_requests += 1
        result = []
        guard = meet(guard, self.families[node].residue_guard(point))
        for (lo, hi, domain), bits in self.facts.get(node, {}).items():
            self.coverage_fact_scans += 1
            if bits & bit:
                active = meet(meet(domain, guard), meet(le(lo, point), le(point, hi)))
                if active is not None:
                    result.append(active)
        self.coverage_result_items += len(result)
        return result

    def coverage_many(self, node, point, origins, guard):
        """Read one endpoint/domain once, retaining each single-bit origin's guards.

        Repeated origins preserve request order and receive independent lists.
        The scalar reader retains its more general any-matching-bit semantics.
        No result survives this call or observes a different flow's facts.
        """
        origins = tuple(origins)
        require(
            all(type(bit) is int and bit > 0 and bit & (bit - 1) == 0 for bit in origins),
            "batched symbolic coverage requires single-bit origins",
        )
        self.coverage_batches += 1
        self.coverage_requests += len(origins)
        result = {bit: [] for bit in origins}
        requested = 0
        for bit in result:
            requested |= bit
        guard = meet(guard, self.families[node].residue_guard(point))
        if requested and guard is not None:
            for (lo, hi, domain), bits in self.facts.get(node, {}).items():
                self.coverage_fact_scans += 1
                if self.coverage_fact_scans % 4096 == 0:
                    poll("symbolic completion coverage", facts=self.coverage_fact_scans)
                selected = bits & requested
                if not selected:
                    continue
                active = meet(meet(domain, guard), meet(le(lo, point), le(point, hi)))
                if active is None:
                    continue
                # Keep this exact guard paired with only its matching origins;
                # unioning all selected masks would mix disjoint count paths.
                while selected:
                    bit = selected & -selected
                    result[bit].append(active)
                    selected ^= bit
        ordered = [list(result[bit]) for bit in origins]
        self.coverage_result_items += sum(map(len, ordered))
        return ordered

    def statistics(self):
        """Stored condition records and actual work, not translated instances."""
        return {
            "fact_nodes": sum(bool(facts) for facts in self.facts.values()),
            "condition_facts": self.fact_records,
            "peak_condition_facts": self.peak_fact_records,
            "fact_attempts": self.fact_attempts,
            "fact_comparisons": self.fact_comparisons,
            "fact_index_steps": sum(
                frontier.retired_index_steps
                + (0 if frontier.index is None else frontier.index.steps)
                for frontier in self._frontiers.values()
            ),
            "fact_interval_joins": sum(frontier.joins for frontier in self._frontiers.values()),
            "fact_updates": self.fact_updates,
            "fact_transfers": self.transfers,
            "pending_frontier_steps": self._pending_frontier_steps,
            "pending_interval_joins": self._pending_interval_joins,
            "coverage_batches": self.coverage_batches,
            "coverage_requests": self.coverage_requests,
            "coverage_fact_scans": self.coverage_fact_scans,
            "coverage_result_items": self.coverage_result_items,
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

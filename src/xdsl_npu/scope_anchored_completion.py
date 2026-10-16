"""Anchored parent boundaries on the existing guarded completion topology.

A single origin bit must reach every active member of a target span. The
reverse query uses h=N-1-g, reverses actual arcs, and reflects native family
bounds. It preserves the original count predicates; N is never instantiated.
"""

from .budget import bounded, poll
from .completion import CountDomain, Position, coalesce, covers, uncovered
from .completion_flow import ALL
from .model import require
from .scope_domains import affine
from .scope_guarded_completion import GuardedCompletion, _activation
from .scope_transfer_queries import (
    VirtualNode,
)
from .scope_transfer_queries import (
    condition as transfer_condition,
)
from .scope_transfer_queries import (
    direct as transfer_direct,
)
from .scope_transfer_queries import (
    family as transfer_family,
)
from .scope_transfer_queries import (
    point_guard as transfer_point_guard,
)
from .scope_transfer_queries import (
    prepare as prepare_virtual,
)
from .scope_transfer_queries import (
    revision as transfer_revision,
)
from .scope_transfer_queries import (
    routes as transfer_routes,
)
from .symbolic_completion import (
    FACT_STATISTICS,
    CompletionTopology,
    Family,
    SymbolicCompletion,
    add,
    choice,
    le,
    meet,
)

MAX_ANCHORED_CASES = 65_536
MAX_ANCHORED_ORIGINS = 65_536
MAX_ANCHORED_TRANSFERS = 2_000_000
MAX_ANCHORED_SPAN_WORK = 2_000_000
MAX_ANCHORED_SEGMENTS = 16_384
MAX_ANCHORED_REVERSE_NODES = 100_000
MAX_ANCHORED_REVERSE_ARCS = 1_000_000


def _reflect(point):
    return 1 - point[0], -1 - point[1]


def _position(point):
    require(
        point is None
        or isinstance(point, Position)
        and type(point.end) is bool
        and type(point.offset) is int,
        "anchored completion needs a parent Position or None",
    )


def _aligned(point, residue, period, count_residue, *, upper=False):
    actual = (point[0] * count_residue + point[1]) % period
    return add(point, -((actual - residue) % period) if upper else (residue - actual) % period)


class _Reverse:
    """One residue of original N, with append-only reversed native arcs."""

    def __init__(self, count_residue):
        self.count_residue = count_residue
        self.topology = CompletionTopology()
        self.flow = SymbolicCompletion(topology=self.topology)
        self.origins, self.arcs = {}, 0
        self.bindings, self.transfer_revision = {}, 0

    def sync(self, guarded):
        period = guarded.graph.period
        for node in range(len(self.topology.families), len(guarded.topology.families)):
            if node % 4096 == 0:
                poll("anchored reverse families", nodes=node)
            family = guarded.topology.families[node]
            require(family.horizon[0] == 1, "anchored reversal needs a parent-count horizon")
            reflected = Family(
                -1 - family.horizon[1],
                (1, -1 - family.lower),
                family.period,
                (self.count_residue - 1 - family.residue) % family.period,
                None if family.last is None else _reflect(family.last),
                family.monotone,
            )
            require(
                self.topology.node(reflected) == node, "reversed native family identity changed"
            )
        for arc in range(self.arcs, len(guarded.graph.edges)):
            if arc % 4096 == 0:
                poll("anchored reverse arcs", arcs=arc)
            source, target, cycles = guarded.graph.edges[arc]
            shift = (
                cycles * period
                + guarded.graph.nodes[target].residue
                - guarded.graph.nodes[source].residue
            )
            # u(g)->v(g+d) becomes v(h)->u(h+d). In particular a certified
            # prefix carry still advances h by P, so its monotone induction
            # remains valid. Ordinary native/Set nodes gain no induction.
            edge = self.topology.edge(
                target,
                source,
                shift,
                guard=meet(guarded.conditions[source], guarded.conditions[target]),
            )
            if edge is not None:
                self.flow.replay(target, edge)
        self.arcs = len(guarded.graph.edges)
        current = transfer_revision(guarded.graph)
        if current != self.transfer_revision:
            for (node, point), bit in self.origins.items():
                if not isinstance(node, VirtualNode):
                    continue
                guard = transfer_point_guard(
                    node,
                    point,
                    self.topology.families,
                    guarded.conditions,
                    count_residue=self.count_residue,
                )
                guard = meet(guard, CountDomain(period, self.count_residue, minimum=0))
                for boundary in transfer_routes(
                    guarded.graph,
                    node,
                    source=True,
                    reverse=True,
                    since=self.transfer_revision,
                ):
                    self.flow.seed(boundary, point, guard, bit)
            self.transfer_revision = current


class AnchoredCompletion:
    """Point/span queries sharing one GuardedCompletion's hardware contract.

    Exactly one or both positions must be anchors. With one anchor, activation
    constrains the unanchored side. With two anchors, only its count domain is
    used. As in GuardedCompletion, inactive endpoint-case combinations create
    no obligation, while unknown endpoint families are errors. Ownership must
    separately justify that a claimed producer/consumer actually exists.
    """

    def __init__(self, guarded):
        require(isinstance(guarded, GuardedCompletion), "anchored completion needs a guarded graph")
        self.guarded = guarded
        self.reverse, self.answers, self.version = {}, {}, None
        self.queries = self.cases = self.reverse_origins = self.span_work = self.propagations = 0

    def _sync(self):
        guarded = self.guarded
        guarded._sync()
        version = (
            len(guarded.topology.families),
            guarded.arcs,
            len(guarded.graph.ids),
            transfer_revision(guarded.graph),
        )
        if version != self.version:
            self.answers.clear()
            self._topology_budget(len(self.reverse))
            for state in self.reverse.values():
                state.sync(guarded)
            self.version = version
        self._check_budget()

    def _topology_budget(self, domains):
        require(
            domains * len(self.guarded.topology.families) <= MAX_ANCHORED_REVERSE_NODES
            and domains * len(self.guarded.graph.edges) <= MAX_ANCHORED_REVERSE_ARCS,
            "anchored completion reverse topology budget exceeded",
        )

    def _check_budget(self):
        require(
            self.guarded.origin_count + self.reverse_origins <= MAX_ANCHORED_ORIGINS,
            "anchored completion origin budget exceeded",
        )
        require(
            sum(flow.transfers for flow in self.guarded.flows.values())
            + sum(state.flow.transfers for state in self.reverse.values())
            <= MAX_ANCHORED_TRANSFERS,
            "anchored completion propagation budget exceeded",
        )

    def _state(self, count_residue):
        if count_residue not in self.reverse:
            self._topology_budget(len(self.reverse) + 1)
            state = self.reverse[count_residue] = _Reverse(count_residue)
            state.sync(self.guarded)
            self._check_budget()
        return self.reverse[count_residue]

    def _point_guard(self, node, point, state):
        return transfer_point_guard(
            node,
            point,
            (self.guarded.topology if state is None else state.topology).families,
            self.guarded.conditions,
            count_residue=None if state is None else state.count_residue,
        )

    def _seed(self, node, point, state):
        if state is None:
            flow, _, _, bit, _ = self.guarded._reach(node, node, point, 0, ALL)
            self._check_budget()
            return flow, bit
        origin = node, point
        if origin not in state.origins:
            self.reverse_origins += 1
            self._check_budget()
            bit = state.origins[origin] = 1 << len(state.origins)
            state.bindings[bit] = node, point
            domain = CountDomain(self.guarded.graph.period, state.count_residue, minimum=0)
            guard = meet(domain, self._point_guard(node, point, state))
            for boundary in transfer_routes(self.guarded.graph, node, source=True, reverse=True):
                state.flow.seed(boundary, point, guard, bit)
        return state.flow, state.origins[origin]

    def _case(self, cases, flow, node, left, right, bit, needed, count_residue, state=None):
        self.cases += 1
        require(self.cases <= MAX_ANCHORED_CASES, "anchored completion case budget exceeded")
        if self.cases % 128 == 0:
            poll("anchored completion cases", cases=self.cases, queries=self.queries)
        cases.append((flow, node, left, right, bit, needed, count_residue, state))

    def _spans(self, node, activation, needed, state, count_residue):
        family = transfer_family(
            node,
            (self.guarded.topology if state is None else state.topology).families,
            count_residue=None if state is None else state.count_residue,
        )
        needed = meet(needed, transfer_condition(node, self.guarded.conditions))
        if needed is None:
            return
        first, last = affine(activation.first), affine(activation.last)
        if state is not None:
            first, last = _reflect(last), _reflect(first)
        if family.last is not None:
            point = family.last
            domain = meet(
                needed,
                meet(
                    meet(le(first, point), le(point, last)), self._point_guard(node, point, state)
                ),
            )
            if domain is not None:
                yield point, point, domain
            return
        for left, lower_domain in choice(first, (0, family.lower), needed, maximum=True):
            for right, domain in choice(last, family.horizon, lower_domain, maximum=False):
                left_aligned = _aligned(left, family.residue, family.period, count_residue)
                right_aligned = _aligned(
                    right, family.residue, family.period, count_residue, upper=True
                )
                domain = meet(domain, le(left_aligned, right_aligned))
                if domain is not None:
                    yield left_aligned, right_aligned, domain

    def _register(
        self, source, destination, source_position, destination_position, activation, prepared
    ):
        _position(source_position)
        _position(destination_position)
        require(
            source_position is not None or destination_position is not None,
            "anchored completion needs at least one anchored endpoint",
        )
        guarded, graph = self.guarded, self.guarded.graph
        _activation(activation, graph.period)
        source = guarded._endpoint(source, None, source=True)
        destination = guarded._endpoint(destination, None, source=False)
        key = source, destination, source_position, destination_position, activation
        if key in self.answers or key in prepared:
            return key
        self.queries += 1
        poll("anchored completion query", queries=self.queries)
        required = guarded._required(activation)
        if required is None:
            self.answers[key] = ()
            return key
        graph.potential()  # Keep the same conservative progress certificate.
        cases = []
        reverse = source_position is None
        anchor = affine(destination_position if reverse else source_position)
        if reverse:
            anchor = _reflect(anchor)
            source, destination = destination, source
        for count_residue in range(graph.period):
            if count_residue % 128 == 0:
                poll("anchored completion count residues", residues=count_residue)
            domain = meet(required, CountDomain(graph.period, count_residue, minimum=0))
            if domain is None:
                continue
            # Choose the original source/destination family at the anchor.
            original = _reflect(anchor) if reverse else anchor
            residue = (original[0] * count_residue + original[1]) % graph.period
            origin = source[residue]
            if origin is None:
                continue
            domain = meet(domain, guarded._point_guard(origin, original))
            if domain is None:
                continue
            state = self._state(count_residue) if reverse else None
            domain = meet(domain, self._point_guard(origin, anchor, state))
            if domain is None:
                continue
            flow, bit = self._seed(origin, anchor, state)
            if source_position is not None and destination_position is not None:
                point = affine(destination_position)
                target = destination[(point[0] * count_residue + point[1]) % graph.period]
                needed = meet(domain, self._point_guard(target, point, None))
                if needed is not None:
                    self._case(cases, flow, target, point, point, bit, needed, count_residue, state)
                continue
            for residue, target in enumerate(destination):
                if target is None or residue % activation.period not in activation.residues:
                    continue
                for left, right, needed in self._spans(
                    target, activation, domain, state, count_residue
                ):
                    self._case(cases, flow, target, left, right, bit, needed, count_residue, state)
        prepared[key] = required, cases
        return key

    def _direct_record(self, flow, node, bit, state):
        bindings = self.guarded.source_bindings[flow] if state is None else state.bindings
        source, point = bindings[bit]
        if transfer_direct(self.guarded.graph, source, node, reverse=state is not None):
            guard = meet(
                self._point_guard(source, point, state), self._point_guard(node, point, state)
            )
            if state is not None:
                guard = meet(
                    guard, CountDomain(self.guarded.graph.period, state.count_residue, minimum=0)
                )
            if guard is not None:
                return point, point, guard
        return None

    def _records(self, flow, node, bit, state, *, needed=None, physical=None):
        records = {}
        for boundary in transfer_routes(
            self.guarded.graph, node, source=False, reverse=state is not None
        ):
            if physical is None:
                for key, bits in flow.facts.get(boundary, {}).items():
                    if bits & bit:
                        records[key] = None
            else:
                records.update(dict.fromkeys(physical[flow, boundary, needed][bit]))
        local = self._direct_record(flow, node, bit, state)
        if local is not None:
            records[local] = None
        return records

    def _coverage(self, flow, node, point, bit, needed, state, physical=None):
        result = []
        needed = meet(needed, self._point_guard(node, point, state))
        for boundary in transfer_routes(
            self.guarded.graph, node, source=False, reverse=state is not None
        ):
            result.extend(
                flow.coverage(boundary, point, bit, needed)
                if physical is None
                else physical[flow, boundary, point, needed][bit]
            )
        local = self._direct_record(flow, node, bit, state)
        for lo, hi, guard in () if local is None else (local,):
            active = meet(meet(needed, guard), meet(le(lo, point), le(point, hi)))
            if active is not None:
                result.append(active)
        return result

    def _span_missing(
        self, flow, node, left, right, bit, needed, count_residue, state=None, physical=None
    ):
        family = transfer_family(
            node,
            flow.families,
            count_residue=None if state is None else state.count_residue,
        )
        period, residue = family.period, family.residue
        missing = [(left, right, needed)]
        for lo, hi, guard in self._records(
            flow, node, bit, state, needed=needed, physical=physical
        ):
            lo = _aligned(lo, residue, period, count_residue)
            hi = _aligned(hi, residue, period, count_residue, upper=True)
            guard = meet(guard, le(lo, hi))
            if guard is None:
                continue
            remaining = []
            for first, last, domain in missing:
                self.span_work += 1
                require(
                    self.span_work <= MAX_ANCHORED_SPAN_WORK,
                    "anchored completion interval-union budget exceeded",
                )
                if self.span_work % 128 == 0:
                    poll("anchored completion interval union", segments=self.span_work)
                remaining.extend((first, last, d) for d in uncovered(domain, (guard,)))
                require(
                    len(remaining) <= MAX_ANCHORED_SEGMENTS,
                    "anchored completion interval-union segment budget exceeded",
                )
                overlap = meet(domain, guard)
                if overlap is None:
                    continue
                for stop, active in choice(last, add(lo, -period), overlap, maximum=False):
                    active = meet(active, le(first, stop))
                    if active is not None:
                        remaining.append((first, stop, active))
                for start, active in choice(first, add(hi, period), overlap, maximum=True):
                    active = meet(active, le(start, last))
                    if active is not None:
                        remaining.append((start, last, active))
                require(
                    len(remaining) <= MAX_ANCHORED_SEGMENTS,
                    "anchored completion interval-union segment budget exceeded",
                )
            missing = remaining
            if not missing:
                break
        return tuple(domain for _, _, domain in missing)

    def _batch_coverage(self, prepared):
        points, spans = {}, {}
        for _, cases in prepared.values():
            for flow, node, left, right, bit, needed, _, state in cases:
                for boundary in transfer_routes(
                    self.guarded.graph, node, source=False, reverse=state is not None
                ):
                    if left == right:
                        guard = meet(needed, self._point_guard(node, left, state))
                        points.setdefault((flow, boundary, left, guard), {})[bit] = None
                    else:
                        spans.setdefault((flow, boundary, needed), {})[bit] = []
        physical_points = {}
        for index, ((flow, boundary, point, needed), origins) in enumerate(points.items()):
            if index % 128 == 0:
                poll("anchored batched point coverage", groups=index)
            values = flow.coverage_many(boundary, point, origins, needed)
            physical_points[flow, boundary, point, needed] = dict(zip(origins, values, strict=True))
        physical_spans = {}
        for index, (key, origins) in enumerate(spans.items()):
            if index % 128 == 0:
                poll("anchored batched span coverage", groups=index)
            flow, boundary, needed = key
            requested = 0
            for bit in origins:
                requested |= bit
            flow.coverage_batches += 1
            flow.coverage_requests += len(origins)
            for (lo, hi, guard), bits in flow.facts.get(boundary, {}).items():
                flow.coverage_fact_scans += 1
                if flow.coverage_fact_scans % 4096 == 0:
                    poll("anchored span fact coverage", facts=flow.coverage_fact_scans)
                selected = bits & requested
                if not selected:
                    continue
                active = meet(guard, needed)
                if active is None:
                    continue
                record = lo, hi, active
                while selected:
                    bit = selected & -selected
                    origins[bit].append(record)
                    flow.coverage_result_items += 1
                    selected ^= bit
            physical_spans[key] = origins
        return physical_points, physical_spans

    def _complete(self, prepared):
        flows = dict.fromkeys(case[0] for _, cases in prepared.values() for case in cases)
        for flow in flows:
            if flow.pending:
                flow.propagate()
                self.propagations += 1
                if flow is self.guarded.flows.get(None):
                    self.guarded.propagations += 1
                self._check_budget()
        physical_points, physical_spans = self._batch_coverage(prepared)
        for index, (key, (required, cases)) in enumerate(prepared.items()):
            if index % 128 == 0:
                poll("anchored completion results", queries=index)
            missing = []
            for flow, node, left, right, bit, needed, count_residue, state in cases:
                if left == right:
                    missing.extend(
                        uncovered(
                            needed,
                            self._coverage(flow, node, left, bit, needed, state, physical_points),
                        )
                    )
                else:
                    missing.extend(
                        self._span_missing(
                            flow, node, left, right, bit, needed, count_residue, state, physical_spans
                        )
                    )
            self.answers[key] = coalesce(uncovered(required, missing))

    @bounded
    def coverage(self, source, destination, source_position, destination_position, activation):
        self._sync()
        prepare_virtual(self.guarded.graph, (source, destination))
        prepared = {}
        key = self._register(
            source, destination, source_position, destination_position, activation, prepared
        )
        self._complete(prepared)
        return self.answers[key]

    def covered(self, source, destination, source_position, destination_position, activation):
        coverage = self.coverage(
            source, destination, source_position, destination_position, activation
        )
        required = self.guarded._required(activation)
        return required is None or covers(required, coverage)

    def missing(self, source, destination, source_position, destination_position, activation):
        coverage = self.coverage(
            source, destination, source_position, destination_position, activation
        )
        required = self.guarded._required(activation)
        return () if required is None else uncovered(required, coverage)

    @bounded
    def covered_many(self, queries):
        self._sync()
        queries = tuple(queries)
        prepare_virtual(
            self.guarded.graph, (endpoint for query in queries for endpoint in query[:2])
        )
        prepared, keys = {}, []
        for index, query in enumerate(queries):
            if index % 128 == 0:
                poll("anchored completion batch registration", queries=index)
            keys.append(self._register(*query, prepared))
        self._complete(prepared)
        answers = []
        for index, key in enumerate(keys):
            if index % 128 == 0:
                poll("anchored completion batch results", queries=index)
            required = self.guarded._required(key[-1])
            answers.append(required is None or covers(required, self.answers[key]))
        return answers

    def statistics(self):
        counts = [state.flow.statistics() for state in self.reverse.values()]
        return {
            "queries": self.queries,
            "cases": self.cases,
            "reverse_domains": len(self.reverse),
            "reverse_origins": self.reverse_origins,
            "propagation_passes": self.propagations,
            "span_subtractions": self.span_work,
            "translated_domain_states": 0,
            **{"reverse_" + key: sum(count[key] for count in counts) for key in FACT_STATISTICS},
        }

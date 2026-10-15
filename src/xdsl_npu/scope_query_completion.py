"""Boundedly many queries on the existing parent/native completion contract.

The only symbolic count remains the KV extent N. A static number of copies
represents neighboring serial invocations; it never expands a query count M.
Only the real per-actor issue/pipe prefixes cross the serial boundary. Neither
a child exit nor the final native instruction becomes query-wide completion.
"""

from dataclasses import dataclass

from .budget import bounded, poll
from .child_completion import ChildPort
from .completion import CountDomain, Position, Site, _first, canonical, coalesce, covers, uncovered
from .model import require
from .periodic import bounded_period
from .scope_domains import affine
from .symbolic_completion import FACT_STATISTICS, CompletionTopology, SymbolicCompletion, meet

MAX_QUERY_NODES = 100_000
MAX_QUERY_ARCS = 1_000_000
MAX_QUERY_CASES = 65_536
MAX_QUERY_ORIGINS = 65_536

POSITIVE = CountDomain(1, 0, minimum=1)


@dataclass(frozen=True)
class QuerySite:
    query: int
    call: str
    site: Site
    parent: Position


class QueryCompletion:
    """Read-only snapshot of a completed parent proof, with exact endpoints.

    ``Site.position`` remains the native's inner position; ``parent`` is its
    independent KV position. The query slot is identity, never another count.
    Coverage includes only counts where BOTH actual endpoints exist and every
    active case has its path. In particular a zero-trip native has no coverage.

    This adapter supplies neither new synchronization nor a proof of outer
    event pairing/progress. Its prefix carries apply to N >= 1; admitting an
    empty query would additionally need a state-port identity contract.
    """

    @bounded
    def __init__(self, parent, *, copies=2):
        from .scope_sync import ParentCompletion

        require(isinstance(parent, ParentCompletion), "query completion needs a parent proof")
        require(
            type(copies) is int and copies >= 2,
            "query completion needs at least two static query copies",
        )
        self.copies = copies
        self.parent, self.guarded = parent, parent.guarded()
        self.guarded._sync()
        parent.graph.potential()
        self.version = self._version()
        self.size = len(self.guarded.topology.families)
        require(
            copies * self.size <= MAX_QUERY_NODES,
            "query completion topology node budget exceeded",
        )
        self.topology = CompletionTopology()
        self.conditions = tuple(self.guarded.conditions) * copies
        for query in range(copies):
            for node, family in enumerate(self.guarded.topology.families):
                if node % 4096 == 0:
                    poll("query completion families", query=query, nodes=node)
                require(
                    self.topology.node(family) == self._node(query, node),
                    "query completion family identity changed",
                )
            for source, edges in enumerate(self.guarded.topology.edges):
                if source % 4096 == 0:
                    poll("query completion arcs", query=query, nodes=source)
                for target, shift, source_point, target_point, guard in edges:
                    self._edge(
                        self._node(query, source),
                        self._node(query, target),
                        shift,
                        source_point=source_point,
                        target_point=target_point,
                        guard=guard,
                    )
        before = self.topology.edge_count
        self._carry_prefixes()
        self.carry_edges = self.topology.edge_count - before
        self.flow = SymbolicCompletion(topology=self.topology)
        self.origins, self.bindings, self.answers = {}, {}, {}
        self.queries = self.cases = 0

    def _version(self):
        return len(self.guarded.topology.families), self.guarded.arcs, len(self.parent.graph.ids)

    def _node(self, query, node):
        return query * self.size + node

    def _edge(self, source, target, shift=0, **kwargs):
        edge = self.topology.edge(source, target, shift, **kwargs)
        require(
            self.topology.edge_count <= MAX_QUERY_ARCS,
            "query completion topology arc budget exceeded",
        )
        return edge

    def _carry_prefixes(self):
        completion = self.parent.completion
        period = self.parent.period
        for actor, calls in completion.call_order.items():
            keys = {
                (pipe, kind) for _, owner, pipe, kind, _ in completion.prefixes if owner == actor
            }
            for pipe, kind in sorted(keys, key=repr):
                poll("query completion physical prefix", actor=actor, pipe=pipe, kind=kind)
                participating = [
                    uid for uid in calls if (uid, actor, pipe, kind, 0) in completion.prefixes
                ]
                first, last = participating[0], participating[-1]
                first_stage = self.parent.program.leaves[first].call.stage.data
                last_stage = self.parent.program.leaves[last].call.stage.data
                entry = 0, -first_stage
                exit = 1, completion.max_stage[actor] - 1 - last_stage
                target = completion.prefixes[first, actor, pipe, kind, entry[1] % period]
                output = ChildPort(actor, pipe, kind, "output")
                for residue in range(period):
                    # The last physical slot may be an inactive call. Its
                    # incoming prefix still exists, while only an active
                    # child can contribute its separate outgoing state.
                    sources = (
                        completion.prefixes[last, actor, pipe, kind, residue],
                        completion.active_ports[last, output, residue],
                    )
                    for source in sources:
                        guard = meet(
                            POSITIVE,
                            meet(
                                self.guarded._point_guard(source, exit),
                                self.guarded._point_guard(target, entry),
                            ),
                        )
                        for query in range(self.copies - 1):
                            self._edge(
                                self._node(query, source),
                                self._node(query + 1, target),
                                source_point=exit,
                                target_point=entry,
                                guard=guard,
                            )

    def _required(self, domain):
        require(isinstance(domain, CountDomain), "query completion needs a KV count domain")
        require(
            type(domain.period) is int
            and domain.period > 0
            and type(domain.residue) is int
            and 0 <= domain.residue < domain.period
            and all(
                value is None or type(value) is int and value >= 0
                for value in (domain.exact, domain.minimum, domain.maximum)
            ),
            "invalid query completion count domain",
        )
        bounded_period(domain.period, "query completion count predicate")
        require(
            domain.exact is not None or domain.maximum is None or _first(domain) <= domain.maximum,
            "query completion count domain is empty",
        )
        required = canonical(domain)
        if self.parent.graph.maximum is not None:
            required = meet(required, CountDomain(1, 0, exact=self.parent.graph.maximum))
        return required

    def _bindings(self, endpoint):
        require(
            isinstance(endpoint, QuerySite)
            and type(endpoint.query) is int
            and 0 <= endpoint.query < self.copies
            and isinstance(endpoint.call, str)
            and isinstance(endpoint.site, Site)
            and isinstance(endpoint.parent, Position)
            and type(endpoint.parent.end) is bool
            and type(endpoint.parent.offset) is int,
            "query completion needs a full QuerySite identity",
        )
        require(endpoint.call in self.parent.program.leaves, "unknown query completion call")
        require(
            endpoint.site.task[0] == self.parent.program.leaves[endpoint.call].actor,
            "query completion endpoint changed its physical actor",
        )
        if endpoint not in self.bindings:
            point = affine(endpoint.parent)
            selected = {}
            for command in self.parent.completion.commands_for(endpoint.call, endpoint.site):
                for node in self.guarded._endpoint(command, None, source=True):
                    if node is None:
                        continue
                    guard = self.guarded._point_guard(node, point)
                    if guard is not None:
                        selected[self._node(endpoint.query, node), point] = guard
            self.bindings[endpoint] = tuple(
                (node, point, guard) for (node, point), guard in selected.items()
            )
        return self.bindings[endpoint]

    def _origin(self, node, point, guard):
        key = node, point
        if key not in self.origins:
            require(
                len(self.origins) < MAX_QUERY_ORIGINS, "query completion origin budget exceeded"
            )
            bit = self.origins[key] = 1 << len(self.origins)
            self.flow.seed(node, point, guard, bit)
        return self.origins[key]

    @bounded
    def coverage(self, source, destination, domain=POSITIVE):
        self.guarded._sync()
        require(
            self._version() == self.version, "parent query completion snapshot changed; rebuild"
        )
        required = self._required(domain)
        sources, targets = self._bindings(source), self._bindings(destination)
        key = source, destination, domain
        if key in self.answers:
            return self.answers[key]
        self.queries += 1
        require(self.queries <= MAX_QUERY_CASES, "query completion query budget exceeded")
        cases = []
        for first, point, source_guard in sources:
            for last, target_point, target_guard in targets:
                self.cases += 1
                require(self.cases <= MAX_QUERY_CASES, "query completion case budget exceeded")
                if self.cases % 128 == 0:
                    poll("query completion cases", cases=self.cases, queries=self.queries)
                needed = meet(required, meet(source_guard, target_guard))
                if needed is not None:
                    bit = self._origin(first, point, source_guard)
                    cases.append((last, target_point, bit, needed))
        self.flow.propagate()
        active = coalesce(needed for _, _, _, needed in cases)
        missing = [
            failure
            for node, point, bit, needed in cases
            for failure in uncovered(needed, self.flow.coverage(node, point, bit, needed))
        ]
        result = coalesce(covered for needed in active for covered in uncovered(needed, missing))
        self.answers[key] = result
        return result

    def covered(self, source, destination, domain=POSITIVE):
        coverage = self.coverage(source, destination, domain)
        required = self._required(domain)
        return required is not None and covers(required, coverage)

    def statistics(self):
        facts = self.flow.statistics()
        return {
            "query_copies": self.copies,
            "parent_node_families": self.size,
            "topology_nodes": len(self.topology.families),
            "topology_edges": self.topology.edge_count,
            "boundary_prefix_edges": self.carry_edges,
            "queries": self.queries,
            "cases": self.cases,
            "origins": len(self.origins),
            "expanded_query_instances": 0,
            **{key: facts[key] for key in FACT_STATISTICS},
        }

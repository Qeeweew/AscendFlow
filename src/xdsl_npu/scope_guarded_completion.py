"""Count-conditioned scope queries on the existing hardware completion graph.

Boundary facts retain their original count predicates. Interior queries use
translation only after a potential certificate separates all fixed first/last
nodes from their paths. Native arcs, milestones and prefix induction are the
existing Graph/SymbolicCompletion contracts; no hardware model is rebuilt.
"""

from collections.abc import Mapping
from dataclasses import replace

from .budget import bounded, poll
from .completion import CountDomain, coalesce, covers, uncovered
from .completion_flow import ALL
from .loop_graph import Command
from .model import require
from .scope_domains import Activation, affine
from .scope_transfer_queries import (
    VirtualCut,
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
    SymbolicCompletion,
    add,
    le,
    meet,
    native_family,
)

MAX_GUARDED_CASES = 16_384
MAX_GUARDED_ORIGINS = 65_536
MAX_GUARDED_TRANSFERS = 2_000_000


class NodeActivations(Mapping):
    """An owned append-only table; imported graph nodes have frozen bindings.

    Existing immutable Activation values may be published again unchanged.
    A new binding on an already imported node is forbidden too: absence is
    part of that node's established activation contract.
    """

    def __init__(self, values=()):
        self._values, self._entries, self._sealed_nodes = {}, [], 0
        self.update(values)

    def __getitem__(self, node):
        return self._values[node]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __setitem__(self, node, activation):
        require(type(node) is int and node >= 0, "invalid guarded scope activation node")
        require(isinstance(activation, Activation), "scope completion needs an Activation")
        if node in self._values:
            require(
                self._values[node] == activation,
                "an existing guarded scope node activation changed",
            )
            return
        require(node >= self._sealed_nodes, "an existing guarded scope node activation changed")
        self._values[node] = activation
        self._entries.append((node, activation))

    def __delitem__(self, node):
        require(False, "an existing guarded scope node activation changed")

    def update(self, values):
        for node, activation in values.items() if isinstance(values, Mapping) else values:
            self[node] = activation

    @property
    def revision(self):
        return len(self._entries)

    def appended(self, revision):
        require(
            type(revision) is int and 0 <= revision <= len(self._entries),
            "guarded scope activation table revision changed",
        )
        return self._entries[revision:]

    def seal_nodes(self, count):
        require(count >= self._sealed_nodes, "guarded scope activation nodes must be append-only")
        self._sealed_nodes = count


def _domain(domain):
    require(
        isinstance(domain, CountDomain)
        and type(domain.period) is int
        and domain.period > 0
        and type(domain.residue) is int
        and 0 <= domain.residue < domain.period
        and all(
            value is None or type(value) is int and value >= 0
            for value in (domain.exact, domain.minimum, domain.maximum)
        ),
        "invalid guarded scope count domain",
    )


def _activation(activation, period):
    require(isinstance(activation, Activation), "scope completion needs an Activation")
    _domain(activation.count)
    require(
        type(activation.period) is int
        and activation.period > 0
        and period % activation.period == 0
        and activation.residues
        and activation.residues == tuple(sorted(set(activation.residues)))
        and all(type(r) is int and 0 <= r < activation.period for r in activation.residues),
        "guarded scope activation period must divide the graph period",
    )
    for point in (activation.first, activation.last):
        require(
            type(point.end) is bool and type(point.offset) is int,
            "invalid guarded scope parent position",
        )
    first, last = affine(activation.first), affine(activation.last)
    require(
        first == last or first[0] == 0 and last[0] == 1,
        "guarded scope activation needs a first/last singleton or a constant-to-end interval",
    )
    return first, last


class GuardedCompletion:
    """Append-only adapter; node activations constrain their own generations.

    Query activations constrain the SOURCE epoch g, and distance targets g+d.
    An integer endpoint identifies the whole residue family of that node's
    command/point; a Command uses Graph's ordinary begin/end defaults.
    """

    def __init__(self, graph, node_activations):
        self.graph, self.node_activations = graph, node_activations
        self.topology = CompletionTopology()
        self.flows, self.origins, self.answers = {}, {}, {}
        self.endpoints, self.boundaries = {}, {}
        self.conditions, self.registered = [], {}
        self.source_bindings = {}
        self._transfer_revision = transfer_revision(graph)
        self._activation_revision = 0
        self.activation_checks = 0
        self.arcs = self.cases = self.origin_count = self.queries = self.propagations = 0
        self.aliases = 0
        self._sync()

    def _family(self, node):
        meta = self.graph.nodes[node]
        family, condition = native_family(meta, self.graph.period), ALL
        activation = self.node_activations.get(node)
        if activation is not None:
            first, last = _activation(activation, self.graph.period)
            condition = (
                activation.count
                if meta.residue % activation.period in activation.residues
                else None
            )
            if first == last:
                if family.last is not None:
                    condition = meet(
                        condition, meet(le(first, family.last), le(family.last, first))
                    )
                family = replace(family, last=first, monotone=False)
            else:
                require(
                    family.horizon[0] == 1, "guarded scope interval needs a parent-count horizon"
                )
                family = replace(
                    family,
                    lower=max(family.lower, first[1]),
                    horizon=(1, min(family.horizon[1], last[1])),
                )
        return family, condition

    def _sync(self):
        poll("guarded scope graph import", nodes=len(self.graph.nodes), arcs=self.arcs)
        old_nodes = len(self.topology.families)
        require(
            len(self.graph.nodes) >= old_nodes and len(self.graph.edges) >= self.arcs,
            "guarded scope graph must be append-only",
        )
        owned = type(self.node_activations) is NodeActivations
        if owned:
            additions = self.node_activations.appended(self._activation_revision)
        else:
            for node, activation in self.registered.items():
                self.activation_checks += 1
                require(
                    self.node_activations.get(node) == activation,
                    "an existing guarded scope node activation changed",
                )
            additions = self.node_activations.items()
        for node, activation in additions:
            self.activation_checks += 1
            require(
                type(node) is int and 0 <= node < len(self.graph.nodes),
                "guarded scope activation has no actual graph node",
            )
            require(
                node >= old_nodes or self.registered.get(node) == activation,
                "an existing guarded scope node activation changed",
            )
        current_transfer = transfer_revision(self.graph)
        changed = (
            old_nodes != len(self.graph.nodes)
            or self.arcs != len(self.graph.edges)
            or current_transfer != self._transfer_revision
        )
        for node in range(old_nodes, len(self.graph.nodes)):
            if node % 4096 == 0:
                poll("guarded scope topology", nodes=node)
            family, condition = self._family(node)
            require(self.topology.node(family) == node, "guarded scope family identity changed")
            self.conditions.append(condition)
            if node in self.node_activations:
                self.registered[node] = self.node_activations[node]
        for arc in range(self.arcs, len(self.graph.edges)):
            if arc % 4096 == 0:
                poll("guarded scope arcs", arcs=arc)
            source, target, distance = self.graph.edges[arc]
            shift = (
                distance * self.graph.period
                + self.graph.nodes[target].residue
                - self.graph.nodes[source].residue
            )
            edge = self.topology.edge(
                source, target, shift, guard=meet(self.conditions[source], self.conditions[target])
            )
            if edge is not None:
                for flow in self.flows.values():
                    flow.replay(source, edge)
        self.arcs = len(self.graph.edges)
        if current_transfer != self._transfer_revision:
            for flow, bindings in self.source_bindings.items():
                for bit, (node, point) in bindings.items():
                    if isinstance(node, VirtualNode):
                        guard = self._point_guard(node, point)
                        for boundary in transfer_routes(
                            self.graph, node, source=True, since=self._transfer_revision
                        ):
                            flow.seed(boundary, point, guard, bit)
            self._transfer_revision = current_transfer
        if owned:
            self.node_activations.seal_nodes(len(self.graph.nodes))
            self._activation_revision = self.node_activations.revision
        if changed or self.aliases != len(self.graph.ids):
            # Appended paths cannot invalidate a proof covering its entire
            # required count domain. Partial answers still need fresh queries.
            # Endpoint tuples are part of the key: newly published residue
            # aliases receive a different key after clearing the lookup below.
            for key in tuple(self.answers):
                required = self._required(key[-1])
                if required is not None and not covers(required, self.answers[key]):
                    del self.answers[key]
            self.endpoints.clear()
            self.boundaries.clear()
        self.aliases = len(self.graph.ids)
        if changed:
            self.graph.potentials = None
        self._check_transfers()

    def _endpoint(self, endpoint, point, *, source):
        if isinstance(endpoint, VirtualCut):
            require(
                point is None or point == endpoint.site.point,
                "virtual native cut milestone changed",
            )
            require(
                endpoint.parent_period == self.graph.period,
                "virtual native cut belongs to a different parent period",
            )
            return tuple(endpoint.resolve(residue) for residue in range(self.graph.period))
        if type(endpoint) is int:
            require(0 <= endpoint < len(self.graph.nodes), "unknown guarded scope endpoint")
            meta = self.graph.nodes[endpoint]
            command, point = meta.command, meta.point if point is None else point
        else:
            require(isinstance(endpoint, Command), "guarded scope endpoint needs a node or Command")
            command = endpoint
        if point is None:
            point = (
                "end"
                if command.kind in {"set", "cross_set"} or source and command.kind == "task"
                else "begin"
            )
        require(type(command.index) is int and type(point) is str, "invalid guarded scope endpoint")
        key = command.index, point
        if key not in self.endpoints:
            nodes = tuple(
                self.graph.ids.get((command.index, residue, point))
                for residue in range(self.graph.period)
            )
            require(
                any(node is not None for node in nodes), "unknown guarded scope endpoint family"
            )
            self.endpoints[key] = nodes
        return self.endpoints[key]

    def _node(self, endpoint, residue):
        return endpoint[residue]

    def _point_guard(self, node, point):
        return transfer_point_guard(node, point, self.topology.families, self.conditions)

    def _flow(self, extent):
        if extent not in self.flows:
            self.flows[extent] = SymbolicCompletion(
                topology=self.topology, count=extent, terminal=extent is None
            )
            self.origins[extent] = {}
            self.source_bindings[self.flows[extent]] = {}
        return self.flows[extent], self.origins[extent]

    def _reach(self, source, target, point, distance, needed, *, extent=None):
        flow, origins = self._flow(extent)
        origin = source, point
        if origin not in origins:
            self.origin_count += 1
            require(
                self.origin_count <= MAX_GUARDED_ORIGINS, "guarded scope origin budget exceeded"
            )
            bit = origins[origin] = 1 << len(origins)
            self.source_bindings[flow][bit] = source, point
            guard = (
                self._point_guard(source, point)
                if isinstance(source, VirtualNode)
                else self.conditions[source]
            )
            for boundary in transfer_routes(self.graph, source, source=True):
                flow.seed(boundary, point, guard, bit)
        return flow, target, add(point, distance), origins[origin], needed

    def _coverage(self, flow, node, point, bit, needed, physical=None):
        result = []
        for boundary in transfer_routes(self.graph, node, source=False):
            result.extend(
                flow.coverage(boundary, point, bit, needed)
                if physical is None
                else physical[flow, boundary, point, needed][bit]
            )
        source, source_point = self.source_bindings[flow][bit]
        if transfer_direct(self.graph, source, node):
            guard = meet(
                needed,
                meet(self._point_guard(source, source_point), self._point_guard(node, point)),
            )
            guard = meet(guard, meet(le(source_point, point), le(point, source_point)))
            if guard is not None:
                result.append(guard)
        return result

    def _check_transfers(self):
        require(
            sum(f.transfers for f in self.flows.values()) <= MAX_GUARDED_TRANSFERS,
            "guarded scope propagation budget exceeded",
        )

    def _case(self):
        self.cases += 1
        require(
            self.cases <= MAX_GUARDED_CASES, "guarded scope boundary/query case budget exceeded"
        )
        if self.cases % 128 == 0:
            poll("guarded scope query cases", cases=self.cases, origins=self.origin_count)

    def _point_case(self, source, target, point, distance, activation, required):
        self._case()
        first, last = affine(activation.first), affine(activation.last)
        needed = meet(required, meet(le(first, point), le(point, last)))
        needed = meet(
            needed,
            meet(self._point_guard(source, point), self._point_guard(target, add(point, distance))),
        )
        if needed is None:
            return ()
        return (self._reach(source, target, point, distance, needed),)

    def _boundary_measures(self, required, potential):
        """Count-specific singleton extrema, shared by all endpoint queries."""
        if required in self.boundaries:
            return self.boundaries[required]
        stride, ranks = potential
        period = self.graph.period
        head = tail = None
        for node, family in enumerate(self.topology.families):
            if node % 4096 == 0:
                poll("guarded scope boundary bounds", nodes=node)
            if family.last is None or meet(required, self._point_guard(node, family.last)) is None:
                continue
            a, b = family.last
            residue = self.graph.nodes[node].residue
            if a == 0:
                measure = stride * (b - residue) + period * ranks[node]
                head = measure if head is None else max(head, measure)
            else:
                measure = stride * (-b + residue) - period * ranks[node]
                tail = measure if tail is None else max(tail, measure)
        result = self.boundaries[required] = head, tail
        return result

    def _bounds(self, source, target, distance, boundary, stride, ranks):
        """Necessary potential inequalities for a path touching a singleton.

        Q(node,g) = stride*(g-residue)/P + rank[node] increases on every
        hardware arc. Q(source,g) <= Q(first,k) bounds the head, and
        Q(last,N-gap) <= Q(target,g+d) bounds the tail. Maxima can be
        taken before integer division, so one count-specific scan suffices.
        """
        period = self.graph.period
        source_nodes = transfer_routes(self.graph, source, source=True)
        target_nodes = transfer_routes(self.graph, target, source=False)
        if not source_nodes or not target_nodes:
            return None, None
        rs = source.residue if isinstance(source, VirtualNode) else self.graph.nodes[source].residue
        rt = target.residue if isinstance(target, VirtualNode) else self.graph.nodes[target].residue
        source_rank = min(ranks[node] for node in source_nodes)
        target_rank = max(ranks[node] for node in target_nodes)
        head, tail = boundary
        return (
            None if head is None else rs + (head - period * source_rank) // stride,
            None if tail is None else distance - rt - 1 + (tail + period * target_rank) // stride,
        )

    def _interval_cases(self, source, target, distance, activation, required, boundary, potential):
        first, last = affine(activation.first), affine(activation.last)
        lo, gap = first[1], -1 - last[1]
        source_family, target_family = (
            transfer_family(source, self.topology.families),
            transfer_family(target, self.topology.families),
        )
        lo = max(lo, source_family.lower, target_family.lower - distance)
        gap = max(gap, -1 - source_family.horizon[1], -1 - target_family.horizon[1] + distance)
        required = meet(
            required,
            meet(transfer_condition(source, self.conditions), transfer_condition(target, self.conditions)),
        )
        required = meet(required, le((0, lo), (1, -1 - gap)))
        if required is None:
            return ()
        # A singleton endpoint fixes the only possible source coordinate.
        # The ordinary point query intersects both endpoint guards, including
        # their count/residue constraints and a second singleton's equality.
        if source_family.last is not None:
            return self._point_case(
                source, target, source_family.last, distance, activation, required
            )
        if target_family.last is not None:
            return self._point_case(
                source, target, add(target_family.last, -distance), distance, activation, required
            )
        head, tail = self._bounds(source, target, distance, boundary, *potential)
        cases = []
        residue = source.residue if isinstance(source, VirtualNode) else self.graph.nodes[source].residue
        period = self.graph.period
        if head is not None:
            begin = lo + (residue - lo) % period
            require(
                max(0, (head - begin) // period + 1) + self.cases <= MAX_GUARDED_CASES,
                "guarded scope boundary/query case budget exceeded",
            )
            for generation in range(begin, head + 1, period):
                cases.extend(
                    self._point_case(
                        source, target, (0, generation), distance, activation, required
                    )
                )
            lo = max(lo, head + 1)
        if tail is not None:
            require(
                max(0, tail - gap + 1) + self.cases <= MAX_GUARDED_CASES,
                "guarded scope boundary/query case budget exceeded",
            )
            for offset in range(gap, tail + 1):
                cases.extend(
                    self._point_case(
                        source, target, (1, -1 - offset), distance, activation, required
                    )
                )
            gap = max(gap, tail + 1)
        generation = lo + (residue - lo) % period
        extent = generation + gap + 1
        needed = meet(required, CountDomain(1, 0, minimum=max(0, extent)))
        if needed is not None:
            self._case()
            require(extent > 0, "guarded scope translation needs a positive extent")
            cases.append(
                self._reach(source, target, (0, generation), distance, needed, extent=extent)
            )
        return tuple(cases)

    def _required(self, activation):
        required = activation.count
        if self.graph.maximum is not None:
            required = meet(required, CountDomain(1, 0, exact=self.graph.maximum))
        return required

    def _register(
        self,
        source,
        destination,
        distance,
        activation,
        *,
        prepared,
        source_point=None,
        destination_point=None,
    ):
        require(type(distance) is int, "guarded scope distance must be signed integer")
        first, last = _activation(activation, self.graph.period)
        source = self._endpoint(source, source_point, source=True)
        target = self._endpoint(destination, destination_point, source=False)
        key = source, target, distance, activation
        if key in self.answers or key in prepared:
            return key
        self.queries += 1
        poll("guarded scope query", queries=self.queries)
        required = self._required(activation)
        if required is None:
            self.answers[key] = ()
            return key
        if distance == 0 and all(
            a is None or b is None or transfer_direct(self.graph, a, b)
            for a, b in zip(source, target, strict=True)
        ):
            self.answers[key] = (required,)
            return key
        cases = []
        potential = self.graph.potential()
        boundary = self._boundary_measures(required, potential) if first != last else None
        for residue in range(self.graph.period):
            if residue % 256 == 0:
                poll("guarded scope residues", residues=residue)
            if residue % activation.period not in activation.residues:
                continue
            a, b = (
                self._node(source, residue),
                self._node(target, (residue + distance) % self.graph.period),
            )
            if a is None or b is None:
                continue
            if first == last:
                cases.extend(self._point_case(a, b, first, distance, activation, required))
            else:
                cases.extend(
                    self._interval_cases(a, b, distance, activation, required, boundary, potential)
                )
        prepared[key] = required, cases
        return key

    def _complete(self, prepared):
        # Register every source before propagating. Distinct origins retain
        # distinct bits, while shared intervals and guards travel once.
        flows = dict.fromkeys(case[0] for _, cases in prepared.values() for case in cases)
        for flow in flows:
            if flow.pending:
                self.propagations += 1
                flow.propagate()
                self._check_transfers()
        groups = {}
        for _, cases in prepared.values():
            for flow, node, point, bit, needed in cases:
                for boundary in transfer_routes(self.graph, node, source=False):
                    groups.setdefault((flow, boundary, point, needed), {})[bit] = None
        physical = {}
        for index, ((flow, boundary, point, needed), origins) in enumerate(groups.items()):
            if index % 128 == 0:
                poll("guarded scope batched coverage", groups=index)
            results = flow.coverage_many(boundary, point, origins, needed)
            physical[flow, boundary, point, needed] = dict(zip(origins, results, strict=True))
        for query, (key, (required, cases)) in enumerate(prepared.items()):
            if query % 128 == 0:
                poll("guarded scope query results", queries=query)
            missing = []
            for index, (flow, node, point, bit, needed) in enumerate(cases):
                if index % 128 == 0:
                    poll("guarded scope coverage", cases=index, queries=self.queries)
                missing.extend(
                    uncovered(needed, self._coverage(flow, node, point, bit, needed, physical))
                )
            self.answers[key] = coalesce(uncovered(required, missing))

    @bounded
    def coverage(
        self,
        source,
        destination,
        distance,
        activation,
        *,
        source_point=None,
        destination_point=None,
    ):
        """Counts where every active source epoch has its required path.

        A known endpoint family's missing residue copies are inactive. Unknown
        commands or milestone names are errors. As with Graph queries, the
        obligation exists only where both endpoints are active; memory
        ownership must separately validate their claimed activations.
        """
        self._sync()
        prepare_virtual(self.graph, (source, destination))
        prepared = {}
        key = self._register(
            source,
            destination,
            distance,
            activation,
            prepared=prepared,
            source_point=source_point,
            destination_point=destination_point,
        )
        self._complete(prepared)
        return self.answers[key]

    def covered(self, source, destination, distance, activation, **kwargs):
        coverage = self.coverage(source, destination, distance, activation, **kwargs)
        required = self._required(activation)
        return required is None or covers(required, coverage)

    def missing(self, source, destination, distance, activation, **kwargs):
        coverage = self.coverage(source, destination, distance, activation, **kwargs)
        required = self._required(activation)
        return () if required is None else uncovered(required, coverage)

    @bounded
    def covered_many(self, queries):
        """Seed all unique obligations before propagating shared domains."""
        self._sync()
        queries = tuple(queries)
        prepare_virtual(self.graph, (endpoint for query in queries for endpoint in query[:2]))
        prepared, keys = {}, []
        for index, query in enumerate(queries):
            if index % 128 == 0:
                poll("guarded scope batch registration", queries=index)
            keys.append(self._register(*query, prepared=prepared))
        self._complete(prepared)
        answers = []
        for index, key in enumerate(keys):
            if index % 128 == 0:
                poll("guarded scope batch results", queries=index)
            required = self._required(key[-1])
            answers.append(required is None or covers(required, self.answers[key]))
        return answers

    def statistics(self):
        counts = [flow.statistics() for flow in self.flows.values()]
        return {
            "queries": self.queries,
            "topology_instances": 1,
            "topology_nodes": len(self.topology.families),
            "topology_edges": self.topology.edge_count,
            "query_domains": len(self.flows),
            "origins": self.origin_count,
            "cases": self.cases,
            "propagation_passes": self.propagations,
            "boundary_domains": len(self.boundaries),
            "activation_checks": self.activation_checks,
            "translated_domain_states": 0,
            **{key: sum(count[key] for count in counts) for key in FACT_STATISTICS},
        }

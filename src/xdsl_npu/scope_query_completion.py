"""Boundedly many queries on the existing parent/native completion contract.

The only symbolic count remains the KV extent N. A static number of copies
represents neighboring serial invocations; it never expands a query count M.
Real per-actor issue/pipe prefixes, exclusive cross-counter Set FIFO and
actual QueryChannel/QuerySync transitions cross the serial boundary. Neither
a child exit nor the final native instruction becomes query-wide completion.
"""

from dataclasses import dataclass

from .budget import bounded, poll
from .child_completion import ChildPort
from .completion import CountDomain, Position, Site, _first, canonical, coalesce, covers, uncovered
from .loop_graph import Command
from .model import require
from .periodic import bounded_period
from .scope_domains import affine, shifted
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


@dataclass(frozen=True)
class QueryEvent:
    """An actual parent/query Set effect or Wait consumption at its KV position."""

    query: int
    actor: str
    channel: str
    action: str
    parent: Position


class QueryCompletion:
    """Completed parent snapshot with append-only actual query transitions.

    ``Site.position`` remains the native's inner position; ``parent`` is its
    independent KV position. The query slot is identity, never another count.
    Coverage includes only counts where BOTH actual endpoints exist and every
    active case has its path. In particular a zero-trip native has no coverage.

    This adapter reads synchronization only from actual IR. It is not a proof
    of outer event pairing/progress. Its prefix carries apply to N >= 1;
    admitting an empty query additionally needs a state-port identity contract.
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
        from .scope_resources import invocation_snapshot

        self._native_snapshot = invocation_snapshot(parent.program)
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
        before = self.topology.edge_count
        self._carry_cross_counters()
        self.counter_edges = self.topology.edge_count - before
        self.origins, self.bindings, self.answers = {}, {}, {}
        self._incomplete = {}
        self.queries = self.cases = 0
        from .scope_query_transitions import install_query_transitions

        install_query_transitions(self)
        self.flow = SymbolicCompletion(topology=self.topology)

    def _version(self):
        from .scope_query_protocols import read_query_protocols

        return (
            len(self.guarded.topology.families),
            self.guarded.arcs,
            len(self.parent.graph.ids),
            tuple(self.parent.protocols),
            tuple(self.parent.events.items()),
            read_query_protocols(self.parent.program, self.parent.access)
            if self.parent.program.invocations
            else (),
        )

    def _node(self, query, node):
        return query * self.size + node

    def _edge(self, source, target, shift=0, **kwargs):
        edge = self.topology.edge(source, target, shift, **kwargs)
        require(
            self.topology.edge_count <= MAX_QUERY_ARCS,
            "query completion topology arc budget exceeded",
        )
        if edge is not None and (flow := getattr(self, "flow", None)) is not None:
            flow.replay(source, edge)
        return edge

    def _append_snapshot(self):
        """Reconstruct an actual suffix without accepting edits to old inputs."""
        from .scope_resources import invocation_snapshot

        self.guarded._sync()
        current = self._version()
        previous = tuple(self.query_protocols.values())
        require(
            current[:-1] == self.version[:-1]
            and self.version[-1] == previous
            and current[-1][: len(previous)] == previous
            and self._native_snapshot == invocation_snapshot(self.parent.program),
            "query append changed the native snapshot or an existing protocol prefix",
        )
        return current, current[-1][len(previous) :]

    @bounded
    def append_protocols(self):
        """Import only newly installed actual protocols into this same flow.

        New arcs replay the existing source facts. A complete old answer is
        monotone; partial answers keep their exact cases and must be read again
        after the new deltas propagate. No origin or native topology is rebuilt.
        """
        from .scope_query_transitions import append_query_transitions

        current, additions = self._append_snapshot()
        require(
            all(protocol.target.query < self.copies for protocol in additions),
            "query proof window omits a protocol consumer",
        )
        if additions:
            append_query_transitions(self, additions)
            for key in self._incomplete:
                self.answers.pop(key, None)
        self.version = current
        return additions

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

    def _event_command(self, endpoint):
        require(
            endpoint.actor in self.parent.access.loop.actors,
            "unknown query completion event actor",
        )
        protocols = [p for p in self.parent.protocols if p.name == endpoint.channel]
        require(len(protocols) == 1, "unknown query completion event channel")
        protocol = protocols[0]
        require(protocol.kind in {"local", "cross"}, "query completion event has no Set/Wait")
        participants = protocol.sources if endpoint.action == "set" else protocol.destinations
        owners = [owner for owner in participants if owner[0] == endpoint.actor]
        require(len(owners) == 1, "query completion event has no actual participant")
        owner = owners[0]
        tasks = [
            task
            for task in self.parent.access.loop.actors[endpoint.actor].program.tasks
            if task.uid == owner[1]
        ]
        require(len(tasks) == 1, "query completion event has no actual participant task")
        require(
            owner in self.parent.access.endpoints, "query completion event has no native endpoint"
        )
        uid, site = self.parent.access.endpoints[owner]
        leaf = self.parent.program.leaves.get(uid)
        require(
            leaf is not None
            and leaf.actor == endpoint.actor
            and site.task[0] == endpoint.actor
            and site.region == uid
            and any(
                task.uid == site.task[1]
                for data in leaf.loop.actors.values()
                for task in data.program.tasks
            ),
            "query completion event changed its actual native participant",
        )
        command = self.parent.events.get((endpoint.actor, endpoint.channel, endpoint.action))
        expected_kind = ("cross_" if protocol.kind == "cross" else "") + endpoint.action
        activation = (
            None
            if protocol.activation is None
            else shifted(protocol.activation, 0 if endpoint.action == "set" else protocol.distance)
        )
        residues = (
            protocol.residues
            if endpoint.action == "set"
            else tuple(sorted((r + protocol.distance) % protocol.period for r in protocol.residues))
        )
        lower, upper = (
            (0, protocol.distance) if endpoint.action == "set" else (protocol.distance, 0)
        )
        if activation is not None:
            residues = activation.residues
            lower = 0 if activation.first.end else activation.first.offset
            upper = activation.last.offset if activation.last.end else 0
        require(
            isinstance(command, Command)
            and command.actor == endpoint.actor
            and command.channel == endpoint.channel
            and command.kind == expected_kind
            and command.task is tasks[0]
            and (command.period, command.residues, command.lower, command.upper)
            == (protocol.period, residues, lower, upper),
            "query completion event has no matching actual command",
        )
        point = "end" if endpoint.action == "set" else "begin"
        for node in self.guarded._endpoint(command, point, source=True):
            require(
                node is None or self.parent.graph.nodes[node].command is command,
                "query completion event changed its actual command",
            )
        return command

    def _carry_cross_counters(self):
        from .scope_native_activation import validate_scope_protocol

        used, period = set(), self.parent.period
        for protocol in self.parent.protocols:
            if protocol.kind != "cross":
                continue
            require(
                len(protocol.ids) == 1
                and type(protocol.ids[0]) is int
                and 0 <= protocol.ids[0] < 8
                and protocol.ids[0] not in used,
                "query completion cross counter needs one allocated exclusive ID",
            )
            used.update(protocol.ids)
            require(
                protocol.activation is None and not protocol.last and not protocol.carry,
                "query completion cross counter activation/case form is unsupported",
            )
            require(
                type(protocol.period) is int
                and protocol.period > 0
                and period % protocol.period == 0
                and protocol.residues
                and protocol.residues == tuple(sorted(set(protocol.residues)))
                and all(type(r) is int and 0 <= r < protocol.period for r in protocol.residues),
                "query completion cross counter has invalid actual residues",
            )
            require(
                len({a for a, _ in protocol.sources}) == len(protocol.sources)
                and len({a for a, _ in protocol.destinations}) == len(protocol.destinations)
                and ({a for a, _ in protocol.sources}, {a for a, _ in protocol.destinations})
                in (({"aic"}, {"aiv0", "aiv1"}), ({"aiv0", "aiv1"}, {"aic"})),
                "query completion cross counter needs its actual AIC/two-AIV quorum",
            )
            validate_scope_protocol(self.parent.program, self.parent.access, protocol)
            sends = []
            for participants, action in (
                (protocol.sources, "set"),
                (protocol.destinations, "wait"),
            ):
                for actor, _ in participants:
                    command = self._event_command(
                        QueryEvent(0, actor, protocol.name, action, Position(False, 0))
                    )
                    if action == "set":
                        sends.append(command)
            for command in sends:
                head = min(
                    command.lower + (r - command.lower) % command.period for r in command.residues
                )
                entry = 0, head
                target = self.parent.graph.ids.get((command.index, head % period, "end"))
                require(
                    target is not None, "query completion cross counter has no actual Set effect"
                )
                for residue in range(period):
                    poll(
                        "query completion cross counter FIFO",
                        channel=protocol.name,
                        residue=residue,
                    )
                    gap = command.upper + min(
                        (residue - 1 - command.upper - r) % command.period for r in command.residues
                    )
                    exit = 1, -1 - gap
                    source = self.parent.graph.ids.get(
                        (command.index, (residue + exit[1]) % period, "end")
                    )
                    require(
                        source is not None,
                        "query completion cross counter has no actual Set effect",
                    )
                    guard = meet(
                        CountDomain(period, residue, minimum=1),
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

    def _validate_identity(self, endpoint):
        """Validate scalar types before equality can hit an already checked key."""
        require(
            isinstance(endpoint, (QuerySite, QueryEvent))
            and type(endpoint.query) is int
            and 0 <= endpoint.query < self.copies
            and isinstance(endpoint.parent, Position)
            and type(endpoint.parent.end) is bool
            and type(endpoint.parent.offset) is int,
            "query completion needs a full QuerySite or QueryEvent identity",
        )
        if isinstance(endpoint, QueryEvent):
            require(
                isinstance(endpoint.actor, str)
                and isinstance(endpoint.channel, str)
                and isinstance(endpoint.action, str)
                and endpoint.action in {"set", "wait"},
                "query completion needs a Set/Wait QueryEvent identity",
            )
        else:
            require(
                isinstance(endpoint.call, str)
                and isinstance(endpoint.site, Site)
                and isinstance(endpoint.site.position, Position)
                and type(endpoint.site.position.end) is bool
                and type(endpoint.site.position.offset) is int
                and type(endpoint.site.call) is int,
                "query completion needs a full QuerySite identity",
            )

    def _bindings(self, endpoint):
        self._validate_identity(endpoint)
        if isinstance(endpoint, QueryEvent):
            if endpoint.channel in self.query_protocols:
                from .scope_query_transitions import query_event_bindings

                return query_event_bindings(self, endpoint)
            commands = (self._event_command(endpoint),)
            event_point = "end" if endpoint.action == "set" else "begin"
        else:
            require(endpoint.call in self.parent.program.leaves, "unknown query completion call")
            require(
                endpoint.site.task[0] == self.parent.program.leaves[endpoint.call].actor,
                "query completion endpoint changed its physical actor",
            )
            commands = self.parent.completion.commands_for(endpoint.call, endpoint.site)
            event_point = None
        if endpoint not in self.bindings:
            point = affine(endpoint.parent)
            selected = {}
            for command in commands:
                for node in self.guarded._endpoint(command, event_point, source=True):
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

    def _check_snapshot(self):
        self.guarded._sync()
        require(
            self._version() == self.version, "parent query completion snapshot changed; rebuild"
        )

    def _register(self, source, destination, domain, prepared):
        required = self._required(domain)
        self._validate_identity(source)
        self._validate_identity(destination)
        key = source, destination, domain
        if key in self.answers or key in prepared:
            return key
        if key in self._incomplete:
            prepared[key] = self._incomplete[key]
            return key
        sources, targets = self._bindings(source), self._bindings(destination)
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
        prepared[key] = required, cases
        return key

    def _complete(self, cases, coverage):
        active = coalesce(needed for _, _, _, needed in cases)
        missing = [
            failure
            for node, point, bit, needed in cases
            for failure in uncovered(needed, coverage[node, point, needed][bit])
        ]
        result = coalesce(covered for needed in active for covered in uncovered(needed, missing))
        return result

    @bounded
    def coverage_many(self, queries):
        """Cover ordered (source, destination, domain) triples in one propagation."""
        self._check_snapshot()
        # Some low-level coverage fixtures construct this adapter around an
        # existing flow. The case ledger belongs to its first completed batch.
        if not hasattr(self, "_incomplete"):
            self._incomplete = {}
        prepared, ordered = {}, []
        for source, destination, domain in queries:
            if len(ordered) % 128 == 0:
                poll("query completion batch", queries=len(ordered))
            ordered.append(self._register(source, destination, domain, prepared))
        self._check_snapshot()
        self.flow.propagate()
        groups = {}
        for _, cases in prepared.values():
            for node, point, bit, needed in cases:
                # Node identity includes the static query copy. Keep the full
                # count domain; equal residue alone does not mean equal need.
                groups.setdefault((node, point, needed), {})[bit] = None
        coverage = {}
        for (node, point, needed), origins in groups.items():
            if len(coverage) % 128 == 0:
                poll("query completion coverage", groups=len(coverage))
            results = self.flow.coverage_many(node, point, origins, needed)
            coverage[node, point, needed] = dict(zip(origins, results, strict=True))
        for key, (required, cases) in prepared.items():
            result = self.answers[key] = self._complete(cases, coverage)
            if required is not None and covers(required, result):
                self._incomplete.pop(key, None)
            else:
                self._incomplete[key] = required, cases
        return [self.answers[key] for key in ordered]

    @bounded
    def covered_many(self, queries):
        """Require every requested count for each ordered endpoint pair."""
        queries = tuple(queries)
        coverage = self.coverage_many(queries)
        return [
            required is not None and covers(required, result)
            for (_, _, domain), result in zip(queries, coverage, strict=True)
            for required in (self._required(domain),)
        ]

    def coverage(self, source, destination, domain=POSITIVE):
        return self.coverage_many(((source, destination, domain),))[0]

    def covered(self, source, destination, domain=POSITIVE):
        return self.covered_many(((source, destination, domain),))[0]

    def statistics(self):
        facts = self.flow.statistics()
        return {
            "query_copies": self.copies,
            "parent_node_families": self.size,
            "topology_nodes": len(self.topology.families),
            "topology_edges": self.topology.edge_count,
            "boundary_prefix_edges": self.carry_edges,
            "cross_counter_fifo_edges": self.counter_edges,
            "query_protocols": len(self.query_protocols),
            "query_protocol_edges": self.query_protocol_edges,
            "queries": self.queries,
            "cases": self.cases,
            "origins": len(self.origins),
            "expanded_query_instances": 0,
            **{key: facts[key] for key in FACT_STATISTICS},
        }

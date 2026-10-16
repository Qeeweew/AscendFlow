"""Compose guarded blocking interfaces on actual invocation regions.

Only selected notification effects and real queue/controller gates participate.
A child exports parameterized queue summaries and actual ancestor-open ports,
never a native completion at child exit. The ancestor owning a clock composes
cross-generation gates from actual call positions and coordinate translations;
those gates need not be child-local zero-step transfers. Loop streams keep
their clock; finite ports keep the same parameter expressions in every
condition. No runtime iteration or branch history is expanded.
"""

from __future__ import annotations

from dataclasses import replace

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from ..target.ascend910b import Actor
from . import progress_regions as regions
from .event_contracts import ProgressPoint, StructuredProgress
from .progress_conditions import ProgressConditions
from .progress_connectivity import RegionConnectivity
from .progress_endpoints import ProgressEndpoints
from .progress_ranks import ProgressRanks
from .progress_regions import is_source, is_wait, port_key
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard, Comparison


def _guard(tests):
    return ActionGuard(tuple(dict.fromkeys(tests)))


class _Composer:
    MAX_PATH_ALTERNATIVES = 64
    MAX_PATH_COMBINATIONS = 100_000

    def __init__(self, domains):
        self.domains = domains
        self.conditions = ProgressConditions(domains.arithmetic)
        self.endpoints = ProgressEndpoints(domains, self.conditions)
        self.ranks = ProgressRanks(domains, self.conditions)
        self.control = domains.control
        self.dispatch = domains.dispatch
        self._path_work = {}
        self.root = domains.memory.module.body.block.first_op
        self.events = tuple(
            n for n in domains.notifications if n.mechanism is not NotificationKind.PIPE_FENCE
        )
        self.parents = {self.root: None}
        self.children = {self.root: []}
        for loop in self.control.loops:
            chain = tuple(
                frame.loop.operation
                for frame in self.control.frames[loop].ancestors()
                if frame.loop is not None
            )
            parent = chain[-2] if len(chain) > 1 else self.root
            self.parents[loop] = parent
            self.children.setdefault(parent, []).append(loop)
            self.children.setdefault(loop, [])
        self.owners = {n: self._owner(n) for n in self.events}
        self.components = []

    def fail(self, region, message, witness=(), *, kind=ErrorKind.UNSUPPORTED):
        raise CompileError(
            message,
            kind=kind,
            phase="structured-event-progress",
            location=str(region.location),
            witness=witness,
        )

    def _owner(self, notification):
        bounds = notification.sources + notification.consumers
        paths = tuple(self.endpoints.ancestors(p.operation) for p in bounds)
        common = self.root
        for candidates in zip(*paths):
            if len(set(candidates)) != 1:
                break
            common = candidates[0]
        while common is not self.root:
            strict = self.endpoints.ancestors(common)[:-1]
            reference = dict(bounds[0].coordinates)
            assumptions = self.domains.tests[notification]
            if all(
                self.domains.arithmetic.prove(
                    Comparison(reference[loop], "==", dict(bound.coordinates)[loop]), assumptions
                )
                for loop in strict
                if loop is not self.root
                for bound in bounds[1:]
            ):
                break
            common = self.parents[common]
        return common

    def _ports(self, region):
        result = []
        for notification in self.events:
            if not self._is_ancestor(self.owners[notification], region):
                continue
            for phase, bounds in (("set", notification.sources), ("wait", notification.consumers)):
                for case in self.domains.cases[notification]:
                    if case.empty:
                        continue
                    for bound in bounds:
                        if self.endpoints.inside(bound, region):
                            port = regions.ProgressPort(notification, bound, phase, case)
                            if not self.conditions.impossible(self.endpoints.ambient(port)):
                                result.append(port)
        return tuple(result)

    def _is_ancestor(self, ancestor, region):
        while region is not None:
            if region is ancestor:
                return True
            region = self.parents[region]
        return False

    def _graph(self, region, ports, stream, arcs, interfaces):
        """Import child interfaces; their native witnesses remain opaque."""
        stream_nodes = () if stream is None else stream.nodes
        graph_ports = list(ports + stream_nodes)
        graph_arcs = list(arcs)
        if stream is not None:
            graph_arcs.extend(
                replace(
                    arc, source=arc.source + len(ports), destination=arc.destination + len(ports)
                )
                for arc in stream.relations
            )
        index = {port_key(port): i for i, port in enumerate(graph_ports)}

        def endpoint(port):
            key = port_key(port)
            if key not in index:
                index[key] = len(graph_ports)
                graph_ports.append(port)
            return index[key]

        for interface in interfaces:
            for relation in interface.relations:
                graph_arcs.append(
                    regions.GuardedTransfer(
                        endpoint(relation.source),
                        endpoint(relation.destination),
                        0,
                        relation.guard,
                        "actual child queue interface",
                        relation.exact,
                        relation.projections,
                    )
                )
        stream_keys = {port_key(node) for node in stream_nodes}
        finite_keys = {port_key(node) for node in ports}
        for wi, wait in enumerate(graph_ports):
            poll("progress interface queue relations", ports=len(graph_ports), row=wi)
            if not is_wait(wait):
                continue
            for si, item in enumerate(graph_ports):
                if not is_source(item) or not self.endpoints.route(wait, item):
                    continue
                if (
                    port_key(wait) in stream_keys
                    and port_key(item) in stream_keys
                    or port_key(wait) in finite_keys
                    and port_key(item) in finite_keys
                ):
                    continue
                invocation = self.endpoints.invocation_guard(region, wait, item)
                if invocation is None:
                    continue
                if region is not self.root and all(
                    self.endpoints.stream_offset(node, region) is not None for node in (wait, item)
                ):
                    steps, guard, exact = self.endpoints.gate_step(wait, item, region)
                    conditions = ((guard, exact),)
                else:
                    steps = 0
                    conditions = self.endpoints.order_conditions(wait, item)
                for guard, exact in conditions:
                    graph_arcs.append(
                        regions.GuardedTransfer(
                            wi,
                            si,
                            steps,
                            _guard(guard.comparisons + invocation.comparisons),
                            "actual region interface admission gate",
                            exact,
                        )
                    )
        # These cuts expose only the region's queues. They are sources/sinks
        # of blocking implications, never events, tasks or completion fences.
        coordinates = tuple(
            (loop, self.control.loops[loop].ordinal)
            for loop in self.endpoints.ancestors(region)[:-1]
            if loop is not self.root
        )
        # Open notifications retain their actual ports and ordering. Only
        # closed, owned publications and child cuts need queue abstraction.
        closed_nodes = tuple(
            node
            for node in graph_ports
            if isinstance(node, regions.QueueCut) or self.owners[node.notification] is region
        )
        resources = tuple(
            dict.fromkeys(
                (
                    node.actor,
                    node.engine
                    if is_source(node)
                    else None
                    if self.endpoints.controller(node)
                    else node.engine,
                    "input" if is_source(node) else "output",
                )
                for node in closed_nodes
            )
        )
        cuts = tuple(
            regions.QueueCut(region, actor, engine, phase, coordinates)
            for actor, engine, phase in resources
        )
        local_nodes = tuple(graph_ports)
        for cut in cuts:
            ci = endpoint(cut)
            for ni, node in enumerate(local_nodes):
                if (
                    isinstance(node, regions.ProgressPort)
                    and self.owners[node.notification] is not region
                ):
                    continue
                if cut.actor is not node.actor or (cut.phase == "input") != is_source(node):
                    continue
                engine = (
                    node.engine
                    if is_source(node)
                    else None
                    if self.endpoints.controller(node)
                    else node.engine
                )
                if cut.engine is not engine:
                    continue
                invocation = self.endpoints.invocation_guard(region, cut, node)
                if invocation is None:
                    continue
                binding = regions.InvocationBinding(
                    region, coordinates, getattr(node, "bindings", ()), invocation
                )
                graph_arcs.append(
                    regions.GuardedTransfer(
                        ci if cut.phase == "input" else ni,
                        ni if cut.phase == "input" else ci,
                        0,
                        _guard(self.endpoints.ambient(node) + invocation.comparisons),
                        "region queue admission input"
                        if cut.phase == "input"
                        else "region queue/controller blocking output",
                        False,
                        (binding,),
                    )
                )
        graph = regions.ChannelProjection(
            region,
            tuple(graph_ports),
            self._unique_arcs(graph_arcs),
            tuple(
                dict.fromkeys(
                    window
                    for port in graph_ports
                    if isinstance(port, regions.ProgressPort)
                    for window in port.case.projections
                )
            ),
        )
        return graph, cuts

    def _paths(self, graph):
        """Bounded ordered conjunctions, with OR across alternative routes."""
        ports, region = graph.ports, graph.operation
        paths = RegionConnectivity(
            len(ports), lambda alternatives, guard: self._add_path(region, alternatives, guard)
        )
        direct = {}
        self._path_work[region] = 0
        for arc in graph.relations:
            a, b = ports[arc.source], ports[arc.destination]
            guard, projections = arc.guard, arc.projections
            if (
                isinstance(a, regions.ProgressPort)
                and isinstance(b, regions.ProgressPort)
                and a.phase == "set"
                and b.phase == "wait"
                and a.notification is b.notification
                and a.case is b.case
                and region in self.domains._clocks(a.notification)
            ):
                guard = self.endpoints.window_guard(a, region)
                projections = a.case.projections
            exported = self.endpoints.export_guard(guard, region)
            witness = replace(
                arc,
                guard=exported,
                exact=arc.exact and exported == guard,
                projections=projections,
            )
            paths.add(arc.source, arc.destination, exported)
            direct.setdefault((arc.source, arc.destination), []).append(witness)
        return paths, direct

    def _add_path(self, region, alternatives, guard):
        poll("progress interface composition", combinations=self._path_work[region])
        self._path_work[region] += 1
        self.domains.work.progress_path_combinations += 1
        if self._path_work[region] > self.MAX_PATH_COMBINATIONS:
            self.fail(
                region,
                "ordered region-interface alternative composition exceeds its budget",
                kind=ErrorKind.ANALYSIS_LIMIT,
            )
        condition = self.conditions.intern(guard.comparisons)
        tests = condition.members
        if any(prior <= tests for prior in alternatives) or self.conditions.impossible(condition):
            return
        for prior in tuple(alternatives):
            if tests < prior:
                del alternatives[prior]
        # Keep the actual ordered witness, never sort predicates. A weaker
        # existing conjunction absorbs a stronger alternative by Boolean OR.
        alternatives[tests] = guard
        self.domains.work.progress_path_alternatives += 1
        if len(alternatives) > self.MAX_PATH_ALTERNATIVES:
            self.fail(
                region,
                "region-interface OR alternatives exceed their explicit budget",
                kind=ErrorKind.ANALYSIS_LIMIT,
            )

    def _boundary(self, graph, paths, direct, count):
        result = []
        for i, j, guard in paths.retained(range(count)):
            conditions = self.conditions.intern(guard.comparisons).members
            witnesses = tuple(
                arc
                for arc in direct.get((i, j), ())
                if self.conditions.intern(arc.guard.comparisons).members == conditions
            )
            if witnesses:
                result.extend(witnesses)
            else:
                result.append(
                    regions.GuardedTransfer(
                        i,
                        j,
                        0,
                        guard,
                        "projected actual region blocking/publication interface",
                        False,
                        (graph,),
                    )
                )
        return self._unique_arcs(result)

    def _interface(self, graph, paths, direct, ports, cuts):
        """Export only open notifications and this region's queue cuts."""
        index = {port_key(node): i for i, node in enumerate(graph.ports)}
        retained = tuple(index[port_key(node)] for node in ports + cuts)
        result = []
        for i, j, guard in paths.retained(retained):
            conditions = self.conditions.intern(guard.comparisons).members
            witness = next(
                (
                    arc
                    for arc in direct.get((i, j), ())
                    if arc.exact
                    and self.conditions.intern(arc.guard.comparisons).members == conditions
                ),
                None,
            )
            result.append(
                regions.InterfaceTransfer(
                    graph.ports[i],
                    graph.ports[j],
                    guard,
                    witness is not None,
                    witness.projections if witness is not None else (graph,),
                )
            )
        return regions.RegionInterface(graph.operation, ports, cuts, tuple(result))

    def _controller_entries(self, region, interfaces):
        """Closed cross-controller children need one actual call ordering.

        A child's progress certificate assumes its participating controllers
        can reach that invocation. Opposite entry orders can prevent that even
        when both child graphs progress in isolation. These are dispatch cuts,
        not child completion or memory-retirement claims.
        """
        collective = tuple(
            interface
            for interface in interfaces
            if any(
                channel.source.actor is not channel.destination.actor
                for channel in interface.relations
            )
        )
        if not collective:
            return ()
        actors = tuple(
            actor
            for actor in Actor
            if any(
                actor in {channel.source.actor, channel.destination.actor}
                for child in collective
                for channel in child.relations
            )
        )
        entries = []
        for actor in actors:
            calls = []
            for child in collective:
                scope = self.dispatch.scopes.get((child.operation, actor))
                if scope is None:
                    self.fail(
                        region,
                        "closed collective child lacks an actual participating-controller entry",
                    )
                calls.append((child.operation, scope.parent_stage, scope.parent_lexical))
            entries.append(
                regions.ControllerEntries(actor, tuple(sorted(calls, key=lambda call: call[2])))
            )
        reference = tuple((operation, stage) for operation, stage, _ in entries[0].calls)
        if any(
            tuple((operation, stage) for operation, stage, _ in entry.calls) != reference
            for entry in entries[1:]
        ):
            self.fail(
                region,
                "closed child controller dispatch order has no common invocation ordering certificate",
                tuple(
                    f"{entry.actor.value}: "
                    + ", ".join(
                        f"{self.control.loops[operation].ordinal.text()} stage={stage} lexical={lexical}"
                        for operation, stage, lexical in entry.calls
                    )
                    for entry in entries
                ),
            )
        return tuple(entries)

    def _local_arcs(self, region, ports, own, interfaces, *, stream):
        index = {port_key(port): i for i, port in enumerate(ports)}
        arcs = []
        for notification in own:
            sources = [
                i
                for i, port in enumerate(ports)
                if isinstance(port, regions.ProgressPort)
                and port.notification is notification
                and port.phase == "set"
            ]
            consumers = [
                i
                for i, port in enumerate(ports)
                if isinstance(port, regions.ProgressPort)
                and port.notification is notification
                and port.phase == "wait"
            ]
            for a in sources:
                for b in consumers:
                    if ports[a].case is not ports[b].case:
                        continue
                    invocation = self.endpoints.invocation_guard(region, ports[a], ports[b])
                    if invocation is None:
                        continue
                    guard = _guard(
                        self.endpoints.ambient(ports[a])
                        + self.endpoints.ambient(ports[b])
                        + invocation.comparisons
                    )
                    if not self.conditions.impossible(guard.comparisons):
                        arcs.append(
                            regions.GuardedTransfer(
                                a, b, 0, guard, "selected effect before its matching actual Wait"
                            )
                        )
        for interface in interfaces:
            for relation in interface.relations:
                a, b = relation.source, relation.destination
                if any(
                    isinstance(node, regions.ProgressPort) and node.notification not in own
                    for node in (a, b)
                ):
                    continue
                if port_key(a) in index and port_key(b) in index:
                    arcs.append(
                        regions.GuardedTransfer(
                            index[port_key(a)],
                            index[port_key(b)],
                            0,
                            relation.guard,
                            "actual child queue interface",
                            relation.exact,
                            relation.projections,
                        )
                    )
        for wi, wait in enumerate(ports):
            poll("progress local queue relations", ports=len(ports), row=wi)
            if not is_wait(wait):
                continue
            for si, item in enumerate(ports):
                if not is_source(item) or not self.endpoints.route(wait, item):
                    continue
                invocation = self.endpoints.invocation_guard(region, wait, item)
                if invocation is None:
                    continue
                if stream:
                    steps, guard, exact = self.endpoints.gate_step(wait, item, region)
                    conditions = ((guard, exact),)
                else:
                    steps = 0
                    conditions = self.endpoints.order_conditions(wait, item)
                for guard, exact in conditions:
                    guard = _guard(guard.comparisons + invocation.comparisons)
                    if not self.conditions.impossible(guard.comparisons):
                        arcs.append(
                            regions.GuardedTransfer(
                                wi,
                                si,
                                steps,
                                guard,
                                "actual controller admission gate"
                                if self.endpoints.controller(wait)
                                else f"actual {wait.engine.value} queue admission gate",
                                exact,
                            )
                        )
        return self._unique_arcs(arcs)

    @staticmethod
    def _unique_arcs(arcs):
        # Equal actual relations are one structural edge, not distinct paths.
        unique = {}
        for arc in arcs:
            unique.setdefault((arc.source, arc.destination, arc.steps, arc.guard), arc)
        return tuple(unique.values())

    def compose(self, region):
        poll("progress region composition", regions=len(self.components))
        children = tuple(self.compose(child) for child in self.children[region])
        own = tuple(n for n in self.events if self.owners[n] is region)
        streams = tuple(
            n for n in own if region is not self.root and region in self.domains._clocks(n)
        )
        interfaces = tuple(
            self.endpoints.import_interface(region, child.interface, streams) for child in children
        )
        controller_entries = self._controller_entries(region, interfaces)
        child_ports = tuple(port for interface in interfaces for port in interface.ports)
        ports, seen = [], set()
        own_ports = tuple(
            imported
            for port in self._ports(region)
            if (
                imported := self.endpoints.stream_port(port, region)
                if port.notification in streams
                else port
            )
            is not None
        )
        for port in own_ports + child_ports:
            if port_key(port) not in seen:
                ports.append(port)
                seen.add(port_key(port))
        stream_ports = tuple(port for port in ports if port.notification in streams)
        ports = tuple(port for port in ports if port.notification not in streams)
        if region is not self.root and any(
            region not in (coordinates := dict(self.endpoints.coordinates(port)))
            or not self.control.invariant_over(coordinates[region], region)
            for port in ports
        ):
            self.fail(
                region,
                "boundary port retains a varying current-loop coordinate without an owned stream interface",
            )
        stream_proof = None
        if streams:
            stream_cuts = tuple(cut for interface in interfaces for cut in interface.cuts)
            nodes = stream_ports + stream_cuts
            stream_arcs = self._local_arcs(region, nodes, streams, interfaces, stream=True)
            stride, rank, conditional = self.ranks.stream(region, nodes, stream_arcs)
            stream_proof = regions.StreamProgress(
                region, stream_ports, stream_cuts, stream_arcs, stride, rank, conditional
            )
            self.components.append((region, nodes, stream_arcs, rank, stride))
        own_finite = tuple(n for n in own if n not in streams)
        direct_arcs = self._local_arcs(region, ports, own_finite, (), stream=False)
        graph, cuts = self._graph(region, ports, stream_proof, direct_arcs, interfaces)
        paths, direct = self._paths(graph)
        own_cut_keys = {port_key(cut) for cut in cuts}
        paths.eliminate(
            (
                i
                for i in range(len(ports), len(graph.ports))
                if port_key(graph.ports[i]) not in own_cut_keys
            ),
        )
        arcs = self._boundary(graph, paths, direct, len(ports))
        excluded, conditional = self.ranks.finite(region, ports, arcs)
        open_ports = tuple(port for port in ports if self.owners[port.notification] is not region)
        open_keys = {port_key(port) for port in open_ports}
        paths.eliminate(i for i in range(len(ports)) if port_key(ports[i]) not in open_keys)
        interface = self._interface(graph, paths, direct, open_ports, cuts)
        proof = regions.RegionProgress(
            region,
            ports,
            arcs,
            interface,
            children,
            stream_proof,
            excluded,
            controller_entries,
            conditional,
        )
        self.components.append((region, ports, arcs, (), 1))
        return proof

    def run(self):
        root = self.compose(self.root)
        points, relations = [], []
        for component, (_, ports, arcs, _, _) in enumerate(self.components):
            base = len(points)
            points.extend(
                ProgressPoint(
                    port.operation if isinstance(port, regions.QueueCut) else port.bound.operation,
                    port.actor,
                    port.phase,
                    None if isinstance(port, regions.QueueCut) else port.notification,
                    component,
                    "queue-interface" if isinstance(port, regions.QueueCut) else "actual-region",
                )
                for port in ports
            )
            relations.extend(
                replace(arc, source=arc.source + base, destination=arc.destination + base)
                for arc in arcs
            )
        self.domains.work.progress_condition_sets = self.conditions.condition_count
        self.domains.work.progress_condition_queries = self.conditions.proof_queries
        self.domains.work.progress_condition_hits = self.conditions.cache_hits
        self.domains.work.progress_points = len(points)
        self.domains.work.progress_arcs = len(relations)
        return StructuredProgress(
            tuple(points),
            tuple(relations),
            tuple(operation for operation, _, _, _, _ in self.components),
            tuple(rank for _, _, _, rank, _ in self.components),
            tuple(stride for _, _, _, _, stride in self.components),
            root,
        )


def prove_progress(domains):
    """Prove owned region progress or diagnose the unproved interface relation."""
    return _Composer(domains).run()

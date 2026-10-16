"""Exact native child transfers with virtual query cuts and real insertion cuts.

Only physical prefix ports and actual synchronization insertions enter the
parent graph. An immutable native Site remains a query handle, not a child-done
node. Injection and extraction are distinct directed local reachability maps.
"""

from dataclasses import dataclass, field, replace

from .child_completion import ChildPort, FixedChildProjection, native_fixed_child
from .child_transfer_reach import LocalTransfer
from .completion import Position, Site
from .completion_flow import ALL
from .diagnostics import require
from .loop_graph import Command
from .scope_domains import Activation, affine
from .symbolic_completion import Family, le, meet


class FixedChildTransfer:
    """One owned actual native body, retaining exact constant inner positions."""

    def __init__(self, analysis, actor):
        self.view = native_fixed_child(analysis, insertion_cuts=True)
        self.count, self.actor = self.view.count, actor
        self.local_actor = next(iter(self.view.streams))
        require(len(self.view.streams) == 1, "scope child transfer needs one actual actor")
        self.reach = LocalTransfer(self.view.flow.topology)
        self.inputs = tuple(replace(port, actor=actor) for port in self.view.inputs)
        self.outputs = tuple(replace(port, actor=actor) for port in self.view.outputs)
        self.native_templates = sum(
            command.kind == "task" for sequence in self.view.streams.values() for command in sequence
        )
        self.commands = {
            ((self.actor, command.task.uid), command.channel,
             "" if command.kind == "task" else command.kind)
            for sequence in self.view.streams.values() for command in sequence
        }
        self.bindings = {}
        # This body topology is immutable. Exact negative local answers are
        # reusable too: adding another origin cannot add a native path.
        self.relations = {}
        self.relation_requests = self.relation_reuses = 0
        self.route_requests = self.route_reuses = self.route_boundary_checks = 0
        self.route_frontier_checks = 0

    def accepts(self, site):
        return (site.task, site.channel, site.action) in self.commands

    def binding(self, point):
        if isinstance(point, Site):
            require(isinstance(point.position, Position)
                    and type(point.position.end) is bool and type(point.position.offset) is int
                    and point.position.offset >= 0,
                    "invalid child transfer inner position")
        if point not in self.bindings:
            if isinstance(point, ChildPort):
                local = replace(point, actor=self.local_actor)
                require(local in self.view.port_bindings, "unknown child transfer prefix port")
                value = self.view.port_bindings[local]
            else:
                require(isinstance(point, Site) and self.accepts(point),
                        "unknown scoped child completion query endpoint")
                value = self.view.bind_site(replace(point, task=(self.local_actor, point.task[1])))
            self.bindings[point] = value
        return self.bindings[point]

    def covered(self, first, last):
        self.relation_requests += 1
        key = self.binding(first), self.binding(last)
        previous = self.relations.get(key)
        if previous is not None:
            self.relation_reuses += 1
            return previous
        result = self.relations[key] = self.reach.covered(*key)
        return result

    def projection(self):
        """Expose only physical ports; retain every other native cut virtually."""
        points = (*self.inputs, *self.outputs)
        relations = set()
        targets = tuple(self.binding(point) for point in points)
        if targets:
            self.reach.predecessors(targets, targets[0])
        for source in points:
            binding = self.binding(source)
            answers = self.reach.reachable(binding, targets)
            self.relations.update(((binding, target), answer)
                                  for target, answer in zip(targets, answers, strict=True))
            relations.update((source, target) for target, covered in zip(points, answers, strict=True)
                             if source != target and covered)
        return FixedChildProjection(
            self.count, self.inputs, self.outputs, (), frozenset(relations),
            len(self.view.flow.families), self.view.flow.edge_count, self.native_templates,
            len(self.view.graph.nodes), len(self.view.graph.edges), (),
        )

    def statistics(self):
        return {
            "native_templates": self.native_templates,
            "native_nodes": len(self.view.graph.nodes),
            "native_edges": len(self.view.graph.edges),
            "projection_nodes": len(self.view.flow.families),
            "projection_edges": self.view.flow.edge_count,
            "virtual_bindings": len(self.bindings),
            "local_relation_requests": self.relation_requests,
            "local_relation_reuses": self.relation_reuses,
            "local_relation_pairs": len(self.relations),
            "route_requests": self.route_requests,
            "route_reuses": self.route_reuses,
            "route_boundary_checks": self.route_boundary_checks,
            "route_frontier_checks": self.route_frontier_checks,
            **self.reach.statistics(),
        }


@dataclass(frozen=True)
class TransferCase:
    label: str | None
    activation: Activation | None
    transfer: FixedChildTransfer


@dataclass(frozen=True)
class ConditionalChildTransfer:
    count: int
    native_templates: int
    cases: tuple[TransferCase, ...]

    def statistics(self):
        return {
            "native_templates": self.native_templates,
            "parent_cases": {case.label: case.transfer.statistics() for case in self.cases},
        }


@dataclass(frozen=True)
class VirtualCut(Command):
    """A Command-compatible exact native cut, without a parent graph vertex."""

    uid: str = field(kw_only=True)
    site: Site = field(kw_only=True)
    case_label: str | None = field(kw_only=True)
    activation: Activation | None = field(kw_only=True)
    transfer: FixedChildTransfer = field(kw_only=True, repr=False)
    parent_period: int = field(kw_only=True)

    def resolve(self, residue):
        require(type(residue) is int and 0 <= residue < self.parent_period,
                "invalid virtual cut residue")
        family = Family(0, (1, -1), self.parent_period, residue)
        condition = ALL
        if self.activation is not None:
            active = self.activation
            condition = active.count if residue % active.period in active.residues else None
            first, last = affine(active.first), affine(active.last)
            if first == last:
                family = replace(family, last=first)
            else:
                family = replace(family, lower=max(family.lower, first[1]),
                                 horizon=(1, min(family.horizon[1], last[1])))
        return VirtualNode(self, residue, family, condition)


@dataclass(frozen=True)
class VirtualNode:
    cut: VirtualCut
    residue: int
    family: Family
    condition: object


@dataclass(frozen=True)
class _Boundary:
    point: object
    nodes: tuple[int, ...]
    revision: int


@dataclass
class _RouteMap:
    processed: int = 0
    frontier: list[_Boundary] = field(default_factory=list)
    results: dict = field(default_factory=dict)


def _include_frontier(transfer, frontier, added, *, earliest):
    """Keep a stable antichain under the exact native local preorder."""
    def precedes(first, last):
        transfer.route_frontier_checks += 1
        return transfer.covered(first.point, last.point)

    def dominates(first, last):
        return precedes(first, last) if earliest else precedes(last, first)

    if any(dominates(previous, added) for previous in frontier):
        return frontier
    return [previous for previous in frontier if not dominates(added, previous)] + [added]


def _nearest(transfer, records, *, earliest):
    frontier = []
    for record in records:
        frontier = _include_frontier(transfer, frontier, record, earliest=earliest)
    return frontier


class TransferComposition:
    """Append-only parent interface; old virtual source identities never change."""

    def __init__(self, completion, calls):
        self.completion, self.calls = completion, calls
        self.revision, self.virtual_number = 0, -1
        self.virtual, self.boundaries, self.promoted, self.route_cache = {}, {}, {}, {}
        self.activation_sink = None
        for uid, (call, value) in calls.items():
            cases = (value.cases if isinstance(value, ConditionalChildTransfer)
                     else (TransferCase(None, None, value),))
            for case in cases:
                records = self.boundaries[uid, case.label] = []
                for point in (*case.transfer.inputs, *case.transfer.outputs):
                    nodes = tuple(
                        completion.active_ports[uid, point, residue]
                        if case.label is None else completion.case_ports[uid, case.label, point, residue]
                        for residue in range(completion.graph.period)
                    )
                    records.append(_Boundary(point, nodes, 0))

    def cases(self, uid):
        value = self.calls[uid][1]
        return (value.cases if isinstance(value, ConditionalChildTransfer)
                else (TransferCase(None, None, value),))

    def commands_for(self, uid, site):
        require(uid in self.calls and isinstance(site, Site),
                "unknown scoped child completion query endpoint")
        require(site.region == uid and type(site.call) is int and site.call == 0,
                "scoped child completion changed its actual lexical call")
        result = []
        for case in self.cases(uid):
            if not case.transfer.accepts(site):
                continue
            case.transfer.binding(site)  # Validate actual inner existence before caching.
            key = uid, case.label, site
            if key not in self.virtual:
                call = self.calls[uid][0]
                # Import lazily to keep the completion adapter's types independent.
                from .scope_completion import _ClockSlot

                self.virtual[key] = VirtualCut(
                    self.virtual_number, site.task[0], _ClockSlot(uid, call.stage.data),
                    "scope_virtual", operation=call,
                    uid=uid, site=site, case_label=case.label, activation=case.activation,
                    transfer=case.transfer, parent_period=self.completion.graph.period,
                )
                self.virtual_number -= 1
            result.append(self.virtual[key])
        require(result, "unknown scoped child completion query endpoint")
        return tuple(result)

    def prepare(self, cuts):
        groups = {}
        for cut in cuts:
            if isinstance(cut, VirtualCut):
                groups.setdefault(cut.transfer, set()).add(cut.transfer.binding(cut.site))
        for transfer, sources in groups.items():
            ordered = tuple(sorted(sources))
            if ordered:
                transfer.reach.predecessors(ordered, ordered[0])

    def routes(self, node, *, source, since=None):
        require(isinstance(node, VirtualNode), "transfer routes need an exact virtual residue")
        cut = node.cut
        records = self.boundaries[cut.uid, cut.case_label]
        transfer = cut.transfer
        transfer.route_requests += 1
        # Equivalent names/parent residues share the same local mapping. The
        # case and call remain part of the key; only physical inner binding is
        # canonicalized. A map examines each new boundary at most once.
        key = cut.uid, cut.case_label, transfer.binding(cut.site), source
        mapping = self.route_cache.setdefault(key, _RouteMap())
        if mapping.processed == len(records):
            transfer.route_reuses += 1
        for record in records[mapping.processed :]:
            transfer.route_boundary_checks += 1
            first, last = (cut.site, record.point) if source else (record.point, cut.site)
            if transfer.covered(first, last):
                frontier = _include_frontier(transfer, mapping.frontier, record, earliest=source)
                if frontier is not mapping.frontier:
                    mapping.frontier = frontier
                    mapping.results.clear()
        mapping.processed = len(records)
        result_key = node.residue, since
        if result_key not in mapping.results:
            mapping.results[result_key] = tuple(sorted({
                record.nodes[node.residue] for record in mapping.frontier
                if since is None or record.revision > since
            }))
        return mapping.results[result_key]

    def promote(self, uid, site):
        """Expose only an actual insertion point, preserving all old native arcs."""
        graph, completion = self.completion.graph, self.completion
        for cut in self.commands_for(uid, site):
            key = uid, cut.case_label, site
            if key in self.promoted:
                continue
            records = self.boundaries[uid, cut.case_label]
            incoming, outgoing = [], []
            equal = None
            for record in records:
                before = cut.transfer.covered(record.point, site)
                after = cut.transfer.covered(site, record.point)
                if before and after:
                    equal = record
                    break
                if before:
                    incoming.append(record)
                if after:
                    outgoing.append(record)
            if equal is not None:
                nodes = equal.nodes
            else:
                number = 1 + max(node.command.index for node in graph.nodes)
                command = Command(number, cut.actor, cut.task, "scope_component", operation=cut.operation)
                nodes = []
                for residue in range(graph.period):
                    node = graph.node(command, residue, "component")
                    graph.ids[command.index, residue, "begin"] = node
                    if cut.activation is not None:
                        completion.node_activations[node] = cut.activation
                        if self.activation_sink is not None:
                            self.activation_sink[node] = cut.activation
                    nodes.append(node)
                nodes = tuple(nodes)
                # Keep only the nearest retained frontiers. Existing macro arcs
                # remain sound witnesses and need not be deleted on promotion.
                incoming = _nearest(cut.transfer, incoming, earliest=False)
                outgoing = _nearest(cut.transfer, outgoing, earliest=True)
                for residue in range(graph.period):
                    for record in incoming:
                        completion.edge((record.nodes[residue], 0), (nodes[residue], 0))
                    for record in outgoing:
                        completion.edge((nodes[residue], 0), (record.nodes[residue], 0))
                self.revision += 1
                records.append(_Boundary(site, nodes, self.revision))
            self.promoted[key] = nodes
            for residue, node in enumerate(nodes):
                if cut.case_label is None:
                    completion.sites[uid, site, residue] = node
                else:
                    values = completion.case_sites.get((uid, site, residue), ())
                    completion.case_sites[uid, site, residue] = (*values, node)
        return tuple(self.promoted[uid, cut.case_label, site] for cut in self.commands_for(uid, site))


def direct(first, last):
    """Native semantic reachability, including active physical identity."""
    return (
        isinstance(first, VirtualNode) and isinstance(last, VirtualNode)
        and first.cut.transfer is last.cut.transfer
        and first.cut.uid == last.cut.uid and first.cut.case_label == last.cut.case_label
        and first.residue == last.residue
        and first.cut.transfer.covered(first.cut.site, last.cut.site)
    )


def point_guard(node, point):
    family = node.family
    result = meet(node.condition, family.residue_guard(point))
    result = meet(result, meet(le((0, family.lower), point), le(point, family.horizon)))
    if family.last is not None:
        result = meet(result, meet(le(point, family.last), le(family.last, point)))
    return result

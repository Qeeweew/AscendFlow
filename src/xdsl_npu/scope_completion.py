"""Compose retained fixed children on their real parent pipeline clock.

Native completion comes only from each verified child projection. Parent
prefixes carry actor/pipe state through every physical call slot, including
skipped prologue/epilogue slots. Active child components have a separate guard;
their controller gates cannot survive a skipped call. Neither clock expands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from .budget import bounded, poll
from .child_completion import ChildPort, FixedChildProjection, ProjectionPoint
from .completion import Site
from .diagnostics import require
from .dialects.physical import CallOp, OnceOp, StreamOp
from .loop_graph import Command, Graph
from .periodic import MAX_PERIOD_TEMPLATES, bounded_period
from .scope_child_transfer import (
    ConditionalChildTransfer,
    TransferComposition,
    VirtualCut,
    VirtualNode,
    direct,
)

MAX_SCOPE_ARCS = 1_000_000
__all__ = ("ScopeCompletion", "VirtualCut", "VirtualNode", "compose_calls", "compose_transfers")


@dataclass(frozen=True)
class _ClockSlot:
    """Query/guard metadata, deliberately without a native completion contract."""

    uid: str
    stage: int


@dataclass(frozen=True)
class ScopeCompletion:
    graph: Graph
    sites: dict[tuple[str, Site, int], int]
    active_ports: dict[tuple[str, ChildPort, int], int]
    prefixes: dict[tuple[str, str, str | None, str, int], int]
    aliases: dict[tuple[str, ProjectionPoint], int]
    commands: dict[tuple[str, Site], Command]
    call_order: dict[str, tuple[str, ...]]
    max_stage: dict[str, int]
    projections: dict[str, FixedChildProjection]
    case_sites: dict = field(default_factory=dict)
    case_commands: dict = field(default_factory=dict)
    node_activations: dict = field(default_factory=dict)
    case_ports: dict = field(default_factory=dict)
    transfer_state: object | None = None

    @property
    def transfer_revision(self):
        return 0 if self.transfer_state is None else self.transfer_state.revision

    def inject(self, node, *, since=None):
        require(self.transfer_state is not None, "scope completion has no native transfer")
        return self.transfer_state.routes(node, source=True, since=since)

    def extract(self, node, *, since=None):
        require(self.transfer_state is not None, "scope completion has no native transfer")
        return self.transfer_state.routes(node, source=False, since=since)

    def direct(self, first, last):
        return direct(first, last)

    def prepare_virtual(self, cuts):
        if self.transfer_state is not None:
            self.transfer_state.prepare(cuts)

    def ats(self, uid, point, generation):
        if self.transfer_state is not None and isinstance(point, Site):
            self.transfer_state.promote(uid, point)
        key = uid, point, generation % self.graph.period
        if key in self.case_sites:
            return tuple((node, generation // self.graph.period) for node in self.case_sites[key])
        return (self.at(uid, point, generation),)

    def commands_for(self, uid, point):
        if self.transfer_state is not None:
            return self.transfer_state.commands_for(uid, point)
        key = uid, point
        if key in self.case_commands:
            return self.case_commands[key]
        require(key in self.commands, "unknown scoped child completion query endpoint")
        return (self.commands[key],)

    @property
    def ports(self):
        return self.active_ports

    def at(self, uid, point, generation):
        """Locate an exact child milestone on the parent generation clock."""
        if self.transfer_state is not None and isinstance(point, Site):
            self.transfer_state.promote(uid, point)
        table = self.sites if isinstance(point, Site) else self.active_ports
        key = uid, point, generation % self.graph.period
        require(key in table, "unknown scoped child completion endpoint")
        return table[key], generation // self.graph.period

    def prefix_at(self, uid, actor, pipe, kind, generation):
        key = uid, actor, pipe, kind, generation % self.graph.period
        require(key in self.prefixes, "unknown scoped physical prefix")
        return self.prefixes[key], generation // self.graph.period

    def query(self, source_uid, source, source_g, target_uid, target, target_g, *, minimum=0):
        """Adapt exact Site handles to Graph.covered_many's translation queries.

        These commands locate already-proved component nodes. They are never
        passed to native_order and grant no instruction begin/end relation.
        """
        if self.transfer_state is not None:
            first = self.commands_for(source_uid, source)
            last = self.commands_for(target_uid, target)
            require(len(first) == len(last) == 1, "conditional transfer query needs explicit cases")
            return first[0], source_g, last[0], target_g, minimum, False
        require(
            (source_uid, source) in self.commands and (target_uid, target) in self.commands,
            "unknown scoped child completion query endpoint",
        )
        return (
            self.commands[source_uid, source],
            source_g,
            self.commands[target_uid, target],
            target_g,
            minimum,
            False,
        )

    def edge(self, source, target):
        """Append a proved protocol arc and invalidate structural query caches."""
        before = len(self.graph.edges)
        self.graph.edge(source, target)
        if len(self.graph.edges) != before:
            self.graph.potentials = None
            self.graph.domains.clear()

    def statistics(self):
        return {
            "calls": len(self.projections),
            "actors": len(self.call_order),
            "period": self.graph.period,
            "node_families": len(self.graph.nodes),
            "arc_families": len(self.graph.edges),
            "physical_prefixes": len(self.prefixes),
            "active_components": len(
                {
                    node.command.index
                    for node in self.graph.nodes
                    if node.command.kind == "scope_component"
                }
            ),
            "native_templates": sum(p.native_templates for p in self.projections.values()),
        }


def _transitive_reduction(component_count, arcs):
    """Reduce a local DAG before attaching its uniform active-call guard.

    Visit each node's successors in topological order. A successor already
    covered by a prior successor contributes no new reachability. Integer
    bitsets compute the descendant sets once in reverse topological order;
    no graph edge is removed and re-queried during analysis.
    """
    following = [[] for _ in range(component_count)]
    preceding = [[] for _ in range(component_count)]
    indegree = [0] * component_count
    for i, (source, target) in enumerate(arcs):
        if i % 4096 == 0:
            poll("child projection DAG", relations=i, components=component_count)
        following[source].append(target)
        preceding[target].append(source)
        indegree[target] += 1
    pending = deque(node for node, degree in enumerate(indegree) if degree == 0)
    order, visited = [], 0
    while pending:
        source = pending.popleft()
        if len(order) % 4096 == 0:
            poll("child projection topological order", components=len(order))
        order.append(source)
        for target in following[source]:
            if visited % 4096 == 0:
                poll("child projection topological order", relations=visited)
            visited += 1
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    require(len(order) == component_count, "child projection components are not a DAG")
    # Iterating targets by rank orders all successor lists in O(C + R).
    successors = [[] for _ in range(component_count)]
    visited = 0
    for rank, target in enumerate(order):
        if rank % 4096 == 0:
            poll("child projection successor order", components=rank)
        for source in preceding[target]:
            if visited % 4096 == 0:
                poll("child projection successor order", relations=visited)
            visited += 1
            successors[source].append(target)
    descendants, reduced = [0] * component_count, []
    visited = 0
    for rank, source in enumerate(reversed(order)):
        if rank % 4096 == 0:
            poll("child projection transitive reduction", components=rank)
        covered = 0
        for target in successors[source]:
            if visited % 4096 == 0:
                poll("child projection transitive reduction", relations=visited)
            visited += 1
            bit = 1 << target
            if covered & bit:
                continue
            reduced.append((source, target))
            covered |= bit | descendants[target]
        descendants[source] = covered
    return tuple(reduced)


def _components(projection):
    """Collapse only the child's certified completion preorder, never parent arcs."""
    points = (*projection.inputs, *projection.endpoints, *projection.outputs)
    require(len(set(points)) == len(points), "duplicate child projection point")
    index = {point: i for i, point in enumerate(points)}
    following, preceding = [set() for _ in points], [set() for _ in points]
    for i, (source, target) in enumerate(projection.relations):
        if i % 4096 == 0:
            poll("child projection components", relations=i, points=len(points))
        require(
            source in index and target in index, "child projection relation has an unknown point"
        )
        a, b = index[source], index[target]
        following[a].add(b)
        preceding[b].add(a)
    following = [tuple(sorted(items)) for items in following]
    seen, finished = set(), []
    for start in range(len(points)):
        if start in seen:
            continue
        seen.add(start)
        pending = [(start, iter(following[start]))]
        while pending:
            node, edges = pending[-1]
            for target in edges:
                if target not in seen:
                    seen.add(target)
                    pending.append((target, iter(following[target])))
                    break
            else:
                finished.append(node)
                pending.pop()
    assigned, groups = {}, []
    for start in reversed(finished):
        if start in assigned:
            continue
        component = len(groups)
        members, pending = [], [start]
        assigned[start] = component
        while pending:
            node = pending.pop()
            members.append(node)
            for source in sorted(preceding[node], reverse=True):
                if source not in assigned:
                    assigned[source] = component
                    pending.append(source)
        groups.append(tuple(points[node] for node in sorted(members)))
    aliases = {point: assigned[i] for i, point in enumerate(points)}
    arcs = {
        (aliases[source], aliases[target])
        for source, target in projection.relations
        if aliases[source] != aliases[target]
    }
    return tuple(groups), aliases, _transitive_reduction(len(groups), tuple(sorted(arcs)))


@bounded
def compose_calls(
    calls: dict[str, tuple[CallOp, FixedChildProjection]], *, period: int, count: int | None
) -> ScopeCompletion:
    """Read actual CallOp slots and compose fixed children without expanding N/C.

    Each actor's calls share their actual parent block and generation operand.
    Stage chooses the parent generation; issue_order and lexical position choose
    dispatch order. logical_phase supplies no completion or dispatch relation.
    All child Site kinds, including insertion cuts, remain opaque exact points.
    """
    if any(not isinstance(projection, FixedChildProjection) for _, projection in calls.values()):
        from .scope_case_completion import compose_conditional_calls

        return compose_conditional_calls(calls, period=period, count=count)
    require(type(period) is int, "scope completion needs an integer protocol period")
    bounded_period(period, "scope completion")
    require(count is None or type(count) is int and count >= 0, "invalid parent scope count")
    require(calls, "scope completion needs an actual child call")
    orders, projections, partitions = {}, {}, {}
    for uid, (op, projection) in calls.items():
        require(
            isinstance(op, CallOp)
            and isinstance(projection, FixedChildProjection)
            and isinstance(uid, str)
            and uid
            and op.uid.data == uid,
            "scope completion needs the actual uniquely named CallOp and child projection",
        )
        require(
            0 <= op.stage.data <= 32
            and 0 <= op.logical_phase.data <= 255
            and 0 <= op.issue_order.data < 1 << 31,
            "invalid parent call stage/logical phase",
        )
        op.verify()
        children = [child for child in op.body.block.ops if isinstance(child, (StreamOp, OnceOp))]
        require(
            len(children) == 1
            and children[0].limit is None
            and children[0].count is not None
            and children[0].count.data == projection.count,
            "parent call must retain its actual fixed-count child",
        )
        require(op.parent is not None, "parent calls need their actual lexical block")
        require(
            all(port.side == "input" for port in projection.inputs)
            and {replace(port, side="input") for port in projection.outputs}
            == set(projection.inputs),
            "child projection has unmatched native prefix ports",
        )
        actors = {port.actor for port in projection.inputs}
        require(actors, "child projection has no actual actor ports")
        for actor in actors:
            orders.setdefault(actor, []).append(uid)
        projections[uid] = projection
        partitions[uid] = _components(projection)
    maxima = {}
    for actor, members in orders.items():
        block = calls[members[0]][0].parent
        generation = calls[members[0]][0].generation
        require(
            all(
                calls[uid][0].parent is block and calls[uid][0].generation is generation
                for uid in members
            ),
            "one actor's parent calls must share their actual lexical block and generation",
        )
        lexical = {op: i for i, op in enumerate(block.ops)}
        members.sort(key=lambda uid: (calls[uid][0].issue_order.data, lexical[calls[uid][0]]))
        maxima[actor] = max(calls[uid][0].stage.data for uid in members)
    size = sum(
        len(groups) + len(projections[uid].inputs) for uid, (groups, _, _) in partitions.items()
    )
    require(
        period * size <= MAX_PERIOD_TEMPLATES,
        "parent scope completion node budget exceeded; no clock expansion fallback",
    )
    require(
        period
        * sum(
            len(arcs) + 4 * len(projections[uid].inputs) for uid, (_, _, arcs) in partitions.items()
        )
        <= MAX_SCOPE_ARCS,
        "parent scope completion arc budget exceeded",
    )
    graph = Graph(period, count)
    sites, ports, prefixes, aliases, commands = {}, {}, {}, {}, {}
    active = {}
    number = 0
    for uid, (op, projection) in calls.items():
        poll("parent child completion nodes", call=uid, nodes=len(graph.nodes))
        groups, mapping, arcs = partitions[uid]
        aliases.update(((uid, point), component) for point, component in mapping.items())
        for component, members in enumerate(groups):
            actors = {
                point.actor if isinstance(point, ChildPort) else point.task[0] for point in members
            }
            command = Command(
                number,
                next(iter(actors)) if len(actors) == 1 else "",
                _ClockSlot(uid, op.stage.data),
                "scope_component",
                operation=op,
            )
            number += 1
            for residue in range(period):
                node = graph.node(command, residue, "component")
                # Query handles locate this exact component, not native begins.
                graph.ids[command.index, residue, "begin"] = node
                active[uid, component, residue] = node
                for point in members:
                    if isinstance(point, Site):
                        sites[uid, point, residue] = node
                        commands[uid, point] = command
                    else:
                        ports[uid, point, residue] = node
        for residue in range(period):
            for source, target in arcs:
                graph.edge((active[uid, source, residue], 0), (active[uid, target, residue], 0))
        for port in projection.inputs:
            command = Command(
                number, port.actor, _ClockSlot(uid, op.stage.data), "scope_prefix", operation=op
            )
            number += 1
            guard = -op.stage.data, op.stage.data - maxima[port.actor]
            for residue in range(period):
                node = graph.node(command, residue, port.kind, guard=guard)
                prefixes[uid, port.actor, port.pipe, port.kind, residue] = node
                graph.edge((node, 0), (ports[uid, port, residue], 0))

    # Each key is a distinct physical state; no actor or pipe shares a prefix.
    for actor, members in orders.items():
        keys = sorted(
            {
                (port.pipe, port.kind)
                for uid in members
                for port in projections[uid].inputs
                if port.actor == actor
            },
            key=repr,
        )
        for pipe, kind in keys:
            port = ChildPort(actor, pipe, kind, "input")
            output = replace(port, side="output")
            participating = [uid for uid in members if port in projections[uid].inputs]
            timeline = []
            for step in range(period):
                for uid in participating:
                    generation = step - calls[uid][0].stage.data
                    residue, cycle = generation % period, generation // period
                    timeline.append(
                        (
                            (prefixes[uid, actor, pipe, kind, residue], cycle),
                            (ports[uid, output, residue], cycle),
                        )
                    )
            for i, (before, after) in enumerate(timeline):
                following = timeline[(i + 1) % len(timeline)][0]
                if i + 1 == len(timeline):
                    following = following[0], following[1] + 1
                graph.edge(before, following)  # Transparent even if this call is skipped.
                graph.edge(after, following)  # The child-specific contribution is active only.
    return ScopeCompletion(
        graph,
        sites,
        ports,
        prefixes,
        aliases,
        commands,
        {actor: tuple(members) for actor, members in orders.items()},
        maxima,
        projections,
    )


def compose_transfers(calls, *, period, count):
    """Compose physical child interfaces, retaining native query points virtually."""
    from .scope_case_completion import ConditionalProjection, ProjectionCase

    projected = {}
    for uid, (call, transfer) in calls.items():
        if isinstance(transfer, ConditionalChildTransfer):
            value = ConditionalProjection(
                transfer.count, transfer.native_templates,
                tuple(ProjectionCase(case.label, case.activation, case.transfer.projection())
                      for case in transfer.cases),
            )
        else:
            value = transfer.projection()
        projected[uid] = call, value
    completion = compose_calls(projected, period=period, count=count)
    state = TransferComposition(completion, calls)
    from dataclasses import replace

    completion = replace(completion, projections={uid: value for uid, (_, value) in calls.items()},
                         transfer_state=state)
    state.completion = completion
    completion.graph.child_transfer = completion
    return completion

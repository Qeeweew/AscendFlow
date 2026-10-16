"""Literal fixed-native bodies for the small serial-query test oracle.

This model never asks LocalTransfer for reachability, injection or extraction.
It expands the owned native topology's integer positions and follows ordinary
arcs. Parent component macros are replaced by these bodies. An independent
local closure check rejects incorrectly merged parent interface components.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from xdsl_npu.completion import Site
from xdsl_npu.model import CompileError

MILESTONES = (
    "begin",
    "end",
    "prefix",
    "issue",
    "queue",
    "retire",
    "implicit",
    "admit_issue",
    "admit_queue",
    "after_queue",
)


def active(activation, count, generation):
    return activation is None or (
        activation.count.contains(count)
        and activation.first.evaluate(count) <= generation <= activation.last.evaluate(count)
        and generation % activation.period in activation.residues
    )


def input_ports(value):
    """Read the declared physical interface for both Graph and transfer fixtures."""
    if hasattr(value, "inputs"):
        return set(value.inputs)
    return {
        port
        for case in value.cases
        for port in getattr(case, "transfer", getattr(case, "projection", None)).inputs
    }


@dataclass(frozen=True)
class BodyCase:
    uid: str
    label: str | None
    activation: object
    transfer: object


def body_cases(completion):
    state = completion.transfer_state
    if state is None:
        return ()
    result = []
    for uid, (_, value) in state.calls.items():
        if hasattr(value, "cases"):
            result.extend(
                BodyCase(uid, case.label, case.activation, case.transfer) for case in value.cases
            )
        else:
            result.append(BodyCase(uid, None, None, value))
    return tuple(result)


class FixedBody:
    """Ordinary finite directed graph, with no symbolic propagation shortcut."""

    def __init__(self, view):
        assert type(view.count) is int and 0 <= view.count <= 64, (
            "finite oracle needs a small child"
        )
        assert len(view.streams) == 1
        self.view = view
        self.local_actor = next(iter(view.streams))
        self.nodes, self.following, self.closures = set(), defaultdict(set), {}
        topology = view.flow.topology
        generations = {}
        for index, family in enumerate(topology.families):
            assert family.horizon[0] == 0 and (family.last is None or family.last[0] == 0)
            generations[index] = tuple(
                generation
                for generation in range(family.lower, family.horizon[1] + 1)
                if generation % family.period == family.residue
                and (family.last is None or generation == family.last[1])
            )
            self.nodes.update((index, generation) for generation in generations[index])
        for source, edges in enumerate(topology.edges):
            for target, shift, source_point, target_point, guard in edges:
                if not guard.contains(view.count):
                    continue
                if source_point is None:
                    pairs = (
                        ((source, generation), (target, generation + shift))
                        for generation in generations[source]
                    )
                else:
                    assert source_point[0] == target_point[0] == 0
                    pairs = (((source, source_point[1]), (target, target_point[1])),)
                for first, last in pairs:
                    if first in self.nodes and last in self.nodes:
                        self.following[first].add(last)

    def binding(self, site):
        binding = self.view.bind_site(replace(site, task=(self.local_actor, site.task[1])))
        node, (coefficient, generation) = binding
        assert coefficient == 0
        return node, generation

    def reaches(self, source, target):
        if source not in self.nodes or target not in self.nodes:
            return False
        if source not in self.closures:
            seen, pending = {source}, [source]
            while pending:
                for last in self.following[pending.pop()]:
                    if last not in seen:
                        seen.add(last)
                        pending.append(last)
            self.closures[source] = seen
        return target in self.closures[source]


def native_sites(parent):
    """Enumerate actual typed endpoints even when no command table was populated."""
    completion = parent.completion
    known = set(completion.commands) | set(completion.case_commands)
    cases = body_cases(completion)
    if not cases:
        return sorted(known, key=repr)
    requested = set(parent.access.endpoints.values()) | known
    for uid, wait, send, _ in parent.resource_pairs:
        requested.update(((uid, wait), (uid, send)))
    state = completion.transfer_state
    requested.update((uid, site) for uid, _, site in state.virtual)
    requested.update((uid, site) for uid, _, site in state.promoted)
    views = defaultdict(list)
    for case in cases:
        view = case.transfer.view
        views[case.uid].append((view, case.transfer.actor))
    result = set()
    for uid, site in requested:
        assert isinstance(site, Site) and site.region == uid and site.call == 0
        for point in MILESTONES:
            candidate = replace(site, point=point)
            for view, actor in views[uid]:
                if candidate.task[0] != actor:
                    continue
                local_actor = next(iter(view.streams))
                try:
                    view.bind_site(replace(candidate, task=(local_actor, candidate.task[1])))
                except CompileError:
                    continue
                result.add((uid, candidate))
                break
    assert result and any(not site.action for _, site in result), (
        "finite oracle lost actual natives"
    )
    return sorted(result, key=repr)


class ExpandedNativeBodies:
    """Attach literal child instances to existing parent prefix/event vertices."""

    def __init__(self, parent, count, copies):
        self.parent, self.count, self.copies = parent, count, copies
        self.cases = body_cases(parent.completion)
        self.bodies = {}
        for case in self.cases:
            if id(case.transfer.view) not in self.bodies:
                self.bodies[id(case.transfer.view)] = FixedBody(case.transfer.view)
        self.nodes = set()

    def native_macro(self, first, last):
        return bool(
            self.cases
            and first.command.kind == last.command.kind == "scope_component"
            and first.command.task.uid == last.command.task.uid
            and first.command.task.uid in self.parent.completion.transfer_state.calls
        )

    @staticmethod
    def point(query, uid, label, local, generation):
        return query, ("native", uid, label, *local), generation

    def attach(self, finite):
        completion = self.parent.completion
        if not self.cases:
            return
        # Conditional aggregate ports are transparent incoming state, including
        # a generation with no active body case. This carries no native end.
        for uid, (_, value) in completion.transfer_state.calls.items():
            if not hasattr(value, "cases"):
                continue
            for query in range(self.copies):
                for generation in range(self.count):
                    residue = generation % self.parent.period
                    for port in input_ports(value):
                        first = completion.active_ports[uid, port, residue]
                        last = completion.active_ports[uid, replace(port, side="output"), residue]
                        finite.edge((query, first, generation), (query, last, generation))
        for case in self.cases:
            body = self.bodies[id(case.transfer.view)]
            for query in range(self.copies):
                for generation in range(self.count):
                    if not active(case.activation, self.count, generation):
                        continue
                    residue = generation % self.parent.period

                    def point(local, *, query=query, case=case, generation=generation):
                        return self.point(query, case.uid, case.label, local, generation)

                    expanded = {point(local) for local in body.nodes}
                    self.nodes.update(expanded)
                    finite.nodes.update(expanded)
                    for source, targets in body.following.items():
                        for target in targets:
                            finite.edge(point(source), point(target))
                    aliases = defaultdict(list)
                    for local_port, (node, (_, inner)) in body.view.port_bindings.items():
                        port = replace(local_port, actor=case.transfer.actor)
                        local = node, inner
                        if case.label is None:
                            parent_node = completion.active_ports[case.uid, port, residue]
                        else:
                            parent_node = completion.case_ports[case.uid, case.label, port, residue]
                            aggregate = completion.active_ports[case.uid, port, residue]
                            first, last = (
                                (aggregate, parent_node)
                                if port.side == "input"
                                else (parent_node, aggregate)
                            )
                            finite.edge((query, first, generation), (query, last, generation))
                        bound = query, parent_node, generation
                        assert bound in finite.nodes
                        finite.edge(bound, point(local))
                        finite.edge(point(local), bound)
                        aliases[parent_node].append(local)
                    for (
                        uid,
                        label,
                        site,
                    ), parent_nodes in completion.transfer_state.promoted.items():
                        if (uid, label) != (case.uid, case.label):
                            continue
                        local = body.binding(site)
                        parent_node = parent_nodes[residue]
                        bound = query, parent_node, generation
                        assert bound in finite.nodes and local in body.nodes
                        finite.edge(bound, point(local))
                        finite.edge(point(local), bound)
                        aliases[parent_node].append(local)
                    # The parent may merge genuine SCC-equivalent interface
                    # points. Check that independently before sharing a node.
                    for bindings in aliases.values():
                        if len(bindings) > 1:
                            first = bindings[0]
                            assert all(
                                body.reaches(first, last) and body.reaches(last, first)
                                for last in bindings[1:]
                            ), "parent merged distinct native points"

    def endpoint(self, endpoint):
        generation = endpoint.parent.evaluate(self.count)
        if not (
            0 <= generation < self.count
            and endpoint.site.region == endpoint.call
            and endpoint.site.call == 0
        ):
            return ()
        result = []
        for case in self.cases:
            if (
                case.uid != endpoint.call
                or endpoint.site.task[0] != case.transfer.actor
                or not active(case.activation, self.count, generation)
            ):
                continue
            body = self.bodies[id(case.transfer.view)]
            try:
                local = body.binding(endpoint.site)
            except CompileError:
                continue
            point = self.point(endpoint.query, case.uid, case.label, local, generation)
            if point in self.nodes:
                result.append(point)
        return tuple(result)

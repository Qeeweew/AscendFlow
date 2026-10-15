"""Completion propagation with direct insertion of selected native protocols.

The initial graph contains only native instructions and their insertion cuts.
An obligation chooses its own capture/acquire; no graph of possible protocols
and no protocol-route search exists. Selected instructions extend the finite
translation domains in place. Final checking reconstructs ordinary command
streams, including independent event-lifetime and progress checks.
"""

from collections import deque
from dataclasses import dataclass

from .budget import poll
from .loop_graph import Command, Graph
from .model import require
from .native_order import instruction_points, prefix_inputs
from .periodic import MAX_PROOF_STATES


@dataclass(frozen=True)
class Protocol:
    name: str
    kind: str
    sources: tuple
    destinations: tuple
    distance: int
    period: int
    residues: tuple
    last: bool = False


class _DomainFlow:
    """Incremental multi-source reachability on an actual translation domain."""

    def __init__(self, graph, guard):
        self.graph, self.guard = graph, guard
        self.index, self.cycles, self.origins = {}, {}, {}
        self.facts, self.delta, self.successors = [], [], []
        self.pending = deque()
        self.transfers = 0
        for node in range(len(graph.nodes)):
            self.node(node)
        for arc in graph.edges:
            self.edge(*arc)

    def node(self, node):
        lower, upper = self.graph.bounds(node, *self.guard)
        self.cycles[node] = range(lower, upper + 1)
        require(
            self.graph.domain_states + max(0, upper - lower + 1) <= MAX_PROOF_STATES,
            "direct synthesis completion-domain budget exceeded",
        )
        for cycle in self.cycles[node]:
            self.index[node, cycle] = len(self.facts)
            self.facts.append(0)
            self.delta.append(0)
            self.successors.append(set())
            self.graph.domain_states += 1

    def seed(self, node, bits):
        bits &= ~self.facts[node]
        if bits:
            self.facts[node] |= bits
            if not self.delta[node]:
                self.pending.append(node)
            self.delta[node] |= bits

    def edge(self, source, target, distance):
        for cycle in self.cycles[source]:
            destination = self.index.get((target, cycle + distance))
            if destination is not None:
                origin = self.index[source, cycle]
                if destination not in self.successors[origin]:
                    self.successors[origin].add(destination)
                    self.seed(destination, self.facts[origin])

    def query(self, source, target):
        source, target = self.index.get(source), self.index.get(target)
        require(source is not None and target is not None, "missing direct synthesis endpoint")
        if source not in self.origins:
            self.origins[source] = 1 << len(self.origins)
            self.seed(source, self.origins[source])
        return self, target, self.origins[source]

    def propagate(self):
        while self.pending:
            source = self.pending.popleft()
            bits, self.delta[source] = self.delta[source], 0
            for target in self.successors[source]:
                self.seed(target, bits)
                self.transfers += 1
                if self.transfers % 4096 == 0:
                    poll("direct synchronization facts", transfers=self.transfers)


def _key(query):
    source, sg, destination, dg, *rest = query
    return source.index, sg, destination.index, dg, tuple(rest)


class DirectCompletion(Graph):
    def __init__(self, period, maximum):
        super().__init__(period, maximum)
        self.cuts, self.next_queue = {}, {}
        self.flows, self.queries = {}, {}
        self.selected = []
        self.endpoints = {}
        self.command_count = 0

    def node(self, *args, **kwargs):
        previous = len(self.nodes)
        index = super().node(*args, **kwargs)
        if len(self.nodes) != previous:
            for flow in self.flows.values():
                flow.node(index)
        return index

    def edge(self, source, target):
        previous = len(self.edges)
        super().edge(source, target)
        if len(self.edges) != previous:
            for flow in self.flows.values():
                flow.edge(*self.edges[-1])

    def admit(self, command, residue, milestones):
        """Fixed native insertion cuts, independent of any possible protocol."""
        require(command.kind == "task", "direct synthesis expects unsynchronized native work")
        self.command_count = max(self.command_count, command.index + 1)
        before = dict(milestones)
        for point in ("issue", "queue"):
            incoming, cycle = milestones[point]
            gate = self.node(command, residue, "admit_" + point, guard=self.nodes[incoming].guard)
            before[point] = gate, cycle
            self.edge(milestones[point], before[point])
        self.cuts[command.index, residue] = {
            "entry": dict(milestones),
            "admit": before,
            "waits": dict(milestones),
        }
        return before

    def link(self, previous_global, previous_pipe, current):
        """Forward selected admission effects without treating native end as a gate."""
        for previous, point in ((previous_global, "issue"), (previous_pipe, "queue")):
            command, milestones, *_ = previous
            node, cycle = milestones[point]
            key = command.index, self.nodes[node].residue
            admission = self.cuts[key]["admit"][point][0]
            target = current[1][point]
            self.edge((admission, cycle), target)
            if point == "queue":
                self.next_queue[key] = target[0], target[1] - cycle

    def register(self, queries):
        for query in queries:
            key = _key(query)
            if key in self.queries:
                continue
            normalized = self._query(*query)
            if normalized is None:
                self.queries[key] = None
                continue
            guard, source, target = normalized
            if guard not in self.flows:
                self.flows[guard] = _DomainFlow(self, guard)
            self.queries[key] = self.flows[guard].query(source, target)

    def covered(self, query):
        item = self.queries[_key(query)]
        if item is None:
            return True
        flow, target, bit = item
        flow.propagate()
        return bool(flow.facts[target] & bit)

    def _forward(self, command, residue, cut, kind, before, added):
        """A skipped sync retains its input prefix for later instructions."""
        entrance, cycle = cut["entry"][kind]
        after = (
            self.node(command, residue, "after_" + kind, guard=self.nodes[entrance].guard),
            cycle,
        )
        self.edge(before, after)
        self.edge(added, after)
        return after

    def insert(self, protocol, native):
        """Insert exactly this required capture/acquire, never an alternate route."""
        poll("direct protocol insertion", selected=len(self.selected), nodes=len(self.nodes))
        require(protocol not in self.selected, "duplicate direct protocol")
        self.selected.append(protocol)
        points = {}
        placements = (
            [(owner, "barrier") for owner in protocol.destinations]
            if protocol.kind == "barrier"
            else [(owner, "set") for owner in protocol.sources]
            + [(owner, "wait") for owner in protocol.destinations]
        )
        for owner, action in placements:
            task = native[owner]
            kind = "cross_" + action if protocol.kind == "cross" else action
            residues = (
                protocol.residues
                if action != "wait"
                else tuple(
                    sorted((r + protocol.distance) % protocol.period for r in protocol.residues)
                )
            )
            lower = protocol.distance if action != "set" else 0
            upper = protocol.distance if action == "set" else 0
            last_gap = upper if protocol.last else 0 if task.task.last else None
            command = Command(
                self.command_count,
                owner[0],
                task.task,
                kind,
                protocol.name,
                lower,
                upper,
                protocol.period,
                residues,
                last_gap=last_gap,
            )
            self.command_count += 1
            self.endpoints[owner[0], protocol.name, action] = command
            for residue in range(self.period):
                if residue % command.period not in command.residues:
                    continue
                cut = self.cuts[task.index, residue]
                cycle = cut["entry"]["issue"][1]
                if action == "set":
                    before = dict(cut["admit"])
                    before["queue"] = cut.get(
                        "last_set", (self.ids[task.index, residue, "begin"], cycle)
                    )
                    before["retire"] = self.ids[task.index, residue, "prefix"], cycle
                    if task.task.pipe == "S":
                        before["issue"] = self.ids[task.index, residue, "end"], cycle
                else:
                    before = cut["waits"]
                actual = instruction_points(
                    command,
                    before,
                    lambda point, c=command, r=residue, k=cycle: (self.node(c, r, point), k),
                    self.edge,
                )
                points[owner[0], action, residue] = actual
                if action == "set":
                    cut["last_set"] = self._forward(
                        command, residue, cut, "queue", before["queue"], actual["begin"]
                    )
                    node, shift = self.next_queue[task.index, residue]
                    self.edge(cut["last_set"], (node, cycle + shift))
                else:
                    for name, value in prefix_inputs(command, actual).items():
                        after = self._forward(command, residue, cut, name, before[name], value)
                        cut["waits"][name] = after
                        self.edge(after, cut["admit"][name])
        if protocol.kind == "barrier":
            return
        active = [r for r in range(self.period) if r % protocol.period in protocol.residues]
        for i, residue in enumerate(active):
            for actor, _ in protocol.sources:
                source = points[actor, "set", residue]["end"][0]
                if protocol.kind == "cross":
                    prior = points[actor, "set", active[i - 1]]["end"][0]
                    self.edge((prior, -int(i == 0)), (source, 0))
                for destination, _ in protocol.destinations:
                    generation = residue + protocol.distance
                    target = points[destination, "wait", generation % self.period]["begin"][0]
                    self.edge((source, 0), (target, generation // self.period))

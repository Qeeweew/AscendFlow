"""Completion propagation with direct insertion of selected native protocols.

The initial graph contains only native instructions and their insertion cuts.
An obligation chooses its own capture/acquire; no graph of possible protocols
and no protocol-route search exists. Selected instructions extend the finite
translation domains in place. Final checking reconstructs ordinary command
streams, including independent event-lifetime and progress checks.
"""

from dataclasses import dataclass

from .budget import poll
from .loop_graph import Command, Graph
from .model import require
from .native_order import instruction_points, prefix_inputs


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


class DirectCompletion(Graph):
    def __init__(self, period, maximum):
        super().__init__(period, maximum)
        self.cuts, self.next_queue = {}, {}
        self.selected = []
        self.endpoints = {}
        self.command_count = 0

    def admit(self, command, residue, milestones):
        """Forward-only insertion cuts before an actual native command.

        Synthesis starts from tasks; a child projection also rebuilds the real
        synchronized stream. Existing syncs still pass through the ordinary
        instruction/prefix rules in completion_graph, including Wait gates.
        """
        require(
            command.kind in {"task", "set", "wait", "barrier", "cross_set", "cross_wait"},
            "insertion cuts require an actual native command",
        )
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
        self.completion().register(queries)

    def covered(self, query):
        return self.completion().covered(query)

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

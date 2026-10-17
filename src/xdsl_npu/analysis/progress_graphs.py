"""Finite nonempty reachability on an immutable conditional-graph topology.

An enabled edge mask is supplied by the caller. Hidden SCCs only accelerate
that graph's reachability; their existence never proves a guard satisfiable,
an exact projection, stream displacement, ownership or native completion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

Charge = Callable[[str, int], None]


@dataclass(frozen=True, slots=True)
class Projection:
    """Boundary destinations, indexed by original source node.

    Hidden source rows are zero. ``through_hidden`` includes only paths that
    visit at least one hidden node. No row contains a zero-length identity.
    """

    rows: tuple[int, ...]
    through_hidden: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PathTree:
    """A completed source search, frozen independently of any conditional proof.

    The read-only predecessor map has no mutable alias outside its builder.
    Its source entry exists only when a real nonempty path returns to source.
    Every other retained vertex is a terminal; hidden entries allow retracing.
    """

    topology: GraphTopology
    source: int
    hidden: frozenset[int]
    enabled: int
    predecessors: Mapping[int, int]

    def __init__(self):
        raise TypeError("PathTree is created by a completed GraphTopology.search")

    def path(self, destination: int, charge: Charge) -> tuple[int, ...] | None:
        """Reconstruct a shortest nonempty path to a retained destination."""
        self.topology._node(destination)
        charge("graph path lookup", 1)
        if destination in self.hidden:
            raise ValueError("graph path destination must be retained")
        index = self.predecessors.get(destination)
        if index is None:
            return None
        result = []
        while True:
            charge("graph path predecessor", 1)
            result.append(index)
            before = self.topology.edges[index][0]
            if before == self.source:
                break
            index = self.predecessors[before]
        charge("graph path result", len(result))
        return tuple(reversed(result))


@dataclass(frozen=True, slots=True, eq=False)
class GraphTopology:
    """An owned topology identity with stable original edge indices.

    The caller charges construction by ``size + 2 * len(edges)``. Adjacency
    tuples retain edge order, including parallel edges and nonempty self-loops.
    """

    size: int
    edges: tuple[tuple[int, int], ...]
    outgoing: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    incoming: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.size) is not int or self.size < 0:
            raise ValueError("graph topology size must be a nonnegative integer")
        edges = tuple((source, destination) for source, destination in self.edges)
        outgoing: list[list[int]] = [[] for _ in range(self.size)]
        incoming: list[list[int]] = [[] for _ in range(self.size)]
        for index, (source, destination) in enumerate(edges):
            self._node(source)
            self._node(destination)
            outgoing[source].append(index)
            incoming[destination].append(index)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "outgoing", tuple(map(tuple, outgoing)))
        object.__setattr__(self, "incoming", tuple(map(tuple, incoming)))

    def _node(self, node: int):
        if type(node) is not int or not 0 <= node < self.size:
            raise IndexError("graph topology node is out of range")

    def _hidden(self, hidden: frozenset[int], charge: Charge):
        if not isinstance(hidden, frozenset):
            raise TypeError("hidden graph nodes must be a frozenset")
        charge("graph hidden validation", len(hidden))
        for node in hidden:
            self._node(node)

    def _enabled(self, enabled: int, charge: Charge) -> bytes:
        if type(enabled) is not int or enabled < 0 or enabled >> len(self.edges):
            raise ValueError("enabled graph edges must be an in-range nonnegative bitmask")
        charge("graph enabled mask", max(1, (len(self.edges) + 63) // 64))
        return enabled.to_bytes((len(self.edges) + 7) // 8, "little")

    def project(self, hidden: frozenset[int], enabled: int, charge: Charge) -> Projection:
        """Project through only hidden nodes using their SCC condensation DAG.

        Boundary rows stop at every other retained node, including one omitted
        from a caller's output selection. Bitset unions charge full topology
        width in 64-bit word units. There is no predecessor/successor product.
        """
        self._hidden(hidden, charge)
        mask = self._enabled(enabled, charge)
        charge("graph projection initialization", self.size)
        hidden_nodes = tuple(node for node in range(self.size) if node in hidden)
        words = max(1, (self.size + 63) // 64)

        # Iterative Kosaraju avoids recursion depth depending on graph size.
        seen: set[int] = set()
        finished: list[int] = []
        for root in hidden_nodes:
            charge("graph first DFS root", 1)
            if root in seen:
                continue
            seen.add(root)
            pending = [(root, iter(self.outgoing[root]))]
            while pending:
                node, edges = pending[-1]
                index = next(edges, None)
                if index is None:
                    charge("graph first DFS finish", 1)
                    finished.append(node)
                    pending.pop()
                    continue
                charge("graph first DFS edge", 1)
                if not mask[index >> 3] & (1 << (index & 7)):
                    continue
                destination = self.edges[index][1]
                if destination in hidden and destination not in seen:
                    seen.add(destination)
                    pending.append((destination, iter(self.outgoing[destination])))

        components: list[list[int]] = []
        component_of = [-1] * self.size
        for root in reversed(finished):
            charge("graph second DFS root", 1)
            if component_of[root] != -1:
                continue
            component = len(components)
            members: list[int] = []
            components.append(members)
            component_of[root] = component
            pending_nodes = [root]
            while pending_nodes:
                node = pending_nodes.pop()
                charge("graph second DFS node", 1)
                members.append(node)
                for index in self.incoming[node]:
                    charge("graph second DFS edge", 1)
                    if not mask[index >> 3] & (1 << (index & 7)):
                        continue
                    source = self.edges[index][0]
                    if source in hidden and component_of[source] == -1:
                        component_of[source] = component
                        pending_nodes.append(source)

        targets = [0] * len(components)
        successors: list[dict[int, None]] = [{} for _ in components]
        indegree = [0] * len(components)
        for node in hidden_nodes:
            charge("graph condensation node", 1)
            component = component_of[node]
            for index in self.outgoing[node]:
                charge("graph condensation edge", 1)
                if not mask[index >> 3] & (1 << (index & 7)):
                    continue
                destination = self.edges[index][1]
                if destination in hidden:
                    other = component_of[destination]
                    if other != component and other not in successors[component]:
                        successors[component][other] = None
                        indegree[other] += 1
                else:
                    charge("graph component boundary union", words)
                    targets[component] |= 1 << destination

        ordered: list[int] = []
        ready = deque()
        for component, count in enumerate(indegree):
            charge("graph condensation root", 1)
            if count == 0:
                ready.append(component)
        while ready:
            component = ready.popleft()
            charge("graph condensation order node", 1)
            ordered.append(component)
            for other in successors[component]:
                charge("graph condensation order edge", 1)
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
        for component in reversed(ordered):
            charge("graph component propagation node", 1)
            for other in successors[component]:
                charge("graph component propagation edge", 1)
                if targets[other]:
                    charge("graph component target union", words)
                    targets[component] |= targets[other]

        rows = [0] * self.size
        through_hidden = [0] * self.size
        for source in range(self.size):
            charge("graph boundary source", 1)
            if source in hidden:
                continue
            visited_components: set[int] = set()
            for index in self.outgoing[source]:
                charge("graph boundary edge", 1)
                if not mask[index >> 3] & (1 << (index & 7)):
                    continue
                destination = self.edges[index][1]
                if destination in hidden:
                    component = component_of[destination]
                    if component not in visited_components:
                        visited_components.add(component)
                        if targets[component]:
                            charge("graph hidden boundary union", words)
                            through_hidden[source] |= targets[component]
                else:
                    charge("graph direct boundary union", words)
                    rows[source] |= 1 << destination
            if through_hidden[source]:
                charge("graph boundary row union", words)
                rows[source] |= through_hidden[source]
        return Projection(tuple(rows), tuple(through_hidden))

    def search(
        self,
        source: int,
        hidden: frozenset[int],
        enabled: int,
        charge: Charge,
    ) -> PathTree:
        """Complete one BFS for every boundary target, expanding only hidden nodes.

        The source is expanded once. A returning edge records a nonempty cycle
        without inserting an identity or expanding any other retained vertex.
        No partial tree escapes if charging or traversal raises an exception.
        """
        self._node(source)
        self._hidden(hidden, charge)
        if source in hidden:
            raise ValueError("graph search source must be retained")
        mask = self._enabled(enabled, charge)
        charge("graph search initialization", 1)
        previous: dict[int, int] = {}
        pending = deque((source,))
        while pending:
            node = pending.popleft()
            charge("graph search node", 1)
            for index in self.outgoing[node]:
                charge("graph search edge", 1)
                if not mask[index >> 3] & (1 << (index & 7)):
                    continue
                after = self.edges[index][1]
                if after not in previous:
                    previous[after] = index
                    if after in hidden:
                        pending.append(after)
        charge("graph search completion", 1)
        tree = object.__new__(PathTree)
        object.__setattr__(tree, "topology", self)
        object.__setattr__(tree, "source", source)
        object.__setattr__(tree, "hidden", hidden)
        object.__setattr__(tree, "enabled", enabled)
        object.__setattr__(tree, "predecessors", MappingProxyType(previous))
        return tree

"""Owned graph views of nonempty paths through explicitly hidden nodes.

The original graph remains a shared mathematical witness. Hiding nodes never
constructs fill edges or eagerly materializes all-path Boolean formulas. Each
returned path freezes its graph version and hidden set in the activation owner.
"""

from collections.abc import Iterable, Iterator

from .progress_activations import Activation, ActivationGraph, ActivationScope
from .progress_graphs import GraphTopology, Projection


class RegionConnectivity:
    """Nonempty boundary reachability without clock or exactness projection."""

    def __init__(self, size: int, scope: ActivationScope):
        if type(size) is not int or size < 0:
            raise ValueError("region connectivity size must be a nonnegative integer")
        self._size = size
        self._active = set(range(size))
        self._hidden: set[int] = set()
        self._relations: dict[tuple[int, int], Activation] = {}
        self._scope = scope
        self._false = scope.false
        self._topology: GraphTopology | None = None
        self._graph: ActivationGraph | None = None
        self._projection: Projection | None = None
        self._view_hidden: frozenset[int] | None = None

    @property
    def active(self) -> frozenset[int]:
        return frozenset(self._active)

    def _node(self, node: int):
        if type(node) is not int or not 0 <= node < self._size:
            raise IndexError("region connectivity node is out of range")

    def _active_node(self, node: int):
        self._node(node)
        if node not in self._active:
            raise ValueError("region connectivity node has been eliminated")

    def add(self, source: int, destination: int, activation: Activation):
        """OR one original edge between active endpoints into the next version."""
        self._active_node(source)
        self._active_node(destination)
        self._scope.charge("progress relation insertion")
        pair = source, destination
        previous = self._relations.get(pair, self._false)
        combined = self._scope.any_of(previous, activation)
        if combined is previous:
            return
        self._relations[pair] = combined
        self._topology = self._graph = self._projection = None
        self._view_hidden = None

    def _view(self) -> tuple[ActivationGraph, frozenset[int], Projection]:
        self._scope.charge("progress projection access")
        if self._graph is None:
            self._scope.charge(
                "progress topology construction", self._size + 2 * len(self._relations)
            )
            topology = GraphTopology(self._size, tuple(self._relations))
            graph = self._scope.graph(topology, tuple(self._relations.values()))
            self._topology, self._graph = topology, graph
        assert self._topology is not None
        if self._view_hidden is None:
            self._scope.charge("progress hidden snapshot", len(self._hidden))
            self._view_hidden = frozenset(self._hidden)
        if self._projection is None:
            count = len(self._topology.edges)
            self._scope.charge("progress may edge mask", max(1, (count + 63) // 64))
            self._projection = self._scope.project(self._graph, self._view_hidden, (1 << count) - 1)
        return self._graph, self._view_hidden, self._projection

    def _activation(
        self,
        source: int,
        destination: int,
        direct: Activation,
        graph: ActivationGraph,
        hidden: frozenset[int],
        projection: Projection,
    ) -> Activation:
        if direct.kind == "true":
            return direct
        bit = 1 << destination
        if not projection.rows[source] & bit:
            return self._false
        if not projection.through_hidden[source] & bit:
            return direct
        return self._scope.path(graph, hidden, source, destination)

    def relation(self, source: int, destination: int) -> Activation:
        """Return false for an absent pair or an endpoint already hidden."""
        self._node(source)
        self._node(destination)
        self._scope.charge("progress relation lookup")
        if source not in self._active or destination not in self._active:
            return self._false
        direct = self._relations.get((source, destination), self._false)
        if direct.kind == "true":
            return direct
        graph, hidden, projection = self._view()
        return self._activation(source, destination, direct, graph, hidden, projection)

    def eliminate(self, nodes: Iterable[int]):
        """Hide the selected nodes while retaining immutable original witnesses."""
        selected = set()
        for node in nodes:
            self._node(node)
            self._scope.charge("progress hidden selection")
            selected.add(node)
        self._scope.charge("progress hidden membership", len(selected))
        newly_hidden = selected & self._active
        if not newly_hidden:
            return
        self._scope.charge("progress vertex hiding", len(newly_hidden))
        self._hidden.update(newly_hidden)
        self._active.difference_update(newly_hidden)
        self._projection = None
        self._view_hidden = None

    def retained(self, boundary: Iterable[int]) -> Iterator[tuple[int, int, Activation]]:
        """Select output endpoints; every other active node remains a boundary."""
        members = set()
        for node in boundary:
            self._active_node(node)
            self._scope.charge("progress boundary selection")
            members.add(node)
        if not members:
            return
        graph, hidden, projection = self._view()
        words = max(1, (self._size + 63) // 64)
        for source in sorted(members):
            self._scope.charge("progress boundary source traversal")
            targets = projection.rows[source]
            while targets:
                self._scope.charge("progress boundary target traversal", words)
                bit = targets & -targets
                targets ^= bit
                destination = bit.bit_length() - 1
                if destination in members:
                    direct = self._relations.get((source, destination), self._false)
                    yield (
                        source,
                        destination,
                        self._activation(source, destination, direct, graph, hidden, projection),
                    )

"""Sparse factored Boolean paths for one owned progress region.

The activation scope owns formula composition, mathematical proofs and budgets.
This container charges its graph visits to that same scope. It does not project
clocks or change any progress step/exactness contract.
"""

from collections.abc import Iterable, Iterator

from .progress_activations import Activation, ActivationScope


class RegionConnectivity:
    """Nonempty directed paths whose internal nodes have been eliminated.

    There is no implicit zero-length relation, including on the diagonal.
    Neighbor and output traversal follows node index order. Each endpoint pair
    stores one activation root; joining a path never distributes AND over OR.
    """

    def __init__(self, size: int, scope: ActivationScope):
        if type(size) is not int or size < 0:
            raise ValueError("region connectivity size must be a nonnegative integer")
        self._size = size
        self._active = set(range(size))
        self._outgoing: dict[int, dict[int, Activation]] = {}
        self._incoming: dict[int, set[int]] = {}
        self._scope = scope
        self._false = scope.false

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

    def _disconnect(self, source: int, destination: int):
        self._scope.charge("progress relation removal")
        outgoing = self._outgoing[source]
        del outgoing[destination]
        if not outgoing:
            del self._outgoing[source]
        incoming = self._incoming[destination]
        incoming.remove(source)
        if not incoming:
            del self._incoming[destination]

    def add(self, source: int, destination: int, activation: Activation):
        self._active_node(source)
        self._active_node(destination)
        self._scope.charge("progress relation insertion")
        previous = self._outgoing.get(source, {}).get(destination, self._false)
        combined = self._scope.any_of(previous, activation)
        if combined is self._false:
            return
        self._outgoing.setdefault(source, {})[destination] = combined
        self._incoming.setdefault(destination, set()).add(source)

    def relation(self, source: int, destination: int) -> Activation:
        """Return false when the pair is absent or an endpoint was eliminated."""
        self._node(source)
        self._node(destination)
        self._scope.charge("progress relation lookup")
        return self._outgoing.get(source, {}).get(destination, self._false)

    def eliminate(self, nodes: Iterable[int]):
        """Remove nodes after composing only their active predecessors/successors.

        An internal self-loop's Boolean star needs no expansion. Traversing it
        adds conjuncts; skipping that traversal already supplies a weaker path
        through the node. Retained self-cycles remain real nonempty paths.
        This reasoning concerns Boolean connectivity, not stream displacement.
        """
        selected = sorted(set(nodes))
        for node in selected:
            self._node(node)
        for node in selected:
            if node not in self._active:
                continue
            self._scope.charge("progress vertex elimination")
            incoming = self._incoming.get(node, ())
            outgoing = self._outgoing.get(node, {})
            if incoming:
                self._scope.charge("progress predecessor snapshot", len(incoming))
            if outgoing:
                self._scope.charge("progress successor snapshot", len(outgoing))
            predecessors = tuple(
                (source, self._outgoing[source][node])
                for source in sorted(incoming)
                if source != node
            )
            successors = tuple(
                (destination, outgoing[destination])
                for destination in sorted(outgoing)
                if destination != node
            )
            for source, left in predecessors:
                for destination, right in successors:
                    self.add(source, destination, self._scope.all_of(left, right))
            for source in sorted(self._incoming.get(node, ())):
                self._disconnect(source, node)
            for destination in sorted(self._outgoing.get(node, ())):
                self._disconnect(node, destination)
            self._active.remove(node)

    def retained(self, boundary: Iterable[int]) -> Iterator[tuple[int, int, Activation]]:
        """Yield only stored relations between the requested active nodes."""
        selected = sorted(set(boundary))
        for node in selected:
            self._active_node(node)
        members = set(selected)
        for source in selected:
            self._scope.charge("progress boundary source traversal")
            for destination in sorted(self._outgoing.get(source, ())):
                self._scope.charge("progress boundary relation traversal")
                if destination in members:
                    yield source, destination, self._outgoing[source][destination]

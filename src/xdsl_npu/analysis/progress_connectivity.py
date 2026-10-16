"""Sparse Boolean path composition for one owned progress region.

The caller owns alternative absorption, condition proofs and their budgets.
This container only joins ordered conjunctions at a shared graph node. It
does not project clocks or change any progress step/exactness contract.
"""

from collections.abc import Callable, Iterable, Iterator

from .structured_schedule import ActionGuard, Comparison

GuardAlternatives = dict[frozenset[Comparison], ActionGuard]
AlternativeOwner = Callable[[GuardAlternatives, ActionGuard], None]


class RegionConnectivity:
    """Nonempty directed paths whose internal nodes have been eliminated.

    There is no implicit zero-length relation, including on the diagonal.
    Neighbor and output traversal follows node index order. The owner callback
    receives every insertion or path combination, including a duplicate which
    it may absorb, and retains the ordered witnesses in each alternatives map.
    """

    def __init__(self, size: int, add_alternative: AlternativeOwner):
        if type(size) is not int or size < 0:
            raise ValueError("region connectivity size must be a nonnegative integer")
        self._size = size
        self._active = set(range(size))
        self._outgoing: dict[int, dict[int, GuardAlternatives]] = {}
        self._incoming: dict[int, set[int]] = {}
        self._add_alternative = add_alternative

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
        outgoing = self._outgoing[source]
        del outgoing[destination]
        if not outgoing:
            del self._outgoing[source]
        incoming = self._incoming[destination]
        incoming.remove(source)
        if not incoming:
            del self._incoming[destination]

    def add(self, source: int, destination: int, guard: ActionGuard):
        self._active_node(source)
        self._active_node(destination)
        alternatives = self._outgoing.get(source, {}).get(destination)
        if alternatives is None:
            alternatives = {}
            self._add_alternative(alternatives, guard)
            if alternatives:
                self._outgoing.setdefault(source, {})[destination] = alternatives
                self._incoming.setdefault(destination, set()).add(source)
        else:
            self._add_alternative(alternatives, guard)
            if not alternatives:
                self._disconnect(source, destination)

    def alternatives(self, source: int, destination: int) -> tuple[ActionGuard, ...]:
        """Return ordered witnesses; an eliminated endpoint has no relation."""
        self._node(source)
        self._node(destination)
        return tuple(self._outgoing.get(source, {}).get(destination, {}).values())

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
            predecessors = tuple(
                (source, self.alternatives(source, node))
                for source in sorted(self._incoming.get(node, ()))
                if source != node
            )
            successors = tuple(
                (destination, self.alternatives(node, destination))
                for destination in sorted(self._outgoing.get(node, ()))
                if destination != node
            )
            for source, left in predecessors:
                for destination, right in successors:
                    for before in left:
                        for after in right:
                            self.add(
                                source,
                                destination,
                                ActionGuard(
                                    tuple(dict.fromkeys(before.comparisons + after.comparisons))
                                ),
                            )
            for source in sorted(self._incoming.get(node, ())):
                self._disconnect(source, node)
            for destination in sorted(self._outgoing.get(node, ())):
                self._disconnect(node, destination)
            self._active.remove(node)

    def retained(self, boundary: Iterable[int]) -> Iterator[tuple[int, int, ActionGuard]]:
        """Yield only stored relations between the requested active nodes."""
        selected = sorted(set(boundary))
        for node in selected:
            self._active_node(node)
        members = set(selected)
        for source in selected:
            for destination in sorted(self._outgoing.get(source, ())):
                if destination in members:
                    for guard in self.alternatives(source, destination):
                        yield source, destination, guard

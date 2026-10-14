"""Private planning search over one immutable completion DAG.

Candidate deletion invalidates only paths containing a removed constraint.
Every accepted query retains a concrete path. Final verification constructs a
new graph from actual IR and never reads this planner's path cache.
"""

from collections import defaultdict
from dataclasses import dataclass

from .model import require


@dataclass(frozen=True, slots=True)
class _Path:
    arc: tuple[int, int]
    tail: "_Path | None"

    def __iter__(self):
        path = self
        while path is not None:
            yield path.arc
            path = path.tail


class ProofSearch:
    def __init__(self, graph, queries, *, removable=None):
        require(graph.reach is not None, "planning graph must be closed")
        self.graph = graph
        self.queries = frozenset(queries)
        self.removable = None if removable is None else frozenset(removable)
        self.paths = {}
        self.watched = {}
        self.users = defaultdict(set)
        self.omitted = frozenset()
        ranks = {node: rank for rank, node in enumerate(graph.order)}
        self.successors = tuple(
            tuple(sorted(targets, key=ranks.__getitem__, reverse=True)) for targets in graph.edges
        )
        # A consumer's proofs often share a long completion-prefix suffix.
        # Share that exact path within ONE graph state, never across deletions.
        suffixes = {}
        for query in sorted(self.queries, key=lambda q: (ranks[q[1]], -ranks[q[0]])):
            path = self._find(query, self.omitted, suffixes)
            require(
                path is not None,
                "fixed schedule cannot cover its physical dependencies within local banks",
            )
            self._install(query, path)

    def _find(self, query, omitted, suffixes=None):
        source, target = query
        if not self.graph.precedes(source, target):
            return None
        suffixes = {} if suffixes is None else suffixes
        seen = {source}
        stack = [(source, iter(self.successors[source]))]
        prefix = []
        target_bit = 1 << target

        def found(tail):
            for arc in reversed(prefix):
                tail = _Path(arc, tail)
                suffixes[target, arc[0]] = tail
            return tail

        while stack:
            node, successors = stack[-1]
            if (tail := suffixes.get((target, node))) is not None:
                return found(tail)
            for successor in successors:
                if (node, successor) in omitted or successor in seen:
                    continue
                if successor != target and not self.graph.reach[successor] & target_bit:
                    continue
                seen.add(successor)
                prefix.append((node, successor))
                if successor == target:
                    return found(None)
                stack.append((successor, iter(self.successors[successor])))
                break
            else:
                stack.pop()
                if prefix:
                    prefix.pop()
        return None

    def _install(self, query, path):
        for arc in self.watched.get(query, ()):
            self.users[arc].discard(query)
        self.paths[query] = path
        arcs = frozenset(arc for arc in path if self.removable is None or arc in self.removable)
        self.watched[query] = arcs
        for arc in arcs:
            self.users[arc].add(query)

    def _deletions(self, arcs):
        arcs = frozenset(arcs)
        require(
            self.removable is None or arcs <= self.removable,
            "planning attempted to delete an untracked native constraint",
        )
        return arcs

    def precedes_without(self, query, arcs):
        candidate = self.omitted | self._deletions(arcs)
        path = self.paths.get(query)
        if path is not None and not self.watched[query] & candidate:
            return True
        return self._find(query, candidate) is not None

    def remove(self, arcs, *, drop_queries=()):
        """Commit only if ALL required queries still have a concrete path."""
        arcs = self._deletions(arcs) - self.omitted
        candidate = self.omitted | arcs
        dropped = self.queries & frozenset(drop_queries)
        affected = (set().union(*(self.users[arc] for arc in arcs)) if arcs else set()) - dropped
        replacements, suffixes = {}, {}
        for query in affected:
            path = self._find(query, candidate, suffixes)
            if path is None:
                return False
            replacements[query] = path
        for query, path in replacements.items():
            self._install(query, path)
        for query in dropped:
            self.paths.pop(query)
            for arc in self.watched.pop(query):
                self.users[arc].discard(query)
        self.queries -= dropped
        self.omitted = candidate
        return True

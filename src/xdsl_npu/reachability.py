"""Private planning search over one immutable completion DAG.

Candidate deletion invalidates only paths containing a removed constraint.
Every accepted query retains a concrete path. Final verification constructs a
new graph from actual IR and never reads this planner's path cache.
"""

from collections import defaultdict

from .model import require


class ProofSearch:
    def __init__(self, graph, queries):
        require(graph.reach is not None, "planning graph must be closed")
        self.graph = graph
        self.queries = frozenset(queries)
        self.paths = {}
        self.users = defaultdict(set)
        self.omitted = frozenset()
        ranks = {node: rank for rank, node in enumerate(graph.order)}
        self.successors = tuple(
            tuple(sorted(targets, key=ranks.__getitem__, reverse=True)) for targets in graph.edges
        )
        for query in self.queries:
            path = self._find(query, self.omitted)
            require(
                path is not None,
                "fixed schedule cannot cover its physical dependencies within local banks",
            )
            self._install(query, path)

    def _find(self, query, omitted):
        source, target = query
        if not self.graph.precedes(source, target):
            return None
        seen = {source}
        stack = [(source, iter(self.successors[source]))]
        path = []
        target_bit = 1 << target
        while stack:
            node, successors = stack[-1]
            for successor in successors:
                if (node, successor) in omitted or successor in seen:
                    continue
                if successor != target and not self.graph.reach[successor] & target_bit:
                    continue
                seen.add(successor)
                if successor == target:
                    return (*path, (node, successor))
                stack.append((successor, iter(self.successors[successor])))
                path.append((node, successor))
                break
            else:
                stack.pop()
                if path:
                    path.pop()
        return None

    def _install(self, query, path):
        for arc in self.paths.get(query, ()):
            self.users[arc].discard(query)
        self.paths[query] = path
        for arc in path:
            self.users[arc].add(query)

    def precedes_without(self, query, arcs):
        candidate = self.omitted | frozenset(arcs)
        path = self.paths.get(query)
        if path is not None and all(arc not in candidate for arc in path):
            return True
        return self._find(query, candidate) is not None

    def remove(self, arcs, *, drop_queries=()):
        """Commit only if ALL required queries still have a concrete path."""
        arcs = frozenset(arcs) - self.omitted
        candidate = self.omitted | arcs
        dropped = self.queries & frozenset(drop_queries)
        affected = (set().union(*(self.users[arc] for arc in arcs)) if arcs else set()) - dropped
        replacements = {}
        for query in affected:
            path = self._find(query, candidate)
            if path is None:
                return False
            replacements[query] = path
        for query, path in replacements.items():
            self._install(query, path)
        for query in dropped:
            for arc in self.paths.pop(query):
                self.users[arc].discard(query)
        self.queries -= dropped
        self.omitted = candidate
        return True

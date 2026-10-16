"""Batch completion facts on explicit, guarded hardware edges.

An edge is supplied by a native contract or an accepted protocol, never by a
reachability oracle. Bits name demanded source milestones. A fact's predicate
is the conjunction along its real path; alternative paths retain their guards.
"""

from heapq import heappop, heappush

from .budget import poll
from .completion import CountDomain, canonical, intersection
from .completion import includes as _includes
from .diagnostics import require

ALL = CountDomain(1, 0, minimum=0)


def propagate_ordered(successors, seeds):
    """The unconditional specialization for a certified topological domain."""
    facts = [0] * len(successors)
    for node, bits in seeds:
        facts[node] |= bits
    transfers = 0
    for node, following in enumerate(successors):
        if node % 4096 == 0:
            poll("ordered completion facts", nodes=len(facts), processed=node)
        bits = facts[node]
        if bits:
            for target, _ in following:
                facts[target] |= bits
                transfers += 1
    return facts, transfers


class CompletionFlow:
    def __init__(self):
        self.edges, self.facts = [], []
        self.pending = []
        self.delta = {}
        self.transfers = 0
        self.edge_count = 0

    def node(self):
        index = len(self.edges)
        require(index < 100_000, "completion transfer node budget exceeded")
        self.edges.append([])
        self.facts.append({})
        return index

    def add_fact(self, node, guard, bits):
        guard = canonical(guard)
        known = self.facts[node]
        for prior, mask in known.items():
            if bits & mask and _includes(prior, guard):
                bits &= ~mask
                if not bits:
                    return
        # Keep a condition antichain per fact. A newly discovered wider path
        # supersedes narrower records and their not-yet-propagated work.
        for prior, mask in tuple(known.items()):
            if bits & mask and _includes(guard, prior):
                if remaining := mask & ~bits:
                    known[prior] = remaining
                else:
                    del known[prior]
        known[guard] = known.get(guard, 0) | bits
        if node not in self.delta:
            self.delta[node] = {}
            heappush(self.pending, node)
        pending = self.delta[node]
        pending[guard] = pending.get(guard, 0) | bits

    def edge(self, source, destination, guard=ALL):
        if guard is None or source == destination:
            return
        guard = canonical(guard)
        if (destination, guard) in self.edges[source]:
            return
        self.edges[source].append((destination, guard))
        self.edge_count += 1
        for domain, bits in tuple(self.facts[source].items()):
            overlap = intersection(domain, guard)
            if overlap is not None:
                self.add_fact(destination, overlap, bits)

    def propagate(self):
        while self.pending:
            source = heappop(self.pending)
            for domain, bits in self.delta.pop(source).items():
                bits &= self.facts[source].get(domain, 0)
                if not bits:
                    continue
                for target, guard in self.edges[source]:
                    self.transfers += 1
                    if self.transfers % 1024 == 0:
                        poll("completion fact propagation", transfers=self.transfers)
                    require(self.transfers <= 2_000_000, "completion fact transfer budget exceeded")
                    overlap = intersection(domain, guard)
                    if overlap is not None:
                        self.add_fact(target, overlap, bits)

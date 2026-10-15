"""Select protocols on the real instruction graph, without trial deletion.

Candidate instructions are detached IR. Their native graph positions are
transparent until selected: every semantic edge incident on their begin/end
is gated by that candidate. Native prefix edges remain unconditional. A gap
is resolved by selecting a concrete route through this graph, then propagating
all demanded completion facts along newly enabled edges. No unselected route
can discharge another obligation.
"""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

from .budget import poll
from .model import require


@dataclass(frozen=True)
class Candidate:
    declaration: object
    placements: tuple


class _DomainFlow:
    def __init__(self, domain, owners, selected):
        self.domain, self.owners = domain, owners
        self.facts = [0] * len(domain.successors)
        self.delta = [0] * len(self.facts)
        self.pending = []
        self.successors = [[] for _ in self.facts]
        self.waiting = defaultdict(list)
        self.active_edges = set()
        self.transfers = 0
        for source, following in enumerate(domain.successors):
            for target, arc in following:
                mask = owners[arc[0]] | owners[arc[1]]
                if not mask & ~selected:
                    self.edge(source, target)
                else:
                    bits = mask
                    while bits:
                        bit = bits & -bits
                        self.waiting[bit].append((source, target, mask))
                        bits ^= bit

    def seed(self, node, bits):
        bits &= ~self.facts[node]
        if bits:
            self.facts[node] |= bits
            if not self.delta[node]:
                heappush(self.pending, node)
            self.delta[node] |= bits

    def edge(self, source, target):
        if (source, target) not in self.active_edges:
            self.active_edges.add((source, target))
            self.successors[source].append(target)
            self.seed(target, self.facts[source])

    def activate(self, bit, selected):
        for source, target, mask in self.waiting.pop(bit, ()):
            if not mask & ~selected:
                self.edge(source, target)

    def propagate(self):
        while self.pending:
            source = heappop(self.pending)
            bits, self.delta[source] = self.delta[source], 0
            for target in self.successors[source]:
                self.seed(target, bits)
                self.transfers += 1
                if self.transfers % 4096 == 0:
                    poll("selected protocol facts", transfers=self.transfers)

    def route(self, source, target, selected):
        """A concrete completion route with few additional protocols.

        This is bounded route selection on existing native edges, not another
        oracle used to discover edges. Keeping one cost per vertex is a
        deterministic policy, not globally optimal protocol set selection.
        """
        costs = [None] * len(self.facts)
        costs[source] = 0
        for node in range(source, target + 1):
            if node % 4096 == 0:
                poll("protocol route selection", states=len(costs), processed=node)
            cost = costs[node]
            if cost is None:
                continue
            for following, arc in self.domain.successors[node]:
                if following > target:
                    continue
                candidate = cost | ((self.owners[arc[0]] | self.owners[arc[1]]) & ~selected)
                previous = costs[following]
                if previous is None or (candidate.bit_count(), candidate) < (
                    previous.bit_count(),
                    previous,
                ):
                    costs[following] = candidate
        return costs[target]


def _key(query):
    source, sg, destination, dg, *rest = query
    return source.index, sg, destination.index, dg, tuple(rest)


class ProtocolSelection:
    def __init__(self, graph, candidates, queries):
        operations = {
            op: 1 << number for number, c in enumerate(candidates) for _, _, op in c.placements
        }
        self.owners = [
            operations.get(node.command.operation, 0) if node.point in {"begin", "end"} else 0
            for node in graph.nodes
        ]
        self.selected = 0
        self.queries, self.flows = {}, {}
        self.routes = 0
        groups = defaultdict(list)
        for query in queries:
            key = _key(query)
            if key in self.queries:
                continue
            normalized = graph._query(*query)
            self.queries[key] = None  # Empty activation is vacuously satisfied.
            if normalized is not None:
                groups[normalized[0]].append((key, *normalized[1:]))
        # The all-candidate potential certifies the order of every selectable
        # subgraph. It supplies no completion fact to the selected subgraph.
        graph.potential()
        for guard, items in groups.items():
            domain = graph.domain(*guard)
            flow = self.flows[guard] = _DomainFlow(domain, self.owners, self.selected)
            origins = {}
            for key, source, target in items:
                source, target = domain.index.get(source), domain.index.get(target)
                require(
                    source is not None and target is not None,
                    "completion obligation lies outside its actual guard domain",
                )
                bit = origins.setdefault(source, 1 << len(origins))
                flow.seed(source, bit)
                self.queries[key] = flow, source, target, bit

    def covered(self, query):
        item = self.queries[_key(query)]
        if item is None:
            return True
        flow, _, target, bit = item
        flow.propagate()
        return bool(flow.facts[target] & bit)

    def ensure(self, query, diagnostic):
        if self.covered(query):
            return
        flow, source, target, _ = self.queries[_key(query)]
        self.routes += 1
        choices = flow.route(source, target, self.selected)
        require(choices is not None and choices != 0, diagnostic)
        self.selected |= choices
        while choices:
            bit = choices & -choices
            for current in self.flows.values():
                current.activate(bit, self.selected)
            choices ^= bit
        require(self.covered(query), "selected native route failed to establish its obligation")

"""Demanded reachability within one fixed child's owned native topology.

Only constant native bounds/points and unconditional local arcs belong here.
Parent activations, parent clocks and serial-query transitions remain outside
this transfer. Sources are registered on demand in one interval flow; the
engine never expands a fixed trip count or constructs all endpoint pairs.
"""

from .budget import poll
from .completion_flow import ALL
from .diagnostics import require
from .symbolic_completion import CompletionTopology, Family, SymbolicCompletion

Binding = tuple[int, tuple[int, int]]


def _constant(point):
    return (
        type(point) is tuple
        and len(point) == 2
        and type(point[0]) is int
        and point[0] == 0
        and type(point[1]) is int
    )


def _owned_topology(topology):
    require(isinstance(topology, CompletionTopology), "local transfer needs a native topology")
    size = len(topology.families)
    require(size == len(topology.edges), "local transfer topology has unmatched families")
    require(size <= 100_000, "symbolic completion node budget exceeded")
    for index, family in enumerate(topology.families):
        if index % 4096 == 0:
            poll("local child transfer families", nodes=index)
        require(
            isinstance(family, Family)
            and type(family.lower) is int
            and _constant(family.horizon)
            and type(family.period) is int
            and family.period > 0
            and type(family.residue) is int
            and 0 <= family.residue < family.period
            and (family.last is None or _constant(family.last))
            and type(family.monotone) is bool,
            "local transfer needs constant native family bounds",
        )
    for source, edges in enumerate(topology.edges):
        if source % 4096 == 0:
            poll("local child transfer arcs", nodes=source)
        for edge in edges:
            require(type(edge) is tuple and len(edge) == 5, "invalid local transfer arc")
            target, shift, source_point, target_point, guard = edge
            require(
                type(target) is int
                and 0 <= target < size
                and type(shift) is int
                and (
                    source_point is None
                    and target_point is None
                    or _constant(source_point)
                    and _constant(target_point)
                ),
                "local transfer needs constant native arc bindings",
            )
            require(guard == ALL, "local transfer needs unconditional native arcs")
    # Families and arc records are immutable values. Copy their mutable owner
    # lists once so later edits to the caller's builder cannot change this proof.
    owned = CompletionTopology()
    owned.families = list(topology.families)
    owned.edges = [tuple(edges) for edges in topology.edges]
    owned.edge_count = sum(map(len, owned.edges))
    return owned


class LocalTransfer:
    """One fixed native transfer, queried by actual physical point bindings.

    Coverage includes an active binding's identity, as native seed coverage
    does. A caller with a nonidentity Site contract must apply that rule using
    Site identity, since different Sites can have the same physical binding.
    """

    def __init__(self, topology: CompletionTopology):
        self._topology = _owned_topology(topology)
        self._flow = SymbolicCompletion(topology=self._topology)
        self._origins: dict[Binding, int] = {}
        self._targets: dict[Binding, tuple[int, int]] = {}
        self._revision = 0
        self._requests = self._source_requests = self._source_reuses = 0
        self._propagations = self._target_reads = self._target_reuses = 0
        self._target_scans = self._hits = 0

    def _binding(self, binding):
        require(
            type(binding) is tuple
            and len(binding) == 2
            and type(binding[0]) is int
            and 0 <= binding[0] < len(self._topology.families)
            and _constant(binding[1]),
            "local transfer needs an actual node and constant point",
        )
        return binding

    def _sources(self, bindings):
        bits = []
        for index, binding in enumerate(bindings):
            if index % 128 == 0:
                poll("local child transfer sources", sources=len(self._origins))
            self._source_requests += 1
            if binding in self._origins:
                self._source_reuses += 1
            else:
                bit = self._origins[binding] = 1 << len(self._origins)
                self._flow.seed(*binding, ALL, bit)
                self._revision += 1
            bits.append(self._origins[binding])
        return bits

    def _propagate(self):
        if self._flow.pending:
            self._propagations += 1
            self._flow.propagate()

    def _read(self, binding):
        self._target_reads += 1
        previous = self._targets.get(binding)
        if previous is not None and previous[0] == self._revision:
            self._target_reuses += 1
            return previous[1]
        node, point = binding
        family = self._topology.families[node]
        result = 0
        # Constant bounds and unconditional seeds/arcs cannot create a count
        # predicate or an affine endpoint. Read the full origin mask once.
        if (
            family.lower <= point[1] <= family.horizon[1]
            and (family.last is None or point == family.last)
            and family.residue_guard(point) is not None
        ):
            for (lo, hi, _), bits in self._flow.facts.get(node, {}).items():
                self._target_scans += 1
                if self._target_scans % 4096 == 0:
                    poll("local child transfer point coverage", facts=self._target_scans)
                if lo[1] <= point[1] <= hi[1]:
                    result |= bits
        self._targets[binding] = self._revision, result
        return result

    def covered(self, source: Binding, target: Binding) -> bool:
        return self.reachable(source, (target,))[0]

    def reachable(self, source: Binding, targets: tuple[Binding, ...]) -> tuple[bool, ...]:
        source = self._binding(source)
        targets = tuple(self._binding(target) for target in targets)
        self._requests += len(targets)
        if not targets:
            return ()
        bit = self._sources((source,))[0]
        self._propagate()
        result = tuple(bool(self._read(target) & bit) for target in targets)
        self._hits += sum(result)
        return result

    def predecessors(self, sources: tuple[Binding, ...], target: Binding) -> tuple[bool, ...]:
        sources = tuple(self._binding(source) for source in sources)
        target = self._binding(target)
        self._requests += len(sources)
        if not sources:
            return ()
        bits = self._sources(sources)
        self._propagate()
        reached = self._read(target)
        result = tuple(bool(reached & bit) for bit in bits)
        self._hits += sum(result)
        return result

    def statistics(self) -> dict:
        return {
            "topology_instances": 1,
            "topology_nodes": len(self._topology.families),
            "topology_edges": self._topology.edge_count,
            "query_domains": int(bool(self._origins)),
            "translated_domain_states": 0,
            "local_builds": 1,
            "local_requests": self._requests,
            "local_origins": len(self._origins),
            "local_source_requests": self._source_requests,
            "local_source_reuses": self._source_reuses,
            "local_propagation_passes": self._propagations,
            "local_target_reads": self._target_reads,
            "local_target_reuses": self._target_reuses,
            **self._flow.statistics(),
            "coverage_batches": self._target_reads - self._target_reuses,
            "coverage_requests": self._requests,
            "coverage_fact_scans": self._target_scans,
            "coverage_result_items": self._hits,
        }

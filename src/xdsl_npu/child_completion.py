"""Project a verified fixed child onto real native sites and prefix ports.

The leaf's guarded native families remain intact. Only requested endpoints and
actor/pipe prefix ports become parent-visible vertices; child count never
controls the number of native nodes or templates. A retirement output carries
captured facts on its own pipe, not completion of the child as a whole.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from sys import int_info

from .budget import poll
from .completion import Position, Site
from .completion_flow import ALL
from .diagnostics import require
from .native_order import prefix_inputs
from .symbolic_completion import Family, SymbolicCompletion, native_family

MAX_PROJECTION_RELATIONS = 1_000_000


@dataclass(frozen=True)
class ChildPort:
    actor: str
    pipe: str | None
    kind: str
    side: str

    def __post_init__(self):
        require(
            self.side in {"input", "output"}
            and (
                self.kind == "issue"
                and self.pipe is None
                or self.kind in {"queue", "retire", "implicit"}
                and isinstance(self.pipe, str)
            ),
            "invalid child completion port",
        )


ProjectionPoint = ChildPort | Site
Binding = tuple[int, tuple[int, int]]


@dataclass(frozen=True)
class FixedNativeView:
    """One actual fixed native topology, before any endpoint-origin projection."""

    count: int
    inputs: tuple[ChildPort, ...]
    outputs: tuple[ChildPort, ...]
    flow: SymbolicCompletion
    graph: object
    streams: dict
    port_bindings: dict[ChildPort, Binding]
    bind_site: Callable[[Site], Binding]


def native_fixed_child(analysis, *, insertion_cuts=False) -> FixedNativeView:
    """Retain one native body for exact local transfer, without projecting sites."""
    return _fixed_child(analysis, (), insertion_cuts=insertion_cuts, native_only=True)


@dataclass(frozen=True)
class FixedChildProjection:
    count: int
    inputs: tuple[ChildPort, ...]
    outputs: tuple[ChildPort, ...]
    endpoints: tuple[Site, ...]
    relations: frozenset[tuple[ProjectionPoint, ProjectionPoint]]
    node_count: int
    edge_count: int
    native_templates: int
    native_node_count: int
    native_edge_count: int
    fact_statistics: tuple[tuple[str, int], ...]

    @cached_property
    def _following(self):
        following = {}
        for source, target in self.relations:
            following.setdefault(source, []).append(target)
        return following

    def reaches(self, source, target):
        """A proved nonidentity relation, with full endpoint identity."""
        if source == target:
            return False
        seen, pending = {source}, [source]
        while pending:
            for point in self._following.get(pending.pop(), ()):
                if point == target:
                    return True
                if point not in seen:
                    seen.add(point)
                    pending.append(point)
        return False

    def statistics(self):
        return {
            "native_templates": self.native_templates,
            "native_nodes": self.native_node_count,
            "native_edges": self.native_edge_count,
            "projection_nodes": self.node_count,
            "projection_edges": self.edge_count,
            "relations": len(self.relations),
            **dict(self.fact_statistics),
        }


def _seed_words(count):
    whole, remainder = divmod(count, int_info.bits_per_digit)
    return int_info.bits_per_digit * whole * (whole + 1) // 2 + (whole + 1) * remainder


@dataclass
class _ProjectionBudget:
    words: int
    visits: int = 0

    def visit(self):
        self.visits += 1
        require(
            self.visits <= MAX_PROJECTION_RELATIONS, "fixed child projection work budget exceeded"
        )

    def mask(self, bits):
        self.words += (bits.bit_length() + int_info.bits_per_digit - 1) // int_info.bits_per_digit
        require(
            self.words <= MAX_PROJECTION_RELATIONS, "fixed child projection mask budget exceeded"
        )

    def edge(self, relations, source, target):
        self.visit()
        relations.add((source, target))
        require(
            len(relations) <= MAX_PROJECTION_RELATIONS,
            "fixed child endpoint relation budget exceeded (edges)",
        )


def _fixed_origins(flow, node, point, *, _budget=None):
    """Read every covered origin once at one constant, unconditional point.

    This specialization belongs to the fixed-child construction below. Its
    native horizons, port links and seed points are constants; every seed and
    arc has guard ALL. Propagation therefore retains constant intervals with
    guard ALL, so interval membership suffices for each fact's entire mask.
    Parent count-conditioned facts do not satisfy this contract.
    """
    family = flow.families[node]
    require(
        point[0] == family.horizon[0] == 0 and (family.last is None or family.last[0] == 0),
        "fixed child coverage requires constant points and family bounds",
    )
    if family.residue_guard(point) is None:
        return 0
    result = 0
    for (lo, hi, guard), bits in flow.facts.get(node, {}).items():
        if _budget is not None:
            _budget.visit()
        require(
            lo[0] == hi[0] == 0 and guard == ALL,
            "fixed child coverage requires constant intervals and unconditional facts",
        )
        if lo[1] <= point[1] <= hi[1]:
            result |= bits
    return result


def _fixed_relations(flow, sources, targets, bindings):
    """Build a sparse graph with exactly the fixed native preorder's closure.

    Every source has one origin per actual binding. Equal predecessor masks
    (including that origin) characterize endpoint SCCs. Ports remain distinct:
    an output has no origin, so equal output masks do not prove equivalence.
    """
    budget = _ProjectionBudget(_seed_words(len(sources)))
    require(budget.words <= MAX_PROJECTION_RELATIONS, "fixed child projection mask budget exceeded")
    source_bits = {bindings[source]: index for index, source in enumerate(sources)}
    require(len(source_bits) == len(sources), "duplicate fixed child origin binding")
    covered, groups, relations = {}, {}, set()
    for index, target in enumerate(targets):
        budget.visit()
        if index % 128 == 0:
            poll("fixed child endpoint coverage", target=index, sources=len(sources))
        binding = bindings[target]
        if binding not in covered:
            covered[binding] = _fixed_origins(flow, *binding, _budget=budget)
            budget.mask(covered[binding])
        if isinstance(target, Site):
            groups.setdefault(covered[binding], []).append(target)

    components, destinations = [], []
    for index, source in enumerate(sources):
        budget.visit()
        if isinstance(source, ChildPort):
            require(source.side == "input", "fixed child output cannot be an origin")
            bit = 1 << index
            budget.mask(bit)
            components.append((source, bit, bit))
    for ancestors, members in groups.items():
        budget.visit()
        representative = members[0]
        bit = 1 << source_bits[bindings[representative]]
        require(ancestors & bit, "fixed child endpoint lost its own origin")
        budget.mask(bit)
        components.append((representative, bit, ancestors))
        destinations.append((representative, ancestors, bit))
        for member in members[1:]:
            budget.edge(relations, representative, member)
            budget.edge(relations, member, representative)
    destinations.extend(
        (target, covered[bindings[target]], 0)
        for target in targets
        if isinstance(target, ChildPort)
    )

    # Strict ancestors have strictly smaller inclusive predecessor masks.
    # Suffix masks select the greatest remaining rank by binary search without
    # enumerating dense closure pairs or scanning all components per target.
    ordered = sorted(components, key=lambda component: component[2].bit_count())
    suffix = [0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        budget.visit()
        suffix[index] = suffix[index + 1] | ordered[index][1]
        budget.mask(suffix[index])
    for index, (target, ancestors, own) in enumerate(destinations):
        budget.visit()
        if index % 128 == 0:
            poll("fixed child sparse relations", target=index, edges=len(relations))
        remaining = ancestors & suffix[0] & ~own
        while remaining:
            low, high = 0, len(ordered)
            while high - low > 1:
                budget.visit()
                middle = (low + high) // 2
                if remaining & suffix[middle]:
                    low = middle
                else:
                    high = middle
            source, _, preceding = ordered[low]
            budget.edge(relations, source, target)
            remaining &= ~preceding
    return frozenset(relations)


def project_fixed_child(analysis, endpoints, *, insertion_cuts=False):
    """Project one verified leaf invocation with an actual constant count.

    ``analysis`` is the result of the existing leaf verifier. All requested
    ``Site`` objects must name the same lexical child invocation, and retain
    their actor, path, call, inner position and native/sync milestone. Unknown
    or inactive sites are rejected, including nonexistent terminal instances.

    Sparse relations have the same closure from inputs or requested sites to
    requested sites or outputs. Source-equals-target queries are false.
    Different Site identities at one physical milestone remain mutually reachable.
    Ports preserve incoming prefix facts even for a zero-trip child; they never
    create a child completion milestone.

    ``insertion_cuts`` rebuilds the actual synchronized command stream with
    forward-only issue/queue admission cuts. Sites may additionally name its
    physical issue/queue/retire/implicit prefixes or admit_issue/admit_queue.
    A fact acquired at admission reaches following commands, never the prefix
    before that admission. A native after_queue exit forwards inserted Set
    begins to the actual FIFO successor without entering the source begin.
    These cuts do not create asynchronous completion.
    """
    return _fixed_child(analysis, endpoints, insertion_cuts=insertion_cuts, native_only=False)


def _fixed_child(analysis, endpoints, *, insertion_cuts, native_only):
    loop, channel_map, streams, graph, _ = analysis
    count = loop.constant_count
    require(
        type(count) is int and count >= 0 and loop.cyclic is None,
        "child completion projection needs an actual constant child count",
    )
    require(
        (loop.invocation is None or loop.region is not None)
        and set(streams) == set(loop.actors)
        and graph.maximum == count,
        "child completion projection needs one verified leaf invocation",
    )
    endpoints = tuple(endpoints)
    require(all(isinstance(site, Site) for site in endpoints), "invalid child completion endpoint")
    endpoints = tuple(dict.fromkeys(endpoints))
    require(
        len({(site.region, site.call) for site in endpoints}) <= 1,
        "child completion endpoints mix lexical scopes or calls",
    )
    ports = sum(1 + 3 * len({c.task.pipe for c in sequence}) for sequence in streams.values())
    require(
        2 * ports + len(endpoints) <= MAX_PROJECTION_RELATIONS,
        "fixed child endpoint relation budget exceeded (points)",
    )
    poll("fixed child completion projection", endpoints=len(endpoints), ports=ports)
    if insertion_cuts:
        from .loop_graph import completion_graph
        from .loop_planning import DirectCompletion

        graph, streams, _, _ = completion_graph(
            loop,
            channel_map,
            period=graph.period,
            _planning=DirectCompletion(graph.period, count),
        )
        for sequence in streams.values():
            for command in sequence:
                if command.kind != "task":
                    continue
                poll("fixed child FIFO exits", command=command.index)
                for residue in range(graph.period):
                    begin = graph.ids.get((command.index, residue, "begin"))
                    if begin is None:
                        continue
                    after = graph.node(command, residue, "after_queue")
                    following, shift = graph.next_queue[command.index, residue]
                    graph.edge((begin, 0), (after, 0))
                    graph.edge((after, 0), (following, shift))
    flow = SymbolicCompletion()
    mapping = tuple(
        flow.node(native_family(meta, graph.period, (0, count))) for meta in graph.nodes
    )
    for source, target, distance in graph.edges:
        shift = distance * graph.period + graph.nodes[target].residue - graph.nodes[source].residue
        flow.edge(mapping[source], mapping[target], shift)

    def binding(command, kind, generation):
        if kind == "end" and command.kind in {"wait", "barrier", "cross_wait"}:
            kind = "begin"
        node = graph.ids.get((command.index, generation % graph.period, kind))
        if node is None:
            return None
        family = flow.families[mapping[node]]
        if not (
            family.lower <= generation <= family.horizon[1]
            and (family.last is None or generation == family.last[1])
        ):
            return None
        return mapping[node], (0, generation)

    commands, ambiguous = {}, set()
    for actor, sequence in streams.items():
        for command in sequence:
            key = (
                (actor, command.task.uid),
                command.channel,
                "" if command.kind == "task" else command.kind,
            )
            if key in commands:
                ambiguous.add(key)
            commands[key] = command
    bindings = {}
    milestones = {"begin", "end", "prefix"}
    if insertion_cuts:
        milestones |= {
            "issue",
            "queue",
            "retire",
            "implicit",
            "admit_issue",
            "admit_queue",
            "after_queue",
        }
    def site_binding(site):
        require(
            site.point in milestones
            and isinstance(site.position, Position)
            and type(site.position.end) is bool
            and type(site.position.offset) is int
            and site.position.offset >= 0,
            "invalid child completion endpoint milestone or inner position",
        )
        key = site.task, site.channel, site.action
        require(
            not insertion_cuts or key not in ambiguous,
            "child completion endpoint names ambiguous native commands",
        )
        command = commands.get(key)
        require(
            command is not None, "child completion endpoint has no matching actor/native command"
        )
        require(
            site.point != "after_queue" or command.kind == "task",
            "child FIFO exit requires a native task",
        )
        generation = site.position.evaluate(count)
        require(0 <= generation < count, "inactive child completion endpoint")
        endpoint = binding(command, site.point, generation)
        require(endpoint is not None, "inactive or unsupported child completion endpoint milestone")
        return endpoint

    for site in endpoints:
        bindings[site] = site_binding(site)

    inputs, outputs = [], []
    for actor, sequence in streams.items():
        require(sequence, "child completion actor has no native command")
        maximum = max(command.task.stage for command in sequence)
        boundaries = [(None, "issue", sequence[0], sequence[-1])]
        for pipe in sorted({command.task.pipe for command in sequence}):
            commands_on_pipe = [command for command in sequence if command.task.pipe == pipe]
            boundaries.extend(
                (pipe, kind, commands_on_pipe[0], commands_on_pipe[-1])
                for kind in ("queue", "retire", "implicit")
            )
        for pipe, kind, first, last in boundaries:
            before, after = (
                ChildPort(actor, pipe, kind, "input"),
                ChildPort(actor, pipe, kind, "output"),
            )
            inputs.append(before)
            outputs.append(after)
            entry = flow.node(Family(0, (0, 0)))
            exit = flow.node(Family(0, (0, 0)))
            bindings[before], bindings[after] = (entry, (0, 0)), (exit, (0, 0))
            flow.edge(entry, exit)  # Preserve input facts; do not complete work.
            if point := binding(first, kind, -first.task.stage):
                flow.edge(entry, point[0], source_point=(0, 0), target_point=point[1])
            generation = count + maximum - 1 - last.task.stage
            if point := binding(last, kind, generation):
                flow.edge(point[0], exit, source_point=point[1], target_point=(0, 0))
            # An acquisition at the final command's cut remains in the
            # outgoing prefix, even if that command itself is skipped.
            # Never feed admission back into the pre-insertion prefix.
            if (
                insertion_cuts
                and kind in {"issue", "queue"}
                and (point := binding(last, "admit_" + kind, generation))
            ):
                flow.edge(point[0], exit, source_point=point[1], target_point=(0, 0))
            if (
                insertion_cuts
                and kind == "queue"
                and (point := binding(last, "after_queue", generation))
            ):
                # The final native's successor lies outside this invocation.
                # An inserted Set begin still belongs to the outgoing FIFO.
                flow.edge(point[0], exit, source_point=point[1], target_point=(0, 0))
            points = {
                point: bound[0]
                for point in ("begin", "end", "prefix")
                if (bound := binding(last, point, generation)) is not None
            }
            if "begin" in points:
                points.setdefault("end", points["begin"])
                if (gate := prefix_inputs(last, points).get(kind)) is not None:
                    flow.edge(gate, exit, source_point=(0, generation), target_point=(0, 0))

    if native_only:
        return FixedNativeView(
            count,
            tuple(inputs),
            tuple(outputs),
            flow,
            graph,
            streams,
            dict(bindings),
            site_binding,
        )

    # Head/tail names and normalized wait milestones may bind to one actual
    # native point. Share their instrumentation origin, keeping every Site.
    origins = {}
    for source in (*inputs, *endpoints):
        origins.setdefault(bindings[source], source)
    sources, targets = tuple(origins.values()), (*endpoints, *outputs)
    require(
        _seed_words(len(sources)) <= MAX_PROJECTION_RELATIONS,
        "fixed child projection mask budget exceeded",
    )
    for bit, source in enumerate(sources):
        node, point = bindings[source]
        flow.seed(node, point, ALL, 1 << bit)
    flow.propagate()
    relations = _fixed_relations(flow, sources, targets, bindings)
    return FixedChildProjection(
        count,
        tuple(inputs),
        tuple(outputs),
        endpoints,
        relations,
        len(flow.families),
        flow.edge_count,
        sum(command.kind == "task" for sequence in streams.values() for command in sequence),
        len(graph.nodes),
        len(graph.edges),
        tuple(flow.statistics().items()),
    )

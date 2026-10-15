"""Project a verified fixed child onto real native sites and prefix ports.

The leaf's guarded native families remain intact. Only requested endpoints and
actor/pipe prefix ports become parent-visible vertices; child count never
controls the number of native nodes or templates. A retirement output carries
captured facts on its own pipe, not completion of the child as a whole.
"""

from dataclasses import dataclass

from .budget import poll
from .completion import Position, Site
from .completion_flow import ALL
from .model import require
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

    def reaches(self, source, target):
        """A proved nonidentity relation, with full endpoint identity."""
        return (source, target) in self.relations

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


def _fixed_origins(flow, node, point):
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
        require(
            lo[0] == hi[0] == 0 and guard == ALL,
            "fixed child coverage requires constant intervals and unconditional facts",
        )
        if lo[1] <= point[1] <= hi[1]:
            result |= bits
    return result


def _fixed_relations(flow, sources, targets, bindings):
    covered, relations = {}, set()
    for index, target in enumerate(targets):
        if index % 128 == 0:
            poll("fixed child endpoint coverage", target=index, sources=len(sources))
        binding = bindings[target]
        if binding not in covered:
            covered[binding] = _fixed_origins(flow, *binding)
        bits = covered[binding]
        while bits:
            bit = bits & -bits
            bits ^= bit
            source = sources[bit.bit_length() - 1]
            if source != target:
                relations.add((source, target))
    return frozenset(relations)


def project_fixed_child(analysis, endpoints, *, insertion_cuts=False):
    """Project one verified leaf invocation with an actual constant count.

    ``analysis`` is the result of the existing leaf verifier. All requested
    ``Site`` objects must name the same lexical child invocation, and retain
    their actor, path, call, inner position and native/sync milestone. Unknown
    or inactive sites are rejected, including nonexistent terminal instances.

    Relations run from inputs or requested sites to requested sites or outputs.
    Only source-equals-target identity edges are omitted. Different Site
    identities at the same physical milestone retain their proved relations.
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
        (ports + len(endpoints)) ** 2 <= MAX_PROJECTION_RELATIONS,
        "fixed child endpoint relation budget exceeded",
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
    for site in endpoints:
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
        bindings[site] = endpoint

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

    sources, targets = (*inputs, *endpoints), (*endpoints, *outputs)
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

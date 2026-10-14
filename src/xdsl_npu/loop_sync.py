"""Automatic synchronization and independent parametric loop checking."""

from collections import defaultdict

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .ir import KernelOp, LoopChannelOp, LoopSyncOp, StreamOp
from .loop_graph import channels, completion_graph
from .loop_model import extract_loop, loop_dependencies
from .model import CompileError, require
from .periodic import project, union


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _obligation_queries(obligation, graph, native):
    o = obligation
    for residue, first in project(o.period, o.residues, graph.period, o.distance):
        yield (native[o.source], residue - o.distance, native[o.destination], residue, first)


def _coverage_queries(obligations, graph, native):
    # Byte cells with the same endpoint relation share one proof. The earliest
    # real occurrence is the strongest guard domain; later cells need no new
    # graph/domain. Keep a real obligation for a concrete failure diagnostic.
    queries = {}
    for o in sorted(obligations, key=lambda o: o.hazard != "WAR"):
        for query in _obligation_queries(o, graph, native):
            source, _, destination, residue, first = query
            key = source.index, destination.index, o.distance, residue
            if key not in queries or first < queries[key][1][-1]:
                queries[key] = o, query
    return queries.values()


def _coverage(loop, obligations, graph, native):
    queries = _coverage_queries(obligations, graph, native)
    for o, query in queries:
        _, _, _, residue, first = query
        if graph.path(*query) is None:
            raise CompileError(
                f"missing loop {o.hazard} {'read_retired' if o.hazard == 'WAR' else 'write_done'}: "
                f"{o.source} -> {o.destination}, distance {o.distance}, protocol residue "
                f"{residue}/{graph.period}, first {first}, storage residues "
                f"{o.residues}/{o.period}, on {o.storage}[{o.start}:{o.stop}]"
            )
    return len(queries)


def _successor(channel, residue, count, period):
    active = [r for r in range(period) if r % channel.period in channel.residues]
    index = active.index(residue)
    cycles, following = divmod(index + count, len(active))
    return active[following] + cycles * period


def _counter_queries(period, channel_map, endpoints):
    for name, channel in channel_map.items():
        if channel.kind != "cross":
            continue
        for residue in range(period):
            if residue % channel.period not in channel.residues:
                continue
            following = _successor(channel, residue, 15, period)
            for source_actor, _ in channel.sources:
                for destination_actor, _ in channel.destinations:
                    yield (
                        name,
                        (
                            endpoints[destination_actor, name, "wait"],
                            residue + channel.distance,
                            endpoints[source_actor, name, "set"],
                            following,
                        ),
                    )


class _ProofSearch:
    """Delete command families without rebuilding the guarded completion graph.

    Stored paths are concrete, not just reachability bits. Removing a command
    invalidates its FIFO, capture, notification AND controller-gate edges.
    Only queries whose current path used that command need another search.
    The final public verifier reconstructs a graph from actual IR independently.
    """

    def __init__(self, loop, obligations, graph, native, channel_map, endpoints):
        self.graph = graph
        self.queries, self.paths, self.watched = {}, {}, {}
        self.users = defaultdict(set)
        self.channel_queries = defaultdict(set)
        self.omitted = frozenset()
        self.removable = frozenset(
            i
            for i, node in enumerate(graph.nodes)
            if node.command.kind != "task" and node.point in {"begin", "end"}
        )
        self.searches = 0
        for _, query in _coverage_queries(obligations, graph, native):
            self._add(query)
        for name, query in _counter_queries(graph.period, channel_map, endpoints):
            key = self._add(query)
            self.channel_queries[name].add(key)

    def _add(self, query):
        source, sg, destination, dg, *minimum = query
        key = source.index, sg, destination.index, dg, tuple(minimum)
        if key not in self.queries:
            self.queries[key] = query
            path = self.graph.path(*query)
            require(path is not None, "fixed loop schedule cannot cover a required completion")
            self._install(key, path)
        return key

    def _install(self, key, path):
        for node in self.watched.get(key, ()):
            self.users[node].discard(key)
        self.paths[key] = path
        nodes = frozenset(n for a, b, _ in path for n in (a, b) if n in self.removable)
        self.watched[key] = nodes
        for node in nodes:
            self.users[node].add(key)

    def remove(self, operations, channel):
        operations = set(operations)
        removed = frozenset(
            i for i in self.removable if self.graph.nodes[i].command.operation in operations
        )
        dropped = self.channel_queries.get(channel, set())
        affected = set().union(*(self.users[node] for node in removed)) - dropped
        candidate = self.omitted | removed
        replacements = {}
        for key in sorted(affected):
            self.searches += 1
            path = self.graph.path(*self.queries[key], omitted=candidate)
            if path is None:
                return False
            replacements[key] = path
        for key, path in replacements.items():
            self._install(key, path)
        for key in dropped:
            self.queries.pop(key)
            self.paths.pop(key)
            for node in self.watched.pop(key):
                self.users[node].discard(key)
        self.omitted = candidate
        return True


def _lifetime(loop, channel_map, graph, endpoints, *, local=True, counters=True):
    banks, cross_ids = defaultdict(set), set()
    local_queries = 0
    for name, channel in channel_map.items():
        if channel.kind == "local" and local:
            source_actor, _ = channel.sources[0]
            destination_actor, _ = channel.destinations[0]
            source = endpoints[source_actor, name, "set"]
            destination = endpoints[destination_actor, name, "wait"]
            if channel.ids:
                key = source_actor, source.task.pipe, destination.task.pipe
                reserved = (
                    {0}
                    if key[1:] == ("S", "FIX") and loop.actors[source_actor].program.config
                    else set()
                )
                require(
                    not (banks[key] | reserved) & set(channel.ids),
                    f"{key}: overlapping local event assignments",
                )
                banks[key].update(channel.ids)
                require(len(banks[key] | reserved) <= 8, "local event bank exhausted")
            width = len(channel.ids)
            require(width > 0, "local event width must be assigned")
            for residue in range(graph.period):
                if residue % channel.period not in channel.residues:
                    continue
                following = _successor(channel, residue, width, graph.period)
                require(
                    graph.path(destination, residue + channel.distance, source, following)
                    is not None,
                    f"{name}: binary event re-Set can precede its old Wait",
                )
                local_queries += 1
        elif channel.kind == "cross" and counters:
            if channel.ids:
                require(
                    len(channel.ids) == 1
                    and 0 <= channel.ids[0] < 8
                    and channel.ids[0] not in cross_ids,
                    "cross counter IDs overlap/exceed dav-2201 mode-2 bank",
                )
                cross_ids.add(channel.ids[0])
    if counters:
        for name, query in _counter_queries(graph.period, channel_map, endpoints):
            require(
                graph.path(*query) is not None,
                f"{name}: mode-2 notification backlog may exceed 15",
            )
    return banks, cross_ids, local_queries


def verify_loop(module, *, _details=False):
    loop = extract_loop(module)
    obligations = loop_dependencies(loop)
    channel_map = channels(loop)
    graph, sequence, endpoints, native = completion_graph(loop, channel_map)
    stride, potential = graph.potential()
    queries = _coverage(loop, obligations, graph, native)
    scalar_queries = 0
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            if task.scalar_source is None:
                continue
            source, destination = native[actor, task.scalar_source], native[actor, task.uid]
            for residue in range(graph.period):
                require(
                    graph.path(source, residue, destination, residue) is not None,
                    "Scalar value lacks an actual capture-to-dispatch completion path",
                )
                scalar_queries += 1
    banks, cross_ids, lifetime_queries = _lifetime(loop, channel_map, graph, endpoints)
    report = {
        "kernel": loop.op.sym_name.data,
        "groups": loop.groups,
        "runtime_length": None
        if loop.bound is None
        else loop.size.sym_name.data
        if loop.size
        else loop.bound.text(),
        "runtime_parameters": [size.sym_name.data for size in loop.parameters],
        "static_length": loop.constant_count,
        "native_templates": sum(len(actor.program.tasks) for actor in loop.actors.values()),
        "storage_periods": loop.storage_periods,
        "native_accesses": sum(
            len(task.refs) for actor in loop.actors.values() for task in actor.program.tasks
        ),
        "storage_residue_accesses": sum(
            loop.storage_periods[loop.root(actor, ref)]
            for actor, data in loop.actors.items()
            for task in data.program.tasks
            for ref in task.refs
        ),
        "protocol_period": graph.period,
        "physical_dependence_templates": len(obligations),
        "completion_node_families": len(graph.nodes),
        "completion_arc_families": len(graph.edges),
        "proof_domains": len(graph.domains),
        "proof_domain_states": graph.domain_states,
        "proof_domain_edges": graph.domain_edges,
        "proof_path_queries": graph.path_queries,
        "coverage_queries": queries,
        "scalar_capture_queries": scalar_queries,
        "binary_lifetime_queries": lifetime_queries,
        "channels": len(channel_map),
        "event_ids": {":".join(key): sorted(values) for key, values in banks.items()},
        "cross_counter_ids": sorted(cross_ids),
        "verification": "parametric periodic physical frontier, guarded translation paths, and integer completion potential; no runtime-length expansion",
        "input_contract": "uniform nonnegative by-value scalars; declared GM extents and disjoint aligned bases; full tiles or verified aligned final prefixes",
        "terminal_drain": "notifications only for real future consumers; each actor drains issued pipes at return",
        "valid_prefixes": sorted(
            {
                (ref.prefix_extent.text(), ref.prefix_elements)
                for actor in loop.actors.values()
                for task in actor.program.tasks
                for ref in task.refs
                if ref.prefix_extent is not None
            }
        ),
    }
    analysis = loop, channel_map, sequence, graph, (stride, potential)
    return (report, analysis) if _details else report


def explain_loop(module):
    report, (loop, _, _, graph, _) = verify_loop(module, _details=True)
    native = {
        (node.command.actor, node.command.task.uid): node.command
        for node in graph.nodes
        if node.command.kind == "task"
    }
    witnesses = []
    for o in loop_dependencies(loop):
        for query in _obligation_queries(o, graph, native):
            _, _, _, residue, first = query
            path = graph.path(*query)
            witnesses.append(
                {
                    "source": "/".join(o.source),
                    "destination": "/".join(o.destination),
                    "hazard": o.hazard,
                    "storage": o.storage,
                    "bytes": [o.start, o.stop],
                    "distance": o.distance,
                    "period": o.period,
                    "storage_residues": list(o.residues),
                    "protocol_period": graph.period,
                    "destination_residue": residue,
                    "first_generation": first,
                    "required_milestone": "read_retired" if o.hazard == "WAR" else "write_done",
                    "completion_arc_families": list(path),
                }
            )
    return {
        "verification": report,
        "physical_obligations": witnesses,
        "scope": "guarded periodic translation paths under declared affine input contracts",
    }


def _sync(stream, action, channel, lower, upper, period, residues):
    return LoopSyncOp.create(
        operands=[stream.body.block.args[0]],
        attributes={
            "action": StringAttr(action),
            "channel": StringAttr(channel),
            "lower": IntAttr(lower),
            "upper_gap": IntAttr(upper),
            "period": IntAttr(period),
            "residues": _ints(residues),
        },
    )


def _install(loop, specs, barriers):
    operations = []
    for actor, uid, lower, family in barriers:
        data = loop.actors[actor]
        period, chosen = family
        op = _sync(data.stream, "barrier", "", lower, 0, period, chosen)
        data.stream.body.block.insert_op_before(op, data.operations[uid])
        operations.append((None, [op]))
    for number, (kind, sources, destinations, distance, family) in enumerate(specs):
        period, chosen = family
        name = f"loop_c{number}"
        op = LoopChannelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "kind": StringAttr(kind),
                "sources": ArrayAttr(StringAttr("/".join(source)) for source in sources),
                "destinations": ArrayAttr(
                    StringAttr("/".join(destination)) for destination in destinations
                ),
                "distance": IntAttr(distance),
                "period": IntAttr(period),
                "residues": _ints(chosen),
                "event_ids": _ints(()),
            }
        )
        first_region = next(
            child for child in loop.op.body.block.ops if isinstance(child, (StreamOp, KernelOp))
        )
        loop.op.body.block.insert_op_before(op, first_region)
        placed = []
        for actor, uid in sources:
            data = loop.actors[actor]
            sync = _sync(data.stream, "set", name, 0, distance, period, chosen)
            data.stream.body.block.insert_op_after(sync, data.operations[uid])
            placed.append(sync)
        shifted = tuple(sorted((r + distance) % period for r in chosen))
        for actor, uid in destinations:
            data = loop.actors[actor]
            sync = _sync(data.stream, "wait", name, distance, 0, period, shifted)
            data.stream.body.block.insert_op_before(sync, data.operations[uid])
            placed.append(sync)
        operations.append((op, placed))
    return operations


def _proposals(loop, obligations):
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    local, barrier_map, cross = {}, {}, {}
    for o in obligations:
        a, b = tasks[o.source], tasks[o.destination]
        if o.source[0] != o.destination[0]:
            # Acquire at this actual consumer, joining only its overlapping
            # contributors. A different native subview gets its own gate.
            key = o.storage, o.distance, o.hazard, o.destination[1]
            item = cross.setdefault(key, ({}, {}, {}))
            source_actor, destination_actor = o.source[0], o.destination[0]
            old_source, old_destination = item[0].get(source_actor), item[1].get(destination_actor)
            if old_source is None or tasks[old_source].issue_order < a.issue_order:
                item[0][source_actor] = o.source
            if old_destination is None or tasks[old_destination].issue_order > b.issue_order:
                item[1][destination_actor] = o.destination
            item[2].setdefault(o.period, set()).update(
                (r - o.distance) % o.period for r in o.residues
            )
        elif a.pipe == b.pipe:
            item = barrier_map.setdefault(o.destination, [o.distance, {}])
            item[0] = min(item[0], o.distance)
            item[1].setdefault(o.period, set()).update(o.residues)
        else:
            local.setdefault((o.source, o.destination, o.distance), {}).setdefault(
                o.period, set()
            ).update((r - o.distance) % o.period for r in o.residues)
    specs = [
        (
            "local",
            (source,),
            (destination,),
            distance,
            union(residues, f"channel {source}->{destination}"),
        )
        for (source, destination, distance), residues in sorted(local.items())
    ]
    specs += [
        (
            "cross",
            tuple(sorted(sources.values())),
            tuple(sorted(destinations.values())),
            distance,
            union(residues, "cross-core channel"),
        )
        for (_, distance, _, _), (sources, destinations, residues) in sorted(cross.items())
    ]
    barriers = [
        (actor, uid, lower, union(residues, f"barrier {actor}/{uid}"))
        for (actor, uid), (lower, residues) in sorted(barrier_map.items())
    ]
    return specs, barriers


def autosync_loop(module, *, _capture=None):
    from .sync import _finish

    result = module.clone()
    loop = extract_loop(result)
    obligations = loop_dependencies(loop)
    require(
        not any(isinstance(op, (LoopChannelOp, LoopSyncOp)) for op in result.walk()),
        "AutoSync expects an unsynchronized symbolic loop",
    )
    specs, barriers = _proposals(loop, obligations)
    installed = _install(loop, specs, barriers)

    def check():
        channel_map = channels(loop, virtual=True)
        graph, _, endpoints, native = completion_graph(loop, channel_map)
        graph.potential()
        _coverage(loop, obligations, graph, native)
        _lifetime(loop, channel_map, graph, endpoints, local=False)
        return graph, channel_map, endpoints, native

    graph, channel_map, endpoints, native = check()
    search = _ProofSearch(loop, obligations, graph, native, channel_map, endpoints)
    # Whole-group paths can retire local storage as well. Every deletion is
    # checked over the same symbolic query domains, including boundary guards.
    for channel, operations in reversed(installed):
        name = None if channel is None else channel.sym_name.data
        if search.remove(operations, name):
            for op in ([channel] if channel is not None else []) + operations:
                op.parent.erase_op(op)
    graph, channel_map, endpoints, _ = check()
    banks, cross_id = defaultdict(int), 0
    for name, channel in channel_map.items():
        op = next(
            op for op in result.walk() if isinstance(op, LoopChannelOp) and op.sym_name.data == name
        )
        if channel.kind == "cross":
            require(cross_id < 8, "symbolic mode-2 counter bank exhausted")
            op.attributes["event_ids"] = _ints([cross_id])
            cross_id += 1
            continue
        actor = channel.sources[0][0]
        source, destination = endpoints[actor, name, "set"], endpoints[actor, name, "wait"]
        width = None
        for candidate in range(1, 9):
            if all(
                graph.path(
                    destination,
                    r + channel.distance,
                    source,
                    _successor(channel, r, candidate, graph.period),
                )
                is not None
                for r in range(graph.period)
                if r % channel.period in channel.residues
            ):
                width = candidate
                break
        require(width is not None, f"{name}: no legal binary event reuse width <=8")
        key = actor, source.task.pipe, destination.task.pipe
        reserved = 1 if key[1:] == ("S", "FIX") and loop.actors[actor].program.config else 0
        base = max(banks[key], reserved)
        require(
            base + width <= 8, f"{key}: symbolic event bank exhausted; no serialization fallback"
        )
        op.attributes["event_ids"] = _ints(range(base, base + width))
        banks[key] = base + width
    return _finish(result, _capture=_capture)

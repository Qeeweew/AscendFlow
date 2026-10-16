"""Automatic synchronization and independent parametric loop checking."""

from collections import defaultdict
from dataclasses import replace
from math import lcm

from xdsl.dialects import scf
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import bounded
from .diagnostics import CompileError, require
from .dialects.physical import (
    InvocationChannelOp,
    InvocationSyncOp,
    KernelOp,
    LoopChannelOp,
    LoopSyncOp,
    StreamOp,
)
from .loop_graph import channels, commands, completion_graph
from .loop_model import extract_loop, loop_dependencies
from .loop_planning import DirectCompletion, Protocol
from .periodic import bounded_period, project, union


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _obligation_queries(obligation, graph, native):
    o = obligation
    for residue, first in project(o.period, o.residues, graph.period, o.distance):
        yield (
            native[o.source],
            residue - o.distance,
            native[o.destination],
            residue,
            first,
            o.prefix,
        )


def _coverage_queries(obligations, graph, native):
    # Byte cells with the same endpoint relation share one proof. The earliest
    # real occurrence is the strongest guard domain; later cells need no new
    # graph/domain. Keep a real obligation for a concrete failure diagnostic.
    queries = {}
    for o in sorted(obligations, key=lambda o: o.hazard != "WAR"):
        for query in _obligation_queries(o, graph, native):
            source, _, destination, residue, first, prefix = query
            key = source.index, destination.index, o.distance, residue, prefix
            if key not in queries or first < queries[key][1][4]:
                queries[key] = o, query
    return queries.values()


def _coverage(loop, obligations, graph, native):
    queries = tuple(_coverage_queries(obligations, graph, native))
    covered = graph.covered_many(query for _, query in queries)
    for (o, query), valid in zip(queries, covered):
        _, _, _, residue, first, prefix = query
        if not valid:
            raise CompileError(
                f"missing loop {o.hazard} {'read_retired_prefix' if prefix else 'read_retired' if o.hazard == 'WAR' else 'write_done'}: "
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


def _lifetime(loop, channel_map, graph, endpoints, *, local=True, counters=True):
    banks, cross_ids = defaultdict(set), set()
    local_checks = []
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
            if channel.last:
                require(width == 1, "terminal local channel executes at most once per invocation")
                continue
            for residue in range(graph.period):
                if residue % channel.period not in channel.residues:
                    continue
                following = _successor(channel, residue, width, graph.period)
                local_checks.append(
                    (name, (destination, residue + channel.distance, source, following))
                )
        elif channel.kind == "cross" and counters:
            if channel.ids:
                require(
                    len(channel.ids) == 1
                    and 0 <= channel.ids[0] < 8
                    and channel.ids[0] not in cross_ids,
                    "cross counter IDs overlap/exceed dav-2201 mode-2 bank",
                )
                cross_ids.add(channel.ids[0])
    for (name, _), valid in zip(local_checks, graph.covered_many(q for _, q in local_checks)):
        require(valid, f"{name}: binary event re-Set can precede its old Wait")
    if counters:
        checks = tuple(_counter_queries(graph.period, channel_map, endpoints))
        for (name, _), valid in zip(checks, graph.covered_many(q for _, q in checks)):
            require(
                valid,
                f"{name}: mode-2 notification backlog may exceed 15",
            )
    return banks, cross_ids, len(local_checks)


@bounded
def verify_loop(module, *, _details=False):
    loop = extract_loop(module)
    require(
        loop.invocation is not None
        or not any(isinstance(op, (InvocationChannelOp, InvocationSyncOp)) for op in module.walk()),
        "invocation synchronization requires its actual parent scope",
    )
    return _verify_leaf(loop, _details=_details)


def _verify_leaf(loop, *, live_in=(), _details=False):
    """Conditional leaf proof; its parent must discharge every live-in byte.

    This is not a public successful-kernel verifier. Only verify_loop with no
    live-ins, or the complete sequence verifier, grants that result.
    """
    obligations = loop_dependencies(loop, live_in=live_in)
    channel_map = channels(loop)
    graph, sequence, endpoints, native = completion_graph(loop, channel_map)
    stride, potential = graph.potential()
    queries = _coverage(loop, obligations, graph, native)
    scalar_queries = []
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            if task.scalar_source is None:
                continue
            source, destination = native[actor, task.scalar_source], native[actor, task.uid]
            for residue in range(graph.period):
                scalar_queries.append((source, residue, destination, residue))
    require(
        all(graph.covered_many(scalar_queries)),
        "Scalar value lacks an actual capture-to-dispatch completion path",
    )
    banks, cross_ids, lifetime_queries = _lifetime(loop, channel_map, graph, endpoints)
    result = _leaf_result(
        loop,
        obligations,
        channel_map,
        graph,
        sequence,
        (stride, potential),
        live_in=live_in,
        queries=queries,
        scalar_queries=scalar_queries,
        banks=banks,
        cross_ids=cross_ids,
        lifetime_queries=lifetime_queries,
    )
    return result if _details else result[0]


def _leaf_result(
    loop,
    obligations,
    channel_map,
    graph,
    sequence,
    potential,
    *,
    live_in,
    queries,
    scalar_queries,
    banks,
    cross_ids,
    lifetime_queries,
    invocation_plan=None,
):
    """Describe facts already established by a planner or the import checker."""
    stride, potential = potential
    native = {
        (command.actor, command.task.uid): command
        for stream in sequence.values()
        for command in stream
        if command.kind == "task"
    }
    invocation_report = {}
    if loop.invocation is not None and loop.region is None:
        from .invocation_sync import verify_invocation

        invocation_report = (
            verify_invocation(loop, channel_map, graph, sequence)
            if invocation_plan is None
            else invocation_plan.complete(loop, channel_map, graph, sequence)
        )
    report = {
        **invocation_report,
        "kernel": loop.op.sym_name.data,
        "groups": loop.groups,
        "runtime_length": None
        if loop.bound is None
        else loop.size.sym_name.data
        if loop.size
        else loop.bound.text(),
        "runtime_parameters": [size.sym_name.data for size in loop.parameters],
        "static_length": loop.constant_count,
        "terminal_templates": sum(
            task.last for data in loop.actors.values() for task in data.program.tasks
        ),
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
        "completion_statistics": graph.completion_statistics(),
        "proof_path_queries": graph.path_queries,
        "completion_batch_queries": graph.batch_queries,
        "completion_transfer_passes": graph.batch_passes,
        "completion_fact_transfers": graph.fact_transfers,
        "coverage_queries": queries,
        "scalar_capture_queries": len(scalar_queries),
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
    if live_in:
        report["open_entry_roots"] = sorted(live_in)
    for protocol in loop.op.body.block.ops:
        if isinstance(protocol, InvocationChannelOp):
            source = native[tuple(protocol.source.data.split("/"))]
            destination = native[tuple(protocol.destination.data.split("/"))]
            bank = f"{source.actor}:{source.task.pipe}:{destination.task.pipe}"
            report["event_ids"].setdefault(bank, []).append(protocol.event_id.data)
            report["event_ids"][bank].sort()
    if loop.invocation is not None and loop.region is None:
        report["invocation_channels"] = sum(
            isinstance(op, InvocationChannelOp) for op in loop.op.walk()
        )
    analysis = loop, channel_map, sequence, graph, (stride, potential)
    return report, analysis


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
            _, _, _, residue, first, prefix = query
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
                    "terminal": o.last,
                    "source_completion_prefix": prefix,
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


def _materialize(loop, protocols):
    """Only selected protocols acquire xDSL operations and SSA uses."""
    scope = loop.protocol_scope.body.block
    first = (
        scope.first_op
        if loop.region is not None
        else next(op for op in scope.ops if isinstance(op, (StreamOp, KernelOp, scf.ForOp)))
    )
    for protocol in protocols:
        p = protocol
        if p.kind != "barrier":
            declaration = LoopChannelOp.create(
                attributes={
                    "sym_name": StringAttr(p.name),
                    "kind": StringAttr(p.kind),
                    "sources": ArrayAttr(StringAttr("/".join(x)) for x in p.sources),
                    "destinations": ArrayAttr(StringAttr("/".join(x)) for x in p.destinations),
                    "distance": IntAttr(p.distance),
                    "period": IntAttr(p.period),
                    "residues": _ints(p.residues),
                    "event_ids": _ints(()),
                    **({"last": IntAttr(1)} if p.last else {}),
                }
            )
            scope.insert_op_before(declaration, first)
        placements = (
            [(owner, "barrier") for owner in p.destinations]
            if p.kind == "barrier"
            else [(owner, "set") for owner in p.sources]
            + [(owner, "wait") for owner in p.destinations]
        )
        for (actor, uid), action in placements:
            residues = (
                p.residues
                if action != "wait"
                else tuple(sorted((r + p.distance) % p.period for r in p.residues))
            )
            lower, upper = (0, p.distance) if action == "set" else (p.distance, 0)
            op = _sync(loop.actors[actor].stream, action, p.name, lower, upper, p.period, residues)
            if p.last:
                op.attributes["last_gap"] = IntAttr(upper)
            target = loop.actors[actor].operations[uid]
            if action == "set":
                # commands() reads this after-list in reverse, retaining the
                # selected issuance order in both the proof and ASC emitter.
                target.parent.insert_op_after(op, target)
            else:
                target.parent.insert_op_before(op, target)


def _requirements(loop, obligations):
    """Group accesses that require the same physical capture/acquire protocol."""
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    local, barriers, cross = {}, {}, {}
    for o in obligations:
        a, b = tasks[o.source], tasks[o.destination]
        if o.source[0] != o.destination[0]:
            require(not o.last, "terminal cross-core protocols require collective region analysis")
            key = o.storage, o.distance, o.hazard, o.destination[1]
            sources, destinations, residues, members = cross.setdefault(key, ({}, {}, {}, []))
            for table, actor, endpoint, later in (
                (sources, o.source[0], o.source, True),
                (destinations, o.destination[0], o.destination, False),
            ):
                previous = table.get(actor)
                if previous is None or (
                    tasks[previous].issue_order < tasks[endpoint].issue_order
                    if later
                    else tasks[previous].issue_order > tasks[endpoint].issue_order
                ):
                    table[actor] = endpoint
            residues.setdefault(o.period, set()).update(
                (r - o.distance) % o.period for r in o.residues
            )
            members.append(o)
        elif a.pipe == b.pipe:
            members = barriers.setdefault(o.destination, [])
            members.append(o)
        else:
            local.setdefault((o.source, o.destination, o.distance, o.last), []).append(o)

    def family(members, *, capture):
        residues = {}
        for o in members:
            residues.setdefault(o.period, set()).update(
                (r - o.distance if capture else r) % o.period for r in o.residues
            )
        return union(residues, "required synchronization activation")

    result = []
    for destination, members in sorted(barriers.items()):
        period, residues = family(members, capture=False)
        result.append(
            (
                Protocol(
                    "",
                    "barrier",
                    (),
                    (destination,),
                    min(o.distance for o in members),
                    period,
                    residues,
                ),
                tuple(members),
            )
        )
    number = 0
    for (source, destination, distance, terminal), members in sorted(local.items()):
        period, residues = family(members, capture=True)
        result.append(
            (
                Protocol(
                    f"loop_c{number}",
                    "local",
                    (source,),
                    (destination,),
                    distance,
                    period,
                    residues,
                    terminal,
                ),
                tuple(members),
            )
        )
        number += 1
    for (_, distance, _, _), (sources, destinations, residues, members) in sorted(cross.items()):
        period, residues = union(residues, "cross-core required activation")
        result.append(
            (
                Protocol(
                    f"loop_c{number}",
                    "cross",
                    tuple(sorted(sources.values())),
                    tuple(sorted(destinations.values())),
                    distance,
                    period,
                    residues,
                ),
                tuple(members),
            )
        )
        number += 1
    return result


def establishes_directly(protocol, obligation):
    """The capture/acquire constructor's exact-endpoint invariant.

    A Set after the named native captures its retirement prefix, and its
    matching Wait admits the named consumer. This proves end AND prefix
    obligations in the protocol's declared activation. Coalesced different
    endpoints and same-pipe fences still need their separate inclusion fact.
    """
    if (
        protocol.kind not in {"local", "cross"}
        or obligation.source not in protocol.sources
        or obligation.destination not in protocol.destinations
        or obligation.distance != protocol.distance
        or getattr(obligation, "last", False) != protocol.last
    ):
        return False
    activation = getattr(obligation, "activation", None)
    if activation is not None:
        return activation == getattr(protocol, "activation", None)
    return (
        obligation.period % protocol.period == 0
        and all(
            (r - obligation.distance) % protocol.period in protocol.residues
            for r in obligation.residues
        )
        and getattr(protocol, "activation", None) is None
    )


def autosync_loop(module, *, _capture=None, _facts=None):
    result = module.clone() if _facts is None else module
    loop = extract_loop(result) if _facts is None else _facts.extract_loop(result)
    report, analysis = _plan_leaf(loop, _facts=_facts)
    if _capture is not None:
        _capture((report, analysis, 0.0))
    return result


def _plan_leaf(loop, *, live_in=(), banks=None, _facts=None):
    """Plan an owned leaf under explicit, still-open entry assumptions."""
    obligations = (
        loop_dependencies(loop, live_in=live_in)
        if _facts is None
        else _facts.loop_dependencies(loop, live_in=live_in)
    )
    require(
        not any(isinstance(op, (LoopChannelOp, LoopSyncOp)) for op in loop.protocol_scope.walk()),
        "AutoSync expects an unsynchronized symbolic loop",
    )
    requirements = _requirements(loop, obligations)
    period = 1
    for protocol, _ in requirements:
        period = bounded_period(lcm(period, protocol.period), "required synchronization predicates")
    graph = DirectCompletion(period, loop.constant_count)
    graph, _, _, native = completion_graph(loop, {}, period=period, _planning=graph)
    groups = [(p, tuple(_coverage_queries(members, graph, native))) for p, members in requirements]
    graph.register(q for _, memory in groups for _, q in memory)

    gated = {
        destination
        for protocol, _ in groups
        for destination in protocol.destinations
        if protocol.kind == "cross"
        or (protocol.kind == "local" and native[destination].task.pipe == "S")
    }

    def order(item):
        protocol, memory = item
        carried = all(o.distance > 0 for o, _ in memory)
        destination = min((q[2].task.stage, q[2].index) for _, q in memory)
        # Body acquisitions precede backedges. At a dispatch gate retain the
        # consumer's own native fence before its blocking acquire; elsewhere
        # external acquisitions may already retire its previous native work.
        fence_first = not carried and any(d in gated for d in protocol.destinations)
        kind = (
            (0 if fence_first else 3)
            if protocol.kind == "barrier"
            else (2 if protocol.kind == "cross" else 1)
        )
        source = min((o.distance, -q[0].index) for o, q in memory)
        return carried, destination, kind, source

    for protocol, memory in sorted(groups, key=order):
        missing = [(o, q) for o, q in memory if not graph.covered(q)]
        if not missing:
            continue
        graph.insert(protocol, native)
        for o, q in missing:
            require(
                establishes_directly(protocol, o) or graph.covered(q),
                f"direct {protocol.kind} capture cannot cover {o.hazard}: "
                f"{o.source} -> {o.destination}",
            )
    channel_map = {p.name: p for p in graph.selected if p.kind != "barrier"}
    counters = tuple(_counter_queries(period, channel_map, graph.endpoints))
    graph.register(q for _, q in counters)
    for name, query in counters:
        require(
            graph.covered(query),
            f"{name}: selected protocols do not bound mode-2 notification backlog to 15; "
            "an explicit consumption milestone protocol is required",
        )
    _materialize(loop, graph.selected)
    channel_map = channels(loop, virtual=True)
    graph, sequence, endpoints, native = completion_graph(loop, channel_map)
    potential = graph.potential()
    # _materialize preserves each selected capture/acquire, clock and guard.
    # Memory coverage was established by direct synthesis; ID allocation does
    # not change it. The independently reconstructed checker remains public.
    queries = len(tuple(_coverage_queries(obligations, graph, native)))
    scalar_queries = []
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            if task.scalar_source is not None:
                scalar_queries.extend(
                    (native[actor, task.scalar_source], r, native[actor, task.uid], r)
                    for r in range(graph.period)
                )
    require(
        all(graph.covered_many(scalar_queries)),
        "Scalar value lacks an actual capture-to-dispatch completion path",
    )
    banks, cross_id = (defaultdict(int) if banks is None else banks), 0
    used, lifetime_queries = defaultdict(set), 0
    for name, channel in channel_map.items():
        op = next(
            op
            for op in loop.protocol_scope.walk()
            if isinstance(op, LoopChannelOp) and op.sym_name.data == name
        )
        if channel.kind == "cross":
            require(cross_id < 8, "symbolic mode-2 counter bank exhausted")
            op.attributes["event_ids"] = _ints([cross_id])
            cross_id += 1
            continue
        actor = channel.sources[0][0]
        source, destination = endpoints[actor, name, "set"], endpoints[actor, name, "wait"]
        width = 1 if channel.last else None
        for candidate in () if channel.last else range(1, 9):
            if all(
                graph.covered_many(
                    (
                        destination,
                        r + channel.distance,
                        source,
                        _successor(channel, r, candidate, graph.period),
                    )
                    for r in range(graph.period)
                    if r % channel.period in channel.residues
                )
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
        used[key].update(range(base, base + width))
        if not channel.last:
            lifetime_queries += sum(
                r % channel.period in channel.residues for r in range(graph.period)
            )
    invocation_plan = None
    if loop.invocation is not None and loop.region is None:
        from .invocation_sync import plan_invocation

        invocation_plan = plan_invocation(
            loop, channel_map=channels(loop), graph=graph, streams=sequence
        )

    channel_map = channels(loop)
    sequence, _, _ = commands(loop, channel_map)
    return _leaf_result(
        loop,
        obligations,
        channel_map,
        graph,
        sequence,
        potential,
        live_in=live_in,
        queries=queries,
        scalar_queries=scalar_queries,
        banks=used,
        cross_ids=set(range(cross_id)),
        lifetime_queries=lifetime_queries,
        invocation_plan=invocation_plan,
    )


def rename_leaf_plan(result, renaming):
    """Apply a channel-name/ID substitution to an owned, established leaf.

    Variant merging changes neither command indices, guards nor native cuts.
    Completion topology and its propagated facts therefore remain valid.
    """
    report, (loop, old_channels, sequence, graph, potential) = result
    require(set(renaming) == set(old_channels), "leaf renaming must cover its channels exactly")
    for op in loop.protocol_scope.walk():
        if isinstance(op, LoopChannelOp):
            name, ids = renaming[op.sym_name.data]
            op.attributes.update(sym_name=StringAttr(name), event_ids=_ints(ids))
        elif isinstance(op, LoopSyncOp) and op.action.data != "barrier":
            op.attributes["channel"] = StringAttr(renaming[op.channel.data][0])
    current = channels(loop)
    require(
        len(current) == len(old_channels)
        and all(
            replace(old, name=renaming[name][0], ids=renaming[name][1])
            == current[renaming[name][0]]
            for name, old in old_channels.items()
        ),
        "leaf resource substitution changed notification semantics",
    )
    replacements = {
        command.index: replace(command, channel=renaming[command.channel][0])
        for stream in sequence.values()
        for command in stream
        if command.channel in renaming
    }
    graph.nodes[:] = [
        replace(node, command=replacements.get(node.command.index, node.command))
        for node in graph.nodes
    ]
    sequence = {
        actor: tuple(replacements.get(c.index, c) for c in stream)
        for actor, stream in sequence.items()
    }
    used = defaultdict(set)
    tasks = {(a, t.uid): t for a, d in loop.actors.items() for t in d.program.tasks}
    for channel in current.values():
        a, b = channel.sources[0], channel.destinations[0]
        used[f"{a[0]}:{tasks[a].pipe}:{tasks[b].pipe}"].update(channel.ids)
    report = {**report, "event_ids": {bank: sorted(ids) for bank, ids in used.items()}}
    return report, (loop, current, sequence, graph, potential)

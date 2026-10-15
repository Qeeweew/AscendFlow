"""Compose local child completion with explicit cross-invocation protocols.

The leaf proof covers internal work. Tail Sets never fence their source pipe;
head Waits only depend on a previous invocation. Hence induction on serial
invocations supplies progress, while the checks below separately cover memory
and event reuse. No finite parent/child expansion participates in this proof.
"""

from dataclasses import dataclass, replace
from math import lcm

from xdsl.dialects.builtin import IntAttr, StringAttr

from .completion import LeafCompletion, Site, _first, canonical, covered_path
from .ir import InvocationChannelOp, InvocationSyncOp, LoopChannelOp, TaskOp
from .loop_boundary import stream_operations
from .model import require
from .periodic import MAX_PERIOD_TEMPLATES
from .region_summary import (
    Access,
    CountDomain,
    Position,
    ReadSpan,
    summarize_local_events,
    summarize_reused_storage,
)


@dataclass(frozen=True)
class Bridge:
    name: str
    event_id: int
    source: Site
    destination: Site
    domain: CountDomain
    carried: bool


def _protocols(loop, channel_map):
    bridges, declarations = [], {}
    for channel in channel_map.values():
        if not channel.carry:
            continue
        width = channel.distance
        for slot in range(width):
            for residue in range(width):
                gap = (residue - 1 - slot) % width
                bridges.append(
                    Bridge(
                        channel.name,
                        channel.ids[slot],
                        Site(channel.sources[0], Position(True, gap), "prefix"),
                        Site(channel.destinations[0], Position(False, slot), "begin"),
                        CountDomain(width, residue, minimum=slot + 1),
                        True,
                    )
                )
    tasks = {(a, t.uid): t for a, d in loop.actors.items() for t in d.program.tasks}
    used = {}
    for c in channel_map.values():
        bank = c.sources[0][0], tasks[c.sources[0]].pipe, tasks[c.destinations[0]].pipe
        used.setdefault(bank, set()).update(c.ids)
    for actor, data in loop.actors.items():
        if data.program.config:
            used.setdefault((actor, "S", "FIX"), set()).add(0)
    for op in loop.op.body.block.ops:
        if not isinstance(op, InvocationChannelOp):
            continue
        name = op.sym_name.data
        source, destination = (
            tuple(op.source.data.split("/")),
            tuple(op.destination.data.split("/")),
        )
        require(
            name not in channel_map and name not in declarations and name.isidentifier(),
            "duplicate/invalid invocation channel",
        )
        require(
            source in tasks and destination in tasks and source[0] == destination[0],
            "invocation channel endpoints must belong to its local actor",
        )
        gap, index = op.source_gap.data, op.destination_index.data
        require(
            0 <= gap <= 840
            and 0 <= index <= 840
            and (not tasks[source].last or gap == 0)
            and not tasks[destination].last,
            "invocation channel requires a real tail and a nonterminal head",
        )
        bank = source[0], tasks[source].pipe, tasks[destination].pipe
        require(bank[1] != bank[2], "same-pipe invocation completion needs a proved pipe fence")
        event_id = op.event_id.data
        require(
            0 <= event_id < 8 and event_id not in used.setdefault(bank, set()),
            "invocation event overlaps a live/reserved bank ID",
        )
        used[bank].add(event_id)
        declarations[name] = op
        bridges.append(
            Bridge(
                name,
                event_id,
                Site(source, Position(True, gap), "prefix"),
                Site(destination, Position(False, index), "begin"),
                CountDomain(1, 0, minimum=max(gap, index) + 1),
                False,
            )
        )

    actual = {}
    for actor, data in loop.actors.items():
        pending, previous, parent = [], None, None
        for op, _ in stream_operations(data.stream):
            if op.parent is not parent:
                require(not pending, "invocation Wait escaped its physical region")
                previous, parent = None, op.parent
            if isinstance(op, TaskOp):
                for wait in pending:
                    actual[wait.channel.data, "wait"] = actor, op.uid.data
                pending, previous = [], (actor, op.uid.data)
            elif isinstance(op, InvocationSyncOp):
                require(
                    op.generation is data.stream.body.block.args[0]
                    and op.channel.data in declarations
                    and op.action.data in {"set", "wait"},
                    "invocation synchronization uses a wrong scope/channel/action",
                )
                key = op.channel.data, op.action.data
                require(
                    key not in actual and all(w.channel != op.channel for w in pending),
                    "duplicate invocation synchronization endpoint",
                )
                if op.action.data == "set":
                    require(previous is not None, "invocation Set must follow a native source")
                    actual[key] = previous
                else:
                    pending.append(op)
        require(not pending, "unattached invocation Wait")
    for bridge in bridges:
        if not bridge.carried:
            require(
                actual.get((bridge.name, "set")) == bridge.source.task
                and actual.get((bridge.name, "wait")) == bridge.destination.task,
                "invocation protocol endpoints do not match actual synchronization sites",
            )
    return tuple(bridges)


def _site(access, *, source):
    if isinstance(access, Access):
        return Site(access.task, access.position, "end" if source else "begin")
    return Site(
        access.task,
        access.last if source else access.first,
        "prefix" if source and access.first != access.last else "end" if source else "begin",
    )


def _domains(domain, period, maximum):
    if domain.exact is not None:
        return (domain,) if maximum is None or domain.exact == maximum else ()
    joint = lcm(domain.period, period)
    require(joint <= MAX_PERIOD_TEMPLATES, "coupled invocation predicate budget exceeded")
    return tuple(
        d
        for r in range(domain.residue, joint, domain.period)
        if (d := CountDomain(joint, r, minimum=_first(domain), maximum=domain.maximum))
        and (maximum is None or d.contains(maximum))
    )


def _covered(left, right, domain, groups, inside):
    if inside.loop.constant_count is not None and not domain.contains(inside.loop.constant_count):
        return True
    # Adjacent calls remain distinct identities. Only actual carried protocols
    # connect them; lexical position does not supply completion.
    right = replace(right, call=1)
    edges, nodes = {}, {left, right}
    for group in groups.values():
        for bridge in group:
            source, destination = bridge.source, replace(bridge.destination, call=1)
            edges.setdefault(source, []).append((destination, canonical(bridge.domain)))
            nodes.update((source, destination))
    nodes = tuple(sorted(nodes, key=repr))

    def successors(site, guard):
        for candidate in nodes:
            if candidate != site and candidate.call == site.call:
                for source_guard in inside.activation(site, guard):
                    for predicate in inside.activation(candidate, source_guard):
                        if inside.prove(site, candidate, predicate):
                            yield candidate, predicate

    return covered_path(
        left, right, domain, edges, successors, lambda site, guard: inside.prove(site, site, guard)
    )


def verify_invocation(loop, channel_map, graph, endpoints, native):
    require(
        loop.invocation is not None and len(loop.actors) == 1,
        "invocation verification needs one retained local child",
    )
    bridges = _protocols(loop, channel_map)
    groups = {}
    for bridge in bridges:
        groups.setdefault(bridge.name, []).append(bridge)
    roots = {
        loop.root(a, r)
        for a, d in loop.actors.items()
        for t in d.program.tasks
        for r in t.refs
        if not r.storage.external
    }
    summary = summarize_reused_storage(loop, roots)
    inside = LeafCompletion(loop, graph, native, endpoints)
    obligations = summary.invocation_dependencies()
    resources = summarize_local_events(loop)
    repeated = loop.invocation.bound.constant is None or loop.invocation.bound.constant > 1
    if repeated:
        for dep in obligations:
            if (
                isinstance(dep.destination, ReadSpan)
                and dep.destination.first != dep.destination.last
            ):
                require(
                    inside.prefix_monotonic(dep.destination.task, "begin"),
                    "child reader family lacks FIFO start order",
                )
            require(
                _covered(
                    _site(dep.source, source=True),
                    _site(dep.destination, source=False),
                    dep.domain,
                    groups,
                    inside,
                ),
                f"missing invocation {dep.hazard}: {dep.storage}[{dep.start}:{dep.stop}], "
                f"{dep.source} -> {dep.destination}, {dep.domain}",
            )
        for resource in resources:
            left = Site(
                resource.consumed.task,
                resource.consumed.position,
                "begin",
                resource.channel,
                "wait",
            )
            right = Site(
                resource.republished.task,
                resource.republished.position,
                "end",
                resource.channel,
                "set",
            )
            require(
                _covered(left, right, resource.domain, groups, inside),
                f"{resource.channel}: event {resource.event_id} can be re-Set across invocations",
            )
        for bridge in bridges:
            domain = bridge.domain
            if loop.constant_count is not None and not domain.contains(loop.constant_count):
                continue
            # Its head Wait is consumed before the next tail Set of the same
            # boundary ID, even when the child has no interior credit pair.
            require(
                inside.prove(bridge.destination, bridge.source, domain),
                f"{bridge.name}: boundary credit may be re-Set before its head Wait",
            )
            if not bridge.carried:
                continue
            # A carried ID is also used inside the child. Check both seams.
            for resource in resources:
                if (resource.channel, resource.event_id) != (bridge.name, bridge.event_id):
                    continue
                for case in _domains(resource.domain, domain.period, loop.constant_count):
                    if not domain.contains(_first(case)):
                        continue
                    left = Site(
                        resource.consumed.task,
                        resource.consumed.position,
                        "begin",
                        resource.channel,
                        "wait",
                    )
                    right = Site(
                        resource.republished.task,
                        resource.republished.position,
                        "end",
                        resource.channel,
                        "set",
                    )
                    require(
                        inside.prove(left, bridge.source, case)
                        and inside.prove(bridge.destination, right, case),
                        f"{bridge.name}: carried/interior event lifetimes overlap",
                    )
    return {
        "invocation_bound": loop.invocation.bound.text(),
        "invocation_frames": len(loop.invocation.frames),
        "invocation_memory_obligations": len(obligations),
        "invocation_event_domains": len(resources),
        "invocation_bridges": len(bridges),
        "invocation_proof_queries": inside.queries,
    }


def plan_invocation(loop):
    """Extend proven slot credits, then propose precise terminal-to-head edges."""
    from .loop_graph import channels, completion_graph

    if loop.invocation.bound.constant in {0, 1} or loop.constant_count == 0:
        return
    for op in loop.op.body.block.ops:
        if (
            isinstance(op, LoopChannelOp)
            and op.kind.data == "local"
            and op.last is None
            and op.period.data == 1
            and op.distance.data == len(op.event_ids)
            and op.distance.data > 0
        ):
            op.attributes["carry"] = IntAttr(1)
    channel_map = channels(loop)
    graph, _, endpoints, native = completion_graph(loop, channel_map)
    bridges = _protocols(loop, channel_map)
    groups = {}
    for bridge in bridges:
        groups.setdefault(bridge.name, []).append(bridge)
    roots = {
        loop.root(a, r)
        for a, d in loop.actors.items()
        for t in d.program.tasks
        for r in t.refs
        if not r.storage.external
    }
    inside = LeafCompletion(loop, graph, native, endpoints)
    summary = summarize_reused_storage(loop, roots)
    candidates = {}
    for dep in summary.invocation_dependencies():
        left, right = _site(dep.source, source=True), _site(dep.destination, source=False)
        if _covered(left, right, dep.domain, groups, inside):
            continue
        if not left.position.end and dep.domain.exact is not None:
            left = Site(
                left.task, Position(True, dep.domain.exact - 1 - left.position.offset), left.point
            )
        require(
            left.position.end and not right.position.end,
            "uncovered invocation relation needs an explicit tail-to-head protocol",
        )
        key = left.task, right.task, left.position.offset, right.position.offset
        candidates.setdefault(key, []).append((left, right, dep.domain))
    for (source, destination, gap, index), required in candidates.items():
        if all(_covered(left, right, domain, groups, inside) for left, right, domain in required):
            continue
        a, b = native[source].task, native[destination].task
        if a.pipe == b.pipe:
            # A later cross-pipe reader may supply the necessary reuse edge;
            # do not manufacture a whole-pipe fence before examining it.
            continue
        used = {
            i
            for c in channel_map.values()
            if c.sources[0][0] == source[0]
            and native[c.sources[0]].task.pipe == a.pipe
            and native[c.destinations[0]].task.pipe == b.pipe
            for i in c.ids
        }
        for old in loop.op.body.block.ops:
            if isinstance(old, InvocationChannelOp) and (
                native[tuple(old.source.data.split("/"))].task.pipe,
                native[tuple(old.destination.data.split("/"))].task.pipe,
            ) == (a.pipe, b.pipe):
                used.add(old.event_id.data)
        if a.pipe == "S" and b.pipe == "FIX" and loop.actors[source[0]].program.config:
            used.add(0)
        free = next((i for i in range(8) if i not in used), None)
        require(free is not None, f"invocation {a.pipe}->{b.pipe} event bank exhausted")
        name = f"invocation_c{len([o for o in loop.op.body.block.ops if isinstance(o, InvocationChannelOp)])}"
        op = InvocationChannelOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "source": StringAttr("/".join(source)),
                "destination": StringAttr("/".join(destination)),
                "source_gap": IntAttr(gap),
                "destination_index": IntAttr(index),
                "event_id": IntAttr(free),
            }
        )
        loop.op.body.block.insert_op_before(op, loop.invocation.op)
        for action, endpoint in [("set", source), ("wait", destination)]:
            data = loop.actors[endpoint[0]]
            task = data.operations[endpoint[1]]
            sync = InvocationSyncOp.create(
                operands=[data.stream.body.block.args[0]],
                attributes={"action": StringAttr(action), "channel": StringAttr(name)},
            )
            if action == "set":
                task.parent.insert_op_after(sync, task)
            else:
                task.parent.insert_op_before(sync, task)
        for bridge in _protocols(loop, channel_map):
            if bridge.name == name:
                groups.setdefault(name, []).append(bridge)
    # autosync_loop performs the mandatory final check from materialized IR.
    # Do not repeat that same full parent proof as a planner preflight.

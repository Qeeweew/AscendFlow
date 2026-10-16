"""Compose local child completion with explicit cross-invocation protocols.

The leaf proof covers internal work. Tail Sets never fence their source pipe;
head Waits only depend on a previous invocation. Hence induction on serial
invocations supplies progress, while the checks below separately cover memory
and event reuse. No finite parent/child expansion participates in this proof.
"""

from dataclasses import dataclass, replace

from xdsl.dialects.builtin import IntAttr, StringAttr

from .completion import Point, Requirement, Site, covers, intersection
from .diagnostics import require
from .dialects.physical import InvocationChannelOp, InvocationSyncOp, LoopChannelOp, TaskOp
from .loop_boundary import stream_operations
from .region_completion import RegionCompletion
from .region_summary import (
    Access,
    CountDomain,
    Position,
    summarize_local_events,
    summarize_reused_storage,
)
from .sequence_plan import Bridge as RegionBridge
from .sequence_plan import materialization_snapshot

_AUTHORITY = object()


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


def _domain(loop, domain):
    if loop.constant_count is None:
        return domain
    return (
        CountDomain(1, 0, exact=loop.constant_count)
        if domain.contains(loop.constant_count)
        else None
    )


def _requirement(loop, left, right, domain, reason, *, across=False):
    domain = _domain(loop, domain)
    if domain is None:
        return None
    return Requirement(
        replace(left, region="child"),
        replace(right, region="child", call=int(across)),
        domain,
        reason,
    )


def _memory_requirements(loop, summary):
    result = []
    for dep in summary.invocation_dependencies():
        left, right = _site(dep.source, source=True), _site(dep.destination, source=False)
        domain = _domain(loop, dep.domain)
        if domain is None:
            continue
        if not left.position.end and domain.exact is not None:
            left = replace(left, position=Position(True, domain.exact - 1 - left.position.offset))
        result.append(
            _requirement(
                loop,
                left,
                right,
                domain,
                f"missing invocation {dep.hazard}: {dep.storage}[{dep.start}:{dep.stop}]",
                across=True,
            )
        )
    return tuple(result)


def _resource_requirements(loop, resources, bridges):
    result = []
    for resource in resources:
        left = Site(
            resource.consumed.task, resource.consumed.position, "begin", resource.channel, "wait"
        )
        right = Site(
            resource.republished.task, resource.republished.position, "end", resource.channel, "set"
        )
        result.append(
            _requirement(
                loop,
                left,
                right,
                resource.domain,
                f"{resource.channel}: event {resource.event_id} can be re-Set across invocations",
                across=True,
            )
        )
    for bridge in bridges:
        result.append(
            _requirement(
                loop,
                bridge.destination,
                bridge.source,
                bridge.domain,
                f"{bridge.name}: boundary credit may be re-Set before its head Wait",
            )
        )
        if not bridge.carried:
            continue
        for resource in resources:
            if (resource.channel, resource.event_id) != (bridge.name, bridge.event_id):
                continue
            domain = intersection(resource.domain, bridge.domain)
            if domain is None:
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
            for source, target in ((left, bridge.source), (bridge.destination, right)):
                result.append(
                    _requirement(
                        loop,
                        source,
                        target,
                        domain,
                        f"{bridge.name}: carried/interior event lifetimes overlap",
                    )
                )
    return tuple(r for r in result if r is not None)


def _transfer(loop, bridge):
    domain = _domain(loop, bridge.domain)
    if domain is None:
        return None
    return RegionBridge(
        Point("child", bridge.source.task[1], bridge.source.position),
        Point("child", bridge.destination.task[1], bridge.destination.position),
        (domain,),
        bridge.event_id,
        bridge.name,
        True,
    )


def _proof(loop, channel_map, graph, streams, requirements, bridges):
    # One actual retained child is a serial scope, using the SAME native
    # transfers as multi-region calls. No separate completion semantics.
    return RegionCompletion(
        {"child": loop},
        {"child": (loop, channel_map, streams, graph, None)},
        requirements,
        bridges=tuple(b for old in bridges if (b := _transfer(loop, old)) is not None),
    )


def _summary(loop):
    roots = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
        if not ref.storage.external
    }
    return summarize_reused_storage(loop, roots)


def verify_invocation(loop, channel_map, graph, streams):
    require(
        loop.invocation is not None and len(loop.actors) == 1,
        "invocation verification needs one retained local child",
    )
    bridges = _protocols(loop, channel_map)
    summary, resources = _summary(loop), summarize_local_events(loop)
    memory = _memory_requirements(loop, summary)
    requirements = memory + _resource_requirements(loop, resources, bridges)
    repeated = loop.invocation.bound.constant is None or loop.invocation.bound.constant > 1
    proof = None
    if repeated:
        proof = _proof(loop, channel_map, graph, streams, requirements, bridges)
        for requirement in requirements:
            require(proof.covered(requirement), requirement.reason)
    return _report(loop, memory, resources, bridges, proof)


def _report(loop, memory, resources, bridges, proof):
    return {
        "invocation_bound": loop.invocation.bound.text(),
        "invocation_frames": len(loop.invocation.frames),
        "invocation_memory_obligations": len(memory),
        "invocation_event_domains": len(resources),
        "invocation_bridges": len(bridges),
        "invocation_proof_queries": 0 if proof is None else proof.queries,
        "invocation_fact_transfers": 0 if proof is None else proof.flow.transfers,
    }


@dataclass(frozen=True)
class InvocationPlan:
    loop: object
    channel_map: dict
    graph: object
    streams: tuple
    bridges: tuple[Bridge, ...]
    proof: RegionCompletion | None
    memory: tuple[Requirement, ...]
    resources: tuple
    resource_requirements: tuple[Requirement, ...]
    memory_coverage: tuple[tuple[CountDomain, ...], ...]
    resource_coverage: tuple[tuple[CountDomain, ...], ...]
    snapshot: tuple
    _authority: object

    def complete(self, loop, channel_map, graph, streams):
        """Match actual insertion output and consume completed owned results."""
        require(
            self._authority is _AUTHORITY
            and self.loop is loop
            and self.graph is graph
            and self.channel_map == channel_map
            and self.streams
            == tuple((actor, tuple(commands)) for actor, commands in streams.items())
            and self.bridges == _protocols(loop, channel_map)
            and self.snapshot
            == materialization_snapshot(loop.op, (InvocationChannelOp, InvocationSyncOp)),
            "invocation materialization changed its owned native/protocol snapshot",
        )
        self._check_results()
        return _report(loop, self.memory, self.resources, self.bridges, self.proof)

    def _check_results(self):
        if self.proof is not None:
            for requests, coverage in (
                (self.memory, self.memory_coverage),
                (self.resource_requirements, self.resource_coverage),
            ):
                require(
                    len(requests) == len(coverage),
                    "invocation plan lacks its actual completed memory/resource result",
                )
                for request, result in zip(requests, coverage, strict=True):
                    require(covers(request.domain, result), request.reason)


def plan_invocation(loop, *, channel_map=None, graph=None, streams=None):
    """Extend proven slot credits, then propose precise terminal-to-head edges."""
    from .loop_graph import channels, completion_graph

    repeated = loop.invocation.bound.constant not in {0, 1} and loop.constant_count != 0
    for op in loop.op.body.block.ops if repeated else ():
        if (
            isinstance(op, LoopChannelOp)
            and op.kind.data == "local"
            and op.last is None
            and op.period.data == 1
            and op.distance.data == len(op.event_ids)
            and op.distance.data > 0
        ):
            op.attributes["carry"] = IntAttr(1)
    actual = channels(loop)
    require(
        channel_map is None
        or channel_map.keys() == actual.keys()
        and all(
            replace(old, carry=actual[name].carry) == actual[name]
            for name, old in channel_map.items()
        ),
        "invocation carry selection changed its child protocol beyond startup credits",
    )
    channel_map = actual
    if graph is None:
        graph, streams, _, native = completion_graph(loop, channel_map)
    else:
        require(
            streams is not None and graph.period == loop.protocol_period,
            "invocation planning needs its actual child graph",
        )
        native = {
            (command.actor, command.task.uid): command
            for sequence in streams.values()
            for command in sequence
            if command.kind == "task"
        }
    bridges = _protocols(loop, channel_map)
    summary, resources = _summary(loop), summarize_local_events(loop)
    requirements = _memory_requirements(loop, summary)
    initial_resources = _resource_requirements(loop, resources, bridges)
    reserved = tuple(
        Requirement(
            replace(request.destination, call=0),
            replace(request.source, point="prefix", call=0),
            request.domain,
            "potential invocation boundary credit",
        )
        for request in requirements
    )
    proof = (
        _proof(
            loop, channel_map, graph, streams, requirements + initial_resources + reserved, bridges
        )
        if repeated
        else None
    )
    snapshot = materialization_snapshot(loop.op, (InvocationChannelOp, InvocationSyncOp))
    revision, completed, pending = 0, {}, {}

    def coverage(request):
        key = request.source, request.destination, request.domain
        if key in completed:
            return completed[key]
        if key not in pending or pending[key][0] != revision:
            result = tuple(proof.coverage(request))
            pending[key] = revision, result
            if covers(request.domain, result):
                completed[key] = result
        return pending[key][1]

    candidates = {}
    for requirement in requirements if repeated else ():
        if covers(requirement.domain, coverage(requirement)):
            continue
        left, right = requirement.source, requirement.destination
        require(
            left.position.end and not right.position.end,
            "uncovered invocation relation needs an explicit tail-to-head protocol",
        )
        key = left.task, right.task, left.position.offset, right.position.offset
        candidates.setdefault(key, []).append(requirement)
    for (source, destination, gap, index), required in candidates.items():
        if all(covers(requirement.domain, coverage(requirement)) for requirement in required):
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
        bridge = Bridge(
            name,
            free,
            Site(source, Position(True, gap), "prefix"),
            Site(destination, Position(False, index), "begin"),
            CountDomain(1, 0, minimum=max(gap, index) + 1),
            False,
        )
        bridges = (*bridges, bridge)
        if transfer := _transfer(loop, bridge):
            proof.add_bridge(transfer)
        revision += 1
    memory_coverage = tuple(coverage(request) for request in requirements) if repeated else ()
    resource_requirements = _resource_requirements(loop, resources, bridges)
    resource_coverage = (
        tuple(coverage(request) for request in resource_requirements) if repeated else ()
    )
    plan = InvocationPlan(
        loop,
        channel_map,
        graph,
        tuple((actor, tuple(commands)) for actor, commands in streams.items()),
        tuple(bridges),
        proof,
        requirements,
        tuple(resources),
        resource_requirements,
        memory_coverage,
        resource_coverage,
        snapshot,
        _AUTHORITY,
    )
    # The caller matches actual declarations/instructions once at completion.
    # Existing native transfers and ordering results remain owned here.
    return plan

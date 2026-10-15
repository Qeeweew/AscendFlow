"""Compose fixed child proofs at retained parent pipeline callsites."""

from collections import defaultdict
from dataclasses import dataclass, replace
from math import lcm

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import bounded, poll
from .child_completion import ChildPort, project_fixed_child
from .ir import LoopChannelOp, ScopeChannelOp, ScopeSyncOp, TaskOp
from .loop_graph import Command
from .loop_sync import _lifetime, _plan_leaf, _requirements, _verify_leaf
from .model import require
from .native_order import instruction_points, prefix_inputs
from .periodic import bounded_period, project
from .region_summary import summarize_local_events
from .scope_completion import compose_calls
from .scope_events import event_choices, resource_inputs, resource_sites
from .scope_model import extract_scopes


@dataclass(frozen=True)
class ScopeProtocol:
    name: str
    kind: str
    sources: tuple
    destinations: tuple
    distance: int
    period: int
    residues: tuple[int, ...]
    ids: tuple[int, ...] = ()
    last: bool = False
    carry: bool = False


@dataclass(frozen=True)
class CheckedScopes:
    program: object
    analyses: dict
    access: object
    channels: dict[str, ScopeProtocol]


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _point(site, point):
    return replace(site, point=point)


class ParentCompletion:
    def __init__(self, program, analyses, access, period, *, resource_inputs=None):
        self.program, self.access = program, access
        self.events, self.protocols = {}, []
        marks = defaultdict(set)
        for uid, site in access.endpoints.values():
            marks[uid].update(
                _point(site, point)
                for point in (
                    "begin",
                    "end",
                    "prefix",
                    "issue",
                    "queue",
                    "retire",
                    "implicit",
                    "admit_issue",
                    "admit_queue",
                    "after_queue",
                )
            )
        self.resource_pairs = []
        for uid, analysis in analyses.items():
            poll("child event resource projection")
            leaf = program.leaves[uid]
            resources = (
                summarize_local_events(analysis[0])
                if resource_inputs is None
                else resource_inputs.get(uid, ())
            )
            for resource in resources:
                if not resource.domain.contains(analysis[0].constant_count):
                    continue
                sites = resource_sites(uid, leaf.actor, analysis[0].constant_count, resource)
                marks[uid].update(sites)
                self.resource_pairs.append((uid, *sites, resource))
        calls = {}
        for uid, leaf in program.leaves.items():
            poll("child completion projection")
            local_actor = next(iter(analyses[uid][0].actors))
            local = tuple(
                replace(site, task=(local_actor, site.task[1]))
                for site in sorted(marks[uid], key=repr)
            )
            projection = project_fixed_child(analyses[uid], local, insertion_cuts=True)

            def retarget(point, *, actor=leaf.actor):
                return (
                    replace(point, actor=actor)
                    if isinstance(point, ChildPort)
                    else replace(point, task=(actor, point.task[1]))
                )

            # Retarget each boundary point once. Rebuilding Site objects for
            # both ends of every proved relation adds quadratic allocation
            # work without changing any physical participant or relation.
            renamed = {
                p: retarget(p)
                for p in (*projection.inputs, *projection.outputs, *projection.endpoints)
            }
            projection = replace(
                projection,
                inputs=tuple(renamed[p] for p in projection.inputs),
                outputs=tuple(renamed[p] for p in projection.outputs),
                endpoints=tuple(renamed[p] for p in projection.endpoints),
                relations=frozenset((renamed[a], renamed[b]) for a, b in projection.relations),
            )
            calls[uid] = leaf.call, projection
        self.completion = compose_calls(calls, period=period, count=program.count)
        self.graph = self.completion.graph
        self.number = 1 + max(node.command.index for node in self.graph.nodes)
        self.waits, self.sends = {}, {}

    def site(self, endpoint, point, generation):
        uid, site = self.access.endpoints[endpoint]
        return self.completion.at(uid, _point(site, point), generation)

    def queries(self, obligation):
        source_uid, source = self.access.endpoints[obligation.source]
        target_uid, target = self.access.endpoints[obligation.destination]
        for residue, first in project(
            obligation.period, obligation.residues, self.graph.period, obligation.distance
        ):
            yield self.completion.query(
                source_uid,
                _point(source, "prefix" if obligation.prefix else "end"),
                residue - obligation.distance,
                target_uid,
                _point(target, "begin"),
                residue,
                minimum=first,
            )

    def covered(self, obligation):
        return all(self.graph.covered_many(tuple(self.queries(obligation))))

    def _forward(self, command, residue, kind, before, added):
        # A synchronization may be absent in fill/drain even while its native
        # is active. Preserve the incoming prefix through that skipped sync.
        after = self.graph.node(command, residue, "after_" + kind, guard=(0, 0)), 0
        self.completion.edge(before, after)
        self.completion.edge(added, after)
        return after

    def insert(self, protocol):
        """Real Set/Wait transitions at child native admission cuts, not child exits."""
        require(protocol.name not in {p.name for p in self.protocols}, "duplicate scope protocol")
        self.protocols.append(protocol)
        tasks = {
            (actor, t.uid): t
            for actor, d in self.access.loop.actors.items()
            for t in d.program.tasks
        }
        placements = (
            [(owner, "barrier") for owner in protocol.destinations]
            if protocol.kind == "barrier"
            else [(owner, "set") for owner in protocol.sources]
            + [(owner, "wait") for owner in protocol.destinations]
        )
        points = {}
        for owner, action in placements:
            poll("parent synchronization insertion")
            task = tasks[owner]
            kind = "cross_" + action if protocol.kind == "cross" else action
            residues = (
                protocol.residues
                if action != "wait"
                else tuple(
                    sorted((r + protocol.distance) % protocol.period for r in protocol.residues)
                )
            )
            command = Command(
                self.number,
                owner[0],
                task,
                kind,
                protocol.name,
                protocol.distance if action != "set" else 0,
                protocol.distance if action == "set" else 0,
                protocol.period,
                residues,
            )
            self.number += 1
            self.events[owner[0], protocol.name, action] = command
            for residue in range(self.graph.period):
                if residue % command.period not in command.residues:
                    continue
                before = {
                    point: self.site(owner, point, residue)
                    for point in ("issue", "queue", "retire", "implicit")
                }
                key = owner, residue
                if action == "set":
                    before["issue"] = self.site(
                        owner, "end" if task.pipe == "S" else "admit_issue", residue
                    )
                    before["queue"] = self.sends.get(key, self.site(owner, "begin", residue))
                    before["retire"] = self.site(owner, "prefix", residue)
                else:
                    before.update(self.waits.get(key, {}))
                actual = instruction_points(
                    command,
                    before,
                    lambda point, c=command, r=residue: (self.graph.node(c, r, point), 0),
                    self.completion.edge,
                )
                points[owner[0], action, residue] = actual
                if action == "set":
                    self.sends[key] = self._forward(
                        command, residue, "queue", before["queue"], actual["begin"]
                    )
                    self.completion.edge(self.sends[key], self.site(owner, "after_queue", residue))
                else:
                    for point, value in prefix_inputs(command, actual).items():
                        after = self._forward(command, residue, point, before[point], value)
                        # Native admission cuts carry the acquire forward on
                        # the actual destination pipe/controller only.
                        self.completion.edge(after, self.site(owner, "admit_" + point, residue))
                        self.waits.setdefault(key, {})[point] = after
        if protocol.kind == "barrier":
            return
        active = [r for r in range(self.graph.period) if r % protocol.period in protocol.residues]
        for i, residue in enumerate(active):
            for actor, _ in protocol.sources:
                source = points[actor, "set", residue]["end"]
                if protocol.kind == "cross":
                    prior = points[actor, "set", active[i - 1]]["end"]
                    self.completion.edge((prior[0], -int(i == 0)), source)
                for actor, _ in protocol.destinations:
                    generation = residue + protocol.distance
                    target = points[actor, "wait", generation % self.graph.period]["begin"]
                    self.completion.edge(source, (target[0], generation // self.graph.period))

    def check_child_resources(self):
        queries = 0
        for uid, consumed, republished, resource in self.resource_pairs:
            poll("child event reuse verification")
            require(
                self.resource_covered(uid, consumed, republished),
                f"{uid}/{resource.channel}: child event {resource.event_id} may be re-Set before its previous parent call Wait",
            )
            queries += self.graph.period
        return queries

    def resource_covered(self, uid, consumed, republished):
        """A selected ID's final Wait is consumed before this call republishes it."""
        return all(
            self.graph.covered_many(
                self.completion.query(uid, consumed, r, uid, republished, r + 1, minimum=1)
                for r in range(self.graph.period)
            )
        )


def _analyses(program):
    reports, analyses = {}, {}
    for uid, leaf in program.leaves.items():
        roots = {
            leaf.loop.root(actor, ref)
            for actor, data in leaf.loop.actors.items()
            for task in data.program.tasks
            for ref in task.refs
        }
        reports[uid], analyses[uid] = _verify_leaf(leaf.loop, live_in=roots, _details=True)
    return reports, analyses


def _encode(access, owner):
    uid, site = access.endpoints[owner]
    return "/".join((owner[0], uid, site.task[1], str(site.position.offset)))


def _install(program, access, protocols):
    root = program.root.body.block
    first = (
        next(iter(program.kernels.values()))
        if len(program.kernels) > 1
        else next(iter(program.parents.values()))
    )
    for p in protocols:
        declaration = ScopeChannelOp.create(
            attributes={
                "sym_name": StringAttr(p.name),
                "kind": StringAttr(p.kind),
                "sources": ArrayAttr(StringAttr(_encode(access, key)) for key in p.sources),
                "destinations": ArrayAttr(
                    StringAttr(_encode(access, key)) for key in p.destinations
                ),
                "distance": IntAttr(p.distance),
                "period": IntAttr(p.period),
                "residues": _ints(p.residues),
                "event_ids": _ints(p.ids),
            }
        )
        root.insert_op_before(declaration, first)
        placements = (
            [(owner, "barrier") for owner in p.destinations]
            if p.kind == "barrier"
            else [(owner, "set") for owner in p.sources]
            + [(owner, "wait") for owner in p.destinations]
        )
        for owner, action in placements:
            uid, site = access.endpoints[owner]
            leaf = program.leaves[uid]
            task = next(
                op
                for op in leaf.stream.walk()
                if isinstance(op, TaskOp) and op.uid.data == site.task[1]
            )
            sync = ScopeSyncOp.create(
                operands=[leaf.call.generation, leaf.stream.body.block.args[0]],
                attributes={
                    "action": StringAttr(action),
                    "channel": StringAttr(p.name),
                    "position": IntAttr(site.position.offset),
                },
            )
            if action == "set":
                anchor = task
                while (
                    isinstance(anchor.next_op, ScopeSyncOp) and anchor.next_op.action.data == "set"
                ):
                    anchor = anchor.next_op
                task.parent.insert_op_after(sync, anchor)
            else:
                task.parent.insert_op_before(sync, task)


def _configuration_banks(program, analyses):
    used = defaultdict(set)
    for uid, analysis in analyses.items():
        leaf = program.leaves[uid]
        data = next(iter(analysis[0].actors.values()))
        if data.program.config:
            used[leaf.actor, "S", "FIX"].add(0)
        if data.program.load3d is not None:
            used[leaf.actor, "S", "MTE1"].add(0)
    return used


def _leaf_banks(program, analyses):
    used = _configuration_banks(program, analyses)
    for uid, analysis in analyses.items():
        leaf, loop, channels = program.leaves[uid], analysis[0], analysis[1]
        data = next(iter(loop.actors.values()))
        tasks = {task.uid: task for task in data.program.tasks}
        for channel in channels.values():
            require(
                channel.kind == "local" and not channel.carry,
                "nested leaf channels need local fixed-child contracts",
            )
            key = (
                leaf.actor,
                tasks[channel.sources[0][1]].pipe,
                tasks[channel.destinations[0][1]].pipe,
            )
            require(
                not used[key].intersection(channel.ids),
                "child event IDs overlap across nested callsites",
            )
            used[key].update(channel.ids)
    return used


def _allocate_child_resources(program, analyses, access, proof, selected):
    """Extend a leaf's ID rotation only when its actual parent lifetime needs it.

    Common valid plans reuse their existing projection and IDs. If a boundary
    fails, all locally legal widths share one enlarged parent projection; no
    native or synchronization instruction is added, moved, or removed.
    """
    valid = all(
        proof.resource_covered(uid, consumed, republished)
        for uid, consumed, republished, _ in proof.resource_pairs
    )
    if valid:
        return proof, _leaf_banks(program, analyses)
    choices = event_choices(program, analyses)
    proof = ParentCompletion(
        program, analyses, access, proof.graph.period, resource_inputs=resource_inputs(choices)
    )
    for protocol in selected:
        proof.insert(protocol)
    proof.graph.potential()
    used = _configuration_banks(program, analyses)
    for (uid, name), choice in choices.items():
        poll("parent-aware child event allocation", call=uid, channel=name)
        option = next(
            (
                option
                for option in choice.options
                if all(
                    proof.resource_covered(
                        uid, *resource_sites(uid, choice.bank[0], choice.count, resource)
                    )
                    for resource in option.resources
                )
            ),
            None,
        )
        require(
            option is not None,
            f"{uid}/{name}: no child event width <=8 proves consumption before the next parent call Set",
        )
        free = [i for i in range(8) if i not in used[choice.bank]]
        require(
            len(free) >= option.width,
            f"{choice.bank}: parent-aware child event pools exhaust 8 IDs",
        )
        ids = tuple(free[: option.width])
        declaration = next(
            op
            for op in program.leaves[uid].stream.walk()
            if isinstance(op, LoopChannelOp) and op.sym_name.data == name
        )
        declaration.attributes["event_ids"] = _ints(ids)
        used[choice.bank].update(ids)
    return proof, used


@bounded
def autosync_scopes(module, *, _capture=None):
    from .scope_access import analyze_scope_access
    from .sync import _finish

    result = module.clone()
    require(
        not any(
            isinstance(op, (ScopeChannelOp, ScopeSyncOp, LoopChannelOp)) for op in result.walk()
        ),
        "nested AutoSync requires unsynchronized physical calls",
    )
    program = extract_scopes(result)
    banks = defaultdict(lambda: defaultdict(int))
    for leaf in program.leaves.values():
        poll("child synchronization planning")
        loop = leaf.loop
        roots = {
            loop.root(actor, ref)
            for actor, data in loop.actors.items()
            for task in data.program.tasks
            for ref in task.refs
        }
        data = next(iter(loop.actors.values()))
        if data.program.load3d is not None:
            key = next(iter(loop.actors)), "S", "MTE1"
            banks[leaf.actor][key] = max(1, banks[leaf.actor][key])
        _plan_leaf(loop, live_in=roots, banks=banks[leaf.actor])
        aliases = {leaf.value_map[c.op.ref]: c.op.ref for c in leaf.captures}
        replacement = data.stream.clone(value_mapper=aliases)
        leaf.call.body.block.insert_op_before(replacement, leaf.stream)
        leaf.call.body.block.erase_op(leaf.stream)
    program = extract_scopes(result)
    access = analyze_scope_access(program)
    _, analyses = _analyses(program)
    candidates = _requirements(access.loop, access.obligations)
    period = (
        bounded_period(lcm(*(p.period for p, _ in candidates)), "parent protocols")
        if candidates
        else 1
    )
    proof = ParentCompletion(program, analyses, access, period)
    # For one acquire, try the required latest producer capture before earlier
    # ones; only the already selected actual protocols establish coverage.
    tasks = {(a, t.uid): t for a, d in access.loop.actors.items() for t in d.program.tasks}
    ordered = sorted(
        candidates,
        key=lambda item: (
            all(o.distance > 0 for o in item[1]),
            min((tasks[d].stage, tasks[d].issue_order) for d in item[0].destinations),
            {"local": 0, "cross": 1, "barrier": 2}[item[0].kind],
            -max((tasks[s].issue_order for s in item[0].sources), default=0),
        ),
    )
    selected = []
    for candidate, obligations in ordered:
        poll("parent synchronization selection")
        if all(proof.covered(o) for o in obligations):
            continue
        p = ScopeProtocol(
            f"scope_c{len(selected)}",
            candidate.kind,
            candidate.sources,
            candidate.destinations,
            candidate.distance,
            candidate.period,
            candidate.residues,
        )
        proof.insert(p)
        require(
            all(proof.covered(o) for o in obligations),
            "parent protocol cannot cover its actual child access obligation",
        )
        selected.append(p)
    proof.graph.potential()
    proof, used = _allocate_child_resources(program, analyses, access, proof, selected)
    from .scope_event_pools import allocate_parent_pools

    selected, _ = allocate_parent_pools(selected, proof, used)
    cross_id = 0
    allocated = []
    for p in selected:
        if p.kind == "barrier":
            allocated.append(p)
            continue
        if p.kind == "cross":
            require(cross_id < 8, "nested CV counter bank exhausted")
            ids, cross_id = (cross_id,), cross_id + 1
        else:
            ids = p.ids
        allocated.append(replace(p, ids=ids))
    _install(program, access, allocated)
    return _finish(result, _capture=_capture)


def _read_protocols(program, access, analyses):
    from .loop_boundary import stream_operations

    by_encoding = {_encode(access, key): key for key in access.endpoints}
    tasks = {(a, t.uid): t for a, d in access.loop.actors.items() for t in d.program.tasks}
    channels, expected = {}, {}
    expected_order, actual_order = defaultdict(list), defaultdict(list)
    reserved = _leaf_banks(program, analyses)
    used = defaultdict(set, {key: set(ids) for key, ids in reserved.items()})
    cross_ids = set()
    for op in program.module.walk():
        if not isinstance(op, ScopeChannelOp):
            continue
        require(
            op.parent_op() is program.root, "parent channel must belong to the actual kernel/group"
        )
        name = op.sym_name.data
        require(name.isidentifier() and name not in channels, "invalid/duplicate parent channel")
        encoded = [tuple(item.data for item in group) for group in (op.sources, op.destinations)]
        require(
            all(key in by_encoding for group in encoded for key in group),
            "parent channel has no actual child native endpoint",
        )
        sources, destinations = (tuple(by_encoding[key] for key in group) for group in encoded)
        kind, distance, period = op.kind.data, op.distance.data, op.period.data
        residues, ids = tuple(v.data for v in op.residues), tuple(v.data for v in op.event_ids)
        bounded_period(period, "parent protocol")
        require(
            distance >= 0
            and residues
            and residues == tuple(sorted(set(residues)))
            and all(0 <= r < period for r in residues),
            "invalid parent protocol activation",
        )
        require(
            len(set(ids)) == len(ids) and all(0 <= i < 8 for i in ids), "invalid parent event IDs"
        )
        require(
            len({a for a, _ in sources}) == len(sources)
            and len({a for a, _ in destinations}) == len(destinations),
            "parent protocol duplicates an actor contribution",
        )
        if kind == "barrier":
            require(
                not sources and len(destinations) == 1 and not ids,
                "parent native fence has one actual destination and no event resource",
            )
        elif kind == "local":
            require(
                len(sources) == len(destinations) == 1
                and sources[0][0] == destinations[0][0]
                and tasks[sources[0]].pipe != tasks[destinations[0]].pipe
                and ids,
                "parent local event needs two distinct pipes in one actor",
            )
            key = sources[0][0], tasks[sources[0]].pipe, tasks[destinations[0]].pipe
            require(
                not reserved[key].intersection(ids),
                "parent event overlaps a child/configuration resource",
            )
            used[key].update(ids)
        else:
            require(
                kind == "cross"
                and ({a for a, _ in sources}, {a for a, _ in destinations})
                in (({"aic"}, {"aiv0", "aiv1"}), ({"aiv0", "aiv1"}, {"aic"})),
                "parent CV protocol requires the actual AIC/two-AIV quorum",
            )
            require(len(ids) == 1 and ids[0] not in cross_ids, "parent CV counter IDs overlap")
            cross_ids.update(ids)
        channels[name] = ScopeProtocol(
            name, kind, sources, destinations, distance, period, residues, ids
        )
        placements = (
            [(key, "barrier") for key in destinations]
            if kind == "barrier"
            else [(key, "set") for key in sources] + [(key, "wait") for key in destinations]
        )
        for key, action in placements:
            uid, site = access.endpoints[key]
            expected[name, key[0], action] = uid, site.task[1], site.position.offset
            expected_order[uid, site.task[1], action == "set"].append(
                (name, action, site.position.offset)
            )
    actual, observed = {}, set()
    for uid, leaf in program.leaves.items():
        poll("actual parent synchronization extraction")
        pending, previous, parent = [], None, None
        for op, _ in stream_operations(leaf.stream):
            if op.parent is not parent:
                require(not pending, "parent acquire escaped its actual native region")
                parent, previous = op.parent, None
            if isinstance(op, TaskOp):
                for sync in pending:
                    key = sync.channel.data, leaf.actor, sync.action.data
                    require(key not in actual, "duplicate parent synchronization")
                    actual[key] = uid, op.uid.data, sync.position.data
                    actual_order[uid, op.uid.data, False].append(
                        (sync.channel.data, sync.action.data, sync.position.data)
                    )
                pending, previous = [], op
            elif isinstance(op, ScopeSyncOp):
                observed.add(op)
                require(
                    op.parent_generation is leaf.call.generation
                    and op.child_generation is leaf.stream.body.block.args[0],
                    "parent synchronization uses the wrong parent/child clock",
                )
                require(
                    op.channel.data in channels and op.action.data in {"set", "wait", "barrier"},
                    "parent synchronization refers to an unknown protocol/action",
                )
                require(
                    0 <= op.position.data < leaf.loop.constant_count,
                    "parent synchronization has no active inner position",
                )
                if op.action.data == "set":
                    require(
                        previous is not None and not pending,
                        "parent Set must follow its actual native producer",
                    )
                    key = op.channel.data, leaf.actor, "set"
                    require(key not in actual, "duplicate parent synchronization")
                    actual[key] = uid, previous.uid.data, op.position.data
                    actual_order[uid, previous.uid.data, True].append(
                        (op.channel.data, "set", op.position.data)
                    )
                else:
                    pending.append(op)
            else:
                # The child projection retains every local synchronization.
                # A parent acquire is immediately before its native; a Set
                # immediately follows it. Crossing a child command would
                # change the instruction order used by both proof and ASC.
                require(not pending, "parent acquire must be adjacent to its actual native")
                previous = None
        require(not pending, "unattached parent acquire")
    require(
        actual == expected,
        "parent protocol is missing or changes an actual native Set/Wait placement",
    )
    require(
        actual_order == expected_order,
        "parent synchronization order differs from its declared insertion order",
    )
    require(
        observed == {op for op in program.module.walk() if isinstance(op, ScopeSyncOp)},
        "parent synchronization escaped its actual child scope",
    )
    return channels, used


@bounded
def verify_scopes(module, *, _details=False):
    from .scope_access import analyze_scope_access

    program = extract_scopes(module)
    access = analyze_scope_access(program)
    reports, analyses = _analyses(program)
    channels, banks = _read_protocols(program, access, analyses)
    period = (
        bounded_period(lcm(*(p.period for p in channels.values())), "parent protocols")
        if channels
        else 1
    )
    proof = ParentCompletion(program, analyses, access, period)
    for channel in channels.values():
        proof.insert(channel)
    potential = proof.graph.potential()
    for obligation in access.obligations:
        require(
            proof.covered(obligation),
            f"missing scoped {obligation.hazard}: {obligation.source} -> {obligation.destination}, "
            f"distance {obligation.distance} on {obligation.storage}[{obligation.start}:{obligation.stop}]",
        )
    child_lifetimes = proof.check_child_resources()
    from .scope_event_pools import verify_parent_pools

    pool_statistics = {}
    banks = verify_parent_pools(
        channels.values(), proof, _leaf_banks(program, analyses), statistics=pool_statistics
    )
    _, counters, _ = _lifetime(access.loop, channels, proof.graph, proof.events, local=False)
    report = {
        "kernel": program.root.sym_name.data,
        "verification": "actual retained child proofs, parent physical frontiers, guarded completion, and event lifetimes",
        "native_templates": sum(r["native_templates"] for r in reports.values()),
        "parent_count": program.count,
        "parent_bound": None if program.bound is None else program.bound.text(),
        "children": reports,
        "scope_completion": proof.completion.statistics(),
        "scope_projection_statistics": {
            uid: p.statistics() for uid, p in proof.completion.projections.items()
        },
        "scope_completion_facts": proof.graph.completion_statistics(),
        "parent_memory_obligations": len(access.obligations),
        "parent_access_visits": access.visits,
        "parent_channels": len(channels),
        "parent_binary_lifetime_queries": pool_statistics["self_queries"],
        "child_parent_lifetime_queries": child_lifetimes,
        "parent_event_pool_proof": pool_statistics,
        "event_ids": {":".join(key): sorted(ids) for key, ids in banks.items()},
        "cross_counter_ids": sorted(counters),
        "progress_stride": potential[0],
        "terminal_drain": "only kernel actor return; child exit is not completion",
    }
    checked = CheckedScopes(program, analyses, access, channels)
    return (report, checked) if _details else report

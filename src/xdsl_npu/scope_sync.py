"""Compose fixed child proofs at retained parent pipeline callsites."""

from collections import defaultdict
from dataclasses import dataclass, replace
from math import lcm

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import bounded, poll
from .diagnostics import require
from .dialects.physical import (
    LoopChannelOp,
    QueryChannelOp,
    QuerySyncOp,
    ScopeChannelOp,
    ScopeSyncOp,
    TaskOp,
)
from .loop_graph import Command
from .loop_sync import _plan_leaf, _requirements, establishes_directly
from .native_order import instruction_points, prefix_inputs
from .periodic import bounded_period, project
from .region_summary import summarize_local_events
from .scope_child_transfer import ConditionalChildTransfer, FixedChildTransfer, TransferCase
from .scope_completion import compose_transfers
from .scope_domains import (
    Activation,
    ScopedDependence,
    decode_activation,
    encode_activation,
    legacy_activation,
    shifted,
    validate_activation,
)
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
    activation: Activation | None = None


@dataclass(frozen=True)
class CheckedScopes:
    program: object
    analyses: dict
    access: object
    channels: dict[str, ScopeProtocol]
    query_summary: object | None = None
    query_report: dict | None = None
    resource_proof: object | None = None


@dataclass(frozen=True)
class PlanningScopes:
    """Owned planning inputs, without a final memory/resource/progress gate."""

    program: object
    analyses: dict
    access: object
    channels: dict[str, ScopeProtocol]
    query_summary: object | None = None
    resource_proof: object | None = None
    child_reports: object | None = None


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _point(site, point):
    return replace(site, point=point)


class ParentCompletion:
    def __init__(
        self,
        program,
        analyses,
        access,
        period,
        *,
        resource_inputs=None,
        extra_protocols=(),
        boundary_sites=(),
    ):
        self.program, self.analyses, self.access = program, analyses, access
        self.events, self.protocols = {}, []
        from .scope_guarded_completion import NodeActivations

        self.node_activations, self._guarded, self._anchored = NodeActivations(), None, None
        # Access and resource endpoints remain exact virtual handles. Only
        # actual protocol insertion cuts are promoted into the parent graph.
        for uid, site in boundary_sites:
            require(uid in program.leaves, "query boundary names an unknown actual call")
            leaf = program.leaves[uid]
            require(
                site.region == uid
                and site.task[0] == leaf.actor
                and not site.channel
                and not site.action
                and site.task[1]
                in {task.uid for data in leaf.loop.actors.values() for task in data.program.tasks},
                "query boundary needs an actual native in its original call and actor",
            )
        self.resource_pairs = []
        for uid in analyses:
            poll("child event resource projection")
            leaf = program.leaves[uid]
            variants = (
                analyses.variants(uid)
                if hasattr(analyses, "variants")
                else ((None, analyses[uid]),)
            )
            for _, analysis in variants:
                resources = (
                    summarize_local_events(analysis[0])
                    if resource_inputs is None
                    else resource_inputs.get(uid, ())
                )
                for resource in resources:
                    if not resource.domain.contains(analysis[0].constant_count):
                        continue
                    sites = resource_sites(uid, leaf.actor, analysis[0].constant_count, resource)
                    self.resource_pairs.append((uid, *sites, resource))
        self.resource_pairs = list(dict.fromkeys(self.resource_pairs))
        calls = {}
        for uid, leaf in program.leaves.items():
            poll("child completion projection")

            if leaf.cases:
                projection = ConditionalChildTransfer(
                    leaf.loop.constant_count,
                    sum(len(data.program.tasks) for data in leaf.loop.actors.values()),
                    tuple(
                        TransferCase(label, leaf.cases[label].activation,
                                     FixedChildTransfer(analysis, leaf.actor))
                        for label, analysis in analyses.variants(uid)
                    ),
                )
            else:
                projection = FixedChildTransfer(analyses[uid], leaf.actor)
            calls[uid] = leaf.call, projection
        self.completion = compose_transfers(calls, period=period, count=program.count)
        self.graph = self.completion.graph
        self.node_activations.update(self.completion.node_activations)
        self.completion.transfer_state.activation_sink = self.node_activations
        self.number = 1 + max(node.command.index for node in self.graph.nodes)
        self.waits, self.sends = {}, {}

    def site(self, endpoint, point, generation):
        uid, site = self.access.endpoints[endpoint]
        points = self.completion.ats(uid, _point(site, point), generation)
        return points[0] if len(points) == 1 else points

    def edge(self, source, target):
        sources = (source,) if isinstance(source[0], int) else source
        targets = (target,) if isinstance(target[0], int) else target
        for first in sources:
            for last in targets:
                self.completion.edge(first, last)

    @property
    def period(self):
        return self.graph.period

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
        return self.covered_obligations((obligation,))[0]

    def _guarded_obligation(self, obligation):
        from .completion import Position

        source_uid, source = self.access.endpoints[obligation.source]
        target_uid, target = self.access.endpoints[obligation.destination]
        endpoints = tuple(
            (a, b, obligation.distance)
            for a in self.completion.commands_for(
                source_uid, _point(source, "prefix" if obligation.prefix else "end")
            )
            for b in self.completion.commands_for(target_uid, _point(target, "begin"))
        )
        if isinstance(obligation, ScopedDependence):
            return tuple((*pair, obligation.activation) for pair in endpoints)
        # Prove the stronger translation family from the first real alias
        # intersection, as in ordinary Graph queries. The guarded solver still
        # checks every epoch, including all first/last paths; no conditional
        # path is assumed translation invariant by this projection.
        return tuple(
            (
                *pair,
                Activation(
                    first=Position(False, first - obligation.distance),
                    last=Position(True, obligation.distance),
                    period=self.graph.period,
                    residues=((residue - obligation.distance) % self.graph.period,),
                ),
            )
            for residue, first in project(
                obligation.period, obligation.residues, self.graph.period, obligation.distance
            )
            for pair in endpoints
        )

    def covered_obligations(self, obligations):
        obligations = tuple(obligations)
        if any(hasattr(o, "source_position") for o in obligations):
            results = [False] * len(obligations)
            fixed = [(i, o) for i, o in enumerate(obligations) if not hasattr(o, "source_position")]
            for (index, _), result in zip(
                fixed, self.covered_obligations(o for _, o in fixed), strict=True
            ):
                results[index] = result
            anchored = self.anchored()
            for index, obligation in enumerate(obligations):
                if not hasattr(obligation, "source_position"):
                    continue
                source_uid, source = self.access.endpoints[obligation.source]
                target_uid, target = self.access.endpoints[obligation.destination]
                results[index] = all(
                    anchored.covered(
                        a,
                        b,
                        obligation.source_position,
                        obligation.destination_position,
                        obligation.activation,
                    )
                    for a in self.completion.commands_for(
                        source_uid, _point(source, "prefix" if obligation.prefix else "end")
                    )
                    for b in self.completion.commands_for(target_uid, _point(target, "begin"))
                )
            return results
        if self.completion.transfer_state is not None or self.node_activations or any(
            isinstance(o, ScopedDependence) for o in obligations
        ):
            groups = [self._guarded_obligation(o) for o in obligations]
            self.completion.prepare_virtual(command for group in groups for query in group
                                            for command in query[:2])
            answers = self.guarded().covered_many(q for group in groups for q in group)
            results, offset = [], 0
            for group in groups:
                end = offset + len(group)
                results.append(all(answers[offset:end]))
                offset = end
            return results
        return [all(self.graph.covered_many(tuple(self.queries(o)))) for o in obligations]

    def guarded(self):
        if self._guarded is None:
            from .scope_guarded_completion import GuardedCompletion

            self._guarded = GuardedCompletion(self.graph, self.node_activations)
        return self._guarded

    def anchored(self):
        if self._anchored is None:
            from .scope_anchored_completion import AnchoredCompletion

            self._anchored = AnchoredCompletion(self.guarded())
        return self._anchored

    def covered_many(self, queries):
        """Resource queries must see every actual conditional protocol too."""
        if not self.node_activations and self.completion.transfer_state is None:
            return self.graph.covered_many(queries)
        converted = []
        for source, sg, destination, dg, *rest in queries:
            require(not rest, "guarded resource query needs explicit endpoint activation")
            distance = dg - sg
            from .completion import Position

            first = max(source.lower, destination.lower - distance, 0, -distance)
            gap = max(source.upper, destination.upper + distance, 0, distance)
            activation = Activation(
                first=Position(False, first),
                last=Position(True, gap),
                period=self.graph.period,
                residues=(sg % self.graph.period,),
            )
            converted.append((source, destination, distance, activation))
        return self.guarded().covered_many(converted)

    def _forward(self, command, residue, kind, before, added):
        # A synchronization may be absent in fill/drain even while its native
        # is active. Preserve the incoming prefix through that skipped sync.
        after = self.graph.node(command, residue, "after_" + kind, guard=(0, 0)), 0
        self.edge(before, after)
        self.edge(added, after)
        return after

    def insert(self, protocol):
        """Real Set/Wait transitions at child native admission cuts, not child exits."""
        require(protocol.name not in {p.name for p in self.protocols}, "duplicate scope protocol")
        if any(leaf.cases for leaf in self.program.leaves.values()):
            from .scope_native_activation import validate_scope_protocol

            validate_scope_protocol(self.program, self.access, protocol)
        if protocol.activation is not None:
            validate_activation(protocol.activation, distance=protocol.distance)
            require(
                (protocol.period, protocol.residues)
                == (protocol.activation.period, protocol.activation.residues),
                "scope channel has inconsistent activation residues",
            )
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
        promoted_sites = []
        for owner, action in placements:
            task = tasks[owner]
            names = (("end" if task.pipe == "S" else "admit_issue", "begin", "prefix", "after_queue")
                     if action == "set" else
                     ("issue", "queue", "retire", "admit_issue", "admit_queue")
                     if action == "barrier" else
                     ("issue", "queue", "admit_issue", "admit_queue"))
            uid, site = self.access.endpoints[owner]
            for name in names:
                promoted_sites.append((uid, _point(site, name)))
        self.completion.prepare_virtual(cut for uid, site in promoted_sites
                                        for cut in self.completion.commands_for(uid, site))
        for uid, site in promoted_sites:
            self.completion.transfer_state.promote(uid, site)
        self.number = 1 + max(node.command.index for node in self.graph.nodes)
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
            activation = (
                None
                if protocol.activation is None
                else shifted(protocol.activation, 0 if action == "set" else protocol.distance)
            )
            if activation is not None:
                residues = activation.residues
            lower, upper = (
                (
                    protocol.distance if action != "set" else 0,
                    protocol.distance if action == "set" else 0,
                )
                if activation is None
                else (
                    0 if activation.first.end else activation.first.offset,
                    activation.last.offset if activation.last.end else 0,
                )
            )
            command = Command(
                self.number,
                owner[0],
                task,
                kind,
                protocol.name,
                lower,
                upper,
                protocol.period,
                residues,
            )
            self.number += 1
            self.events[owner[0], protocol.name, action] = command
            for residue in range(self.graph.period):
                if residue % command.period not in command.residues:
                    continue
                before = {}
                key = owner, residue
                if action == "set":
                    before["issue"] = self.site(
                        owner, "end" if task.pipe == "S" else "admit_issue", residue
                    )
                    before["queue"] = self.sends.get(key, self.site(owner, "begin", residue))
                    before["retire"] = self.site(owner, "prefix", residue)
                else:
                    before = {
                        point: self.site(owner, point, residue)
                        for point in (
                            ("issue", "queue", "retire")
                            if action == "barrier"
                            else ("issue", "queue")
                        )
                    }
                    before.update(self.waits.get(key, {}))

                def node(point, c=command, r=residue, active=activation):
                    index = self.graph.node(c, r, point)
                    if active is not None:
                        self.node_activations[index] = active
                    return index, 0

                actual = instruction_points(
                    command,
                    before,
                    node,
                    self.edge,
                )
                points[owner[0], action, residue] = actual
                if action == "set":
                    self.sends[key] = self._forward(
                        command, residue, "queue", before["queue"], actual["begin"]
                    )
                    self.edge(self.sends[key], self.site(owner, "after_queue", residue))
                else:
                    for point, value in prefix_inputs(command, actual).items():
                        after = self._forward(command, residue, point, before[point], value)
                        # Native admission cuts carry the acquire forward on
                        # the actual destination pipe/controller only.
                        self.edge(after, self.site(owner, "admit_" + point, residue))
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
        conditional = any(leaf.cases for leaf in self.program.leaves.values())
        answers = self.resource_covered_many(
            (uid, consumed, republished) for uid, consumed, republished, _ in self.resource_pairs
        )
        for (uid, _, _, resource), valid in zip(self.resource_pairs, answers, strict=True):
            poll("child event reuse verification")
            require(
                valid,
                (
                    f"{uid}/{resource.channel}: conditional child event {resource.event_id} may be re-Set before its previous Wait"
                    if conditional
                    else f"{uid}/{resource.channel}: child event {resource.event_id} may be re-Set before its previous parent call Wait"
                ),
            )
        return len(self.resource_pairs) * self.graph.period

    def resource_covered(self, uid, consumed, republished):
        """A selected ID's final Wait is consumed before this call republishes it."""
        return self.resource_covered_many(((uid, consumed, republished),))[0]

    def resource_covered_many(self, resources):
        """Seed every real resource/case before shared completion propagation.

        Conditional children also need their actual first-to-last reuse path:
        an interior invocation can omit the channel entirely. Keep that check
        per resource and per native case, while registering its origins once
        for the whole resource batch. Neither protocol placement nor IDs change.
        """
        resources = tuple(resources)
        if not resources:
            return []
        groups, queries = [], []
        if self.node_activations or self.completion.transfer_state is not None:
            from .completion import CountDomain, Position

            pairs = []
            for uid, consumed, republished in resources:
                poll("child event resource batch", call=uid)
                current = tuple(
                    (a, b)
                    for a in self.completion.commands_for(uid, consumed)
                    for b in self.completion.commands_for(uid, republished)
                )
                pairs.append(current)
                first = len(queries)
                queries.extend((a, b, 1, legacy_activation(1)) for a, b in current)
                groups.append((first, len(queries)))
            answers = self.guarded().covered_many(queries)
            results = [all(answers[first:last]) for first, last in groups]
            anchored_groups, anchored_queries = [], []
            for index, (uid, _, _) in enumerate(resources):
                if not results[index] or not self.program.leaves[uid].cases:
                    continue
                first = len(anchored_queries)
                anchored_queries.extend(
                    (
                        a,
                        b,
                        Position(False, 0),
                        Position(True, 0),
                        Activation(CountDomain(1, 0, minimum=2)),
                    )
                    for a, b in pairs[index]
                )
                anchored_groups.append((index, first, len(anchored_queries)))
            if anchored_queries:
                answers = self.anchored().covered_many(anchored_queries)
                for index, first, last in anchored_groups:
                    results[index] = all(answers[first:last])
            return results
        for uid, consumed, republished in resources:
            poll("child event resource batch", call=uid)
            first = len(queries)
            queries.extend(
                self.completion.query(uid, consumed, r, uid, republished, r + 1, minimum=1)
                for r in range(self.graph.period)
            )
            groups.append((first, len(queries)))
        answers = self.graph.covered_many(queries)
        return [all(answers[first:last]) for first, last in groups]


def _analyses(program):
    from .scope_analysis import verify_children

    return verify_children(program)


def _scope_access(program):
    from .scope_access import analyze_scope_access

    access = analyze_scope_access(program, definitions=True)
    if any(leaf.cases for leaf in program.leaves.values()):
        from .scope_guarded_access import analyze_guarded_access

        access = analyze_guarded_access(program, base=access)
    if access.definition_roots:
        from .scope_lifetimes import plan_lifetimes

        plan = plan_lifetimes(program, access)
        access = replace(
            access,
            obligations=plan.dependencies,
            lifetime_boundaries=plan.boundaries,
            lifetime_report={
                "reordered_roots": list(plan.reordered_roots),
                "retained_logical_roots": dict(plan.fallbacks),
                "visits": plan.visits,
            },
        )
    return access


def _query_boundary_summary(program, access):
    if not program.invocations:
        return None, ()
    from .scope_query_summary import query_native_sites, summarize_query_patterns

    summary = summarize_query_patterns(program, access)
    return summary, query_native_sites(summary, access)


def _scope_requirements(access):
    """Keep conditional retirement clauses separate from ordinary protocols."""
    ordinary, guarded = [], {}
    tasks = {(a, t.uid): t for a, data in access.loop.actors.items() for t in data.program.tasks}
    for obligation in access.obligations:
        if hasattr(obligation, "source_position"):
            continue  # A fixed first/last acquire must prove the entire span below.
        if not isinstance(obligation, ScopedDependence):
            ordinary.append(obligation)
            continue
        require(
            obligation.source[0] == obligation.destination[0],
            "conditional ownership currently requires actor-private storage",
        )
        guarded.setdefault(
            (obligation.source, obligation.destination, obligation.distance, obligation.activation),
            [],
        ).append(obligation)
    result = _requirements(access.loop, ordinary)
    for (source, destination, distance, activation), obligations in guarded.items():
        validate_activation(activation, distance=distance)
        kind = "barrier" if tasks[source].pipe == tasks[destination].pipe else "local"
        result.append(
            (
                ScopeProtocol(
                    "",
                    kind,
                    () if kind == "barrier" else (source,),
                    (destination,),
                    distance,
                    activation.period,
                    activation.residues,
                    activation=activation,
                ),
                tuple(obligations),
            )
        )
    return result


def _scope_period(access, protocols):
    periods = [p.period for p in protocols]
    periods.extend(
        o.activation.period for o in access.obligations if isinstance(o, ScopedDependence)
    )
    return bounded_period(lcm(*periods), "parent protocols") if periods else 1


def _encode(access, owner):
    uid, site = access.endpoints[owner]
    return "/".join((owner[0], uid, site.task[1], str(site.position.offset)))


def _install(program, access, protocols):
    root = program.root.body.block
    first = (
        next(iter(program.kernels.values()))
        if len(program.kernels) > 1
        else next(iter(program.invocations.values())).op
        if program.invocations
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
        if p.activation is not None:
            declaration.attributes["activation"] = encode_activation(p.activation)
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
    from .scope_resources import capture_child_resources

    answers = proof.resource_covered_many(
        (uid, consumed, republished) for uid, consumed, republished, _ in proof.resource_pairs
    )
    if all(answers):
        proof.child_resources = capture_child_resources(proof, answers)
        return proof, _leaf_banks(program, analyses)
    require(
        not any(leaf.cases for leaf in program.leaves.values()),
        "conditional child event reuse needs a proved parent lifetime; width adaptation is unsupported",
    )
    choices = event_choices(program, analyses)
    proof = ParentCompletion(
        program,
        analyses,
        access,
        proof.graph.period,
        resource_inputs=resource_inputs(choices),
        extra_protocols=selected,
    )
    for protocol in selected:
        proof.insert(protocol)
    proof.graph.potential()
    used = _configuration_banks(program, analyses)
    established = set()
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
        loop, channel_map, sequence, graph, potential = analyses[uid]
        for op in loop.protocol_scope.walk():
            if isinstance(op, LoopChannelOp) and op.sym_name.data == name:
                op.attributes["event_ids"] = _ints(ids)
        channel_map = dict(channel_map)
        channel_map[name] = replace(channel_map[name], ids=ids)
        analyses[uid] = loop, channel_map, sequence, graph, potential
        established.update(
            (uid, *resource_sites(uid, choice.bank[0], choice.count, resource))
            for resource in option.resources
        )
        used[choice.bank].update(ids)
    pairs = []
    for uid, analysis in analyses.items():
        leaf = program.leaves[uid]
        for resource in summarize_local_events(analysis[0]):
            if resource.domain.contains(analysis[0].constant_count):
                sites = resource_sites(uid, leaf.actor, leaf.loop.constant_count, resource)
                require(
                    (uid, *sites) in established, "assigned child rotation lost its lifetime proof"
                )
                pairs.append((uid, *sites, resource))
    proof.resource_pairs = list(dict.fromkeys(pairs))
    proof.child_resources = capture_child_resources(proof, (True for _ in proof.resource_pairs))
    return proof, used


@bounded
def autosync_scopes(module, *, _capture=None, _facts=None):
    result = module.clone() if _facts is None else module
    require(
        not any(
            isinstance(
                op, (ScopeChannelOp, ScopeSyncOp, QueryChannelOp, QuerySyncOp, LoopChannelOp)
            )
            for op in result.walk()
        ),
        "nested AutoSync requires unsynchronized physical calls",
    )
    program = extract_scopes(result) if _facts is None else _facts.extract_scopes(result)
    planning, parent = _plan_scope_parent(program, _facts=_facts)
    query_plan = None
    if program.invocations:
        from .scope_query_planning import plan_query_memory

        query_plan = plan_query_memory(planning, parent)
    report, checked = _complete_scope_plan(planning, parent, query_plan)
    if _capture is not None:
        _capture((report, checked, 0.0))
    return result


def _plan_scope_parent(program, *, _facts=None) -> tuple[PlanningScopes, ParentCompletion]:
    """Install child/parent protocols in an owned actual program.

    This composes one invocation only. Query normalization retains the outer
    lifetime contracts, but does not discharge them. The public entry point
    must separately prove those contracts before returning an executable plan.
    """
    from .loop_graph import channels, commands
    from .scope_analysis import ScopeAnalyses

    banks = defaultdict(lambda: defaultdict(int))
    analyses, reports, leaves = ScopeAnalyses(), {}, {}
    for uid, leaf in program.leaves.items():
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
        if leaf.cases:
            from .scope_variants import plan_child_variants

            variants = plan_child_variants(
                leaf.stream,
                leaf.call,
                {label: case.loop for label, case in leaf.cases.items()},
                live_in=roots,
                banks=banks[leaf.actor],
                _facts=_facts,
            )
            # The union descriptor is emission-only; each guarded case keeps
            # the completed native analysis under its original activation.
            from .scope_variants import describe_child

            values = dict(leaf.value_map)
            union = describe_child(leaf.stream, leaf.call, value_mapper=values)
            data.stream.parent.insert_op_before(union, data.stream)
            data.stream.parent.erase_op(data.stream)
            operations = {op.uid.data: op for op in union.walk() if isinstance(op, TaskOp)}
            local_actor = next(iter(loop.actors))
            loop = replace(
                loop,
                region=union,
                actors={local_actor: replace(data, stream=union, operations=operations)},
            )
            channel_map = channels(loop)
            sequences, _, _ = commands(loop, channel_map)
            analyses[uid] = loop, channel_map, sequences, None, None
            current = {}
            for label, (report, analysis) in variants.plans.items():
                current[label] = report
                analyses.cases[uid, label] = analysis
            reports[uid] = {"native_templates": len(data.program.tasks), "cases": current}
            leaves[uid] = replace(leaf, loop=loop, value_map=values)
            continue
        reports[uid], analyses[uid] = _plan_leaf(
            loop, live_in=roots, banks=banks[leaf.actor], _facts=_facts
        )
        aliases = {leaf.value_map[c.op.ref]: c.op.ref for c in leaf.captures}
        replacement = data.stream.clone(value_mapper=aliases)
        leaf.call.body.block.insert_op_before(replacement, leaf.stream)
        leaf.call.body.block.erase_op(leaf.stream)
        leaves[uid] = replace(
            leaf,
            stream=replacement,
            value_map={actual: normalized for normalized, actual in aliases.items()},
        )
    program = replace(program, leaves=leaves)
    access = _scope_access(program)
    query_summary, boundary_sites = _query_boundary_summary(program, access)
    candidates = _scope_requirements(access)
    period = _scope_period(access, (p for p, _ in candidates))
    proof = ParentCompletion(program, analyses, access, period, boundary_sites=boundary_sites)
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
    if proof.node_activations:
        # Register the actual candidate sources together. Otherwise every
        # candidate injects a few new bits and scans their reachable graph
        # again. Bitsets share that work when origins enter one propagation.
        # These answers add no edges: selection still rechecks each candidate
        # against only the protocols already accepted below.
        proof.covered_obligations(o for _, obligations in ordered for o in obligations)
    selected = []
    for candidate, obligations in ordered:
        poll("parent synchronization selection")
        if all(proof.covered_obligations(obligations)):
            continue
        p = ScopeProtocol(
            f"scope_c{len(selected)}",
            candidate.kind,
            candidate.sources,
            candidate.destinations,
            candidate.distance,
            candidate.period,
            candidate.residues,
            activation=getattr(candidate, "activation", None),
        )
        proof.insert(p)
        remaining = tuple(o for o in obligations if not establishes_directly(p, o))
        require(
            not remaining or all(proof.covered_obligations(remaining)),
            "parent protocol cannot cover its actual child access obligation",
        )
        selected.append(p)
    for obligation in access.obligations:
        if hasattr(obligation, "source_position"):
            require(
                proof.covered(obligation),
                f"uncovered anchored {obligation.hazard} on {obligation.storage}: "
                "first/last native protocols do not cover the complete access span",
            )
    proof.graph.potential()
    proof, used = _allocate_child_resources(program, analyses, access, proof, selected)
    # Resource substitutions preserve the leaf proof but not its old report's
    # concrete ID tuples. Derive only those fields from the assigned map.
    for uid, analysis in analyses.items():
        if program.leaves[uid].cases:
            continue
        loop, channel_map, _, _, _ = analysis
        tasks = {(a, t.uid): t for a, d in loop.actors.items() for t in d.program.tasks}
        assigned = defaultdict(set)
        for channel in channel_map.values():
            if channel.kind == "local":
                source, destination = channel.sources[0], channel.destinations[0]
                bank = f"{source[0]}:{tasks[source].pipe}:{tasks[destination].pipe}"
                assigned[bank].update(channel.ids)
        reports[uid]["event_ids"] = {bank: sorted(ids) for bank, ids in assigned.items()}
    from .scope_event_pools import allocate_parent_pools

    pool_results = []
    selected, _ = allocate_parent_pools(selected, proof, used, _capture=pool_results.append)
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
    require(
        len(proof.protocols) == len(allocated)
        and all(
            replace(before, ids=after.ids) == after
            for before, after in zip(proof.protocols, allocated, strict=True)
        ),
        "parent allocation changed a planned protocol beyond its event IDs",
    )
    _install(program, access, allocated)
    # IDs do not change completion topology, native effects, or the selected
    # child cuts. Even width adaptation retains its exact candidate projection;
    # resource_pairs below contains only the actually assigned rotations.
    proof.protocols[:] = allocated
    planning = PlanningScopes(
        program,
        analyses,
        access,
        {p.name: p for p in allocated},
        query_summary,
        child_reports=reports,
    )
    from .scope_resources import establish_invocation_resources

    actual, _ = _read_protocols(program, access, analyses)
    require(actual == planning.channels, "materialized parent protocols changed their plan")
    resources = establish_invocation_resources(
        planning, proof, children=proof.child_resources, pools=pool_results[0]
    )
    planning = replace(planning, resource_proof=resources)
    return planning, proof


def _read_protocols(program, access, analyses):
    from .scope_variants import scope_operations

    by_encoding = {_encode(access, key): key for key in access.endpoints}
    tasks = {(a, t.uid): t for a, d in access.loop.actors.items() for t in d.program.tasks}
    channels, expected = {}, {}
    expected_order, actual_order = defaultdict(list), defaultdict(list)
    reserved = _leaf_banks(program, analyses)
    borrow_slots = None
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
            (distance >= 0 or op.activation is not None)
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
            if reserved[key].intersection(ids):
                # Full lifetime checking needs the actual parent transitions.
                # Only a real child owner can defer an overlap to that proof;
                # kernel-entry configuration resources never enter this path.
                if borrow_slots is None:
                    from .scope_event_borrow import child_slots

                    borrow_slots = child_slots(program, analyses)
                require(
                    all(
                        (key, event_id) in borrow_slots
                        for event_id in ids
                        if event_id in reserved[key]
                    ),
                    "parent event overlaps a child/configuration resource without an actual child owner",
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
            name,
            kind,
            sources,
            destinations,
            distance,
            period,
            residues,
            ids,
            activation=None
            if op.activation is None
            else decode_activation(
                op.activation, distance=distance, period=period, residues=residues
            ),
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
        for op, _ in scope_operations(leaf.stream):
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
            elif isinstance(op, QuerySyncOp):
                # The query protocol reader checks the joint instruction
                # order. Retain the original parent placement while rebuilding
                # the invocation-only proof; query effects enter its adapter.
                pass
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
    program = extract_scopes(module)
    report, checked, parent = _verify_scope_parent(program)
    if program.invocations:
        from .scope_query_verification import verify_query_invocations

        query_report = verify_query_invocations(checked, parent)
        checked = replace(checked, query_report=query_report)
        report.update(
            cross_query_verified=True,
            verification="actual child, parent and serial query memory/event/progress proofs",
            query_verification=query_report,
        )
    return (report, checked) if _details else report


def _verify_scope_parent(program):
    """Rebuild the actual invocation proof, retaining it for outer composition.

    An invocation-local report alone never verifies serial query execution.
    """
    access = _scope_access(program)
    reports, analyses = _analyses(program)
    channels, _ = _read_protocols(program, access, analyses)
    period = _scope_period(access, channels.values())
    query_summary, boundary_sites = _query_boundary_summary(program, access)
    if program.invocations:
        from .scope_query_protocols import read_query_protocols

        boundary_sites += tuple(
            (endpoint.call, _point(endpoint.site, point))
            for protocol in read_query_protocols(program, access)
            for endpoint in (protocol.source, protocol.target)
            for point in (
                "end",
                "prefix",
                "begin",
                "issue",
                "queue",
                "retire",
                "admit_issue",
                "admit_queue",
                "after_queue",
            )
        )
    proof = ParentCompletion(
        program,
        analyses,
        access,
        period,
        extra_protocols=channels.values(),
        boundary_sites=boundary_sites,
    )
    for channel in channels.values():
        proof.insert(channel)
    potential = proof.graph.potential()
    for obligation, valid in zip(
        access.obligations, proof.covered_obligations(access.obligations), strict=True
    ):
        require(
            valid,
            f"missing scoped {obligation.hazard}: {obligation.source} -> {obligation.destination}, "
            f"distance {getattr(obligation, 'distance', 'anchored')} on {obligation.storage}[{obligation.start}:{obligation.stop}]",
        )
    checked = CheckedScopes(program, analyses, access, channels, query_summary)
    from .scope_resources import establish_invocation_resources

    resources = establish_invocation_resources(checked, proof)
    checked = replace(checked, resource_proof=resources)
    report = _scope_report(
        program, access, reports, analyses, channels, proof, query_summary, resources, potential
    )
    return report, checked, proof


def _scope_report(
    program, access, reports, analyses, channels, proof, query_summary, resources, potential
):
    child_lifetimes = resources.child_lifetime_queries
    pool_statistics = dict(resources.pool_statistics)
    banks, counters = resources.used_banks, resources.cross_counter_ids
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
        "scope_completion_facts": (
            proof.guarded().statistics()
            if proof.node_activations
            else proof.graph.completion_statistics()
        ),
        "potential_work": dict(proof.graph.potential_work),
        "scope_anchored_facts": None if proof._anchored is None else proof._anchored.statistics(),
        "parent_memory_obligations": len(access.obligations),
        "parent_access_visits": access.visits,
        "definition_lifetimes": access.lifetime_report,
        "parent_channels": len(channels),
        "parent_binary_lifetime_queries": pool_statistics["self_queries"],
        "child_parent_lifetime_queries": child_lifetimes,
        "parent_event_pool_proof": pool_statistics,
        "event_ids": {":".join(key): sorted(ids) for key, ids in banks.items()},
        "cross_counter_ids": sorted(counters),
        "progress_stride": potential[0],
        "terminal_drain": "only kernel actor return; child exit is not completion",
    }
    if program.invocations:
        report["verification"] = (
            "invocation-local child and parent proof only; "
            "cross-query memory and event lifetimes remain unverified"
        )
        report["cross_query_verified"] = False
        report["query_memory_summary"] = query_summary.statistics
    return report


def _complete_scope_plan(planning, parent, query_plan):
    """Consume established child/parent facts and discharge only query deltas.

    Linear instruction correspondence checks the materializer, without rebuilding
    accesses, completion graphs or any already established resource obligation.
    """
    channels, _ = _read_protocols(planning.program, planning.access, planning.analyses)
    require(channels == planning.channels, "materialized parent protocols changed their plan")
    resources = planning.resource_proof
    if not planning.program.invocations:
        resources.check(planning, parent)
    checked = CheckedScopes(
        planning.program,
        planning.analyses,
        planning.access,
        channels,
        planning.query_summary,
        resource_proof=resources,
    )
    report = _scope_report(
        planning.program,
        planning.access,
        planning.child_reports,
        planning.analyses,
        channels,
        parent,
        planning.query_summary,
        resources,
        parent.graph.potential(),
    )
    if planning.program.invocations:
        from .scope_query_verification import verify_query_invocations

        query_report = verify_query_invocations(
            checked, parent, resources=resources, query_plan=query_plan
        )
        checked = replace(checked, query_report=query_report)
        report.update(
            cross_query_verified=True,
            verification="actual child, parent and serial query memory/event/progress proofs",
            query_verification=query_report,
        )
    report["verification_origin"] = "composed owned transformation contracts"
    return report, checked

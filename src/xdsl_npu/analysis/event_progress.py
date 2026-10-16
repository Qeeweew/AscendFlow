"""Compose guarded blocking interfaces on actual invocation regions.

Only selected notification effects and real queue/controller gates participate.
A child exports actual ports and structural channel relations, never a native
completion at child exit. Loop streams keep their clock; finite ports keep the
same parameter expressions in every condition. No runtime iteration or branch
history is expanded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind
from ..scalar import Expr, expression, literal, substitute
from ..target.ascend910b import Actor
from .event_contracts import ProgressPoint, StructuredProgress, _Order
from .event_domains import _normalize
from .structured_protocols import BoundPlacement, LogicalNotification, NotificationKind
from .structured_schedule import ActionGuard, Comparison


@dataclass(frozen=True, slots=True)
class ProgressPort:
    notification: LogicalNotification
    bound: BoundPlacement
    phase: str
    bindings: tuple[tuple[Expr, Expr], ...] = ()

    @property
    def actor(self):
        return self.bound.actor

    @property
    def engine(self):
        return (
            self.notification.source_engine
            if self.phase == "set"
            else self.notification.destination_engine
        )


@dataclass(frozen=True, slots=True)
class GuardedTransfer:
    source: int
    destination: int
    steps: int
    guard: ActionGuard
    reason: str
    exact: bool = True
    projections: tuple = ()


@dataclass(frozen=True, slots=True)
class ExcludedCycle:
    relations: tuple[int, ...]
    condition: ActionGuard


@dataclass(frozen=True, slots=True)
class ChannelTransfer:
    source: ProgressPort
    destination: ProgressPort
    guard: ActionGuard
    exact: bool
    projections: tuple


@dataclass(frozen=True, slots=True)
class ChannelProjection:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    relations: tuple[GuardedTransfer, ...]
    windows: tuple


@dataclass(frozen=True, slots=True)
class InvocationBinding:
    operation: Operation
    coordinates: tuple[tuple[Operation, Expr], ...]
    substitutions: tuple[tuple[Expr, Expr], ...]
    condition: ActionGuard


@dataclass(frozen=True, slots=True)
class StreamProgress:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    relations: tuple[GuardedTransfer, ...]
    stride: int
    rank: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ControllerEntries:
    actor: Actor
    calls: tuple[tuple[Operation, int, int], ...]


@dataclass(frozen=True, slots=True)
class RegionProgress:
    operation: Operation
    ports: tuple[ProgressPort, ...]
    relations: tuple[GuardedTransfer, ...]
    channels: tuple[ChannelTransfer, ...]
    children: tuple[RegionProgress, ...]
    stream: StreamProgress | None
    excluded_cycles: tuple[ExcludedCycle, ...] = ()
    controller_entries: tuple[ControllerEntries, ...] = ()


def _guard(tests):
    return ActionGuard(tuple(dict.fromkeys(tests)))


def _key(port):
    return (
        id(port.notification),
        port.phase,
        port.actor,
        port.bound.operation,
        port.bound.coordinates,
        port.bindings,
    )


class _Composer:
    def __init__(self, domains):
        self.domains = domains
        self.control = domains.control
        self.dispatch = domains.dispatch
        self.root = domains.memory.module.body.block.first_op
        self.events = tuple(
            n for n in domains.notifications if n.mechanism is not NotificationKind.PIPE_FENCE
        )
        self.parents = {self.root: None}
        self.children = {self.root: []}
        for loop in self.control.loops:
            chain = tuple(
                frame.loop.operation
                for frame in self.control.frames[loop].ancestors()
                if frame.loop is not None
            )
            parent = chain[-2] if len(chain) > 1 else self.root
            self.parents[loop] = parent
            self.children.setdefault(parent, []).append(loop)
            self.children.setdefault(loop, [])
        self.owners = {n: self._owner(n) for n in self.events}
        self.components = []

    def fail(self, region, message, witness=()):
        raise CompileError(
            message,
            kind=ErrorKind.UNSUPPORTED,
            phase="structured-event-progress",
            location=str(region.location),
            witness=witness,
        )

    def _ancestors(self, operation):
        return (self.root,) + tuple(
            frame.loop.operation
            for frame in self.control.frames[operation].ancestors()
            if frame.loop is not None
        )

    def _inside(self, bound, region):
        return region in self._ancestors(bound.operation)

    def _owner(self, notification):
        bounds = notification.sources + notification.consumers
        paths = tuple(self._ancestors(p.operation) for p in bounds)
        common = self.root
        for candidates in zip(*paths):
            if len(set(candidates)) != 1:
                break
            common = candidates[0]
        while common is not self.root:
            strict = self._ancestors(common)[:-1]
            reference = dict(bounds[0].coordinates)
            assumptions = self.domains.tests[notification]
            if all(
                self.domains._prove(
                    Comparison(reference[loop], "==", dict(bound.coordinates)[loop]), assumptions
                )
                for loop in strict
                if loop is not self.root
                for bound in bounds[1:]
            ):
                break
            common = self.parents[common]
        return common

    def _ports(self, region):
        result = []
        for notification in self.events:
            if not self._is_ancestor(self.owners[notification], region):
                continue
            for phase, bounds in (("set", notification.sources), ("wait", notification.consumers)):
                result.extend(
                    ProgressPort(notification, bound, phase)
                    for bound in bounds
                    if self._inside(bound, region)
                )
        return tuple(result)

    def _is_ancestor(self, ancestor, region):
        while region is not None:
            if region is ancestor:
                return True
            region = self.parents[region]
        return False

    def _ambient(self, port):
        return self._instance(port).guard.comparisons

    def _instance(self, port):
        return self.domains._instance(port.bound, dict(port.bindings))

    def _bind_port(self, port, bindings):
        combined = dict(bindings)
        combined.update((symbol, substitute(value, bindings)) for symbol, value in port.bindings)
        return replace(
            port,
            bindings=tuple(
                (symbol, value) for symbol, value in combined.items() if symbol != value
            ),
        )

    def _stream_port(self, port, region):
        """Rebase logical ancestor translations onto one actual invocation."""
        bindings = {}
        coordinates = dict(self._instance(port).coordinates)
        for loop in self._ancestors(region)[:-1]:
            if loop is self.root:
                continue
            ordinal = self.control.loops[loop].ordinal
            offset = _normalize(
                expression("sub", coordinates[loop], ordinal), self.domains.facts
            ).constant
            if offset is None:
                self.fail(region, "stream ancestor binding is not an actual unit-rate invocation")
            bindings[ordinal] = expression("sub", ordinal, literal(offset))
        return self._bind_port(port, bindings)

    def _invocation_guard(self, region, *ports):
        if region is self.root or len(ports) < 2:
            return ActionGuard()
        instances = tuple(self._instance(port) for port in ports)
        # Queue endpoints may be active at different iterations of this loop.
        # Their local membership cannot disprove a shared ancestor invocation.
        assumptions = self._export_guard(
            _guard(tuple(test for item in instances for test in item.guard.comparisons)), region
        ).comparisons
        reference = dict(instances[0].coordinates)
        tests = []
        for loop in self._ancestors(region)[:-1]:
            if loop is self.root:
                continue
            if loop not in reference or any(
                loop not in dict(item.coordinates) for item in instances
            ):
                return None
            for item in instances[1:]:
                test = Comparison(reference[loop], "==", dict(item.coordinates)[loop])
                if not self.domains._prove(test, assumptions):
                    tests.append(test)
        if self._impossible(assumptions + tuple(tests)):
            return None
        return _guard(tuple(tests))

    def _window_guard(self, port, region):
        guard = self.domains._window(port.notification, region).nonempty
        bindings = dict(port.bindings)
        return _guard(
            tuple(
                Comparison(
                    substitute(test.lhs, bindings),
                    test.predicate,
                    substitute(test.rhs, bindings),
                )
                for test in guard.comparisons
            )
        )

    def _impossible(self, tests):
        opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
        tests = tuple(dict.fromkeys(tests))
        return any(
            self.domains._prove(
                Comparison(test.lhs, opposite[test.predicate], test.rhs),
                tests[:index] + tests[index + 1 :],
            )
            for index, test in enumerate(tests)
        )

    def _route(self, wait, source):
        return wait.actor is source.actor and (
            self.domains._gate(wait.notification) or wait.engine is source.engine
        )

    def _position(self, port, region):
        task = self.dispatch.tasks[port.bound.operation, port.actor]
        path = self.domains._path(task)
        owner = next(index for index, scope in enumerate(path) if scope.operation is region)
        if task.scope.operation is region:
            return task.stage, (task.lexical, 1 if port.phase == "set" else 0)
        child = path[owner + 1]
        return child.parent_stage, (child.parent_lexical, 1)

    def _order_condition(self, wait, source):
        """Necessary order condition; exact whenever one unresolved rank remains."""
        before, after = self._instance(wait), self._instance(source)
        assumptions = before.guard.comparisons + after.guard.comparisons
        relation = self.domains._may_follow(
            wait.notification,
            before,
            source.notification,
            after,
            assumptions,
        )
        if relation is _Order.IMPOSSIBLE:
            return None
        if relation is _Order.PROVED:
            return _guard(assumptions), True, True
        a = self.dispatch.tasks[wait.bound.operation, wait.actor]
        b = self.dispatch.tasks[source.bound.operation, source.actor]
        ap, bp = self.domains._path(a), self.domains._path(b)
        common = 0
        while common < min(len(ap), len(bp)) and ap[common] is bp[common]:
            common += 1
        equal = []
        for index in range(common):
            region = ap[index].operation
            ag, ar = self._position(wait, region)
            bg, br = self._position(source, region)
            av = dict(before.coordinates).get(region, literal(0))
            bv = dict(after.coordinates).get(region, literal(0))
            delta = _normalize(
                expression(
                    "sub", expression("add", bv, literal(bg)), expression("add", av, literal(ag))
                ),
                self.domains.facts,
            )
            if delta.constant == 0:
                continue
            predicate = ">=" if index + 1 < common or br >= ar else ">"
            test = Comparison(delta, predicate, literal(0))
            return _guard(assumptions + tuple(equal) + (test,)), index + 1 == common, False
        return _guard(assumptions), False, False

    def _stream_offset(self, port, loop):
        value = dict(self._instance(port).coordinates).get(loop)
        if value is None:
            return None
        ordinal = self.control.loops[loop].ordinal
        return _normalize(expression("sub", value, ordinal), self.domains.facts).constant

    def _gate_step(self, wait, source, loop):
        wo, so = self._stream_offset(wait, loop), self._stream_offset(source, loop)
        if wo is None or so is None:
            self.fail(loop, "current-loop port is not a unit-rate translation of its actual clock")
        ws, wr = self._position(wait, loop)
        ss, sr = self._position(source, loop)
        shift = wo + ws - so - ss + int(sr < wr)
        ordinal = self.control.loops[loop].ordinal
        base = self._instance(source)
        shifted = self.domains._shift_instance(
            base, {ordinal: expression("add", ordinal, literal(shift))}
        )
        old = self._instance(wait)
        tests = old.guard.comparisons + shifted.guard.comparisons
        relation = self.domains._may_follow(
            wait.notification, old, source.notification, shifted, tests
        )
        if relation is _Order.IMPOSSIBLE:
            shift += 1
            shifted = self.domains._shift_instance(
                base, {ordinal: expression("add", ordinal, literal(shift))}
            )
            tests = old.guard.comparisons + shifted.guard.comparisons
        # The physical minimum is a lower bound for *every* later active
        # gate. It need not itself have two active endpoints (for example,
        # even/odd notification streams). Keep the envelope with invocation
        # invariants, never a false guard sampled at that minimum shift.
        invariant = (
            self._window_guard(wait, loop).comparisons
            + self._window_guard(source, loop).comparisons
        )
        return shift, _guard(invariant), False

    def _rank(self, region, ports, arcs):
        count, offsets = len(ports), [0] * len(ports)
        stride, predecessor = count + 1, [None] * count
        for _ in range(count):
            changed = None
            for index, arc in enumerate(arcs):
                required = offsets[arc.source] + 1 - stride * arc.steps
                if required > offsets[arc.destination]:
                    offsets[arc.destination] = required
                    predecessor[arc.destination] = index
                    changed = arc.destination
            if changed is None:
                return stride, tuple(offsets)
        if not count:
            return 1, ()
        cursor = changed
        for _ in range(count):
            cursor = arcs[predecessor[cursor]].source
        first, witness = cursor, []
        while not witness or cursor != first:
            arc = arcs[predecessor[cursor]]
            witness.append(
                f"{ports[arc.source].notification.name}:{ports[arc.source].phase} -> {ports[arc.destination].notification.name}:{ports[arc.destination].phase}, clock step {arc.steps}: {arc.reason}"
            )
            cursor = arc.source
        self.fail(
            region,
            "selected structured notifications have no increasing blocking-interface rank",
            witness,
        )

    def _finite_proof(self, region, ports, arcs):
        """SCC conditions are conjoined once, never expanded into path histories."""
        adjacency = [[] for _ in ports]
        for index, arc in enumerate(arcs):
            adjacency[arc.source].append((arc.destination, index))
        counter, stack, onstack, ids, low, groups = 0, [], set(), {}, {}, []

        def visit(node):
            nonlocal counter
            ids[node] = low[node] = counter
            counter += 1
            stack.append(node)
            onstack.add(node)
            for destination, _ in adjacency[node]:
                if destination not in ids:
                    visit(destination)
                    low[node] = min(low[node], low[destination])
                elif destination in onstack:
                    low[node] = min(low[node], ids[destination])
            if low[node] == ids[node]:
                group = []
                while True:
                    member = stack.pop()
                    onstack.remove(member)
                    group.append(member)
                    if member == node:
                        break
                groups.append(tuple(group))

        for node in range(len(ports)):
            if node not in ids:
                visit(node)
        excluded = []
        for group in groups:
            members = set(group)
            internal = tuple(
                index
                for index, arc in enumerate(arcs)
                if arc.source in members and arc.destination in members
            )
            if len(group) == 1 and not any(
                arcs[index].source == arcs[index].destination for index in internal
            ):
                continue
            # For a simple structural SCC this conjunction is the complete
            # cycle condition. Multiple guarded cycles need a stronger local
            # projection contract; choosing one path would be unsound.
            if any(
                sum(arcs[index].source == node for index in internal) != 1
                or sum(arcs[index].destination == node for index in internal) != 1
                for node in group
            ):
                self.fail(
                    region,
                    "conditional region SCC needs an exact joint interface projection",
                    tuple(arcs[index].reason for index in internal),
                )
            condition = _guard(
                tuple(test for index in internal for test in arcs[index].guard.comparisons)
            )
            if self._impossible(condition.comparisons):
                excluded.append(ExcludedCycle(internal, condition))
                continue
            exact = all(arcs[index].exact for index in internal)
            message = (
                "selected finite region has a possible blocking cycle under its shared activation condition"
                if exact
                else "child blocking relation cannot be projected exactly for a possible parent cycle"
            )
            self.fail(
                region,
                message,
                tuple(
                    f"{ports[arcs[index].source].notification.name}:{ports[arcs[index].source].phase} -> {ports[arcs[index].destination].notification.name}:{ports[arcs[index].destination].phase}: {arcs[index].reason}"
                    for index in internal
                ),
            )
        # DAG components and exclusions certify the finite graph; no
        # unconditional numeric rank is claimed for its guarded cyclic arcs.
        return tuple(excluded)

    def _export_guard(self, guard, region):
        return _guard(
            tuple(
                test
                for test in guard.comparisons
                if self.control.invariant_over(test.lhs, region)
                and self.control.invariant_over(test.rhs, region)
            )
        )

    def _graph(self, region, ports, stream, arcs, children):
        """Own one graph; import each child's boundary summary once."""
        stream_ports = () if stream is None else stream.ports
        graph_ports = list(ports + stream_ports)
        graph_arcs = list(arcs)
        if stream is not None:
            graph_arcs.extend(
                replace(
                    arc, source=arc.source + len(ports), destination=arc.destination + len(ports)
                )
                for arc in stream.relations
            )
        index = {_key(port): i for i, port in enumerate(graph_ports)}

        def endpoint(port):
            key = _key(port)
            if key not in index:
                index[key] = len(graph_ports)
                graph_ports.append(port)
            return index[key]

        stream_events = {port.notification for port in stream_ports}
        for child in children:
            for arc in child.relations:
                a, b = child.ports[arc.source], child.ports[arc.destination]
                # This transfer already belongs to the owned stream proof.
                if a.notification in stream_events and b.notification in stream_events:
                    continue
                graph_arcs.append(replace(arc, source=endpoint(a), destination=endpoint(b)))
            for channel in child.channels:
                graph_arcs.append(
                    GuardedTransfer(
                        endpoint(channel.source),
                        endpoint(channel.destination),
                        0,
                        channel.guard,
                        "actual child exported channel",
                        channel.exact,
                        channel.projections,
                    )
                )
        # The two local proofs already own their direct gates. Add only the
        # interface gates joining those proofs or a hidden child endpoint.
        local_count = len(ports) + len(stream_ports)
        for wi, wait in enumerate(graph_ports):
            if wait.phase != "wait":
                continue
            for si, item in enumerate(graph_ports):
                if item.phase != "set" or not self._route(wait, item):
                    continue
                if wi < local_count and si < local_count and (wi < len(ports)) == (si < len(ports)):
                    continue
                invocation = self._invocation_guard(region, wait, item)
                if invocation is None:
                    continue
                if region is not self.root and all(
                    region in self.domains._clocks(port.notification) for port in (wait, item)
                ):
                    steps, guard, exact = self._gate_step(wait, item, region)
                else:
                    condition = self._order_condition(wait, item)
                    if condition is None:
                        continue
                    guard, exact = condition[:2]
                    steps = 0
                coordinates = dict(self._instance(wait).coordinates)
                binding = InvocationBinding(
                    region,
                    tuple(
                        (loop, coordinates[loop])
                        for loop in self._ancestors(region)[:-1]
                        if loop is not self.root
                    ),
                    item.bindings,
                    invocation,
                )
                graph_arcs.append(
                    GuardedTransfer(
                        wi,
                        si,
                        steps,
                        _guard(guard.comparisons + invocation.comparisons),
                        "actual region interface admission gate",
                        exact,
                        (binding,),
                    )
                )
        return ChannelProjection(
            region,
            tuple(graph_ports),
            self._unique_arcs(graph_arcs),
            tuple(
                dict.fromkeys(
                    window
                    for port in graph_ports
                    for window in self.domains.projections[port.notification]
                )
            ),
        )

    def _paths(self, graph):
        """One necessary-condition matrix, with exact direct witnesses."""
        ports, region = graph.ports, graph.operation
        paths = [[None] * len(ports) for _ in ports]
        direct, ordered_tests = {}, []
        for arc in graph.relations:
            a, b = ports[arc.source], ports[arc.destination]
            guard, projections = arc.guard, arc.projections
            if (
                a.phase == "set"
                and b.phase == "wait"
                and a.notification is b.notification
                and region in self.domains._clocks(a.notification)
            ):
                guard = self._window_guard(a, region)
                projections = self.domains.projections[a.notification]
            exported = self._export_guard(guard, region)
            witness = replace(
                arc,
                guard=exported,
                exact=arc.exact and exported == guard,
                projections=projections,
            )
            conditions = frozenset(exported.comparisons)
            old = paths[arc.source][arc.destination]
            paths[arc.source][arc.destination] = conditions if old is None else old & conditions
            direct.setdefault((arc.source, arc.destination), []).append(witness)
            ordered_tests.extend(exported.comparisons)
        return paths, direct, tuple(dict.fromkeys(ordered_tests))

    @staticmethod
    def _eliminate(paths, members):
        # Conditions compose by union along a path and intersection across
        # alternatives. This enumerates structural ports, never path histories.
        for k in members:
            for i in range(len(paths)):
                if paths[i][k] is None:
                    continue
                for j in range(len(paths)):
                    if paths[k][j] is None:
                        continue
                    conditions = paths[i][k] | paths[k][j]
                    old = paths[i][j]
                    paths[i][j] = conditions if old is None else old & conditions

    @staticmethod
    def _path_guard(conditions, ordered_tests):
        return _guard(tuple(test for test in ordered_tests if test in conditions))

    def _boundary(self, graph, paths, direct, ordered_tests, count):
        result = []
        for i in range(count):
            for j in range(count):
                conditions = paths[i][j]
                if conditions is None:
                    continue
                witnesses = direct.get((i, j), ())
                result.extend(witnesses)
                if any(frozenset(arc.guard.comparisons) == conditions for arc in witnesses):
                    continue
                guard = self._path_guard(conditions, ordered_tests)
                if not self._impossible(guard.comparisons):
                    result.append(
                        GuardedTransfer(
                            i,
                            j,
                            0,
                            guard,
                            "projected actual region blocking/publication interface",
                            False,
                            (graph,),
                        )
                    )
        return self._unique_arcs(
            arc for arc in result if not self._impossible(arc.guard.comparisons)
        )

    def _channels(self, graph, paths, direct, ordered_tests):
        """The sole exported publication/blocking summary for this region."""
        result = []
        for i, source in enumerate(graph.ports):
            if source.phase != "set":
                continue
            for j, destination in enumerate(graph.ports):
                conditions = paths[i][j]
                if destination.phase != "wait" or conditions is None:
                    continue
                guard = self._path_guard(conditions, ordered_tests)
                witness = next(
                    (
                        arc
                        for arc in direct.get((i, j), ())
                        if arc.exact and frozenset(arc.guard.comparisons) == conditions
                    ),
                    None,
                )
                result.append(
                    ChannelTransfer(
                        source,
                        destination,
                        guard,
                        witness is not None,
                        witness.projections if witness is not None else (graph,),
                    )
                )
        return tuple(result)

    def _stream_channels(self, region, ports, children):
        """A structural delay envelope for routes through a real child.

        Both local queue orders bound the child's entry step. Its actual
        source/destination entry stages account for different actor schedules.
        Ignoring lexical ties widens this relation; a successful rank for the
        lower delay therefore also orders every actual later use.
        """
        result = []
        for child in children:
            for channel in child.channels:
                a, _ = self._position(channel.source, region)
                b, _ = self._position(channel.destination, region)
                for wi, wait in enumerate(ports):
                    if wait.phase != "wait" or not self._route(wait, channel.source):
                        continue
                    for si, source in enumerate(ports):
                        if source.phase != "set" or not self._route(channel.destination, source):
                            continue
                        invocation = self._invocation_guard(region, wait, source)
                        if invocation is None:
                            continue
                        wo, so = (
                            self._stream_offset(wait, region),
                            self._stream_offset(source, region),
                        )
                        if wo is None or so is None:
                            self.fail(
                                region, "child route has no unit-rate containing-clock binding"
                            )
                        ws, _ = self._position(wait, region)
                        ss, _ = self._position(source, region)
                        guard = _guard(
                            self._window_guard(wait, region).comparisons
                            + self._window_guard(source, region).comparisons
                            + invocation.comparisons
                        )
                        result.append(
                            GuardedTransfer(
                                wi,
                                si,
                                wo + ws - so - ss + b - a,
                                guard,
                                "conservative actual child channel delay",
                                False,
                                channel.projections,
                            )
                        )
        return result

    def _controller_entries(self, region, children):
        """Closed cross-controller children need one actual call ordering.

        A child's progress certificate assumes its participating controllers
        can reach that invocation. Opposite entry orders can prevent that even
        when both child graphs progress in isolation. These are dispatch cuts,
        not child completion or memory-retirement claims.
        """
        collective = tuple(
            child
            for child in children
            if any(
                channel.source.actor is not channel.destination.actor for channel in child.channels
            )
        )
        if not collective:
            return ()
        actors = tuple(
            actor
            for actor in Actor
            if any(
                actor in {channel.source.actor, channel.destination.actor}
                for child in collective
                for channel in child.channels
            )
        )
        entries = []
        for actor in actors:
            calls = []
            for child in collective:
                scope = self.dispatch.scopes.get((child.operation, actor))
                if scope is None:
                    self.fail(
                        region,
                        "closed collective child lacks an actual participating-controller entry",
                    )
                calls.append((child.operation, scope.parent_stage, scope.parent_lexical))
            entries.append(ControllerEntries(actor, tuple(sorted(calls, key=lambda call: call[2]))))
        reference = tuple((operation, stage) for operation, stage, _ in entries[0].calls)
        if any(
            tuple((operation, stage) for operation, stage, _ in entry.calls) != reference
            for entry in entries[1:]
        ):
            self.fail(
                region,
                "closed child controller dispatch order has no common invocation ordering certificate",
                tuple(
                    f"{entry.actor.value}: "
                    + ", ".join(
                        f"{self.control.loops[operation].ordinal.text()} stage={stage} lexical={lexical}"
                        for operation, stage, lexical in entry.calls
                    )
                    for entry in entries
                ),
            )
        return tuple(entries)

    def _local_arcs(self, region, ports, own, children, *, stream):
        index = {_key(port): i for i, port in enumerate(ports)}
        arcs = []
        for notification in own:
            sources = [
                i
                for i, port in enumerate(ports)
                if port.notification is notification and port.phase == "set"
            ]
            consumers = [
                i
                for i, port in enumerate(ports)
                if port.notification is notification and port.phase == "wait"
            ]
            for source in sources:
                for consumer in consumers:
                    invocation = self._invocation_guard(region, ports[source], ports[consumer])
                    if invocation is None:
                        continue
                    guard = _guard(
                        self._ambient(ports[source])
                        + self._ambient(ports[consumer])
                        + invocation.comparisons
                    )
                    if not self._impossible(guard.comparisons):
                        arcs.append(
                            GuardedTransfer(
                                source,
                                consumer,
                                0,
                                guard,
                                "selected effect before its matching actual Wait",
                            )
                        )
        for child in children:
            for arc in child.relations:
                a, b = child.ports[arc.source], child.ports[arc.destination]
                if stream:
                    if a.notification not in own or b.notification not in own:
                        continue
                    a, b = self._stream_port(a, region), self._stream_port(b, region)
                    if a.bindings != b.bindings:
                        self.fail(region, "child transfer needs a shared actual ancestor binding")
                    bindings = dict(a.bindings)
                    arc = replace(
                        arc,
                        guard=_guard(
                            tuple(
                                Comparison(
                                    substitute(test.lhs, bindings),
                                    test.predicate,
                                    substitute(test.rhs, bindings),
                                )
                                for test in arc.guard.comparisons
                            )
                        ),
                    )
                if _key(a) in index and _key(b) in index:
                    arcs.append(replace(arc, source=index[_key(a)], destination=index[_key(b)]))
        for wi, wait in enumerate(ports):
            if wait.phase != "wait":
                continue
            for si, source in enumerate(ports):
                if source.phase != "set" or not self._route(wait, source):
                    continue
                # A child invocation cannot consume a parent-generation step.
                # The original ports remain exported for that parent's proof.
                invocation = self._invocation_guard(region, wait, source)
                if invocation is None:
                    continue
                if stream:
                    shift, guard, exact = self._gate_step(wait, source, region)
                else:
                    condition = self._order_condition(wait, source)
                    if condition is None:
                        continue
                    guard, exact = condition[:2]
                    shift = 0
                guard = _guard(guard.comparisons + invocation.comparisons)
                if not self._impossible(guard.comparisons):
                    arcs.append(
                        GuardedTransfer(
                            wi,
                            si,
                            shift,
                            guard,
                            "actual controller admission gate"
                            if self.domains._gate(wait.notification)
                            else f"actual {wait.engine.value} queue admission gate",
                            exact,
                        )
                    )
        if stream:
            arcs.extend(self._stream_channels(region, ports, children))
        return self._unique_arcs(arcs)

    @staticmethod
    def _unique_arcs(arcs):
        # Equal actual relations are one structural edge, not distinct paths.
        unique = {}
        for arc in arcs:
            unique.setdefault((arc.source, arc.destination, arc.steps, arc.guard), arc)
        return tuple(unique.values())

    def compose(self, region):
        children = tuple(self.compose(child) for child in self.children[region])
        controller_entries = self._controller_entries(region, children)
        own = tuple(n for n in self.events if self.owners[n] is region)
        child_ports = tuple(port for child in children for port in child.ports)
        ports, seen = [], set()
        for port in self._ports(region) + child_ports:
            if _key(port) not in seen:
                ports.append(port)
                seen.add(_key(port))
        streams = tuple(
            n for n in own if region is not self.root and region in self.domains._clocks(n)
        )
        stream_ports = tuple(
            self._stream_port(port, region) for port in ports if port.notification in streams
        )
        ports = tuple(port for port in ports if port.notification not in streams)
        if region is not self.root and any(
            not self.control.invariant_over(dict(self._instance(port).coordinates)[region], region)
            for port in ports
        ):
            self.fail(
                region,
                "boundary port retains a varying current-loop coordinate without an owned stream interface",
            )
        stream_proof = None
        if streams:
            stream_arcs = self._local_arcs(region, stream_ports, streams, children, stream=True)
            stride, rank = self._rank(region, stream_ports, stream_arcs)
            stream_proof = StreamProgress(
                region,
                stream_ports,
                stream_arcs,
                stride,
                rank,
            )
            self.components.append((region, stream_ports, stream_arcs, rank, stride))
        own_finite = tuple(n for n in own if n not in streams)
        direct_arcs = self._local_arcs(region, ports, own_finite, (), stream=False)
        graph = self._graph(region, ports, stream_proof, direct_arcs, children)
        paths, direct, ordered_tests = self._paths(graph)
        self._eliminate(paths, range(len(ports), len(graph.ports)))
        arcs = self._boundary(graph, paths, direct, ordered_tests, len(ports))
        excluded = self._finite_proof(region, ports, arcs)
        self._eliminate(paths, range(len(ports)))
        proof = RegionProgress(
            region,
            ports,
            arcs,
            self._channels(graph, paths, direct, ordered_tests),
            children,
            stream_proof,
            excluded,
            controller_entries,
        )
        self.components.append((region, ports, arcs, (), 1))
        return proof

    def run(self):
        root = self.compose(self.root)
        points, relations = [], []
        for component, (_, ports, arcs, _, _) in enumerate(self.components):
            base = len(points)
            points.extend(
                ProgressPoint(
                    port.bound.operation,
                    port.actor,
                    port.phase,
                    port.notification,
                    component,
                    "actual-region",
                )
                for port in ports
            )
            relations.extend(
                replace(arc, source=arc.source + base, destination=arc.destination + base)
                for arc in arcs
            )
        self.domains.work.progress_points = len(points)
        self.domains.work.progress_arcs = len(relations)
        return StructuredProgress(
            tuple(points),
            tuple(relations),
            tuple(operation for operation, _, _, _, _ in self.components),
            tuple(rank for _, _, _, rank, _ in self.components),
            tuple(stride for _, _, _, _, stride in self.components),
            root,
        )


def prove_progress(domains):
    """Prove owned region progress or diagnose the unproved interface relation."""
    return _Composer(domains).run()

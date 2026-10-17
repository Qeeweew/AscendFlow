"""A sufficient, compositional order rule for generated notifications.

Each loop uses its actual physical step (logical ordinal plus stage) as the
major order.  Matching publications may advance that step or stay in it;
they may not move backwards.  A finite DAG orders the zero-step queue gates.
Ordinary local waits gate their destination queue, whereas Scalar and cross
waits gate their actor's controller.  Controller issue order alone is never
an order between native completions.

A child exports actual open endpoints and fixed queue cuts.  Its interface
contains only possible blocking transfers between those boundaries, with no
hidden graph or path history.  The abstraction is deliberately sufficient:
an unrankable union of conditional transfers is diagnosed, not solved using
another whole-program progress search.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from xdsl.ir import Operation

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from ..scalar import expression, literal, substitute
from ..target.ascend910b import Actor, Engine
from .event_domains import EventCase, EventDomains
from .structured_protocols import BoundPlacement, LogicalNotification, NotificationKind
from .structured_schedule import Comparison


@dataclass(frozen=True, eq=False, slots=True)
class QueuePoint:
    """An actual endpoint, or one queue boundary of an invocation.

    An engine of None denotes a controller gate.  Cuts have no notification
    identity and provide no completion, consumption or storage-release fact.
    """

    operation: Operation
    actor: Actor
    engine: Engine | None
    phase: str
    notification: LogicalNotification | None = None
    bound: BoundPlacement | None = None
    case: EventCase | None = None


@dataclass(frozen=True, slots=True)
class QueueRegion:
    operation: Operation
    points: tuple[QueuePoint, ...]
    rank: tuple[int, ...]
    # Only these points and their transfers are imported by the parent.
    ports: tuple[QueuePoint, ...]
    cuts: tuple[QueuePoint, ...]
    transfers: tuple[tuple[QueuePoint, QueuePoint], ...]
    children: tuple[QueueRegion, ...]
    positive_relations: int


@dataclass(frozen=True, slots=True)
class QueueOrderWork:
    regions: int
    points: int
    zero_relations: int
    positive_relations: int
    publication_checks: int
    boundary_ranges: int
    summary_visits: int
    boundary_points: int
    boundary_transfers: int


@dataclass(frozen=True, slots=True)
class QueueOrder:
    root: QueueRegion
    regions: tuple[QueueRegion, ...]
    work: QueueOrderWork


@dataclass(frozen=True, slots=True)
class _Position:
    stage: int
    lexical: int
    child: Operation | None


class _OrderConstruction:
    def __init__(self, domains: EventDomains):
        self.domains = domains
        self.control, self.dispatch = domains.control, domains.dispatch
        self.root = domains.memory.module.body.block.first_op
        self.parents: dict[Operation, Operation | None] = {self.root: None}
        self.children: dict[Operation, list[Operation]] = {self.root: []}
        self.chains = {self.root: ()}
        for loop in self.control.loops:
            chain = tuple(
                frame.loop.operation
                for frame in self.control.frames[loop].ancestors()
                if frame.loop is not None
            )
            parent = chain[-2] if len(chain) > 1 else self.root
            self.chains[loop] = chain
            self.parents[loop] = parent
            self.children.setdefault(parent, []).append(loop)
            self.children.setdefault(loop, [])
        self.owners = {}
        self.endpoints = []
        for notification in domains.notifications:
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                continue
            self.owners[notification] = self._owner(notification)
            for case in domains.cases[notification]:
                if not domains.active[notification, case]:
                    continue
                for phase, bounds in (
                    ("set", notification.sources),
                    ("wait", notification.consumers),
                ):
                    for bound in bounds:
                        point = QueuePoint(
                            bound.operation,
                            bound.actor,
                            notification.source_engine
                            if phase == "set"
                            else notification.destination_engine,
                            phase,
                            notification,
                            bound,
                            case,
                        )
                        self.endpoints.append(point)
        self._contexts = {}
        self.boundary_times = {}
        self.boundary_groups = {}
        self.regions = []
        self.work = dict.fromkeys(QueueOrderWork.__dataclass_fields__, 0)

    def _fail(self, region, message, points=()):
        raise CompileError(
            message,
            kind=ErrorKind.UNSUPPORTED,
            phase="notification-generation-order",
            location=str(region.location),
            witness=tuple(self._label(point) for point in points),
        )

    @staticmethod
    def _label(point):
        route = "controller" if point.engine is None else point.engine.value
        channel = "queue boundary" if point.notification is None else point.notification.name
        return f"{channel} {point.actor.value}/{route} {point.phase} at {point.operation.location}"

    def _possible(self, tests):
        opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
        return not any(
            self.domains.arithmetic.prove(
                Comparison(test.lhs, opposite[test.predicate], test.rhs),
                tests[:index] + tests[index + 1 :],
            )
            for index, test in enumerate(tests)
        )

    def _inside(self, operation, region):
        return region is self.root or region in tuple(
            frame.loop.operation
            for frame in self.control.frames[operation].ancestors()
            if frame.loop is not None
        )

    def _ancestor(self, ancestor, region):
        return ancestor is self.root or ancestor in self.chains[region]

    def _owner(self, notification):
        bounds = notification.sources + notification.consumers
        paths = tuple(
            tuple(
                frame.loop.operation
                for frame in self.control.frames[bound.operation].ancestors()
                if frame.loop is not None
            )
            for bound in bounds
        )
        owner = self.root
        for level in zip(*paths):
            if any(loop is not level[0] for loop in level[1:]):
                break
            owner = level[0]
        while owner is not self.root:
            coordinates = dict(bounds[0].coordinates)
            if all(
                self.domains.arithmetic.prove(
                    Comparison(coordinates[loop], "==", dict(bound.coordinates)[loop]),
                    self.domains.tests[notification],
                )
                for loop in self.chains[owner][:-1]
                for bound in bounds[1:]
            ):
                break
            owner = self.parents[owner]
        return owner

    def _position(self, point, region):
        scope = (
            self.dispatch.scopes[point.operation, point.actor]
            if point.bound is None
            else self.dispatch.tasks[point.operation, point.actor].scope
        )
        task = None if point.bound is None else self.dispatch.tasks[point.operation, point.actor]
        if scope.operation is region:
            if task is None:
                return _Position(0, 0, None)
            return _Position(task.stage, task.lexical, None)
        while scope.parent is not None and scope.parent.operation is not region:
            scope = scope.parent
        if scope.parent is None:
            self._fail(region, "generation boundary is outside its actual dispatch scope", (point,))
        return _Position(scope.parent_stage, scope.parent_lexical, scope.operation)

    def _context(self, point, region):
        """Rebase strict ancestor translations to this actual invocation.

        For example the last reader in parent g-1 and the first writer in g
        are different channel coordinates, but each child summary describes
        one actual child invocation.  Their original maps remain on the ports
        and are used again by the parent; no naked ordinals are identified.
        """
        key = point, region
        if key in self._contexts:
            return self._contexts[key]
        if point.bound is None:
            tests = ()
            if point.operation in self.control.loops:
                tests = (Comparison(self.control.loops[point.operation].count, ">", literal(0)),)
            coordinate = literal(0) if region is self.root else self.control.loops[region].ordinal
            result = coordinate, tests
        else:
            bound, case = point.bound, point.case
            assert case is not None
            instance = self.domains._instance(bound, {}, case=case)
            coordinates = dict(instance.coordinates)
            translations, restrictions = {}, []
            for ancestor in self.chains[region][:-1]:
                ordinal = self.control.loops[ancestor].ordinal
                actual = substitute(coordinates[ancestor], translations)
                difference = self.domains.arithmetic.normalize(expression("sub", actual, ordinal))
                if difference.constant is not None:
                    translations[ordinal] = expression("sub", ordinal, difference)
                elif self.control.invariant_over(actual, ancestor):
                    restrictions.append(Comparison(ordinal, "==", actual))
                else:
                    self._fail(
                        region,
                        "ancestor notification coordinate is neither a unit translation nor an invocation selection",
                        (point,),
                    )
            tests = tuple(
                Comparison(
                    substitute(test.lhs, translations),
                    test.predicate,
                    substitute(test.rhs, translations),
                )
                for test in instance.guard.comparisons
            ) + tuple(restrictions)
            coordinate = (
                literal(0) if region is self.root else substitute(coordinates[region], translations)
            )
            result = coordinate, tuple(dict.fromkeys(tests))
        self._contexts[key] = result
        return result

    def _time(self, point, region):
        coordinate, tests = self._context(point, region)
        position = self._position(point, region)
        return expression("add", coordinate, literal(position.stage)), tests

    def _boundary_time(self, point, region):
        key = point, region
        if key not in self.boundary_times:
            ranges = tuple(
                self.domains.arithmetic.bounds(*self._time(member, region))
                for member in self.boundary_groups[region][point]
            )
            lower, upper = zip(*ranges, strict=True)
            self.boundary_times[key] = (
                None if None in lower else min(lower),
                None if None in upper else max(upper),
            )
            self.work["boundary_ranges"] += 1
        return self.boundary_times[key]

    def _coalesce(self, points, region):
        """One may-block point per actual position, not per memory fragment.

        Publications still use every original endpoint and its coordinates.
        Coalescing is a conservative union of possible blockers; finite time
        ranges take the hull of all member domains. It proves no consumption.
        Owners and case substitutions remain distinct at child boundaries.
        """
        unique, groups, keys, index = [], {}, {}, {}
        for point in points:
            key = (
                point
                if point.bound is None
                else (
                    point.operation,
                    point.actor,
                    point.engine,
                    point.phase,
                    self.owners[point.notification],
                    point.bound.coordinates,
                    point.case.condition,
                    point.case.substitutions,
                )
            )
            if key not in keys:
                keys[key] = len(unique)
                unique.append(point)
                groups[point] = []
            slot = keys[key]
            index[point] = slot
            groups[unique[slot]].append(point)
        self.boundary_groups[region] = groups
        return unique, index

    @staticmethod
    def _controller(point):
        return (
            point.engine is None
            if point.bound is None
            else (
                point.notification.mechanism is NotificationKind.CROSS_EVENT
                or point.notification.destination_engine is Engine.S
            )
        )

    def _route(self, wait, source):
        return wait.actor is source.actor and (
            self._controller(wait) or wait.engine is source.engine
        )

    def _publication(self, region, source, wait):
        self.work["publication_checks"] += 1
        a, at = self._time(source, region)
        b, bt = self._time(wait, region)
        delta = self.domains.arithmetic.normalize(expression("sub", b, a))
        if delta.constant is not None and delta.constant >= 0:
            return int(delta.constant > 0)
        tests = at + bt
        if not self._possible(tests):
            return None
        if not self.domains.arithmetic.prove(Comparison(delta, ">=", literal(0)), tests):
            self._fail(
                region,
                "notification publication moves backwards in the actual physical-step order; "
                "this generation rule does not support that schedule",
                (source, wait),
            )
        return 1 if self.domains.arithmetic.prove(Comparison(delta, ">", literal(0)), tests) else 0

    def _gate(self, region, wait, source):
        wp, sp = self._position(wait, region), self._position(source, region)
        # The child's exported transfer is authoritative for one invocation.
        # Reentry into that same call belongs to the next parent step.
        same_child = wp.child is not None and wp.child is sp.child
        if same_child:
            return 1 if region is not self.root else None
        lexical = sp.lexical >= wp.lexical
        if region is not self.root and wait.bound is not None and source.bound is not None:
            a, _ = self._time(wait, region)
            b, _ = self._time(source, region)
            if self.control.invariant_over(a, region) or self.control.invariant_over(b, region):
                # FIRST/LAST ports can be single occurrences. In particular
                # a singleton child has no invented return to an earlier
                # lexical rank. Only this finite boundary relation needs an
                # integer premise; two repeated channels use the rule below.
                (al, ah), (bl, bh) = (
                    self._boundary_time(wait, region),
                    self._boundary_time(source, region),
                )
                delta = self.domains.arithmetic.normalize(expression("sub", b, a))
                if (
                    delta.constant is not None
                    and (delta.constant < 0 if lexical else delta.constant <= 0)
                ) or (bh is not None and al is not None and (bh < al if lexical else bh <= al)):
                    return None
                if (delta.constant is not None and delta.constant > 0) or (
                    bl is not None and ah is not None and bl > ah
                ):
                    return 1
                # Unresolved intervals keep the structural possible transfer.
                # Endpoint ranges are established once, independently of the
                # other channel's canonical generation and guard history.
        # Independent channels do not share a canonical generation variable.
        # Queue issue order is structural: within one actual physical step it
        # follows lexical rank, and a return to an earlier rank advances the
        # step. Keep every possible activation, including selected FIRST/LAST;
        # no pairwise guard solver or identification of naked ordinals is
        # needed. Exact activation remains essential for pairing and reuse.
        if lexical:
            return 0
        return 1 if region is not self.root else None

    def _rank(self, region, points, edges):
        outgoing = [[] for _ in points]
        degree, rank = [0] * len(points), [0] * len(points)
        for source, destination, step in edges:
            if step == 0:
                outgoing[source].append(destination)
                degree[destination] += 1
        pending = deque(index for index, value in enumerate(degree) if value == 0)
        visited = 0
        while pending:
            source = pending.popleft()
            visited += 1
            for destination in outgoing[source]:
                rank[destination] = max(rank[destination], rank[source] + 1)
                degree[destination] -= 1
                if degree[destination] == 0:
                    pending.append(destination)
        if visited != len(points):
            self._fail(
                region,
                "generated queue/controller transfers contain a zero-step blocking cycle; "
                "no common phase order was established",
                tuple(point for point, value in zip(points, degree, strict=True) if value),
            )
        return tuple(rank)

    def compose(self, region):
        children = tuple(self.compose(child) for child in self.children[region])
        own = tuple(n for n, owner in self.owners.items() if owner is region)
        points = [
            point
            for point in self.endpoints
            if self._ancestor(self.owners[point.notification], region)
            and self._inside(point.operation, region)
        ]
        for child in children:
            points.extend(point for point in child.ports + child.cuts if point not in points)
        original = tuple(points)
        points, index = self._coalesce(points, region)
        edges = set()

        def edge(source, destination, step):
            if step is not None:
                edges.add((index[source], index[destination], step))

        for notification in own:
            publications = tuple(
                p for p in original if p.notification is notification and p.phase == "set"
            )
            waits = tuple(
                p for p in original if p.notification is notification and p.phase == "wait"
            )
            for source in publications:
                for wait in waits:
                    if source.case is wait.case:
                        edge(source, wait, self._publication(region, source, wait))
        for child in children:
            for source, destination in child.transfers:
                edge(source, destination, 0)
        for wait in tuple(points):
            if wait.phase not in {"wait", "output"}:
                continue
            for source in tuple(points):
                poll(
                    "notification generation queue order",
                    regions=len(self.regions),
                    points=len(points),
                )
                if source.phase in {"set", "input"} and self._route(wait, source):
                    edge(wait, source, self._gate(region, wait, source))

        # New cuts summarize only closed publications/gates and child cuts.
        # Open endpoints keep their actual identity and ancestor coordinates.
        closed = tuple(p for p in points if p.bound is None or p.notification in own)
        routes = tuple(
            dict.fromkeys(
                (
                    p.actor,
                    p.engine
                    if p.phase in {"set", "input"}
                    else None
                    if self._controller(p)
                    else p.engine,
                    "input" if p.phase in {"set", "input"} else "output",
                )
                for p in closed
            )
        )
        cuts = tuple(QueuePoint(region, actor, engine, phase) for actor, engine, phase in routes)
        for cut in cuts:
            index[cut] = len(points)
            points.append(cut)
            for point in closed:
                source = point.phase in {"set", "input"}
                engine = (
                    point.engine if source else None if self._controller(point) else point.engine
                )
                if (
                    cut.actor is point.actor
                    and cut.engine is engine
                    and (cut.phase == "input") == source
                ):
                    edge(cut if source else point, point if source else cut, 0)
        rank = self._rank(region, points, edges)
        ports = tuple(
            p for p in points if p.bound is not None and self.owners[p.notification] is not region
        )
        retained = ports + cuts
        adjacency = [[] for _ in points]
        for a, b, _ in edges:
            adjacency[a].append(b)
        transfers = []
        # One finite reachability pass per *boundary*, then discard this graph.
        # Positive child steps become an opaque whole-invocation possibility;
        # no native completion or event-consumption fact is exported.
        for source in retained:
            reached, pending = set(), list(adjacency[index[source]])
            while pending:
                poll(
                    "notification generation boundary summary",
                    points=len(points),
                    boundary=len(retained),
                )
                target = pending.pop()
                if target in reached:
                    continue
                reached.add(target)
                self.work["summary_visits"] += 1
                pending.extend(adjacency[target])
            transfers.extend(
                (source, target)
                for target in retained
                if target is not source and index[target] in reached
            )
        positive = sum(step > 0 for _, _, step in edges)
        result = QueueRegion(
            region, tuple(points), rank, ports, cuts, tuple(transfers), children, positive
        )
        self.regions.append(result)
        self.work["regions"] += 1
        self.work["points"] += len(points)
        self.work["zero_relations"] += len(edges) - positive
        self.work["positive_relations"] += positive
        self.work["boundary_points"] += len(retained)
        self.work["boundary_transfers"] += len(transfers)
        return result

    def run(self):
        root = self.compose(self.root)
        return QueueOrder(root, tuple(self.regions), QueueOrderWork(**self.work))


def establish_queue_order(domains: EventDomains) -> QueueOrder:
    """Construct the physical-step/queue order before allocating resources."""
    return _OrderConstruction(domains).run()

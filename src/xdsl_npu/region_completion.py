"""Native relations for the admitted serial region/call control domain.

This adapter uses the explicit serial Sequence contract. It is not valid for
overlapping CV tasks. Conditional path composition and endpoint identity are
shared with leaf/invocation checks in completion.py.
"""

from collections import defaultdict

from .completion import (
    Point,
    Position,
    Requirement,
    Site,
    _first,
    covers,
    uncovered,
)
from .completion_flow import ALL
from .model import require
from .native_order import prefix_inputs
from .region_summary import Access
from .symbolic_completion import Family, SymbolicCompletion, native_family


def _data(regions, name):
    return next(iter(regions[name].actors.values()))


def _task(regions, point):
    return next(t for t in _data(regions, point.region).program.tasks if t.uid == point.task)


def _expression(regions, point):
    count = regions[point.region].constant_count
    p = point.position
    if count is not None:
        return 0, p.evaluate(count)
    return (1, -1 - p.offset) if p.end else (0, p.offset)


def _nonnegative(a, b, domain):
    return a * _first(domain) + b >= 0 and (domain.exact is not None or a >= 0)


def _valid_point(regions, point, domain):
    task = _task(regions, point)
    a, b = _expression(regions, point)
    count = regions[point.region].constant_count
    ca, cb = (1, 0) if count is None else (0, count)
    return (
        _nonnegative(a, b, domain)
        and _nonnegative(ca - a, cb - b - 1, domain)
        and (
            not task.last
            or a == ca
            and b == cb - 1
            or domain.exact is not None
            and a * domain.exact + b == ca * domain.exact + cb - 1
        )
    )


def _point(regions, endpoint, domain, *, source):
    access = endpoint.access
    p = access.position if isinstance(access, Access) else access.last if source else access.first
    count = endpoint.constant_count
    task = next(t for t in _data(regions, endpoint.region).program.tasks if t.uid == access.task[1])
    if count is not None:
        p = Position(False, p.evaluate(count))
    elif domain.exact is not None and (source or task.last):
        p = Position(True, domain.exact - 1 - p.evaluate(domain.exact))
    return Point(endpoint.region, access.task[1], p)


def position(regions, region, point):
    count = regions[region].constant_count
    if count is None:
        return point
    index = point.evaluate(count)
    return Position(False, index) if index <= count // 2 else Position(True, count - 1 - index)


def mark(regions, point, milestone, call=0):
    actor = next(iter(regions[point.region].actors))
    return Site(
        (actor, point.task),
        position(regions, point.region, point.position),
        milestone,
        region=point.region,
        call=call,
    )


class RegionCompletion:
    """Sparse symbolic native families composed at real scope boundaries.

    There is no head/tail instruction expansion. A physical prefix may carry
    a fact across later generations using its transparent native carry cycle;
    individual task completion and notification effects never get that rule.
    Boundary state is still issue/queue/retire/implicit, not child-done.
    """

    def __init__(self, regions, analyses, requirements, *, calls=(0, 1), bridges=()):
        self.regions, self.analyses, self.calls = regions, analyses, calls
        self.order = {name: i for i, name in enumerate(regions)}
        self.requirements = tuple(requirements)
        self.flow = SymbolicCompletion()
        self.flow.node(Family(0, (0, 0)))  # Open input, with no originating fact.
        self.queries = 0
        self.sources = {
            s: i for i, s in enumerate(dict.fromkeys(r.source for r in self.requirements))
        }
        self.bindings = defaultdict(list)
        self.scopes = {}
        state = {}
        self.templates = {name: self._template(name) for name in regions}
        for call in calls:
            for name in regions:
                self._compose(call, name, state)
        marks = {s for r in self.requirements for s in (r.source, r.destination)}
        for r in self.requirements:
            marks.add(mark(regions, r.source.location, "prefix", r.source.call))
        for bridge in bridges:
            for call in calls:
                target = call + int(bridge.carried)
                if target in calls:
                    marks.add(mark(regions, bridge.source, "prefix", call))
                    marks.add(mark(regions, bridge.destination, "begin", target))
        for site in marks:
            graph, commands, mapping = self.scopes[site.call, site.region]
            command = commands[site.task, site.channel, site.action]
            point = _expression(regions, site.location)
            self.bindings[site].extend(self._positions(graph, mapping, command, site.point, point))
        for site, bit in self.sources.items():
            for node, point in self.bindings[site]:
                self.flow.seed(node, point, ALL, 1 << bit)
        self.flow.propagate()
        self.native_coverage = {r: self._coverage(r) for r in self.requirements}
        for bridge in bridges:
            self.add_bridge(bridge)

    def _expr(self, name, a, b):
        count = self.regions[name].constant_count
        return (a, b) if count is None else (0, a * count + b)

    def _template(self, name):
        _, channels, streams, graph, _ = self.analyses[name]
        require(
            len(streams) == 1 and all(c.kind == "local" for c in channels.values()),
            "serial completion transfers require one local actor and local event contracts",
        )
        actor, commands = next(iter(streams.items()))
        families = tuple(
            native_family(meta, graph.period, self._expr(name, 1, 0)) for meta in graph.nodes
        )
        edges = tuple(
            (a, b, d * graph.period + graph.nodes[b].residue - graph.nodes[a].residue)
            for a, b, d in graph.edges
        )
        return actor, commands, graph, families, edges

    def _positions(self, graph, mapping, command, kind, point):
        if kind == "end" and command.kind in {"wait", "barrier"}:
            kind = "begin"
        for residue in range(graph.period):
            node = graph.ids.get((command.index, residue, kind))
            if (
                node is not None
                and self.flow.families[mapping[node]].residue_guard(point) is not None
            ):
                yield mapping[node], point

    def _compose(self, call, name, state):
        actor, commands, graph, families, edges = self.templates[name]
        mapping = tuple(self.flow.node(family) for family in families)
        by_identity = {
            ((actor, c.task.uid), c.channel, "" if c.kind == "task" else c.kind): c
            for c in commands
        }
        self.scopes[call, name] = graph, by_identity, mapping
        for a, b, shift in edges:
            self.flow.edge(mapping[a], mapping[b], shift)
        maximum = max(c.task.stage for c in commands)
        pipes = {c.task.pipe for c in commands}
        boundaries = [("issue", commands[0], commands[-1])]
        for pipe in sorted(pipes):
            stream = [c for c in commands if c.task.pipe == pipe]
            boundaries += [
                (pipe, kind, stream[0], stream[-1]) for kind in ("queue", "retire", "implicit")
            ]
        for *parts, first, last in boundaries:
            key, kind = (actor, *parts), parts[-1]
            before = state.get(key, 0)
            after = self.flow.node(Family(0, (0, 0)))
            state[key] = after
            # Every prefix preserves incoming facts, including a zero-trip
            # child. This edge does not complete any child native work.
            self.flow.edge(before, after)
            for node, point in self._positions(graph, mapping, first, kind, (0, -first.task.stage)):
                self.flow.edge(before, node, source_point=(0, 0), target_point=point)
            exit_point = self._expr(name, 1, maximum - 1 - last.task.stage)
            for node, point in self._positions(graph, mapping, last, kind, exit_point):
                self.flow.edge(node, after, source_point=point, target_point=(0, 0))
            for residue in range(graph.period):
                points = {
                    p: mapping[node]
                    for p in ("begin", "end", "prefix")
                    if (node := graph.ids.get((last.index, residue, p))) is not None
                }
                if "begin" not in points:
                    continue
                points.setdefault("end", points["begin"])
                if (node := prefix_inputs(last, points).get(kind)) is not None:
                    self.flow.edge(node, after, source_point=exit_point, target_point=(0, 0))

    def add_bridge(self, bridge):
        for call in self.calls:
            target_call = call + int(bridge.carried)
            if target_call not in self.calls:
                continue
            source = mark(self.regions, bridge.source, "prefix", call)
            target = mark(self.regions, bridge.destination, "begin", target_call)
            require(
                source in self.bindings and target in self.bindings,
                "protocol endpoint missing from symbolic native transfers",
            )
            for left, a in self.bindings[source]:
                for right, b in self.bindings[target]:
                    for domain in bridge.domains:
                        self.flow.edge(left, right, source_point=a, target_point=b, guard=domain)

    def _coverage(self, requirement):
        bit = 1 << self.sources[requirement.source]
        return tuple(
            domain
            for node, point in self.bindings[requirement.destination]
            for domain in self.flow.coverage(node, point, bit, requirement.domain)
        )

    def coverage(self, requirement):
        self.queries += 1
        self.flow.propagate()
        return self._coverage(requirement)

    def covered(self, requirement):
        return covers(requirement.domain, self.coverage(requirement))

    def missing(self, requirement):
        return uncovered(requirement.domain, self.coverage(requirement))

    def inherited(self, requirement):
        return covers(requirement.domain, self.native_coverage[requirement])


def memory_requirement(sequence, source, destination, domain, obligation):
    access = obligation.source.access
    point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
    return Requirement(
        mark(sequence.regions, source, point),
        mark(sequence.regions, destination, "begin"),
        domain,
        obligation.hazard,
    )

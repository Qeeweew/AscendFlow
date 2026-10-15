"""Native relations for the admitted serial region/call control domain.

This adapter uses the explicit serial Sequence contract. It is not valid for
overlapping CV tasks. Conditional path composition and endpoint identity are
shared with leaf/invocation checks in completion.py.
"""

from collections import defaultdict

from .completion import (
    CountDomain,
    Point,
    Position,
    Requirement,
    Site,
    _first,
    covers,
    intersection,
    uncovered,
)
from .completion_flow import ALL, CompletionFlow
from .model import require
from .region_summary import Access


def _data(sequence, name):
    return next(iter(sequence.regions[name].actors.values()))


def _task(sequence, point):
    return next(t for t in _data(sequence, point.region).program.tasks if t.uid == point.task)


def _expression(sequence, point):
    count = sequence.regions[point.region].constant_count
    p = point.position
    if count is not None:
        return 0, p.evaluate(count)
    return (1, -1 - p.offset) if p.end else (0, p.offset)


def _nonnegative(a, b, domain):
    return a * _first(domain) + b >= 0 and (domain.exact is not None or a >= 0)


def _valid_point(sequence, point, domain):
    task = _task(sequence, point)
    a, b = _expression(sequence, point)
    count = sequence.regions[point.region].constant_count
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


def _point(sequence, endpoint, domain, *, source):
    access = endpoint.access
    p = access.position if isinstance(access, Access) else access.last if source else access.first
    count = endpoint.constant_count
    task = next(
        t for t in _data(sequence, endpoint.region).program.tasks if t.uid == access.task[1]
    )
    if count is not None:
        p = Position(False, p.evaluate(count))
    elif domain.exact is not None and (source or task.last):
        p = Position(True, domain.exact - 1 - p.evaluate(domain.exact))
    return Point(endpoint.region, access.task[1], p)


def position(sequence, region, point):
    count = sequence.regions[region].constant_count
    if count is None:
        return point
    index = point.evaluate(count)
    return Position(False, index) if index <= count // 2 else Position(True, count - 1 - index)


def mark(sequence, point, milestone, call=0):
    actor = next(iter(sequence.regions[point.region].actors))
    return Site(
        (actor, point.task),
        position(sequence, point.region, point.position),
        milestone,
        region=point.region,
        call=call,
    )


def _ge(a, b):
    """Predicate a*N+b >= 0 for a in {-1,0,1}."""
    if not a:
        return ALL if b >= 0 else None
    if a == 1:
        return CountDomain(1, 0, minimum=max(0, -b))
    require(a == -1, "unsupported completion coordinate slope")
    return CountDomain(1, 0, minimum=0, maximum=b) if b >= 0 else None


def _equal(left, right):
    a, b = left[0] - right[0], left[1] - right[1]
    if not a:
        return ALL if not b else None
    return CountDomain(1, 0, exact=-b // a) if -b % a == 0 and -b // a >= 0 else None


def _meet(left, right):
    return None if left is None or right is None else intersection(left, right)


class RegionCompletion:
    """Batch native transfers for the admitted serial region/call structure.

    A loop's finite head and tail retain all demanded boundary milestones.
    Between them, queue, native-retirement, implicit-completion and issue
    prefixes carry monotonically: each native transfer contains its incoming
    prefix. This is a structural induction, not sampling N or assuming the
    omitted native work completed. Event effects NEVER get such a carry edge.
    Interior paths absent from this summary cannot establish coverage.
    """

    def __init__(self, sequence, analyses, requirements, *, calls=(0, 1), bridges=()):
        self.sequence, self.analyses, self.calls = sequence, analyses, calls
        self.order = {name: i for i, name in enumerate(sequence.regions)}
        self.requirements = tuple(requirements)
        self.flow = CompletionFlow()
        self.flow.node()  # Open entry prefixes; no completion facts originate here.
        self.queries = 0
        self.sources = {
            site: i for i, site in enumerate(dict.fromkeys(r.source for r in self.requirements))
        }
        self.bindings = defaultdict(list)
        marks = {s for r in self.requirements for s in (r.source, r.destination)}
        for r in self.requirements:
            marks.add(mark(sequence, r.source.location, "prefix", r.source.call))
        for bridge in bridges:
            for call in calls:
                target = call + int(bridge.carried)
                if target in calls:
                    marks.add(mark(sequence, bridge.source, "prefix", call))
                    marks.add(mark(sequence, bridge.destination, "begin", target))
        self.marks = defaultdict(list)
        for m in sorted(marks, key=repr):
            self.marks[m.call, m.region, m.task, m.channel, m.action].append(m)
        state = {}
        for call in calls:
            for region in sequence.regions:
                self._region(call, region, state)
        for site, bit in self.sources.items():
            for node, guard in self.bindings[site]:
                self.flow.add_fact(node, guard, 1 << bit)
        self.flow.propagate()
        self.native_coverage = {r: self._coverage(r) for r in self.requirements}
        for bridge in bridges:
            self.add_bridge(bridge)

    def _expr(self, region, a, b):
        count = self.sequence.regions[region].constant_count
        return (a, b) if count is None else (0, a * count + b)

    def _activation(self, region, command, generation, guard):
        a, b = generation
        ca, cb = self._expr(region, 1, 0)
        guard = _meet(guard, _ge(a, b - command.lower))
        guard = _meet(guard, _ge(ca - a, cb - b - command.upper - 1))
        if command.last_gap is not None:
            guard = _meet(guard, _equal(generation, (ca, cb - 1 - command.last_gap)))
        if guard is None:
            return ()
        if not a:
            return (guard,) if b % command.period in command.residues else ()
        return tuple(
            overlap
            for r in command.residues
            if (
                overlap := _meet(
                    guard, CountDomain(command.period, (r - b) % command.period, minimum=0)
                )
            )
            is not None
        )

    def _bind(self, call, region, command, generation, guard, points):
        key = (
            call,
            region,
            (command.actor, command.task.uid),
            command.channel,
            ("" if command.kind == "task" else command.kind),
        )
        for site in self.marks.get(key, ()):
            node = points.get(site.point)
            if node is None:
                continue
            p = site.position
            wanted = self._expr(region, 1, -1 - p.offset) if p.end else (0, p.offset)
            overlap = _meet(guard, _equal(generation, wanted))
            if overlap is not None:
                self.bindings[site].append((node, overlap))

    def _join(self, *inputs):
        result = self.flow.node()
        for source, guard in inputs:
            self.flow.edge(source, result, guard)
        return result

    def _region(self, call, region, state):
        _, channels, streams, _, _ = self.analyses[region]
        require(len(streams) == 1, "serial completion transfers require one local actor")
        actor, commands = next(iter(streams.items()))
        maximum = max(c.task.stage for c in commands)
        points = [
            m
            for (p, name, *_), items in self.marks.items()
            if (p, name) == (call, region)
            for m in items
        ]
        head = max(
            [1]
            + [c.task.stage + c.lower + 1 for c in commands]
            + [m.position.offset + maximum + 1 for m in points if not m.position.end]
        )
        tail = max(
            [1]
            + [maximum + c.upper + 1 for c in commands]
            + [maximum + m.position.offset + 1 for m in points if m.position.end]
        )
        require(head + tail <= 4096, "completion boundary template budget exceeded")
        sends, waits = defaultdict(list), defaultdict(list)
        # Prefix state is monotone through every omitted steady-state step.
        # No event-publication edge or synthetic child-done crosses that gap.
        steps = [(0, step, ALL) for step in range(head)]
        steps += [
            (1, maximum - tail + j, _ge(*self._expr(region, 1, maximum - tail + j - head)))
            for j in range(tail)
        ]
        for a, step, extent in steps:
            for command in commands:
                generation = self._expr(region, a, step - command.task.stage)
                for guard in self._activation(region, command, generation, extent):
                    pipe = actor, command.task.pipe
                    issue = state.get((actor, "issue"), 0)
                    queue = state.get((*pipe, "queue"), 0)
                    retire = state.get((*pipe, "retire"), 0)
                    implicit = state.get((*pipe, "implicit"), 0)
                    begin = self._join((issue, guard), (queue, guard))
                    if command.kind == "task":
                        self.flow.edge(implicit, begin, guard)
                    end = begin
                    if command.kind in {"task", "set"}:
                        end = self._join((begin, guard))
                    if command.kind in {"set", "barrier"}:
                        self.flow.edge(retire, end, guard)
                    state[*pipe, "queue"] = self._join((queue, ALL), (begin, guard))
                    native_points = {"begin": begin, "end": end}
                    if command.kind == "task":
                        prefix = self._join((retire, ALL), (end, guard))
                        state[*pipe, "retire"] = prefix
                        native_points["prefix"] = prefix
                        if command.task.implicit_completion:
                            state[*pipe, "implicit"] = self._join((implicit, ALL), (end, guard))
                    if command.task.pipe == "S" and command.kind in {"task", "wait"}:
                        state[actor, "issue"] = self._join((issue, ALL), (end, guard))
                    self._bind(call, region, command, generation, guard, native_points)
                    if command.kind == "set":
                        sends[command.channel].append((generation, end, guard))
                    if command.kind == "wait":
                        waits[command.channel].append((generation, begin, guard))
        for name, channel in channels.items():
            require(channel.kind == "local", "serial transfers require local event contracts")
            for (a, b), target, target_guard in waits[name]:
                for generation, source, source_guard in sends[name]:
                    guard = _meet(
                        _meet(target_guard, source_guard),
                        _equal(generation, (a, b - channel.distance)),
                    )
                    self.flow.edge(source, target, guard)

    def add_bridge(self, bridge):
        for call in self.calls:
            target_call = call + int(bridge.carried)
            if target_call not in self.calls:
                continue
            source = mark(self.sequence, bridge.source, "prefix", call)
            destination = mark(self.sequence, bridge.destination, "begin", target_call)
            require(
                source in self.bindings and destination in self.bindings,
                "protocol endpoint missing from native completion transfers",
            )
            for left, a in self.bindings[source]:
                for right, b in self.bindings[destination]:
                    for domain in bridge.domains:
                        self.flow.edge(left, right, _meet(_meet(a, b), domain))

    def _coverage(self, requirement):
        bit = 1 << self.sources[requirement.source]
        result = []
        for node, active in self.bindings[requirement.destination]:
            for guard, bits in self.flow.facts[node].items():
                if bits & bit:
                    overlap = _meet(_meet(guard, active), requirement.domain)
                    if overlap is not None:
                        result.append(overlap)
        return tuple(result)

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
        mark(sequence, source, point),
        mark(sequence, destination, "begin"),
        domain,
        obligation.hazard,
    )

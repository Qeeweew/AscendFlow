"""Native relations for the admitted serial region/call control domain.

This adapter uses the explicit serial Sequence contract. It is not valid for
overlapping CV tasks. Conditional path composition and endpoint identity are
shared with leaf/invocation checks in completion.py.
"""

from collections import defaultdict
from dataclasses import replace

from .budget import poll
from .completion import (
    CompletionPaths,
    CountDomain,
    LeafCompletion,
    Point,
    Position,
    Requirement,
    Site,
    _first,
    canonical,
    covers,
    uncovered,
)
from .model import require
from .region_summary import Access, ReadSpan


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


def _before(sequence, left, right, domain):
    """Same-region, same-pipe issue order, for the entire parameter domain."""
    if left.region != right.region:
        names = list(sequence.regions)
        return names.index(left.region) < names.index(right.region)
    tasks = _data(sequence, left.region).program.tasks
    lt, rt = _task(sequence, left), _task(sequence, right)
    a, b = _expression(sequence, left)
    c, d = _expression(sequence, right)
    delta = (c - a) * _first(domain) + d + rt.stage - b - lt.stage
    return (
        delta >= 0
        and (domain.exact is not None or c - a >= 0)
        and (delta > 0 or (lt.issue_order, tasks.index(lt)) <= (rt.issue_order, tasks.index(rt)))
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


class LeafRelations:
    def __init__(self, sequence, analyses):
        self.sequence, self.analyses = sequence, analyses
        self.inside = {}
        self.cache = {}
        self.site_cache = {}
        self.direct_cache, self.active_cache, self.capture_cache = {}, {}, {}
        for name, analysis in analyses.items():
            loop, _, commands, graph, _ = analysis
            native = {
                (actor, c.task.uid): c
                for actor, items in commands.items()
                for c in items
                if c.kind == "task"
            }
            endpoints = {
                (actor, c.channel, c.kind): c
                for actor, items in commands.items()
                for c in items
                if c.kind in {"set", "wait"}
            }
            self.inside[name] = LeafCompletion(loop, graph, native, endpoints)

    def inherited(self, source, destination, domain):
        key = source, destination, domain
        if key not in self.cache:
            self.cache[key] = self._inherited(source, destination, domain)
        return self.cache[key]

    def site_path(self, region, left, right, domain):
        if (left.region, left.call) != (right.region, right.call):
            return False
        left, right = left.leaf, right.leaf
        constant = self.sequence.regions[region].constant_count
        # A fixed-count leaf's boundary proof is independent of the parent's
        # dynamic count family; cache before repeatedly normalizing its sites.
        key = region, left, right, domain if constant is None else None
        if key not in self.site_cache:
            self.site_cache[key] = self._site_path(region, left, right, domain)
        return self.site_cache[key]

    def _site_path(self, region, left, right, domain):
        inside = self.inside[region]
        loop, _, commands, _, _ = self.analyses[region]
        positions = [left.position, right.position]
        if loop.constant_count is not None:
            count = loop.constant_count
            positions = [
                Position(True, count - 1 - p.evaluate(count))
                if p.evaluate(count) > count // 2
                else Position(False, p.evaluate(count))
                for p in positions
            ]
            head = max([0] + [p.offset for p in positions if not p.end])
            tail = max([0] + [p.offset for p in positions if p.end])
            window = max(
                [head + tail + 1, inside.graph.period]
                + [c.lower + c.upper + 1 for items in commands.values() for c in items]
                + [
                    p.offset
                    + (inside.command(site).lower if p.end else inside.command(site).upper)
                    + 1
                    for p, site in zip(positions, (left, right))
                ]
            )
            domain = CountDomain(
                inside.graph.period, count % inside.graph.period, minimum=min(count, window)
            )
        return inside.prove(
            left if left.position == positions[0] else replace(left, position=positions[0]),
            right if right.position == positions[1] else replace(right, position=positions[1]),
            domain,
        )

    def _inherited(self, source, destination, domain, *, capture_only=False):
        seq = self.sequence
        st, dt = _task(seq, source), _task(seq, destination)
        if not capture_only and (st.pipe == "S" or st.pipe == dt.pipe and st.implicit_completion):
            return True
        # Any real Set/barrier in the receiving region on the source pipe also
        # captures preceding regions' native work. Only use an actual completed
        # path from that capture to this first consumer; source order is not one.
        loop, _, commands, _, _ = self.analyses[destination.region]
        inside = self.inside[destination.region]
        local_domain = domain
        target_position = destination.position
        if loop.constant_count is not None:
            # A stronger static boundary family contains this fixed count.
            # Never build an N-step history just to prove an entry gate.
            if dt.last:
                target_position = Position(True, 0)
            window = max(
                [1, inside.graph.period, 0 if target_position.end else target_position.offset + 1]
                + [c.lower + c.upper + 1 for items in commands.values() for c in items]
            )
            local_domain = CountDomain(
                inside.graph.period,
                loop.constant_count % inside.graph.period,
                minimum=min(loop.constant_count, window),
            )
        for actor, items in commands.items():
            for command in items:
                if command.kind not in {"set", "barrier"} or command.task.pipe != st.pipe:
                    continue
                if command.last_gap is not None:
                    positions = [Position(True, command.last_gap)]
                else:
                    positions = [
                        Position(False, command.lower + (r - command.lower) % command.period)
                        for r in command.residues
                    ]
                for position in positions:
                    point = Point(destination.region, command.task.uid, position)
                    if not _valid_point(seq, point, domain):
                        continue
                    ca, cb = _expression(seq, point)
                    n = _first(domain)
                    g = ca * n + cb
                    length = n if loop.constant_count is None else loop.constant_count
                    if not command.lower <= g < length - command.upper:
                        continue
                    if ca and domain.exact is None and domain.period % command.period:
                        continue
                    if g % command.period not in command.residues:
                        continue
                    # For a nonterminal fixed head, lower bound must also ensure
                    # the upper activation gap for every N, not just one sample.
                    if loop.constant_count is None and not _nonnegative(
                        1 - ca, -cb - command.upper - 1, domain
                    ):
                        continue
                    actor_key = actor, destination.task
                    if command.kind == "set":
                        start = Site(
                            (actor, command.task.uid), position, "end", command.channel, "set"
                        )
                        if inside.prove(
                            start, Site(actor_key, target_position, "begin"), local_domain
                        ):
                            return True
                    else:
                        start = Site((actor, command.task.uid), position, "begin", direct=command)
                        if inside.prove(
                            start, Site(actor_key, target_position, "begin"), local_domain
                        ):
                            return True
        return False


class RegionCompletion:
    """Serial control adapter; overlapping CV must supply its own real issue edges."""

    def __init__(self, sequence, analyses, requirements, *, calls=(0, 1), native=None, bridges=()):
        self.sequence, self.analyses = sequence, analyses
        self.calls = calls
        self.leaves = native if native is not None else LeafRelations(sequence, analyses)
        require(
            self.leaves.sequence is sequence and self.leaves.analyses is analyses,
            "completion caches belong to one immutable analysis snapshot",
        )
        self.requirements = tuple(requirements)
        self.required_marks = {m for r in self.requirements for m in (r.source, r.destination)}
        self.direct_cache, self.active_cache = self.leaves.direct_cache, self.leaves.active_cache
        self.capture_cache = self.leaves.capture_cache
        self.candidate_cache = {}
        self.neighbors = {}
        self.queries, self.visits = 0, 0
        self.order = {name: i for i, name in enumerate(sequence.regions)}
        self.paths = CompletionPaths(self.successors, self.active)
        self._index_nodes(bridges)
        for bridge in bridges:
            self.add_bridge(bridge)
        for dep in sequence.invocation_dependencies():
            access = dep.destination.access
            if isinstance(access, ReadSpan) and access.first != access.last:
                require(
                    self.leaves.inside[dep.destination.region].prefix_monotonic(
                        access.task, "begin"
                    ),
                    "invocation reader family lacks FIFO start order",
                )

    def command(self, m):
        return self.leaves.inside[m.region].command(m)

    def active(self, m, domain):
        key = m.region, m.leaf, domain
        if key not in self.active_cache:
            self.active_cache[key] = covers(domain, self.activation(m, domain))
        return self.active_cache[key]

    def activation(self, m, domain):
        return self.leaves.inside[m.region].activation(m, domain)

    def direct(self, left, right, domain):
        # Direct native/leaf relations depend on relative calls, never absolute
        # p. Share their immutable query cache between p and p+1; bridge paths
        # remain explicit and are checked separately for their guards.
        key = left.region, left.leaf, right.region, right.leaf, right.call - left.call, domain
        if key not in self.direct_cache:
            self.direct_cache[key] = self._direct(left, right, domain)
        return self.direct_cache[key]

    def _direct(self, left, right, domain):
        if (left.call, self.order[left.region]) > (right.call, self.order[right.region]):
            return False
        if not self.active(left, domain) or not self.active(right, domain):
            return False
        if left == right:
            return True
        a, b = self.command(left), self.command(right)
        lp, rp = left.point, right.point
        same_pipe = a.task.pipe == b.task.pipe
        local = left.call == right.call and left.region == right.region
        if local:
            x, y = _expression(self.sequence, left.location)
            u, v = _expression(self.sequence, right.location)
            delta = (u - x) * _first(domain) + v + b.task.stage - y - a.task.stage
            ordered = (
                delta >= 0
                and (domain.exact is not None or u >= x)
                and (delta > 0 or a.index <= b.index)
            )
            strict = ordered and (delta > 0 or a.index < b.index)
        else:
            ordered = (left.call, self.order[left.region]) < (right.call, self.order[right.region])
            if not ordered:
                return False
            strict = True
        # FIFO begins followed by a real native begin->end->prefix chain.
        # This never promotes a source end/Set effect into a pipe fence.
        if (
            ordered
            and same_pipe
            and lp == "begin"
            and (rp == "begin" or b.kind == "task" and rp in {"end", "prefix"})
        ):
            return True
        if (
            strict
            and a.kind == "task"
            and lp == "end"
            and rp == "begin"
            and (
                a.task.pipe == "S" or same_pipe and a.task.implicit_completion and b.kind == "task"
            )
        ):
            return True
        if ordered and a.kind == "task" and lp in {"end", "prefix"}:
            if same_pipe and (
                (b.kind == "task" and rp == "prefix")
                or (b.kind == "set" and rp == "end")
                or (b.kind == "barrier" and rp == "begin")
            ):
                return True  # A real retirement capture includes earlier native work.
            if not local and b.kind == "task" and rp == "begin":
                # Do not treat a native prefix as an implicit task completion:
                # only an actual Set/barrier can capture arbitrary earlier work.
                key = a.task.pipe, right.location, domain
                if key not in self.capture_cache:
                    self.capture_cache[key] = self.leaves._inherited(
                        left.location, right.location, domain, capture_only=True
                    )
                return self.capture_cache[key]
        if local:
            return self.leaves.site_path(left.region, left, right, domain)
        return False

    def _index_nodes(self, bridges):
        # Candidate endpoints are known before selection. Adding a protocol
        # changes edges, not the universe or identity of native milestones.
        values = set(self.required_marks)
        for bridge in bridges:
            values.update(
                (
                    mark(self.sequence, bridge.source, "prefix"),
                    mark(self.sequence, bridge.destination, "begin"),
                )
            )
        for value in tuple(values):
            for point in ("begin", "end", "prefix"):
                values.add(replace(value, point=point, channel="", action="", direct=None))
            if self.command(value).kind == "set":
                values.add(replace(value, point="begin"))
        values |= {replace(value, call=call) for value in tuple(values) for call in self.calls}
        require(len(values) <= 2048, "sequence call milestone budget exceeded")
        self.nodes = frozenset(values)
        self.scopes, self.begins, self.captures = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        self.native_begins, self.event_begins = [], []
        for value in sorted(values, key=lambda m: (m.call, self.order[m.region], repr(m))):
            command = self.command(value)
            self.scopes[value.call, value.region].append(value)
            if value.point == "begin":
                self.begins[command.task.pipe].append(value)
                (self.native_begins if command.kind == "task" else self.event_begins).append(value)
            if (
                command.kind == "task"
                and value.point == "prefix"
                or command.kind == "set"
                and value.point == "end"
                or command.kind == "barrier"
                and value.point == "begin"
            ):
                self.captures[command.task.pipe].append(value)

    def add_bridge(self, bridge):
        for call in self.calls:
            target_call = call + int(bridge.carried)
            if target_call not in self.calls:
                continue
            source = mark(self.sequence, bridge.source, "prefix", call)
            destination = mark(self.sequence, bridge.destination, "begin", target_call)
            require(
                source in self.nodes and destination in self.nodes,
                "protocol endpoint absent from immutable completion index",
            )
            for domain in bridge.domains:
                self.paths.add(source, destination, domain)

    def covered(self, requirement):
        return covers(requirement.domain, self.coverage(requirement))

    def missing(self, requirement):
        return uncovered(requirement.domain, self.coverage(requirement))

    def coverage(self, requirement):
        self.queries += 1
        poll("sequence call paths", queries=self.queries, visits=self.visits)
        require(self.queries <= 20000, "sequence call coverage budget exceeded")
        source, destination, domain = (
            requirement.source,
            requirement.destination,
            canonical(requirement.domain),
        )
        require(
            source in self.nodes and destination in self.nodes,
            "completion query absent from immutable endpoint index",
        )
        if self.direct(source, destination, domain):
            return (domain,)
        return self.paths.coverage(source, destination, domain)

    def successors(self, value, domain):
        key = value, domain
        if key not in self.neighbors:
            successors = []
            source_domains = self.activation(value, domain)
            for candidate in self.candidates(value):
                self.visits += 1
                if self.visits % 512 == 0:
                    poll("region completion milestones", visits=self.visits, nodes=len(self.nodes))
                require(self.visits <= 2_000_000, "region completion path-state budget exceeded")
                if candidate == value:
                    continue
                for source_domain in source_domains:
                    for guard in self.activation(candidate, source_domain):
                        if self.direct(value, candidate, guard):
                            successors.append((candidate, guard))
            self.neighbors[key] = tuple(successors)
        return self.neighbors[key]

    def candidates(self, source):
        """Index admissible relation kinds; _direct still proves every edge.

        Same-leaf relations use the real leaf graph. Cross-scope relations can
        only use FIFO begins, actual retirement captures, and native incoming
        captures. Scope identity alone never supplies a completion edge.
        """
        command = self.command(source)
        key = source.call, source.region, command.kind, source.point, command.task.pipe
        if key not in self.candidate_cache:
            later = []
            if source.point == "begin":
                later.extend(self.begins[command.task.pipe])
            if command.kind == "task" and source.point in {"end", "prefix"}:
                later.extend(self.native_begins)
                later.extend(self.captures[command.task.pipe])
                if source.point == "end" and command.task.pipe == "S":
                    later.extend(self.event_begins)
            rank = source.call, self.order[source.region]
            local = self.scopes[source.call, source.region]
            self.candidate_cache[key] = tuple(
                dict.fromkeys(
                    [*local, *(m for m in later if rank < (m.call, self.order[m.region]))]
                )
            )
        return self.candidate_cache[key]


def memory_requirement(sequence, source, destination, domain, obligation):
    access = obligation.source.access
    point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
    return Requirement(
        mark(sequence, source, point),
        mark(sequence, destination, "begin"),
        domain,
        obligation.hazard,
    )

"""Compose actual region milestones across adjacent serial invocations.

The two call coordinates below are symbolic p and p+1, never runtime expansion.
Leaf progress and lexical forward channels give a base case; carried channels
only depend on an earlier call. Actual native prefixes discharge memory, and
separate Wait-to-Set obligations discharge finite event resources.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from math import gcd, lcm

from .budget import poll
from .invocation_sync import Site, _first
from .model import require
from .periodic import bounded_period
from .region_summary import Access, CountDomain, Position, ReadSpan, summarize_local_events
from .sequence_sync import (
    BoundaryProof,
    Bridge,
    Point,
    _covers,
    _data,
    _domain_row,
    _expression,
    _nonnegative,
    _point,
    _task,
    _valid_point,
)


@dataclass(frozen=True)
class Mark:
    region: str
    site: Site
    call: int = 0
    _hash: int = field(init=False, repr=False, compare=False)
    point: Point = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash((self.region, self.site, self.call)))
        object.__setattr__(self, "point", Point(self.region, self.site.task[1], self.site.position))

    def __hash__(self):
        return self._hash


@dataclass(frozen=True)
class Requirement:
    source: Mark
    destination: Mark
    domain: CountDomain
    reason: str


def canonical(domain):
    return (
        CountDomain(1, 0, exact=domain.exact)
        if domain.exact is not None
        else CountDomain(domain.period, domain.residue, minimum=_first(domain))
    )


def position(sequence, region, point):
    count = sequence.regions[region].constant_count
    if count is None:
        return point
    index = point.evaluate(count)
    return Position(False, index) if index <= count // 2 else Position(True, count - 1 - index)


def intersection(left, right):
    if left.exact is not None:
        return left if right.contains(left.exact) else None
    if right.exact is not None:
        return right if left.contains(right.exact) else None
    divisor = gcd(left.period, right.period)
    if (right.residue - left.residue) % divisor:
        return None
    period = bounded_period(lcm(left.period, right.period), "coupled region path predicate")
    other = right.period // divisor
    shift = (
        0
        if other == 1
        else ((right.residue - left.residue) // divisor * pow(left.period // divisor, -1, other))
        % other
    )
    return CountDomain(
        period,
        (left.residue + left.period * shift) % period,
        minimum=max(_first(left), _first(right)),
    )


def mark(sequence, point, milestone, call=0):
    actor = next(iter(sequence.regions[point.region].actors))
    return Mark(
        point.region,
        Site((actor, point.task), position(sequence, point.region, point.position), milestone),
        call,
    )


def memory_requirements(sequence):
    result = []
    for dep in sequence.summary.invocation_dependencies():
        # Invocation GM frames are distinct; these are actual reused local bytes.
        require(not dep.storage.owner.external.data, "call transfer includes unpartitioned GM")
        source = _point(sequence, dep.source, dep.domain, source=True)
        destination = _point(sequence, dep.destination, dep.domain, source=False)
        access = dep.source.access
        point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
        result.append(
            Requirement(
                mark(sequence, source, point),
                mark(sequence, destination, "begin", 1),
                dep.domain,
                f"{dep.hazard} {dep.storage.owner.sym_name.data}[{dep.start}:{dep.stop}]",
            )
        )
    return tuple(result)


def event_requirements(sequence):
    result = []
    for name, loop in sequence.regions.items():
        for event in summarize_local_events(loop):
            # A fixed-count leaf is active independently of the common dynamic N.
            domain = event.domain if loop.constant_count is None else CountDomain(1, 0, minimum=0)
            left = Site(
                event.consumed.task,
                position(sequence, name, event.consumed.position),
                "begin",
                event.channel,
                "wait",
            )
            right = Site(
                event.republished.task,
                position(sequence, name, event.republished.position),
                "end",
                event.channel,
                "set",
            )
            result.append(
                Requirement(
                    Mark(name, left),
                    Mark(name, right, 1),
                    domain,
                    f"event {name}/{event.channel}:{event.event_id} Wait-before-Set",
                )
            )
    return tuple(result)


def boundary_requirements(sequence, bridges):
    result = []
    for bridge in bridges:
        if bridge.event_id < 0:
            continue
        if bridge.carried and sequence.invocation.bound.constant == 2:
            continue  # One send/one receive; no second publication of this ID.
        for domain in bridge.domains:
            # Consumer begin follows the real boundary Wait; source prefix is
            # captured by the real Set. These are sufficient, not equivalent,
            # resource endpoints. They never retire a buffer on Wait alone.
            result.append(
                Requirement(
                    mark(sequence, bridge.destination, "begin"),
                    mark(sequence, bridge.source, "prefix", 0 if bridge.carried else 1),
                    domain,
                    f"event {bridge.name} Wait-before-Set",
                )
            )
    return tuple(result)


class CallProof:
    def __init__(self, sequence, analyses, requirements):
        self.sequence, self.analyses = sequence, analyses
        self.leaves = BoundaryProof(sequence, analyses)
        self.requirements = tuple(requirements)
        self.required_marks = {m for r in self.requirements for m in (r.source, r.destination)}
        self.direct_cache, self.active_cache = {}, {}
        self.capture_cache, self.node_cache = {}, {}
        self.neighbors = {}
        self.queries, self.visits = 0, 0
        self.order = {name: i for i, name in enumerate(sequence.regions)}
        for dep in sequence.summary.invocation_dependencies():
            access = dep.destination.access
            if isinstance(access, ReadSpan) and access.first != access.last:
                require(
                    self.leaves.inside[dep.destination.region].prefix_monotonic(
                        access.task, "begin"
                    ),
                    "invocation reader family lacks FIFO start order",
                )

    def command(self, m):
        return self.leaves.inside[m.region].command(m.site)

    def active(self, m, domain):
        key = m, domain
        if key in self.active_cache:
            return self.active_cache[key]
        seq = self.sequence
        value = _valid_point(seq, m.point, domain)
        command = self.command(m)
        a, b = _expression(seq, m.point)
        count = seq.regions[m.region].constant_count
        ca, cb = (1, 0) if count is None else (0, count)
        value = value and _nonnegative(a, b - command.lower, domain)
        value = value and _nonnegative(ca - a, cb - b - command.upper - 1, domain)
        if command.last_gap is not None:
            da, db = a - ca, b - cb + 1 + command.last_gap
            value = value and (
                da * domain.exact + db == 0 if domain.exact is not None else da == db == 0
            )
        if value:
            start = a * _first(domain) + b
            step = a * domain.period
            residues = (
                1 if domain.exact is not None else command.period // gcd(command.period, step)
            )
            value = all(
                (start + i * step) % command.period in command.residues for i in range(residues)
            )
        self.active_cache[key] = value
        return value

    def direct(self, left, right, domain):
        key = left, right, domain
        if key not in self.direct_cache:
            self.direct_cache[key] = self._direct(left, right, domain)
        return self.direct_cache[key]

    def _direct(self, left, right, domain):
        if not self.active(left, domain) or not self.active(right, domain):
            return False
        if left == right:
            return True
        a, b = self.command(left), self.command(right)
        lp, rp = left.site.point, right.site.point
        same_pipe = a.task.pipe == b.task.pipe
        local = left.call == right.call and left.region == right.region
        if local:
            x, y = _expression(self.sequence, left.point)
            u, v = _expression(self.sequence, right.point)
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
        # FIFO starts/Wait consumption are not FIFO asynchronous completion.
        if ordered and same_pipe and lp == rp == "begin":
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
                key = a.task.pipe, right.point, domain
                if key not in self.capture_cache:
                    self.capture_cache[key] = self.leaves._inherited(
                        left.point, right.point, domain, capture_only=True
                    )
                return self.capture_cache[key]
        if local:
            return self.leaves.site_path(left.region, left.site, right.site, domain)
        return False

    def nodes(self, bridges, source, destination):
        key = (
            tuple((b.source, b.destination, b.carried) for b in bridges),
            None if source in self.required_marks else source,
            None if destination in self.required_marks else destination,
        )
        if key in self.node_cache:
            return self.node_cache[key]
        values = {source, destination}
        for requirement in self.requirements:
            values.update((requirement.source, requirement.destination))
        for bridge in bridges:
            for call in (0, 1):
                if call + int(bridge.carried) <= 1:
                    values.add(mark(self.sequence, bridge.source, "prefix", call))
                    values.add(
                        mark(self.sequence, bridge.destination, "begin", call + int(bridge.carried))
                    )
        # Expose native milestones behind real event endpoints; no done token.
        for value in tuple(values):
            for point in ("begin", "end", "prefix"):
                values.add(replace(value, site=Site(value.site.task, value.site.position, point)))
            if self.command(value).kind == "set":
                values.add(replace(value, site=replace(value.site, point="begin")))
        # The same boundary shape is available in both symbolic calls.
        values |= {replace(value, call=1 - value.call) for value in tuple(values)}
        require(len(values) <= 2048, "sequence call milestone budget exceeded")
        result = (
            len(self.node_cache),
            tuple(sorted(values, key=lambda m: (m.call, self.order[m.region], repr(m.site)))),
        )
        self.node_cache[key] = result
        return result

    def covered(self, requirement, bridges):
        self.queries += 1
        poll("sequence call paths", queries=self.queries, visits=self.visits)
        require(self.queries <= 20000, "sequence call coverage budget exceeded")
        source, destination, domain = (
            requirement.source,
            requirement.destination,
            canonical(requirement.domain),
        )
        pool, nodes = self.nodes(bridges, source, destination)
        edges = defaultdict(list)
        for bridge in bridges:
            for call in (0, 1):
                target_call = call + int(bridge.carried)
                if target_call > 1:
                    continue
                a = mark(self.sequence, bridge.source, "prefix", call)
                b = mark(self.sequence, bridge.destination, "begin", target_call)
                edges[a].extend((b, canonical(guard)) for guard in bridge.domains)
        reached = defaultdict(list)
        pending = deque([(source, domain)])
        while pending:
            value, active = pending.popleft()
            if _covers(active, reached[value]):
                continue
            reached[value].append(active)
            if value == destination:
                if _covers(domain, reached[value]):
                    return True
                continue
            key = pool, value, active
            if key not in self.neighbors:
                successors = []
                for candidate in nodes:
                    self.visits += 1
                    if self.visits % 512 == 0:
                        poll("sequence call milestones", visits=self.visits, nodes=len(nodes))
                    require(self.visits <= 2_000_000, "sequence call path-state budget exceeded")
                    if candidate != value and self.direct(value, candidate, active):
                        successors.append(candidate)
                self.neighbors[key] = tuple(successors)
            for candidate in self.neighbors[key]:
                if not _covers(active, reached[candidate]):
                    pending.append((candidate, active))
            for candidate, guard in edges[value]:
                overlap = intersection(active, guard)
                if (
                    overlap is not None
                    and self.active(value, overlap)
                    and self.active(candidate, overlap)
                ):
                    pending.append((candidate, overlap))
        return _covers(domain, reached[destination])


def verify_calls(sequence, analyses, bridges):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return {}
    memory = memory_requirements(sequence)
    events = event_requirements(sequence) + boundary_requirements(sequence, bridges)
    proof = CallProof(sequence, analyses, memory + events)
    for requirement in memory + events:
        require(
            proof.covered(requirement, bridges),
            f"missing sequence invocation {requirement.reason}: {requirement.source} -> {requirement.destination}, {requirement.domain}",
        )
    return {
        "invocation_bound": sequence.invocation.bound.text(),
        "invocation_frames": len(sequence.invocation.frames),
        "invocation_memory_obligations": len(memory),
        "invocation_event_domains": len(events),
        "invocation_proof_queries": proof.queries,
        "invocation_path_visits": proof.visits,
    }


def plan_calls(sequence, analyses, forward, banks):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return []
    memory, events = memory_requirements(sequence), event_requirements(sequence)
    proof = CallProof(sequence, analyses, memory + events)
    proposals = []
    # A late reader release often covers an earlier writer and several input
    # rings. Examine real cross-pipe WARs before manufacturing same-pipe fences.
    ordered = sorted(
        memory,
        key=lambda r: (
            not r.reason.startswith("WAR"),
            _task(sequence, r.source.point).pipe == _task(sequence, r.destination.point).pipe,
            -proof.order[r.source.region],
            proof.order[r.destination.region],
            repr(r),
        ),
    )
    for requirement in ordered:
        if proof.covered(requirement, [*forward, *proposals]):
            continue
        source, destination = requirement.source.point, requirement.destination.point
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        proposals.append(
            Bridge(
                source,
                destination,
                (requirement.domain,),
                -1 if sp == dp else 0,
                f"call_c{len(proposals)}",
                True,
            )
        )
    grouped = {}
    for bridge in proposals:
        key = bridge.source, bridge.destination, bridge.event_id
        grouped.setdefault(key, set()).update(bridge.domains)
    proposals = [
        Bridge(s, d, tuple(sorted(domains, key=_domain_row)), event_id, f"call_c{i}", True)
        for i, ((s, d, event_id), domains) in enumerate(grouped.items())
    ]
    for candidate in reversed(tuple(proposals)):
        remaining = [b for b in proposals if b is not candidate]
        if all(proof.covered(r, [*forward, *remaining]) for r in memory + events):
            proposals = remaining
    actor = next(iter(next(iter(sequence.regions.values())).actors))
    assigned = []
    for bridge in proposals:
        if bridge.event_id >= 0:
            sp, dp = _task(sequence, bridge.source).pipe, _task(sequence, bridge.destination).pipe
            key = actor, sp, dp
            reserved = int(
                (sp, dp) == ("S", "FIX") and _data(sequence, bridge.source.region).program.config
            )
            event_id = max(banks[key], reserved)
            require(event_id < 8, f"{key}: invocation event bank exhausted; no drain fallback")
            banks[key] = event_id + 1
            bridge = replace(bridge, event_id=event_id)
        assigned.append(bridge)
    return assigned

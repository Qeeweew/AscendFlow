"""Shared symbolic completion endpoints and conditional path composition.

No region, actor or call order is implicit here. Only actual native relations
and explicitly guarded protocol edges can establish a path. CountDomain is
the currently supported one-uniform-count predicate, not a general nonlinear
or multi-axis domain solver.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain
from math import gcd, lcm

from .budget import poll
from .model import require
from .periodic import bounded_period, project


@dataclass(frozen=True)
class CountDomain:
    period: int
    residue: int
    exact: int | None = None
    minimum: int | None = None

    def contains(self, count):
        return (
            count == self.exact
            if self.exact is not None
            else (
                count >= (self.period if self.minimum is None else self.minimum)
                and count % self.period == self.residue
            )
        )


@dataclass(frozen=True)
class Position:
    """A generation relative to this child invocation, never to its parent."""

    end: bool
    offset: int

    def evaluate(self, count):
        return count - 1 - self.offset if self.end else self.offset


@dataclass(frozen=True)
class Point:
    """Native insertion location; deliberately not a completion milestone."""

    region: str
    task: str
    position: Position


def _first(domain):
    if domain.exact is not None:
        return domain.exact
    minimum = domain.period if domain.minimum is None else domain.minimum
    return minimum + (domain.residue - minimum) % domain.period


@dataclass(frozen=True)
class Site:
    """A real milestone with its actor, lexical scope and relative call.

    Scope/call are identity only: their order grants no happens-before edge.
    ``prefix`` denotes an actual native retirement capture, not a block exit.
    """

    task: tuple[str, str]
    position: Position
    point: str
    channel: str = ""
    action: str = ""
    direct: object = None
    region: str = ""
    call: int = 0
    _hash: int = field(init=False, repr=False, compare=False)
    location: Point = field(init=False, repr=False, compare=False)
    local: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        local = (
            self.region,
            self.task,
            self.position,
            self.point,
            self.channel,
            self.action,
            self.direct,
        )
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "_hash", hash((local, self.call)))
        object.__setattr__(self, "location", Point(self.region, self.task[1], self.position))

    def __hash__(self):
        return self._hash

    @cached_property
    def leaf(self):
        """Same command within its leaf; used only after checking scope equality."""
        return replace(self, region="", call=0) if self.region or self.call else self


@dataclass(frozen=True)
class Requirement:
    source: Site
    destination: Site
    domain: CountDomain
    reason: str


def canonical(domain):
    return (
        CountDomain(1, 0, exact=domain.exact)
        if domain.exact is not None
        else CountDomain(domain.period, domain.residue, minimum=_first(domain))
    )


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


def covers(required, guards):
    guards = tuple(guards)
    if not guards:
        return False
    if required in guards:
        return True
    if required.exact is not None:
        return any(g.contains(required.exact) for g in guards)
    # Membership is constant per residue between threshold/exact boundaries.
    # Never enumerate up to a large integer boundary such as one million.
    period = required.period
    cuts = {_first(required)}
    for guard in guards:
        if guard.exact is None:
            period = bounded_period(lcm(period, guard.period), "region activation coverage")
            cuts.add(guard.period if guard.minimum is None else guard.minimum)
        else:
            cuts.update((guard.exact, guard.exact + 1))
    cuts = sorted(c for c in cuts if c >= _first(required))
    for i, lower in enumerate(cuts):
        upper = cuts[i + 1] if i + 1 < len(cuts) else None
        for r in range(period):
            n = lower + (r - lower) % period
            if (
                (upper is None or n < upper)
                and required.contains(n)
                and not any(g.contains(n) for g in guards)
            ):
                return False
    return True


class LeafCompletion:
    def __init__(self, loop, graph, native, endpoints):
        self.loop, self.graph, self.native, self.endpoints = loop, graph, native, endpoints
        self.cache, self.monotonic = {}, {}
        self.activations = {}
        self.queries = 0

    def command(self, site):
        if site.direct is not None:
            return site.direct
        return (
            self.endpoints[site.task[0], site.channel, site.action]
            if site.channel
            else self.native[site.task]
        )

    def prove(self, left, right, domain):
        if (left.region, left.call) != (right.region, right.call):
            return False
        # This adapter has one actual leaf graph. Scope is already checked;
        # its native proof is translation-invariant in the outer call.
        left, right = left.leaf, right.leaf
        key = left, right, domain
        if key not in self.cache:
            self.queries += 1
            poll("invocation completion", queries=self.queries, states=self.graph.domain_states)
            require(self.queries <= 20000, "invocation boundary proof query budget exceeded")
            self.cache[key] = (
                covers(domain, self.activation(left, domain))
                and covers(domain, self.activation(right, domain))
                and self._prove(left, right, domain)
            )
        return self.cache[key]

    def activation(self, site, domain):
        """Intersect a real command's activation with a count predicate.

        Head/tail coordinates make the bounds a threshold or one exact count;
        modulo classification depends on the command period, never on N.
        """
        key = site.leaf, domain
        if key not in self.activations:
            self.activations[key] = self._activation(site, domain)
        return self.activations[key]

    def _activation(self, site, domain):
        command = self.command(site)
        count = self.loop.constant_count
        if count is not None:
            g = site.position.evaluate(count)
            valid = (
                command.lower <= g < count - command.upper
                and g % command.period in command.residues
                and (not command.task.last or g == count - 1)
                and (command.last_gap is None or g == count - 1 - command.last_gap)
            )
            return (domain,) if valid else ()
        p = site.position
        if p.end:
            if (
                p.offset < command.upper
                or command.task.last
                and p.offset != 0
                or command.last_gap is not None
                and p.offset != command.last_gap
            ):
                return ()
            predicates = tuple(
                CountDomain(
                    command.period,
                    (r + 1 + p.offset) % command.period,
                    minimum=p.offset + command.lower + 1,
                )
                for r in command.residues
            )
        else:
            if p.offset < command.lower or p.offset % command.period not in command.residues:
                return ()
            minimum = p.offset + command.upper + 1
            last_gap = 0 if command.task.last else command.last_gap
            if command.task.last and command.last_gap not in (None, 0):
                return ()
            if last_gap is not None:
                exact = p.offset + last_gap + 1
                if exact < minimum:
                    return ()
                predicates = (CountDomain(1, 0, exact=exact),)
            else:
                predicates = (CountDomain(1, 0, minimum=minimum),)
        return tuple(
            overlap for p in predicates if (overlap := intersection(domain, p)) is not None
        )

    def _prove(self, left, right, domain):
        a, b = self.command(left), self.command(right)
        first = _first(domain)
        if domain.exact is not None:
            ag, bg = left.position.evaluate(first), right.position.evaluate(first)
            return (
                self.graph.path(
                    a,
                    ag,
                    b,
                    bg,
                    bg,
                    source_point=left.point,
                    destination_point=right.point,
                    remaining=first - 1 - bg,
                    at_end=True,
                    _shorten=False,
                )
                is not None
            )
        if left.position.end == right.position.end:
            if not left.position.end:
                ag, bg = left.position.offset, right.position.offset
                if max(ag, bg) >= first:
                    return False
                return (
                    self.graph.path(
                        a,
                        ag,
                        b,
                        bg,
                        bg,
                        source_point=left.point,
                        destination_point=right.point,
                        remaining=first - 1 - bg,
                        _shorten=False,
                    )
                    is not None
                )
            for residue, count in project(
                domain.period, (domain.residue,), self.graph.period, first
            ):
                if (
                    self.loop.constant_count is not None
                    and self.loop.constant_count % self.graph.period != residue
                ):
                    continue
                ag, bg = left.position.evaluate(count), right.position.evaluate(count)
                if (
                    min(ag, bg) < 0
                    or self.graph.path(
                        a,
                        ag,
                        b,
                        bg,
                        bg,
                        source_point=left.point,
                        destination_point=right.point,
                        remaining=right.position.offset,
                        at_end=True,
                        _shorten=False,
                    )
                    is None
                ):
                    return False
            return True
        if left.position.end or left.position.offset > right.position.evaluate(first):
            return False
        # A constant head can precede an arbitrary tail through a native
        # retirement prefix. Prove both boundary legs and the one-generation
        # prefix recurrence; induction then spans N without enumerating it.
        for key, command in self.native.items():
            if command.task.last:
                continue
            head = Site(key, left.position, "prefix")
            tail = Site(key, right.position, "prefix")
            if (
                self.prove(left, head, domain)
                and self.prove(tail, right, domain)
                and self.prefix_monotonic(key)
            ):
                return True
        return False

    def prefix_monotonic(self, key, point="prefix"):
        cache_key = key, point
        if cache_key not in self.monotonic:
            command = self.native[key]
            self.monotonic[cache_key] = not command.task.last and all(
                self.graph.path(
                    command,
                    r,
                    command,
                    r + 1,
                    r + 1,
                    source_point=point,
                    destination_point=point,
                    _shorten=False,
                )
                is not None
                for r in range(self.graph.period)
            )
        return self.monotonic[cache_key]


def covered_path(source, destination, domain, edges, successors, active):
    """Prove a whole predicate by composing native and guarded edges.

    A path carries the intersection of *all* its guards. Distinct paths may
    cover different parts of the requested domain; neither an existential
    path nor an endpoint's matching residue suffices. Successors are native
    relations paired with the predicate on which the adapter proves them.
    """
    domain = canonical(domain)
    if source == destination:
        return active(source, domain)
    reached = defaultdict(list)
    pending = deque([(source, domain)])
    visits = 0
    while pending:
        value, guard = pending.popleft()
        visits += 1
        if visits % 256 == 0:
            poll("conditional completion paths", visits=visits)
        require(visits <= 2_000_000, "conditional completion path-state budget exceeded")
        if covers(guard, reached[value]):
            continue
        reached[value].append(guard)
        if value == destination:
            if covers(domain, reached[value]):
                return True
            continue
        for candidate, predicate in chain(successors(value, guard), edges.get(value, ())):
            overlap = intersection(guard, predicate)
            if overlap is not None and active(value, overlap) and active(candidate, overlap):
                if candidate == destination:
                    reached[destination].append(overlap)
                    if covers(domain, reached[destination]):
                        return True
                    continue
                pending.append((candidate, overlap))
    return covers(domain, reached[destination])

"""Shared symbolic completion endpoints and conditional path composition.

No region, actor or call order is implicit here. Only actual native relations
and explicitly guarded protocol edges can establish a path. CountDomain is
the currently supported one-uniform-count predicate, not a general nonlinear
or multi-axis domain solver.
"""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from math import gcd, lcm

from .periodic import bounded_period


@dataclass(frozen=True)
class CountDomain:
    period: int
    residue: int
    exact: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    @cached_property
    def _hash(self):
        return hash((self.period, self.residue, self.exact, self.minimum, self.maximum))

    def __hash__(self):
        return self._hash

    def contains(self, count):
        return (
            count == self.exact
            if self.exact is not None
            else (
                count >= (self.period if self.minimum is None else self.minimum)
                and count % self.period == self.residue
                and (self.maximum is None or count <= self.maximum)
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
    if domain.period == 1:
        return minimum
    return minimum + (domain.residue - minimum) % domain.period


_math_values = ContextVar("xdsl_npu_completion_math", default=None)
_MISSING = object()


class _CompletionMath:
    """Immutable arithmetic values for one compilation, never graph proofs."""

    def __init__(self):
        self.values, self.intersections, self.inclusions = {}, {}, {}
        self.value_requests = self.value_reuses = self.value_creations = 0
        self.canonical_requests = self.canonical_reuses = 0
        self.intersection_requests = self.intersection_reuses = 0
        self.inclusion_requests = self.inclusion_reuses = 0

    def value(self, key, candidate=None):
        self.value_requests += 1
        prior = self.values.get(key)
        if prior is not None:
            self.value_reuses += 1
            return prior
        if candidate is None:
            candidate = CountDomain(*key)
            self.value_creations += 1
        self.values[key] = candidate
        return candidate

    def statistics(self):
        return {
            "value_requests": self.value_requests,
            "value_reuses": self.value_reuses,
            "value_creations": self.value_creations,
            "values": len(self.values),
            "canonical_requests": self.canonical_requests,
            "canonical_reuses": self.canonical_reuses,
            "intersection_requests": self.intersection_requests,
            "intersection_reuses": self.intersection_reuses,
            "intersection_pairs": len(self.intersections),
            "inclusion_requests": self.inclusion_requests,
            "inclusion_reuses": self.inclusion_reuses,
            "inclusion_pairs": len(self.inclusions),
        }


@contextmanager
def completion_math(*, isolated=False):
    """Share pure value results in one invocation; nested analyses compose."""
    active = _math_values.get()
    if active is not None and not isolated:
        yield active
        return
    active = _CompletionMath()
    token = _math_values.set(active)
    try:
        yield active
    finally:
        _math_values.reset(token)


def completion_math_statistics():
    active = _math_values.get()
    return {} if active is None else active.statistics()


def count_domain(period, residue, exact=None, minimum=None, maximum=None):
    """Construct an internal canonical value, sharing it only in this compile."""
    if exact is not None:
        key = 1, 0, exact, None, None
    else:
        first = period if minimum is None else minimum
        if period != 1:
            first += (residue - first) % period
        key = period, residue, None, first, maximum
    active = _math_values.get()
    return CountDomain(*key) if active is None else active.value(key)


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
    region: str = ""
    call: int = 0
    _hash: int = field(init=False, repr=False, compare=False)
    location: Point = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        local = (
            self.region,
            self.task,
            self.position,
            self.point,
            self.channel,
            self.action,
        )
        object.__setattr__(self, "_hash", hash((local, self.call)))
        object.__setattr__(self, "location", Point(self.region, self.task[1], self.position))

    def __hash__(self):
        return self._hash


@dataclass(frozen=True)
class Requirement:
    source: Site
    destination: Site
    domain: CountDomain
    reason: str


def canonical(domain):
    if domain.exact is not None:
        key = 1, 0, domain.exact, None, None
        ready = (domain.period, domain.residue, domain.minimum, domain.maximum) == (
            1,
            0,
            None,
            None,
        )
    else:
        first = _first(domain)
        key = domain.period, domain.residue, None, first, domain.maximum
        ready = domain.minimum == first
    active = _math_values.get()
    if active is None:
        return domain if ready else CountDomain(*key)
    active.canonical_requests += 1
    active.canonical_reuses += ready
    return active.value(key, domain if ready else None)


def intersection(left, right):
    active = _math_values.get()
    if active is None:
        return _intersection(left, right)
    active.intersection_requests += 1
    key = (left, right) if hash(left) <= hash(right) else (right, left)
    known = active.intersections.get(key, _MISSING)
    if known is not _MISSING:
        active.intersection_reuses += 1
        return known
    result = active.intersections[key] = _intersection(left, right)
    return result


def _intersection(left, right):
    if left.exact is not None:
        return canonical(left) if right.contains(left.exact) else None
    if right.exact is not None:
        return canonical(right) if left.contains(right.exact) else None
    first = max(_first(left), _first(right))
    maximum = (
        right.maximum
        if left.maximum is None
        else left.maximum
        if right.maximum is None
        else min(left.maximum, right.maximum)
    )
    if left.period == right.period:
        if left.residue != right.residue:
            return None
        bounded_period(left.period, "coupled region path predicate")
        if maximum is not None and first > maximum:
            return None
        if first == left.minimum and maximum == left.maximum:
            return canonical(left)
        if first == right.minimum and maximum == right.maximum:
            return canonical(right)
        return count_domain(left.period, left.residue, minimum=first, maximum=maximum)
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
    result = count_domain(
        period,
        (left.residue + left.period * shift) % period,
        minimum=first,
        maximum=maximum,
    )
    return result if result.maximum is None or _first(result) <= result.maximum else None


def includes(outer, inner):
    """One arithmetic interval contains another; no path/union proof is cached."""
    if outer is inner or outer == inner:
        return True
    active = _math_values.get()
    if active is None:
        return _includes(outer, inner)
    active.inclusion_requests += 1
    key = outer, inner
    known = active.inclusions.get(key, _MISSING)
    if known is not _MISSING:
        active.inclusion_reuses += 1
        return known
    result = active.inclusions[key] = _includes(outer, inner)
    return result


def _includes(outer, inner):
    first = _first(inner)
    if not outer.contains(first):
        return False
    if inner.exact is not None:
        return True
    last = (
        None
        if inner.maximum is None
        else inner.maximum - (inner.maximum - inner.residue) % inner.period
    )
    if last == first:
        return True
    if outer.exact is not None or inner.period % outer.period:
        return False
    return outer.maximum is None or last is not None and last <= outer.maximum


def covers(required, guards):
    guards = tuple(guards)
    if not guards:
        return False
    if required in guards:
        return True
    if required.exact is not None:
        return any(g.contains(required.exact) for g in guards)
    return next(_uncovered(required, guards), None) is None


def uncovered(required, guards):
    """Exact symbolic difference; interval width never controls enumeration."""
    guards = tuple(guards)
    if required in guards:
        return ()
    if required.exact is not None:
        return () if any(g.contains(required.exact) for g in guards) else (required,)
    return tuple(_uncovered(required, guards))


def coalesce(domains):
    """Join adjacent pieces of the same residue class without filling holes."""
    items = list(dict.fromkeys(canonical(d) for d in domains))
    groups, points = defaultdict(list), {}
    for index, domain in enumerate(items):
        if domain.exact is None:
            groups[domain.period, domain.residue].append(
                (_first(domain), domain.maximum, index, True)
            )
        else:
            points[domain.exact] = index
    result, consumed = [], set()
    for (period, residue), intervals in groups.items():
        # A singleton may bridge two pieces, or a chain of singletons may
        # extend one piece. Sorting once finds that union without restarting
        # an all-pairs scan after every extension. Other residue classes stay
        # separate, and isolated points are emitted once after all groups.
        pieces = intervals + [
            (value, value, index, False)
            for value, index in points.items()
            if value % period == residue
        ]
        pieces.sort(key=lambda piece: piece[0])
        lower = upper = order = None
        family, singles = False, []

        for first, last, index, native in pieces:
            if lower is None or upper is not None and upper + period < first:
                if family:
                    result.append(
                        (order, CountDomain(period, residue, minimum=lower, maximum=upper))
                    )
                    consumed.update(singles)
                lower, upper, order, family = first, last, index, native
                singles = [] if native else [first]
                continue
            upper = None if upper is None or last is None else max(upper, last)
            order = min(order, index)
            family |= native
            if not native:
                singles.append(first)
        if family:
            result.append((order, CountDomain(period, residue, minimum=lower, maximum=upper)))
            consumed.update(singles)
    result.extend((index, items[index]) for value, index in points.items() if value not in consumed)
    result.sort(key=lambda item: item[0])
    return tuple(domain for _, domain in result)


def _uncovered(required, guards):
    # Membership is constant per residue between threshold/exact boundaries.
    # Never enumerate up to a large integer boundary such as one million.
    period = required.period
    cuts = {_first(required)}
    if required.maximum is not None:
        cuts.add(required.maximum + 1)
    for guard in guards:
        if guard.exact is None:
            period = bounded_period(lcm(period, guard.period), "region activation coverage")
            cuts.add(guard.period if guard.minimum is None else guard.minimum)
            if guard.maximum is not None:
                cuts.add(guard.maximum + 1)
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
                last = None if upper is None else upper - 1
                if required.maximum is not None:
                    last = required.maximum if last is None else min(last, required.maximum)
                if last is not None and n + period > last:
                    yield CountDomain(1, 0, exact=n)
                else:
                    yield CountDomain(period, r, minimum=n, maximum=last)

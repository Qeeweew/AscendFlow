"""Shared symbolic completion endpoints and conditional path composition.

No region, actor or call order is implicit here. Only actual native relations
and explicitly guarded protocol edges can establish a path. CountDomain is
the currently supported one-uniform-count predicate, not a general nonlinear
or multi-axis domain solver.
"""

from dataclasses import dataclass, field, replace
from math import gcd, lcm

from .periodic import bounded_period


@dataclass(frozen=True)
class CountDomain:
    period: int
    residue: int
    exact: int | None = None
    minimum: int | None = None
    maximum: int | None = None

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
    return (
        CountDomain(1, 0, exact=domain.exact)
        if domain.exact is not None
        else CountDomain(
            domain.period, domain.residue, minimum=_first(domain), maximum=domain.maximum
        )
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
    maxima = [d.maximum for d in (left, right) if d.maximum is not None]
    result = CountDomain(
        period,
        (left.residue + left.period * shift) % period,
        minimum=max(_first(left), _first(right)),
        maximum=min(maxima) if maxima else None,
    )
    return result if result.maximum is None or _first(result) <= result.maximum else None


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
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                merged = None
                if a.exact is not None or b.exact is not None:
                    single, family = (a, b) if a.exact is not None else (b, a)
                    n = single.exact
                    if family.contains(n):
                        merged = family
                    elif family.exact is None and n % family.period == family.residue:
                        if n + family.period == _first(family):
                            merged = replace(family, minimum=n)
                        elif family.maximum is not None:
                            last = (
                                family.maximum - (family.maximum - family.residue) % family.period
                            )
                            if last + family.period == n:
                                merged = replace(family, maximum=n)
                elif (a.period, a.residue) == (b.period, b.residue):
                    first, second = (a, b) if _first(a) <= _first(b) else (b, a)
                    if first.maximum is None or first.maximum + first.period >= _first(second):
                        maximum = (
                            None
                            if a.maximum is None or b.maximum is None
                            else max(a.maximum, b.maximum)
                        )
                        merged = replace(first, maximum=maximum)
                if merged is not None:
                    items[i] = merged
                    items.pop(j)
                    changed = True
                    break
            if changed:
                break
    return tuple(items)


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

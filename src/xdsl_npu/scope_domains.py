"""One parent clock's activation domains, separate from child native positions.

These intervals constrain a source epoch. A relation with signed distance d
targets parent generation epoch+d; neither number denotes a child exit or
asynchronous completion. Positions with end=True are relative to parent N.
"""

from dataclasses import dataclass, replace

from .completion import CountDomain, Position, _first
from .completion_flow import ALL
from .diagnostics import require
from .periodic import bounded_period
from .symbolic_completion import add, holds, le, meet

FIRST = Position(False, 0)
LAST = Position(True, 0)


@dataclass(frozen=True)
class Activation:
    count: CountDomain = ALL
    first: Position = FIRST
    last: Position = LAST
    period: int = 1
    residues: tuple[int, ...] = (0,)


def validate_activation(value: Activation, *, distance: int = 0):
    """Validate both endpoint epochs on the entire nonempty activation domain."""
    require(isinstance(value, Activation), "scope protocol needs an activation domain")
    require(type(distance) is int, "scope distance must be an integer")
    domain = value.count
    require(isinstance(domain, CountDomain), "scope activation needs a count domain")
    require(
        type(domain.period) is int
        and domain.period > 0
        and type(domain.residue) is int
        and 0 <= domain.residue < domain.period
        and all(
            v is None or type(v) is int and v >= 0
            for v in (domain.exact, domain.minimum, domain.maximum)
        ),
        "invalid scope count domain",
    )
    bounded_period(domain.period, "scope count predicate")
    require(
        domain.exact is not None or domain.maximum is None or _first(domain) <= domain.maximum,
        "scope count domain is empty",
    )
    require(type(value.period) is int, "scope activation period must be an integer")
    bounded_period(value.period, "scope activation")
    require(
        value.residues
        and value.residues == tuple(sorted(set(value.residues)))
        and all(type(r) is int and 0 <= r < value.period for r in value.residues),
        "invalid scope activation residues",
    )
    require(
        all(
            isinstance(p, Position) and type(p.end) is bool and type(p.offset) is int
            for p in (value.first, value.last)
        ),
        "invalid scope activation position",
    )
    first, last = affine(value.first), affine(value.last)
    active = meet(domain, le(first, last))
    require(active is not None, "scope activation is empty")
    require(
        holds((0, 0), first, active)
        and holds(last, (1, -1), active)
        and holds((0, 0), add(first, distance), active)
        and holds(add(last, distance), (1, -1), active),
        "scope activation escapes a source or destination parent domain",
    )
    return value


def shifted(value: Activation, delta: int) -> Activation:
    """Express the same active instances in another endpoint's epoch."""
    return replace(
        value,
        first=position(add(affine(value.first), delta)),
        last=position(add(affine(value.last), delta)),
        residues=tuple(sorted((r + delta) % value.period for r in value.residues)),
    )


def legacy_activation(distance: int, period=1, residues=(0,)) -> Activation:
    require(distance >= 0, "legacy scope distance cannot be negative")
    return Activation(last=Position(True, distance), period=period, residues=residues)


def encode_activation(value: Activation):
    """Named IR fields preserve the complete predicate for final reconstruction."""
    from xdsl.dialects.builtin import DictionaryAttr, IntAttr

    fields = {
        "count_period": value.count.period,
        "count_residue": value.count.residue,
        "first_end": int(value.first.end),
        "first_offset": value.first.offset,
        "last_end": int(value.last.end),
        "last_offset": value.last.offset,
    }
    for key in ("exact", "minimum", "maximum"):
        if (number := getattr(value.count, key)) is not None:
            fields["count_" + key] = number
    return DictionaryAttr({key: IntAttr(number) for key, number in fields.items()})


def decode_activation(attr, *, distance: int, period: int, residues: tuple[int, ...]):
    from xdsl.dialects.builtin import DictionaryAttr, IntAttr

    require(isinstance(attr, DictionaryAttr), "invalid scope activation attribute")
    fields = attr.data
    required = {
        "count_period",
        "count_residue",
        "first_end",
        "first_offset",
        "last_end",
        "last_offset",
    }
    optional = {"count_exact", "count_minimum", "count_maximum"}
    require(
        required <= fields.keys() <= required | optional
        and all(isinstance(v, IntAttr) for v in fields.values()),
        "invalid scope activation fields",
    )
    data = {key: value.data for key, value in fields.items()}
    require(
        data["first_end"] in (0, 1) and data["last_end"] in (0, 1), "invalid scope boundary origin"
    )
    return validate_activation(
        Activation(
            CountDomain(
                data["count_period"],
                data["count_residue"],
                **{key: data.get("count_" + key) for key in ("exact", "minimum", "maximum")},
            ),
            Position(bool(data["first_end"]), data["first_offset"]),
            Position(bool(data["last_end"]), data["last_offset"]),
            period,
            residues,
        ),
        distance=distance,
    )


def affine(position: Position) -> tuple[int, int]:
    return (1, -1 - position.offset) if position.end else (0, position.offset)


def position(point: tuple[int, int]) -> Position:
    a, b = point
    if a not in (0, 1):
        raise ValueError("a scope boundary needs a constant or N-relative position")
    return Position(bool(a), -1 - b if a else b)


@dataclass(frozen=True)
class ScopedDependence:
    source: tuple[str, str]
    destination: tuple[str, str]
    distance: int
    activation: Activation
    hazard: str
    storage: str
    start: int
    stop: int
    prefix: bool = False

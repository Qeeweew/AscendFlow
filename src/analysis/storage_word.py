"""Symbolic adjacency of closed storage packets on one pipeline clock.

Packet existence is 0 <= g < count, and its controller position is
``(g + offset, lexical)``. A packet's internal value/read/retirement contract
belongs to its caller. This module establishes only the neighboring packets
of that fixed word; it supplies no RAW, completion, or ownership facts.

PLE boundaries are bounded by the static offset span. Runtime iterations and
native bodies are never expanded. Short counts partition that static boundary;
long counts share one steady word with exact entry and exit connections.
"""

from dataclasses import dataclass
from itertools import pairwise

from ..diagnostics import ErrorKind, require
from ..indexing import arguments
from ..scalar import Expr, expression, literal
from .structured_schedule import Comparison

MAX_SPAN = 64
MAX_PACKETS = 64
MAX_WORK = 65536
MAX_ADJACENCIES = 16384


def _require(condition, message, *, kind=ErrorKind.INVALID):
    require(condition, message, kind=kind, phase="storage-word")


@dataclass(frozen=True, slots=True)
class StoragePacketPosition:
    key: int
    offset: int
    lexical: int


@dataclass(frozen=True, slots=True)
class PacketPoint:
    key: int
    ordinal: Expr


@dataclass(frozen=True, slots=True)
class PacketAdjacency:
    """A static neighboring pair, or a family in consumer ordinal j.

    ``limit=None`` means one pair under ``guards``. Otherwise the consumer is
    the supplied canonical ordinal j in ``[first, limit)`` and the source is
    j-distance, where distance may be negative. Guards precede evaluation of
    endpoints or limits, including count-dependent boundary expressions.
    """

    source: PacketPoint
    consumer: PacketPoint
    first: int = 0
    limit: Expr | None = None
    guards: tuple[Comparison, ...] = ()


@dataclass(frozen=True, slots=True)
class GuardedPacketPoint:
    point: PacketPoint
    guards: tuple[Comparison, ...] = ()


@dataclass(frozen=True, slots=True)
class StorageWord:
    adjacencies: tuple[PacketAdjacency, ...]
    first: tuple[GuardedPacketPoint, ...]
    last: tuple[GuardedPacketPoint, ...]


def build_storage_word(positions, count: Expr, ordinal: Expr) -> StorageWord:
    """Establish exact packet adjacency, conditional on a nonnegative count.

    Lexical positions identify distinct controller anchors and must be unique.
    The supplied count is fixed throughout this loop invocation; outer clocks
    may remain in it. Every output contains only mathematical scalar Exprs.
    """
    positions = tuple(positions)
    _require(
        len(positions) <= MAX_PACKETS,
        f"storage packet count exceeds analysis budget {MAX_PACKETS}",
        kind=ErrorKind.ANALYSIS_LIMIT,
    )
    _require(
        all(
            isinstance(position, StoragePacketPosition)
            and all(
                type(value) is int for value in (position.key, position.offset, position.lexical)
            )
            for position in positions
        ),
        "storage packet positions need integer keys, offsets, and lexical anchors",
    )
    _require(
        len({position.key for position in positions}) == len(positions),
        "storage packet keys must be distinct",
    )
    _require(
        len({position.lexical for position in positions}) == len(positions),
        "storage packet lexical anchors are ambiguous",
    )
    _require(
        isinstance(count, Expr) and isinstance(ordinal, Expr) and ordinal.kind == "argument",
        "storage word needs a scalar count and canonical ordinal argument",
    )
    _require(
        ordinal.args[0] not in arguments(count), "storage packet count varies over its own clock"
    )
    _require(
        count.constant is None or count.constant >= 0, "storage word count must be nonnegative"
    )
    if not positions or count.constant == 0:
        return StorageWord((), (), ())
    minimum = min(position.offset for position in positions)
    positions = tuple(
        sorted(
            (
                StoragePacketPosition(position.key, position.offset - minimum, position.lexical)
                for position in positions
            ),
            key=lambda position: position.lexical,
        )
    )
    span = max(position.offset for position in positions)
    _require(
        span <= MAX_SPAN,
        f"storage packet offset span exceeds analysis budget {MAX_SPAN}",
        kind=ErrorKind.ANALYSIS_LIMIT,
    )
    adjacencies, work = [], 0

    def charge(amount=1):
        nonlocal work
        work += amount
        _require(
            work <= MAX_WORK,
            f"storage word cumulative work exceeds analysis budget {MAX_WORK}",
            kind=ErrorKind.ANALYSIS_LIMIT,
        )

    def guards(predicate, bound):
        test = Comparison(count, predicate, literal(bound))
        if count.constant is None:
            return (test,)
        return () if test.evaluate({}) else None

    def point(position, value):
        charge()
        return PacketPoint(position.key, value)

    def append(source, consumer, condition, first=0, limit=None):
        charge()
        if limit is not None and limit.constant is not None and limit.constant <= first:
            return
        _require(
            len(adjacencies) < MAX_ADJACENCIES,
            f"storage word adjacency count exceeds analysis budget {MAX_ADJACENCIES}",
            kind=ErrorKind.ANALYSIS_LIMIT,
        )
        adjacencies.append(PacketAdjacency(source, consumer, first, limit, condition))

    def static_word(steps, condition):
        first, previous = None, None
        for points in steps:
            for current in points:
                if previous is not None:
                    append(previous, current, condition)
                first = current if first is None else first
                previous = current
        return first, previous

    def short_steps(size):
        for step in range(size + span):
            charge(len(positions))
            yield tuple(
                point(position, literal(step - position.offset))
                for position in positions
                if 0 <= step - position.offset < size
            )

    # These are statically many count cases, never a runtime-count loop.
    for size in range(1, span + 1):
        if (condition := guards("==", size)) is not None:
            static_word(short_steps(size), condition)

    if (condition := guards(">", span)) is not None:

        def prologue():
            for step in range(span):
                charge(len(positions))
                yield tuple(
                    point(position, literal(step - position.offset))
                    for position in positions
                    if position.offset <= step
                )

        def epilogue():
            for step in range(span):
                charge(len(positions))
                yield tuple(
                    point(position, expression("add", count, literal(step - position.offset)))
                    for position in positions
                    if step < position.offset
                )

        _, prefix_last = static_word(prologue(), condition)
        suffix_first, _ = static_word(epilogue(), condition)
        first_position, last_position = positions[0], positions[-1]
        if prefix_last is not None:
            append(
                prefix_last,
                point(first_position, literal(span - first_position.offset)),
                condition,
            )
        if suffix_first is not None:
            append(
                point(
                    last_position,
                    expression("add", count, literal(-1 - last_position.offset)),
                ),
                suffix_first,
                condition,
            )
        # Within every complete steady step S <= step < count.
        for source, consumer in pairwise(positions):
            distance = source.offset - consumer.offset
            append(
                point(source, expression("sub", ordinal, literal(distance))),
                point(consumer, ordinal),
                condition,
                span - consumer.offset,
                expression("sub", count, literal(consumer.offset)),
            )
        # Between complete steady steps S+1 <= step < count.
        distance = 1 + last_position.offset - first_position.offset
        append(
            point(last_position, expression("sub", ordinal, literal(distance))),
            point(first_position, ordinal),
            condition,
            span + 1 - first_position.offset,
            expression("sub", count, literal(first_position.offset)),
        )

    nonempty = guards(">", 0)
    assert nonempty is not None
    first_position = min(positions, key=lambda position: (position.offset, position.lexical))
    last_position = max(positions, key=lambda position: (position.offset, position.lexical))
    return StorageWord(
        tuple(adjacencies),
        (GuardedPacketPoint(point(first_position, literal(0)), nonempty),),
        (GuardedPacketPoint(point(last_position, expression("sub", count, literal(1))), nonempty),),
    )

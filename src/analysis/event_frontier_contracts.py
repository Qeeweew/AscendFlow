"""Frozen local-event interval certificates and numeric reuse lookup.

These records contain real endpoints, enclosure links and established
consumption handbacks. Numeric group checks and compressed reporting perform
no source, domain, handback or solver queries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, eq=False, slots=True)
class EventFrontier:
    head: object
    tail: object
    members: frozenset
    children: tuple[EventFrontier, ...] = ()
    # Actual unconditional issue enclosures of the child extremal endpoints.
    enclosure: tuple = ()


@dataclass(frozen=True, slots=True)
class FrontierSeparation:
    left: EventFrontier
    right: EventFrontier
    forward: bool | None
    handbacks: tuple
    original: tuple | None = None


@dataclass(frozen=True, eq=False, slots=True)
class CreditSeparation:
    """A finite interface covering every member of both complete credits."""

    interfaces: tuple[FrontierSeparation, ...]
    left: int
    right: int

    def __post_init__(self):
        assert self.left and self.right and not self.left & self.right, (
            "credit axes must be disjoint"
        )


@dataclass(frozen=True, slots=True)
class SelectedSeparation:
    """A selected subrectangle of an already established complete-credit fact."""

    left: int
    right: int
    certificate: CreditSeparation


@dataclass(frozen=True, slots=True)
class LocalEventReuse:
    bank: tuple
    event: int
    credits: tuple
    rectangles: tuple[SelectedSeparation, ...]


@dataclass(frozen=True, init=False, slots=True)
class FrozenEventRelations(Mapping):
    """Pure finite reuse lookup over explicit counters and local rectangles."""

    _counters: Mapping
    _rows: tuple
    _masks: tuple[int, ...]

    def __init__(self, counters, rows):
        rows = tuple(tuple(row) for row in rows)
        for number, row in enumerate(rows):
            bit = 1 << number
            for mask, certificate in row:
                allowed = (
                    certificate.right
                    if bit & certificate.left
                    else certificate.left
                    if bit & certificate.right
                    else 0
                )
                assert mask and not mask & ~allowed, "reuse row escapes its certified rectangle"
        object.__setattr__(self, "_counters", MappingProxyType(dict(counters)))
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_masks", tuple(self._union(row) for row in self._rows))

    @staticmethod
    def _union(row):
        result = 0
        for mask, _ in row:
            result |= mask
        return result

    def __getitem__(self, key):
        if key in self._counters:
            return self._counters[key]
        a, b = key
        if not (0 <= a < len(self._rows) and 0 <= b < len(self._rows)) or not self._masks[a] & (
            1 << b
        ):
            raise KeyError(key)
        for mask, certificate in self._rows[a]:
            if mask & (1 << b):
                return (certificate,)
        raise AssertionError("frozen frontier mask has no covering certificate")

    def local_mask(self, number):
        """Numeric permissions for one original local row, without pair expansion."""
        return self._masks[number] if 0 <= number < len(self._masks) else 0

    def cover_group(self, members):
        """Check all selected pairs by row masks, retaining compressed facts.

        This consumes the same frozen permissions as individual lookup. A
        missing permission in either selected row rejects the whole group.
        No message-pair projection, domain query or handback is performed.
        """
        mask = sum(1 << number for number in members)
        for number in members:
            if not 0 <= number < len(self._rows) or (mask ^ (1 << number)) & ~self._masks[number]:
                return None
        certificates = {}
        for number in sorted(members):
            for others, certificate in self._rows[number]:
                if others & mask:
                    certificates.setdefault(certificate, None)
        return tuple(
            SelectedSeparation(c.left & mask, c.right & mask, c)
            for c in certificates
            if c.left & mask and c.right & mask
        )

    def __iter__(self):
        yield from self._counters
        for a, mask in enumerate(self._masks):
            mask &= ~((1 << (a + 1)) - 1)
            while mask:
                low = mask & -mask
                yield a, low.bit_length() - 1
                mask ^= low

    def __len__(self):
        return len(self._counters) + sum(
            (mask >> (a + 1)).bit_count() for a, mask in enumerate(self._masks)
        )

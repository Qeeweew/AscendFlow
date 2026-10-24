"""Frozen event-consumption interval certificates and numeric reuse lookup.

These records contain real endpoints, enclosure links and established
consumption handbacks. Numeric group checks and compressed reporting perform
no source, domain, handback or solver queries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, eq=False, slots=True)
class EventFrontier:
    head: object
    tail: object
    size: int = 1
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
    """Pure finite lookup of established complete-credit consumption rectangles."""

    _rows: tuple
    _masks: tuple[int, ...]

    def __init__(self, rows):
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
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_masks", tuple(self._union(row) for row in self._rows))

    @staticmethod
    def _union(row):
        result = 0
        for mask, _ in row:
            result |= mask
        return result

    def __getitem__(self, key):
        a, b = key
        if not (0 <= a < len(self._rows) and 0 <= b < len(self._rows)) or not self._masks[a] & (
            1 << b
        ):
            raise KeyError(key)
        for mask, certificate in self._rows[a]:
            if mask & (1 << b):
                return (certificate,)
        raise AssertionError("frozen frontier mask has no covering certificate")

    def row_mask(self, number):
        """Numeric permissions for one original row, without pair expansion."""
        return self._masks[number] if 0 <= number < len(self._masks) else 0

    def permits_group(self, members):
        """Check a complete temporal group without projecting its pairs."""
        members = tuple(members)
        if len(members) < 2:
            return True
        mask = sum(1 << number for number in members)
        return all(
            0 <= number < len(self._rows) and not (mask ^ (1 << number)) & ~self._masks[number]
            for number in members
        )

    def cover_group(self, members):
        return self.cover_groups(tuple(frozenset((number,)) for number in members))

    def cover_groups(self, groups):
        """Check cross-group rows and retain selected certificate rectangles.

        A FIFO child owns its internal word; temporal permissions must cover
        every complete credit in all other children. Singleton children give
        ordinary complete-credit time reuse. Missing either row rejects it.
        """
        masks = tuple(sum(1 << number for number in group) for group in groups)
        mask = sum(masks)
        assert mask == self._union((value, None) for value in masks), "reuse groups overlap"
        certificates = {}
        for group, own in zip(groups, masks, strict=True):
            other = mask ^ own
            if not other:
                continue
            for number in sorted(group):
                if not 0 <= number < len(self._rows) or other & ~self._masks[number]:
                    return None
                for others, certificate in self._rows[number]:
                    if others & other:
                        certificates.setdefault(certificate, None)
        return tuple(
            SelectedSeparation(c.left & mask, c.right & mask, c)
            for c in certificates
            if c.left & mask and c.right & mask
        )

    def __bool__(self):
        return any(self._masks)

    def __iter__(self):
        for a, mask in enumerate(self._masks):
            mask &= ~((1 << (a + 1)) - 1)
            while mask:
                low = mask & -mask
                yield a, low.bit_length() - 1
                mask ^= low

    def __len__(self):
        return sum((mask >> (a + 1)).bit_count() for a, mask in enumerate(self._masks))

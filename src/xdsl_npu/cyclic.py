"""Cyclic ownership of a uniform flat task domain, without expanding its total.

Owner c issues local generation g for flat index c + groups*g. The local
count and the allocation/physical-view relation are checked independently:
an owner-local bound is never evidence that its addresses stay in the total.
"""

from dataclasses import dataclass

from .diagnostics import require
from .indexing import equivalent
from .scalar import Expr, expression, literal

CORE = Expr("argument", ("$core",))


@dataclass(frozen=True)
class CyclicDomain:
    total: Expr
    groups: int

    @property
    def local_bound(self):
        remaining = expression("sub", self.total, CORE)
        return expression("ceildiv", expression("max", literal(0), remaining), literal(self.groups))

    @property
    def host_bound(self):
        return expression("ceildiv", self.total, literal(self.groups))


def check_stream(domain, storage, refs, facts):
    """Prove complete tile ownership, including every read-only GM view.

    Each allocation has one static tile scale shared by all its aliases.
    A view may name exact row/subrange fragments inside that tile. Different
    owners/generations have distinct flat indices, hence disjoint tile frames.
    All enumeration below is over static views, rows and bounded owner IDs.
    """
    strides = {sum(s for _, s, m, _ in ref.terms if not m) for ref in refs}
    require(
        len(strides) == 1 and next(iter(strides)) > 0,
        f"{storage.name}: cyclic GM aliases need one positive stream stride",
    )
    stride = next(iter(strides))
    require(stride % domain.groups == 0, f"{storage.name}: cyclic stride must divide by groups")
    tile = stride // domain.groups
    expected = expression("mul", domain.total, literal(tile))
    actual = (
        expression("mul", storage.extent, literal(storage.count_scale))
        if storage.count_scale
        else literal(storage.nbytes)
    )
    require(
        equivalent(actual, expected, facts),
        f"{storage.name}: cyclic allocation must equal total times tile bytes",
    )
    for ref in refs:
        require(
            not ref.count_offset and not ref.count_core_stride and ref.prefix_extent is None,
            f"{storage.name}: cyclic views cannot change count ownership or clip tiles",
        )
        require(
            all(not s or not m and d == 1 for _, s, m, d in ref.terms),
            f"{storage.name}: cyclic GM access must retain its static affine task index",
        )
        require(
            all(
                ref.address(0, core) - ref.address(0) == core * tile
                for core in range(domain.groups)
            ),
            f"{storage.name}: cyclic owner translation must equal core times tile bytes",
        )
        require(
            all(0 <= lo < hi <= tile for lo, hi in ref.segments(0)),
            f"{storage.name}: cyclic access exceeds its task tile",
        )
    return stride

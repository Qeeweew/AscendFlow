"""Canonical L1 NZ allocation geometry, independent of access generations."""

from dataclasses import dataclass
from math import prod

from ..diagnostics import require


@dataclass(frozen=True, slots=True)
class NZLayout:
    shape: tuple[int, ...]
    itemsize: int

    def __post_init__(self):
        require(
            len(self.shape) >= 2
            and all(type(n) is int and n > 0 for n in self.shape)
            and self.itemsize in {2, 4}
            and self.shape[-2] % 16 == 0
            and self.shape[-1] % self.c0 == 0,
            "L1 NZ allocation needs static planes, row16 and C0-aligned columns",
        )

    @property
    def c0(self):
        return 32 // self.itemsize

    @property
    def nbytes(self):
        return prod(self.shape) * self.itemsize

    @property
    def strides(self):
        """Byte coefficients for C0-aligned source-view column origins.

        A subview retains the allocation's M, including runtime row prefixes.
        Within-C0 lane selection is not a supported native matrix subview.
        """
        return (
            *(prod(self.shape[axis + 1 :]) * self.itemsize for axis in range(len(self.shape) - 2)),
            32,
            self.shape[-2] * self.itemsize,
        )

    def geometry(self, rows, columns, *, prefix=False):
        if rows != self.shape[-2] or prefix:
            return rows * 32, columns // self.c0, self.shape[-2] * 32
        return rows * columns * self.itemsize, 1, 0

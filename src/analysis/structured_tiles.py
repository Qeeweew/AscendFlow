"""Exact aligned ND rectangles and their injective tile-grid coordinates.

These geometry identities are established once by address decoding. They
provide no bounds, ownership, value-version or completion guarantee themselves.
"""

from dataclasses import dataclass

from ..indexing import _ratio, equivalent, normalize, polynomial
from ..scalar import Expr, expression, infer, literal
from ..target.native import ITEMSIZE


@dataclass(frozen=True, slots=True)
class NDGridCell:
    extents: tuple[int, int]
    widths: tuple[int, int]
    index: Expr

    @property
    def grid(self):
        return self.extents, self.widths


def nd_grid_cell(offset, geometry, axes, facts):
    if (
        geometry.bank != "GM"
        or geometry.layout != "linear"
        or axes is None
        or len(axes.origins) != 2
        or axes.axes != (0, 1)
        or axes.displacement != 0
    ):
        return None
    extents = tuple(normalize(value, facts).constant for value in axes.extents)
    widths = axes.widths
    if any(
        width <= 0 or size is None or size <= 0 or size % width
        for size, width in zip(extents, widths, strict=True)
    ):
        return None
    _, columns = extents
    itemsize = ITEMSIZE[geometry.dtype]
    if (
        geometry.rows != widths[0]
        or geometry.nbytes != widths[1] * itemsize
        or geometry.row_stride != columns * itemsize
    ):
        return None
    row, column = axes.origins
    actual = expression(
        "mul",
        expression("add", expression("mul", row, literal(columns)), column),
        literal(itemsize),
    )
    if not equivalent(offset, actual, facts):
        return None
    for origin, width in zip(axes.origins, widths, strict=True):
        for atoms, coefficient in polynomial(origin, facts).items():
            multiple = _ratio(coefficient, width)
            for atom in atoms:
                fact = infer(atom, facts)
                multiple *= 1 if fact is None else fact.multiple
            if multiple.denominator != 1:
                return None
    index = normalize(
        expression(
            "add",
            expression(
                "mul",
                expression("floordiv", row, literal(widths[0])),
                literal(columns // widths[1]),
            ),
            expression("floordiv", column, literal(widths[1])),
        ),
        facts,
    )
    return NDGridCell(extents, widths, index)

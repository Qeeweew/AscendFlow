"""Digit permutations retain injective ND grid identities, not owner trust."""

from itertools import product

import pytest

from ascendflow.analysis.structured_tiles import NDRectangle, nd_grid_cell
from ascendflow.scalar import Expr, Fact, expression, literal


def rectangle(row_count=8, low_max=3):
    rh, rl, ch, cl = (Expr("argument", (name,)) for name in ("rh", "rl", "ch", "cl"))
    row = expression("add", expression("mul", rh, literal(4)), rl)
    col = expression("add", expression("mul", ch, literal(2)), cl)
    facts = {
        "rh": Fact(maximum=1),
        "rl": Fact(maximum=low_max),
        "ch": Fact(maximum=2),
        "cl": Fact(maximum=1),
    }
    return NDRectangle((row_count, 6), (1, 1), (row, col)), facts


@pytest.mark.parametrize("column_first", [False, True])
def test_grid_digit_orders_are_complete_bijections(column_first):
    rect, facts = rectangle()
    cell = nd_grid_cell(rect, facts)
    index = cell.index((4, 2, column_first))
    assert index is not None
    values = [
        index.evaluate({"rh": row // 4, "rl": row % 4, "ch": col // 2, "cl": col % 2})
        for row, col in product(range(8), range(6))
    ]
    assert sorted(values) == list(range(48))


@pytest.mark.parametrize("row_count,low_max", [(8, 4), (10, 3)])
def test_grid_digits_need_bounded_low_digits_and_complete_radices(row_count, low_max):
    rect, facts = rectangle(row_count, low_max)
    cell = nd_grid_cell(rect, facts)
    assert cell.index((4, 2, False)) is None

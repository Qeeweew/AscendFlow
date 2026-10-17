"""Exact aligned ND rectangles and their injective tile-grid coordinates.

These geometry identities are established once by address decoding. They
provide no bounds, ownership, value-version or completion guarantee themselves.
"""

from dataclasses import dataclass
from itertools import product

from ..indexing import _ratio, equivalent, normalize, polynomial
from ..scalar import Expr, expression, infer, literal
from ..target.native import ITEMSIZE


@dataclass(frozen=True, slots=True)
class GridLinearization:
    # Split each axis into high/low digits, then choose the major axis.
    order: tuple[int, int, bool]
    index: Expr


@dataclass(frozen=True, slots=True)
class NDGridCell:
    extents: tuple[int, int]
    widths: tuple[int, int]
    linearizations: tuple[GridLinearization, ...]

    @property
    def grid(self):
        return self.extents, self.widths

    def index(self, order):
        return next((item.index for item in self.linearizations if item.order == order), None)


@dataclass(frozen=True, slots=True)
class NDRectangle:
    extents: tuple[int, int]
    widths: tuple[int, int]
    origins: tuple[Expr, Expr]


def nd_rectangle(offset, geometry, axes, facts):
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
        width <= 0 or size is None or size < width
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
    return NDRectangle(extents, widths, axes.origins)


def _rebuild(terms):
    result = literal(0)
    for atoms, coefficient in terms.items():
        if coefficient.denominator != 1:
            return None
        value = literal(int(coefficient))
        for atom in atoms:
            value = expression("mul", value, atom)
        result = expression("add", result, value)
    return result


def _digit_range(value, facts):
    bound = infer(value, facts)
    if bound is not None and bound.maximum is not None:
        return bound.minimum, bound.maximum
    if value.kind == "rem" and (divisor := value.args[1].constant) is not None and divisor > 0:
        # A Euclidean low digit has this range whenever the original
        # division is defined. This identity does not exempt its raw DAG
        # from the later ordered unsigned-definedness and axis checks.
        return 0, divisor - 1
    if value.kind in {"add", "mul"}:
        left, right = (_digit_range(child, facts) for child in value.args)
        if left is not None and right is not None:
            operation = (lambda a, b: a + b) if value.kind == "add" else (lambda a, b: a * b)
            return operation(left[0], right[0]), operation(left[1], right[1])
    return None


def _axis_digits(origin, width, count, facts):
    # Keep raw div/rem atoms while splitting integral ring coefficients.
    # Expanding x%r first would lose its independent [0,r) range.
    terms = {
        atoms: _ratio(coefficient, width)
        for atoms, coefficient in polynomial(origin, facts, retained=True).items()
    }
    value = _rebuild(terms)
    if value is None:
        value = normalize(expression("floordiv", origin, literal(width)), facts)
        terms = polynomial(value, facts, retained=True)
    factors = {1}
    for atoms, coefficient in terms.items():
        if atoms and coefficient.denominator == 1:
            factor = abs(int(coefficient))
            if factor > 1 and count % factor == 0:
                factors.add(factor)
        for atom in atoms:
            if atom.kind == "rem":
                factor = atom.args[1].constant
                if factor is not None and factor > 1 and count % factor == 0:
                    factors.add(factor)
    digits = {1: (value, literal(0))}
    for factor in sorted(factors - {1}):
        high, low = {}, {}
        for atoms, coefficient in terms.items():
            quotient = _ratio(coefficient, factor)
            (high if quotient.denominator == 1 else low)[atoms] = (
                quotient if quotient.denominator == 1 else coefficient
            )
        major, minor = _rebuild(high), _rebuild(low)
        if major is None or minor is None:
            continue
        bound = _digit_range(minor, facts)
        if bound is not None and 0 <= bound[0] <= bound[1] < factor:
            # value == factor*major + minor, 0<=minor<factor. Hence these
            # are its unique Euclidean digits. Original axis bounds are
            # still established separately by memory admission.
            digits[factor] = major, minor
    return digits


def nd_grid_cell(rectangle, facts):
    if rectangle is None:
        return None
    extents, widths = rectangle.extents, rectangle.widths
    if any(size % width for size, width in zip(extents, widths, strict=True)):
        return None
    for origin, width in zip(rectangle.origins, widths, strict=True):
        for atoms, coefficient in polynomial(origin, facts).items():
            multiple = _ratio(coefficient, width)
            for atom in atoms:
                fact = infer(atom, facts)
                multiple *= 1 if fact is None else fact.multiple
            if multiple.denominator != 1:
                return None
    counts = tuple(size // width for size, width in zip(extents, widths, strict=True))
    rows, columns = (
        _axis_digits(origin, width, count, facts)
        for origin, width, count in zip(rectangle.origins, widths, counts, strict=True)
    )
    indices = []
    for (r, (rh, rl)), (c, (ch, cl)), column_first in product(
        rows.items(), columns.items(), (False, True)
    ):
        major, minor = ((rh, ch), (rl, cl)) if not column_first else ((ch, rh), (cl, rl))
        radices = (counts[1] // c, r, c) if not column_first else (counts[0] // r, c, r)
        index = major[0]
        for digit, radix in zip((major[1], *minor), radices, strict=True):
            index = expression("add", expression("mul", index, literal(radix)), digit)
        # Every order is a bijection of the same bounded grid: just a
        # permutation of Euclidean digits, never an owner assertion.
        indices.append(GridLinearization((r, c, column_first), normalize(index, facts)))
    return NDGridCell(extents, widths, tuple(indices))

"""Exact aligned ND rectangles and their injective tile-grid coordinates.

These geometry identities are established once by address decoding. They
provide no bounds, ownership, value-version or completion guarantee themselves.
"""

from dataclasses import dataclass
from itertools import permutations, product
from math import prod

from ..diagnostics import CompileError
from ..indexing import _ratio, equivalent, interval, nonnegative, normalize, polynomial
from ..scalar import Expr, expression, infer, literal
from ..target.native import ITEMSIZE


@dataclass(frozen=True, slots=True)
class GridLinearization:
    # Split each axis into high/low digits, then choose the major axis.
    order: tuple
    index: Expr
    ranges: tuple[tuple[Expr, int | Expr], ...] = ()


@dataclass(frozen=True, slots=True)
class NDGridCell:
    extents: tuple[int | Expr, ...]
    widths: tuple[int, ...]
    linearizations: tuple[GridLinearization, ...]

    @property
    def grid(self):
        return self.extents, self.widths

    def index(self, order):
        return next((item.index for item in self.linearizations if item.order == order), None)

    def bounded_index(self, order, domains, facts):
        item = next((item for item in self.linearizations if item.order == order), None)
        if item is None:
            return None
        for value, limit in item.ranges:
            limit = literal(limit) if type(limit) is int else limit
            low, high = interval(value, domains, facts)
            if not nonnegative(low, facts) or not nonnegative(
                expression("sub", limit, expression("add", high, literal(1))), facts
            ):
                return None
        return item.index


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


def axis_grid_cell(offset, geometry, axes, facts):
    """Bijections of complete aligned original-axis tiles, including runtime axes.

    Permuting Euclidean digits changes only the proof coordinate. Physical
    strides, native footprints and actual original-axis bounds remain intact.
    At most four nontrivial digits are admitted; no runtime tiles are expanded.
    """
    if (
        axes is None
        or geometry.bank != "GM"
        or geometry.layout != "linear"
        or axes.displacement
        or len(axes.origins) > 4
        or axes.axes
        not in {(len(axes.origins) - 1,), (len(axes.origins) - 2, len(axes.origins) - 1)}
        or any(w != 1 for i, w in enumerate(axes.widths) if i not in axes.axes)
    ):
        return None
    itemsize = ITEMSIZE[geometry.dtype]
    if prod(axes.widths) * itemsize != geometry.total_nbytes:
        return None
    logical_rows = axes.widths[-2] if len(axes.axes) == 2 else 1
    if (
        geometry.rows == 1
        and logical_rows > 1
        and not equivalent(axes.extents[-1], literal(axes.widths[-1]), facts)
    ):
        return None
    if not geometry.contiguous and not (
        geometry.rows == axes.widths[-2]
        and geometry.nbytes == axes.widths[-1] * itemsize
        and equivalent(
            literal(geometry.row_stride),
            expression("mul", axes.extents[-1], literal(itemsize)),
            facts,
        )
    ):
        return None
    stride, expected = literal(itemsize), literal(0)
    for origin, extent in zip(reversed(axes.origins), reversed(axes.extents), strict=True):
        expected = expression("add", expected, expression("mul", origin, stride))
        stride = expression("mul", stride, extent)
    if not equivalent(expected, offset, facts):
        return None
    choices = []
    for axis, (origin, width, extent) in enumerate(
        zip(axes.origins, axes.widths, axes.extents, strict=True)
    ):
        bound = infer(extent, facts)
        if bound is None or bound.minimum < width or bound.multiple % width:
            return None
        terms = {
            atoms: _ratio(coefficient, width)
            for atoms, coefficient in polynomial(origin, facts, retained=True).items()
        }
        index = _rebuild(terms)
        if index is None:
            return None
        count = expression("floordiv", extent, literal(width))
        count_bound = infer(count, facts)
        if count_bound is not None and count_bound.maximum == 1:
            choices.append(((1, () if index.constant == 0 else (((axis, 0), index, count),), ()),))
            continue
        factors = {1}
        for atoms, coefficient in terms.items():
            if atoms and coefficient.denominator == 1:
                factor = abs(int(coefficient))
                if factor > 1 and count_bound is not None and count_bound.multiple % factor == 0:
                    factors.add(factor)
        variants = []
        for factor in sorted(factors):
            high, low = {}, {}
            for atoms, coefficient in terms.items():
                quotient = _ratio(coefficient, factor)
                (high if quotient.denominator == 1 else low)[atoms] = (
                    quotient if quotient.denominator == 1 else coefficient
                )
            major, minor = _rebuild(high), _rebuild(low)
            digit_bound = _digit_range(minor, facts)
            if digit_bound is not None and not 0 <= digit_bound[0] <= digit_bound[1] < factor:
                continue
            digits = [((axis, 0), major, expression("floordiv", count, literal(factor)))]
            if factor > 1:
                digits.append(((axis, 1), minor, literal(factor)))
            variants.append(
                (factor, tuple(digits), () if digit_bound is not None else ((minor, factor),))
            )
        choices.append(tuple(variants))
    indices = []
    for selected in product(*choices):
        factors = tuple(factor for factor, _, _ in selected)
        digits = tuple(digit for _, group, _ in selected for digit in group)
        ranges = tuple(condition for _, _, conditions in selected for condition in conditions)
        if len(digits) > 4:
            continue
        for ordered in permutations(digits):
            index = literal(0)
            for _, value, radix in ordered:
                index = expression("add", expression("mul", index, radix), value)
            # Retain quotient atoms while cancelling exact arithmetic. Unlike
            # address reconstruction this coordinate may have runtime radices.
            index = _rebuild(polynomial(index, facts, retained=True))
            if index is not None:
                raw_index = index
                try:
                    index = normalize(index, facts)
                except CompileError as error:
                    if error.message != "runtime address retains a fractional coefficient":
                        raise
                indices.append(
                    GridLinearization((factors, tuple(d[0] for d in ordered)), index, ranges)
                )
                for key, unshifted, premise in _unshifted_indices(ordered, raw_index, facts):
                    indices.append(
                        GridLinearization(
                            (factors, tuple(d[0] for d in ordered), key),
                            unshifted,
                            (*ranges, premise),
                        )
                    )
    return NDGridCell(axes.extents, axes.widths, tuple(indices)) if indices else None


def _unshifted_indices(digits, index, facts):
    """Invert a cyclic translation of a complete mixed-radix prefix.

    If a prefix of radix N is (x+c)%N, with uniform c and 0<=x<N,
    replacing it by x is another bijection of that same bounded grid. The
    consumer must establish x's actual active range; the inverse is never
    emitted as runtime arithmetic or trusted from a user marker.
    """
    pending, atoms = [index], set()
    while pending:
        value = pending.pop()
        if value in atoms:
            continue
        atoms.add(value)
        pending.extend(arg for arg in value.args if isinstance(arg, Expr))
    candidates = []
    for atom in sorted(atoms, key=lambda value: value.order_key):
        if atom.kind != "rem" or any(
            name not in facts or name.startswith("$") for name in atom.args[1].variables
        ):
            continue
        uniform, varying = {}, {}
        for term, coefficient in polynomial(atom.args[0], facts, retained=True).items():
            (
                uniform
                if all(
                    name in facts and not name.startswith("$")
                    for factor in term
                    for name in factor.variables
                )
                else varying
            )[term] = coefficient
        shift, coordinate = _rebuild(uniform), _rebuild(varying)
        if (
            shift is None
            or coordinate is None
            or shift.constant == 0
            or coordinate.constant is not None
        ):
            continue
        candidates.append((atom, shift, coordinate))
    prefix, radix = literal(0), literal(1)
    for split, (_, digit, count) in enumerate(digits, 1):
        prefix = expression("add", expression("mul", prefix, count), digit)
        radix = expression("mul", radix, count)
        suffix = literal(1)
        for _, _, count in digits[split:]:
            suffix = expression("mul", suffix, count)
        for atom, shift, coordinate in candidates:
            if not equivalent(radix, atom.args[1], facts) or not equivalent(prefix, atom, facts):
                continue
            transformed = expression(
                "add", index, expression("mul", suffix, expression("sub", coordinate, prefix))
            )
            try:
                transformed = normalize(transformed, facts)
            except CompileError as error:
                if error.message != "runtime address retains a fractional coefficient":
                    raise
                continue
            yield (
                ("cyclic-prefix", split, shift, atom.args[1]),
                transformed,
                (coordinate, atom.args[1]),
            )

"""Structural periods and mathematical phases for finite scalar analysis.

A period is a proof of equality after a constant ordinal shift, not a proof
that unsigned source arithmetic is defined. Source guards retain that separate
ordered obligation. Phase evaluation therefore uses mathematical integers,
including negative representatives outside the source's valid interval.
"""

from collections.abc import Callable
from math import gcd, lcm

from ..scalar import Expr
from .structured_schedule import Comparison


def period_of(value: Expr, ordinal: Expr, invariant: Callable[[Expr], bool]) -> int | None:
    """Return a proven positive period, or ``None`` for this grammar.

    Polynomial ordinal dependence is supported below a static remainder.
    Arbitrary scalar operations also preserve an already periodic pair of
    operands. The returned LCM is exact as a bound; callers enforce their own
    enumeration budgets before evaluating any phases.
    """
    periods: dict[Expr, int | None] = {}
    modular_periods: dict[tuple[Expr, int], int | None] = {}

    def combine(a: int | None, b: int | None) -> int | None:
        return None if a is None or b is None else lcm(a, b)

    def modular(current: Expr, modulus: int) -> int | None:
        key = current, modulus
        if key in modular_periods:
            return modular_periods[key]
        if modulus == 1 or invariant(current):
            result = 1
        elif current == ordinal:
            result = modulus
        elif current.kind == "add":
            result = combine(*(modular(child, modulus) for child in current.args))
        elif current.kind == "mul":
            a, b = current.args
            if a.constant is not None:
                result = modular(b, modulus // gcd(modulus, a.constant))
            elif b.constant is not None:
                result = modular(a, modulus // gcd(modulus, b.constant))
            else:
                result = combine(modular(a, modulus), modular(b, modulus))
        else:
            # A bounded periodic subexpression can be a polynomial coefficient
            # or summand. Unbounded min/max and quotient expressions do not
            # acquire a modular polynomial proof through this branch.
            result = visit(current)
        modular_periods[key] = result
        return result

    def visit(current: Expr) -> int | None:
        if current in periods:
            return periods[current]
        if invariant(current):
            result = 1
        elif current.kind in {"constant", "argument"}:
            result = None
        else:
            result = combine(*(visit(child) for child in current.args))
            if result is None and current.kind == "rem":
                numerator, divisor = current.args
                if divisor.constant is not None and divisor.constant > 0:
                    result = modular(numerator, divisor.constant)
        periods[current] = result
        return result

    return visit(value)


def evaluate_phase(value: Expr, ordinal: Expr, phase: int) -> int | None:
    """Evaluate one mathematical phase, leaving unknown parameters unresolved.

    A negative dividend is intentional here: a valid interval may begin after
    phase zero. No fixed-width or source-definedness claim follows from this
    proof-only interpretation.
    """
    values: dict[Expr, int | None] = {}

    def visit(current: Expr) -> int | None:
        if current in values:
            return values[current]
        if current == ordinal:
            result = phase
        elif current.kind == "constant":
            result = current.constant
        elif current.kind == "argument":
            result = None
        else:
            a, b = (visit(child) for child in current.args)
            if a is None or b is None:
                result = None
            elif current.kind == "add":
                result = a + b
            elif current.kind == "mul":
                result = a * b
            elif current.kind == "min":
                result = min(a, b)
            elif current.kind == "max":
                result = max(a, b)
            elif b <= 0:
                result = None
            elif current.kind == "floordiv":
                result = a // b
            elif current.kind == "ceildiv":
                result = a // b + int(a % b != 0)
            else:
                assert current.kind == "rem"
                result = a % b
        values[current] = result
        return result

    return visit(value)


def evaluate_comparison_at(test: Comparison, ordinal: Expr, phase: int) -> bool | None:
    """Compare mathematical phase values, or return ``None`` for a parameter."""
    lhs = evaluate_phase(test.lhs, ordinal, phase)
    rhs = evaluate_phase(test.rhs, ordinal, phase)
    if lhs is None or rhs is None:
        return None
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[test.predicate]

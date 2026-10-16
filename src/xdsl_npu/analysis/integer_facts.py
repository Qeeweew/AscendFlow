"""Integer query facts owned by one immutable scalar input environment.

An answer is reused only for the same expression and complete ordered premise
sequence. Ordered definedness checks validate unsigned operands before other
queries consume a guard's relations; no query establishes completion or ownership.
Tables die with this analysis, never live in process-global memoized methods.
"""

from functools import wraps
from types import MappingProxyType

from ..diagnostics import CompileError
from ..indexing import normalize
from ..scalar import DIVISION, Expr, expression, infer, literal, substitute
from .structured_schedule import Comparison

_OPPOSITE = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}


def _query(method):
    """Reuse a fact in this instance's immutable environment, including False."""

    @wraps(method)
    def query(self, *args):
        key = (method.__name__, *args)
        if key not in self._answers:
            self._answers[key] = method(self, *args)
        return self._answers[key]

    return query


def normalize_integer(value, facts):
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


def _term(test):
    if test.predicate in {">", ">="}:
        value = expression("sub", test.lhs, test.rhs)
        return expression("sub", value, literal(int(test.predicate == ">")))
    if test.predicate in {"<", "<="}:
        value = expression("sub", test.rhs, test.lhs)
        return expression("sub", value, literal(int(test.predicate == "<")))
    return None


class IntegerFacts:
    """Sufficient integer relations; unknown stays false, never becomes safety."""

    def __init__(self, facts):
        self._facts = MappingProxyType(dict(facts))
        self._answers = {}

    @property
    def facts(self):
        return self._facts

    @_query
    def normalize(self, value):
        return normalize_integer(value, self.facts)

    @_query
    def infer(self, value):
        return infer(value, self.facts)

    @_query
    def ordered_definedness(
        self, tests: tuple[Comparison, ...]
    ) -> tuple[Comparison, tuple[Comparison, ...]] | None:
        """Return the first missing unsigned premise and its ordered prefix.

        Every operand in the current comparison is checked before that
        comparison can establish a short-circuiting false guard. A later
        comparison supplies no premise for an earlier expression.
        """

        def proved(test, prefix):
            try:
                return self.prove(test, prefix)
            except CompileError as error:
                if error.message != "index divisor must be provably positive and uniform":
                    raise
                # Polynomial normalization cannot use a prefix-only dynamic
                # divisor fact. This query remains unproved, never a reason
                # to skip a later expression or accept a missing operand fact.
                return False

        def check(value, prefix):
            for child in value.args:
                if isinstance(child, Expr) and (missing := check(child, prefix)) is not None:
                    return missing
            if value.kind in DIVISION:
                numerator, divisor = value.args
                for required in (
                    Comparison(numerator, ">=", literal(0)),
                    Comparison(divisor, ">", literal(0)),
                ):
                    if not proved(required, prefix):
                        return required
            return None

        for index, test in enumerate(tests):
            prefix = tests[:index]
            for value in (test.lhs, test.rhs):
                if (missing := check(value, prefix)) is not None:
                    return missing, prefix
            if proved(Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs), prefix):
                return None
        return None

    @_query
    def bounded(self, value):
        """Normalize integer ring arithmetic while retaining division atoms.

        Quotient expansion is useful for equality, but loses the actual remainder
        interval needed by inequalities. These temporary names are local algebra
        variables only; none enters an event expression or emitted program.
        """
        atoms, replacements = {}, {}

        def encode(current):
            if current.kind in {"constant", "argument"}:
                return current
            if current.kind in {"add", "mul"}:
                return expression(current.kind, *(encode(child) for child in current.args))
            # Cancel the surrounding integer ring without expanding an atom's
            # domain: (g-1)*k can be nonnegative under its factor premise even
            # when interval reasoning cannot bound the expanded g*k-k.
            atom = current
            if atom.kind in {"floordiv", "ceildiv", "rem"} and atom.args[1].constant == 1:
                return encode(literal(0) if atom.kind == "rem" else atom.args[0])
            if atom.kind == "rem" and (period := atom.args[1].constant) is not None:
                numerator, divisor = atom.args
                if numerator.kind == "rem" and numerator.args[1] == divisor:
                    return encode(numerator)
                if numerator.kind == "add":
                    for shift, rest in (numerator.args, tuple(reversed(numerator.args))):
                        if (
                            shift.constant is not None
                            and shift.constant >= 0
                            and shift.constant % period == 0
                            and (rest.kind in DIVISION or self.infer(rest) is not None)
                        ):
                            # Unsigned results are nonnegative whenever defined.
                            # Keep that inner expression and its domain intact;
                            # only a nonnegative whole-period addition leaves.
                            return encode(expression("rem", rest, divisor))
            if atom.kind in {"min", "max"}:
                a, b = (self.infer(arg) for arg in atom.args)
                if a is not None and b is not None:
                    if a.maximum is not None and b.minimum is not None and a.maximum <= b.minimum:
                        return encode(atom.args[0 if atom.kind == "min" else 1])
                    if b.maximum is not None and a.minimum is not None and b.maximum <= a.minimum:
                        return encode(atom.args[1 if atom.kind == "min" else 0])
            if atom not in atoms:
                symbol = Expr("argument", (f"$resource_atom_{len(atoms)}",))
                atoms[atom], replacements[symbol] = symbol, atom
            return atoms[atom]

        encoded = encode(value)
        return substitute(normalize(encoded, {}), replacements)

    @_query
    def bounds(self, value, assumptions=()):
        # Preserve factors/clamps for interval reasoning. Polynomial expansion
        # is for equivalence queries: expanding (g-1)*k to g*k-k would lose
        # the fact that the complete first factor is nonnegative.
        fact = self.infer(value)
        lower, upper = (None, None) if fact is None else (fact.minimum, fact.maximum)
        for test in assumptions:
            if test.rhs.constant is not None and (
                value.kind == "argument" or self.bounded(test.lhs) == self.bounded(value)
            ):
                offset = self.normalize(self.bounded(expression("sub", test.lhs, value))).constant
                if offset is None:
                    continue
                bound = test.rhs.constant - offset
                if test.predicate in {">", ">=", "=="}:
                    candidate = bound + int(test.predicate == ">")
                    lower = candidate if lower is None else max(lower, candidate)
                if test.predicate in {"<", "<=", "=="}:
                    candidate = bound - int(test.predicate == "<")
                    upper = candidate if upper is None else min(upper, candidate)
        if value.kind == "constant":
            return value.constant, value.constant
        if lower is not None and upper is not None:
            return lower, upper
        if value.kind in {"add", "mul", "min", "max", *DIVISION}:
            (al, au), (bl, bu) = (self.bounds(arg, assumptions) for arg in value.args)
            if value.kind in DIVISION:
                # These are unsigned operations, not signed floor division.
                # Neither a zero/unknown denominator nor a possibly negative
                # numerator authorizes an interval result.
                if al is None or al < 0 or bl is None or bl <= 0:
                    return lower, upper
                if value.kind == "rem":
                    highs = [
                        bound for bound in (au, None if bu is None else bu - 1) if bound is not None
                    ]
                    calculated_lower, calculated_upper = 0, min(highs) if highs else None
                else:

                    def quotient(a, b):
                        return a // b + int(value.kind == "ceildiv" and a % b != 0)

                    calculated_lower = (
                        int(value.kind == "ceildiv" and al > 0) if bu is None else quotient(al, bu)
                    )
                    calculated_upper = None if au is None else quotient(au, bl)
            elif value.kind in {"min", "max"}:
                # max(a,b) needs only one lower bound; min(a,b) needs only
                # one upper bound. The opposite endpoint needs both. This
                # includes clamps whose signed input has no nonnegative Fact.
                operation = min if value.kind == "min" else max
                lows = [bound for bound in (al, bl) if bound is not None]
                highs = [bound for bound in (au, bu) if bound is not None]
                calculated_lower = (
                    operation(lows) if len(lows) == 2 or value.kind == "max" and lows else None
                )
                calculated_upper = (
                    operation(highs) if len(highs) == 2 or value.kind == "min" and highs else None
                )
            elif value.kind == "add":
                calculated_lower = None if al is None or bl is None else al + bl
                calculated_upper = None if au is None or bu is None else au + bu
            elif any(arg.constant is not None for arg in value.args):
                scalar, (minimum, maximum) = (
                    (value.args[0].constant, (bl, bu))
                    if value.args[0].constant is not None
                    else (value.args[1].constant, (al, au))
                )
                if scalar == 0:
                    calculated_lower, calculated_upper = 0, 0
                else:
                    if scalar < 0:
                        minimum, maximum = maximum, minimum
                    calculated_lower = None if minimum is None else scalar * minimum
                    calculated_upper = None if maximum is None else scalar * maximum
            elif all(bound is not None for bound in (al, au, bl, bu)):
                products = (al * bl, al * bu, au * bl, au * bu)
                calculated_lower, calculated_upper = min(products), max(products)
            else:
                calculated_lower = (
                    0 if al is not None and bl is not None and al >= 0 and bl >= 0 else None
                )
                calculated_upper = None
            if calculated_lower is not None:
                lower = calculated_lower if lower is None else max(lower, calculated_lower)
            if calculated_upper is not None:
                upper = calculated_upper if upper is None else min(upper, calculated_upper)
        return lower, upper

    @_query
    def prove(self, test, assumptions):
        if test in assumptions:
            return True
        constraint = _term(test)
        if constraint is not None:
            lower, _ = self.bounds(constraint, assumptions)
            if lower is not None and lower >= 0:
                return True
        difference = self.normalize(self.bounded(expression("sub", test.lhs, test.rhs)))
        if difference.constant is not None:
            return Comparison(difference, test.predicate, literal(0)).evaluate({})
        if test.predicate in {"==", "!="}:
            for prior in assumptions:
                if (
                    prior.predicate == test.predicate
                    and self.normalize(
                        expression("sub", expression("sub", prior.lhs, prior.rhs), difference)
                    ).constant
                    == 0
                ):
                    return True
            if test.predicate == "!=":
                lower, upper = self.bounds(difference, assumptions)
                return lower is not None and lower > 0 or upper is not None and upper < 0
            return self.prove(Comparison(test.lhs, ">=", test.rhs), assumptions) and self.prove(
                Comparison(test.lhs, "<=", test.rhs), assumptions
            )
        term = self.bounded(_term(test))
        for prior in assumptions:
            constraint = _term(prior)
            if constraint is not None:
                remainder = self.bounded(expression("sub", term, constraint))
                if remainder.constant is not None:
                    if remainder.constant >= 0:
                        return True
                    continue
                bound, _ = self.bounds(remainder)
                if bound is not None and bound >= 0:
                    return True
        lower, _ = self.bounds(term, assumptions)
        return lower is not None and lower >= 0

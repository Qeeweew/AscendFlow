"""Integer query facts owned by one immutable scalar input environment.

An answer is reused only for the same expression and complete ordered premise
sequence. Ordered definedness checks validate unsigned operands before other
queries consume a guard's relations; no query establishes completion or ownership.
Tables die with this analysis, never live in process-global memoized methods.

Non-trivial relations between original expressions are decided by
:class:`~ascendflow.analysis.smt.IntegerQueryService`, which encodes the raw DAG
under the same complete premises. Bounds stay a cheap interval view of the same
environment; they never replace the query service nor act as a proof.
"""

from dataclasses import dataclass
from functools import cache, wraps
from types import MappingProxyType

from ..indexing import AtomIdentity, exact_constant, normalize_integral
from ..scalar import DIVISION, Expr, expression, infer, literal
from .smt import IntegerQueryService
from .structured_schedule import Comparison

_OPPOSITE = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_NO_ANSWER = object()


def _query(method):
    """Reuse a fact in this instance's immutable environment, including False."""

    @wraps(method)
    def query(self, *args):
        key = (method.__name__, *args)
        answer = self._answers.get(key, _NO_ANSWER)
        if answer is _NO_ANSWER:
            answer = method(self, *args)
            self._answers[key] = answer
        return answer

    return query


@dataclass(frozen=True, slots=True, eq=False)
class _PremiseContext:
    """One owner's exact ordered premises, with an identity-only query key.

    The sequence is neither sorted nor deduplicated. This record establishes
    no additional facts and performs no eager arithmetic or definedness query.
    """

    owner: object
    tests: tuple[Comparison, ...]


class IntegerFacts:
    """Sufficient integer relations; unknown stays false, never becomes safety."""

    def __init__(self, facts):
        self._facts = MappingProxyType(dict(facts))
        self._answers = {}
        self._atom_identity = AtomIdentity()
        self._premise_owner = object()
        self._contexts: dict[tuple[Comparison, ...], _PremiseContext] = {}
        self._queries: IntegerQueryService | None = None

    @property
    def queries(self):
        """The single mathematical decision procedure for this environment."""
        if self._queries is None:
            self._queries = IntegerQueryService(self._facts)
        return self._queries

    @property
    def facts(self):
        return self._facts

    def _context(self, assumptions: tuple[Comparison, ...]) -> _PremiseContext:
        if not isinstance(assumptions, tuple):
            raise TypeError("integer premises must be an immutable tuple")
        context = self._contexts.get(assumptions)
        if context is None:
            context = _PremiseContext(self._premise_owner, assumptions)
            self._contexts[assumptions] = context
        return context

    def _check_context(self, context: _PremiseContext):
        if not isinstance(context, _PremiseContext) or context.owner is not self._premise_owner:
            raise ValueError("integer premise context belongs to another owner")

    @_query
    def _difference(self, lhs, rhs):
        """Construct and bound one operand difference in this facts owner."""
        return self.bounded(expression("sub", lhs, rhs))

    @_query
    def normalize(self, value):
        """Best-effort ring-normal form; a non-integral form keeps ``value``.

        The normalized spelling is a convenience for callers that compare or
        rebuild coordinates. It is not a proof and it never invents a quotient:
        when the declared multiples make a term rational, the original
        expression is returned and callers query relations explicitly.
        """
        return normalize_integral(value, self.facts, identity=self._atom_identity) or value

    @_query
    def exact_constant(self, value):
        """The exact integer value of one normalized expression, or ``None``."""
        return exact_constant(value, self.facts, identity=self._atom_identity)

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
            # One query service decides the premise. A premise that cannot be
            # proved in its own ordered prefix stays unproved; it is never a
            # reason to skip a later expression or accept a missing operand
            # fact.
            return self.prove(test, prefix)

        def check(value, prefix, completed):
            identity = id(value)
            if identity in completed:
                return None
            for child in value.args:
                if (
                    isinstance(child, Expr)
                    and (missing := check(child, prefix, completed)) is not None
                ):
                    return missing
            if value.kind in DIVISION:
                numerator, divisor = value.args
                for required in (
                    Comparison(numerator, ">=", literal(0)),
                    Comparison(divisor, ">", literal(0)),
                ):
                    if not proved(required, prefix):
                        return required
            completed.add(identity)
            return None

        for index, test in enumerate(tests):
            prefix = tests[:index]
            # Only these two operands share one complete ordered premise set.
            # Another comparison must establish its raw nodes under its prefix.
            completed = set()
            for value in (test.lhs, test.rhs):
                if (missing := check(value, prefix, completed)) is not None:
                    return missing, prefix
            if proved(Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs), prefix):
                return None
        return None

    @_query
    def bounded(self, value):
        """Normalize integer ring arithmetic while retaining division atoms.

        Quotient expansion is useful for equality, but loses the actual remainder
        interval needed by inequalities. Congruence keys identify functions
        without rebuilding the factors or unsigned domains of their operands.
        """

        @cache
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
            return atom

        return self._atom_identity.retained(encode(value))

    def bounds(self, value, assumptions: tuple[Comparison, ...] = ()):
        return self._bounds(value, self._context(assumptions))

    @_query
    def _contains_unsigned(self, value):
        return value.kind in DIVISION or any(
            self._contains_unsigned(child) for child in value.args if isinstance(child, Expr)
        )

    @_query
    def _unsigned_defined(self, value, context):
        if not self._contains_unsigned(value):
            return True
        if not all(
            self._unsigned_defined(child, context)
            for child in value.args
            if isinstance(child, Expr)
        ):
            return False
        if value.kind in DIVISION:
            numerator, divisor = value.args
            lower = self._bounds(numerator, context)[0]
            positive = self._bounds(divisor, context)[0]
            return lower is not None and lower >= 0 and positive is not None and positive > 0
        return True

    @_query
    def _bounds(self, value, context: _PremiseContext):
        self._check_context(context)
        assumptions = context.tests
        # Preserve factors/clamps for interval reasoning. Polynomial expansion
        # is for equivalence queries: expanding (g-1)*k to g*k-k would lose
        # the fact that the complete first factor is nonnegative.
        # Algebraic remainder/clamp ranges are conditional on the original
        # unsigned operands being defined. Derive these bounds recursively
        # instead of accepting a complete inferred interval before its domain.
        if not self._unsigned_defined(value, context):
            return None, None
        fact = self.infer(value)
        lower, upper = (None, None) if fact is None else (fact.minimum, fact.maximum)
        for test in assumptions:
            if test.rhs.constant is not None and (
                value.kind == "argument" or self.bounded(test.lhs) == self.bounded(value)
            ):
                offset = self.exact_constant(self._difference(test.lhs, value))
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
            (al, au), (bl, bu) = (self._bounds(arg, context) for arg in value.args)
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

    def prove(self, test, assumptions: tuple[Comparison, ...]):
        return self._prove(test, self._context(assumptions))

    @_query
    def _prove(self, test, context: _PremiseContext):
        """Answered by the shared query service under this exact premise tuple."""
        self._check_context(context)
        if test in context.tests:
            return True
        return self.queries.implies(test, context.tests).proved

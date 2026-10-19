"""One mathematical query service over the original integer expression DAG.

This shared service encodes the raw ``Expr`` DAG that the source program
already wrote, asks Z3 one
question under a complete ordered premise sequence, and returns a structured
answer. It never rewrites the program, never chooses a schedule, and never acts
as a certificate for definedness, ownership or completion.

Three separate semantic domains:

* **mathematics** (unbounded ``Int``): relations between the values the original
  DAG denotes. ``implies``/``equivalent``/``disjoint`` answer here. A
  mathematical proof says nothing about whether the queried expressions are
  legally evaluable; that is the ordered definedness obligation of the original
  program, established separately and consumed by the caller.
* **ordered definedness** (``Def``): every unsigned division/remainder in the
  DAG has a nonnegative numerator and a positive divisor, checked layer by layer
  in source evaluation order. ``defined`` answers here. Its premises are the
  ordered prefix the caller already established, so filtering non-evaluable
  premise states is exactly the contract; no other query may use that filter.
* **machine integers**: whether every original intermediate value fits signed
  64-bit. ``machine_representable`` answers here; mathematical identities such
  as ``(x + 1) - 1 == x`` say nothing about the original addition.

Outcome kinds are explicit. Universal queries return ``proved`` (solver
``unsat``), ``counterexample`` (``sat`` plus an assignment) or ``unknown``.
Existence queries return ``witness``, ``none`` or ``unknown`` in a separate
record, because an unsatisfiable existence question proves no relation and a
witness refutes none. ``sat`` is never a safety fact and ``unknown`` is never
permission: an unproven necessary admission is refused and an unproven optional
rewrite keeps the original expression.

Cached answers are keyed by query kind, the complete premise sequence, the
question and this service's immutable input facts, so different lexical scopes
never share an answer and an unchanged question is decided once. The per-query
timeout is bounded by the remaining compilation deadline. The outer process
group watchdog enforces the build's hard time limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from time import perf_counter

from z3 import (
    Z3_OP_UNINTERPRETED,
    And,
    BitVec,
    BitVecVal,
    BoolVal,
    If,
    Int,
    IntVal,
    Not,
    Or,
    Solver,
    URem,
    is_bv_value,
    sat,
    unsat,
)

from ..budget import poll, remaining
from ..diagnostics import require
from ..scalar import DIVISION, INT64_MAX, Expr, Fact
from .structured_schedule import Comparison

INT64_MIN = -INT64_MAX - 1
PROVED = "proved"
COUNTEREXAMPLE = "counterexample"
UNKNOWN = "unknown"
NONE = "none"
WITNESS = "witness"
_LOCAL_TIMEOUT_MS = 2000
_PREDICATES = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}


@dataclass(frozen=True, slots=True)
class Instance:
    """One concrete assignment returned by the solver."""

    values: tuple[tuple[str, int], ...]

    def text(self):
        return ", ".join(f"{name}={value}" for name, value in self.values)


@dataclass(frozen=True, slots=True)
class Answer:
    """Result of a universal query: does the goal hold for every instance?"""

    kind: str
    query: str
    elapsed: float = 0.0
    instance: Instance | None = None
    reason: str = ""

    def __post_init__(self):
        require(self.kind in {PROVED, COUNTEREXAMPLE, UNKNOWN}, "unknown query outcome kind")

    @property
    def proved(self):
        return self.kind == PROVED

    @property
    def refuted(self):
        return self.kind == COUNTEREXAMPLE

    def require_proved(self, message, **context):
        """Refuse a necessary admission; ``unknown`` is not permission."""
        if self.proved:
            return self
        detail = (
            f"counterexample: {self.instance.text()}"
            if self.instance is not None
            else f"unknown: {self.reason}"
        )
        require(False, message, outcome=self.kind, detail=detail, **context)


@dataclass(frozen=True, slots=True)
class Exists:
    """Result of an existential query: is there an instance satisfying it?"""

    kind: str
    query: str
    elapsed: float = 0.0
    instance: Instance | None = None
    reason: str = ""

    def __post_init__(self):
        require(self.kind in {WITNESS, NONE, UNKNOWN}, "unknown existence outcome kind")

    @property
    def satisfied(self):
        return self.kind == WITNESS

    @property
    def impossible(self):
        return self.kind == NONE

    def require_impossible(self, message, **context):
        if self.impossible:
            return self
        detail = (
            f"witness: {self.instance.text()}"
            if self.instance is not None
            else f"unknown: {self.reason}"
        )
        require(False, message, outcome=self.kind, detail=detail, **context)


@dataclass
class Statistics:
    """Distinct-query and outcome accounting for one compilation."""

    calls: int = 0
    hits: int = 0
    distinct: int = 0
    proved: int = 0
    counterexample: int = 0
    witness: int = 0
    none: int = 0
    unknown: int = 0
    seconds: float = 0.0
    maximum: float = 0.0

    def record(self, kind, elapsed, *, cached):
        self.calls += 1
        if cached:
            self.hits += 1
            return
        self.distinct += 1
        self.seconds += elapsed
        self.maximum = max(self.maximum, elapsed)
        setattr(self, kind, getattr(self, kind) + 1)

    def report(self):
        return {
            "calls": self.calls,
            "cache_hits": self.hits,
            "distinct": self.distinct,
            "proved": self.proved,
            "counterexample": self.counterexample,
            "witness": self.witness,
            "none": self.none,
            "unknown": self.unknown,
            "seconds": round(self.seconds, 6),
            "maximum_seconds": round(self.maximum, 6),
        }


def _cmp(left, predicate, right):
    return _PREDICATES[predicate](left, right)


class IntegerQueryService:
    """Decide integer/boolean questions under one immutable input environment."""

    def __init__(self, facts, *, timeout_ms=_LOCAL_TIMEOUT_MS):
        require(
            type(timeout_ms) is int and 0 < timeout_ms <= 60000,
            "integer query timeout must be a positive bounded number of milliseconds",
        )
        self._facts = dict(facts)
        for name, fact in self._facts.items():
            require(isinstance(fact, Fact), f"integer query facts must be Fact values: {name}")
        self._timeout = timeout_ms
        self._values = {}
        self._expressions = {}
        self._defined = {}
        self._contexts = {}
        self._answers = {}
        self.statistics = Statistics()
        self._solver = Solver()
        self._solver.set(random_seed=0)
        # Input facts are immutable for the lifetime of this service. Establish
        # them once; query push/pop adds only the actual lexical premises.
        for name, fact in self._facts.items():
            integer = self._integer(Expr("argument", (name,)))
            self._solver.add(integer >= fact.minimum)
            if fact.maximum is not None:
                self._solver.add(integer <= fact.maximum)
            if fact.multiple != 1:
                self._solver.add(integer % fact.multiple == 0)

    # -- encoding ---------------------------------------------------------
    def _integer(self, value: Expr):
        cached = self._expressions.get(value)
        if cached is not None:
            return cached
        if value.kind == "constant":
            result = IntVal(value.constant)
        elif value.kind == "argument":
            result = self._values.get(value.args[0])
            if result is None:
                result = Int(value.args[0])
                self._values[value.args[0]] = result
        else:
            a, b = (self._integer(arg) for arg in value.args)
            if value.kind == "add":
                result = a + b
            elif value.kind == "mul":
                result = a * b
            elif value.kind == "min":
                result = If(a < b, a, b)
            elif value.kind == "max":
                result = If(a > b, a, b)
            elif value.kind == "floordiv":
                result = a / b
            elif value.kind == "rem":
                result = a % b
            else:
                result = a / b + If(a % b == 0, IntVal(0), IntVal(1))
        self._expressions[value] = result
        return result

    def _def(self, value: Expr):
        """Ordered definedness of one original DAG node, layer by layer."""
        cached = self._defined.get(value)
        if cached is not None:
            return cached
        if value.kind in {"constant", "argument"}:
            result = BoolVal(True)
        else:
            a, b = value.args
            result = And(self._def(a), self._def(b))
            if value.kind in DIVISION:
                # Source unsigned division: nonnegative numerator, positive
                # divisor. Z3's total division is not source legality.
                result = And(result, self._integer(a) >= 0, self._integer(b) > 0)
        self._defined[value] = result
        return result

    def _comparison(self, test: Comparison):
        return _cmp(self._integer(test.lhs), test.predicate, self._integer(test.rhs))

    def _context(self, assumptions):
        """Input facts plus one complete ordered premise tuple, as relations.

        Premises are consumed as relations between values. Their own
        definedness is a separate obligation established in order by the
        caller's ordered checks; it is never used here to filter away illegal
        execution states.
        """
        require(
            isinstance(assumptions, tuple)
            and all(isinstance(item, Comparison) for item in assumptions),
            "integer query premises must be an immutable comparison tuple",
        )
        cached = self._contexts.get(assumptions)
        if cached is not None:
            return cached
        encoded = tuple(self._comparison(test) for test in assumptions)
        self._contexts[assumptions] = encoded
        return encoded

    # -- query core -------------------------------------------------------
    def _local_timeout(self):
        """The local cap, further bounded by the remaining build deadline."""
        left = remaining("integer query")
        if left is None:
            return self._timeout
        return max(1, min(self._timeout, int(left * 1000)))

    def _decide(self, kind, assumptions, key, goal, *, universal):
        """Decide one question and cache the structured outcome."""
        cache_key = (kind, assumptions, key)
        cached = self._answers.get(cache_key)
        if cached is not None:
            self.statistics.record(cached.kind, cached.elapsed, cached=True)
            return cached
        poll("integer query", distinct=self.statistics.distinct, calls=self.statistics.calls)
        solver = self._solver
        solver.set(timeout=self._local_timeout(), random_seed=0)
        started = perf_counter()
        solver.push()
        try:
            solver.add(*self._context(assumptions))
            solver.add(goal())
            outcome = solver.check()
            elapsed = perf_counter() - started
            if outcome == unsat:
                result = (PROVED if universal else NONE, elapsed, None, "")
            elif outcome == sat:
                result = (
                    COUNTEREXAMPLE if universal else WITNESS,
                    elapsed,
                    self._instance(solver),
                    "",
                )
            else:
                result = (UNKNOWN, elapsed, None, str(solver.reason_unknown()))
        finally:
            solver.pop()
        self.statistics.record(result[0], elapsed, cached=False)
        answer = (Answer if universal else Exists)(result[0], kind, result[1], result[2], result[3])
        self._answers[cache_key] = answer
        return answer

    def _instance(self, solver):
        model = solver.model()
        values = []
        for declaration in sorted(model.decls(), key=lambda item: item.name()):
            if declaration.kind() != Z3_OP_UNINTERPRETED:
                continue
            value = model[declaration]
            if value is None:
                continue
            if is_bv_value(value):
                values.append((declaration.name(), value.as_signed_long()))
            else:
                try:
                    values.append((declaration.name(), value.as_long()))
                except AttributeError:
                    continue
        return Instance(tuple(values))

    # -- mathematical queries --------------------------------------------
    def implies(self, test: Comparison, assumptions: tuple[Comparison, ...] = ()) -> Answer:
        """Prove ``assumptions -> test`` between values.

        Only a mathematical relation: the caller keeps the ordered definedness
        obligation for the guard and its uses.
        """

        def goal():
            return Not(self._comparison(test))

        return self._decide("implies", assumptions, ("implies", test), goal, universal=True)

    def equivalent(self, left: Expr, right: Expr, assumptions: tuple[Comparison, ...] = ()):
        """Prove two original expressions denote the same value.

        Mathematical equality only; it does not authorize evaluating either
        expression outside the ordered domain that admission checked.
        """

        def goal():
            return self._integer(left) != self._integer(right)

        return self._decide(
            "equivalent", assumptions, ("equivalent", left, right), goal, universal=True
        )

    def covers(self, alternatives, assumptions: tuple[Comparison, ...] = ()) -> Answer:
        """Prove a union of conjunctions contains the complete premise domain.

        This is a mathematical query. The caller separately establishes each
        guard's ordered definedness before treating it as executable code.
        """
        alternatives = tuple(alternatives)

        def goal():
            return Not(
                Or(
                    *(
                        And(*(self._comparison(test) for test in guard.comparisons))
                        for guard in alternatives
                    )
                )
            )

        return self._decide("covers", assumptions, ("covers", alternatives), goal, universal=True)

    def disjoint(
        self, left: Expr, right: Expr, width, assumptions: tuple[Comparison, ...] = ()
    ) -> Answer:
        """Prove two ``width``-byte physical accesses cannot overlap.

        ``proved`` means no legal value of the offsets overlaps; a
        counterexample is a concrete pair of overlapping rows.
        """
        require(type(width) is int and width > 0, "physical overlap query needs a positive width")

        def goal():
            return And(
                self._integer(left) < self._integer(right) + IntVal(width),
                self._integer(right) < self._integer(left) + IntVal(width),
            )

        return self._decide(
            "disjoint", assumptions, ("disjoint", left, right, width), goal, universal=True
        )

    # -- existential queries ---------------------------------------------
    def satisfies(self, tests, assumptions: tuple[Comparison, ...] = ()) -> Exists:
        """Ask for one assignment satisfying every test under the premises."""
        tests = tuple(tests)

        def goal():
            return And(*(self._comparison(test) for test in tests))

        return self._decide("satisfies", assumptions, ("satisfies", tests), goal, universal=False)

    # -- ordered definedness ---------------------------------------------
    def defined(self, tests, assumptions: tuple[Comparison, ...] = ()) -> Answer:
        """Prove every operand DAG of ``tests`` is definedly evaluable here.

        The premises are the ordered prefix the caller already established, so
        assuming their own definedness is exactly the contract of an ordered
        check. No other query uses that filter.
        """
        tests = tuple(tests)
        premises = tuple(dict.fromkeys(assumptions))

        def guarded():
            return And(
                Or(*(Not(self._def(value)) for test in tests for value in (test.lhs, test.rhs))),
                *(self._def(value) for test in premises for value in (test.lhs, test.rhs)),
            )

        return self._decide("defined", assumptions, ("defined", tests), guarded, universal=True)

    # -- machine integers -------------------------------------------------
    def machine_representable(self, value: Expr, assumptions: tuple[Comparison, ...] = ()):
        """Prove every original intermediate of ``value`` fits signed 64-bit."""
        nodes = tuple(_nodes(value))

        def goal():
            return Or(
                *(
                    Or(self._integer(node) < INT64_MIN, self._integer(node) > INT64_MAX)
                    for node in nodes
                )
            )

        return self._decide(
            "machine-range", assumptions, ("machine-range", nodes), goal, universal=True
        )

    def unsigned_remainder_mask(self, power: int, *, width=64):
        """Prove ``bits % 2**power == bits & (2**power - 1)`` over all bit patterns.

        A pure bit-vector identity with no extra hypothesis: the unsigned
        remainder of a ``width``-bit value by a power of two is exactly its low
        ``power`` bits. It never authorizes an undefined source remainder.
        """
        require(type(power) is int and 0 < power < 64, "mask query needs a bounded power")
        require(width == 64, "mask query is stated for the 64-bit execution domain")
        key = ("unsigned-mask", power, width)
        cached = self._answers.get(key)
        if cached is not None:
            self.statistics.record(cached.kind, cached.elapsed, cached=True)
            return cached
        bits = BitVec("unsigned_bits", width)
        mask = BitVecVal((1 << power) - 1, width)
        solver = Solver()
        solver.set(timeout=self._local_timeout(), random_seed=0)
        solver.add(URem(bits, BitVecVal(1 << power, width)) != (bits & mask))
        started = perf_counter()
        outcome = solver.check()
        elapsed = perf_counter() - started
        if outcome == unsat:
            answer = Answer(PROVED, "unsigned-mask", elapsed)
        elif outcome == sat:
            answer = Answer(COUNTEREXAMPLE, "unsigned-mask", elapsed, self._instance(solver))
        else:
            answer = Answer(UNKNOWN, "unsigned-mask", elapsed, reason=str(solver.reason_unknown()))
        self.statistics.record(answer.kind, elapsed, cached=False)
        self._answers[key] = answer
        return answer


def _nodes(value: Expr):
    seen, pending = [], [value]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.append(current)
        pending.extend(arg for arg in current.args if isinstance(arg, Expr))
    return seen


__all__ = [
    "COUNTEREXAMPLE",
    "NONE",
    "PROVED",
    "UNKNOWN",
    "WITNESS",
    "Answer",
    "Exists",
    "Instance",
    "IntegerQueryService",
    "Statistics",
]

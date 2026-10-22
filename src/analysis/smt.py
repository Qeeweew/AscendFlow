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

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from operator import eq, ge, gt, le, lt, ne
from time import perf_counter

from xdsl.dialects.builtin import IndexType
from xdsl.ir import SSAValue
from z3 import (
    Z3_OP_UNINTERPRETED,
    BitVec,
    BitVecVal,
    BoolRef,
    BoolVal,
    Function,
    If,
    Int,
    IntSort,
    IntVal,
    Not,
    Solver,
    URem,
    is_bv_value,
    is_true,
    main_ctx,
    sat,
    unsat,
)
from z3.z3 import _to_ast_array
from z3.z3core import Z3_mk_and, Z3_mk_or

from ..budget import poll, remaining
from ..diagnostics import require
from ..memo import memo_table, memoized, memoized_method
from ..scalar import DIVISION, INT64_MAX, Expr, Fact, infer
from ..symbols import Symbol, SymbolKind
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
class Boolean:
    """A mathematical conjunction/union, never an executable guard rewrite."""

    kind: str
    operands: tuple
    _hash: int | None = field(default=None, init=False, compare=False, repr=False)

    def __hash__(self):
        value = self._hash
        if value is None:
            value = hash((self.kind, self.operands))
            object.__setattr__(self, "_hash", value)
        return value

    def __post_init__(self):
        require(self.kind in {"and", "or"}, "unknown boolean query operation")
        require(
            isinstance(self.operands, tuple)
            and all(isinstance(value, (Comparison, Boolean, bool)) for value in self.operands),
            "boolean queries need immutable mathematical predicates",
        )


def all_of(*conditions):
    if any(condition is False for condition in conditions):
        return False
    values = tuple(dict.fromkeys(condition for condition in conditions if condition is not True))
    return True if not values else values[0] if len(values) == 1 else Boolean("and", values)


def any_of(*conditions):
    if any(condition is True for condition in conditions):
        return True
    values = tuple(dict.fromkeys(condition for condition in conditions if condition is not False))
    return False if not values else values[0] if len(values) == 1 else Boolean("or", values)


@memoized(key=lambda facts: frozenset(facts.items()))
def query_service(facts):
    """Reuse precisely the same immutable input environment in this build."""
    return IntegerQueryService(facts)


def query_statistics():
    services = memo_table(query_service)
    return tuple(
        {
            "input_symbols": len(service._facts),
            "expressions": service._integer.cache_info().currsize,
            "contexts": service._encode_context.cache_info().currsize,
            **service.statistics.report(),
        }
        for service in (() if services is None else services.values())
    )


@dataclass(frozen=True)
class Instance:
    """A fixed solver model; decode assignments only when diagnostics need them."""

    _model: object = field(repr=False)
    _symbols: dict = field(repr=False)
    _captures: tuple = field(repr=False)

    @cached_property
    def values(self):
        values = []
        for declaration in sorted(self._model.decls(), key=lambda item: item.name()):
            if declaration.kind() != Z3_OP_UNINTERPRETED:
                continue
            value = self._model[declaration]
            if value is None:
                continue
            if is_bv_value(value):
                values.append((declaration.name(), value.as_signed_long()))
            else:
                try:
                    values.append(
                        (self._symbols.get(declaration.name(), declaration.name()), value.as_long())
                    )
                except AttributeError:
                    continue
        values.extend(
            (symbol, self._model.eval(value, model_completion=True).as_long())
            for symbol, value in self._captures
        )
        return tuple(values)

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
    solver_checks: int = 0
    model_reuses: int = 0
    elementary_proofs: int = 0

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
            "solver_checks": self.solver_checks,
            "model_reuses": self.model_reuses,
            "elementary_proofs": self.elementary_proofs,
        }


def _cmp(left, predicate, right):
    return _PREDICATES[predicate](left, right)


def _boolean_packet(native, values):
    """Construct the exact AST from this closed encoder's typed BoolRefs.

    Every child comes from an integer comparison or another Boolean packet,
    never a caller-provided Z3 expression. Keep the same native constructor,
    operand order and nesting as Z3's general Python And/Or without repeating
    its arithmetic sort coercion. The pinned Z3 binding owns AST references.
    """
    require(not values or isinstance(values[0], BoolRef), "math packet needs typed BoolRefs")
    context = values[0].ctx if values else main_ctx()
    require(
        all(isinstance(value, BoolRef) and value.ctx is context for value in values),
        "math packet needs BoolRefs in one context",
    )
    arguments, size = _to_ast_array(values)
    return BoolRef(native(context.ref(), size, arguments), context)


def _and(*values):
    return _boolean_packet(Z3_mk_and, values)


def _or(*values):
    return _boolean_packet(Z3_mk_or, values)


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
        self._symbols = {}
        self._functions = {}
        self._captures = {}
        self._answers = {}
        self._witnesses = deque(maxlen=32)
        self.statistics = Statistics()
        self._solver = Solver()
        self._solver_timeout = None
        self._solver.set(random_seed=0)
        # Input facts are immutable for the lifetime of this service. Establish
        # them once; each query assumes only its actual lexical premises.
        for name, fact in self._facts.items():
            integer = self._integer(Expr("argument", (name,)))
            self._solver.add(integer >= fact.minimum)
            if (
                isinstance(name, Symbol)
                and name.kind is SymbolKind.PARAMETER
                and isinstance(name.origin, SSAValue)
                and isinstance(name.origin.type, IndexType)
            ):
                # This is a runtime input's machine representation, not a
                # static spatial bound for contracting a whole address family.
                self._solver.add(integer <= INT64_MAX)
            if fact.maximum is not None:
                self._solver.add(integer <= fact.maximum)
            if fact.multiple != 1:
                self._solver.add(integer % fact.multiple == 0)

    # -- encoding ---------------------------------------------------------
    @memoized_method
    def _integer(self, value: Expr):
        if value.kind == "constant":
            result = IntVal(value.constant)
        elif value.kind == "argument":
            symbol = value.args[0]
            result = self._values.get(symbol)
            if result is None:
                if (
                    isinstance(symbol, Symbol)
                    and symbol.kind in {SymbolKind.METADATA, SymbolKind.CARRY}
                    and symbol.captures
                ):
                    key = symbol.kind, symbol._identity
                    function = self._functions.get(key)
                    if function is None:
                        function = Function(
                            f"capture_{len(self._functions)}",
                            *(IntSort() for _ in range(len(symbol.captures) + 1)),
                        )
                        self._functions[key] = function
                    result = function(*(self._integer(capture) for capture in symbol.captures))
                    self._captures[symbol] = result
                else:
                    name = f"scalar_{len(self._values)}"
                    result = Int(name)
                    self._symbols[name] = symbol
                self._values[symbol] = result
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
                numerator, modulus = value.args
                quotient = numerator.args[0] if numerator.kind == "floordiv" else None
                product = (
                    quotient.args[1] if quotient is not None and quotient.kind == "rem" else None
                )
                divisor = numerator.args[1] if numerator.kind == "floordiv" else None
                # 0 <= floor((x mod (d*m))/d) < m for positive d,m.
                # Removing the outer modulo avoids a redundant nonlinear
                # division. This is only a mathematical encoding identity:
                # Def and machine checks still traverse every original node.
                if (
                    product is not None
                    and product.kind == "mul"
                    and product.args in ((divisor, modulus), (modulus, divisor))
                    and (dl := self._range(divisor)[0]) is not None
                    and dl > 0
                    and (ml := self._range(modulus)[0]) is not None
                    and ml > 0
                ):
                    result = a
                else:
                    result = a % b
            elif value.args[1].constant is not None and value.args[1].constant > 0:
                # For a positive divisor, ceil(a/b) = floor((a+b-1)/b)
                # for every mathematical integer a. Avoid a second remainder
                # and conditional per ceildiv in the solver's value encoding.
                # The original unsigned DAG and machine-range checks remain
                # separate; this is never emitted as machine arithmetic.
                result = (a + (value.args[1].constant - 1)) / b
            else:
                result = a / b + If(a % b == 0, IntVal(0), IntVal(1))
        return result

    @memoized_method
    def _def(self, value: Expr):
        """Ordered definedness of one original DAG node, layer by layer."""
        if value.kind in {"constant", "argument"}:
            result = BoolVal(True)
        else:
            a, b = value.args
            result = _and(self._def(a), self._def(b))
            if value.kind in DIVISION:
                # Source unsigned division: nonnegative numerator, positive
                # divisor. Z3's total division is not source legality.
                result = _and(result, self._integer(a) >= 0, self._integer(b) > 0)
        return result

    @memoized_method
    def _comparison(self, test: Comparison | Boolean | bool):
        if type(test) is bool:
            return BoolVal(test)
        if isinstance(test, Boolean):
            operation = _and if test.kind == "and" else _or
            return operation(*(self._comparison(value) for value in test.operands))
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
            and all(isinstance(item, (Comparison, Boolean, bool)) for item in assumptions),
            "integer query premises must be an immutable predicate tuple",
        )
        return self._encode_context(assumptions)

    @memoized_method
    def _encode_context(self, assumptions):
        return (_and(*(self._comparison(test) for test in assumptions)),)

    # -- query core -------------------------------------------------------
    @memoized_method
    def _range(self, value):
        if value.constant is not None:
            return value.constant, value.constant
        bound = infer(value, self._facts)
        return (None, None) if bound is None else (bound.minimum, bound.maximum)

    def _elementary(self, test):
        """Sufficient mathematical facts, independent of lexical premises.

        Exact constants/identity and disjoint enclosing intervals need no
        search. This never authorizes evaluation of an original partial DAG.
        All correlated or unresolved relations still use the complete SMT
        context, including contradictory premises.
        """
        if type(test) is bool:
            return test
        if isinstance(test, Boolean):
            values = (self._elementary(value) for value in test.operands)
            return all(values) if test.kind == "and" else any(values)
        if test.lhs == test.rhs:
            return test.predicate in {"==", "<=", ">="}
        low, high = self._range(test.lhs)
        other_low, other_high = self._range(test.rhs)
        if test.predicate in {">", ">="}:
            low, high, other_low, other_high = other_low, other_high, low, high
            predicate = "<" if test.predicate == ">" else "<="
        else:
            predicate = test.predicate
        if predicate in {"<", "<="}:
            return (
                high is not None
                and other_low is not None
                and (high < other_low if predicate == "<" else high <= other_low)
            )
        if predicate == "==":
            return low is not None and low == high == other_low == other_high
        return (
            high is not None
            and other_low is not None
            and high < other_low
            or other_high is not None
            and low is not None
            and other_high < low
        )

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
        started = perf_counter()
        if kind == "implies" and self._elementary(key[1]):
            require(
                isinstance(assumptions, tuple)
                and all(isinstance(item, (Comparison, Boolean, bool)) for item in assumptions),
                "integer query premises must be an immutable predicate tuple",
            )
            elapsed = perf_counter() - started
            answer = Answer(PROVED, kind, elapsed)
            self.statistics.elementary_proofs += 1
            self.statistics.record(PROVED, elapsed, cached=False)
            self._answers[cache_key] = answer
            return answer
        solver = self._solver
        context, objective = self._context(assumptions), goal()
        question = _and(*context, objective)
        # An already obtained SMT model can witness another question only if
        # that exact model satisfies its COMPLETE context and counterexample
        # formula. This never proves a universal claim, never samples inputs,
        # and never uses model completion to invent a missing value.
        for model, values in self._witnesses:
            # A fixed model gives the same value to each original predicate
            # across lexical contexts. Reuse those evaluations, but require
            # every actual premise and this objective to hold in that model.
            valid = True
            for test in assumptions:
                if test not in values:
                    values[test] = is_true(
                        model.eval(self._comparison(test), model_completion=False)
                    )
                if not values[test]:
                    valid = False
                    break
            if valid and is_true(model.eval(objective, model_completion=False)):
                elapsed = perf_counter() - started
                outcome = COUNTEREXAMPLE if universal else WITNESS
                answer = (Answer if universal else Exists)(
                    outcome, kind, elapsed, self._model_instance(model)
                )
                self.statistics.model_reuses += 1
                self.statistics.record(outcome, elapsed, cached=False)
                self._answers[cache_key] = answer
                return answer
        # Keep each query in its own assertion scope. Assertion preprocessing
        # is important for nonlinear arithmetic; passing the entire formula as
        # a Boolean assumption can leave even simple bounded queries unresolved.
        timeout = self._local_timeout()
        if timeout != self._solver_timeout:
            solver.set(timeout=timeout)
            self._solver_timeout = timeout
        self.statistics.solver_checks += 1
        solver.push()
        try:
            solver.add(question)
            outcome = solver.check()
            elapsed = perf_counter() - started
            if outcome == unsat:
                result = (PROVED if universal else NONE, elapsed, None, "")
            elif outcome == sat:
                model = solver.model()
                self._witnesses.appendleft((model, dict.fromkeys(assumptions, True)))
                result = (
                    COUNTEREXAMPLE if universal else WITNESS,
                    elapsed,
                    self._model_instance(model),
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
        return self._model_instance(solver.model())

    def _model_instance(self, model):
        return Instance(model, self._symbols, tuple(self._captures.items()))

    # -- mathematical queries --------------------------------------------
    def implies(self, test: Comparison | Boolean | bool, assumptions: tuple = ()) -> Answer:
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
                _or(
                    *(
                        _and(*(self._comparison(test) for test in guard.comparisons))
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
            return _and(
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
            return _and(*(self._comparison(test) for test in tests))

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
        require(
            all(isinstance(test, Comparison) for test in (*tests, *premises)),
            "ordered definedness needs original comparisons in evaluation order",
        )

        def guarded():
            return _and(
                _or(*(Not(self._def(value)) for test in tests for value in (test.lhs, test.rhs))),
                *(self._def(value) for test in premises for value in (test.lhs, test.rhs)),
            )

        return self._decide("defined", assumptions, ("defined", tests), guarded, universal=True)

    # -- machine integers -------------------------------------------------
    def machine_representable(self, value: Expr, assumptions: tuple[Comparison, ...] = ()):
        """Prove every original intermediate of ``value`` fits signed 64-bit."""
        nodes = tuple(_nodes(value))

        def goal():
            return _or(
                *(
                    _or(self._integer(node) < INT64_MIN, self._integer(node) > INT64_MAX)
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
        self.statistics.solver_checks += 1
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

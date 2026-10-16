"""Exact single-clock event membership and guarded finite window operations.

Other coordinates must be proved invariant over this actual loop. Eliminating
a child clock is explicit: its existence certificate can then be consumed by
the containing domain stage. This module never projects an unrelated clock or
assumes a sparse activation is an interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import gcd, lcm

from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import arguments, normalize
from ..scalar import INT64_MAX, Expr, Fact, expression, literal, substitute
from .periodic import evaluate_comparison_at, period_of
from .structured_protocols import LogicalNotification
from .structured_schedule import ActionGuard, Comparison

Proof = Callable[[Comparison, tuple[Comparison, ...]], bool]
Bounds = Callable[[Expr, tuple[Comparison, ...]], tuple[int | None, int | None]]
OrderedDefinedness = Callable[
    [tuple[Comparison, ...]], tuple[Comparison, tuple[Comparison, ...]] | None
]

# Analysis budgets for a concrete periodic set, independent of hardware banks.
MAX_ENUMERATED_PERIOD = 64
MAX_RESIDUE_WORK = 4096
MAX_RESIDUES = 64


class WindowPartitionRequired(CompileError):
    """Unresolved exact membership tests available for parameter partitioning."""

    def __init__(self, message, *, tests: tuple[Comparison, ...], **context):
        self.tests = tests
        super().__init__(message, **context)


def _sum(values):
    result = literal(0)
    for value in values:
        result = expression("add", result, value)
    return result


def _minimum(values):
    values = iter(values)
    result = next(values)
    for value in values:
        result = expression("min", result, value)
    return result


def _indicator(value):
    return expression("max", literal(0), expression("min", literal(1), value))


def _prefix_count(value, period, residues):
    """Number of members in [0,value); value is nonnegative in its scope."""
    if not residues:
        return literal(0)
    if period == 1:
        return value
    complete = expression(
        "mul", literal(len(residues)), expression("floordiv", value, literal(period))
    )
    remainder = expression("rem", value, literal(period))
    tail = _sum(_indicator(expression("sub", remainder, residue)) for residue in residues)
    return expression("add", complete, tail)


def _residue_guard(ordinal, period, residues):
    if not residues:
        return Comparison(literal(0), "==", literal(1))
    if len(residues) == 1:
        return Comparison(expression("rem", ordinal, literal(period)), "==", residues[0])
    # The surrounding interval first establishes 0 <= ordinal < count, so
    # ordinal+1 remains representable even at the final int64 member.
    return Comparison(
        _prefix_count(expression("add", ordinal, literal(1)), period, residues),
        ">",
        _prefix_count(ordinal, period, residues),
    )


def _constant_false(test):
    return test.lhs.constant is not None and test.rhs.constant is not None and not test.evaluate({})


@dataclass(frozen=True, slots=True)
class WindowExistence:
    loop: Operation
    ordinal: Expr
    membership: ActionGuard
    guard: ActionGuard


@dataclass(frozen=True, slots=True)
class EventWindow:
    loop: Operation
    first: Expr
    count: Expr
    period: int
    residues: tuple[Expr, ...]
    upper: int | None
    single: bool = False
    ordinal: Expr = field(kw_only=True)
    lower: Expr = field(kw_only=True)
    membership: ActionGuard = field(kw_only=True)
    interval: ActionGuard = field(kw_only=True)
    invariants: ActionGuard = field(kw_only=True)
    nonempty: ActionGuard = field(kw_only=True)

    def __post_init__(self):
        require(type(self.period) is int and 0 < self.period <= INT64_MAX, "invalid event period")
        require(
            isinstance(self.residues, tuple) and all(isinstance(r, Expr) for r in self.residues),
            "event residues must be an immutable expression tuple",
        )
        require(
            len(self.residues) <= MAX_RESIDUES,
            "event residue count exceeds analysis budget",
            kind=ErrorKind.ANALYSIS_LIMIT,
        )
        if len(self.residues) > 1:
            values = tuple(residue.constant for residue in self.residues)
            require(
                all(value is not None and 0 <= value < self.period for value in values)
                and values == tuple(sorted(set(values))),
                "multiple event residues must be ascending distinct static integers",
            )

    def prefix_count(self, value: Expr) -> Expr:
        """Exact H(value); callers establish nonnegative value and invariants."""
        return _prefix_count(value, self.period, self.residues)

    def first_last(self) -> tuple[Expr, Expr]:
        """Both values are members under ``nonempty``; otherwise evaluate neither."""
        if self.single or self.upper == 0 or not self.residues:
            return self.first, self.first
        last = expression("sub", self.count, literal(1))
        if self.period != 1:
            offset = expression("rem", last, literal(self.period))
            backward = _minimum(
                _difference_mod(offset, residue, self.period) for residue in self.residues
            )
            last = expression("sub", last, backward)
        return self.first, last

    def project_exists(self) -> WindowExistence:
        """Exact existential elimination of this clock, retaining all parameters."""
        return WindowExistence(self.loop, self.ordinal, self.membership, self.nonempty)

    def rank(self) -> Expr:
        """Zero-based member rank, evaluated only under ``membership``."""
        if self.single or not self.residues:
            return literal(0)
        if len(self.residues) > 1:
            return expression("sub", self.prefix_count(self.ordinal), self.prefix_count(self.lower))
        return expression(
            "floordiv", expression("sub", self.ordinal, self.first), literal(self.period)
        )

    def next_same_id(self, width: int) -> Expr:
        """Mathematical lifetime index; this expression requests no runtime work.

        A successor exists only under ``same_id_guard``. Emitted event IDs use
        ``rank`` at an actual member, rather than executing this future index.
        """
        require(type(width) is int and width > 0, "event ring width must be positive")
        require(self.residues, "empty event window has no successor")
        if len(self.residues) > 1:
            # At member R[i], dense rank selects R[(i+width) % size].
            # A staircase of its constant shifts avoids expanding H(g) in
            # every residue selector, retaining the ordinary Expr budget.
            deltas = self._reuse_deltas(width)
            position = expression("rem", self.ordinal, literal(self.period))
            delta = expression(
                "add",
                literal(deltas[0]),
                _sum(
                    expression(
                        "mul",
                        literal(deltas[i] - deltas[i - 1]),
                        _indicator(expression("sub", position, self.residues[i - 1])),
                    )
                    for i in range(1, len(self.residues))
                ),
            )
            return expression("add", self.ordinal, delta)
        return expression("add", self.ordinal, literal(width * self.period))

    def _reuse_deltas(self, width: int) -> tuple[int, ...]:
        size = len(self.residues)
        return tuple(
            ((index + width) // size) * self.period
            + self.residues[(index + width) % size].constant
            - residue.constant
            for index, residue in enumerate(self.residues)
        )

    def same_id_guard(self, width: int) -> ActionGuard:
        require(type(width) is int and width > 0, "event ring width must be positive")
        if not self.residues or self.single:
            return _guard(self.membership.comparisons + (Comparison(literal(0), "==", literal(1)),))
        if len(self.residues) > 1:
            return _guard(
                self.membership.comparisons
                + (
                    Comparison(
                        expression(
                            "sub", self.prefix_count(self.count), self.prefix_count(self.ordinal)
                        ),
                        ">",
                        literal(width),
                    ),
                )
            )
        successor = self.next_same_id(width)
        return _guard(
            self.membership.comparisons
            + (
                Comparison(
                    literal(width * self.period),
                    "<",
                    expression("sub", self.count, self.ordinal),
                ),
            )
            + tuple(
                Comparison(
                    substitute(test.lhs, {self.ordinal: successor}),
                    test.predicate,
                    substitute(test.rhs, {self.ordinal: successor}),
                )
                for test in self.membership.comparisons
            )
        )

    def reuse_phases(self, width: int) -> tuple[tuple[ActionGuard, int], ...]:
        """Exact finite current-residue cases for dense same-ID reuse."""
        guard = self.same_id_guard(width)
        if not self.residues:
            return ()
        if len(self.residues) == 1:
            return ((guard, width * self.period),)
        result = []
        for residue, delta in zip(self.residues, self._reuse_deltas(width), strict=True):
            successor = expression("add", self.ordinal, literal(delta))
            phase = _guard(
                guard.comparisons
                + (
                    Comparison(
                        expression("rem", self.ordinal, literal(self.period)), "==", residue
                    ),
                )
                + tuple(
                    Comparison(
                        substitute(test.lhs, {self.ordinal: successor}),
                        test.predicate,
                        substitute(test.rhs, {self.ordinal: successor}),
                    )
                    for test in self.membership.comparisons
                )
            )
            result.append((phase, delta))
        return tuple(result)


def _guard(tests):
    return ActionGuard(
        tuple(
            dict.fromkeys(
                test
                for test in tests
                if not (
                    test.lhs.constant is not None
                    and test.rhs.constant is not None
                    and test.evaluate({})
                )
            )
        )
    )


def _normal(value, facts):
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


def _difference_mod(a, b, period):
    """Canonical (a-b) mod p for 0<=a<p and 0<=b<=p, with safe sums."""
    if period == 1:
        return literal(0)
    if a.constant is not None and b.constant is not None:
        return literal((a.constant - b.constant) % period)
    if b.constant == 0:
        return a
    if 2 * period - 1 <= INT64_MAX:
        return expression(
            "rem", expression("add", a, expression("sub", literal(period), b)), literal(period)
        )
    delta = expression("sub", a, b)
    positive = expression("max", delta, literal(0))
    negative = expression("max", expression("mul", literal(-1), delta), literal(0))
    # Only one summand is nonzero; every intermediate is in [-p,p].
    return expression(
        "add",
        positive,
        expression("rem", expression("sub", literal(period), negative), literal(period)),
    )


class _Builder:
    def __init__(
        self,
        notification,
        loop,
        ordinal,
        count,
        guards,
        facts,
        proves,
        bounds,
        invariant,
        parameter_bindings,
        *,
        ordered_definedness,
    ):
        self.notification, self.loop, self.ordinal = notification, loop, ordinal
        self.facts, self.proves, self.bounds, self.invariant = facts, proves, bounds, invariant
        self.ordered_definedness = ordered_definedness
        self.actual = guards
        self.definedness()
        self.parameter_bindings = dict(parameter_bindings)
        for source, replacement in self.parameter_bindings.items():
            require(
                isinstance(source, Expr)
                and isinstance(replacement, Expr)
                and bool(arguments(source))
                and invariant(source)
                and invariant(replacement)
                and all(
                    name in facts and (not name.startswith("$") or name == "$group")
                    for name in arguments(source) | arguments(replacement)
                ),
                "event parameter bindings must contain only uniform input expressions",
            )
        actual, inactive = [], False
        for test in guards:
            specialized = self.specialize(test)
            actual.append(specialized)
            if _constant_false(specialized):
                inactive = True
                break
        self.actual = tuple(dict.fromkeys(actual))
        self.lower, self.count = (
            literal(0),
            literal(0) if inactive else _normal(substitute(count, self.parameter_bindings), facts),
        )
        self.period, self.residues, self.single = 1, (literal(0),), False
        self.residue_work = 0
        self.invariants, self.pending, self.established = [], [], []

    def specialize(self, test):
        return Comparison(
            substitute(test.lhs, self.parameter_bindings),
            test.predicate,
            substitute(test.rhs, self.parameter_bindings),
        )

    def fail(self, message, *, witness=()):
        raise CompileError(
            f"{self.notification.name}: {message}",
            kind=ErrorKind.UNSUPPORTED,
            phase="event-window-membership",
            location=str(self.loop.location),
            witness=(f"clock={self.ordinal.text()}", *witness),
        )

    def constant(self, value):
        return _normal(value, self.facts)

    def limit(self, condition, message):
        if not condition:
            raise CompileError(
                f"{self.notification.name}: {message}",
                kind=ErrorKind.ANALYSIS_LIMIT,
                phase="event-window-membership",
                location=str(self.loop.location),
                witness=(f"clock={self.ordinal.text()}",),
            )

    def charge(self, amount):
        self.limit(
            self.residue_work + amount <= MAX_RESIDUE_WORK,
            f"concrete residue work exceeds analysis budget {MAX_RESIDUE_WORK}",
        )
        self.residue_work += amount

    def linear(self, value):
        value = self.constant(value)
        if self.invariant(value):
            return 0, value
        if value == self.ordinal:
            return 1, literal(0)
        if value.kind == "add":
            a, b = (self.linear(child) for child in value.args)
            if a is not None and b is not None:
                return a[0] + b[0], self.constant(expression("add", a[1], b[1]))
        if value.kind == "mul":
            for scalar, other in (value.args, tuple(reversed(value.args))):
                if scalar.constant is not None and (part := self.linear(other)) is not None:
                    return part[0] * scalar.constant, self.constant(
                        expression("mul", scalar, part[1])
                    )
        return None

    def add_invariant(self, test):
        if not self.invariant(test.lhs) or not self.invariant(test.rhs):
            self.fail("event membership retains a non-invariant coordinate or unknown scalar")
        self.invariants.append(test)

    def definedness(self):
        """A later guard cannot justify an earlier unsigned division."""
        missing = self.ordered_definedness(self.actual)
        if missing is not None:
            required, prefix = missing
            self.fail(
                "unsigned arithmetic lacks an ordered activation prefix proof",
                witness=(
                    f"required: {required.lhs.text()} {required.predicate} {required.rhs.text()}",
                    "ordered prefix: "
                    + "; ".join(
                        f"{prior.lhs.text()} {prior.predicate} {prior.rhs.text()}"
                        for prior in prefix
                    ),
                ),
            )

    def modulo(self, value, period):
        if period == 1:
            return literal(0)
        if value.constant is not None:
            return literal(value.constant % period)
        if value.kind in {"add", "mul"}:
            a, b = (self.modulo(child, period) for child in value.args)
            if value.kind == "add":
                return _difference_mod(a, expression("sub", literal(period), b), period)
            maximum_a = a.constant if a.constant is not None else period - 1
            maximum_b = b.constant if b.constant is not None else period - 1
            if maximum_a * maximum_b > INT64_MAX:
                self.fail("modular product has no int64 intermediate range proof")
            return expression("rem", expression("mul", a, b), literal(period))
        if self.proves(Comparison(value, ">=", literal(0)), tuple(self.invariants)):
            return expression("rem", value, literal(period))
        self.fail("signed residue offset lacks a nonnegative arithmetic proof")

    def intersect(self, period, residues):
        """Intersect one real periodic set, checking its finite work first."""
        if type(period) is not int or not 0 < period <= INT64_MAX:
            self.fail("event residue period is not a positive int64 constant")
        self.limit(
            len(residues) <= MAX_RESIDUES,
            f"concrete residue count exceeds analysis budget {MAX_RESIDUES}",
        )
        combined, divisor = lcm(self.period, period), gcd(self.period, period)
        if combined > INT64_MAX:
            self.fail("event residue intersection exceeds int64 period")
        if not self.residues or not residues:
            self.period, self.residues = combined, ()
            return
        if any(residue.constant is None for residue in self.residues + residues):
            if len(self.residues) != 1 or len(residues) != 1:
                self.fail(
                    "dynamic residue intersection would require a conditional multi-residue set"
                )
            self._intersect_single(period, residues[0])
            return
        self.charge(len(self.residues) * len(residues))
        left, right = {}, {}
        for table, values in ((left, self.residues), (right, residues)):
            for value in values:
                key = value.constant % divisor
                table[key] = table.get(key, 0) + 1
        size = sum(amount * right.get(key, 0) for key, amount in left.items())
        self.limit(
            size <= MAX_RESIDUES,
            f"concrete residue intersection exceeds analysis budget {MAX_RESIDUES}",
        )
        modulus = period // divisor
        inverse = pow(self.period // divisor, -1, modulus) if modulus != 1 else 0
        result = []
        for left_residue in self.residues:
            for right_residue in residues:
                difference = right_residue.constant - left_residue.constant
                if difference % divisor:
                    continue
                step = difference // divisor * inverse % modulus if modulus != 1 else 0
                result.append((left_residue.constant + self.period * step) % combined)
        self.period, self.residues = combined, tuple(literal(value) for value in sorted(result))
        if len(self.residues) == self.period:
            self.period, self.residues = 1, (literal(0),)

    def congruence(self, period, residue):
        if type(period) is not int or not 0 < period <= INT64_MAX:
            self.fail("event residue period is not a positive int64 constant")
        self.add_invariant(Comparison(residue, ">=", literal(0)))
        self.add_invariant(Comparison(residue, "<", literal(period)))
        # Invalid residues make the original congruence empty. Keeping that
        # invariant permits safe guarded arithmetic for the remaining window.
        if residue.constant is not None:
            residue = literal(residue.constant % period)
        self.intersect(period, (residue,))

    def _intersect_single(self, period, residue):
        p, r = self.period, self.residues[0]
        if period % p == 0:
            self.add_invariant(Comparison(self.modulo(residue, p), "==", r))
            self.period, self.residues = period, (residue,)
        elif p % period == 0:
            self.add_invariant(Comparison(self.modulo(r, period), "==", residue))
        else:
            combined, divisor = lcm(p, period), gcd(p, period)
            if combined > INT64_MAX:
                self.fail("event residue intersection exceeds int64 period")
            difference = expression("sub", expression("add", residue, literal(p)), r)
            self.add_invariant(
                Comparison(expression("rem", difference, literal(divisor)), "==", literal(0))
            )
            modulus = period // divisor
            inverse = pow(p // divisor, -1, modulus)
            if (p + period - 1) // divisor * inverse + modulus - 1 > INT64_MAX:
                self.fail("CRT product has no int64 intermediate range proof")
            step = expression(
                "rem",
                expression(
                    "add",
                    expression(
                        "mul",
                        expression("floordiv", difference, literal(divisor)),
                        literal(inverse),
                    ),
                    literal(modulus - 1),
                ),
                literal(modulus),
            )
            self.period, self.residues = (
                combined,
                (
                    expression(
                        "rem",
                        expression("add", r, expression("mul", literal(p), step)),
                        literal(combined),
                    ),
                ),
            )

    def residue_test(self, test):
        predicate = test.predicate
        for term, target in ((test.lhs, test.rhs), (test.rhs, test.lhs)):
            if term.kind != "rem" or not self.invariant(target):
                continue
            numerator, divisor = term.args
            period = divisor.constant
            part = self.linear(numerator)
            if period is None or part is None:
                continue
            comparison = (
                predicate
                if term is test.lhs
                else {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[predicate]
            )
            if (part[0] not in {-1, 1} or comparison != "==") and (
                target.constant is not None and part[1].constant is not None
            ):
                reduced = period // gcd(period, part[0])
                self.limit(
                    reduced <= MAX_ENUMERATED_PERIOD,
                    f"static affine period {reduced} exceeds enumeration analysis budget {MAX_ENUMERATED_PERIOD}",
                )
                self.charge(reduced)
                residues = tuple(
                    literal(value)
                    for value in range(reduced)
                    if Comparison(
                        literal((part[0] * value + part[1].constant) % period),
                        comparison,
                        target,
                    ).evaluate({})
                )
                self.intersect(reduced, residues)
                return True
            if part[0] not in {-1, 1}:
                continue
            if comparison == "!=":
                if period != 2 or target.constant not in {0, 1}:
                    continue
                target = literal(1 - target.constant)
            elif comparison != "==":
                continue
            if target.constant is not None and not 0 <= target.constant < period:
                self.intersect(period, ())
                return True
            self.add_invariant(Comparison(target, ">=", literal(0)))
            self.add_invariant(Comparison(target, "<", literal(period)))
            # x+c == r mod p, or -x+c == r mod p. Canonical modular
            # arithmetic avoids taking an unsigned remainder of r-c.
            offset = self.modulo(part[1], period)
            a, b = (target, offset) if part[0] == 1 else (offset, target)
            self.congruence(period, _difference_mod(a, b, period))
            return True
        lhs_period = period_of(test.lhs, self.ordinal, self.invariant)
        rhs_period = period_of(test.rhs, self.ordinal, self.invariant)
        if lhs_period is None or rhs_period is None:
            return False
        period = lcm(lhs_period, rhs_period)
        self.limit(
            period <= MAX_ENUMERATED_PERIOD,
            f"structural period {period} exceeds enumeration analysis budget {MAX_ENUMERATED_PERIOD}",
        )
        self.charge(period)
        divisor = gcd(self.period, period)
        possible_phases = (
            {residue.constant % divisor for residue in self.residues}
            if all(residue.constant is not None for residue in self.residues)
            else None
        )
        residues = []
        for phase in range(period):
            # A prior periodic conjunct may exclude a zero divisor phase.
            # Original ordered definedness was checked independently; absent
            # phases need no value when forming this real intersection.
            if possible_phases is not None and phase % divisor not in possible_phases:
                continue
            active = evaluate_comparison_at(test, self.ordinal, phase)
            if active is None:
                return False
            if active:
                residues.append(literal(phase))
        self.intersect(period, tuple(residues))
        return True

    def reduce(self, test):
        if self.invariant(test.lhs) and self.invariant(test.rhs):
            self.add_invariant(test)
            return
        if self.residue_test(test):
            return
        part = self.linear(expression("sub", test.lhs, test.rhs))
        if part is None or part[0] not in {-1, 0, 1}:
            self.pending.append(test)
            return
        coefficient, offset = part
        if coefficient == 0:
            self.add_invariant(Comparison(offset, test.predicate, literal(0)))
            return
        predicate = test.predicate
        if coefficient == -1:
            predicate = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[
                predicate
            ]
        if predicate == "!=":
            self.pending.append(
                Comparison(
                    self.ordinal,
                    "!=",
                    self.constant(expression("mul", literal(-coefficient), offset)),
                )
            )
            return
        negative_count = expression("mul", literal(-1), self.count)
        if coefficient == 1:
            clipped = (
                offset
                if self.proves(Comparison(offset, ">=", negative_count), tuple(self.invariants))
                else expression("max", offset, negative_count)
            )
            bound = expression("mul", literal(-1), clipped)
        else:
            bound = expression("min", offset, self.count)
        bound = self.constant(bound)

        def inclusive_limit(value):
            return expression(
                "max",
                literal(0),
                expression(
                    "add",
                    expression("min", value, expression("sub", self.count, literal(1))),
                    literal(1),
                ),
            )

        if predicate in {">", ">="}:
            bound = inclusive_limit(bound) if predicate == ">" else bound
            self.lower = expression("max", self.lower, bound)
        elif predicate in {"<", "<="}:
            self.count = (
                inclusive_limit(bound)
                if predicate == "<="
                else expression("max", literal(0), expression("min", self.count, bound))
            )
        elif predicate == "==":
            self.lower = expression("max", self.lower, bound)
            self.count = inclusive_limit(bound)
            self.single = True
        self.lower = expression("min", self.lower, self.count)

    def normalized(self):
        return _guard(
            tuple(self.invariants)
            + (
                Comparison(self.ordinal, ">=", self.lower),
                Comparison(self.ordinal, "<", self.count),
                _residue_guard(self.ordinal, self.period, self.residues),
            )
        )

    def encoded_assumptions(self):
        # Successfully encoded conjuncts remain valid membership facts even
        # when interval clipping obscures their spelling. Pending predicates
        # never enter their own proof premises.
        return _guard((*self.normalized().comparisons, *self.established)).comparisons

    def reduce_membership(self):
        hints = [
            Comparison(self.ordinal, ">=", literal(0)),
            Comparison(self.ordinal, "<", self.count),
        ]
        for domain in self.notification.domains:
            if domain.loop is not self.loop:
                continue
            residue = literal(domain.residue) if type(domain.residue) is int else domain.residue
            residue = substitute(residue, self.parameter_bindings)
            hints.extend(
                (
                    Comparison(self.ordinal, ">=", literal(max(domain.first, domain.distance))),
                    Comparison(
                        self.ordinal, "<", substitute(domain.count, self.parameter_bindings)
                    ),
                    Comparison(
                        expression("rem", self.ordinal, literal(domain.period)), "==", residue
                    ),
                )
            )
        # Domain metadata may guide a proof, but cannot discard real emitted
        # occurrences. In particular the actual source's N remains a premise.
        for hint in hints:
            if not self.proves(hint, self.actual):
                self.fail("notification domain is not proved by the actual endpoint activation")
        for test in self.actual:
            if self.invariant(test.lhs) and self.invariant(test.rhs):
                self.add_invariant(test)
        for test in dict.fromkeys((*hints, *self.actual)):
            # Avoid rebuilding interval clamps already implied by encoded
            # facts. Periodic predicates use their finite-set intersection
            # directly; expanding a normalized H(g) to rediscover one of
            # those comparisons can exceed the ordinary scalar budget.
            if (
                period_of(test.lhs, self.ordinal, self.invariant) is None
                or period_of(test.rhs, self.ordinal, self.invariant) is None
            ) and self.proves(test, self.encoded_assumptions()):
                self.established.append(test)
                continue
            pending_count = len(self.pending)
            self.reduce(test)
            if len(self.pending) == pending_count:
                self.established.append(test)
        while self.pending:
            pending, self.pending = self.pending, []
            changed = False
            for test in pending:
                assumptions = self.encoded_assumptions()
                if self.proves(test, assumptions):
                    self.established.append(test)
                    changed = True
                elif test.predicate == "!=" and test.lhs == self.ordinal:
                    if self.proves(Comparison(test.rhs, "==", self.lower), assumptions):
                        self.lower = expression(
                            "add",
                            expression(
                                "min", self.lower, expression("sub", self.count, literal(1))
                            ),
                            literal(1),
                        )
                        self.established.append(test)
                        changed = True
                    elif self.proves(
                        Comparison(test.rhs, "==", expression("sub", self.count, literal(1))),
                        assumptions,
                    ):
                        self.count = expression(
                            "max", literal(0), expression("sub", self.count, literal(1))
                        )
                        self.lower = expression("min", self.lower, self.count)
                        self.established.append(test)
                        changed = True
                    else:
                        self.pending.append(test)
                else:
                    self.pending.append(test)
            if self.pending and not changed:
                raise WindowPartitionRequired(
                    f"{self.notification.name}: loop-varying guard lacks an exact interval/residue membership proof",
                    tests=tuple(self.pending),
                    kind=ErrorKind.UNSUPPORTED,
                    phase="event-window-membership",
                    location=str(self.loop.location),
                    witness=tuple(
                        [f"clock={self.ordinal.text()}"]
                        + [
                            f"unresolved: {test.lhs.text()} {test.predicate} {test.rhs.text()}"
                            for test in self.pending
                        ]
                        + [
                            (
                                "modular sets require statically evaluable periodic phases; "
                                "a dynamic residue supports singleton congruence only"
                            )
                        ]
                    ),
                )

    def finish(self):
        if not self.invariant(self.count):
            self.fail("true loop count is not proved invariant over the actual loop")
        self.definedness()
        if any(_constant_false(test) for test in self.actual):
            # Parameter specialization may make the ordered prefix impossible.
            # Neither later source expressions nor advisory domain selections
            # exist on this case, so they must not be constructed or evaluated.
            self.residues = ()
            self.invariants.append(Comparison(literal(0), "==", literal(1)))
        else:
            self.reduce_membership()
        delta = literal(0)
        if self.residues and self.period != 1:
            position = expression("rem", self.lower, literal(self.period))
            delta = _minimum(
                _difference_mod(residue, position, self.period) for residue in self.residues
            )
        span = expression("sub", self.count, self.lower)
        # An absent alignment may lie beyond INT64_MAX. Its clipped sentinel
        # remains within count even during eager constant folding; nonempty
        # cases retain their exact aligned member.
        safe_delta = expression("min", delta, span) if self.residues and self.period != 1 else delta
        first = expression("add", self.lower, safe_delta)
        invariants = _guard(self.invariants)
        _, maximum = self.bounds(self.count, invariants.comparisons)
        minimum, _ = self.bounds(first, invariants.comparisons)
        minimum = max(0, minimum or 0)  # Alignment above proves first >= 0.
        upper = (
            None
            if maximum is None
            else max(0, (maximum - minimum + self.period - 1) // self.period) * len(self.residues)
        )
        if not self.residues:
            upper = 0
        if self.single:
            upper = 1 if upper is None else min(1, upper)
        return EventWindow(
            self.loop,
            first,
            self.count,
            self.period,
            self.residues,
            upper,
            self.single,
            ordinal=self.ordinal,
            lower=self.lower,
            membership=_guard(self.actual),
            interval=self.normalized(),
            invariants=invariants,
            nonempty=_guard(
                invariants.comparisons
                + (() if self.residues else (Comparison(literal(0), "==", literal(1)),))
                + (
                    Comparison(self.lower, "<", self.count),
                    Comparison(delta, "<", span),
                )
            ),
        )


def build_event_window(
    notification: LogicalNotification,
    loop: Operation,
    *,
    ordinal: Expr,
    count: Expr,
    guards: tuple[Comparison, ...],
    facts: Mapping[str, Fact],
    proves: Proof,
    bounds: Bounds,
    invariant: Callable[[Expr], bool],
    ordered_definedness: OrderedDefinedness,
    parameter_bindings: Mapping[Expr, Expr] | None = None,
) -> EventWindow:
    """Establish one concrete fiber, with optional uniform parameter specialization.

    The caller retains the parameter case that authorizes its bindings. The
    same substitution applies to actual guards, source count and domain hints.
    This builder never specializes a generation or creates parameter families.
    """
    return _Builder(
        notification,
        loop,
        ordinal,
        count,
        guards,
        facts,
        proves,
        bounds,
        invariant,
        {} if parameter_bindings is None else parameter_bindings,
        ordered_definedness=ordered_definedness,
    ).finish()

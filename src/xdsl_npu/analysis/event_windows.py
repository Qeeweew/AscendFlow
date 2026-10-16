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
from ..indexing import normalize
from ..scalar import INT64_MAX, Expr, Fact, expression, literal, substitute
from .structured_protocols import LogicalNotification
from .structured_schedule import ActionGuard, Comparison

Proof = Callable[[Comparison, tuple[Comparison, ...]], bool]
Bounds = Callable[[Expr, tuple[Comparison, ...]], tuple[int | None, int | None]]


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
    residue: Expr
    upper: int | None
    single: bool = False
    ordinal: Expr = field(kw_only=True)
    lower: Expr = field(kw_only=True)
    membership: ActionGuard = field(kw_only=True)
    interval: ActionGuard = field(kw_only=True)
    invariants: ActionGuard = field(kw_only=True)
    nonempty: ActionGuard = field(kw_only=True)

    def first_last(self) -> tuple[Expr, Expr]:
        """Both values are members under ``nonempty``; otherwise evaluate neither."""
        if self.single or self.upper == 0:
            return self.first, self.first
        last = expression("sub", self.count, literal(1))
        if self.period != 1:
            last = expression(
                "sub",
                last,
                expression("rem", expression("sub", last, self.residue), literal(self.period)),
            )
        return self.first, last

    def project_exists(self) -> WindowExistence:
        """Exact existential elimination of this clock, retaining all parameters."""
        return WindowExistence(self.loop, self.ordinal, self.membership, self.nonempty)

    def rank(self) -> Expr:
        """Zero-based member rank, evaluated only under ``membership``."""
        if self.single:
            return literal(0)
        return expression(
            "floordiv", expression("sub", self.ordinal, self.first), literal(self.period)
        )

    def next_same_id(self, width: int) -> Expr:
        """Mathematical lifetime index; this expression requests no runtime work.

        A successor exists only under ``same_id_guard``. Emitted event IDs use
        ``rank`` at an actual member, rather than executing this future index.
        """
        require(type(width) is int and width > 0, "event ring width must be positive")
        return expression("add", self.ordinal, literal(width * self.period))

    def same_id_guard(self, width: int) -> ActionGuard:
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
    """Canonical (a-b) mod p for 0<=a<p and 0<=b<=p, without a+p."""
    if period == 1:
        return literal(0)
    if a.constant is not None and b.constant is not None:
        return literal((a.constant - b.constant) % period)
    if b.constant == 0:
        return a
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
        self, notification, loop, ordinal, count, guards, facts, proves, bounds, invariant
    ):
        self.notification, self.loop, self.ordinal = notification, loop, ordinal
        self.actual = tuple(dict.fromkeys(guards))
        self.facts, self.proves, self.bounds, self.invariant = facts, proves, bounds, invariant
        self.lower, self.count = literal(0), _normal(count, facts)
        self.period, self.residue, self.single = 1, literal(0), False
        self.invariants, self.pending = [], []

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
        prefix = []

        def check(value):
            for child in value.args:
                if isinstance(child, Expr):
                    check(child)
            if value.kind in {"floordiv", "ceildiv", "rem"}:
                a, b = value.args
                for test in (Comparison(a, ">=", literal(0)), Comparison(b, ">", literal(0))):
                    if not self.proves(test, tuple(prefix)):
                        self.fail(
                            "unsigned arithmetic lacks an ordered activation prefix proof",
                            witness=(
                                f"required: {test.lhs.text()} {test.predicate} {test.rhs.text()}",
                                "ordered prefix: "
                                + "; ".join(
                                    f"{prior.lhs.text()} {prior.predicate} {prior.rhs.text()}"
                                    for prior in prefix
                                ),
                            ),
                        )

        for test in self.actual:
            check(test.lhs)
            check(test.rhs)
            prefix.append(test)

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

    def congruence(self, period, residue):
        if type(period) is not int or not 0 < period <= INT64_MAX:
            self.fail("event residue period is not a positive int64 constant")
        self.add_invariant(Comparison(residue, ">=", literal(0)))
        self.add_invariant(Comparison(residue, "<", literal(period)))
        # Invalid residues make the original congruence empty. Keeping that
        # invariant permits safe guarded arithmetic for the remaining window.
        if residue.constant is not None:
            residue = literal(residue.constant % period)
        p, r = self.period, self.residue
        if period % p == 0:
            self.add_invariant(Comparison(self.modulo(residue, p), "==", r))
            self.period, self.residue = period, residue
        elif p % period == 0:
            self.add_invariant(Comparison(self.modulo(r, period), "==", residue))
        else:
            combined, divisor = lcm(p, period), gcd(p, period)
            if combined > INT64_MAX:
                self.fail("event residue intersection exceeds int64 period")
            if r.constant is not None and residue.constant is not None:
                difference = residue.constant - r.constant
                self.add_invariant(Comparison(literal(difference % divisor), "==", literal(0)))
                modulus = period // divisor
                step = difference // divisor * pow(p // divisor, -1, modulus) % modulus
                self.period, self.residue = combined, literal((r.constant + p * step) % combined)
                return
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
            self.period, self.residue = (
                combined,
                expression(
                    "rem",
                    expression("add", r, expression("mul", literal(p), step)),
                    literal(combined),
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
            if period is None or part is None or part[0] not in {-1, 1}:
                continue
            if predicate == "!=":
                if period != 2 or target.constant not in {0, 1}:
                    continue
                target = literal(1 - target.constant)
            elif predicate != "==":
                continue
            self.add_invariant(Comparison(target, ">=", literal(0)))
            self.add_invariant(Comparison(target, "<", literal(period)))
            # x+c == r mod p, or -x+c == r mod p. Canonical modular
            # arithmetic avoids taking an unsigned remainder of r-c.
            offset = self.modulo(part[1], period)
            a, b = (target, offset) if part[0] == 1 else (offset, target)
            self.congruence(period, _difference_mod(a, b, period))
            return True
        return False

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
                Comparison(
                    expression("rem", self.ordinal, literal(self.period)), "==", self.residue
                ),
            )
        )

    def finish(self):
        if not self.invariant(self.count):
            self.fail("true loop count is not proved invariant over the actual loop")
        self.definedness()
        hints = [
            Comparison(self.ordinal, ">=", literal(0)),
            Comparison(self.ordinal, "<", self.count),
        ]
        for domain in self.notification.domains:
            if domain.loop is not self.loop:
                continue
            residue = literal(domain.residue) if type(domain.residue) is int else domain.residue
            hints.extend(
                (
                    Comparison(self.ordinal, ">=", literal(max(domain.first, domain.distance))),
                    Comparison(self.ordinal, "<", domain.count),
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
            self.reduce(test)
        while self.pending:
            pending, self.pending = self.pending, []
            changed = False
            for test in pending:
                assumptions = self.normalized().comparisons
                if self.proves(test, assumptions):
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
                        changed = True
                    elif self.proves(
                        Comparison(test.rhs, "==", expression("sub", self.count, literal(1))),
                        assumptions,
                    ):
                        self.count = expression(
                            "max", literal(0), expression("sub", self.count, literal(1))
                        )
                        self.lower = expression("min", self.lower, self.count)
                        changed = True
                    else:
                        self.pending.append(test)
                else:
                    self.pending.append(test)
            if self.pending and not changed:
                self.fail(
                    "loop-varying guard lacks an exact interval/residue membership proof",
                    witness=tuple(
                        f"unresolved: {test.lhs.text()} {test.predicate} {test.rhs.text()}"
                        for test in self.pending
                    ),
                )
        delta = literal(0)
        if self.period != 1:
            delta = _difference_mod(
                self.residue,
                expression("rem", self.lower, literal(self.period)),
                self.period,
            )
        first = expression("add", self.lower, delta)
        invariants = _guard(self.invariants)
        _, maximum = self.bounds(self.count, invariants.comparisons)
        minimum, _ = self.bounds(first, invariants.comparisons)
        minimum = max(0, minimum or 0)  # Alignment above proves first >= 0.
        upper = (
            None
            if maximum is None
            else max(0, (maximum - minimum + self.period - 1) // self.period)
        )
        if self.single:
            upper = 1 if upper is None else min(1, upper)
        return EventWindow(
            self.loop,
            first,
            self.count,
            self.period,
            self.residue,
            upper,
            self.single,
            ordinal=self.ordinal,
            lower=self.lower,
            membership=_guard(self.actual),
            interval=self.normalized(),
            invariants=invariants,
            nonempty=_guard(
                invariants.comparisons
                + (
                    Comparison(self.lower, "<", self.count),
                    Comparison(delta, "<", expression("sub", self.count, self.lower)),
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
) -> EventWindow:
    """Establish one exact activation fiber, or reject its missing proof."""
    return _Builder(
        notification, loop, ordinal, count, guards, facts, proves, bounds, invariant
    ).finish()

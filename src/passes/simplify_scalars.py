"""Scoped integer simplification after admission and notification insertion.

Exact SSA expressions reuse dominating definitions. Total integer invariants
move to their earliest legal preheader; guarded division and overflow-prone
arithmetic need destination-scope premises before any new evaluation. Optional
source facts and actual control supply local ranges. Native, address,
notification and control operations keep their identities.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from math import gcd
from types import MappingProxyType
from typing import Self

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import IsolatedFromAbove, Pure
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern

from ..diagnostics import CompileError, ErrorKind
from ..dialects.physical import CoreIdOp, RuntimeSizeOp
from ..dialects.structured import ActorDomainOp, PipelineLoopOp, PipelineStageOp
from ..scalar import Expr, Fact, literal
from ..symbols import is_coordinate, symbol_name


@dataclass
class ScalarFacts:
    """Optional source ranges; absence never prevents source execution."""

    limits: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    _range_data: dict = field(init=False, repr=False, compare=False)
    _observed: set = field(default_factory=set, init=False, repr=False, compare=False)
    revision: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._range_data = dict(self.ranges)
        self.ranges = MappingProxyType(self._range_data)

    def bound(self, value):
        self._observed.add(value)
        return self.ranges.get(value)

    def bind_range(self, value, bounds):
        if self.ranges.get(value) != bounds:
            self._range_data[value] = bounds
            if value in self._observed:
                self.revision += 1

    @classmethod
    def from_control(cls, control=None):
        result = cls()
        if control is None:
            return result
        from ..indexing import IndexRanges, nonnegative, normalize_or_identity, polynomial

        facts = {
            name: Fact(
                fact.minimum,
                fact.multiple,
                min(
                    (1 << 63) - 1 if fact.maximum is None else fact.maximum,
                    result.limits.get(symbol_name(name), (1 << 63) - 1),
                ),
            )
            for name, fact in control.environment.facts.items()
        }
        cache = {}

        def bounds(expr):
            if expr in cache:
                return cache[expr]
            if expr.constant is not None:
                answer = (expr.constant, expr.constant)
            elif expr.kind == "argument":
                fact = facts.get(expr.args[0])
                answer = None if fact is None else (fact.minimum, fact.maximum)
            else:
                a, b = expr.args
                left, right = bounds(a), bounds(b)
                answer = None
                if left is not None and right is not None:
                    al, ah = left
                    bl, bh = right
                    if expr.kind == "add":
                        answer = al + bl, ah + bh
                    elif expr.kind == "mul":
                        products = al * bl, al * bh, ah * bl, ah * bh
                        answer = min(products), max(products)
                    elif expr.kind in {"min", "max"}:
                        f = min if expr.kind == "min" else max
                        answer = f(al, bl), f(ah, bh)
                    elif expr.kind == "rem" and bl > 0:
                        answer = 0, min(ah, bh - 1)
                    elif expr.kind in {"floordiv", "ceildiv"} and al >= 0 and bl > 0:
                        ceiling = expr.kind == "ceildiv"
                        answer = (
                            ((al + bh - 1) // bh, (ah + bl - 1) // bl)
                            if ceiling
                            else (al // bh, ah // bl)
                        )
                        if expr.kind == "floordiv" and b.constant is None:
                            numerator, denominator = polynomial(a, facts), polynomial(b, facts)
                            atom = next((term for term in denominator if term), None)
                            if atom is not None:
                                scale = Fraction(numerator.get(atom, 0), denominator[atom])
                                if (
                                    scale.denominator == 1
                                    and scale >= 0
                                    and all(
                                        numerator.get(term, 0) == scale * denominator.get(term, 0)
                                        for term in numerator.keys() | denominator.keys()
                                        if term
                                    )
                                ):
                                    remainder = numerator.get((), 0) - scale * denominator.get(
                                        (), 0
                                    )
                                    if 0 <= remainder < bl:
                                        answer = int(scale), int(scale)
                                    elif scale >= 1 and -bl <= remainder < 0:
                                        answer = int(scale) - 1, int(scale) - 1
            cache[expr] = answer
            return answer

        symbolic = {}
        active_ranges = IndexRanges(facts)
        for operation, frame in control.frames.items():
            for value in (
                *operation.results,
                *(
                    arg
                    for region in operation.regions
                    for block in region.blocks
                    for arg in block.args
                ),
            ):
                if not isinstance(value.type, IndexType):
                    continue
                try:
                    domains = control.result_frames.get(value, frame).domains()
                    expr = control.environment.expression(value)
                    if not any(is_coordinate(name) for name in expr.variables):
                        uniform_range = bounds(expr)
                        if uniform_range is not None:
                            low, high = uniform_range
                            if -(1 << 63) <= low <= high < (1 << 63):
                                result.bind_range(value, uniform_range)
                            continue
                    key = expr, domains
                    if key not in symbolic:
                        low, high = active_ranges.interval(expr, domains)
                        if expr.kind == "floordiv":
                            numerator, divisor = expr.args
                            nl, nh = active_ranges.interval(numerator, domains)
                            gap = Expr("add", (divisor, Expr("mul", (literal(-1), nh))))
                            if nonnegative(nl, facts) and nonnegative(
                                Expr("add", (gap, literal(-1))), facts
                            ):
                                low = high = literal(0)
                        low_range, high_range = bounds(low), bounds(high)
                        symbolic[key] = (
                            bounds(normalize_or_identity(low, facts))
                            if low_range is None
                            else low_range,
                            bounds(normalize_or_identity(high, facts))
                            if high_range is None
                            else high_range,
                        )
                    left, right = symbolic[key]
                    if left is not None and right is not None:
                        low, high = max(-(1 << 63), left[0]), min((1 << 63) - 1, right[1])
                        if low <= high:
                            result.bind_range(value, (low, high))
                except CompileError as error:
                    if error.kind is not ErrorKind.ANALYSIS_LIMIT:
                        raise
                    # Optional precision may exhaust its algebra budget. It
                    # does not determine executable source admission.
                    continue
        return result


_ARITH_OPERATIONS = tuple(arith.Arith.operations)
_UNSIGNED = (arith.DivUIOp, arith.CeilDivUIOp, arith.RemUIOp)
_SIGNS = ({0}, {-1, 1}, {-1}, {-1, 0}, {1}, {0, 1})


@cache
def _arithmetic_type(operation_type):
    """Static dialect membership and class traits, never program facts."""
    return issubclass(operation_type, _ARITH_OPERATIONS) and (
        operation_type.has_trait(Pure) or issubclass(operation_type, _UNSIGNED)
    )


@cache
def _integer_type(attribute_type):
    return issubclass(attribute_type, (IntegerType, IndexType))


def _constant(value):
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, IntegerAttr):
        return owner.value.value.data
    return None


def _modulo(value):
    op = value.owner
    if isinstance(op, arith.RemUIOp):
        period = _constant(op.rhs)
        return (period, op.lhs) if period is not None and period > 0 else None
    if isinstance(op, arith.AndIOp):
        for mask, numerator in ((op.lhs, op.rhs), (op.rhs, op.lhs)):
            constant = _constant(mask)
            if constant is not None and constant >= 0 and constant & (constant + 1) == 0:
                return constant + 1, numerator
    return None


def _limits(value):
    width = 64 if isinstance(value.type, IndexType) else value.type.width.data
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _scalar_arithmetic(op):
    return (
        _arithmetic_type(type(op))
        and len(op.results) == 1
        and all(_integer_type(type(value.type)) for value in (*op.operands, *op.results))
    )


def _key(op):
    operands = tuple(op.operands)
    if isinstance(op, (arith.AddiOp, arith.MuliOp, arith.AndIOp, arith.OrIOp)):
        operands = tuple(sorted(operands, key=id))
    return (
        type(op),
        operands,
        tuple(sorted(op.properties.items())),
        tuple(sorted(op.attributes.items())),
        tuple(value.type for value in op.results),
    )


@dataclass
class _Scope:
    available: dict[tuple, SSAValue] = field(default_factory=dict)
    ranges: dict[SSAValue, tuple[int, int]] = field(default_factory=dict)
    relations: dict[tuple[SSAValue, int, SSAValue], bool] = field(default_factory=dict)
    defined: set[SSAValue] = field(default_factory=set)
    placements: tuple = ()
    parent: Self | None = None
    conditions: dict[SSAValue, bool] = field(default_factory=dict)
    facts: ScalarFacts | None = None
    affines: dict = field(default_factory=dict)
    canonical: dict = field(default_factory=dict)
    multiples: dict = field(default_factory=dict)
    _range_data: dict = field(init=False, repr=False)
    _intervals: dict = field(default_factory=dict, init=False, repr=False)
    _refinements: dict = field(default_factory=dict, init=False, repr=False)
    _revision: int = field(default=0, init=False, repr=False)
    _clock: list[int] = field(default_factory=lambda: [0], init=False, repr=False)
    _observed: tuple = field(default=(), init=False, repr=False)
    _inputs: tuple = field(default=(), init=False, repr=False)
    premises: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._range_data = dict(self.ranges)
        self.ranges = MappingProxyType(self._range_data)
        self.premises.update(self.relations)
        if self.parent is not None:
            self._clock = self.parent._clock

    def _invalidate(self):
        self._intervals.clear()
        self._refinements.clear()
        self.affines.clear()
        self._revision += 1

    def refresh(self):
        # The clock only reports a possible input change. It never invalidates
        # sibling scopes: the actual parent/facts revisions below decide that.
        # Unchanged queries avoid recursively polling the same ancestor chain.
        facts_revision = () if self.facts is None else (id(self.facts), self.facts.revision)
        observed = self._clock[0], facts_revision
        if observed == self._observed:
            return
        if self.parent is not None:
            self.parent.refresh()
        inputs = -1 if self.parent is None else self.parent._revision, facts_revision
        if inputs != self._inputs:
            self._invalidate()
            self._inputs = inputs
        self._observed = observed

    def _changed(self):
        self._invalidate()
        self._clock[0] += 1

    def bind_range(self, value, bounds):
        if self.ranges.get(value) != bounds:
            self._range_data[value] = bounds
            if (
                value in self._intervals
                and self._intervals[value] != bounds
                or value not in self._intervals
                and value in self._refinements
            ):
                self._changed()

    def child(self):
        return _Scope(placements=self.placements, parent=self, facts=self.facts)

    def definition(self, key):
        return self.available.get(key) or (self.parent.definition(key) if self.parent else None)

    def has_definition(self, value):
        return (
            value in self.defined or self.parent is not None and self.parent.has_definition(value)
        )

    def known_relation(self, key):
        if key in self.relations:
            return self.relations[key]
        return self.parent.known_relation(key) if self.parent else None

    def condition(self, value):
        if value in self.conditions:
            return self.conditions[value]
        return self.parent.condition(value) if self.parent else None

    def interval(self, value):
        self.refresh()
        if (cached := self._intervals.get(value)) is not None:
            return cached
        constant = _constant(value)
        if constant is not None:
            return constant, constant
        result = self.ranges.get(value)
        if result is None:
            result = self.parent.interval(value) if self.parent else _limits(value)
            # A dominating expression can become tighter under a child guard
            # without being recomputed there (for example t-s under t>=s).
            if (
                isinstance(value.owner, Operation)
                and _scalar_arithmetic(value.owner)
                and (self.parent is None or self._refines(value))
            ):
                refined = _bounded_interval(value.owner, self)
                if refined is not None:
                    result = max(result[0], refined[0]), min(result[1], refined[1])
        if self.facts is not None and (bound := self.facts.bound(value)) is not None:
            low, high = bound
            result = max(result[0], low), min(result[1], high)
        self._intervals[value] = result
        return result

    def _refines(self, value):
        """Only changed scalar inputs require rebuilding an inherited range."""
        if value not in self._refinements:
            if value in self.ranges:
                self._refinements[value] = self.ranges[value] != self.parent.interval(value)
            else:
                owner = value.owner
                self._refinements[value] = (
                    isinstance(owner, Operation)
                    and _scalar_arithmetic(owner)
                    and any(self._refines(operand) for operand in owner.operands)
                )
        return self._refinements[value]

    def affine_definition(self, key):
        return self.canonical.get(key) or (
            self.parent.affine_definition(key) if self.parent else None
        )

    def multiple(self, value):
        constant = _constant(value)
        if constant is not None:
            return abs(constant)
        if value in self.multiples:
            return self.multiples[value]
        return self.parent.multiple(value) if self.parent else 1

    def relation(self, lhs, predicate, rhs, truth=True):
        self.premises[lhs, predicate, rhs] = truth
        if not self.relations:
            self._changed()
        if predicate >= 6:
            # An unsigned comparison cannot establish a signed range.
            self.relations[lhs, predicate, rhs] = truth
            return
        possible = _SIGNS[predicate] if truth else {-1, 0, 1} - _SIGNS[predicate]
        for a, b, signs in ((lhs, rhs, possible), (rhs, lhs, {-x for x in possible})):
            for index, allowed in enumerate(_SIGNS):
                if signs <= allowed:
                    self.relations[a, index, b] = True
                elif signs.isdisjoint(allowed):
                    self.relations[a, index, b] = False
            low, high = self.interval(a)
            other_low, other_high = self.interval(b)
            if signs == {0}:
                low, high = max(low, other_low), min(high, other_high)
            elif signs <= {-1, 0}:
                high = min(high, other_high - int(0 not in signs))
            elif signs <= {0, 1}:
                low = max(low, other_low + int(0 not in signs))
            if low <= high:
                self.bind_range(a, (low, high))

    def assume(self, condition, truth):
        self.conditions[condition] = truth
        owner = condition.owner
        if isinstance(owner, arith.CmpiOp):
            self.relation(owner.lhs, owner.predicate.value.data, owner.rhs, truth)
        elif (
            isinstance(owner, arith.AndIOp)
            and truth
            or isinstance(owner, arith.OrIOp)
            and not truth
        ):
            for operand in owner.operands:
                self.assume(operand, truth)

    def truth(self, value):
        constant = _constant(value)
        if constant is not None:
            return bool(constant)
        known = self.condition(value)
        if known is not None:
            return known
        return self.comparison(value.owner) if isinstance(value.owner, arith.CmpiOp) else None

    def comparison(self, op):
        predicate = op.predicate.value.data
        if op.lhs is op.rhs:
            return predicate in {0, 3, 5, 7, 9}
        if (known := self.known_relation((op.lhs, predicate, op.rhs))) is not None:
            return known
        a, b = self.interval(op.lhs), self.interval(op.rhs)
        if predicate >= 6:
            if a[0] < 0 or b[0] < 0:
                return None
            predicate -= 4
        possible = set()
        if a[0] < b[1]:
            possible.add(-1)
        if max(a[0], b[0]) <= min(a[1], b[1]):
            possible.add(0)
        if a[1] > b[0]:
            possible.add(1)
        if possible <= _SIGNS[predicate]:
            return True
        if possible.isdisjoint(_SIGNS[predicate]):
            return False
        return None


def _bounded_interval(op, scope):
    """Transfer only bounds whose complete signed result range cannot wrap."""
    if isinstance(op, arith.SelectOp):
        left, right = (scope.interval(v) for v in op.operands[1:])
        return min(left[0], right[0]), max(left[1], right[1])
    if isinstance(op, (arith.AndIOp, arith.OrIOp)) and (modulo := _modulo(op.result)) is None:
        return _limits(op.result)
    if len(op.operands) != 2:
        return None
    (al, ah), (bl, bh) = (scope.interval(value) for value in op.operands)
    if isinstance(op, arith.AddiOp):
        result = al + bl, ah + bh
    elif isinstance(op, arith.SubiOp):
        result = al - bh, ah - bl
    elif isinstance(op, arith.MuliOp):
        products = al * bl, al * bh, ah * bl, ah * bh
        result = min(products), max(products)
    elif isinstance(op, arith.MinSIOp):
        result = min(al, bl), min(ah, bh)
    elif isinstance(op, arith.MaxSIOp):
        result = max(al, bl), max(ah, bh)
    elif isinstance(op, arith.AndIOp) and (modulo := _modulo(op.result)) is not None:
        result = 0, modulo[0] - 1
    elif isinstance(op, arith.RemUIOp) and bl > 0:
        # A positive signed divisor bounds the unsigned remainder even when
        # the dividend's sign bit is unknown. Its result cannot wrap.
        result = 0, min(ah, bh - 1) if al >= 0 else bh - 1
    elif isinstance(op, _UNSIGNED) and al >= 0 and bl > 0:
        if isinstance(op, arith.CeilDivUIOp):
            result = (al + bh - 1) // bh, (ah + bl - 1) // bl
        else:
            result = al // bh, ah // bl
    else:
        return None
    low, high = _limits(op.results[0])
    return result if low <= result[0] <= result[1] <= high else None


def _interval(op, scope):
    result = _bounded_interval(op, scope) or _limits(op.results[0])
    if scope.facts is not None and op.results[0] in scope.facts.ranges:
        low, high = scope.facts.ranges[op.results[0]]
        return max(low, result[0]), min(high, result[1])
    return result


def _affine(value, scope):
    """One exact, scope-safe linear view shared by integers and predicates."""
    scope.refresh()
    if value in scope.affines:
        return scope.affines[value]
    owner, constant = value.owner, _constant(value)
    if constant is not None:
        result = {}, constant
    elif (
        isinstance(owner, (arith.AddiOp, arith.SubiOp, arith.MuliOp))
        and _bounded_interval(owner, scope) is not None
    ):
        if isinstance(owner, arith.MuliOp):
            a, b = owner.operands
            scale, operand = (_constant(a), b) if _constant(a) is not None else (_constant(b), a)
            if scale is None:
                result = {value: 1}, 0
            else:
                terms, bias = _affine(operand, scope)
                result = {v: c * scale for v, c in terms.items() if c * scale}, bias * scale
        else:
            left, a = _affine(owner.lhs, scope)
            right, b = _affine(owner.rhs, scope)
            sign = -1 if isinstance(owner, arith.SubiOp) else 1
            terms = dict(left)
            for v, c in right.items():
                terms[v] = terms.get(v, 0) + sign * c
            result = {v: c for v, c in terms.items() if c}, a + sign * b
    else:
        result = {value: 1}, 0
    scope.affines[value] = result
    return result


def _linear_operation(op, scope):
    """Materialize a two-term difference directly from its shared linear view."""
    if not isinstance(op, (arith.AddiOp, arith.MuliOp)):
        return op
    terms, bias = _affine(op.results[0], scope)
    if bias or len(terms) != 2 or set(terms.values()) != {-1, 1}:
        return op
    positive = next(value for value, coefficient in terms.items() if coefficient == 1)
    negative = next(value for value, coefficient in terms.items() if coefficient == -1)
    candidate = arith.SubiOp(positive, negative)
    if _bounded_interval(candidate, scope) is None:
        candidate.erase()
        return op
    candidate.location = op.location
    scope.affines[candidate.result] = terms, bias
    return candidate


def _place_invariant(op, scope, key):
    """Move only total arithmetic with operands available before the region.

    A new evaluation can occur on an empty loop or an untaken branch. Its
    signed intermediates and unsigned operands must therefore be legal in
    the destination scope, independently of any premises inside the region.
    """
    for anchor, destination in scope.placements:
        if not all(destination.has_definition(operand) for operand in op.operands):
            continue
        if isinstance(op, _UNSIGNED) and (
            destination.interval(op.lhs)[0] < 0 or destination.interval(op.rhs)[0] <= 0
        ):
            continue
        if (
            not isinstance(op, (arith.ConstantOp, arith.CmpiOp))
            and _bounded_interval(op, destination) is None
        ):
            continue
        if (prior := destination.definition(key)) is not None:
            _replace(op, prior, scope)
            scope.available[key] = prior
            scope.defined.add(prior)
        else:
            local_range = _interval(op, scope)
            destination_range = _bounded_interval(op, destination) or _limits(op.results[0])
            op.parent.detach_op(op)
            anchor.parent.insert_op_before(op, anchor)
            result = op.results[0]
            destination.available[key] = scope.available[key] = result
            destination.defined.add(result)
            scope.defined.add(result)
            if not isinstance(op, arith.ConstantOp):
                # A source-domain bound is invalidated when its definition
                # moves. Export the bound of the actual new evaluation domain.
                if scope.facts is not None:
                    scope.facts.bind_range(result, destination_range)
                if destination.facts is not None:
                    destination.facts.bind_range(result, destination_range)
                destination.bind_range(result, destination_range)
                scope.bind_range(result, local_range)
        return True
    return False


def _fold(op, scope):
    if isinstance(op, arith.ConstantOp):
        return None
    if isinstance(op, arith.CmpiOp):
        value = scope.comparison(op)
        return None if value is None else int(value)
    if isinstance(op, arith.SelectOp):
        condition, left, right = op.operands
        constant = scope.truth(condition)
        if constant is not None:
            return left if constant else right
        if left is right:
            return left
    if (
        isinstance(op, (arith.AndIOp, arith.OrIOp))
        and isinstance(op.result.type, IntegerType)
        and op.result.type.width.data == 1
    ):
        # Only i1 logical connectives: a true bit is printed as -1 by xDSL.
        for a, b in ((op.lhs, op.rhs), (op.rhs, op.lhs)):
            if (constant := scope.truth(a)) is not None:
                if isinstance(op, arith.AndIOp):
                    return b if constant else 0
                return 1 if constant else b
    if isinstance(op, (arith.AddiOp, arith.SubiOp)) and _constant(op.rhs) == 0:
        return op.lhs
    if isinstance(op, arith.MuliOp) and _constant(op.rhs) == 1:
        return op.lhs
    if isinstance(op, (arith.MinSIOp, arith.MaxSIOp)):
        (al, ah), (bl, bh) = (scope.interval(value) for value in op.operands)
        if ah <= bl:
            return op.lhs if isinstance(op, arith.MinSIOp) else op.rhs
        if bh <= al:
            return op.rhs if isinstance(op, arith.MinSIOp) else op.lhs
    if isinstance(op, _UNSIGNED):
        (al, ah), (bl, bh) = (scope.interval(value) for value in op.operands)
        if bl == bh == 1:
            # Exact bit-vector identities. Original DSL unsigned premises
            # have already been checked, before this final SSA pass.
            return 0 if isinstance(op, arith.RemUIOp) else op.lhs
        if (
            isinstance(op, arith.RemUIOp)
            and bl == bh
            and bl > 0
            and scope.multiple(op.lhs) % bl == 0
            and al >= 0
        ):
            return 0
        if al >= 0 and bl == bh and ah < bl:
            if isinstance(op, arith.RemUIOp):
                return op.lhs
            if isinstance(op, arith.DivUIOp):
                return 0
    low, high = _interval(op, scope)
    if low == high:
        return low
    return None


def _remainder_basis(value, scope):
    """An exact affine function of one already defined remainder SSA."""
    constant = _constant(value)
    if constant is not None:
        return None, 0, constant
    op = value.owner
    if isinstance(op, arith.RemUIOp):
        return value, 1, 0
    if (
        not isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp))
        or _bounded_interval(op, scope) is None
    ):
        return None
    if isinstance(op, arith.MuliOp):
        for scale, rest in ((op.lhs, op.rhs), (op.rhs, op.lhs)):
            constant = _constant(scale)
            basis = _remainder_basis(rest, scope)
            if constant is not None and basis is not None:
                return basis[0], basis[1] * constant, basis[2] * constant
        return None
    left, right = (_remainder_basis(v, scope) for v in op.operands)
    if (
        left is None
        or right is None
        or left[0] is not None
        and right[0] is not None
        and left[0] is not right[0]
    ):
        return None
    sign = -1 if isinstance(op, arith.SubiOp) else 1
    return (
        left[0] if left[0] is not None else right[0],
        left[1] + sign * right[1],
        left[2] + sign * right[2],
    )


def _replace(op, value, scope=None):
    if scope is not None:
        old_low, old_high = _interval(op, scope)
        low, high = scope.interval(value)
        low, high = max(low, old_low), min(high, old_high)
        if low <= high:
            scope.bind_range(value, (low, high))
    op.results[0].replace_all_uses_with(value)
    op.parent.erase_op(op)


def _seed(block, scope):
    scope.defined.update(block.args)
    for value in block.args:
        scope.canonical[frozenset(((value, 1),)), 0, value.type] = value
    owner = block.parent_op()
    if isinstance(owner, ActorDomainOp):
        scope.bind_range(block.args[0], (0, 1 if owner.kind.data == "vector" else 0))
    elif isinstance(owner, scf.ForOp) and (_constant(owner.step) or 0) > 0:
        induction = block.args[0]
        scope.relation(induction, 5, owner.lb)
        scope.relation(induction, 2, owner.ub)
    elif isinstance(owner, PipelineStageOp):
        loop = owner.parent_op()
        ordinal = block.args[0]
        scope.relation(ordinal, 2, loop.count)
        scope.bind_range(ordinal, (0, scope.interval(ordinal)[1]))


def _materialize_constant(op, scope, value, typ):
    constant = arith.ConstantOp.from_int_and_width(value, typ)
    op.parent.insert_op_before(constant, op)
    key = _key(constant)
    if (prior := scope.definition(key)) is not None:
        constant.parent.erase_op(constant)
    elif _place_invariant(constant, scope, key):
        prior = scope.available[key]
    else:
        prior = scope.available.setdefault(key, constant.result)
    scope.defined.add(prior)
    return prior


def _simplify_block(block, scope, runtime_limits):
    _seed(block, scope)
    for op in tuple(block.ops):
        if isinstance(op, RuntimeSizeOp):
            multiple = op.multiple.data
            low, high = _limits(op.value)
            low = max(low, op.minimum.data)
            high = min(high, runtime_limits.get(op.sym_name.data, high))
            scope.bind_range(op.value, ((low + multiple - 1) // multiple * multiple, high))
            scope.multiples[op.value] = multiple
        elif isinstance(op, CoreIdOp):
            root = op.parent_op()
            scope.bind_range(
                op.value, (0, (root.blocks.data if root.blocks is not None else 1) - 1)
            )
        elif _scalar_arithmetic(op):
            normalized = _linear_operation(op, scope)
            if normalized is not op:
                op.parent.insert_op_before(normalized, op)
                if scope.facts is not None and op.results[0] in scope.facts.ranges:
                    scope.facts.bind_range(normalized.result, scope.facts.ranges[op.results[0]])
                _replace(op, normalized.result, scope)
                op = normalized
            if isinstance(op, arith.RemUIOp) and (period := _constant(op.rhs)) and period > 0:
                basis = _remainder_basis(op.lhs, scope)
                if (
                    basis is not None
                    and basis[0] is not None
                    and _constant(basis[0].owner.rhs) == period
                    and scope.interval(op.lhs)[0] >= 0
                ):
                    remainder, scale, bias = basis
                    if scale % period == 1 and bias % period == 0:
                        _replace(op, remainder, scope)
                        continue
                    if scale % period == period - 1 and bias % period == period - 1:
                        bound = _materialize_constant(op, scope, period - 1, op.result.type)
                        complement = arith.SubiOp(bound, remainder)
                        op.parent.insert_op_before(complement, op)
                        _replace(op, complement.result, scope)
                        scope.bind_range(complement.result, (0, period - 1))
                        scope.defined.add(complement.result)
                        scope.available[_key(complement)] = complement.result
                        _place_invariant(complement, scope, _key(complement))
                        continue
                numerator = op.lhs.owner
                if (
                    isinstance(numerator, arith.AddiOp)
                    and _bounded_interval(numerator, scope) is not None
                ):
                    for shift, rest in (
                        (numerator.lhs, numerator.rhs),
                        (numerator.rhs, numerator.lhs),
                    ):
                        constant = _constant(shift)
                        if (
                            constant is not None
                            and constant % period == 0
                            and scope.interval(rest)[0] >= 0
                        ):
                            op.operands = (rest, op.rhs)
                            break
            folded = _fold(op, scope)
            if isinstance(folded, int):
                prior = _materialize_constant(op, scope, folded, op.results[0].type)
                _replace(op, prior, scope)
                continue
            if isinstance(folded, SSAValue):
                _replace(op, folded, scope)
                continue
            form = None
            if isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp)):
                terms, bias = _affine(op.results[0], scope)
                form = frozenset(terms.items()), bias, op.results[0].type
                if not terms:
                    prior = _materialize_constant(op, scope, bias, op.results[0].type)
                    _replace(op, prior, scope)
                    continue
                if len(terms) == 1 and bias == 0 and next(iter(terms.values())) == 1:
                    value = next(iter(terms))
                    if value is not op.results[0]:
                        _replace(op, value, scope)
                        continue
                if (prior := scope.affine_definition(form)) is not None:
                    _replace(op, prior, scope)
                    continue
                if len(terms) == 1 and bias == 0:
                    value, coefficient = next(iter(terms.items()))
                    low, high = _limits(op.results[0])
                    if (
                        value is not op.results[0]
                        and low <= coefficient <= high
                        and isinstance(op, arith.MuliOp)
                        and value not in op.operands
                    ):
                        constant = _materialize_constant(op, scope, coefficient, op.results[0].type)
                        op.operands = (value, constant)
            if isinstance(op, (arith.AddiOp, arith.SubiOp)):
                scope.multiples[op.results[0]] = gcd(*(scope.multiple(v) for v in op.operands))
            elif isinstance(op, arith.MuliOp):
                scope.multiples[op.results[0]] = scope.multiple(op.lhs) * scope.multiple(op.rhs)
            key = _key(op)
            if (prior := scope.definition(key)) is not None:
                if form is not None:
                    scope.canonical[form] = prior
                _replace(op, prior, scope)
                continue
            if _place_invariant(op, scope, key):
                if form is not None:
                    scope.canonical[form] = scope.available[key]
                continue
            scope.available[key] = op.results[0]
            if form is not None:
                scope.canonical[form] = op.results[0]
            if not isinstance(op, arith.ConstantOp):
                scope.bind_range(op.results[0], _interval(op, scope))
                if scope.facts is not None:
                    scope.facts.bind_range(op.results[0], scope.ranges[op.results[0]])
        elif isinstance(op, scf.IfOp) and _constant(op.cond) is None:
            known = scope.condition(op.cond)
            if known is None and isinstance(op.cond.owner, arith.CmpiOp):
                known = scope.comparison(op.cond.owner)
            if known is not None:
                op.operands = (_materialize_constant(op, scope, int(known), op.cond.type),)
        for branch, region in enumerate(op.regions):
            for child in region.blocks:
                nested = (
                    _Scope(facts=scope.facts) if op.has_trait(IsolatedFromAbove) else scope.child()
                )
                if isinstance(op, (scf.ForOp, scf.IfOp, PipelineLoopOp, ActorDomainOp)):
                    nested.placements += ((op, scope),)
                if isinstance(op, scf.IfOp):
                    nested.assume(op.cond, branch == 0)
                _simplify_block(child, nested, runtime_limits)
        scope.defined.update(op.results)


class _PureIntegerArithmetic(RewritePattern):
    def __init__(self):
        self.pattern = GreedyRewritePatternApplier([CanonicalizationRewritePattern()])

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, _ARITH_OPERATIONS) or not op.has_trait(Pure):
            return
        if not all(
            isinstance(value.type, (IntegerType, IndexType))
            for value in (*op.operands, *op.results)
        ):
            return
        self.pattern.match_and_rewrite(op, rewriter)


def _cse_block(block, scope, changed):
    """Merge exact definitions exposed by the final local canonicalization.

    Reuse the admitted ranges and local control premises. Only changed or
    newly introduced definitions need their legal placement checked again.
    """
    _seed(block, scope)
    for op in tuple(block.ops):
        if _scalar_arithmetic(op):
            folded = _fold(op, scope)
            if isinstance(folded, int):
                folded = _materialize_constant(op, scope, folded, op.results[0].type)
            if isinstance(folded, SSAValue):
                _replace(op, folded, scope)
                continue
        if _scalar_arithmetic(op):
            key = _key(op)
            if (prior := scope.definition(key)) is not None:
                _replace(op, prior, scope)
                continue
            if op in changed and _place_invariant(op, scope, key):
                continue
            scope.available[key] = op.results[0]
            if not isinstance(op, arith.ConstantOp):
                scope.bind_range(op.results[0], _interval(op, scope))
        elif isinstance(op, scf.IfOp) and (known := scope.truth(op.cond)) is not None:
            if _constant(op.cond) is None:
                op.operands = (_materialize_constant(op, scope, int(known), op.cond.type),)
        for branch, region in enumerate(op.regions):
            for child in region.blocks:
                nested = (
                    _Scope(facts=scope.facts) if op.has_trait(IsolatedFromAbove) else scope.child()
                )
                if isinstance(op, (scf.ForOp, scf.IfOp, ActorDomainOp)):
                    nested.placements += ((op, scope),)
                if isinstance(op, scf.IfOp):
                    nested.assume(op.cond, branch == 0)
                _cse_block(child, nested, changed)
        scope.defined.update(op.results)


def simplify_scalars(module: ModuleOp, *, control=None, facts=None, finalize=True):
    """Simplify owned arithmetic in place without revisiting execution facts."""
    facts = ScalarFacts.from_control(control) if facts is None else facts
    runtime_limits = facts.limits
    for block in module.body.blocks:
        _simplify_block(block, _Scope(facts=facts), runtime_limits)
    if finalize:
        cleanup_scalar_syntax(module, facts=facts)
    else:
        # The condition pass follows immediately and owns final syntax CSE.
        # Scoped simplification has already folded/shared the current tree;
        # only dead definitions must be removed before packet collection.
        _erase_dead_scalars(module)


def _erase_dead_scalars(module):
    for op in reversed(tuple(module.walk())):
        if _scalar_arithmetic(op) and not op.results[0].uses:
            op.parent.erase_op(op)


def cleanup_scalar_syntax(module, *, canonicalize=True, facts=None, changed=()):
    """Fold/CSE/DCE syntax; reconsider placement only for changed definitions."""
    changed = set(changed)
    if canonicalize:
        previous = {op: _key(op) for op in module.walk() if _scalar_arithmetic(op)}
        PatternRewriteWalker(_PureIntegerArithmetic()).rewrite_module(module)
        changed.update(
            op
            for op in module.walk()
            if _scalar_arithmetic(op) and (op not in previous or _key(op) != previous[op])
        )
    _erase_dead_scalars(module)
    for block in module.body.blocks:
        _cse_block(block, _Scope(facts=facts), changed)
    _erase_dead_scalars(module)

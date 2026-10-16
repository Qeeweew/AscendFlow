"""Proof-only canonical predicates over one bounded integer modulus.

These records describe mathematical congruences. They neither evaluate source
expressions nor establish unsigned division domains or fixed-width safety.
Callers must check the original ordered arithmetic premises before using them.
All polynomial work and unit selection belong to one bounded query; no loop
coordinate or runtime parameter is enumerated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd

from ..diagnostics import ErrorKind, require
from ..scalar import Expr
from .structured_schedule import Comparison

MAX_MODULUS = 64
MAX_TERMS = 64
MAX_WORK = 4096

Monomial = tuple[Expr, ...]
Terms = tuple[tuple[Monomial, int], ...]
Polynomial = dict[Monomial, int]


def _limit(condition, message):
    require(condition, message, kind=ErrorKind.ANALYSIS_LIMIT, phase="modular-predicate")


def _monomial_key(monomial: Monomial):
    return tuple(factor.order_key for factor in monomial)


@dataclass(frozen=True, slots=True)
class ModularAtom:
    """The residue of one integer polynomial, with its constant removed."""

    modulus: int
    terms: Terms

    def __post_init__(self):
        require(type(self.modulus) is int and self.modulus > 0, "invalid modular atom modulus")
        _limit(
            self.modulus <= MAX_MODULUS, f"modular modulus exceeds analysis budget {MAX_MODULUS}"
        )
        require(isinstance(self.terms, tuple), "modular atom terms must be immutable")
        _limit(
            len(self.terms) <= MAX_TERMS, f"modular term count exceeds analysis budget {MAX_TERMS}"
        )
        keys = []
        for term in self.terms:
            require(isinstance(term, tuple) and len(term) == 2, "invalid modular polynomial term")
            monomial, coefficient = term
            require(
                isinstance(monomial, tuple)
                and bool(monomial)
                and all(
                    isinstance(factor, Expr) and factor.constant is None for factor in monomial
                ),
                "modular monomials need nonconstant integer expression factors",
            )
            key = _monomial_key(monomial)
            require(key == tuple(sorted(key)), "modular monomial factors must be sorted")
            require(
                type(coefficient) is int and 0 < coefficient < self.modulus,
                "modular coefficients must be nonzero canonical residues",
            )
            keys.append(key)
        require(keys == sorted(set(keys)), "modular polynomial terms must be sorted and distinct")


@dataclass(frozen=True, slots=True)
class ModularPredicate:
    """Membership of one canonical polynomial residue in a finite static set."""

    atom: ModularAtom
    residues: frozenset[int]

    def __post_init__(self):
        require(isinstance(self.atom, ModularAtom), "modular predicate needs a typed atom")
        require(
            isinstance(self.residues, frozenset)
            and all(
                type(value) is int and 0 <= value < self.atom.modulus for value in self.residues
            ),
            "modular predicate residues must be an immutable canonical integer set",
        )


class _Polynomial:
    def __init__(self, modulus: int, bindings: Mapping[Expr, int] | None = None):
        self.modulus = modulus
        self.work = 0
        self.memo: dict[Expr, Polynomial] = {}
        self.exact_memo: dict[Expr, int | None] = {}
        self.charge(0 if bindings is None else len(bindings))
        self.bindings = {} if bindings is None else dict(bindings)
        require(
            all(
                isinstance(value, Expr) and type(integer) is int
                for value, integer in self.bindings.items()
            ),
            "modular bindings must map integer expressions to exact integers",
        )

    def charge(self, amount: int):
        _limit(self.work + amount <= MAX_WORK, f"modular work exceeds analysis budget {MAX_WORK}")
        self.work += amount

    @staticmethod
    def size(value: Polynomial) -> int:
        return sum(1 + len(monomial) for monomial in value)

    def accumulate(self, result: Polynomial, monomial: Monomial, coefficient: int):
        coefficient = (result.get(monomial, 0) + coefficient) % self.modulus
        if coefficient:
            if monomial not in result and monomial:
                _limit(
                    len(result) - int(() in result) < MAX_TERMS,
                    f"modular term count exceeds analysis budget {MAX_TERMS}",
                )
            result[monomial] = coefficient
        else:
            result.pop(monomial, None)

    def combine(self, left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
        self.charge(self.size(left) + self.size(right))
        result = dict(left)
        for monomial, coefficient in right.items():
            self.accumulate(result, monomial, sign * coefficient)
        return result

    def product(self, left: Polynomial, right: Polynomial) -> Polynomial:
        # Charge all candidate products and factor copies before expansion.
        self.charge(
            len(right) * self.size(left) + len(left) * sum(len(monomial) for monomial in right)
        )
        result: Polynomial = {}
        for a, coefficient_a in left.items():
            for b, coefficient_b in right.items():
                monomial = tuple(sorted((*a, *b), key=lambda factor: factor.order_key))
                self.accumulate(result, monomial, coefficient_a * coefficient_b)
        return result

    def rebase(self, modulus: int, terms: Terms) -> Polynomial:
        """Normalize reduced coefficients when more nested remainders peel."""
        self.modulus = modulus
        self.memo = {}
        result: Polynomial = {}
        for monomial, coefficient in terms:
            term = {(): coefficient}
            for factor in monomial:
                term = self.product(term, self.visit(factor))
            result = self.combine(result, term)
        return result

    def exact(self, value: Expr) -> int | None:
        """One mathematical integer under caller-proved exact bindings."""
        if value in self.bindings:
            return self.bindings[value]
        if value.constant is not None:
            return value.constant
        if not self.bindings:
            return None
        if value in self.exact_memo:
            return self.exact_memo[value]
        self.charge(1)
        if value.kind == "argument":
            result = None
        else:
            a, b = (self.exact(child) for child in value.args)
            if a is None or b is None:
                result = None
            elif value.kind == "add":
                result = a + b
            elif value.kind == "mul":
                result = a * b
            elif value.kind == "min":
                result = min(a, b)
            elif value.kind == "max":
                result = max(a, b)
            elif b <= 0:
                result = None
            elif value.kind == "floordiv":
                result = a // b
            elif value.kind == "ceildiv":
                result = a // b + int(a % b != 0)
            else:
                assert value.kind == "rem"
                result = a % b
        self.exact_memo[value] = result
        return result

    def visit(self, value: Expr) -> Polynomial:
        if value in self.memo:
            return self.memo[value]
        self.charge(1)
        if self.modulus == 1:
            result = {}
        elif (integer := self.exact(value)) is not None:
            coefficient = integer % self.modulus
            result = {(): coefficient} if coefficient else {}
        elif value.kind == "add":
            result = self.combine(*(self.visit(child) for child in value.args))
        elif value.kind == "mul":
            if any(
                child.constant is not None and child.constant % self.modulus == 0
                for child in value.args
            ):
                result = {}
            else:
                result = self.product(*(self.visit(child) for child in value.args))
        elif (
            value.kind == "rem"
            and value.args[1].constant is not None
            and value.args[1].constant % self.modulus == 0
        ):
            # (x mod K) mod M == x mod M when positive K is a multiple of M.
            result = self.visit(value.args[0])
        else:
            # Keep min/max, quotients and unrelated remainders intact. Their
            # internal variables retain their original correlation and scope.
            result = {(value,): 1}
        self.memo[value] = result
        return result


def _constant(truth: bool) -> ModularPredicate:
    return ModularPredicate(ModularAtom(1, ()), frozenset((0,)) if truth else frozenset())


def _equal_remainders(test: Comparison) -> tuple[int, Expr, Expr] | None:
    if test.predicate not in {"==", "!="}:
        return None
    lhs, rhs = test.lhs, test.rhs
    if lhs.kind != "rem" or (modulus := lhs.args[1].constant) is None:
        return None
    if rhs.kind == "rem" and rhs.args[1].constant == modulus:
        return modulus, lhs, rhs
    return None


def _compare(lhs: int, predicate: str, rhs: int) -> bool:
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[predicate]


def _affine_remainder(query: _Polynomial, value: Expr) -> tuple[Expr | None, int, int] | None:
    """One remainder basis with exact integer scale and offset, without IR."""
    memo: dict[Expr, tuple[Expr | None, int, int] | None] = {}

    def visit(current: Expr) -> tuple[Expr | None, int, int] | None:
        if current in memo:
            return memo[current]
        query.charge(1)
        if current.kind == "rem" and current.args[1].constant is not None:
            result = current, 1, 0
        elif (integer := query.exact(current)) is not None:
            result = None, 0, integer
        elif current.kind == "add":
            left, right = (visit(child) for child in current.args)
            if (
                left is None
                or right is None
                or left[0] is not None
                and right[0] is not None
                and left[0] != right[0]
            ):
                result = None
            else:
                coefficient, offset = left[1] + right[1], left[2] + right[2]
                result = (left[0] if left[0] is not None else right[0], coefficient, offset)
                if coefficient == 0:
                    result = None, 0, offset
        elif current.kind == "mul":
            result = None
            for scalar, other in (current.args, tuple(reversed(current.args))):
                if (integer := query.exact(scalar)) is not None and (
                    part := visit(other)
                ) is not None:
                    coefficient, offset = integer * part[1], integer * part[2]
                    result = (part[0], coefficient, offset) if coefficient else (None, 0, offset)
                    break
        else:
            result = None
        memo[current] = result
        return result

    return visit(value)


def _normalize(
    query: _Polynomial, polynomial: Polynomial, residues: frozenset[int]
) -> ModularPredicate:
    modulus = query.modulus
    while True:
        if not residues:
            return _constant(False)
        if len(residues) == modulus:
            return _constant(True)
        constant = polynomial.pop((), 0)
        query.charge(len(residues))
        residues = frozenset((residue - constant) % modulus for residue in residues)
        if not polynomial:
            return _constant(0 in residues)
        terms = tuple(sorted(polynomial.items(), key=lambda term: _monomial_key(term[0])))
        divisor = gcd(modulus, *(coefficient for _, coefficient in terms))
        if divisor == 1:
            break
        query.charge(len(residues))
        residues = frozenset(residue // divisor for residue in residues if residue % divisor == 0)
        modulus //= divisor
        terms = tuple((monomial, coefficient // divisor) for monomial, coefficient in terms)
        if not residues:
            return _constant(False)
        if len(residues) == modulus:
            return _constant(True)
        polynomial = query.rebase(modulus, terms)
    # Primitive coefficients have a trivial unit stabilizer: two units giving
    # the same complete vector must be equal modulo M. Thus selecting the
    # least vector also determines one unambiguous target transformation.
    leading = terms[0][1]
    if gcd(leading, modulus) == 1:
        query.charge(len(terms))
        unit = pow(leading, -1, modulus)
    else:
        query.charge(modulus)
        units = tuple(unit for unit in range(1, modulus) if gcd(unit, modulus) == 1)
        query.charge(len(units) * len(terms))
        _, unit = min(
            (tuple(unit * coefficient % modulus for _, coefficient in terms), unit)
            for unit in units
        )
    terms = tuple((monomial, unit * coefficient % modulus) for monomial, coefficient in terms)
    query.charge(len(residues))
    return ModularPredicate(
        ModularAtom(modulus, terms), frozenset(unit * residue % modulus for residue in residues)
    )


def modular_predicate(
    test: Comparison, *, bindings: Mapping[Expr, int] | None = None
) -> ModularPredicate | None:
    """Canonicalize a supported remainder predicate, or return unknown.

    Equality/inequality may compare two remainders with the same static positive
    modulus. All six comparisons also support one remainder's static affine
    function against an exact integer. The ordered comparison is evaluated over
    its original finite residue range before gcd/unit normalization; it is never
    interpreted as an ordering of modular differences. Caller-proved bindings
    enter only the mathematical parser. Runtime coordinates are never enumerated.
    """
    query = _Polynomial(1, bindings)
    lhs, rhs = query.exact(test.lhs), query.exact(test.rhs)
    if lhs is not None and rhs is not None:
        return _constant(_compare(lhs, test.predicate, rhs))
    if lhs is not None or rhs is not None:
        affine = _affine_remainder(query, test.rhs if lhs is not None else test.lhs)
        if affine is None:
            return None
        remainder, scale, offset = affine
        if remainder is None:
            return _constant(
                _compare(lhs, test.predicate, offset)
                if lhs is not None
                else _compare(offset, test.predicate, rhs)
            )
        modulus = remainder.args[1].constant
        assert modulus is not None
        _limit(modulus <= MAX_MODULUS, f"modular modulus exceeds analysis budget {MAX_MODULUS}")
        query.modulus = modulus
        query.charge(modulus)
        residues = frozenset(
            residue
            for residue in range(modulus)
            if (
                _compare(lhs, test.predicate, scale * residue + offset)
                if lhs is not None
                else _compare(scale * residue + offset, test.predicate, rhs)
            )
        )
        return _normalize(query, query.combine(query.visit(remainder), {}), residues)
    operands = _equal_remainders(test)
    if operands is None:
        return None
    modulus, lhs, rhs = operands
    _limit(modulus <= MAX_MODULUS, f"modular modulus exceeds analysis budget {MAX_MODULUS}")
    query.modulus = modulus
    left = query.visit(lhs)
    right = query.visit(rhs)
    difference = query.combine(left, right, -1)
    query.charge(1 if test.predicate == "==" else modulus)
    residues = (
        frozenset((0,))
        if test.predicate == "=="
        else frozenset(value for value in range(modulus) if value != 0)
    )
    return _normalize(query, difference, residues)


def specialize_residues(
    predicate: ModularPredicate, bindings: Mapping[Expr, int]
) -> ModularPredicate:
    """Rekey a complete allowed residue set under caller-proved exact values.

    Binding does not construct source expressions or establish definedness.
    Operations kept as opaque factors remain original expressions unless their
    complete mathematical value becomes fixed. One query budget covers parsing,
    reachable-residue filtering, repeated gcd reduction and unit normalization.
    """
    query = _Polynomial(predicate.atom.modulus, bindings)
    if not predicate.residues:
        return _constant(False)
    if len(predicate.residues) == predicate.atom.modulus:
        return _constant(True)
    polynomial = query.rebase(predicate.atom.modulus, predicate.atom.terms)
    return _normalize(query, polynomial, predicate.residues)

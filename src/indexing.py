"""Bounded integer algebra for source axes and physical address contracts.

The polynomial is a proof representation, never a generated task history.
Runtime radices stay symbolic. Euclidean division identities and declared
multiples suffice for ordinary row-major split/recombine indexing; unknown
relations fail closed. A fixed term budget bounds distributive expansion.
"""

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

from .diagnostics import CompileError, require
from .scalar import DIVISION, Expr, expression, literal
from .symbols import Symbol, SymbolKind, is_coordinate

MAX_TERMS = 256
MAX_WORK = 16384


def _charge(work, amount):
    work[0] += amount
    require(work[0] <= MAX_WORK, "index algebra cumulative work budget exceeded")


def _sum(a, b):
    result = dict(a)
    for term, coefficient in b.items():
        result[term] = result.get(term, 0) + coefficient
        if not result[term]:
            del result[term]
    require(len(result) <= MAX_TERMS, "index algebra term budget exceeded")
    return result


def _scale(a, coefficient):
    return {term: c * coefficient for term, c in a.items() if c * coefficient}


def _ratio(numerator, denominator):
    """Exact coefficient division, retaining the common integer case.

    Python int also implements Rational's numerator/denominator contract.
    Never use int/int's floating point division in the proof representation.
    """
    if isinstance(numerator, int) and isinstance(denominator, int):
        quotient, remainder = divmod(numerator, denominator)
        return Fraction(numerator, denominator) if remainder else quotient
    return numerator / denominator


def _product(a, b):
    require(len(a) * len(b) <= MAX_TERMS, "index algebra product budget exceeded")
    result = {}
    for (at, ac), (bt, bc) in product(a.items(), b.items()):
        term = tuple(sorted((*at, *bt), key=lambda atom: atom.order_key))
        result = _sum(result, {term: ac * bc})
    return result


def _multiple(atom, facts):
    from .scalar import infer

    value = infer(atom, facts)
    return 1 if value is None else value.multiple


def _integral(polynomial, facts):
    """Whether a term's *value* is integral under declared input multiples.

    This relaxation lets equality queries cancel a quotient against its
    numerator (``4*(n//4) == n`` for ``n % 4 == 0``). The rational coefficient is
    a proof-form artifact only: it is never rebuilt into an executable
    expression, which requires :func:`_exact_integers` instead.
    """
    for term, coefficient in polynomial.items():
        value = coefficient
        for atom in term:
            value *= _multiple(atom, facts)
        if value.denominator != 1:
            return False
    return True


def _to_expr(polynomial):
    """Rebuild one executable expression from an *integral* proof polynomial.

    Rational coefficients are a property of the proof representation, not of
    the program: the executable DAG keeps the original quotient atoms. A caller
    that only has a rational form must keep the original expression (or ask the
    query service for the relation it needs) instead of reconstructing a new
    product/division DAG here.
    """
    result = literal(0)
    for term, coefficient in sorted(
        polynomial.items(), key=lambda item: tuple(atom.order_key for atom in item[0])
    ):
        require(coefficient.denominator == 1, "runtime address retains a fractional coefficient")
        value = literal(int(coefficient))
        for atom in term:
            value = expression("mul", value, atom)
        result = expression("add", result, value)
    return result


@dataclass(frozen=True)
class _PolynomialKey:
    terms: tuple
    order_key: tuple = field(init=False, compare=False, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash(self.terms))
        object.__setattr__(
            self,
            "order_key",
            tuple(
                (
                    tuple(atom.order_key for atom in term),
                    coefficient.numerator,
                    coefficient.denominator,
                )
                for term, coefficient in self.terms
            ),
        )

    def __hash__(self):
        return self._hash


@dataclass(frozen=True)
class _FunctionKey:
    kind: str
    operands: tuple[_PolynomialKey, ...]
    order_key: tuple = field(init=False, compare=False, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash((self.kind, self.operands)))
        object.__setattr__(
            self, "order_key", (self.kind, tuple(arg.order_key for arg in self.operands))
        )

    def __hash__(self):
        return self._hash


class AtomIdentity:
    """Scoped function congruence keys, with no stored raw representative.

    Only integer-ring operations enter an operand's key. Unsigned operations
    remain uninterpreted functions here, so a prefix-only divisor needs no
    global positivity fact. Keys say nothing about an original DAG's domain.
    """

    def __init__(self):
        self._polynomials, self._keys, self._atoms = {}, {}, {}
        self._polynomial_keys, self._function_keys = {}, {}

    def key(self, value, work):
        _charge(work, 1)
        if value not in self._keys:
            polynomial = _PolynomialQuery(
                {}, identity=self, retained=True, keys=True, memo=self._polynomials, work=work
            ).visit(value)
            _charge(work, sum(1 + len(term) for term in polynomial))
            key = _PolynomialKey(
                tuple(
                    sorted(
                        polynomial.items(),
                        key=lambda item: tuple(atom.order_key for atom in item[0]),
                    )
                )
            )
            self._keys[value] = self._polynomial_keys.setdefault(key, key)
        return self._keys[value]

    def atom(self, value, work):
        _charge(work, 1)
        if value.kind == "argument":
            return value
        if value not in self._atoms:
            operands = tuple(self.key(arg, work) for arg in value.args)
            if value.kind in {"min", "max"}:
                operands = tuple(sorted(operands, key=lambda arg: arg.order_key))
            key = _FunctionKey(value.kind, operands)
            self._atoms[value] = self._function_keys.setdefault(key, key)
        return self._atoms[value]

    def retained(self, value):
        """Cancel ring terms using this query's original function operands."""
        return _to_expr(_PolynomialQuery({}, identity=self, retained=True).visit(value))


class _PolynomialQuery:
    """One bounded polynomial query and its local raw atom representatives."""

    def __init__(
        self,
        facts,
        *,
        periodic=False,
        identity=None,
        retained=False,
        keys=False,
        memo=None,
        work=None,
        opaque_quotients=False,
    ):
        self.facts, self.periodic = facts, periodic
        # A relation between quotient *atoms* is weaker and therefore safe; it
        # is what a caller wants when it must name the original quotient in an
        # executable expression instead of expanding it over declared multiples.
        self.opaque_quotients = opaque_quotients
        self.identity = AtomIdentity() if identity is None else identity
        self.retained, self.keys = retained, keys
        self.memo = {} if memo is None else memo
        self.work = [0] if work is None else work
        self.representatives = {}

    def atom(self, value):
        key = self.identity.atom(value, self.work)
        if self.keys:
            return key
        # Equal mathematical operands need not have the same definedness.
        # Never borrow a raw expression from an earlier query's guard/domain.
        return self.representatives.setdefault(key, value)

    def visit(self, expr):
        from .scalar import positive

        _charge(self.work, 1)
        if expr in self.memo:
            return self.memo[expr]
        if expr.kind == "constant":
            result = {(): expr.constant} if expr.constant else {}
        elif expr.kind in {"add", "mul"}:
            a, b = (self.visit(arg) for arg in expr.args)
            _charge(
                self.work,
                len(a) + len(b)
                if expr.kind == "add"
                else (
                    len(b) * sum(1 + len(term) for term in a)
                    + len(a) * sum(len(term) for term in b)
                ),
            )
            result = _sum(a, b) if expr.kind == "add" else _product(a, b)
        elif self.retained or self.opaque_quotients and expr.kind in DIVISION:
            result = {(self.atom(expr),): 1}
        elif expr.kind in {"floordiv", "rem"}:
            numerator, divisor = expr.args
            require(
                positive(divisor, self.facts), "index divisor must be provably positive and uniform"
            )
            coordinate = numerator
            if coordinate.kind == "floordiv" and coordinate.args[1].constant is not None:
                coordinate = coordinate.args[0]
            if (
                self.periodic
                and expr.kind == "rem"
                and divisor.constant is not None
                and coordinate.kind == "argument"
                and is_coordinate(coordinate.args[0])
            ):
                result = {(self.atom(expr),): 1}
            elif expr.kind == "rem":
                quotient = self.visit(expression("floordiv", numerator, divisor))
                result = _sum(
                    self.visit(numerator),
                    _scale(_product(quotient, self.visit(divisor)), -1),
                )
            else:
                # floor(floor(x/a)/b) == floor(x/(a*b)) for positive integer a,b.
                if numerator.kind == "floordiv":
                    numerator, inner = numerator.args
                    divisor = expression("mul", inner, divisor)
                top, bottom = self.visit(numerator), self.visit(divisor)
                quotient, remainder = {}, dict(top)
                _charge(self.work, sum(1 + len(term) for term in top))
                if len(bottom) == 1:
                    ((dt, dc),) = bottom.items()
                    for term, coefficient in top.items():
                        counts = Counter(term)
                        needed = Counter(dt)
                        if needed <= counts:
                            remaining = tuple(
                                sorted(
                                    (counts - needed).elements(), key=lambda atom: atom.order_key
                                )
                            )
                            candidate = {remaining: _ratio(coefficient, dc)}
                            if _integral(candidate, self.facts):
                                quotient = _sum(quotient, candidate)
                                del remainder[term]
                if not remainder:
                    result = quotient
                elif remainder == {(): -1} and quotient:
                    result = _sum(quotient, {(): -1})
                else:
                    # Keep integral operand reconstruction for existing index
                    # identities. Congruence no longer depends on this spelling.
                    if all(c.denominator == 1 for p in (top, bottom) for c in p.values()):
                        numerator = _to_expr(top)
                        divisor = _to_expr(bottom)
                    atom = expression("floordiv", numerator, divisor)
                    result = {(self.atom(atom),): 1}
        else:
            result = {(self.atom(expr),): 1}
        self.memo[expr] = result
        return result


def polynomial(expr, facts, *, periodic=False, retained=False):
    """Exact rational polynomial whose public term atoms remain raw Exprs."""
    return _PolynomialQuery(facts, periodic=periodic, retained=retained).visit(expr)


def equivalent(a, b, facts):
    query = _PolynomialQuery(facts)
    return query.visit(a) == query.visit(b)


def _exact_integers(polynomial):
    return all(coefficient.denominator == 1 for coefficient in polynomial.values())


def normalize(expr, facts, *, identity=None):
    """Ring-normalize one expression that stays integral over the integers.

    A declared input multiple can make a rational proof coefficient denote an
    integer value, but the executable DAG then has no quotient term to name it.
    Such a case keeps the original expression at its caller instead of a new
    product/division DAG; see :func:`normalize_integral`.
    """
    return _to_expr(_PolynomialQuery(facts, identity=identity).visit(expr))


def normalize_integral(expr, facts, *, identity=None):
    """Rebuild a normalized expression, or ``None`` when it is not integral.

    A caller that can proceed with less information (for example a window
    reduction that only wants its clock-invariant part) must not turn a
    fractional proof coefficient into an exception, and must never receive a
    fabricated integer.
    """
    polynomial = _PolynomialQuery(facts, identity=identity).visit(expr)
    if not _exact_integers(polynomial):
        return None
    return _to_expr(polynomial)


def normalize_or_identity(expr, facts, *, identity=None):
    """Best-effort ring normalization that keeps the original when it cannot.

    Some captures legitimately have a rational proof coefficient because of a
    declared input multiple. Such a caller needs a canonical spelling *or* the
    untouched syntax, never an invented quotient and never a string-matched
    exception path. Both outcomes stay unproved expressions; every non-trivial
    relation is asked through the query service.
    """
    integral = normalize_integral(expr, facts, identity=identity)
    return expr if integral is None else integral


def exact_constant(expr, facts, *, identity=None):
    """The exact integer value of a normalized expression, or ``None``.

    Callers that only ask whether two coordinates differ by a constant must not
    force a full executable reconstruction: a term whose coefficient is only
    integral under an input multiple is still not a constant.
    """
    polynomial = _PolynomialQuery(facts, identity=identity).visit(expr)
    if any(term for term in polynomial):
        return None
    value = polynomial.get((), 0)
    return int(value) if value.denominator == 1 else None


def arguments(expr):
    # Free variables are compositional expression facts, independent of any
    # input range or guard. Establish them with each immutable node once.
    return set(expr.variables)


def physical_terms(expr, facts):
    """Separate a uniform coefficient from one physical clock per term."""
    grouped = {}
    for term, coefficient in polynomial(expr, facts, periodic=True).items():
        coordinates = [atom for atom in term if any(is_coordinate(n) for n in arguments(atom))]
        require(len(coordinates) <= 1, "address combines unsupported task clocks")
        coordinate = coordinates[0] if coordinates else None
        uniform = list(term)
        if coordinate is not None:
            uniform.remove(coordinate)
        grouped[coordinate] = _sum(grouped.get(coordinate, {}), {tuple(uniform): coefficient})
    return {coordinate: _to_expr(coefficient) for coordinate, coefficient in grouped.items()}


def nonnegative(expr, facts):
    """Mathematical sign under input facts; original evaluation is separate."""
    from .analysis.smt import query_service
    from .analysis.structured_schedule import Comparison

    if expr.constant is not None:
        return expr.constant >= 0
    return query_service(facts).implies(Comparison(expr, ">=", literal(0))).proved


def _joint_affine_interval(expr, domains, facts, *, atomic=False, queries=None):
    """Recover a proved joint coordinate after ordinary polynomial expansion.

    A captured address may spell ``tile*core + tile*groups*query`` rather
    than ``tile*(core + groups*query)``. Independent intervals lose the
    owner/query correlation. Use an exact polynomial identity with a positive
    or negative constant scale and an integral uniform bias; never infer a new
    iteration domain. Every returned bound is additionally proved as an
    implication of the coordinate's own domain by the shared query service, so
    the concrete integer DAG is never trusted from algebra alone.
    """
    candidates = [
        (coordinate, lower, upper)
        for coordinate, lower, upper in domains
        if coordinate.kind in ({"argument", "add", "mul"} if atomic else {"add", "mul"})
        and any(is_coordinate(name) for name in arguments(coordinate))
    ]
    if not candidates or expr.kind not in {"add", "mul"}:
        return None

    def varying(poly):
        return {
            term: coefficient
            for term, coefficient in poly.items()
            if any(is_coordinate(name) for atom in term for name in arguments(atom))
        }

    query = _PolynomialQuery(facts, opaque_quotients=True)
    top = query.visit(expr)
    left = varying(top)
    if not left:
        return None
    for coordinate, lower, upper in candidates:
        bottom = query.visit(coordinate)
        right = varying(bottom)
        if not right or left.keys() != right.keys():
            continue
        ratios = {_ratio(left[term], coefficient) for term, coefficient in right.items()}
        if len(ratios) != 1:
            continue
        (ratio,) = ratios
        if ratio == 0 or ratio.denominator != 1:
            continue
        remainder = _sum(top, _scale(bottom, -ratio))
        if not _exact_integers(remainder):
            continue
        bias = _to_expr(remainder)
        scale = literal(int(ratio))
        selected = (lower, upper) if ratio > 0 else (upper, lower)
        bounds = tuple(
            expression("add", expression("mul", scale, bound), bias) for bound in selected
        )
        if queries is not None and not _bounds_hold(
            expr, bounds, coordinate, lower, upper, queries
        ):
            return None
        return bounds
    return None


def _bounds_hold(expr, bounds, coordinate, lower, upper, queries):
    """Ask the shared query service whether the joint bounds are implied.

    The coordinate's own lexical domain is the only premise; no new iteration
    domain and no expression's own legality is assumed.
    """
    from .analysis.structured_schedule import Comparison

    premises = (Comparison(coordinate, ">=", lower), Comparison(coordinate, "<=", upper))
    low, high = bounds
    return (
        queries.implies(Comparison(expr, ">=", low), premises).proved
        and queries.implies(Comparison(expr, "<=", high), premises).proved
    )


def interval(expr, domains, facts, *, queries=None):
    """Inclusive symbolic range on an active loop iteration.

    Each domain entry is (expression, lower, upper). It can describe a joint
    coordinate such as core + g*groups, preserving cyclic ownership information
    which separate core/g intervals would lose.
    """
    matched = None
    for coordinate, lower, upper in domains:
        if equivalent(expr, coordinate, facts):
            matched = (
                (lower, upper)
                if matched is None
                else (expression("max", matched[0], lower), expression("min", matched[1], upper))
            )
    if matched is not None:
        return matched
    correlated = _joint_affine_interval(expr, domains, facts, queries=queries)
    if correlated is not None:
        return correlated
    if expr.kind in {"constant", "argument"}:
        return expr, expr
    a, b = expr.args
    al, ah = interval(a, domains, facts)
    bl, bh = interval(b, domains, facts)
    if expr.kind in {"add", "min", "max"}:
        return expression(expr.kind, al, bl), expression(expr.kind, ah, bh)
    if expr.kind == "mul":
        if a.constant is not None:
            low, high = (bh, bl) if a.constant < 0 else (bl, bh)
            return expression("mul", a, low), expression("mul", a, high)
        if b.constant is not None:
            low, high = (ah, al) if b.constant < 0 else (al, ah)
            return expression("mul", b, low), expression("mul", b, high)
        require(
            domain_nonnegative(al, domains, facts) and domain_nonnegative(bl, domains, facts),
            "index product has an unproven sign",
            witness=(
                f"product={expr.text()}",
                f"lower_left={al.text()}",
                f"lower_right={bl.text()}",
            ),
        )
        return expression("mul", al, bl), expression("mul", ah, bh)
    if expr.kind in {"floordiv", "ceildiv", "rem"}:
        from .scalar import positive

        require(
            domain_nonnegative(al, domains, facts),
            "unsigned index division has an unproven nonnegative dividend",
        )
        require(al == ah or b == bl == bh, "index divisor is not uniform")
        require(positive(b, facts), "index divisor must be provably positive and uniform")
        if expr.kind == "rem":
            return literal(0), expression("sub", b, literal(1))
        return expression(expr.kind, al, b), expression(expr.kind, ah, b)
    require(False, f"unsupported index range operation {expr.kind}")


def domain_nonnegative(expr, domains, facts):
    """Keep the complete lexical range relations in one sign query.

    Loaded metadata and ancestor/child coordinates keep their actual identities.
    Do not recursively replace correlated bounds by independent extrema, or
    borrow a later condition to authorize an original unsigned evaluation.
    """
    from .analysis.smt import query_service
    from .analysis.structured_schedule import Comparison

    premises = tuple(
        test
        for coordinate, lower, upper in domains
        for test in (Comparison(coordinate, ">=", lower), Comparison(coordinate, "<=", upper))
    )
    return query_service(facts).implies(Comparison(expr, ">=", literal(0)), premises).proved


def check_axis(origin, width, extent, domains, facts):
    # Validate the original DAG before algebra can cancel an invalid unsigned
    # division. For example divmod(g-5, 3) recombines mathematically, but g-5
    # cannot be an unsigned dividend on the first iterations.
    seen = set()

    def divisions(expr):
        if expr in seen:
            return
        seen.add(expr)
        for arg in expr.args:
            if isinstance(arg, Expr):
                divisions(arg)
        if expr.kind in {"floordiv", "rem", "ceildiv"}:
            lower, _ = interval(expr.args[0], domains, facts)
            require(
                domain_nonnegative(lower, domains, facts),
                "unsigned index division has an unproven nonnegative dividend",
            )

    divisions(origin)
    lower, _ = interval(origin, domains, facts)
    require(
        domain_nonnegative(origin, domains, facts) or domain_nonnegative(lower, domains, facts),
        "logical axis origin may be negative",
    )
    remaining = expression("sub", extent, expression("add", origin, literal(width)))
    require(
        domain_nonnegative(remaining, domains, facts),
        "logical axis access is not bounded by its tensor extent",
    )


def validate_source_arithmetic(program, controls):
    """Check every original division before value/address simplification.

    Even an unused div followed by ``bad-bad`` must satisfy its source domain.
    Walking the retained scopes avoids mistaking a simplified Expr DAG for the
    original SSA. Branch predicates do not silently grant stronger ranges.
    """
    from xdsl.dialects import arith, scf

    from .dialects.source import DomainOp, IndexReadOp, IterationOp
    from .scalar import Fact, facts_from_sizes, from_ssa, positive

    # Only unsigned divisions need these range premises. Keep all of them,
    # including dead expressions, but do not demand an access-bound formula
    # for unrelated SCF carries. Propagate relevance once up the actual IR;
    # walking each division's ancestor chain would repeat work at every depth.
    divisions = (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)
    relevant = set()
    for operation in reversed(tuple(program.walk())):
        if isinstance(operation, divisions) or operation in relevant:
            relevant.add(operation)
            if (parent := operation.parent_op()) is not None:
                relevant.add(parent)

    facts = facts_from_sizes(program.body.block.ops)
    owner = program.body.block.args[0]
    core = Expr("argument", (Symbol("$core", SymbolKind.GROUP, program),))
    facts[core.args[0]] = Fact(maximum=program.groups.data - 1)
    number = 0

    def walk(block, coordinates, domains):
        nonlocal number
        for op in block.ops:
            if isinstance(op, IndexReadOp):
                name = Symbol(
                    f"$metadata{number}",
                    SymbolKind.METADATA,
                    op.value,
                    tuple(
                        Expr("argument", (symbol,))
                        for symbol in coordinates.values()
                        if isinstance(symbol, Symbol)
                        and symbol.kind
                        in {SymbolKind.GROUP, SymbolKind.PARTICIPANT, SymbolKind.ORDINAL}
                    ),
                )
                number += 1
                lower, upper = (
                    from_ssa(v, coordinates=coordinates, facts=facts)
                    for v in (op.minimum, op.maximum)
                )
                coordinates = coordinates | {op.value: name}
                coordinate = Expr("argument", (name,))
                domains = [
                    *domains,
                    (coordinate, lower, upper),
                    (
                        expression("sub", coordinate, lower),
                        literal(0),
                        expression("sub", upper, lower),
                    ),
                ]
                continue
            if op not in relevant:
                continue
            if isinstance(op, divisions):
                lhs, rhs = (
                    from_ssa(value, coordinates=coordinates, facts=facts) for value in op.operands
                )
                try:
                    lower, _ = interval(lhs, domains, facts)
                    for _ in range(len(domains) + 1):
                        bound = interval(lower, domains, facts)[0]
                        if bound == lower:
                            break
                        lower = bound
                except CompileError as error:
                    raise CompileError(
                        f"{op.location}: source unsigned division: {error}"
                    ) from None
                require(
                    nonnegative(lower, facts),
                    "source unsigned division has an unproven nonnegative dividend",
                    location=str(op.location),
                    witness=(f"dividend={lhs.text()}", f"lower={lower.text()}"),
                )
                require(
                    positive(rhs, facts),
                    "source index divisor must be provably positive and uniform",
                )
            local, ranges = coordinates, domains
            if isinstance(op, (IterationOp, DomainOp, scf.ForOp)):
                name = Symbol(
                    f"$index{number}",
                    SymbolKind.PARTICIPANT if isinstance(op, DomainOp) else SymbolKind.ORDINAL,
                    op,
                )
                number += 1
                value = op.body.block.args[0]
                if isinstance(op, DomainOp):
                    lower = literal(0)
                    bound = literal(
                        2 if program.kind.data == "mixed" and op.kind.data == "vector" else 1
                    )
                else:
                    domain = controls.by_operation[op].domain
                    lower = (
                        literal(domain.lower)
                        if type(domain.lower) is int
                        else from_ssa(domain.lower, coordinates=coordinates, facts=facts)
                    )
                    bound = (
                        literal(domain.upper)
                        if type(domain.upper) is int
                        else from_ssa(domain.upper, coordinates=coordinates, facts=facts)
                    )
                if lower == bound or (
                    lower.constant is not None
                    and bound.constant is not None
                    and lower.constant >= bound.constant
                ):
                    continue  # No scalar operation in this loop can execute.
                local = coordinates | {value: name}
                ranges = [
                    *domains,
                    (Expr("argument", (name,)), lower, expression("sub", bound, literal(1))),
                ]
            for region in op.regions:
                for child in region.blocks:
                    walk(child, local, ranges)

    walk(
        program.body.block,
        {owner: core.args[0]},
        [(core, literal(0), literal(program.groups.data - 1))],
    )

"""Bounded integer algebra for source axes and physical address contracts.

The polynomial is a proof representation, never a generated task history.
Runtime radices stay symbolic. Euclidean division identities and declared
multiples suffice for ordinary row-major split/recombine indexing; unknown
relations fail closed. A fixed term budget bounds distributive expansion.
"""

from collections import Counter
from fractions import Fraction
from itertools import product

from .diagnostics import CompileError, require
from .scalar import Expr, expression, literal

MAX_TERMS = 256
MAX_WORK = 16384


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
    for term, coefficient in polynomial.items():
        value = coefficient
        for atom in term:
            value *= _multiple(atom, facts)
        if value.denominator != 1:
            return False
    return True


def _to_expr(polynomial):
    """Reconstruct only integral coefficients; proof fractions stay internal."""
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


def polynomial(expr, facts, *, periodic=False, _memo=None, _work=None):
    """Exact rational polynomial over integer atoms, under explicit facts."""
    from .scalar import positive

    memo = {} if _memo is None else _memo
    work = [0] if _work is None else _work

    def charge(amount):
        work[0] += amount
        require(work[0] <= MAX_WORK, "index algebra cumulative work budget exceeded")

    charge(1)
    if expr in memo:
        return memo[expr]
    if expr.kind == "constant":
        result = {(): Fraction(expr.constant)} if expr.constant else {}
    elif expr.kind in {"add", "mul"}:
        a, b = (
            polynomial(arg, facts, _memo=memo, _work=work, periodic=periodic) for arg in expr.args
        )
        charge(
            len(a) + len(b)
            if expr.kind == "add"
            else (len(b) * sum(1 + len(term) for term in a) + len(a) * sum(len(term) for term in b))
        )
        result = _sum(a, b) if expr.kind == "add" else _product(a, b)
    elif expr.kind in {"floordiv", "rem"}:
        numerator, divisor = expr.args
        require(positive(divisor, facts), "index divisor must be provably positive and uniform")
        coordinate = numerator
        if coordinate.kind == "floordiv" and coordinate.args[1].constant is not None:
            coordinate = coordinate.args[0]
        if (
            periodic
            and expr.kind == "rem"
            and divisor.constant is not None
            and coordinate.kind == "argument"
            and coordinate.args[0].startswith("$")
        ):
            result = {(expr,): Fraction(1)}
        elif expr.kind == "rem":
            quotient = polynomial(
                expression("floordiv", numerator, divisor),
                facts,
                _memo=memo,
                _work=work,
                periodic=periodic,
            )
            result = _sum(
                polynomial(numerator, facts, _memo=memo, _work=work, periodic=periodic),
                _scale(
                    _product(
                        quotient,
                        polynomial(divisor, facts, _memo=memo, _work=work, periodic=periodic),
                    ),
                    -1,
                ),
            )
        else:
            # floor(floor(x/a)/b) == floor(x/(a*b)) for positive integer a,b.
            if numerator.kind == "floordiv":
                numerator, inner = numerator.args
                divisor = expression("mul", inner, divisor)
            top = polynomial(numerator, facts, _memo=memo, _work=work, periodic=periodic)
            bottom = polynomial(divisor, facts, _memo=memo, _work=work, periodic=periodic)
            quotient, remainder = {}, dict(top)
            charge(sum(1 + len(term) for term in top))
            if len(bottom) == 1:
                ((dt, dc),) = bottom.items()
                for term, coefficient in top.items():
                    counts = Counter(term)
                    needed = Counter(dt)
                    if needed <= counts:
                        remaining = tuple(
                            sorted((counts - needed).elements(), key=lambda atom: atom.order_key)
                        )
                        candidate = {remaining: coefficient / dc}
                        if _integral(candidate, facts):
                            quotient = _sum(quotient, candidate)
                            del remainder[term]
            if not remainder:
                result = quotient
            elif remainder == {(): Fraction(-1)} and quotient:
                result = _sum(quotient, {(): Fraction(-1)})
            else:
                # Canonicalize atom operands where integral coefficients allow it.
                if all(c.denominator == 1 for p in (top, bottom) for c in p.values()):
                    numerator = _to_expr(top)
                    divisor = _to_expr(bottom)
                atom = expression("floordiv", numerator, divisor)
                result = {(atom,): Fraction(1)}
    else:
        result = {(expr,): Fraction(1)}
    memo[expr] = result
    return result


def equivalent(a, b, facts):
    return polynomial(a, facts) == polynomial(b, facts)


def normalize(expr, facts):
    return _to_expr(polynomial(expr, facts))


def arguments(expr):
    result, seen, pending = set(), set(), [expr]
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current.kind == "argument":
            result.add(current.args[0])
        else:
            pending.extend(arg for arg in current.args if isinstance(arg, Expr))
    return result


def physical_terms(expr, facts):
    """Separate a uniform coefficient from one physical clock per term."""
    grouped = {}
    for term, coefficient in polynomial(expr, facts, periodic=True).items():
        coordinates = [atom for atom in term if any(n.startswith("$") for n in arguments(atom))]
        require(len(coordinates) <= 1, "address combines unsupported task clocks")
        coordinate = coordinates[0] if coordinates else None
        uniform = list(term)
        if coordinate is not None:
            uniform.remove(coordinate)
        grouped[coordinate] = _sum(grouped.get(coordinate, {}), {tuple(uniform): coefficient})
    return {coordinate: _to_expr(coefficient) for coordinate, coefficient in grouped.items()}


def constant_ratio(a, b, facts):
    left, right = polynomial(a, facts), polynomial(b, facts)
    if not right or left.keys() != right.keys():
        return None
    ratios = {left[term] / value for term, value in right.items()}
    if len(ratios) == 1:
        ratio = next(iter(ratios))
        if ratio.denominator == 1:
            return int(ratio)
    return None


def nonnegative(expr, facts):
    """Sufficient algebraic sign proof; no parameter sampling."""
    from .scalar import infer

    inferred = infer(expr, facts)
    if inferred is not None and inferred.minimum >= 0:
        return True
    if expr.kind == "max" and any(nonnegative(arg, facts) for arg in expr.args):
        return True
    if expr.kind == "min" and all(nonnegative(arg, facts) for arg in expr.args):
        return True
    lower = Fraction(0)
    for term, coefficient in polynomial(expr, facts).items():
        if coefficient < 0 and term:
            return False
        value = coefficient
        for atom in term:
            inferred = infer(atom, facts)
            if inferred is None:
                return False
            value *= inferred.minimum
        lower += value
    return lower >= 0


def _joint_affine_interval(expr, domains, facts):
    """Recover a proved joint coordinate after ordinary polynomial expansion.

    A captured address may spell ``tile*core + tile*groups*query`` rather
    than ``tile*(core + groups*query)``. Independent intervals lose the
    owner/query correlation. Use an exact polynomial identity with a positive
    constant scale and a uniform bias; never infer a new iteration domain.
    """
    candidates = [
        (coordinate, lower, upper)
        for coordinate, lower, upper in domains
        if coordinate.kind in {"add", "mul"}
        and any(name.startswith("$") for name in arguments(coordinate))
    ]
    if not candidates or expr.kind not in {"add", "mul"}:
        return None

    def varying(poly):
        return {
            term: coefficient
            for term, coefficient in poly.items()
            if any(name.startswith("$") for atom in term for name in arguments(atom))
        }

    top = polynomial(expr, facts)
    left = varying(top)
    if not left:
        return None
    for coordinate, lower, upper in candidates:
        bottom = polynomial(coordinate, facts)
        right = varying(bottom)
        if not right or left.keys() != right.keys():
            continue
        ratios = {left[term] / coefficient for term, coefficient in right.items()}
        if len(ratios) != 1:
            continue
        (ratio,) = ratios
        if ratio <= 0 or ratio.denominator != 1:
            continue
        bias = _to_expr(_sum(top, _scale(bottom, -ratio)))
        scale = literal(int(ratio))
        return tuple(
            expression("add", expression("mul", scale, bound), bias) for bound in (lower, upper)
        )
    return None


def interval(expr, domains, facts):
    """Inclusive symbolic range on an active loop iteration.

    Each domain entry is (expression, lower, upper). It can describe a joint
    coordinate such as core + g*groups, preserving cyclic ownership information
    which separate core/g intervals would lose.
    """
    for coordinate, lower, upper in domains:
        if equivalent(expr, coordinate, facts):
            return lower, upper
    correlated = _joint_affine_interval(expr, domains, facts)
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
        if a.constant is not None and a.constant < 0:
            return expression("mul", a, bh), expression("mul", a, bl)
        if b.constant is not None and b.constant < 0:
            return expression("mul", b, ah), expression("mul", b, al)
        require(
            nonnegative(al, facts) and nonnegative(bl, facts),
            "index product has an unproven sign",
        )
        return expression("mul", al, bl), expression("mul", ah, bh)
    if expr.kind in {"floordiv", "ceildiv", "rem"}:
        from .scalar import positive

        require(
            al == literal(0) or nonnegative(al, facts),
            "unsigned index division has an unproven nonnegative dividend",
        )
        require(al == ah or b == bl == bh, "index divisor is not uniform")
        require(positive(b, facts), "index divisor must be provably positive and uniform")
        if expr.kind == "rem":
            return literal(0), expression("sub", b, literal(1))
        return expression(expr.kind, al, b), expression(expr.kind, ah, b)
    require(False, f"unsupported index range operation {expr.kind}")


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
                nonnegative(lower, facts),
                "unsigned index division has an unproven nonnegative dividend",
            )

    divisions(origin)
    lower, upper = interval(origin, domains, facts)
    require(nonnegative(lower, facts), "logical axis origin may be negative")
    remaining = expression("sub", extent, expression("add", upper, literal(width)))
    require(
        nonnegative(remaining, facts), "logical axis access is not bounded by its tensor extent"
    )


def validate_source_arithmetic(program):
    """Check every original division before value/address simplification.

    Even an unused div followed by ``bad-bad`` must satisfy its source domain.
    Walking the retained scopes avoids mistaking a simplified Expr DAG for the
    original SSA. Branch predicates do not silently grant stronger ranges.
    """
    from xdsl.dialects import arith, scf

    from .analysis.iteration import iteration_domain
    from .dialects.source import DomainOp, PipelineOp
    from .scalar import facts_from_sizes, from_ssa, positive

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
    core = Expr("argument", ("$core",))
    number = 0

    def walk(block, coordinates, domains):
        nonlocal number
        for op in block.ops:
            if op not in relevant:
                continue
            if isinstance(op, divisions):
                lhs, rhs = (
                    from_ssa(value, coordinates=coordinates, facts=facts) for value in op.operands
                )
                try:
                    lower, _ = interval(lhs, domains, facts)
                except CompileError as error:
                    raise CompileError(
                        f"{op.location}: source unsigned division: {error}"
                    ) from None
                require(
                    nonnegative(lower, facts),
                    "source unsigned division has an unproven nonnegative dividend",
                )
                require(
                    positive(rhs, facts),
                    "source index divisor must be provably positive and uniform",
                )
            local, ranges = coordinates, domains
            if isinstance(op, (PipelineOp, DomainOp, scf.ForOp)):
                name = f"$index{number}"
                number += 1
                value = op.body.block.args[0]
                if isinstance(op, DomainOp):
                    lower = literal(0)
                    bound = literal(
                        2 if program.kind.data == "mixed" and op.kind.data == "vector" else 1
                    )
                else:
                    domain = iteration_domain(op)
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
        program.body.block, {owner: "$core"}, [(core, literal(0), literal(program.groups.data - 1))]
    )

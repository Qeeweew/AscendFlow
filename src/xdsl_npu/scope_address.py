"""Separate a captured ancestor address from a static child reference.

This is address algebra, not a new storage, ownership or completion contract.
The caller validates original source SSA (including divisions) before algebra,
and supplies clock names from the actual lexical context. Captures must retain
the original allocation identity and the caller's stage/guard activation.
"""

from collections.abc import Collection, Mapping

from .indexing import _to_expr, arguments, physical_terms, polynomial
from .model import require
from .scalar import Expr, Fact, expression, literal


def _replace_argument(expr: Expr, name: str, value: Expr) -> Expr:
    memo = {}

    def visit(current):
        if current in memo:
            return memo[current]
        if current.kind == "argument":
            result = value if current.args[0] == name else current
        elif current.kind == "constant":
            result = current
        else:
            result = expression(current.kind, *(visit(arg) for arg in current.args))
        memo[current] = result
        return result

    return visit(expr)


def _subtract_static_bias(expr: Expr, bias: int) -> Expr:
    """Fold additive constants without expanding quotient/remainder atoms."""
    terms, pending, constant = [], [expr], -bias
    while pending:
        current = pending.pop()
        if current.kind == "add":
            pending.extend(reversed(current.args))
        elif current.constant is not None:
            constant += current.constant
        else:
            terms.append(current)
    result = literal(constant)
    for term in terms:
        result = expression("add", result, term)
    return result


def _check_relative(relative: Expr, child_clock: str | None) -> None:
    names = arguments(relative)
    require(
        names <= ({child_clock} if child_clock is not None else set()),
        "child relative offset still depends on captured variables",
    )
    if child_clock is None:
        require(relative.constant is not None, "ordinary relative offset must be static")
        return
    # physical_terms recognizes '$' task clocks. Validation uses one private
    # spelling; returned expressions retain the caller's actual coordinate name.
    child_name = "$scope_child"
    canonical = _replace_argument(relative, child_clock, Expr("argument", (child_name,)))
    for coordinate, coefficient in physical_terms(canonical, {}).items():
        require(
            coefficient.constant is not None,
            "child index coefficient must be static after ancestor capture",
        )
        if coordinate is None:
            continue
        if coordinate.kind == "rem":
            coordinate, modulus = coordinate.args
            require(modulus.constant is not None, "child storage period must be static")
        if coordinate.kind == "floordiv":
            coordinate, divisor = coordinate.args
            require(divisor.constant is not None, "child task quotient must be static")
        require(
            coordinate == Expr("argument", (child_name,)),
            f"unsupported child index atom {coordinate.kind}",
        )


def split_offset(
    address: Expr,
    child_clock: str | None,
    captured_variables: Collection[str],
    facts: Mapping[str, Fact],
) -> tuple[Expr, Expr]:
    """Return ``(parent_offset, relative_offset)`` without changing root identity.

    ``child_clock`` names the leaf's own coordinate; None denotes ordinary work.
    ``captured_variables`` names actual ancestor/core/lane coordinates. Uniform
    parameter names come from ``facts``. Static bias stays in the relative
    reference, so views with different fixed offsets can share one capture.

    The relative offset must fit existing static affine/div/mod native refs.
    A caller groups refs by allocation, physical actor and source scope and
    checks their parent offsets for equivalence; this function does not promise
    that different calls are disjoint, fresh, initialized or complete.
    """
    require(isinstance(address, Expr), "scope address must be an index Expr")
    require(
        child_clock is None or type(child_clock) is str,
        "child clock must be a coordinate name or None",
    )
    require(
        isinstance(captured_variables, Collection)
        and not isinstance(captured_variables, str)
        and all(type(name) is str for name in captured_variables),
        "captured variables must be coordinate names",
    )
    require(
        isinstance(facts, Mapping)
        and all(type(name) is str and isinstance(fact, Fact) for name, fact in facts.items()),
        "scope address facts must describe declared uniform parameters",
    )
    captured = frozenset(captured_variables) | facts.keys()
    require(child_clock not in captured, "child clock cannot also be captured or uniform")
    allowed = captured | ({child_clock} if child_clock is not None else set())
    unknown = arguments(address) - allowed
    require(not unknown, f"scope address uses uncaptured variables: {', '.join(sorted(unknown))}")

    parent_terms, relative_terms = {}, {}
    for term, coefficient in polynomial(address, facts, periodic=True).items():
        names = set().union(*(arguments(atom) for atom in term))
        if child_clock is not None and child_clock in names:
            for atom in term:
                dependencies = arguments(atom)
                require(
                    child_clock not in dependencies or dependencies <= {child_clock},
                    f"mixed parent/child index atom {atom.kind}: {', '.join(sorted(dependencies))}",
                )
            other = names - {child_clock}
            require(
                not other,
                "child index coefficient depends on runtime captured variables: "
                + ", ".join(sorted(other)),
            )
            relative_terms[term] = coefficient
        elif names:
            parent_terms[term] = coefficient
        else:
            relative_terms[term] = coefficient

    relative = _to_expr(relative_terms)
    if not arguments(relative):
        relative = literal(relative.evaluate({}))
    _check_relative(relative, child_clock)
    if not parent_terms:
        parent = literal(0)
    elif all(coefficient.denominator == 1 for coefficient in parent_terms.values()) and not any(
        atom.kind == "floordiv" for term in parent_terms for atom in term
    ):
        parent = _to_expr(parent_terms)
    else:
        # Proof fractions (Sq/128) and expanded divmod correlations must not
        # become the emitted address. Retain the original quotient/remainder
        # tree, whose range is known, rather than x - d*(x/d). At child=0,
        # subtract the entire static relative value; periodic child atoms may
        # themselves have nonzero bias. The polynomial still proves the split.
        parent = address
        zero_relative = relative
        if child_clock is not None:
            parent = _replace_argument(parent, child_clock, literal(0))
            zero_relative = _replace_argument(relative, child_clock, literal(0))
        parent = _subtract_static_bias(parent, zero_relative.evaluate({}))
        require(
            polynomial(parent, facts, periodic=True) == parent_terms,
            "ancestor capture does not match the canonical parent address",
        )
    require(
        arguments(parent) <= captured,
        "ancestor capture still depends on the child clock",
    )
    return parent, relative

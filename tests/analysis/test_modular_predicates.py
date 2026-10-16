"""Independent small integer oracles for proof-only modular predicates."""

from dataclasses import FrozenInstanceError
from itertools import product
from math import gcd, prod

import pytest

from xdsl_npu.analysis import modular
from xdsl_npu.analysis.modular import (
    ModularAtom,
    ModularPredicate,
    modular_predicate,
    specialize_residues,
)
from xdsl_npu.analysis.structured_schedule import Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, expression, literal

G = Expr("argument", ("g",))
P = Expr("argument", ("p",))
M = Expr("argument", ("m",))


def _rem(value, modulus):
    return expression("rem", value, literal(modulus))


def _sum(values):
    values = tuple(values)
    if len(values) == 1:
        return values[0]
    middle = len(values) // 2
    return expression("add", _sum(values[:middle]), _sum(values[middle:]))


def _holds(predicate, context):
    value = (
        sum(
            coefficient * prod(factor.evaluate(context) for factor in monomial)
            for monomial, coefficient in predicate.atom.terms
        )
        % predicate.atom.modulus
    )
    return value in predicate.residues


def _oracle(test, contexts):
    predicate = modular_predicate(test)
    assert predicate is not None
    for context in contexts:
        assert _holds(predicate, context) == test.evaluate(context)
    return predicate


def test_correlated_parity_siblings_share_one_atom_and_complementary_residues():
    product_remainder = _rem(expression("mul", P, M), 2)
    tests = tuple(
        Comparison(_rem(G, 2), "==", _rem(expression("sub", literal(base), product_remainder), 2))
        for base in (2, 3)
    )
    contexts = tuple(
        dict(zip(("g", "p", "m"), values, strict=True)) for values in product(range(5), repeat=3)
    )
    even, odd = (_oracle(test, contexts) for test in tests)
    assert even.atom == odd.atom == ModularAtom(2, (((G,), 1), ((M, P), 1)))
    assert even.residues == frozenset((0,))
    assert odd.residues == frozenset((1,))
    assert even.residues.isdisjoint(odd.residues)
    assert even.residues | odd.residues == frozenset(range(2))


def test_distributivity_and_same_modulus_remainders_preserve_products():
    expanded = expression("add", expression("mul", G, M), expression("mul", P, M))
    factored = expression("mul", expression("add", G, P), M)
    test = Comparison(_rem(factored, 5), "==", literal(3))
    contexts = tuple(
        dict(zip(("g", "p", "m"), values, strict=True)) for values in product(range(5), repeat=3)
    )
    predicate = _oracle(test, contexts)
    assert predicate == modular_predicate(Comparison(literal(3), "==", _rem(expanded, 5)))
    assert predicate.atom.terms == (((G, M), 1), ((M, P), 1))
    repeated = Comparison(
        _rem(expression("mul", expression("add", G, P), expression("add", G, P)), 5),
        "==",
        literal(2),
    )
    _oracle(repeated, ({"g": g, "p": p} for g, p in product(range(5), repeat=2)))


@pytest.mark.parametrize("modulus", [2, 3, 5, 6, 8, 12, 30, 64])
@pytest.mark.parametrize("predicate", ["==", "!="])
def test_unit_scaling_and_reversing_sides_have_identical_canonical_predicates(modulus, predicate):
    left = expression("add", expression("mul", literal(2), G), expression("mul", literal(3), P))
    right = expression("add", M, literal(1))
    original = Comparison(_rem(left, modulus), predicate, _rem(right, modulus))
    contexts = tuple(
        dict(zip(("g", "p", "m"), values, strict=True)) for values in product(range(4), repeat=3)
    )
    canonical = _oracle(original, contexts)
    assert canonical == modular_predicate(Comparison(original.rhs, predicate, original.lhs))
    for unit in range(1, modulus):
        if gcd(unit, modulus) != 1:
            continue
        scaled = Comparison(
            _rem(expression("mul", literal(unit), left), modulus),
            predicate,
            _rem(expression("mul", literal(unit), right), modulus),
        )
        assert modular_predicate(scaled) == canonical


@pytest.mark.parametrize(
    "modulus,coefficients", [(6, (2, 3)), (10, (2, 5)), (30, (6, 10, 15)), (64, (8, 4, 1))]
)
def test_nonunit_leading_coefficient_has_a_deterministic_unit_orbit(modulus, coefficients):
    symbols = tuple(Expr("argument", (f"v{i}",)) for i in range(len(coefficients)))
    numerator = _sum(
        expression("mul", literal(c), symbol)
        for c, symbol in zip(coefficients, symbols, strict=True)
    )
    original = Comparison(_rem(numerator, modulus), "==", literal(1))
    canonical = modular_predicate(original)
    assert canonical is not None and canonical.atom.modulus == modulus
    for unit in range(1, modulus):
        if gcd(unit, modulus) == 1:
            scaled = Comparison(
                _rem(expression("mul", literal(unit), numerator), modulus), "==", literal(unit)
            )
            assert modular_predicate(scaled) == canonical
    contexts = (
        {f"v{i}": value for i, value in enumerate(values)}
        for values in product(range(3), repeat=len(symbols))
    )
    _oracle(original, contexts)


def test_common_coefficient_gcd_reduces_the_atom_and_the_target_together():
    numerator = expression("add", expression("mul", literal(4), G), literal(2))
    for comparison in ("==", "!="):
        test = Comparison(_rem(numerator, 6), comparison, literal(0))
        predicate = _oracle(test, ({"g": g} for g in range(15)))
        assert predicate == modular_predicate(Comparison(_rem(G, 3), comparison, literal(1)))
        impossible = _oracle(
            Comparison(_rem(numerator, 6), comparison, literal(1)), ({"g": g} for g in range(15))
        )
        assert impossible.atom == ModularAtom(1, ())
        assert impossible.residues == (frozenset() if comparison == "==" else frozenset((0,)))


def test_gcd_reduction_repeats_nested_modular_normalization_with_the_same_budget():
    remainder = _rem(G, 3)
    numerator = expression("mul", literal(2), remainder)
    for comparison in ("==", "!="):
        test = Comparison(_rem(numerator, 6), comparison, literal(2))
        predicate = _oracle(test, ({"g": g} for g in range(20)))
        assert predicate == modular_predicate(Comparison(_rem(G, 3), comparison, literal(1)))
    cancellation = expression("add", numerator, expression("mul", literal(4), G))
    constant = _oracle(
        Comparison(_rem(cancellation, 6), "==", literal(2)), ({"g": g} for g in range(20))
    )
    assert constant == ModularPredicate(ModularAtom(1, ()), frozenset())


@pytest.mark.parametrize("comparison", ["==", "!="])
def test_constant_polynomials_have_one_constant_truth_atom(comparison):
    numerator = expression("add", expression("mul", literal(6), G), literal(3))
    for target in (0, 3):
        test = Comparison(_rem(numerator, 6), comparison, literal(target))
        predicate = _oracle(test, ({"g": g} for g in range(10)))
        truth = (target == 3) == (comparison == "==")
        assert predicate == ModularPredicate(
            ModularAtom(1, ()), frozenset((0,)) if truth else frozenset()
        )
    identical = _oracle(
        Comparison(_rem(expression("mul", G, P), 5), comparison, _rem(expression("mul", P, G), 5)),
        ({"g": 2, "p": 3},),
    )
    assert identical.atom == ModularAtom(1, ())
    assert identical.residues == (frozenset((0,)) if comparison == "==" else frozenset())
    assert modular_predicate(Comparison(_rem(G, 1), comparison, literal(0))) == identical


def test_only_nested_moduli_that_are_multiples_of_the_outer_modulus_are_removed():
    direct = Comparison(_rem(G, 3), "==", literal(1))
    nested = Comparison(_rem(expression("add", _rem(G, 12), literal(6)), 3), "==", literal(1))
    assert modular_predicate(nested) == modular_predicate(direct)
    _oracle(nested, ({"g": g} for g in range(30)))
    unrelated = _rem(G, 4)
    opaque = _oracle(
        Comparison(_rem(unrelated, 3), "==", literal(1)), ({"g": g} for g in range(30))
    )
    assert opaque.atom.terms == (((unrelated,), 1),)
    wide_nested = Comparison(_rem(_rem(G, 128), 2), "==", literal(1))
    assert modular_predicate(wide_nested) == modular_predicate(
        Comparison(_rem(G, 2), "==", literal(1))
    )


def test_other_integer_operations_remain_original_opaque_atoms():
    values = (
        expression("min", G, P),
        expression("max", G, P),
        expression("floordiv", G, literal(2)),
        expression("ceildiv", G, literal(3)),
    )
    for value in values:
        predicate = _oracle(
            Comparison(_rem(value, 5), "!=", literal(2)),
            ({"g": g, "p": p} for g, p in product(range(8), repeat=2)),
        )
        assert predicate.atom.terms == (((value,), 1),)


def test_unsupported_comparisons_and_modulus_shapes_remain_unknown():
    unknown = (
        Comparison(_rem(G, 3), "==", _rem(P, 4)),
        Comparison(_rem(G, 3), "==", P),
        Comparison(expression("rem", G, M), "==", literal(0)),
        Comparison(G, "==", literal(0)),
    )
    assert all(modular_predicate(test) is None for test in unknown)


@pytest.mark.parametrize("comparison", ["==", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale,offset", [(3, -4), (-2, 7), (0, 5)])
def test_static_affine_remainder_comparisons_have_exact_original_residue_masks(
    comparison, reverse, scale, offset
):
    remainder = _rem(expression("add", expression("mul", literal(2), G), literal(1)), 6)
    value = expression("add", expression("mul", literal(scale), remainder), literal(offset))
    test = (
        Comparison(literal(2), comparison, value)
        if reverse
        else Comparison(value, comparison, literal(2))
    )
    _oracle(test, ({"g": g} for g in range(25)))


def test_remainder_equality_outside_its_range_has_constant_truth():
    for target in (-1, 3, 10):
        for comparison in ("==", "!="):
            test = Comparison(_rem(G, 3), comparison, literal(target))
            predicate = _oracle(test, ({"g": g} for g in range(10)))
            assert predicate == ModularPredicate(
                ModularAtom(1, ()), frozenset((0,)) if comparison == "!=" else frozenset()
            )


def test_last_slot_existence_comparison_retains_parent_parity():
    inner = _rem(expression("mul", literal(3), P), 2)
    residue = _rem(expression("sub", literal(2), inner), 2)
    test = Comparison(literal(3), ">", expression("add", literal(2), residue))
    predicate = _oracle(test, ({"p": p} for p in range(20)))
    assert predicate == ModularPredicate(ModularAtom(2, (((P,), 1),)), frozenset((0,)))


def test_bound_scale_offset_and_count_are_interpreted_only_as_mathematical_integers(monkeypatch):
    remainder = _rem(G, 5)
    scale = expression("sub", M, literal(4))
    offset = expression("max", expression("sub", M, literal(1)), literal(0))
    count = expression("ceildiv", offset, literal(2))
    value = expression("add", expression("mul", scale, remainder), offset)
    test = Comparison(value, "<", count)
    with monkeypatch.context() as patch:
        patch.setattr(Expr, "evaluate", lambda *_: pytest.fail("source evaluation in affine query"))
        predicate = modular_predicate(test, bindings={M: 3})
    assert predicate == ModularPredicate(ModularAtom(5, (((G,), 1),)), frozenset((2, 3, 4)))
    for g in range(15):
        assert _holds(predicate, {"g": g}) == test.evaluate({"g": g, "m": 3})
    inner = _rem(expression("mul", M, P), 2)
    last = expression("add", literal(2), _rem(expression("sub", literal(2), inner), 2))
    test = Comparison(count, ">", last)
    predicate = modular_predicate(test, bindings={M: 7})
    assert predicate == ModularPredicate(ModularAtom(2, (((P,), 1),)), frozenset((0,)))
    for p in range(15):
        assert _holds(predicate, {"p": p}) == test.evaluate({"m": 7, "p": p})


def test_unknown_remainder_orderings_and_nonaffine_functions_remain_unknown():
    r, other = _rem(G, 3), _rem(P, 3)
    unknown = (
        Comparison(r, "<", other),
        Comparison(r, ">=", _rem(P, 4)),
        Comparison(expression("add", literal(2), r), "<=", other),
        Comparison(expression("add", r, other), "<", literal(3)),
        Comparison(expression("mul", r, r), "<", literal(3)),
        Comparison(expression("mul", M, r), "<", literal(3)),
        Comparison(expression("add", M, r), "<", literal(3)),
    )
    assert all(modular_predicate(test) is None for test in unknown)


def test_affine_residue_enumeration_checks_modulus_before_any_phase(monkeypatch):
    test = Comparison(expression("add", literal(1), _rem(G, 65)), "<", literal(2))
    with monkeypatch.context() as patch:
        patch.setattr(
            modular,
            "range",
            lambda *_: pytest.fail("enumerated before modulus limit"),
            raising=False,
        )
        with pytest.raises(CompileError, match="modulus") as failure:
            modular_predicate(test)
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT


def test_normalization_never_evaluates_source_or_claims_its_unsigned_domain(monkeypatch):
    numerator = expression("sub", G, literal(2))
    test = Comparison(_rem(numerator, 3), "==", literal(1))
    with monkeypatch.context() as patch:
        patch.setattr(Expr, "evaluate", lambda *_: pytest.fail("source evaluation in proof query"))
        predicate = modular_predicate(test)
    assert predicate == ModularPredicate(ModularAtom(3, (((G,), 1),)), frozenset((0,)))
    # g=0 does not define the original unsigned remainder. Only the caller's
    # valid source region may consume the congruence equivalence.
    with pytest.raises(CompileError, match="nonnegative numerator"):
        test.evaluate({"g": 0})
    _oracle(test, ({"g": g} for g in range(2, 20)))
    assert predicate is not None
    assert hash(predicate) == hash(modular_predicate(test))
    with pytest.raises(FrozenInstanceError):
        predicate.residues = frozenset()


def test_modulus_term_and_work_budgets_are_checked_before_expansion(monkeypatch):
    with pytest.raises(CompileError, match="modulus") as failure:
        modular_predicate(Comparison(_rem(G, 65), "==", literal(0)))
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    symbols = tuple(Expr("argument", (f"x{i}",)) for i in range(65))
    with pytest.raises(CompileError, match="term count") as failure:
        modular_predicate(Comparison(_rem(_sum(symbols), 5), "==", literal(0)))
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    left = _sum(symbols[:64])
    right = _sum(Expr("argument", (f"y{i}",)) for i in range(64))
    test = Comparison(_rem(expression("mul", left, right), 5), "==", literal(0))
    with monkeypatch.context() as patch:
        patch.setattr(
            modular,
            "sorted",
            lambda *_args, **_kwargs: pytest.fail("expanded before work limit"),
            raising=False,
        )
        with pytest.raises(CompileError, match="work") as failure:
            modular_predicate(test)
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert failure.value.phase == "modular-predicate"


def test_query_work_and_memoization_are_scoped_to_each_call():
    symbols = tuple(Expr("argument", (f"x{i}",)) for i in range(20))
    test = Comparison(_rem(_sum(symbols), 5), "==", literal(1))
    first = modular_predicate(test)
    assert first is not None
    for _ in range(30):
        assert modular_predicate(test) == first


@pytest.mark.parametrize("known_g,residue", [(0, 0), (1, 1)])
def test_exact_ordinal_bindings_preserve_correlated_parity(known_g, residue):
    test = Comparison(_rem(expression("add", P, G), 2), "==", literal(0))
    original = modular_predicate(test)
    assert original is not None
    expected = ModularPredicate(ModularAtom(2, (((P,), 1),)), frozenset((residue,)))
    direct = modular_predicate(test, bindings={G: known_g})
    rekeyed = specialize_residues(original, {G: known_g})
    assert direct == rekeyed == expected
    for p in range(10):
        context = {"g": known_g, "p": p}
        assert _holds(rekeyed, context) == test.evaluate(context)


def test_exact_parameter_and_ordinal_bindings_reduce_one_correlated_product():
    right = _rem(expression("sub", literal(2), _rem(expression("mul", P, M), 2)), 2)
    test = Comparison(_rem(G, 2), "==", right)
    original = modular_predicate(test)
    assert original is not None
    for g in (0, 1):
        bindings = {M: 3, G: g}
        expected = ModularPredicate(ModularAtom(2, (((P,), 1),)), frozenset((g,)))
        assert modular_predicate(test, bindings=bindings) == expected
        specialized = specialize_residues(original, bindings)
        assert specialized == expected
        for p in range(10):
            context = {"g": g, "p": p, "m": 3}
            assert _holds(specialized, context) == test.evaluate(context)
        constant = ModularPredicate(ModularAtom(1, ()), frozenset((0,)) if g == 0 else frozenset())
        assert modular_predicate(test, bindings={M: 0, G: g}) == constant
        assert specialize_residues(original, {M: 0, G: g}) == constant


def test_old_masks_shift_and_change_unit_together_under_exact_bindings():
    predicate = ModularPredicate(ModularAtom(5, (((G,), 1), ((P,), 2))), frozenset((0, 2)))
    specialized = specialize_residues(predicate, {G: 2})
    assert specialized == ModularPredicate(ModularAtom(5, (((P,), 1),)), frozenset((0, 4)))
    for p in range(15):
        context = {"g": 2, "p": p}
        assert _holds(specialized, context) == _holds(predicate, context)


@pytest.mark.parametrize(
    "allowed,expected",
    [((1, 2, 4), frozenset((1, 2))), ((1, 3, 5), frozenset()), ((0, 2, 4), frozenset((0,)))],
)
def test_binding_gcd_removes_unreachable_allowed_residues(allowed, expected):
    predicate = ModularPredicate(ModularAtom(6, (((G,), 2), ((P,), 1))), frozenset(allowed))
    specialized = specialize_residues(predicate, {P: 0})
    if allowed == (1, 2, 4):
        assert specialized == ModularPredicate(ModularAtom(3, (((G,), 1),)), expected)
    else:
        assert specialized == ModularPredicate(ModularAtom(1, ()), expected)
    for g in range(15):
        context = {"g": g, "p": 0}
        assert _holds(specialized, context) == _holds(predicate, context)


@pytest.mark.parametrize(
    "allowed,truth", [((0,), True), ((1,), False), ((0, 1), True), ((), False)]
)
def test_all_constant_empty_and_full_masks_use_the_constant_atom(allowed, truth):
    predicate = ModularPredicate(ModularAtom(2, (((G,), 1), ((P,), 1))), frozenset(allowed))
    specialized = specialize_residues(predicate, {G: 1, P: 1})
    assert specialized == ModularPredicate(
        ModularAtom(1, ()), frozenset((0,)) if truth else frozenset()
    )
    if allowed in ((0,), (1,)):
        test = Comparison(_rem(expression("add", G, P), 2), "==", literal(allowed[0]))
        assert modular_predicate(test, bindings={G: 1, P: 1}) == specialized


def test_exact_bindings_can_fix_an_original_opaque_integer_factor():
    quotient = expression("floordiv", expression("add", G, P), literal(2))
    test = Comparison(_rem(quotient, 5), "==", literal(1))
    predicate = modular_predicate(test)
    assert predicate is not None
    true = ModularPredicate(ModularAtom(1, ()), frozenset((0,)))
    assert specialize_residues(predicate, {quotient: 1}) == true
    assert specialize_residues(predicate, {G: 1, P: 2}) == true
    assert modular_predicate(test, bindings={quotient: 1}) == true
    assert modular_predicate(test, bindings={G: 1, P: 2}) == true


def test_binding_can_fix_the_original_outer_remainder_expression():
    remainder = _rem(G, 3)
    test = Comparison(remainder, "==", literal(2))
    assert modular_predicate(test, bindings={remainder: 2}) == ModularPredicate(
        ModularAtom(1, ()), frozenset((0,))
    )


def test_signed_binding_representatives_do_not_construct_or_evaluate_source(monkeypatch):
    test = Comparison(_rem(expression("sub", G, literal(1)), 2), "==", literal(1))
    predicate = modular_predicate(test)
    assert predicate is not None
    with monkeypatch.context() as patch:
        patch.setattr(Expr, "evaluate", lambda *_: pytest.fail("source evaluation in bound query"))
        direct = modular_predicate(test, bindings={G: 0})
        rekeyed = specialize_residues(predicate, {G: 0})
    assert direct == rekeyed == ModularPredicate(ModularAtom(1, ()), frozenset((0,)))
    with pytest.raises(CompileError, match="nonnegative numerator"):
        test.evaluate({"g": 0})


def test_binding_queries_have_independent_budgets_and_keep_unbound_factors():
    test = Comparison(_rem(expression("add", G, P), 2), "==", literal(0))
    predicate = modular_predicate(test)
    assert predicate is not None
    for g in (0, 1, 0, 1):
        expected = ModularPredicate(ModularAtom(2, (((P,), 1),)), frozenset((g,)))
        assert modular_predicate(test, bindings={G: g}) == expected
        assert specialize_residues(predicate, {G: g}) == expected
    assert specialize_residues(predicate, {}) == predicate
    with pytest.raises(CompileError, match="exact integers"):
        modular_predicate(test, bindings={G: True})
    with pytest.raises(CompileError, match="exact integers"):
        specialize_residues(predicate, {G: literal(0)})

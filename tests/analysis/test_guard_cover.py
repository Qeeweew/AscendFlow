"""Local guard covers checked against independent finite integer examples.

These are mathematical obligations, without notification selection or resource
allocation. In particular, period five here makes no claim about a real memory
frontier accepting a five-slot operator.
"""

from itertools import product

import pytest

from ascendflow.analysis.guard_cover import prove_guard_cover
from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.scalar import Expr, Fact, expression, literal

M, P, G, X, Y = (Expr("argument", (name,)) for name in ("m", "p", "g", "x", "y"))
WORD = expression("add", expression("mul", P, M), G)


def comparison(value, predicate, point):
    return Comparison(value, predicate, literal(point) if type(point) is int else point)


def guard(*tests):
    return ActionGuard(tuple(tests))


def slot_problem(period=2):
    assumptions = (
        comparison(M, ">=", 3),
        comparison(P, ">=", 0),
        comparison(G, ">=", 0),
        comparison(expression("add", G, literal(2)), "<", M),
    )
    residue = expression("rem", WORD, literal(period))
    alternatives = tuple(
        guard(comparison(M, predicate, 3), comparison(residue, "==", slot))
        for predicate, slot in product(("==", ">"), range(period))
    )
    return assumptions, alternatives


def finite_slot_inputs():
    for m in range(3, 11):
        for p in range(4):
            for g in range(m - 2):
                yield {"m": m, "p": p, "g": g}


def covered_alternative(proof, values):
    """Evaluate the original guards independently of SMT and its encoding."""
    assert all(test.evaluate(values) for test in proof.assumptions)
    matches = [i for i, guard in enumerate(proof.alternatives) if guard.evaluate(values)]
    assert matches
    return matches[0]


@pytest.mark.parametrize("period", (2, 5))
def test_parameter_and_whole_slot_expression_have_a_complete_local_cover(period):
    assumptions, alternatives = slot_problem(period)
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    assert proof is not None
    assert proof.assumptions == assumptions
    assert proof.alternatives == alternatives
    assert proof.answer.proved
    for values in finite_slot_inputs():
        m, p, g = (values[name] for name in ("m", "p", "g"))
        expected = (0 if m == 3 else period) + (p * m + g) % period
        assert all(test.evaluate(values) for test in assumptions)
        assert [i for i, h in enumerate(alternatives) if h.evaluate(values)] == [expected]
        assert covered_alternative(proof, values) == expected


@pytest.mark.parametrize(
    ("missing", "values"),
    (
        (0, {"m": 3, "p": 0, "g": 0}),
        (1, {"m": 3, "p": 1, "g": 0}),
        (2, {"m": 4, "p": 0, "g": 0}),
        (3, {"m": 4, "p": 0, "g": 1}),
    ),
)
def test_every_parameter_and_slot_branch_is_required(missing, values):
    assumptions, alternatives = slot_problem()
    incomplete = alternatives[:missing] + alternatives[missing + 1 :]
    assert values["m"] >= 3 and values["g"] + 2 < values["m"]
    assert (0 if values["m"] == 3 else 2) + (values["p"] * values["m"] + values["g"]) % 2 == missing
    assert all(test.evaluate(values) for test in assumptions)
    assert not any(h.evaluate(values) for h in incomplete)
    assert prove_guard_cover(IntegerFacts({}), assumptions, incomplete) is None


def test_different_moduli_cannot_share_a_residue_partition():
    assumptions, _ = slot_problem()
    alternatives = (
        guard(comparison(expression("rem", WORD, literal(2)), "==", 0)),
        guard(comparison(expression("rem", WORD, literal(3)), "==", 1)),
    )
    values = {"m": 3, "p": 1, "g": 0}
    assert (values["p"] * values["m"] + values["g"]) % 2 == 1
    assert (values["p"] * values["m"] + values["g"]) % 3 == 0
    assert all(test.evaluate(values) for test in assumptions)
    assert not any(h.evaluate(values) for h in alternatives)
    assert prove_guard_cover(IntegerFacts({}), assumptions, alternatives) is None


def test_a_shifted_expression_keeps_its_actual_residue_meaning():
    assumptions, _ = slot_problem()
    shifted = expression("add", WORD, literal(1))
    alternatives = (
        guard(comparison(expression("rem", WORD, literal(2)), "==", 0)),
        guard(comparison(expression("rem", shifted, literal(2)), "==", 1)),
    )
    values = {"m": 3, "p": 1, "g": 0}
    assert not any(h.evaluate(values) for h in alternatives)
    assert prove_guard_cover(IntegerFacts({}), assumptions, alternatives) is None


def test_distinct_polynomials_do_not_borrow_each_others_allowed_residues():
    assumptions = (comparison(X, ">=", 0), comparison(Y, ">=", 0))
    alternatives = (
        guard(comparison(expression("rem", X, literal(2)), "==", 0)),
        guard(comparison(expression("rem", Y, literal(2)), "==", 1)),
    )
    values = {"x": 1, "y": 0}
    assert not any(h.evaluate(values) for h in alternatives)
    assert prove_guard_cover(IntegerFacts({}), assumptions, alternatives) is None


def test_inequality_keeps_its_noncontiguous_residue_set():
    assumptions, _ = slot_problem()
    residue = expression("rem", WORD, literal(5))
    alternatives = (
        guard(comparison(residue, "!=", 3)),
        guard(comparison(residue, "==", 3)),
    )
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    assert proof is not None
    assert proof.answer.proved
    for values in finite_slot_inputs():
        expected = int((values["p"] * values["m"] + values["g"]) % 5 == 3)
        assert covered_alternative(proof, values) == expected


def test_reduced_modulus_predicates_share_only_a_proved_canonical_atom():
    assumptions, _ = slot_problem()
    twice = expression("mul", literal(2), WORD)
    alternatives = (
        guard(comparison(expression("rem", twice, literal(4)), "==", 0)),
        guard(comparison(expression("rem", WORD, literal(2)), "==", 1)),
    )
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    assert proof is not None
    assert proof.answer.proved
    for values in finite_slot_inputs():
        expected = (values["p"] * values["m"] + values["g"]) % 2
        assert covered_alternative(proof, values) == expected


@pytest.mark.parametrize("missing_odd", (False, True))
def test_a_later_exact_integer_rekeys_the_existing_residue_branch(missing_odd):
    assumptions, _ = slot_problem()
    combined = expression("rem", expression("add", P, G), literal(2))
    parent = expression("rem", P, literal(2))
    alternatives = tuple(
        guard(
            comparison(combined, "==", residue),
            comparison(M, "==", 3),
            comparison(parent, "==", residue),
        )
        for residue in range(1 if missing_odd else 2)
    ) + (guard(comparison(M, ">", 3)),)
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    if missing_odd:
        values = {"m": 3, "p": 1, "g": 0}
        assert all(test.evaluate(values) for test in assumptions)
        assert not any(h.evaluate(values) for h in alternatives)
        assert proof is None
        return
    assert proof is not None
    for values in finite_slot_inputs():
        # At m=3, 0 <= g and g+2<m force g=0. A residue split made
        # before establishing m=3 must keep that exact-value correlation.
        expected = values["p"] % 2 if values["m"] == 3 else 2
        assert covered_alternative(proof, values) == expected


def test_integer_thresholds_preserve_the_equality_branch():
    assumptions = (comparison(X, ">=", 0),)
    incomplete = (guard(comparison(X, "<", 3)), guard(comparison(X, ">", 3)))
    assert not any(h.evaluate({"x": 3}) for h in incomplete)
    assert prove_guard_cover(IntegerFacts({}), assumptions, incomplete) is None
    complete = incomplete + (guard(comparison(X, "==", 3)),)
    proof = prove_guard_cover(IntegerFacts({}), assumptions, complete)
    assert proof is not None
    for x in range(8):
        assert covered_alternative(proof, {"x": x}) == (0 if x < 3 else 2 if x == 3 else 1)


@pytest.mark.parametrize("complete", (False, True))
def test_shifted_loop_end_equality_terminates_and_keeps_the_uncovered_tail(complete):
    count = expression(
        "ceildiv", expression("max", expression("sub", literal(128), G), literal(0)), literal(20)
    )
    next_end = expression("add", X, literal(2))
    assumptions = (
        comparison(G, ">=", 0),
        comparison(G, "<", 20),
        comparison(X, ">=", 0),
        comparison(count, ">", expression("add", X, literal(1))),
    )
    alternatives = (guard(comparison(count, ">", next_end)),)
    if complete:
        alternatives += (guard(comparison(count, "==", next_end)),)
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    if not complete:
        assert proof is None
    else:
        assert proof is not None
        for group in range(20):
            for index in range(8):
                values = {"g": group, "x": index}
                if all(test.evaluate(values) for test in assumptions):
                    assert covered_alternative(proof, values) == int(
                        count.evaluate(values) == index + 2
                    )


@pytest.mark.parametrize("reverse", (False, True))
def test_integer_trichotomy_covers_signed_values_and_reversed_operands(reverse):
    alternatives = tuple(
        guard(comparison(literal(0), swapped, X) if reverse else comparison(X, predicate, 0))
        for predicate, swapped in (("<", ">"), ("==", "=="), (">", "<"))
    )
    proof = prove_guard_cover(IntegerFacts({}), (), alternatives)
    assert proof is not None
    for x in range(-5, 6):
        assert covered_alternative(proof, {"x": x}) == (0 if x < 0 else 1 if x == 0 else 2)


def test_the_next_obligation_cannot_borrow_a_previous_residue_state():
    arithmetic = IntegerFacts({})
    residue = expression("rem", X, literal(2))
    nonnegative = comparison(X, ">=", 0)
    even, odd = (comparison(residue, "==", i) for i in range(2))
    alternatives = (guard(even),)
    first = (nonnegative, even)
    second = (nonnegative, odd)
    assert prove_guard_cover(arithmetic, first, alternatives) is not None
    assert prove_guard_cover(arithmetic, second, alternatives) is None
    assert prove_guard_cover(arithmetic, first, alternatives) is not None


def test_each_alternative_supplies_its_own_ordered_unsigned_prefix():
    assumptions = (comparison(X, ">=", 0),)
    prerequisite = comparison(X, ">=", 1)
    residue = expression("rem", expression("sub", X, literal(1)), literal(2))
    alternatives = tuple(guard(prerequisite, comparison(residue, "==", r)) for r in range(2)) + (
        guard(comparison(X, "==", 0)),
    )
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    assert proof is not None
    for x in range(9):
        expected = 2 if x == 0 else (x - 1) % 2
        assert [i for i, h in enumerate(alternatives) if h.evaluate({"x": x})] == [expected]
        assert covered_alternative(proof, {"x": x}) == expected


@pytest.mark.parametrize("invalid_assumptions", (False, True))
def test_unsigned_definedness_is_checked_before_normalization_or_coverage(invalid_assumptions):
    arithmetic = IntegerFacts({"x": Fact()})
    numerator = expression("sub", X, literal(1))
    # Mathematical mod-one simplification must not conceal x=0's invalid use.
    use = comparison(expression("rem", numerator, literal(1)), "==", 0)
    prerequisite = comparison(X, ">=", 1)
    assumptions = (use, prerequisite) if invalid_assumptions else (comparison(X, ">=", 0),)
    alternatives = (guard(),) if invalid_assumptions else (guard(), guard(use, prerequisite))
    with pytest.raises(CompileError) as raised:
        prove_guard_cover(arithmetic, assumptions, alternatives)
    assert raised.value.kind is ErrorKind.UNSUPPORTED
    assert raised.value.phase == "event-guard-cover"


def test_a_proved_false_prefix_leaves_its_unsigned_tail_unreachable():
    assumptions = (comparison(X, ">=", 0),)
    never = comparison(expression("rem", X, literal(1)), "!=", 0)
    unsafe = comparison(expression("rem", expression("sub", X, literal(1)), literal(2)), "==", 0)
    alternatives = (guard(never, unsafe), guard())
    proof = prove_guard_cover(IntegerFacts({}), assumptions, alternatives)
    assert proof is not None
    assert not alternatives[0].evaluate({"x": 0})
    assert covered_alternative(proof, {"x": 0}) == 1


def test_empty_alternatives_do_not_cover_a_possible_obligation():
    assert prove_guard_cover(IntegerFacts({}), (), ()) is None


def test_an_impossible_obligation_keeps_its_contradiction_witness():
    assumptions = (comparison(X, ">=", 0), comparison(X, "<", 0))
    proof = prove_guard_cover(IntegerFacts({}), assumptions, ())
    assert proof is not None
    assert proof.answer.proved


def test_an_empty_conjunction_covers_a_possible_obligation():
    proof = prove_guard_cover(IntegerFacts({}), (), (guard(),))
    assert proof is not None
    assert covered_alternative(proof, {}) == 0

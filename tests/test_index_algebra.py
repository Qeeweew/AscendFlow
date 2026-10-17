"""Symbolic index contracts with independent numerical counterexamples.

The finite values below check interpretation of the proved identities. They do
not replace the symbolic range and ownership obligations with parameter samples.
"""

import pytest

from ascendflow import indexing
from ascendflow.diagnostics import CompileError
from ascendflow.indexing import check_axis, equivalent, physical_terms, polynomial
from ascendflow.scalar import Expr, Fact, expression, literal


def argument(name):
    return Expr("argument", (name,))


def split(task, heads, tiles):
    quotient = expression("floordiv", task, tiles)
    row = expression("rem", task, tiles)
    head = expression("rem", quotient, heads)
    batch = expression("floordiv", task, expression("mul", heads, tiles))
    rebuilt = expression(
        "add",
        expression("mul", expression("add", expression("mul", batch, heads), head), tiles),
        row,
    )
    return batch, head, row, rebuilt


def evaluate_physical(terms, values):
    return sum(
        coefficient.evaluate(values) * (1 if coordinate is None else coordinate.evaluate(values))
        for coordinate, coefficient in terms.items()
    )


def test_runtime_euclidean_split_recombines_symbolically_and_matches_divmod():
    task, heads, tiles = (argument(name) for name in ("$g", "heads", "tiles"))
    facts = {"heads": Fact(minimum=1), "tiles": Fact(minimum=1)}
    batch, head, row, rebuilt = split(task, heads, tiles)
    assert equivalent(rebuilt, task, facts)
    assert physical_terms(rebuilt, facts) == {task: literal(1)}
    for head_count, tile_count, flat in [(1, 1, 0), (3, 5, 4), (3, 5, 5), (3, 5, 29), (2, 7, 51)]:
        quotient, expected_row = divmod(flat, tile_count)
        expected_batch, expected_head = divmod(quotient, head_count)
        values = {"$g": flat, "heads": head_count, "tiles": tile_count}
        assert tuple(value.evaluate(values) for value in (batch, head, row)) == (
            expected_batch,
            expected_head,
            expected_row,
        )
        assert rebuilt.evaluate(values) == flat


def test_nested_positive_quotients_and_last_index_have_exact_integer_identities():
    task, a, b = (argument(name) for name in ("$g", "a", "b"))
    facts = {"a": Fact(minimum=1), "b": Fact(minimum=1)}
    nested = expression("floordiv", expression("floordiv", task, a), b)
    direct = expression("floordiv", task, expression("mul", a, b))
    assert equivalent(nested, direct, facts)
    last = expression("floordiv", expression("sub", expression("mul", a, b), literal(1)), a)
    assert equivalent(last, expression("sub", b, literal(1)), facts)
    for left, right, flat in [(1, 1, 0), (3, 5, 14), (3, 5, 15), (4, 7, 83)]:
        values = {"$g": flat, "a": left, "b": right}
        assert nested.evaluate(values) == flat // left // right
        assert direct.evaluate(values) == flat // (left * right)
        assert last.evaluate(values) == (left * right - 1) // left


def test_large_rational_coefficients_never_pass_through_floating_point():
    value, coefficient = argument("x"), (1 << 60) + 3
    facts = {"x": Fact(multiple=3)}
    divided = expression("floordiv", expression("mul", literal(coefficient), value), literal(3))
    factored = expression("mul", literal(coefficient), expression("floordiv", value, literal(3)))
    wrong = expression("floordiv", expression("mul", literal(coefficient + 1), value), literal(3))
    assert equivalent(divided, factored, facts)
    assert not equivalent(divided, wrong, facts)
    for x in (3, 6, 15):
        assert divided.evaluate({"x": x}) == factored.evaluate({"x": x}) == coefficient * x // 3
        assert wrong.evaluate({"x": x}) != divided.evaluate({"x": x})


def test_axis_bounds_preserve_each_mixed_radix_axis_and_complete_tiles():
    task, tiles = argument("$g"), argument("tiles")
    heads, batches, tile = literal(3), literal(2), literal(16)
    facts = {"tiles": Fact(minimum=1)}
    batch, head, row, _ = split(task, heads, tiles)
    total = expression("mul", expression("mul", batches, heads), tiles)
    domains = [(task, literal(0), expression("sub", total, literal(1)))]
    check_axis(batch, 1, batches, domains, facts)
    check_axis(head, 1, heads, domains, facts)
    check_axis(expression("mul", row, tile), 16, expression("mul", tiles, tile), domains, facts)
    for tile_count, flat in [(1, 5), (2, 7), (5, 29)]:
        quotient, expected_row = divmod(flat, tile_count)
        expected_batch, expected_head = divmod(quotient, 3)
        assert 0 <= expected_batch < 2
        assert 0 <= expected_head < 3
        assert 0 <= expected_row * 16 <= tile_count * 16 - 16
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(row, 1, heads, domains, facts)
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(expression("floordiv", task, tiles), 1, batches, domains, facts)


def test_changed_decode_radix_is_not_equivalent_to_original_flat_task():
    task, tiles, heads = argument("$g"), argument("tiles"), literal(3)
    facts = {"tiles": Fact(minimum=1)}
    batch, _, row, _ = split(task, heads, tiles)
    wrong_head = expression(
        "rem", expression("floordiv", task, expression("add", tiles, literal(1))), heads
    )
    rebuilt = expression(
        "add",
        expression("mul", expression("add", expression("mul", batch, heads), wrong_head), tiles),
        row,
    )
    assert not equivalent(rebuilt, task, facts)
    values = {"$g": 5, "tiles": 5}
    assert rebuilt.evaluate(values) == 0
    assert values["$g"] == 5


def test_a_multiple_is_a_required_contract_and_cannot_be_invented_by_algebra():
    n = argument("n")
    rounded = expression("mul", expression("floordiv", n, literal(4)), literal(4))
    origin = expression("sub", n, literal(4))
    aligned = {"n": Fact(minimum=4, multiple=4)}
    unaligned = {"n": Fact(minimum=4)}
    assert equivalent(rounded, n, aligned)
    check_axis(origin, 4, rounded, [], aligned)
    # A fresh proof with weaker facts must not inherit the previous divisibility.
    assert not equivalent(rounded, n, unaligned)
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(origin, 4, rounded, [], unaligned)
    for length in (4, 8, 20):
        assert (length // 4) * 4 == length
    length = 5
    allocation = (length // 4) * 4
    assert length - 4 + 4 > allocation


@pytest.mark.parametrize("kind", ["floordiv", "rem"])
def test_unproven_positive_divisor_cannot_authorize_split_identity_or_axis(kind):
    task, divisor = argument("$g"), argument("divisor")
    value = expression(kind, task, divisor)
    facts = {"divisor": Fact(multiple=4)}
    domains = [(task, literal(0), literal(7))]
    with pytest.raises(CompileError, match="divisor.*positive"):
        equivalent(value, task, facts)
    with pytest.raises(CompileError, match="divisor.*positive"):
        check_axis(value, 1, literal(8), domains, facts)


def test_nonuniform_task_coordinate_is_not_a_runtime_radix():
    task = argument("$g")
    divisor = expression("add", task, literal(1))
    value = expression("rem", task, divisor)
    with pytest.raises(CompileError, match="divisor.*positive"):
        check_axis(value, 1, literal(8), [(task, literal(0), literal(7))], {})


@pytest.mark.parametrize("kind", ["floordiv", "rem"])
def test_negative_unsigned_dividend_is_rejected_on_its_original_axis(kind):
    task, divisor = argument("$g"), argument("divisor")
    numerator = expression("sub", task, literal(1))
    value = expression(kind, numerator, divisor)
    facts = {"divisor": Fact(minimum=1)}
    with pytest.raises(CompileError, match="nonnegative dividend"):
        check_axis(value, 1, divisor, [(task, literal(0), literal(7))], facts)
    assert 0 - 1 < 0


def test_flat_recombination_does_not_authorize_a_negative_unsigned_intermediate():
    task, divisor = argument("$g"), argument("divisor")
    numerator = expression("sub", task, literal(1))
    quotient = expression("floordiv", numerator, divisor)
    remainder = expression("rem", numerator, divisor)
    rebuilt = expression(
        "add", expression("add", expression("mul", quotient, divisor), remainder), literal(1)
    )
    facts = {"divisor": Fact(minimum=1)}
    assert equivalent(rebuilt, task, facts)
    with pytest.raises(CompileError, match="nonnegative dividend"):
        check_axis(remainder, 1, divisor, [(task, literal(0), literal(7))], facts)


def test_child_generation_cannot_select_an_unrelated_parent_axis():
    parent, task = argument("$call"), argument("$g")
    parents, children = argument("parents"), argument("children")
    facts = {"parents": Fact(minimum=1), "children": Fact(minimum=1)}
    domains = [
        (parent, literal(0), expression("sub", parents, literal(1))),
        (task, literal(0), expression("sub", children, literal(1))),
    ]
    check_axis(parent, 1, parents, domains, facts)
    check_axis(task, 1, children, domains, facts)
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(task, 1, parents, domains, facts)
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(parent, 1, children, domains, facts)
    parent_count, child_count, child = 2, 5, 4
    assert child < child_count and child >= parent_count


def test_flat_address_in_allocation_cannot_hide_an_out_of_range_head_axis():
    tiles = argument("tiles")
    facts = {"tiles": Fact(minimum=1)}
    invalid_head, heads, batches = literal(3), literal(3), literal(2)
    flat = expression("mul", invalid_head, tiles)
    total = expression("mul", expression("mul", batches, heads), tiles)
    next_batch = expression("mul", expression("mul", literal(1), heads), tiles)
    assert equivalent(flat, next_batch, facts)
    check_axis(flat, 1, total, [], facts)
    with pytest.raises(CompileError, match="not bounded"):
        check_axis(invalid_head, 1, heads, [], facts)
    batch, head, tile_count = 0, 3, 5
    address = (batch * 3 + head) * tile_count
    assert address == (1 * 3 + 0) * tile_count
    assert address < 2 * 3 * tile_count and head >= 3


def test_physical_terms_preserve_uniform_parent_strides_and_cyclic_core_mapping():
    task, core, parent = (argument(name) for name in ("$g", "$core", "$call"))
    parents, tiles = argument("parents"), argument("tiles")
    frame = expression("mul", tiles, literal(32))
    address = expression(
        "add",
        expression(
            "add",
            expression("mul", parent, frame),
            expression("mul", core, expression("mul", parents, frame)),
        ),
        expression("mul", task, literal(64)),
    )
    facts = {"parents": Fact(minimum=1), "tiles": Fact(minimum=1)}
    terms = physical_terms(address, facts)
    assert set(terms) == {task, core, parent}
    for invocation_count, tile_count, c, call, g in [(2, 3, 0, 1, 4), (5, 7, 2, 3, 1)]:
        values = {
            "parents": invocation_count,
            "tiles": tile_count,
            "$core": c,
            "$call": call,
            "$g": g,
        }
        expected = (c * invocation_count + call) * tile_count * 32 + g * 64
        assert evaluate_physical(terms, values) == expected
    flat = expression("add", core, expression("mul", task, literal(3)))
    cyclic = physical_terms(expression("mul", flat, literal(64)), {})
    assert cyclic == {core: literal(64), task: literal(192)}
    assert evaluate_physical(cyclic, {"$core": 2, "$g": 4}) == (2 + 3 * 4) * 64


def test_physical_terms_keep_static_ring_clocks_without_turning_radices_into_periods():
    task = argument("$g")
    ring = expression("rem", task, literal(3))
    divided_ring = expression("rem", expression("floordiv", task, literal(2)), literal(4))
    address = expression(
        "add",
        literal(16),
        expression(
            "add",
            expression("mul", ring, literal(32)),
            expression("mul", divided_ring, literal(128)),
        ),
    )
    terms = physical_terms(address, {})
    assert terms == {None: literal(16), ring: literal(32), divided_ring: literal(128)}
    for generation in (0, 1, 3, 7, 11):
        expected = 16 + (generation % 3) * 32 + ((generation // 2) % 4) * 128
        assert evaluate_physical(terms, {"$g": generation}) == expected


@pytest.mark.parametrize("other", ["$g", "$core", "$call"])
def test_physical_address_cannot_multiply_task_clocks(other):
    address = expression("mul", argument("$g"), argument(other))
    with pytest.raises(CompileError, match="unsupported task clocks"):
        physical_terms(address, {})


def test_term_and_product_budgets_reject_before_unbounded_distribution(monkeypatch):
    monkeypatch.setattr(indexing, "MAX_TERMS", 4)
    atoms = [argument(name) for name in "abcdef"]
    four = literal(0)
    for atom in atoms[:4]:
        four = expression("add", four, atom)
    assert len(polynomial(four, {})) == 4
    with pytest.raises(CompileError, match="term budget"):
        polynomial(expression("add", four, atoms[4]), {})
    with pytest.raises(CompileError, match="product budget"):
        polynomial(expression("mul", four, expression("add", atoms[4], atoms[5])), {})


def test_cumulative_budget_limits_repeated_small_normal_forms(monkeypatch):
    base = expression("add", argument("a"), argument("b"))
    value = literal(0)
    for coefficient in range(2, 14):
        value = expression("add", value, expression("mul", literal(coefficient), base))
    # Every normal form has two terms, and each product has only two pairs.
    # The accumulated work still needs a separate, per-proof bound.
    assert len(polynomial(value, {})) == 2
    monkeypatch.setattr(indexing, "MAX_WORK", 16)
    with pytest.raises(CompileError, match="index algebra.*work.*budget"):
        polynomial(value, {})


def _congruent_quotient_products():
    task, size = argument("$g"), argument("size")
    quotient = expression("floordiv", size, literal(4))
    first = expression("sub", task, literal(1))
    second = expression("sub", expression("add", first, literal(1)), literal(1))
    return (
        {"size": Fact(minimum=1, multiple=4)},
        expression("mul", first, quotient),
        expression("mul", second, quotient),
    )


@pytest.mark.parametrize("kind", ("rem", "floordiv", "ceildiv", "min", "max"))
def test_function_atoms_share_integer_argument_identity_without_emitting_fractions(kind):
    facts, first, second = _congruent_quotient_products()
    lhs, rhs = (expression(kind, value, literal(2)) for value in (first, second))
    for g, size in ((1, 4), (2, 4), (3, 12), (4, 8), (8, 28)):
        value = (g - 1) * (size // 4)
        expected = {
            "rem": value % 2,
            "floordiv": value // 2,
            "ceildiv": (value + 1) // 2,
            "min": min(value, 2),
            "max": max(value, 2),
        }[kind]
        assert lhs.evaluate({"$g": g, "size": size}) == expected
        assert rhs.evaluate({"$g": g, "size": size}) == expected
    assert equivalent(lhs, rhs, facts)
    assert indexing.normalize(expression("sub", lhs, rhs), facts).constant == 0


@pytest.mark.parametrize("kind", ("min", "max"))
def test_commutative_atom_identity_survives_raw_argument_order_changes(kind):
    z, a = argument("z"), argument("a")
    same_z = expression("sub", expression("add", z, literal(1)), literal(1))
    lhs, rhs = expression(kind, z, a), expression(kind, same_z, a)
    assert lhs.args[0] == a and rhs.args[0] == same_z
    for zv, av in ((-2, 3), (0, 0), (5, 1)):
        expected = min(zv, av) if kind == "min" else max(zv, av)
        assert lhs.evaluate({"z": zv, "a": av}) == rhs.evaluate({"z": zv, "a": av}) == expected
    assert equivalent(lhs, rhs, {})
    assert indexing.normalize(expression("sub", lhs, rhs), {}).constant == 0


@pytest.mark.parametrize("difference", ("numerator", "divisor", "kind"))
def test_function_atom_identity_preserves_every_actual_argument_and_operation(difference):
    facts, value, _ = _congruent_quotient_products()
    if difference == "kind":
        lhs = expression("floordiv", value, literal(2))
        rhs = expression("ceildiv", value, literal(2))
        g = 2
    else:
        lhs = expression("rem", value, literal(2))
        rhs = expression(
            "rem",
            expression("add", value, literal(1)) if difference == "numerator" else value,
            literal(3 if difference == "divisor" else 2),
        )
        g = 3 if difference == "divisor" else 2
    values = {"$g": g, "size": 4}
    assert lhs.evaluate(values) != rhs.evaluate(values)
    assert not equivalent(lhs, rhs, facts)


def test_function_atom_keys_remain_scoped_to_the_declared_divisibility_facts():
    size = argument("size")
    rounded = expression("mul", expression("floordiv", size, literal(4)), literal(4))
    lhs, rhs = (expression("rem", value, literal(2)) for value in (rounded, size))
    aligned, unaligned = {"size": Fact(minimum=4, multiple=4)}, {"size": Fact(minimum=4)}
    assert equivalent(lhs, rhs, aligned)
    assert lhs.evaluate({"size": 5}) != rhs.evaluate({"size": 5})
    assert not equivalent(lhs, rhs, unaligned)
    assert equivalent(lhs, rhs, aligned)


@pytest.mark.parametrize("wrapper", ("zero_product", "remainder_one"))
def test_axis_checks_raw_unsigned_atoms_before_their_mathematical_zero(wrapper):
    facts, numerator, _ = _congruent_quotient_products()
    task = argument("$g")
    inner = expression("rem", numerator, literal(2))
    origin = (
        Expr("mul", (literal(0), inner))
        if wrapper == "zero_product"
        else expression("rem", inner, literal(1))
    )
    assert equivalent(origin, literal(0), facts)
    with pytest.raises(CompileError, match="nonnegative dividend|unproven sign"):
        check_axis(origin, 1, literal(1), [(task, literal(0), literal(3))], facts)
    with pytest.raises(CompileError, match="nonnegative numerator"):
        origin.evaluate({"$g": 0, "size": 4})
    check_axis(origin, 1, literal(1), [(task, literal(1), literal(3))], facts)

"""Immutable scalar DAG identity, deterministic ordering and bounded visits."""

from dataclasses import FrozenInstanceError

import pytest

from ascendflow import scalar
from ascendflow.diagnostics import CompileError
from ascendflow.indexing import arguments, normalize
from ascendflow.memo import memo_scope
from ascendflow.scalar import Expr, Fact, expression, literal, substitute, substitution


def _argument(name):
    return Expr("argument", (name,))


def _doubled(value, depth):
    for _ in range(depth):
        value = Expr("add", (value, value))
    return value


def test_equal_independent_structures_keep_equal_hashes_without_rehashing_children(monkeypatch):
    first = _doubled(_argument("n"), 8)
    second = _doubled(_argument("n"), 8)
    assert first is not second and first == second
    assert first.order_key == second.order_key
    original = Expr.__hash__
    calls = []

    def observed(node):
        calls.append(id(node))
        return original(node)

    monkeypatch.setattr(Expr, "__hash__", observed)
    assert hash(first) == hash(second)
    assert {first: "same expression"}[second] == "same expression"
    assert set(calls) == {id(first), id(second)}
    with pytest.raises(FrozenInstanceError):
        first.kind = "mul"
    with pytest.raises(CompileError, match="immutable tuple"):
        Expr("constant", [1])


def test_commutative_and_polynomial_ordering_do_not_render_or_hash_sort_keys(monkeypatch):
    a, b = _argument("a"), _argument("b")

    def forbidden_repr(node):
        raise AssertionError("expression ordering must not render a DAG")

    monkeypatch.setattr(Expr, "__repr__", forbidden_repr)
    left = expression("add", a, b)
    right = expression("add", b, a)
    assert left == right
    product = expression("mul", expression("add", a, literal(2)), expression("add", b, literal(3)))
    expanded = normalize(product, {})
    assert expanded.evaluate({"a": 7, "b": 11}) == (7 + 2) * (11 + 3)
    quotient = expression("floordiv", expression("mul", a, b), a)
    assert normalize(quotient, {"a": Fact(minimum=1)}) == b
    # This order is structural; numeric hash values never participate in it.
    assert (
        expression("add", literal(10), a).order_key == expression("add", a, literal(10)).order_key
    )


def test_substitution_rebuilds_each_affected_shared_node_once_and_stays_simultaneous(monkeypatch):
    source, replacement = _argument("source"), _argument("replacement")
    root = _doubled(source, 8)
    original = scalar.expression
    calls = []

    def observed(kind, left, right):
        calls.append((kind, left, right))
        return original(kind, left, right)

    monkeypatch.setattr(scalar, "expression", observed)
    result = substitute(root, {source: replacement, replacement: literal(99)})
    assert len(calls) == 8
    assert result.weight == root.weight and result.depth == root.depth
    assert result.evaluate({"replacement": 3}) == 3 * 2**8
    current = result
    for _ in range(8):
        assert current.args[0] is current.args[1]
        current = current.args[0]
    assert current is replacement
    assert substitute(root, {source: literal(2)}).evaluate({}) == 2 * 2**8


def test_empty_and_unrelated_substitutions_preserve_original_node_identities():
    left = expression("add", _argument("a"), literal(1))
    right = expression("add", _argument("a"), literal(1))
    assert left == right and left is not right
    root = Expr("mul", (left, right))
    assert substitute(root, {}) is root
    assert substitute(root, {_argument("unrelated"): literal(7)}) is root
    assert root.args == (left, right)


def test_bulk_substitution_keeps_its_snapshot_and_simultaneous_replacements():
    source, replacement = _argument("source"), _argument("replacement")
    root = Expr("add", (source, source))
    bindings = {source: replacement, replacement: literal(99)}
    with memo_scope(fresh=True):
        translate = substitution(bindings)
        bindings[source] = literal(4)
        result = translate(root)
        assert result.evaluate({"replacement": 3}) == 6
        assert translate(root) is result
        assert substitute(root, bindings).evaluate({}) == 8


def test_free_variables_are_immutable_compositional_facts():
    first, second = _argument("first"), _argument("second")
    root = _doubled(Expr("mul", (first, second)), 8)
    collected = arguments(root)
    assert collected == {"first", "second"}
    collected.clear()
    assert arguments(root) == {"first", "second"}
    replaced = substitute(root, {first: _argument("third")})
    assert arguments(replaced) == {"second", "third"}
    assert arguments(root) == {"first", "second"}
    assert not arguments(substitute(root, {first: literal(0)}))


def test_retained_integer_normalization_bounds_work_by_shared_ring_nodes(monkeypatch):
    from ascendflow.analysis import integer_facts

    depth = 10
    root = _doubled(_argument("n"), depth)
    original = integer_facts.expression
    calls = []

    def observed(kind, left, right):
        calls.append(kind)
        return original(kind, left, right)

    monkeypatch.setattr(integer_facts, "expression", observed)
    result = integer_facts.IntegerFacts({}).bounded(root)
    assert len(calls) <= 2 * depth
    assert root.weight == 2 ** (depth + 1) - 1
    for n in (-3, 0, 7):
        assert result.evaluate({"n": n}) == n * 2**depth


def test_shared_graphs_still_obey_expanded_weight_and_depth_budgets():
    root = _doubled(_argument("n"), 11)
    assert root.weight == 4095
    with pytest.raises(CompileError, match="4096 terms / depth 128"):
        Expr("add", (root, root))
    chain = _argument("n")
    one = literal(1)
    for _ in range(127):
        chain = Expr("add", (chain, one))
    assert chain.depth == 128
    with pytest.raises(CompileError, match="4096 terms / depth 128"):
        Expr("add", (chain, one))


def test_expression_scope_shares_only_identical_raw_syntax_and_restores_outer_scope():
    with memo_scope(fresh=True):
        first = expression("add", _argument("n"), literal(3))
        assert expression("add", _argument("n"), literal(3)) is first
        assert expression("mul", _argument("n"), literal(3)) is not first
        raw_division = Expr("floordiv", (_argument("n"), literal(1)))
        assert raw_division.kind == "floordiv" and raw_division is not _argument("n")
        with pytest.raises(RuntimeError, match="inner"), memo_scope(fresh=True):
            inner = expression("add", _argument("n"), literal(3))
            assert inner == first and inner is not first
            raise RuntimeError("inner")
        assert expression("add", _argument("n"), literal(3)) is first
    outside = expression("add", _argument("n"), literal(3))
    assert outside == first and outside is not first
    with memo_scope(fresh=True):
        assert expression("add", _argument("n"), literal(3)) is not first


@pytest.mark.parametrize("invalid", (True, 1.0))
def test_expression_arena_does_not_mask_invalid_constants_equal_to_cached_integers(invalid):
    with memo_scope(fresh=True):
        valid = Expr("constant", (1,))
        for _ in range(2):
            with pytest.raises(CompileError, match="invalid scalar constant"):
                Expr("constant", (invalid,))
        assert Expr("constant", (1,)) is valid
        with pytest.raises(CompileError, match="immutable tuple"):
            Expr("constant", [1])


def test_expression_arena_never_shares_definedness_between_fact_environments():
    from ascendflow.analysis.integer_facts import IntegerFacts
    from ascendflow.analysis.structured_schedule import Comparison

    with memo_scope(fresh=True):
        quotient = expression("floordiv", literal(1), _argument("n"))
        assert expression("floordiv", literal(1), _argument("n")) is quotient
        test = Comparison(quotient, ">=", literal(0))
        positive = IntegerFacts({"n": Fact(minimum=1)})
        possibly_zero = IntegerFacts({"n": Fact()})
        assert positive.ordered_definedness((test,)) is None
        assert possibly_zero.ordered_definedness((test,)) is not None
        assert positive.ordered_definedness((test,)) is None


def test_compilation_releases_its_expression_scope_after_failure(monkeypatch):
    from ascendflow import compile
    from ascendflow.compilation import Compilation
    from examples.vector.scale import scale

    captured = []

    def fail(*args, **kwargs):
        captured.append(expression("add", _argument("scope_cleanup"), literal(3)))
        assert expression("add", _argument("scope_cleanup"), literal(3)) is captured[0]
        raise RuntimeError("compile failure")

    monkeypatch.setattr(Compilation, "_build_structured", fail)
    with pytest.raises(RuntimeError, match="compile failure"):
        compile(scale.specialize(B=64))
    outside = expression("add", _argument("scope_cleanup"), literal(3))
    assert outside == captured[0] and outside is not captured[0]

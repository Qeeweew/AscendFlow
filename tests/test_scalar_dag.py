"""Immutable scalar DAG identity, deterministic ordering and bounded visits."""

from dataclasses import FrozenInstanceError

import pytest

from xdsl_npu import scalar
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.indexing import arguments, normalize
from xdsl_npu.scalar import Expr, Fact, expression, literal, substitute


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
    from xdsl_npu.analysis import integer_facts

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

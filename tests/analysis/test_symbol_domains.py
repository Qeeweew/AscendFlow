"""Identity, invocation and capture distinctions survive spelling changes."""

from dataclasses import replace

from ascendflow.analysis.smt import IntegerQueryService, query_scope, query_service
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.scalar import Expr, Fact, expression, expression_scope, literal, substitute
from ascendflow.symbols import Symbol, SymbolKind, is_coordinate


def test_same_printed_name_does_not_merge_two_actual_scalar_bindings():
    left = Symbol("same", SymbolKind.METADATA, object())
    right = Symbol("same", SymbolKind.METADATA, object())
    with expression_scope():
        a, b = (Expr("argument", (symbol,)) for symbol in (left, right))
        assert a is not b
        query = IntegerQueryService({left: Fact(maximum=0), right: Fact(minimum=1)})
        assert query.implies(Comparison(a, "<", b)).proved
        assert query.equivalent(a, b).refuted


def test_printed_names_neither_change_a_binding_nor_assign_its_role():
    original = Symbol("ordinary", SymbolKind.ORDINAL, object())
    renamed = replace(original, name="$parameter_like_spelling")
    assert original == renamed
    assert hash(original) == hash(renamed)
    a, b = (Expr("argument", (symbol,)) for symbol in (original, renamed))
    assert a == b and a.order_key == b.order_key
    assert is_coordinate(original) and is_coordinate(renamed)
    assert not is_coordinate(Symbol("$loop0", SymbolKind.PARAMETER))
    assert not is_coordinate("$loop0")


def test_metadata_loads_from_distinct_iterations_cannot_borrow_one_value():
    clock = Symbol("iteration", SymbolKind.ORDINAL, object())
    ordinal = Expr("argument", (clock,))
    load = Symbol("metadata", SymbolKind.METADATA, object(), (ordinal,))
    current = Expr("argument", (load,))
    previous = substitute(current, {ordinal: expression("sub", ordinal, literal(1))})
    assert previous != current
    assert clock in previous.variables
    query = IntegerQueryService({clock: Fact(minimum=1, maximum=4)})
    assert query.equivalent(previous, current).refuted


def test_equivalent_capture_coordinates_name_the_same_actual_occurrence():
    clock = Symbol("iteration", SymbolKind.ORDINAL, object())
    ordinal = Expr("argument", (clock,))
    load = Symbol("metadata", SymbolKind.METADATA, object(), (ordinal,))
    current = Expr("argument", (load,))
    recovered = Expr("add", (Expr("add", (ordinal, literal(1))), literal(-1)))
    specialized = substitute(current, {ordinal: recovered})
    assert specialized != current
    query = IntegerQueryService({clock: Fact(maximum=4)})
    assert query.equivalent(specialized, current).proved


def test_capture_substitution_is_simultaneous_and_does_not_shift_twice():
    outer = Expr("argument", (Symbol("outer", SymbolKind.ORDINAL, object()),))
    inner = Expr("argument", (Symbol("inner", SymbolKind.ORDINAL, object()),))
    load = Symbol("metadata", SymbolKind.METADATA, object(), (outer, inner))
    value = Expr("argument", (load,))
    shifted = expression("sub", outer, literal(1))
    selected = substitute(value, {outer: shifted, inner: outer})
    assert selected.args[0].captures == (shifted, outer)


def test_complete_fact_snapshots_reuse_queries_only_inside_one_compilation():
    x = Expr("argument", (Symbol("input", SymbolKind.PARAMETER, object()),))
    positive = Comparison(x, ">", literal(0))
    facts = {x.args[0]: Fact(minimum=1)}
    with query_scope():
        first = query_service(facts)
        assert first.implies(positive).proved
        assert query_service(dict(facts)) is first
        facts[x.args[0]] = Fact(maximum=0)
        changed = query_service(facts)
        assert changed is not first
        assert changed.implies(positive).refuted
        assert first.implies(positive).proved
    with query_scope():
        fresh = query_service(facts)
        assert fresh is not changed
        assert fresh.statistics.calls == 0
        assert fresh.implies(positive).refuted

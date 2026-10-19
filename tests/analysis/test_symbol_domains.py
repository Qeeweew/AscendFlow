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


def test_metadata_invariance_retains_definition_and_every_specialized_capture():
    from ascendflow import language as N
    from ascendflow.analysis.structured_control import build_control_facts
    from ascendflow.dialects.physical import TaskOp
    from ascendflow.passes.lower_native import lower_native_structure

    source = N.from_source("""
    def captured(Offsets:N.Tensor((4,),N.i64)):
        with N.Kernel():
            first=N.load_index(Offsets[0:1],minimum=0,maximum=4)
            last=N.load_index(Offsets[1:2],minimum=first,maximum=4)
            for outer in N.serial(2):
                middle=N.load_index(Offsets[2:3],minimum=0,maximum=4)
                for inner in N.serial(first,last):
                    current=N.load_index(Offsets[3:4],minimum=0,maximum=4)
    """)
    control = build_control_facts(lower_native_structure(source).module)
    outer, inner = control.loops.values()
    first, _last, middle, current = (
        control.environment.expression(op.value)
        for op in control.frames
        if isinstance(op, TaskOp) and op.native.data == "index_load"
    )
    selected = substitute(inner.count, {control.sid: literal(0)})
    assert selected != inner.count
    assert control.invariant_over(selected, inner.operation)
    assert control.invariant_over(substitute(first, {control.sid: literal(0)}), outer.operation)
    assert control.invariant_over(middle, inner.operation)
    assert not control.invariant_over(middle, outer.operation)
    assert not control.invariant_over(
        substitute(middle, {outer.ordinal: literal(0)}), outer.operation
    )
    assert not control.invariant_over(
        substitute(current, {inner.ordinal: literal(0)}), inner.operation
    )
    variant = replace(first.args[0], captures=(control.group, inner.ordinal))
    assert not control.invariant_over(Expr("argument", (variant,)), inner.operation)
    carry = replace(first.args[0], kind=SymbolKind.CARRY)
    assert not control.invariant_over(Expr("argument", (carry,)), inner.operation)

"""A metadata value belongs to its actual load and invocation coordinates."""

from ascendflow.analysis.region_coordinates import ScalarScope
from ascendflow.scalar import Expr, Fact, expression, literal, substitute
from ascendflow.symbols import Symbol, SymbolKind


def test_inverse_translation_recovers_the_dominating_load_without_reloading():
    clock = Expr("argument", (Symbol("j", SymbolKind.ORDINAL),))
    metadata = Expr("argument", (Symbol("length", SymbolKind.METADATA, object(), (clock,)),))
    previous = substitute(metadata, {clock: expression("sub", clock, literal(1))})
    restored = substitute(previous, {clock: expression("add", clock, literal(1))})
    assert restored != metadata
    scope = ScalarScope((metadata,), {clock.args[0]: Fact()})
    assert scope.resolve(restored) == metadata
    # The preceding invocation is a different load value even with one origin.
    assert scope.resolve(previous) == previous


def test_diagnostic_names_do_not_identify_a_scalar_definition():
    clock = Expr("argument", (Symbol("j", SymbolKind.ORDINAL),))
    left = Expr("argument", (Symbol("length", SymbolKind.METADATA, object(), (clock,)),))
    right = Expr("argument", (Symbol("length", SymbolKind.METADATA, object(), (clock,)),))
    assert ScalarScope((left,), {}).resolve(right) == right


def test_each_capture_including_private_participant_must_match():
    clock = Expr("argument", (Symbol("j", SymbolKind.ORDINAL),))
    origin = object()
    left = Expr("argument", (Symbol("value", SymbolKind.METADATA, origin, (literal(0), clock)),))
    right = Expr("argument", (Symbol("value", SymbolKind.METADATA, origin, (literal(1), clock)),))
    assert ScalarScope((left,), {}).resolve(right) == right

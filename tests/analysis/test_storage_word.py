"""Symbolic storage words compared with an independent finite issue ordering."""

from collections import Counter
from itertools import pairwise, permutations

import pytest

from xdsl_npu.analysis import storage_word
from xdsl_npu.analysis.storage_word import StoragePacketPosition as Position
from xdsl_npu.analysis.storage_word import build_storage_word
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, expression, literal

N = Expr("argument", ("count",))
J = Expr("argument", ("ordinal",))


def _integer(value, inputs):
    """Independent arithmetic for the affine expressions this module emits."""
    if value.kind == "constant":
        return value.args[0]
    if value.kind == "argument":
        return inputs[value.args[0]]
    left, right = (_integer(arg, inputs) for arg in value.args)
    assert value.kind in {"add", "mul"}
    return left + right if value.kind == "add" else left * right


def _active(guards, inputs):
    operations = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
    }
    return all(
        operations[test.predicate](_integer(test.lhs, inputs), _integer(test.rhs, inputs))
        for test in guards
    )


def _point(point, inputs):
    return point.key, _integer(point.ordinal, inputs)


def _materialized(word, count):
    inputs = {"count": count}
    pairs = []
    for relation in word.adjacencies:
        if not _active(relation.guards, inputs):
            continue
        if relation.limit is None:
            pairs.append((_point(relation.source, inputs), _point(relation.consumer, inputs)))
        else:
            for ordinal in range(relation.first, _integer(relation.limit, inputs)):
                selected = inputs | {"ordinal": ordinal}
                assert _point(relation.consumer, selected)[1] == ordinal
                pairs.append(
                    (_point(relation.source, selected), _point(relation.consumer, selected))
                )
    first = [_point(row.point, inputs) for row in word.first if _active(row.guards, inputs)]
    last = [_point(row.point, inputs) for row in word.last if _active(row.guards, inputs)]
    return Counter(pairs), first, last


def _finite(positions, count):
    # This oracle expands only small test inputs and sorts actual positions.
    # It does not consume the production PLE decomposition or its evaluators.
    issued = sorted(
        (ordinal + position.offset, position.lexical, position.key, ordinal)
        for position in positions
        for ordinal in range(count)
    )
    points = [(key, ordinal) for _, _, key, ordinal in issued]
    return Counter(pairwise(points)), points[:1], points[-1:]


@pytest.mark.parametrize("offsets", [(0,), (0, 0), (0, 1), (0, 3), (0, 1, 3)])
@pytest.mark.parametrize("reverse", [False, True])
def test_symbolic_ple_matches_every_finite_neighbor_including_empty_and_short(offsets, reverse):
    positions = tuple(
        Position(key, offset, len(offsets) - key if reverse else key)
        for key, offset in enumerate(offsets)
    )
    word = build_storage_word(positions, N, J)
    for count in range(9):
        assert _materialized(word, count) == _finite(positions, count)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_all_lexical_permutations_preserve_cross_step_and_negative_distance(order):
    positions = tuple(Position(key, offset, order[key]) for key, offset in enumerate((0, 1, 3)))
    word = build_storage_word(positions, N, J)
    for count in range(9):
        assert _materialized(word, count) == _finite(positions, count)


def test_two_offset_word_includes_start_steady_interleaving_and_final_packet():
    positions = (Position(0, 0, 0), Position(1, 1, 1))
    word = build_storage_word(positions, N, J)
    expected = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
    pairs, first, last = _materialized(word, 3)
    assert pairs == Counter(pairwise(expected))
    assert first == expected[:1] and last == expected[-1:]
    # The within-step A(j+1)->B(j) relation has negative distance.
    family = next(
        row for row in word.adjacencies if row.limit is not None and row.consumer.key == 1
    )
    assert _integer(family.source.ordinal, {"ordinal": 4}) == 5


@pytest.mark.parametrize("translation", [-100, -3, 0, 7, 1000000])
def test_uniform_offset_translation_preserves_the_exact_word(translation):
    base = (Position(0, 0, 5), Position(1, 1, 1), Position(2, 3, 3))
    translated = tuple(Position(row.key, row.offset + translation, row.lexical) for row in base)
    assert build_storage_word(translated, N, J) == build_storage_word(base, N, J)


@pytest.mark.parametrize("count", range(9))
def test_constant_count_specialization_matches_the_independent_word(count):
    positions = (Position(0, -1, 1), Position(1, 0, 0), Position(2, 2, 2))
    word = build_storage_word(positions, literal(count), J)
    assert _materialized(word, count) == _finite(positions, count)
    if count == 0:
        assert word.adjacencies == word.first == word.last == ()


def test_large_runtime_count_changes_no_static_relation_or_work_size(monkeypatch):
    positions = (Position(0, 0, 2), Position(1, 1, 1), Position(2, 3, 0))
    # A small work limit catches a loop or list proportional to the numeric N.
    monkeypatch.setattr(storage_word, "MAX_WORK", 256)
    small = build_storage_word(positions, literal(8), J)
    large = build_storage_word(positions, literal(100000), J)
    larger = build_storage_word(positions, literal(1000000), J)
    assert len(small.adjacencies) == len(large.adjacencies) == len(larger.adjacencies)
    assert len(large.first) == len(large.last) == 1
    symbolic = build_storage_word(positions, N, J)
    assert len(symbolic.adjacencies) < 100


def test_empty_packet_set_has_no_edges_or_endpoints():
    assert build_storage_word((), N, J).adjacencies == ()
    assert build_storage_word((), N, J).first == ()
    assert build_storage_word((), N, J).last == ()


@pytest.mark.parametrize(
    "positions,diagnostic",
    [
        ((Position(0, 0, 0), Position(0, 1, 1)), "keys must be distinct"),
        ((Position(0, 0, 0), Position(1, 1, 0)), "lexical anchors are ambiguous"),
        ((Position(0, 0.5, 0),), "integer keys, offsets"),
    ],
)
def test_ambiguous_or_untyped_packet_positions_are_rejected(positions, diagnostic):
    with pytest.raises(CompileError, match=diagnostic):
        build_storage_word(positions, N, J)


def test_noncanonical_or_self_varying_clock_is_rejected():
    with pytest.raises(CompileError, match="canonical ordinal argument"):
        build_storage_word((Position(0, 0, 0),), N, literal(0))
    with pytest.raises(CompileError, match="varies over its own clock"):
        build_storage_word((Position(0, 0, 0),), expression("add", J, literal(1)), J)
    with pytest.raises(CompileError, match="count must be nonnegative"):
        build_storage_word((Position(0, 0, 0),), literal(-1), J)


@pytest.mark.parametrize("limit", ["span", "packets", "work", "adjacencies"])
def test_static_representation_budget_has_a_distinct_diagnostic(monkeypatch, limit):
    positions = (Position(0, 0, 0), Position(1, 3, 1))
    setting, value = {
        "span": ("MAX_SPAN", 2),
        "packets": ("MAX_PACKETS", 1),
        "work": ("MAX_WORK", 2),
        "adjacencies": ("MAX_ADJACENCIES", 1),
    }[limit]
    monkeypatch.setattr(storage_word, setting, value)
    with pytest.raises(CompileError) as raised:
        build_storage_word(positions, N, J)
    assert raised.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert raised.value.phase == "storage-word" and "budget" in raised.value.message

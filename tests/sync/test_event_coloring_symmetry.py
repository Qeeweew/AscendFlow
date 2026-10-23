"""Unused event numbers are symmetric only under the full reservation map."""

from itertools import combinations, product

from ascendflow.analysis.event_coloring import _color, _ColoringProblem
from ascendflow.analysis.event_contracts import EventDemand
from ascendflow.target.ascend910b import Actor, Engine

BANK = (Actor.AIV0, Engine.MTE2, Engine.V)


def problem(edges):
    pairs = tuple(combinations(range(5), 2))
    return _ColoringProblem(
        tuple(EventDemand((), 1, (BANK,)) for _ in range(5)),
        (),
        {pair: () for pair in pairs if pair not in edges},
    )


def test_odd_cycle_does_not_enumerate_equivalent_unused_hardware_ids():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    colors, nodes = _color(
        problem(edges), tuple(range(5)), {BANK: set(range(2, 8))}, 8, 5, reuse=True
    )
    assert colors is None and nodes == 5


def test_all_five_vertex_graphs_match_an_independent_two_color_oracle():
    pairs = tuple(combinations(range(5), 2))
    for mask in range(1 << len(pairs)):
        edges = {pair for i, pair in enumerate(pairs) if mask & (1 << i)}
        expected = any(
            all(values[a] != values[b] for a, b in edges) for values in product(range(2), repeat=5)
        )
        colors, _ = _color(
            problem(edges), tuple(range(5)), {BANK: set(range(2, 8))}, 8, 100, reuse=True
        )
        assert (colors is not None) == expected
        if colors is not None:
            assert all(colors[a] != colors[b] for a, b in edges)


def test_different_joint_bank_reservations_keep_distinct_unused_id_choices():
    other = (Actor.AIV1, Engine.MTE2, Engine.V)
    demands = (EventDemand((), 1, (BANK,)), EventDemand((), 1, (BANK, other)))
    # Both vertices need different IDs in BANK. Only 0 is available in the
    # other bank: vertex 0 must backtrack from 0 to the initially unused 1.
    numeric = _ColoringProblem(demands, (), {})
    colors, _ = _color(
        numeric,
        (0, 1),
        {BANK: set(range(2, 8)), other: set(range(1, 8))},
        8,
        20,
        reuse=False,
    )
    assert colors == {0: (1,), 1: (0,)}

"""Frozen numeric constraints; these probes grant no asynchronous DSL permission."""

from itertools import combinations
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import NoEventAssignment, color_event_resources
from ascendflow.analysis.event_contracts import EventDemand, EventIdPools
from ascendflow.analysis.event_frontier_contracts import CreditSeparation, FrozenEventRelations
from ascendflow.diagnostics import CompileError
from ascendflow.target.ascend910b import Actor, Engine


class CountedRows:
    """Finite masks and ordinary numeric lookup, independent of the allocator."""

    def __init__(self, masks):
        self.masks = tuple(masks)
        self.pair_queries = self.row_queries = 0
        self.frozen = FrozenEventRelations(
            tuple(
                ((mask, CreditSeparation((), 1 << n, mask)),) if mask else ()
                for n, mask in enumerate(masks)
            ),
        )

    def row_mask(self, number):
        self.row_queries += 1
        return self.masks[number]

    def get(self, pair, default=None):
        self.pair_queries += 1
        return self.frozen.get(pair, default)


def path_constraints(size, *, width=1, peers=False):
    banks = tuple((actor, Engine.MTE2, Engine.V) for actor in (Actor.AIV0, Actor.AIV1))
    copies = 2 if peers else 1
    edges = set()
    masks = []
    for copy in range(copies):
        radius = 2 if copy else 1
        base = copy * size
        full = ((1 << size) - 1) << base
        for n in range(size):
            blocked = sum(
                1 << (base + m) for m in range(max(0, n - radius), min(size, n + radius + 1))
            )
            masks.append(full & ~blocked)
            edges.update((n, m) for m in range(n + 1, min(size, n + radius + 1)))
    relations = CountedRows(masks)
    flags = (0, 3, 7) if peers else (1, 3, 5, 7) if width == 2 else (2, 6)
    return (
        SimpleNamespace(
            demands=tuple(
                EventDemand((), width, (banks[copy],))
                for copy in range(copies)
                for _ in range(size)
            ),
            reservations=(),
            peers=tuple((n, size + n) for n in range(size)) if peers else (),
            counter_groups=(),
            counter_reuses=FrozenEventRelations(()),
            relations=relations,
            id_pools=EventIdPools(local=flags),
        ),
        edges,
        relations,
    )


@pytest.mark.parametrize("size", [16, 64, 256])
@pytest.mark.parametrize("width,peers", [(1, False), (2, False), (1, True)])
def test_sparse_constraints_only_expand_actual_conflicts(size, width, peers):
    numeric, edges, relations = path_constraints(size, width=width, peers=peers)
    try:
        assigned, option = color_event_resources(numeric)
    except CompileError as error:
        pytest.fail(f"sparse numeric graph used {relations.pair_queries} pair lookups: {error}")
    assert option == 0
    assert relations.pair_queries < 24 * size
    assert relations.row_queries <= 2 * size
    for n, values in enumerate(assigned):
        assert len(values) == width and len(set(values)) == width
        assert set(values) <= set(numeric.id_pools.local)
        if peers and n >= size:
            assert values == assigned[n - size]
    for a, b in edges:
        assert set(assigned[a]).isdisjoint(assigned[b])
    # Check every same-bank constraint independently on the small input;
    # growth probes above only count actual sparse conflicts and row access.
    if size == 16:
        for a, b in combinations(range(len(assigned)), 2):
            if set(numeric.demands[a].banks).isdisjoint(numeric.demands[b].banks):
                continue
            if relations.frozen.get((a, b)) is None:
                assert set(assigned[a]).isdisjoint(assigned[b])


def test_an_overfull_fixed_clique_does_not_expand_its_dense_remainder():
    numeric, _, _ = path_constraints(1024)
    rows = numeric.relations = CountedRows((0,) * len(numeric.demands))
    with pytest.raises(NoEventAssignment, match="not a hardware minimum"):
        color_event_resources(numeric)
    assert rows.pair_queries < 16
    assert rows.row_queries == 0

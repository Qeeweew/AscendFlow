"""Independent finite paths for immutable hidden-node graph projection."""

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from xdsl_npu.analysis.progress_graphs import GraphTopology, PathTree


class Work:
    def __init__(self):
        self.counts = Counter()
        self.union_units = set()

    def charge(self, detail, amount):
        self.counts[detail] += amount
        if "union" in detail:
            self.union_units.add(amount)

    @property
    def total(self):
        return sum(self.counts.values())


def finite_paths(edges, hidden, enabled, source):
    """Raw-edge DFS with no production SCC, projection or search helpers."""
    adjacent = defaultdict(list)
    for index, (before, after) in enumerate(edges):
        if enabled & (1 << index):
            adjacent[before].append(after)
    reached, through, visited = set(), set(), set()
    pending = [(after, after in hidden) for after in adjacent[source]]
    while pending:
        node, crossed = pending.pop()
        if node not in hidden:
            reached.add(node)
            if crossed:
                through.add(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        pending.extend((after, True) for after in adjacent[node])
    return (
        sum(1 << node for node in reached),
        sum(1 << node for node in through),
    )


def assert_witness(topology, source, destination, hidden, enabled, witness):
    assert witness
    node = source
    for offset, index in enumerate(witness):
        assert enabled & (1 << index)
        before, after = topology.edges[index]
        assert before == node
        if offset + 1 < len(witness):
            assert after in hidden
        node = after
    assert node == destination


def test_every_three_node_enabled_graph_and_hidden_set_matches_nonempty_dfs():
    edges = tuple(product(range(3), repeat=2))
    topology = GraphTopology(3, edges)
    work = Work()
    for enabled in range(1 << len(edges)):
        for hidden_bits in range(1 << topology.size):
            hidden = frozenset(node for node in range(topology.size) if hidden_bits & (1 << node))
            projection = topology.project(hidden, enabled, work.charge)
            for source in range(topology.size):
                if source in hidden:
                    assert projection.rows[source] == projection.through_hidden[source] == 0
                    continue
                expected, expected_hidden = finite_paths(edges, hidden, enabled, source)
                assert projection.rows[source] == expected
                assert projection.through_hidden[source] == expected_hidden
                tree = topology.search(source, hidden, enabled, work.charge)
                for destination in range(topology.size):
                    if destination in hidden:
                        continue
                    witness = tree.path(destination, work.charge)
                    assert (witness is not None) == bool(expected & (1 << destination))
                    if witness is not None:
                        assert_witness(topology, source, destination, hidden, enabled, witness)
    # One complete source search supplies all destinations for that source.
    searches = (1 << len(edges)) * topology.size * (1 << (topology.size - 1))
    assert work.counts["graph search initialization"] == searches
    assert work.counts["graph search completion"] == searches


def test_hidden_scc_dag_propagates_exits_without_crossing_other_retained_nodes():
    edges = ((0, 1), (1, 2), (2, 1), (2, 3), (3, 4), (4, 3), (4, 5), (5, 6))
    topology = GraphTopology(7, edges)
    hidden = frozenset((1, 2, 3, 4))
    projection = topology.project(hidden, (1 << len(edges)) - 1, Work().charge)
    assert projection.rows[0] == projection.through_hidden[0] == 1 << 5
    assert projection.rows[5] == 1 << 6
    assert projection.through_hidden[5] == 0
    assert not projection.rows[0] & (1 << 6)


def test_direct_and_hidden_nonempty_self_cycles_are_distinguished():
    topology = GraphTopology(3, ((0, 0), (0, 1), (1, 1), (1, 0), (2, 2)))
    work = Work()
    hidden = frozenset((1,))
    for enabled, direct, crossed in ((0, 0, 0), (1, 1, 0), (14, 1, 1), (15, 1, 1)):
        projection = topology.project(hidden, enabled, work.charge)
        assert projection.rows[0] == direct
        assert projection.through_hidden[0] == crossed
        assert projection.rows[2] == 0
    assert topology.search(0, hidden, 1, work.charge).path(0, work.charge) == (0,)
    assert topology.search(0, hidden, 14, work.charge).path(0, work.charge) == (1, 3)
    assert topology.search(0, hidden, 6, work.charge).path(0, work.charge) is None


def test_parallel_edges_keep_original_indices_and_bfs_returns_a_shortest_path():
    topology = GraphTopology(4, ((0, 1), (0, 2), (1, 2), (2, 3), (0, 3), (0, 3)))
    hidden = frozenset((1, 2))
    work = Work()
    assert topology.outgoing[0] == (0, 1, 4, 5)
    assert topology.incoming[3] == (3, 4, 5)
    assert topology.search(0, hidden, 63, work.charge).path(3, work.charge) == (4,)
    assert topology.search(0, hidden, 47, work.charge).path(3, work.charge) == (5,)
    assert topology.search(0, hidden, 15, work.charge).path(3, work.charge) == (1, 3)
    assert topology.search(0, hidden, 13, work.charge).path(3, work.charge) == (0, 2, 3)


def test_one_search_serves_many_destinations_and_stops_at_retained_boundaries():
    topology = GraphTopology(6, ((0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 0)))
    hidden = frozenset((1, 2))
    work = Work()
    tree = topology.search(0, hidden, 63, work.charge)
    assert isinstance(tree, PathTree)
    assert work.counts["graph search initialization"] == work.counts["graph search completion"] == 1
    visits = work.counts["graph search node"], work.counts["graph search edge"]
    assert visits == (3, 5)
    assert tree.path(0, work.charge) == (0, 1, 5)
    assert tree.path(3, work.charge) == (0, 1, 2)
    assert tree.path(4, work.charge) == (0, 3)
    assert tree.path(5, work.charge) is None
    assert (work.counts["graph search node"], work.counts["graph search edge"]) == visits
    assert 1 in tree.predecessors and 2 in tree.predecessors
    assert tree.predecessors[0] == 5 and 5 not in tree.predecessors
    assert work.counts["graph path lookup"] == 4
    assert work.counts["graph path predecessor"] == work.counts["graph path result"] == 8


def test_search_facts_keep_their_original_source_hidden_set_and_enabled_mask():
    topology = GraphTopology(5, ((0, 1), (1, 2), (2, 3), (0, 4), (4, 3)))
    work = Work()
    earlier = topology.search(0, frozenset((1,)), 31, work.charge)
    later = topology.search(0, frozenset((1, 2)), 7, work.charge)
    disabled = topology.search(0, frozenset((1, 2)), 0, work.charge)
    other_source = topology.search(4, frozenset((1, 2)), 31, work.charge)
    assert earlier.path(2, work.charge) == (0, 1)
    assert earlier.path(3, work.charge) is None
    assert later.path(3, work.charge) == (0, 1, 2)
    assert disabled.path(3, work.charge) is None
    assert other_source.path(3, work.charge) == (4,)
    assert earlier.topology is later.topology is disabled.topology is topology
    assert earlier.source == later.source == disabled.source == 0 and other_source.source == 4
    assert earlier.hidden == {1} and later.hidden == {1, 2}
    assert earlier.enabled == 31 and later.enabled == 7 and disabled.enabled == 0


def test_projection_charges_full_bitset_word_width_and_linear_scc_visits():
    hidden_count = 80
    destination = hidden_count + 1
    hidden = frozenset(range(1, destination))
    edges = (
        tuple((0, node) for node in hidden)
        + tuple(product(hidden, repeat=2))
        + tuple((node, destination) for node in hidden)
    )
    topology = GraphTopology(destination + 1, edges)
    work = Work()
    projection = topology.project(hidden, (1 << len(edges)) - 1, work.charge)
    assert projection.rows[0] == projection.through_hidden[0] == 1 << destination
    assert work.union_units == {2}
    assert work.total < 5 * len(edges) + 10 * topology.size


def test_topology_projection_and_search_trees_are_immutable():
    topology = GraphTopology(2, ((0, 1),))
    other = GraphTopology(2, ((0, 1),))
    assert topology is not other and topology != other
    assert len({topology, other}) == 2
    projection = topology.project(frozenset(), 1, Work().charge)
    tree = topology.search(0, frozenset(), 1, Work().charge)
    with pytest.raises(TypeError, match="completed.*search"):
        PathTree()
    with pytest.raises(FrozenInstanceError):
        topology.edges = ()
    with pytest.raises(FrozenInstanceError):
        projection.rows = ()
    for field, value in (
        ("topology", other),
        ("source", 1),
        ("hidden", frozenset((1,))),
        ("enabled", 0),
        ("predecessors", {}),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(tree, field, value)
    with pytest.raises(TypeError):
        tree.predecessors[1] = 0


def test_projection_search_and_reconstruction_propagate_budget_failures():
    class Limit(Exception):
        pass

    def exhausted(detail, amount):
        raise Limit("owned graph work exhausted")

    topology = GraphTopology(2, ((0, 1),))
    with pytest.raises(Limit, match="owned graph work"):
        topology.project(frozenset(), 1, exhausted)
    with pytest.raises(Limit, match="owned graph work"):
        topology.search(0, frozenset(), 1, exhausted)
    tree = topology.search(0, frozenset(), 1, Work().charge)
    with pytest.raises(Limit, match="owned graph work"):
        tree.path(1, exhausted)
    assert tree.path(1, Work().charge) == (0,)


def test_an_interrupted_search_never_returns_a_partial_tree_or_negative_fact():
    class Limit(Exception):
        pass

    topology = GraphTopology(5, ((0, 1), (1, 2), (2, 3)))
    hidden = frozenset((1, 2))
    for failure_point in ("graph search edge", "graph search completion"):
        work = Work()

        def limited(detail, amount, *, work=work, failure_point=failure_point):
            work.charge(detail, amount)
            if detail == failure_point:
                raise Limit("source search interrupted")

        tree = None
        with pytest.raises(Limit, match="interrupted"):
            tree = topology.search(0, hidden, 7, limited)
        assert tree is None
        complete = topology.search(0, hidden, 7, Work().charge)
        assert complete.path(3, Work().charge) == (0, 1, 2)
        assert complete.path(4, Work().charge) is None


def test_empty_topology_and_invalid_inputs():
    work = Work()
    empty = GraphTopology(0, ())
    projection = empty.project(frozenset(), 0, work.charge)
    assert projection.rows == projection.through_hidden == ()
    for size in (-1, True, 1.5):
        with pytest.raises(ValueError, match="size"):
            GraphTopology(size, ())
    for edge in ((-1, 0), (0, 2), (True, 0)):
        with pytest.raises(IndexError, match="out of range"):
            GraphTopology(2, (edge,))
    topology = GraphTopology(2, ((0, 1),))
    for enabled in (-1, 2, True, 0.5):
        with pytest.raises(ValueError, match="bitmask"):
            topology.project(frozenset(), enabled, work.charge)
        with pytest.raises(ValueError, match="bitmask"):
            topology.search(0, frozenset(), enabled, work.charge)
    with pytest.raises(TypeError, match="frozenset"):
        topology.project(set(), 1, work.charge)
    with pytest.raises(IndexError, match="out of range"):
        topology.project(frozenset((2,)), 1, work.charge)
    with pytest.raises(ValueError, match="retained"):
        topology.search(0, frozenset((0,)), 1, work.charge)
    with pytest.raises(IndexError, match="out of range"):
        topology.search(2, frozenset(), 1, work.charge)
    tree = topology.search(0, frozenset((1,)), 1, work.charge)
    with pytest.raises(ValueError, match="retained"):
        tree.path(1, work.charge)
    with pytest.raises(IndexError, match="out of range"):
        tree.path(2, work.charge)

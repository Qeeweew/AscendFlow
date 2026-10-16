"""Independent finite Boolean reachability for sparse region elimination."""

from collections import defaultdict
from itertools import permutations, product
from random import Random

import pytest

from xdsl_npu.analysis.progress_connectivity import RegionConnectivity
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.scalar import Expr, expression, literal

P, Q, R = (Expr("argument", (name,)) for name in ("p", "q", "r"))
YES = ActionGuard()
P_TRUE, P_FALSE = (Comparison(P, "==", literal(value)) for value in (1, 0))
Q_TRUE, Q_FALSE = (Comparison(Q, "==", literal(value)) for value in (1, 0))
R_TRUE = Comparison(R, "==", literal(1))
GUARDS = (
    YES,
    ActionGuard((P_TRUE,)),
    ActionGuard((P_FALSE,)),
    ActionGuard((Q_TRUE,)),
    ActionGuard((Q_FALSE,)),
    ActionGuard((P_TRUE, Q_TRUE)),
    ActionGuard((P_FALSE, R_TRUE)),
)
VALUES = tuple(dict(zip(("p", "q", "r"), bits)) for bits in product(range(2), repeat=3))


class Conditions:
    """A test owner with syntactic Boolean absorption and no graph knowledge."""

    def __init__(self):
        self.calls = []

    def add(self, alternatives, guard):
        self.calls.append(guard)
        tests = frozenset(guard.comparisons)
        if any(prior <= tests for prior in alternatives):
            return
        for prior in tuple(alternatives):
            if tests < prior:
                del alternatives[prior]
        alternatives[tests] = guard


def graph(size, edges, owner=None):
    owner = Conditions() if owner is None else owner
    result = RegionConnectivity(size, owner.add)
    for source, destination, guard in edges:
        result.add(source, destination, guard)
    return result, owner


def reachable(edges, source, values, *, through=None):
    """Nonempty finite DFS; optionally stop on reaching a retained boundary."""
    adjacent = defaultdict(list)
    for before, after, guard in edges:
        if guard.evaluate(values):
            adjacent[before].append(after)
    reached, pending = set(), list(adjacent[source])
    while pending:
        destination = pending.pop()
        if destination in reached:
            continue
        reached.add(destination)
        if through is None or destination in through:
            pending.extend(adjacent[destination])
    return reached


def assert_projection(result, edges, boundary, size):
    boundary = set(boundary)
    hidden = set(range(size)) - boundary
    projected = tuple(result.retained(boundary))
    assert result.active == boundary
    assert {(a, b) for a, b, _ in projected} <= set(product(boundary, repeat=2))
    for values in VALUES:
        for source in boundary:
            # Each stored relation has only eliminated internal vertices.
            expected = reachable(edges, source, values, through=hidden) & boundary
            actual = {
                destination
                for destination in boundary
                if any(guard.evaluate(values) for guard in result.alternatives(source, destination))
            }
            assert actual == expected, (source, values, actual, expected)
            # The resulting sparse graph preserves all retained-pair paths,
            # including paths through other retained vertices and self-cycles.
            assert reachable(projected, source, values) == (
                reachable(edges, source, values) & boundary
            )


def test_all_three_node_edge_subsets_match_finite_boolean_dfs():
    possible = tuple(
        (source, destination, GUARDS[(3 * source + destination) % len(GUARDS)])
        for source, destination in product(range(3), repeat=2)
    )
    for mask in range(1 << len(possible)):
        edges = tuple(edge for bit, edge in enumerate(possible) if mask & (1 << bit))
        result, _ = graph(3, edges)
        result.eliminate((1,))
        assert_projection(result, edges, (0, 2), 3)


def test_elimination_orders_preserve_all_retained_pairs_and_conditional_cycles():
    random = Random(910)
    for _ in range(12):
        edges = tuple(
            (source, destination, random.choice(GUARDS))
            for source, destination in product(range(5), repeat=2)
            if random.randrange(3) == 0
        )
        for order in permutations((1, 2, 3)):
            result, _ = graph(5, edges)
            # Single-node calls deliberately exercise different orders; batch
            # elimination itself sorts node indices for a deterministic witness.
            for node in order:
                result.eliminate((node,))
            assert_projection(result, edges, (0, 4), 5)


def test_or_alternatives_remain_distinct_and_keep_ordered_guard_witnesses():
    prefix = Comparison(P, ">", literal(0))
    residue = Comparison(
        expression("rem", expression("sub", P, literal(1)), literal(2)), "==", literal(0)
    )
    edges = (
        (0, 1, ActionGuard((prefix, residue))),
        (1, 3, ActionGuard((Q_TRUE, prefix))),
        (0, 2, ActionGuard((P_FALSE,))),
        (2, 3, ActionGuard((Q_FALSE,))),
    )
    result, _ = graph(4, edges)
    result.eliminate((2, 1))
    assert tuple(guard.comparisons for guard in result.alternatives(0, 3)) == (
        (prefix, residue, Q_TRUE),
        (P_FALSE, Q_FALSE),
    )
    assert_projection(result, edges, (0, 3), 4)


def test_internal_self_loop_star_adds_no_weaker_path_and_retained_cycle_is_nonempty():
    edges = (
        (0, 1, ActionGuard((P_TRUE,))),
        (1, 1, ActionGuard((Q_TRUE,))),
        (1, 2, ActionGuard((R_TRUE,))),
        (2, 0, YES),
        (2, 2, ActionGuard((Q_FALSE,))),
    )
    result, owner = graph(3, edges)
    result.eliminate((1,))
    assert result.alternatives(0, 2) == (ActionGuard((P_TRUE, R_TRUE)),)
    assert len(owner.calls) == len(edges) + 1
    assert result.alternatives(0, 0) == ()
    assert result.alternatives(2, 2) == (ActionGuard((Q_FALSE,)),)
    assert_projection(result, edges, (0, 2), 3)
    result.eliminate((2,))
    assert result.alternatives(0, 0) == (ActionGuard((P_TRUE, R_TRUE)),)
    assert_projection(result, edges, (0,), 3)


def test_deleted_nodes_have_no_edges_or_stale_neighbor_indexes():
    edges = ((0, 1, YES), (1, 2, YES), (2, 3, YES), (1, 1, YES))
    result, owner = graph(5, edges)
    result.eliminate((1,))
    calls = len(owner.calls)
    assert result.active == {0, 2, 3, 4}
    assert all(not result.alternatives(a, b) for a, b in product(range(5), repeat=2) if 1 in (a, b))
    result.eliminate((1, 1))
    assert len(owner.calls) == calls
    with pytest.raises(ValueError, match="eliminated"):
        result.add(1, 4, YES)
    with pytest.raises(ValueError, match="eliminated"):
        tuple(result.retained((0, 1)))
    result.add(4, 2, ActionGuard((Q_TRUE,)))
    edges += ((4, 2, ActionGuard((Q_TRUE,))),)
    result.eliminate((2,))
    assert len(owner.calls) == calls + 3
    assert_projection(result, edges, (0, 3, 4), 5)


def test_retained_iteration_is_sparse_deterministic_and_has_no_identity_edges():
    edges = ((3, 3, ActionGuard((P_TRUE,))), (2, 3, YES), (0, 2, YES))
    result, _ = graph(5, edges)
    assert tuple(result.retained((3, 2, 0, 3))) == (edges[2], edges[1], edges[0])
    assert result.alternatives(4, 4) == ()
    result.eliminate((3, 2, 0, 4, 1))
    assert not result.active
    assert tuple(result.retained(())) == ()


def test_owner_alone_can_reject_conditions_and_enforce_combination_budgets():
    class OwnerLimit(Exception):
        pass

    owner = Conditions()

    def add(alternatives, guard):
        if len(owner.calls) >= 2:
            raise OwnerLimit("owned combination budget")
        owner.add(alternatives, guard)

    result = RegionConnectivity(3, add)
    result.add(0, 1, YES)
    result.add(1, 2, YES)
    with pytest.raises(OwnerLimit, match="owned combination budget"):
        result.eliminate((1,))
    assert len(owner.calls) == 2

    def reject(alternatives, guard):
        assert guard is YES

    result = RegionConnectivity(2, reject)
    result.add(0, 1, YES)
    assert tuple(result.retained((0, 1))) == ()
    result.eliminate((1,))
    assert result.active == {0}


def test_empty_graph_and_invalid_nodes_fail_at_the_container_boundary():
    result, _ = graph(0, ())
    result.eliminate(())
    assert not result.active and tuple(result.retained(())) == ()
    with pytest.raises(ValueError, match="size"):
        RegionConnectivity(-1, Conditions().add)
    with pytest.raises(IndexError, match="out of range"):
        result.add(0, 0, YES)

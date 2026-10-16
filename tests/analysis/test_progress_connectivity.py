"""Independent finite Boolean reachability for sparse region elimination."""

from collections import defaultdict
from itertools import permutations, product
from random import Random

import pytest
from xdsl.dialects.builtin import ModuleOp

from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.progress_activations import ProgressActivations
from xdsl_npu.analysis.progress_conditions import ProgressConditions
from xdsl_npu.analysis.progress_connectivity import RegionConnectivity
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
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


def graph(size, edges):
    work = ResourceWork()
    owner = ProgressActivations(ProgressConditions(IntegerFacts({})), work)
    scope = owner.region(ModuleOp([]))
    result = RegionConnectivity(size, scope)
    for source, destination, guard in edges:
        result.add(source, destination, scope.guard(guard))
    return result, scope, work


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
                if result.relation(source, destination).evaluate(values)
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
        result, _, _ = graph(3, edges)
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
            result, _, _ = graph(5, edges)
            # Single-node calls deliberately exercise different orders; batch
            # elimination itself sorts node indices for deterministic formulas.
            for node in order:
                result.eliminate((node,))
            assert_projection(result, edges, (0, 4), 5)


def test_factored_or_retains_original_ordered_guard_leaves():
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
    result, scope, _ = graph(4, edges)
    result.eliminate((2, 1))
    relation = result.relation(0, 3)
    assert relation.kind == "or"
    assert len(relation.operands) == 2
    assert all(branch.kind == "and" for branch in relation.operands)
    leaves = tuple(leaf for branch in relation.operands for leaf in branch.operands)
    assert {id(leaf.guard) for leaf in leaves} == {id(guard) for _, _, guard in edges}
    for _, _, guard in edges:
        assert scope.guard(guard) in leaves
    assert_projection(result, edges, (0, 3), 4)


def test_internal_self_loop_star_adds_no_weaker_path_and_retained_cycle_is_nonempty():
    edges = (
        (0, 1, ActionGuard((P_TRUE,))),
        (1, 1, ActionGuard((Q_TRUE,))),
        (1, 2, ActionGuard((R_TRUE,))),
        (2, 0, YES),
        (2, 2, ActionGuard((Q_FALSE,))),
    )
    result, scope, _ = graph(3, edges)
    result.eliminate((1,))
    assert result.relation(0, 2) is scope.all_of(scope.guard(edges[0][2]), scope.guard(edges[2][2]))
    assert result.relation(0, 0) is scope.false
    assert result.relation(2, 2) is scope.guard(edges[4][2])
    assert_projection(result, edges, (0, 2), 3)
    result.eliminate((2,))
    assert result.relation(0, 0) is scope.all_of(scope.guard(edges[0][2]), scope.guard(edges[2][2]))
    assert_projection(result, edges, (0,), 3)


def test_deleted_nodes_have_no_edges_or_stale_neighbor_indexes():
    edges = ((0, 1, YES), (1, 2, YES), (2, 3, YES), (1, 1, YES))
    result, scope, work = graph(5, edges)
    result.eliminate((1,))
    assert result.active == {0, 2, 3, 4}
    assert all(
        result.relation(a, b) is scope.false for a, b in product(range(5), repeat=2) if 1 in (a, b)
    )
    prior = work.progress_formula_work
    result.eliminate((1, 1))
    assert work.progress_formula_work == prior
    with pytest.raises(ValueError, match="eliminated"):
        result.add(1, 4, scope.true)
    with pytest.raises(ValueError, match="eliminated"):
        tuple(result.retained((0, 1)))
    result.add(4, 2, scope.guard(ActionGuard((Q_TRUE,))))
    edges += ((4, 2, ActionGuard((Q_TRUE,))),)
    result.eliminate((2,))
    assert_projection(result, edges, (0, 3, 4), 5)


def test_retained_iteration_is_sparse_deterministic_and_has_no_identity_edges():
    edges = ((3, 3, ActionGuard((P_TRUE,))), (2, 3, YES), (0, 2, YES))
    result, scope, _ = graph(5, edges)
    assert tuple(result.retained((3, 2, 0, 3))) == tuple(
        (a, b, scope.guard(guard)) for a, b, guard in (edges[2], edges[1], edges[0])
    )
    assert result.relation(4, 4) is scope.false
    result.eliminate((3, 2, 0, 4, 1))
    assert not result.active
    assert tuple(result.retained(())) == ()


def test_false_relations_are_not_stored():
    result, scope, _ = graph(2, ())
    result.add(0, 1, scope.false)
    assert result.relation(0, 1) is scope.false
    assert tuple(result.retained((0, 1))) == ()
    active = scope.guard(ActionGuard((P_TRUE,)))
    result.add(0, 1, active)
    result.add(0, 1, scope.false)
    assert result.relation(0, 1) is active
    result.eliminate((1,))
    assert result.active == {0}


def test_graph_traversal_uses_the_real_owner_region_budget():
    work = ResourceWork()
    owner = ProgressActivations(ProgressConditions(IntegerFacts({})), work)
    scope = owner.region(ModuleOp([]))
    result = RegionConnectivity(3, scope)
    result.add(0, 1, scope.true)
    result.add(1, 2, scope.true)
    owner.MAX_REGION_WORK = work.progress_formula_work
    with pytest.raises(CompileError) as failure:
        result.eliminate((1,))
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert failure.value.phase == "structured-event-progress"
    assert result.active == {0, 1, 2}


def test_seven_independent_diamonds_keep_a_small_factored_relation():
    levels = 7
    names = tuple(f"{choice}{index}" for index in range(levels) for choice in ("p", "q"))
    predicates = tuple(
        ActionGuard((Comparison(Expr("argument", (name,)), "==", literal(1)),)) for name in names
    )
    edges = tuple(
        edge
        for index in range(levels)
        for edge in (
            (3 * index, 3 * index + 1, predicates[2 * index]),
            (3 * index, 3 * index + 2, predicates[2 * index + 1]),
            (3 * index + 1, 3 * index + 3, YES),
            (3 * index + 2, 3 * index + 3, YES),
        )
    )
    result, scope, work = graph(3 * levels + 1, edges)
    result.eliminate(range(1, 3 * levels))
    relation = result.relation(0, 3 * levels)
    assert tuple(result.retained((0, 3 * levels))) == ((0, 3 * levels, relation),)
    assert result.relation(0, 0) is scope.false
    assert result.relation(3 * levels, 3 * levels) is scope.false
    # There are 2**7 non-absorbable DNF terms. Construction instead retains
    # shared subformulas and a bounded number of actual graph/formula visits.
    assert work.progress_formula_nodes <= 8 * levels + 2
    assert work.progress_formula_work < 1000
    for bits in product(range(2), repeat=len(names)):
        values = dict(zip(names, bits))
        expected = all(bits[2 * index] or bits[2 * index + 1] for index in range(levels))
        assert relation.evaluate(values) == expected, values


def test_empty_graph_and_invalid_nodes_fail_at_the_container_boundary():
    result, scope, _ = graph(0, ())
    result.eliminate(())
    assert not result.active and tuple(result.retained(())) == ()
    with pytest.raises(ValueError, match="size"):
        RegionConnectivity(-1, scope)
    with pytest.raises(IndexError, match="out of range"):
        result.add(0, 0, scope.true)

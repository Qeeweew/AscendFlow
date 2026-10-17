"""Finite truth oracles and work bounds for shared progress activation circuits."""

from dataclasses import FrozenInstanceError
from itertools import product
from types import SimpleNamespace

import pytest

from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.guard_cover import opposite
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.progress_activations import ProgressActivations
from xdsl_npu.analysis.progress_conditions import ProgressConditions
from xdsl_npu.analysis.progress_graphs import GraphTopology
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, Fact, expression, literal, substitute

X, Y = (Expr("argument", (name,)) for name in ("x", "y"))
P = Comparison(X, "==", literal(1))
Q = Comparison(Y, "==", literal(1))


def context(*, facts=None):
    arithmetic = IntegerFacts(
        {"x": Fact(minimum=0, maximum=1), "y": Fact(minimum=0, maximum=1)}
        if facts is None
        else facts
    )
    work = ResourceWork()
    owner = ProgressActivations(ProgressConditions(arithmetic), work)
    region = SimpleNamespace(location="activation-test:3")
    return owner, owner.region(region), arithmetic, work


def leaf(scope, *tests):
    return scope.guard(ActionGuard(tests))


def test_invocation_interning_constants_and_direct_boolean_laws():
    owner, scope, _, work = context()
    a, b = leaf(scope, P), leaf(scope, Q)
    another = owner.region(SimpleNamespace(location="another-region"))
    assert owner.region(scope._region) is scope
    assert another.guard(ActionGuard((P,))) is a
    assert another.true is scope.true and another.false is scope.false
    assert scope.guard(ActionGuard()) is scope.true
    assert scope.all_of(a, a) is scope.any_of(a, a) is a
    assert scope.all_of(a, scope.true) is scope.any_of(a, scope.false) is a
    assert scope.all_of(a, scope.false) is scope.false
    assert scope.any_of(a, scope.true) is scope.true
    conjunction, disjunction = scope.all_of(a, b), scope.any_of(a, b)
    assert scope.all_of(b, a) is conjunction
    assert scope.any_of(b, a) is disjunction
    assert scope.any_of(a, conjunction) is scope.all_of(a, disjunction) is a
    assert len(conjunction.operands) == len(disjunction.operands) == 2
    with pytest.raises(FrozenInstanceError):
        a.kind = "false"
    assert work.progress_formula_nodes == 6
    assert work.progress_formula_work >= work.progress_formula_operations > 0


def test_factored_formulas_match_an_independent_truth_table_and_fixed_premises():
    _, scope, _, _ = context()
    a, b = leaf(scope, P), leaf(scope, Q)
    na, nb = leaf(scope, opposite(P)), leaf(scope, opposite(Q))
    formulas = (
        (scope.all_of(a, b), lambda x, y: x and y),
        (scope.any_of(a, b), lambda x, y: x or y),
        (scope.all_of(scope.any_of(a, b), na), lambda x, y: (x or y) and not x),
        (scope.any_of(scope.all_of(a, na), scope.all_of(b, nb)), lambda x, y: False),
        (scope.all_of(scope.any_of(a, b), scope.all_of(na, nb)), lambda x, y: False),
    )
    for root, oracle in formulas:
        expected = []
        for x, y in product(range(2), repeat=2):
            result = bool(oracle(bool(x), bool(y)))
            expected.append(result)
            assert root.evaluate({"x": x, "y": y}) is result
            premises = ActionGuard(
                (Comparison(X, "==", literal(x)), Comparison(Y, "==", literal(y)))
            )
            assert scope.refuted(root, premises) is not result
            assert scope.split(root, premises) is None
        assert scope.refuted(root) is (not any(expected))


def test_seven_diamonds_stay_linear_and_match_all_fourteen_boolean_inputs():
    _, scope, _, work = context()
    root = scope.true
    predicates = []
    for index in range(7):
        pair = tuple(
            Comparison(Expr("argument", (f"{name}{index}",)), "==", literal(1))
            for name in ("p", "q")
        )
        predicates.extend(pair)
        root = scope.all_of(root, scope.any_of(*(leaf(scope, test) for test in pair)))
    assert work.progress_formula_nodes == 28
    for bits in product(range(2), repeat=14):
        values = {test.lhs.args[0]: bit for test, bit in zip(predicates, bits, strict=True)}
        assert root.evaluate(values) is all(
            bits[index] or bits[index + 1] for index in range(0, 14, 2)
        )
    before = work.progress_formula_query_states
    assert not scope.refuted(root)
    assert 1 <= work.progress_formula_query_states - before <= 64
    # Mandatory negative guards refute the factored formula without a list
    # of its 128 possible path conjunctions.
    negative = leaf(scope, *(opposite(test) for test in predicates))
    contradiction = scope.all_of(root, negative)
    before = work.progress_formula_query_states
    scope._owner.MAX_QUERY_BRANCHES = 3
    assert scope.refuted(contradiction)
    assert 1 <= work.progress_formula_query_states - before <= 3


def test_mandatory_guards_are_collected_before_any_or_branch():
    owner, scope, _, work = context()
    root = scope.any_of(leaf(scope, P), leaf(scope, Q))
    contradiction = scope.all_of(root, leaf(scope, P, opposite(P)))
    owner.MAX_QUERY_BRANCHES = 1
    assert scope.refuted(contradiction)
    assert work.progress_formula_query_states == 1


def ordered_guard():
    prefix = Comparison(X, ">", literal(0))
    remainder = expression("rem", expression("sub", X, literal(1)), literal(2))
    residue = Comparison(remainder, "==", literal(0))
    upper = Comparison(X, "<", literal(6))
    return ActionGuard((prefix, residue, upper))


def test_unsigned_leaf_order_and_split_prerequisites_survive_mathematical_queries(monkeypatch):
    _, scope, arithmetic, _ = context(facts={"x": Fact(minimum=0, maximum=8)})
    raw = ordered_guard()
    prefix, residue, upper = raw.comparisons
    assert arithmetic.ordered_definedness(raw.comparisons) is None

    def no_readmission(*args):
        pytest.fail("activation queries must consume established endpoint admission")

    monkeypatch.setattr(arithmetic, "ordered_definedness", no_readmission)
    node = scope.guard(raw)
    assert node.kind == "guard" and node.guard is raw
    assert node.guard.comparisons[1].lhs is residue.lhs
    assert scope.guard(ActionGuard(tuple(reversed(raw.comparisons)))) is not node
    assert scope.split(node) is prefix
    assert scope.split(node, ActionGuard((prefix,))) is residue
    assert scope.split(node, ActionGuard((prefix, residue))) is upper
    assert scope.split(node, ActionGuard((opposite(prefix),))) is None
    assert not node.evaluate({"x": 0})
    assert node.evaluate({"x": 1}) and not node.evaluate({"x": 2})


@pytest.mark.parametrize("operator", ("and", "or"))
@pytest.mark.parametrize("known", (False, True))
def test_split_combines_three_valued_branches_without_splitting_a_decided_formula(operator, known):
    _, scope, _, _ = context()
    undecided = leaf(scope, P)
    decided = leaf(scope, Comparison(literal(1), "==", literal(int(known))))
    combine = scope.all_of if operator == "and" else scope.any_of
    # Create the unknown branch first, so a right-hand True OR branch must
    # override its pending split; a right-hand False AND branch does likewise.
    root = combine(undecided, decided)
    expected = None if (operator == "or" and known or operator == "and" and not known) else P
    assert scope.split(root) is expected


def test_mapping_visits_each_shared_guard_once_and_preserves_binding_and_order():
    _, scope, _, _ = context(facts={"x": Fact(minimum=0), "y": Fact(minimum=0)})
    raw = ordered_guard()
    shared, a, b = scope.guard(raw), leaf(scope, P), leaf(scope, Q)
    root = scope.all_of(scope.any_of(shared, a), scope.any_of(shared, b))
    calls = []
    bindings = {X: expression("add", Y, literal(1))}

    def bind(guard):
        calls.append(guard)
        return ActionGuard(
            tuple(
                Comparison(
                    substitute(test.lhs, bindings), test.predicate, substitute(test.rhs, bindings)
                )
                for test in guard.comparisons
            )
        )

    (mapped,) = scope.map_guards((root,), bind)
    assert len(calls) == 3 and sum(item is raw for item in calls) == 1
    assert shared.guard is raw
    for y in range(8):
        assert mapped.evaluate({"y": y}) == root.evaluate({"x": y + 1, "y": y})
    assert scope.map_guards((root,), lambda guard: guard) == (root,)
    assert scope.map_guards((root,), lambda guard: ActionGuard()) == (scope.true,)


def test_one_forest_mapping_preserves_sharing_across_distinct_and_repeated_roots():
    _, scope, _, _ = context()
    shared, a, b = leaf(scope, P), leaf(scope, Q), leaf(scope, opposite(Q))
    first, second = scope.all_of(shared, a), scope.all_of(shared, b)
    calls = []

    def transform(guard):
        calls.append(guard)
        return guard

    roots = scope.map_guards(iter((first, second, shared, first)), transform)
    assert roots == (first, second, shared, first)
    assert len(calls) == 3 and sum(guard is shared.guard for guard in calls) == 1
    assert scope.map_guards((), transform) == ()


def test_unknown_refutation_is_reused_only_with_its_complete_premises(monkeypatch):
    _, scope, _, work = context()
    root = leaf(scope, P)
    assert not scope.refuted(root)
    hits, before = work.progress_formula_query_hits, work.progress_formula_work
    assert not scope.refuted(root)
    assert work.progress_formula_query_hits == hits + 1
    assert work.progress_formula_work > before
    assert scope.refuted(root, ActionGuard((opposite(P),)))
    assert scope.refuted(scope.true, ActionGuard((P, opposite(P))))
    assert not scope.refuted(scope.true)
    # Refutation failure, even for an actually false formula, stays may.
    monkeypatch.setattr(scope._owner.conditions, "impossible", lambda tests: False)
    unproved = leaf(scope, Comparison(literal(0), "==", literal(1)))
    assert not scope.refuted(unproved)


def test_query_branch_limit_has_region_diagnostic_and_does_not_cache_exceptions():
    owner, scope, _, work = context()
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (2, 3))), (leaf(scope, P), leaf(scope, opposite(P)))
    )
    root = scope.all_of(scope.path(graph, frozenset(), 0, 1), scope.path(graph, frozenset(), 2, 3))
    owner.MAX_QUERY_BRANCHES = 1
    for _ in range(2):
        with pytest.raises(CompileError, match="query branches.*budget") as caught:
            scope.refuted(root)
        assert caught.value.kind is ErrorKind.ANALYSIS_LIMIT
        assert caught.value.phase == "structured-event-progress"
        assert caught.value.location == "activation-test:3"
        assert (root, ActionGuard()) not in owner._refutations
        assert not owner._statuses and not owner._graph_states
        assert not owner._path_trees
    owner.MAX_QUERY_BRANCHES = 64
    assert scope.refuted(root)
    hits = work.progress_formula_query_hits
    assert scope.refuted(root)
    assert work.progress_formula_query_hits == hits + 1


def test_region_work_includes_intern_hits_and_resets_only_for_another_region():
    arithmetic, work = IntegerFacts({}), ResourceWork()
    owner = ProgressActivations(ProgressConditions(arithmetic), work)
    owner.MAX_REGION_WORK = 3
    region = SimpleNamespace(location="bounded-region")
    scope = owner.region(region)
    node = leaf(scope, P)
    with pytest.raises(CompileError, match="region activation work.*budget") as caught:
        scope.all_of(node, node)
    assert caught.value.phase == "structured-event-progress"
    assert caught.value.location == "bounded-region"
    assert "work=4" in str(caught.value) and "limit=3" in str(caught.value)
    assert "detail=and construction" in str(caught.value)
    another = owner.region(SimpleNamespace(location="fresh-region"))
    assert leaf(another, P) is node
    assert work.progress_formula_nodes == 1
    assert work.progress_formula_work == 7


@pytest.mark.parametrize("operation", ("all_of", "any_of", "map_guards", "refuted", "split"))
def test_foreign_invocation_nodes_are_rejected(operation):
    _, scope, _, _ = context()
    _, foreign_scope, _, _ = context()
    foreign = leaf(foreign_scope, P)
    with pytest.raises(ValueError, match="another owner"):
        if operation in {"all_of", "any_of"}:
            getattr(scope, operation)(scope.false, foreign)
        elif operation == "map_guards":
            scope.map_guards((foreign,), lambda guard: guard)
        else:
            getattr(scope, operation)(foreign)


def test_poll_failure_has_progress_diagnostic_and_leaves_no_query_answer(monkeypatch):
    import xdsl_npu.analysis.progress_activations as module

    _, scope, _, work = context()
    node = leaf(scope, P)

    def stopped(*args, **kwargs):
        raise CompileError("test cancellation", kind=ErrorKind.ANALYSIS_LIMIT)

    with monkeypatch.context() as patch:
        patch.setattr(module, "poll", stopped)
        with pytest.raises(CompileError, match="test cancellation") as caught:
            scope.refuted(node)
    assert caught.value.phase == "structured-event-progress"
    assert (node, ActionGuard()) not in scope._owner._refutations
    assert not scope.refuted(node)
    assert work.progress_formula_queries > 0


def test_path_evaluation_matches_a_branch_oracle_without_reading_disconnected_variables():
    _, scope, _, _ = context()
    z = Expr("argument", ("z",))
    unrelated = Expr("argument", ("unrelated",))
    graph = scope.graph(
        GraphTopology(5, ((0, 1), (1, 3), (0, 2), (2, 3), (3, 0), (4, 4))),
        (
            leaf(scope, P),
            leaf(scope, Q),
            leaf(scope, opposite(P)),
            leaf(scope, Comparison(z, "==", literal(1))),
            scope.true,
            leaf(scope, Comparison(unrelated, "==", literal(1))),
        ),
    )
    forward = scope.path(graph, frozenset((1, 2)), 0, 3)
    blocked_cycle = scope.path(graph, frozenset((1, 2)), 0, 0)
    actual_cycle = scope.path(graph, frozenset((1, 2, 3)), 0, 0)
    for x, y, value in product(range(2), repeat=3):
        values = {"x": x, "y": y, "z": value}
        expected = bool(x and y or not x and value)
        assert forward.evaluate(values) is expected
        assert not blocked_cycle.evaluate(values)
        assert actual_cycle.evaluate(values) is expected


def test_joint_paths_keep_one_parameter_context_and_refute_incompatible_routes():
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (2, 3))), (leaf(scope, P), leaf(scope, opposite(P)))
    )
    first = scope.path(graph, frozenset(), 0, 1)
    second = scope.path(graph, frozenset(), 2, 3)
    assert not scope.refuted(first) and not scope.refuted(second)
    assert scope.refuted(scope.all_of(first, second))
    assert scope.refuted(first, ActionGuard((opposite(P),)))
    assert not scope.refuted(first, ActionGuard((P,)))
    for value in range(2):
        assert not scope.all_of(first, second).evaluate({"x": value})


def test_one_path_cannot_combine_edges_from_incompatible_parameter_cases():
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(3, ((0, 1), (1, 2))), (leaf(scope, P), leaf(scope, opposite(P)))
    )
    root = scope.path(graph, frozenset((1,)), 0, 2)
    assert scope.refuted(root)
    assert not root.evaluate({"x": 0}) and not root.evaluate({"x": 1})


def test_must_projection_does_not_borrow_unknown_edges_from_a_may_scc():
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(3, ((0, 1), (1, 0), (1, 2))), (leaf(scope, P), scope.true, scope.true)
    )
    root = scope.path(graph, frozenset((1,)), 0, 2)
    assert scope.split(root) is P
    assert scope.refuted(root, ActionGuard((opposite(P),)))
    assert not scope.refuted(root, ActionGuard((P,)))


def test_possible_path_returns_unrefuted_without_enumerating_unknown_assignments():
    owner, scope, _, work = context()
    graph = scope.graph(GraphTopology(3, ((0, 1), (1, 2))), (leaf(scope, P), leaf(scope, Q)))
    root = scope.path(graph, frozenset((1,)), 0, 2)
    owner.MAX_QUERY_BRANCHES = 1
    assert not scope.refuted(root)
    assert work.progress_formula_query_states <= owner.MAX_QUERY_BRANCHES


def test_graph_mapping_is_shared_across_path_roots_and_keeps_immutable_views():
    _, scope, _, _ = context(facts={"x": Fact(minimum=0), "y": Fact(minimum=0)})
    raw = ordered_guard()
    shared = scope.guard(raw)
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (1, 2), (2, 3))), (shared, leaf(scope, P), leaf(scope, Q))
    )
    first_hidden, second_hidden = frozenset((1,)), frozenset((2,))
    first = scope.path(graph, first_hidden, 0, 2)
    second = scope.path(graph, second_hidden, 1, 3)
    calls = []
    bindings = {X: expression("add", Y, literal(1))}

    def bind(guard):
        calls.append(guard)
        return ActionGuard(
            tuple(
                Comparison(
                    substitute(test.lhs, bindings), test.predicate, substitute(test.rhs, bindings)
                )
                for test in guard.comparisons
            )
        )

    left, right, repeated = scope.map_guards((first, second, first), bind)
    assert len(calls) == 3 and sum(guard is raw for guard in calls) == 1
    assert left is repeated
    assert left.path is not None and right.path is not None
    rebound = left.path[0]
    assert rebound is right.path[0] and rebound is not graph
    assert rebound.topology is graph.topology
    assert left.path[1:] == (first_hidden, 0, 2)
    assert right.path[1:] == (second_hidden, 1, 3)
    assert left.path[1] is first_hidden and right.path[1] is second_hidden
    assert shared.guard is raw
    assert rebound.inputs[0].guard is not None
    assert tuple(test.predicate for test in rebound.inputs[0].guard.comparisons) == tuple(
        test.predicate for test in raw.comparisons
    )
    for y in range(6):
        assert left.evaluate({"y": y}) == first.evaluate({"x": y + 1, "y": y})
        assert right.evaluate({"y": y}) == second.evaluate({"x": y + 1, "y": y})


def test_query_reuses_container_projection_and_graph_states_across_path_roots(monkeypatch):
    import xdsl_npu.analysis.progress_activations as module

    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (1, 2), (1, 3))),
        (leaf(scope, P), leaf(scope, Q), leaf(scope, opposite(Q))),
    )
    hidden = frozenset((1,))
    first, second = scope.path(graph, hidden, 0, 2), scope.path(graph, hidden, 0, 3)
    projections, guards = [], []
    project, guard_status = GraphTopology.project, module._Query._guard_status

    def observed_project(topology, selected, mask, charge):
        projections.append((topology, selected, mask))
        return project(topology, selected, mask, charge)

    def observed_guard(query, guard, premises):
        guards.append(guard)
        return guard_status(query, guard, premises)

    monkeypatch.setattr(GraphTopology, "project", observed_project)
    monkeypatch.setattr(module._Query, "_guard_status", observed_guard)
    full = (1 << len(graph.inputs)) - 1
    initial = scope.project(graph, hidden, full)
    assert scope.split(first) is P
    assert scope.split(second) is P
    assert len(guards) == 3
    assert len(projections) == 2 and {mask for _, _, mask in projections} == {0, full}
    assert scope.project(graph, hidden, full) is initial
    scope.split(first, ActionGuard((P,)))
    assert len(guards) == 6


def test_path_queries_share_one_complete_search_tree_across_destinations(monkeypatch):
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (1, 2), (1, 3))),
        (leaf(scope, P), leaf(scope, Q), leaf(scope, opposite(Q))),
    )
    hidden = frozenset((1,))
    first, second = scope.path(graph, hidden, 0, 2), scope.path(graph, hidden, 0, 3)
    searches = []
    search = GraphTopology.search

    def observed(topology, source, selected, mask, charge):
        searches.append((topology, source, selected, mask))
        return search(topology, source, selected, mask, charge)

    monkeypatch.setattr(GraphTopology, "search", observed)
    assert scope.split(first) is P and scope.split(second) is P
    assert not scope.refuted(first) and not scope.refuted(second)
    full = (1 << len(graph.inputs)) - 1
    assert searches == [(graph.topology, 0, hidden, full)]
    assert (graph, hidden, 0, full) in scope._owner._path_trees


def test_path_search_tree_key_separates_graph_hidden_source_and_enabled_mask(monkeypatch):
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(5, ((0, 1), (0, 1), (1, 2), (1, 3), (4, 1))),
        (
            leaf(scope, P),
            leaf(scope, opposite(P)),
            leaf(scope, Q),
            leaf(scope, opposite(Q)),
            scope.true,
        ),
    )
    hidden = frozenset((1,))
    first = scope.path(graph, hidden, 0, 2)
    searches = []
    search = GraphTopology.search

    def observed(topology, source, selected, mask, charge):
        searches.append((topology, source, selected, mask))
        return search(topology, source, selected, mask, charge)

    monkeypatch.setattr(GraphTopology, "search", observed)
    assert scope.split(first) is P
    full = (1 << len(graph.inputs)) - 1
    assert len(searches) == 1
    # A different complete premise set can reuse a topology tree when the
    # resulting enabled mask is identical; its edge truth states stay separate.
    lower = Comparison(X, ">=", literal(0))
    assert scope.split(first, ActionGuard((lower,))) is P
    assert len(searches) == 1
    assert scope.split(scope.path(graph, hidden, 4, 2)) is Q
    assert searches[-1][1:] == (4, hidden, full)
    assert len(searches) == 2
    larger = frozenset((1, 3))
    assert scope.split(scope.path(graph, larger, 0, 2)) is P
    assert searches[-1][1:] == (0, larger, full)
    assert len(searches) == 3
    assert scope.split(first, ActionGuard((P,))) is Q
    assert searches[-1][1:] == (0, hidden, full & ~(1 << 1))
    assert len(searches) == 4
    rebound = scope.graph(graph.topology, (graph.inputs[1], graph.inputs[0], *graph.inputs[2:]))
    assert scope.split(scope.path(rebound, hidden, 0, 2)) == opposite(P)
    assert len(searches) == 5
    assert (rebound, hidden, 0, full) in scope._owner._path_trees


def test_evaluation_shares_search_tree_across_paths_without_reading_other_variables(monkeypatch):
    _, scope, _, _ = context()
    unrelated = Expr("argument", ("unrelated",))
    graph = scope.graph(
        GraphTopology(5, ((0, 1), (1, 2), (1, 3), (4, 4))),
        (
            scope.true,
            leaf(scope, P),
            leaf(scope, Q),
            leaf(scope, Comparison(unrelated, "==", literal(1))),
        ),
    )
    hidden = frozenset((1,))
    root = scope.all_of(scope.path(graph, hidden, 0, 2), scope.path(graph, hidden, 0, 3))
    searches = []
    search = GraphTopology.search

    def observed(topology, source, selected, mask, charge):
        searches.append((topology, source, selected, mask))
        return search(topology, source, selected, mask, charge)

    monkeypatch.setattr(GraphTopology, "search", observed)
    assert root.evaluate({"x": 1, "y": 1})
    assert searches == [(graph.topology, 0, hidden, (1 << len(graph.inputs)) - 1)]


def test_evaluation_uses_a_new_tree_after_disabling_a_false_edge(monkeypatch):
    _, scope, _, _ = context()
    graph = scope.graph(
        GraphTopology(4, ((0, 1), (1, 3), (0, 2), (2, 3))),
        (leaf(scope, P), scope.true, leaf(scope, Q), scope.true),
    )
    hidden = frozenset((1, 2))
    root = scope.path(graph, hidden, 0, 3)
    masks = []
    search = GraphTopology.search

    def observed(topology, source, selected, mask, charge):
        masks.append(mask)
        return search(topology, source, selected, mask, charge)

    monkeypatch.setattr(GraphTopology, "search", observed)
    assert root.evaluate({"x": 0, "y": 1})
    full = (1 << len(graph.inputs)) - 1
    assert masks == [full, full & ~1]


def test_nested_graphs_share_query_and_mapping_with_their_bound_inputs():
    _, scope, _, _ = context()
    child = scope.graph(GraphTopology(2, ((0, 1),)), (leaf(scope, P),))
    child_path = scope.path(child, frozenset(), 0, 1)
    parent = scope.graph(GraphTopology(3, ((0, 1), (1, 2))), (child_path, leaf(scope, Q)))
    root = scope.path(parent, frozenset((1,)), 0, 2)
    assert not scope.refuted(root)
    assert scope.refuted(root, ActionGuard((opposite(P),)))
    assert scope.map_guards((root, child_path), lambda guard: guard) == (root, child_path)
    for x, y in product(range(2), repeat=2):
        assert root.evaluate({"x": x, "y": y}) is bool(x and y)


def test_graph_and_path_inputs_validate_owner_endpoints_and_immutable_hidden_snapshot():
    owner, scope, _, _ = context()
    _, foreign, _, _ = context()
    topology = GraphTopology(3, ((0, 1),))
    graph = scope.graph(topology, (leaf(scope, P),))
    assert scope.graph(topology, graph.inputs) is graph
    assert scope.path(graph, frozenset(), 0, 1) is scope.path(graph, frozenset(), 0, 1)
    with pytest.raises(FrozenInstanceError):
        graph.inputs = ()
    with pytest.raises(ValueError, match="another owner"):
        foreign.path(graph, frozenset(), 0, 1)
    with pytest.raises(ValueError, match="another owner"):
        foreign.project(graph, frozenset(), 1)
    with pytest.raises(ValueError, match="another owner"):
        scope.graph(topology, (leaf(foreign, P),))
    with pytest.raises(ValueError, match="differ"):
        scope.graph(topology, ())
    with pytest.raises(TypeError, match="frozenset"):
        scope.path(graph, set(), 0, 1)
    for hidden, source, destination in ((frozenset((3,)), 0, 1), (frozenset((1,)), 0, 1)):
        with pytest.raises(ValueError, match="vertex|vertices"):
            scope.path(graph, hidden, source, destination)
    scope.path(graph, frozenset((0,)), 1, 2)
    with pytest.raises(ValueError, match="vertex"):
        scope.path(graph, frozenset((False,)), 1, 2)
    for mask in (-1, 2, True):
        with pytest.raises(ValueError, match="mask"):
            scope.project(graph, frozenset(), mask)
    assert graph.owner is owner

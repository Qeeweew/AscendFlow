"""Required cut conclusions agree with independent finite least-model semantics.

The test models only the already established legal relation. Real DSL/native
models separately check which queue, capture and quorum edges may exist.
"""

from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.protocol_transfers import (
    CompletionOrigin,
    ConsumptionOrigin,
    ProtocolTransfers,
    TransferNode,
    TransferRule,
)
from ascendflow.analysis.smt import IntegerQueryService, all_of, any_of, negate
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.scalar import Expr, expression, literal


def relation(nodes, captures, root):
    graph = object.__new__(ProtocolTransfers)
    graph.nodes, graph.work = nodes, ResourceWork()
    graph.domains = SimpleNamespace(arithmetic=IntegerFacts({}))
    graph.positions = SimpleNamespace(position=lambda _: (0,))
    graph.snapshot = lambda *args, **kwargs: root
    graph.propositions, graph.rules = [], []
    graph._rule_system = None
    graph._compile_rules()

    def seeds(origin, excluded):
        for number in captures:
            node = nodes[number]
            if node.notification not in excluded:
                yield TransferRule(node, node.condition, graph.propositions[number], True)

    graph._capture_seeds = seeds
    return graph


def derive(graph, point, premises, excluded=frozenset(), *, consumption=False):
    origin = ConsumptionOrigin(point, object()) if consumption else CompletionOrigin(point, None)
    return graph._derive(
        origin, point, None, excluded=excluded, assumptions=premises, strict_target=consumption
    )


def assert_owned(graph, proof, premises, excluded):
    if proof is None:
        return
    queries = graph.domains.arithmetic.queries
    pending, seen = [proof], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        assert current.node.notification not in excluded
        if current.deduction is not None:
            deduction = current.deduction
            assert deduction.premises == premises
            assert queries.owns_implication(deduction.answer, current.condition, premises)
            # Independently solve the frozen implication without the temporary
            # context. The proof must retain the actual rules it needs.
            assert IntegerQueryService({}).implies(current.condition, premises).proved
            assert current.condition == any_of(
                *(negate(rule.predicate) for rule in deduction.rules), deduction.conclusion
            )
        pending.extend(current.inputs)


@pytest.mark.parametrize("excluded", ((), (1,), (4,), (1, 4)))
@pytest.mark.parametrize("assume_b", (False, True))
@pytest.mark.parametrize("contradict", (False, True))
def test_guarded_cycle_matches_concrete_least_fixed_point(excluded, assume_b, contradict):
    x, y = (Expr("argument", (name,)) for name in ("x", "y"))
    a, other = Comparison(x, "<", literal(0)), Comparison(x, ">=", literal(0))
    b = other if contradict else Comparison(y, ">=", literal(0))
    channels, point = (object(), object()), object()
    nodes = [
        TransferNode("union", None, None, (), False),
        TransferNode("capture", channels[0], point, (), a),
        TransferNode("union", None, None, (1, 3)),
        TransferNode("union", None, None, (2,), b),
        TransferNode("capture", channels[1], point, (), other),
        TransferNode("acquire", None, point, (3, 4)),
    ]
    graph = relation(nodes, (1, 4), 5)
    unavailable = frozenset(nodes[number].notification for number in excluded)
    premises = (b,) if assume_b else ()
    for xv in (-1, 0, 1):
        for yv in (-1, 0, 1):
            values = {"x": xv, "y": yv}
            if assume_b and not b.evaluate(values):
                continue
            reached = set()
            while True:
                updated = {
                    number
                    for number, node in enumerate(nodes)
                    if node.notification not in unavailable
                    and (
                        node.condition
                        if type(node.condition) is bool
                        else node.condition.evaluate(values)
                    )
                    and (number in (1, 4) or any(i in reached for i in node.inputs))
                }
                if updated == reached:
                    break
                reached = updated
            scope = (*premises, Comparison(x, "==", literal(xv)), Comparison(y, "==", literal(yv)))
            proof = derive(graph, point, scope, unavailable)
            assert (proof is not None) == (5 in reached)
            assert_owned(graph, proof, scope, unavailable)
    expected = any_of(
        *(all_of(a, b) for _ in (0,) if 1 not in excluded),
        *(other for _ in (0,) if 4 not in excluded),
    )
    required = graph.domains.arithmetic.queries.implies(expected, premises).proved
    proof = derive(graph, point, premises, unavailable)
    assert (proof is not None) == required
    assert_owned(graph, proof, premises, unavailable)


@pytest.mark.parametrize("layers", (8, 32, 64))
def test_branch_product_needs_only_linear_rules_and_one_required_conclusion(layers):
    nodes = [
        TransferNode("union", None, None, (), False),
        TransferNode("capture", object(), object(), ()),
    ]
    previous, factors = 1, []
    for i in range(layers):
        a, b = (Expr("argument", (f"{name}{i}",)) for name in ("a", "b"))
        left, right = Comparison(a, "<", literal(0)), Comparison(b, "<", literal(0))
        factors.append(any_of(left, right))
        number = len(nodes)
        nodes.extend(
            (
                TransferNode("union", None, None, (previous,), left),
                TransferNode("union", None, None, (previous,), right),
                TransferNode("union", None, None, (number, number + 1)),
            )
        )
        previous = number + 2
    graph = relation(nodes, (1,), previous)
    # No full conditional function is requested: under weak premises the
    # target is not guaranteed; under all layer choices it is guaranteed.
    assert derive(graph, object(), ()) is None
    checks = graph.domains.arithmetic.queries.statistics.solver_checks
    proof = derive(graph, object(), tuple(factors))
    assert proof is not None
    assert graph.work.transfer_rules == len(nodes)
    assert graph.work.completion_rule_instances == 2 * (len(nodes) + 1)
    assert graph.domains.arithmetic.queries.statistics.solver_checks == checks + 1
    assert len(proof.deduction.rules) <= len(nodes) + 1
    assert_owned(graph, proof, tuple(factors), frozenset())


def test_unseeded_cycle_and_missing_acquire_arm_cannot_export_a_fact():
    point = object()
    x = Expr("argument", ("x",))
    a = Comparison(x, "<", literal(0))
    nodes = [
        TransferNode("union", None, None, (), False),
        TransferNode("capture", object(), point, (), a),
        TransferNode("acquire", object(), point, (1, 3)),
        TransferNode("union", None, None, (2,)),
    ]
    graph = relation(nodes, (), 3)
    assert derive(graph, point, ()) is None
    graph = relation(nodes, (1,), 3)
    assert derive(graph, point, (), consumption=True) is None
    proof = derive(graph, point, (a,), consumption=True)
    assert proof is not None
    acquires = [child for child in proof.inputs if child.node.kind == "acquire"]
    assert len(acquires) == 1
    assert_owned(graph, proof, (a,), frozenset())
    assert graph.domains.arithmetic.queries.implies(acquires[0].condition, (a,)).proved
    assert derive(graph, point, (a,), frozenset((nodes[1].notification,))) is None


def test_related_integer_guards_remain_correlated_inside_rules():
    n, point = Expr("argument", ("n",)), object()
    positive = Comparison(n, ">=", literal(1))
    clipped = expression("max", literal(0), expression("min", n, expression("add", n, literal(1))))
    equal = Comparison(clipped, "==", n)
    nodes = [
        TransferNode("union", None, None, (), False),
        TransferNode("capture", object(), point, (), positive),
        TransferNode("acquire", object(), point, (1,), equal),
    ]
    graph = relation(nodes, (1,), 2)
    assert derive(graph, point, ()) is None
    proof = derive(graph, point, (positive,))
    assert proof is not None
    assert_owned(graph, proof, (positive,), frozenset())


def test_unconditional_exit_does_not_export_either_conditional_acquire():
    point = object()
    selector = Expr("argument", ("branch",))
    left = Comparison(selector, "<", literal(0))
    right = Comparison(selector, ">=", literal(0))
    nodes = [
        TransferNode("union", None, None, (), False),
        TransferNode("capture", object(), point, ()),
        TransferNode("acquire", object(), point, (1,), left),
        TransferNode("acquire", object(), point, (1,), right),
        TransferNode("union", None, None, (2, 3)),
    ]
    graph = relation(nodes, (1,), 4)
    proof = derive(graph, point, (), consumption=True)
    assert proof is not None
    assert all(child.node.kind != "acquire" for child in proof.inputs)
    assert_owned(graph, proof, (), frozenset())

"""Exact endpoint and invocation facts belong to one progress composition."""

from dataclasses import replace
from itertools import product

import pytest

from tests.analysis.test_progress_parent_interfaces import PARENT_REUSE
from tests.models.structured_domains import actual_dispatch
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.access_frontier import AccessEndpoint
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_progress import _Composer
from xdsl_npu.analysis.progress_regions import ProgressPort
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.scalar import Expr, expression, literal
from xdsl_npu.target.ascend910b import Actor


@pytest.fixture(scope="module")
def domains():
    return prepared(selected(PARENT_REUSE))


def invocation(domains):
    composer = _Composer(domains)
    parent, child = domains.control.loops
    notification = next(n for n in domains.notifications if composer.owners[n] is parent)
    (case,) = domains.cases[notification]
    source = ProgressPort(notification, notification.sources[0], "set", case)
    wait = ProgressPort(notification, notification.consumers[0], "wait", case)
    return composer, parent, child, source, wait


def count_instances(monkeypatch):
    calls = []
    original = EventDomains._instance

    def instance(self, bound, bindings, *, case, extra_coordinate_guards=()):
        calls.append((bound, bindings, case, extra_coordinate_guards))
        return original(
            self, bound, bindings, case=case, extra_coordinate_guards=extra_coordinate_guards
        )

    monkeypatch.setattr(EventDomains, "_instance", instance)
    return calls


def count_proofs(domains, monkeypatch):
    calls = []
    arithmetic = type(domains.arithmetic)
    original = arithmetic.prove

    def prove(self, test, assumptions):
        calls.append((test, assumptions))
        return original(self, test, assumptions)

    monkeypatch.setattr(arithmetic, "prove", prove)
    return calls


def test_equal_own_and_imported_endpoints_share_one_bound_fact(domains, monkeypatch):
    composer, _, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    bound = composer.endpoints.instance(source)
    imported = replace(source, bound=replace(source.bound))
    assert composer.endpoints.instance(imported) is bound
    assert composer.endpoints.ambient(source) is bound.guard.comparisons
    assert composer.endpoints.coordinates(imported) is bound.coordinates
    assert len(calls) == 1


def test_guard_and_proof_binding_remain_part_of_the_endpoint_fact(domains, monkeypatch):
    composer, parent, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    impossible = Comparison(literal(0), "==", literal(1))
    inactive = replace(
        source,
        bound=replace(
            source.bound, guard=ActionGuard(source.bound.guard.comparisons + (impossible,))
        ),
    )
    ordinal = domains.control.loops[parent].ordinal
    shifted = replace(source, bindings=((ordinal, expression("add", ordinal, literal(1))),))
    original = composer.endpoints.instance(source)
    assert impossible not in original.guard.comparisons
    assert impossible in composer.endpoints.ambient(inactive)
    shifted_coordinates = dict(composer.endpoints.coordinates(shifted))
    original_coordinates = dict(original.coordinates)
    context = {ordinal.args[0]: 2}
    assert shifted_coordinates[parent].evaluate(context) == 2
    assert original_coordinates[parent].evaluate(context) == 1
    assert source.bound.coordinates == original.coordinates
    assert len(calls) == 3


def test_actual_placement_is_not_aliased_by_notification_and_coordinates(domains, monkeypatch):
    composer, _, _, source, _ = invocation(domains)
    calls = count_instances(monkeypatch)
    before = replace(source, bound=replace(source.bound, placement=Placement.BEFORE))
    assert composer.endpoints.instance(source).placement is Placement.AFTER
    assert composer.endpoints.instance(before).placement is Placement.BEFORE
    assert len(calls) == 2


@pytest.mark.parametrize("same_invocation", (True, False))
def test_invocation_result_is_established_once_including_impossible(
    domains, monkeypatch, same_invocation
):
    composer, _, child, source, wait = invocation(domains)
    calls = count_proofs(domains, monkeypatch)
    other = source if same_invocation else wait
    result = composer.endpoints.invocation_guard(child, source, other)
    assert (result is not None) is same_invocation
    first_checks = len(calls)
    assert first_checks
    assert composer.endpoints.invocation_guard(child, replace(source), replace(other)) is result
    assert len(calls) == first_checks


def test_invocation_fact_includes_the_complete_premise_context(domains, monkeypatch):
    composer, _, child, source, _ = invocation(domains)
    original = composer.endpoints.ambient
    premises = []
    # Supply two invariant premise contexts for the same actual placement.
    # A stronger, inconsistent context must not reuse the first answer.
    monkeypatch.setattr(
        composer.endpoints, "ambient", lambda port: original(port) + tuple(premises)
    )
    assert composer.endpoints.invocation_guard(child, source, source) is not None
    premises.append(Comparison(literal(0), "==", literal(1)))
    assert composer.endpoints.invocation_guard(child, source, source) is None


def test_bound_facts_do_not_outlive_the_endpoint_owner(domains, monkeypatch):
    first, _, _, source, _ = invocation(domains)
    second = _Composer(domains)
    calls = count_instances(monkeypatch)
    assert first.endpoints.instance(source) is not second.endpoints.instance(source)
    assert len(calls) == 2


@pytest.fixture(scope="module")
def conditional_order_domains():
    """Two finite selected coordinates; no allocation or operator compile."""
    selection = selected("""
    def coordinates(X:N.Tensor((64,),N.f32),n:N.int64,m:N.int64,
                    p:N.int64,q:N.int64,i:N.int64,j:N.int64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            N.copy(X[:],x)
            for a in N.Pipelined(n,stages=1):
                for b in N.Pipelined(m,stages=1):
                    N.copy(X[:],x)
                    N.fill(y,1.0)
            N.copy(X[:],x)
    """)
    memory = selection.dependencies.memory
    outer, inner = memory.control.loops.values()
    arguments = {name: Expr("argument", (name,)) for name in ("p", "q", "i", "j")}
    before = ((outer.operation, arguments["p"]), (inner.operation, arguments["i"]))
    after = ((outer.operation, arguments["q"]), (inner.operation, arguments["j"]))

    def channel(name, source_index, source_coordinates, consumer_index, consumer_coordinates):
        activation = ActionGuard(
            tuple(
                test
                for loop, value in source_coordinates + consumer_coordinates
                for test in (
                    Comparison(value, ">=", literal(0)),
                    Comparison(memory.control.loops[loop].count, ">", value),
                )
            )
        )

        def bound(index, coordinates, placement):
            return BoundPlacement(
                AccessOccurrence(
                    AccessEndpoint.from_template(memory.templates[index], Actor.AIV0),
                    tuple(LoopBinding(*coordinate) for coordinate in coordinates),
                ),
                placement,
                activation,
                (),
                (),
                (),
                coordinates,
            )

        return LogicalNotification(
            name,
            (bound(source_index, source_coordinates, Placement.AFTER),),
            (bound(consumer_index, consumer_coordinates, Placement.BEFORE),),
            memory.templates[source_index].contract.engine,
            memory.templates[consumer_index].contract.engine,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        )

    return prepared(
        replace(
            selection,
            dependencies=replace(selection.dependencies, required=()),
            notifications=(
                channel("incoming", 0, (), 2, before),
                channel("outgoing", 2, after, 3, ()),
            ),
            coverage=(),
        )
    )


@pytest.mark.parametrize("interface", (False, True))
def test_progress_retains_each_unknown_native_order_clause(conditional_order_domains, interface):
    domains = conditional_order_domains
    composer = _Composer(domains)
    incoming, outgoing = domains.notifications
    (wait_case,), (source_case,) = (domains.cases[n] for n in domains.notifications)
    wait = ProgressPort(incoming, incoming.consumers[0], "wait", wait_case)
    source = ProgressPort(outgoing, outgoing.sources[0], "set", source_case)
    outer, inner = domains.control.loops
    if interface:
        # An actual child interface binds one outer invocation. Its parent
        # must also retain gates from earlier invocations of that child.
        child = composer.compose(inner).interface
        graph, _ = composer._graph(outer, (wait,), None, (), (child,))
        arcs = tuple(
            arc for arc in graph.relations if arc.reason == "actual region interface admission gate"
        )
    else:
        arcs = composer._local_arcs(composer.root, (wait, source), (), (), stream=False)
    assert len(arcs) == 2
    assert all(arc.exact and arc.steps == 0 for arc in arcs)

    native = domains.memory.templates[2].operation
    for p, q, i, j in product(range(2), repeat=4):
        context = {"n": 2, "m": 2, "p": p, "q": q, "i": i, "j": j, "$group": 0}
        positions = {
            key: index
            for index, (key, _) in enumerate(
                actual_dispatch(domains.selection, context, Actor.AIV0)
            )
        }
        before = native, Actor.AIV0, ((outer, p), (inner, i))
        after = native, Actor.AIV0, ((outer, q), (inner, j))
        assert sum(arc.guard.evaluate(context) for arc in arcs) == int(
            positions[before] <= positions[after]
        )
    assert all(
        not arc.guard.evaluate({"n": 0, "m": 2, "p": 0, "q": 0, "i": 0, "j": 0}) for arc in arcs
    )

"""Signed controller issue queries on actual retained native/loop anchors.

Construction uses the real DSL and analysis inputs. Finite order expectations
come from retained dispatch, not the production issue-order implementation.
These comparisons supply no native completion or retirement evidence.
"""

from dataclasses import replace

import pytest

from ascendflow.analysis.access_frontier import AccessEndpoint
from ascendflow.analysis.structured_protocols import BoundPlacement
from ascendflow.analysis.structured_regions import (
    AccessOccurrence,
    CountRelation,
    LoopBinding,
    LoopCountGuard,
    LoopDomain,
)
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement
from ascendflow.passes.notification_binding import _IssueOrder
from ascendflow.scalar import Expr, expression, literal
from ascendflow.target.ascend910b import Actor, Engine
from tests.models.structured_domains import actual_dispatch, need_cases, occurrence_key, point_key
from tests.support.structured import selected


@pytest.mark.parametrize("minimum", (0, 1))
def test_real_compound_two_slot_source_reaches_notification_selection(minimum):
    selection = selected(f"""
    def slots(X:N.Tensor((n,m,64),N.f32),n:N.i64,m:N.i64(minimum={minimum})):
        with N.Kernel():
            x,y=N.alloc_ub((2,64)),N.alloc_ub((2,64))
            for p in N.serial(n):
                for g in N.Pipelined(m):
                    slot=(p*m+g)%2
                    with N.Stage(0,name="load"):
                        N.copy(X[p,g,:],x[slot,:])
                    with N.Stage(1,name="read"):
                        N.mul(x[slot,:],2.0,out=y[slot,:])
""")
    assert len(selection.dependencies.memory.templates) == 2
    assert len(selection.coverage) == len(selection.dependencies.required)
    assert {id(item.requirement) for item in selection.coverage} == {
        id(need) for need in selection.dependencies.required
    }
    assert any(
        item.source_engine is Engine.V and item.destination_engine is Engine.MTE2
        for item in selection.notifications
    )
    # Uniform extent partitions establish populated residues, so captures
    # remain at actual native accesses instead of draining the invocation.
    assert all(
        isinstance(p.occurrence.endpoint, AccessEndpoint)
        for n in selection.notifications
        for p in n.sources + n.consumers
    )
    memory = selection.dependencies.memory
    for count in range(minimum, 6):
        uniform = {"n": 3, "m": count, "$group": 0, "$sid": 0}
        positions = {
            key: index
            for index, (key, _) in enumerate(
                actual_dispatch(selection, uniform, Actor.AIV0, ports=True)
            )
        }
        for coverage in selection.coverage:
            need, notification = coverage.requirement, coverage.notification
            for context in need_cases(need, uniform, memory):
                source = occurrence_key(need.source, context, memory)
                consumer = occurrence_key(need.consumer, context, memory)
                assert source in positions and consumer in positions
                if notification is None:
                    assert coverage.ordering is not None
                    assert positions[source] < positions[consumer]
                else:
                    capture = point_key(notification.sources[0], context, memory)
                    acquire = point_key(notification.consumers[0], context, memory)
                    assert positions[source] <= positions[capture]
                    assert positions[acquire] <= positions[consumer]


@pytest.fixture(scope="module")
def clocks():
    selection = selected("""
    def clocks(n:N.i64):
        with N.Kernel(kind="mixed"):
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            for i in N.Dataflow(n):
                with N.Vector():
                    N.fill(x,1.0)
                    N.mul(x,2.0,out=y)
""")
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))
    need = next(
        need
        for need in selection.dependencies.required
        if need.source.operation is memory.templates[0].operation
        and need.consumer.operation is memory.templates[1].operation
        and need.source.participant is Actor.AIV0
    )
    return selection, loop, need


def point(selection, loop, index, coordinate, phase, actor=Actor.AIV0):
    template = selection.dependencies.memory.templates[index]
    coordinates = ((loop.operation, coordinate),)
    return BoundPlacement(
        AccessOccurrence(
            AccessEndpoint.from_template(template, actor),
            (LoopBinding(loop.operation, coordinate),),
        ),
        phase,
        ActionGuard(),
        (),
        (),
        (),
        coordinates,
    )


def finite_position(selection, bound, context):
    actual = dict(actual_dispatch(selection, {"n": context["n"], "$group": 0}, bound.actor))
    key = (
        bound.operation,
        bound.actor,
        tuple((loop, value.evaluate(context)) for loop, value in bound.coordinates),
    )
    assert key in actual
    return list(actual).index(key)


def test_forward_reverse_lexical_and_phase_orders_keep_their_actual_direction(clocks):
    selection, loop, need = clocks
    order = _IssueOrder(selection.dispatch)
    current = point(selection, loop, 0, loop.ordinal, Placement.AFTER)
    following = point(selection, loop, 1, loop.ordinal, Placement.BEFORE)
    earlier = point(
        selection, loop, 0, expression("sub", loop.ordinal, literal(1)), Placement.AFTER
    )
    active = replace(need, domains=(LoopDomain(loop.operation, loop.count, first=1),))
    for invocation in (1, 2, 3):
        context = {"n": 4, loop.ordinal.args[0]: invocation}
        assert finite_position(selection, earlier, context) < finite_position(
            selection, current, context
        )
        assert finite_position(selection, current, context) < finite_position(
            selection, following, context
        )
    for before, after in ((earlier, current), (current, following)):
        assert order.compare(before, after, active) == -1
        assert order.compare(after, before, active) == 1
    before = replace(current, placement=Placement.BEFORE)
    assert order.compare(before, current, active) == -1
    assert order.compare(current, before, active) == 1
    assert order.compare(current, current, active) == 0
    other_actor = point(selection, loop, 1, loop.ordinal, Placement.BEFORE, Actor.AIV1)
    assert order.compare(current, other_actor, active) is None


def test_varying_clock_order_stays_unknown_in_both_directions(clocks):
    selection, loop, need = clocks
    order = _IssueOrder(selection.dispatch)
    varying = point(selection, loop, 0, loop.ordinal, Placement.AFTER)
    fixed = point(selection, loop, 1, literal(1), Placement.BEFORE)
    active = replace(
        need,
        domains=(LoopDomain(loop.operation, loop.count),),
        guards=(LoopCountGuard(loop.operation, loop.count, CountRelation.MORE, 2),),
    )
    observations = []
    for invocation in range(3):
        context = {"n": 3, loop.ordinal.args[0]: invocation}
        observations.append(
            finite_position(selection, varying, context)
            < finite_position(selection, fixed, context)
        )
    assert any(observations) and not all(observations)
    assert order.compare(varying, fixed, active) is None
    assert order.compare(fixed, varying, active) is None


def test_count_and_residue_premises_are_owned_by_the_complete_need(clocks):
    selection, loop, need = clocks
    order = _IssueOrder(selection.dispatch)
    before = point(selection, loop, 0, literal(1), Placement.AFTER)
    after = point(selection, loop, 1, expression("sub", loop.count, literal(1)), Placement.BEFORE)
    domain = LoopDomain(loop.operation, loop.count)
    strong = replace(
        need,
        domains=(domain,),
        guards=(LoopCountGuard(loop.operation, loop.count, CountRelation.MORE, 2),),
    )
    weak = replace(
        strong, guards=(LoopCountGuard(loop.operation, loop.count, CountRelation.MORE, 0),)
    )
    assert order.compare(before, after, strong) == -1
    assert order.compare(after, before, strong) == 1
    assert order.compare(before, after, weak) is None
    assert order.premises(strong) is order.premises(strong)
    assert order.premises(weak) != order.premises(strong)

    before = point(selection, loop, 0, expression("rem", loop.ordinal, literal(2)), Placement.AFTER)
    after = point(selection, loop, 1, literal(0), Placement.BEFORE)
    even = replace(strong, domains=(replace(domain, period=2, residue=0),))
    assert order.compare(before, after, even) == -1
    assert order.compare(before, after, strong) is None
    assert Comparison(
        expression("rem", loop.ordinal, literal(2)), "==", literal(0)
    ) in order.premises(even)


def test_real_branch_premise_is_not_reused_after_its_removal():
    selection = selected("""
    def branch(n:N.i64):
        with N.Kernel():
            x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
            for i in N.Pipelined(n):
                N.fill(x,1.0)
                if n>=3:
                    N.mul(x,2.0,out=y)
""")
    memory = selection.dependencies.memory
    loop = next(iter(memory.control.loops.values()))
    n = next(name for name in memory.control.environment.facts if name.name == "n")
    need = next(
        need
        for need in selection.dependencies.required
        if need.source.operation is memory.templates[0].operation
        and need.consumer.operation is memory.templates[1].operation
    )
    assert any(
        g.comparison == Comparison(Expr("argument", (n,)), ">=", literal(3)) for g in need.guards
    )
    before = point(selection, loop, 0, literal(1), Placement.AFTER)
    after = point(selection, loop, 1, expression("sub", loop.count, literal(1)), Placement.BEFORE)
    order = _IssueOrder(selection.dispatch)
    for count in (3, 4, 5):
        context = {"n": count, loop.ordinal.args[0]: 0}
        assert finite_position(selection, before, context) < finite_position(
            selection, after, context
        )
    assert order.compare(before, after, need) == -1
    assert order.compare(after, before, need) == 1
    assert order.compare(before, after, replace(need, guards=())) is None
    assert Comparison(Expr("argument", (n,)), ">=", literal(3)) in order.premises(need)

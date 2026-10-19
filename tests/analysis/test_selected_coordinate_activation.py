"""Selected-coordinate prerequisites precede the arithmetic they authorize.

The finite oracle uses original parent/child ordinals and modular membership.
Only the production window's ordered definedness check is exercised here;
no correlated parent membership model, allocator or operator build is assumed.
"""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_frontier import (
    AccessEndpoint,
    LoopResidueSelection,
    ResiduePosition,
)
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.event_windows import _Builder
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_protocols import (
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from ascendflow.analysis.structured_regions import (
    AccessOccurrence,
    LoopBinding,
    LoopDomain,
    LoopOrdinalGuard,
    StorageCell,
    StructuredNeed,
)
from ascendflow.analysis.structured_schedule import ActionGuard
from ascendflow.diagnostics import CompileError
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.notification_binding import _bind_endpoint, _IssueOrder
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.conditional_mad import _retarget, _structure


@pytest.fixture(scope="module")
def selected_context():
    source = N.from_source("""
    def selected_clocks(X:N.Tensor((64,),N.f32),n:N.int64,m:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for p in N.Pipelined(n,stages=1):
                for g in N.Pipelined(m,stages=2):
                    with N.Stage(0):
                        value=N.copy(X[:],x[:])
                    with N.Stage(1):
                        N.mul(value,2.0,out=x[:])
    """)
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_structured_dependencies(memory)
    base = dependencies.required[0]
    selection = SynchronizationSelection(
        replace(dependencies, required=()), bind_structured_dispatch(memory), (), ()
    )
    return EventDomains(selection, ResourceWork()), base


def bound_coordinate(selected_context, *, family, source, translated_residue=True):
    domains, base = selected_context
    parent, child = domains.control.loops.values()
    previous = expression("sub", parent.ordinal, literal(1))
    residue = (
        expression(
            "rem",
            expression(
                "sub",
                literal(2),
                expression("rem", expression("mul", previous, child.count), literal(2)),
            ),
            literal(2),
        )
        if translated_residue
        else 1
    )
    selected = (
        child.ordinal
        if family
        else LoopResidueSelection(
            child.operation,
            2,
            residue,
            ResiduePosition.LAST if source else ResiduePosition.FIRST,
        )
    )
    occurrence = AccessOccurrence(
        AccessEndpoint.from_template(domains.memory.templates[0 if source else 1], Actor.AIV0),
        (LoopBinding(parent.operation, previous), LoopBinding(child.operation, selected)),
        (LoopDomain(child.operation, child.count, period=2, residue=residue),) if family else (),
    )
    need = replace(base, domains=(), guards=())
    bound = _bind_endpoint(need, occurrence, domains.memory, source=source)
    return bound, parent, child


def definedness(domains, bound, parent):
    notification = LogicalNotification(
        "selected_coordinate_activation",
        (bound,),
        (bound,),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        parent.operation,
        parent.ordinal,
        1,
        None,
    )
    guards = domains._ambient(bound)
    builder = _Builder(
        notification,
        parent.operation,
        parent.ordinal,
        parent.count,
        guards,
        domains.facts,
        domains.arithmetic.prove,
        domains.arithmetic.bounds,
        lambda value: domains.control.invariant_over(value, parent.operation),
        parameter_bindings={},
        ordered_definedness=domains.arithmetic.ordered_definedness,
    )
    builder.definedness()
    return ActionGuard(guards)


@pytest.mark.parametrize("family", (False, True))
@pytest.mark.parametrize("source", (False, True))
@pytest.mark.parametrize("translated_residue", (False, True))
def test_selected_coordinate_has_an_ordered_nonempty_prerequisite(
    selected_context, family, source, translated_residue
):
    domains, _ = selected_context
    bound, parent, child = bound_coordinate(
        selected_context, family=family, source=source, translated_residue=translated_residue
    )
    guard = definedness(domains, bound, parent)
    (prerequisite,) = dict(bound.coordinate_guards)[child.operation].comparisons
    selected = next(item for item in bound.selected if item.loop is child.operation)
    assert selected.value == dict(bound.coordinates)[child.operation]
    # The constructor supplies FIRST/LAST membership. The independent finite
    # evaluations below still execute the raw selector, including unsigned
    # arithmetic on empty fibers and translated parent coordinates.
    assert prerequisite in guard.comparisons
    for count in (0, 1, 2):
        for canonical_parent in (1, 2, 3, 4):
            actual_parent = canonical_parent - 1
            residue = (-actual_parent * count) % 2 if translated_residue else 1
            members = [ordinal for ordinal in range(count) if ordinal % 2 == residue]
            canonical = {
                "n": 5,
                "m": count,
                "$group": 0,
                "$sid": 0,
                parent.ordinal.args[0]: canonical_parent,
            }
            assert prerequisite.evaluate(canonical) == bool(members)
            # Evaluation checks unsigned operands and short-circuits absent
            # fibers before attempting count-1-residue.
            assert guard.evaluate(canonical) == bool(members)
            if members:
                chosen = members[-1] if source else members[0]
                assert dict(bound.coordinates)[child.operation].evaluate(canonical) == chosen
            for actual_child in range(count):
                actual = canonical | {
                    parent.ordinal.args[0]: actual_parent,
                    child.ordinal.args[0]: actual_child,
                }
                chosen = (members[-1] if source else members[0]) if members else None
                assert bound.guard.evaluate(actual) == (actual_child == chosen)
        # The absent previous parent invocation precedes even the child
        # prerequisite's parent-dependent remainder.
        assert not guard.evaluate(canonical | {parent.ordinal.args[0]: 0})


@pytest.mark.parametrize("family", (False, True))
def test_a_late_action_guard_cannot_authorize_selected_last(selected_context, family):
    domains, _ = selected_context
    bound, parent, _ = bound_coordinate(selected_context, family=family, source=True)
    with pytest.raises(CompileError, match="ordered activation prefix"):
        definedness(domains, replace(bound, coordinate_guards=()), parent)


@pytest.mark.parametrize("period,distance", ((None, 0), (3, 1)))
def test_native_conditional_read_guard_uses_the_consumers_actual_ordinal(period, distance):
    structure = _structure(
        "N.matmul(aa,bb,accum=c0,initialize=True)\nN.matmul(aa,bb,accum=c0,initialize=(k==0))"
        if distance == 0
        else None
    )
    if period is not None:
        _retarget(structure, period)
    memory = collect_native_memory(structure.module)
    consumer = next(template for template in memory.templates if template.spec.read_resets)
    writer = (
        next(template for template in memory.templates if template.spec.native == "mad")
        if distance == 0
        else consumer
    )
    accumulator = consumer.accesses[2]
    condition = accumulator.read_condition
    assert condition is not None
    logical = memory.control.loops[condition.loop]

    def occurrence(template, ordinal):
        return AccessOccurrence(
            AccessEndpoint.from_template(template, Actor.AIC),
            (LoopBinding(logical.operation, ordinal),),
        )

    need = StructuredNeed(
        StorageCell(
            accumulator.address.root,
            Actor.AIC,
            literal(0),
            0,
            accumulator.address.geometry.nbytes,
        ),
        HazardKind.RAW,
        occurrence(writer, expression("sub", logical.ordinal, literal(distance))),
        occurrence(consumer, logical.ordinal),
        logical.operation,
        domains=(LoopDomain(logical.operation, logical.count, distance=distance),),
        guards=(LoopOrdinalGuard(logical.operation, condition.read),),
        loop=logical.operation,
        distance=distance,
    )
    source = _bind_endpoint(need, need.source, memory, source=True)
    destination = _bind_endpoint(need, need.consumer, memory, source=False)
    premises = ActionGuard(_IssueOrder(bind_structured_dispatch(memory)).premises(need))

    def active(ordinal, count):
        reset = ordinal == 0 if period is None else ordinal % period == 0
        return distance <= ordinal < count and not reset

    for count in (0, 1, 2, 3, 4, 7):
        for ordinal in range(max(count, 1)):
            values = {"n": count, "$group": 0, "$sid": 0, logical.ordinal.args[0]: ordinal}
            assert destination.guard.evaluate(values) == active(ordinal, count)
            assert premises.evaluate(values) == active(ordinal, count)
            # The source runs at g-d: its predicate must test consumer g,
            # including a later native reset that suppresses the old-value read.
            assert source.guard.evaluate(values) == active(ordinal + distance, count)

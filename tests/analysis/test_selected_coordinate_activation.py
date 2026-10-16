"""Selected-coordinate prerequisites precede the arithmetic they authorize.

The finite oracle uses original parent/child ordinals and modular membership.
Only the production window's ordered definedness check is exercised here;
no correlated parent membership model, allocator or operator build is assumed.
"""

from dataclasses import replace

import pytest

from xdsl_npu import language as N
from xdsl_npu.analysis.access_frontier import AccessEndpoint, LoopResidueSelection, ResiduePosition
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_windows import _Builder
from xdsl_npu.analysis.structured_dependencies import build_structured_dependencies
from xdsl_npu.analysis.structured_dispatch import bind_structured_dispatch
from xdsl_npu.analysis.structured_memory import collect_native_memory
from xdsl_npu.analysis.structured_protocols import (
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding, LoopDomain
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.passes.select_notifications import _bind_endpoint
from xdsl_npu.scalar import expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine


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
    assert guard.comparisons.index(prerequisite) < guard.comparisons.index(
        Comparison(dict(bound.coordinates)[child.operation], ">=", literal(0))
    )
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

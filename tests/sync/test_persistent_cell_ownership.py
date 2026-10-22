"""Automatic ownership rounds, true last readers and real bank capacity."""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_resources import allocate_structured_events
from ascendflow.diagnostics import CompileError
from ascendflow.passes.generate_notifications import generate_notifications
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import SyncKind
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)
from tests.support.credit_primitives import LOCAL_SOURCE
from tests.support.structured import native_memory


def two_readers():
    source = LOCAL_SOURCE.replace(
        "n:N.i64):",
        'Mirror:N.Tensor((n,4,64),N.f32,access="write",write_disjoint=True),n:N.i64):',
    )
    source = source.replace(
        "N.copy(scratch[child%2,:],Out[parent,child,:])",
        "N.copy(scratch[child%2,:],Out[parent,child,:])\n"
        "                        N.copy(scratch[child%2,:],Mirror[parent,child,:])",
    )
    return N.from_source(source, disjoint_external=True)


def test_automatic_credit_retires_every_reader_without_a_parent_boundary_drain():
    plan = synchronize_structure(two_readers())
    assert len(plan.generation.credits) == 2
    assert all(
        len(body.readers) == 2
        for lifetime in plan.dependencies.order.cell_lifetimes
        for body in lifetime.rounds
    )
    for n in (0, 3, 7, 10, 17):
        check_compiled_events(plan, {"n": n}, invocations=2)


def test_returning_credit_after_only_the_first_reader_has_a_real_war_counterexample():
    memory = native_memory(two_readers())
    dependencies = build_structured_dependencies(memory)
    program = generate_notifications(dependencies, dependencies.dispatch)
    resources = allocate_structured_events(plan_event_resources(program))
    original = collect_native_memory(memory.module.clone())
    earlier = {
        (credit.name, body.release.operation): raw.readers[0].operation
        for credit in program.credits
        for body, raw in zip(credit.rounds, credit.lifetime.rounds, strict=True)
    }
    actions = tuple(
        replace(action, anchor=earlier[action.protocols[0], action.anchor])
        if action.kind is SyncKind.LOCAL_SET
        and len(action.protocols) == 1
        and (action.protocols[0], action.anchor) in earlier
        else action
        for action in resources.actions
    )
    materialize_notifications(memory, actions)
    with pytest.raises(EventModelError):
        check_structured_events(
            collect_native_memory(memory.module),
            {"n": 10},
            logical_reference=original,
            invocations=2,
        )


def test_persistent_cells_use_the_real_eight_id_local_bank():
    source = (
        LOCAL_SOURCE.replace("4,64", "11,64")
        .replace("(2,64)", "(9,64)")
        .replace("child%2", "child%9")
        .replace("(parent//2)%5", "parent%11")
    )
    with pytest.raises(CompileError, match="capacity 8"):
        synchronize_structure(N.from_source(source, disjoint_external=True))


def test_a_shared_conditional_activation_preserves_credit_when_the_call_is_skipped():
    source = """
def conditional(X:N.Tensor((n,4,64),N.f32),
                Out:N.Tensor((n,4,64),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(1,kind="vector"):
        scratch=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for parent in N.serial(n):
                for child in N.Pipelined(parent%4):
                    with N.Stage(0):
                        if parent%3==1:
                            N.copy(X[parent,child,:],scratch[child%2,:])
                            N.copy(scratch[child%2,:],Out[parent,child,:])
"""
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    assert len(plan.generation.credits) == 2
    for n in (0, 2, 7, 13):
        check_compiled_events(plan, {"n": n}, invocations=2)


@pytest.mark.parametrize("quantum,credits", [(1, 1), (2, 0)])
def test_slot_visits_use_parent_entry_relations_without_assuming_a_child_exists(quantum, credits):
    # Parent visibility is aligned to quantum. An active tile has at least
    # quantum elements left; with quantum=1 the second slot really can be
    # absent. The loop's own body domain must not prove its slot populated.
    source = f"""
def prefix(X:N.Tensor((n,2,4,64),N.f32),
           Out:N.Tensor((n,2,4,64),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(1,kind="vector"):
        scratch=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for parent in N.serial(n):
                end={quantum}*(parent%4+1)
                for tile in N.Pipelined(N.ceildiv(end,4)):
                    with N.Stage(0):
                        count=N.min(4,N.max(0,end-tile*4))
                        for child in N.Pipelined(count):
                            with N.Stage(0):
                                N.copy(X[parent,tile,child,:],scratch[child%2,:])
                            with N.Stage(1):
                                N.copy(scratch[child%2,:],Out[parent,tile,child,:])
"""
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    assert len(plan.generation.credits) == credits
    for n in (0, 1, 2, 4, 7):
        check_compiled_events(plan, {"n": n}, invocations=2)

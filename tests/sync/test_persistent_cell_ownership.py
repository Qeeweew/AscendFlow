"""Automatic ownership rounds, true last readers and real bank capacity."""

from dataclasses import replace
from importlib import import_module

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_order import _Builder, build_access_order
from ascendflow.analysis.cell_ownership import bind_cell_credits, infer_cell_lifetimes
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_resources import allocate_structured_events
from ascendflow.diagnostics import CompileError
from ascendflow.passes.generate_notifications import generate_notifications
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.notification_binding import bind_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import SyncKind
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)
from tests.support.credit_primitives import LOCAL_SOURCE
from tests.support.structured import native_memory, prepared


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
        for lifetime in plan.dependencies.cell_lifetimes
        for body in lifetime.rounds
    )
    for n in (0, 3, 7, 10, 17):
        check_compiled_events(plan, {"n": n}, invocations=2)


def test_cell_retirement_consumes_original_bindings_without_rebinding_or_selecting(monkeypatch):
    selection = bind_notifications(build_access_order(native_memory(two_readers())))
    domains = prepared(selection)

    def unexpected(*args, **kwargs):
        pytest.fail("cell retirement rebuilt an original endpoint or selected a notification")

    module = import_module("ascendflow.passes.notification_binding")
    for name in ("_bind_endpoint", "_bind_requirement", "_notification_for"):
        monkeypatch.setattr(module, name, unexpected)
    credits, _ = bind_cell_credits(domains)
    assert len(credits) == 2
    for credit in credits:
        for actual, original in zip(credit.rounds, credit.lifetime.rounds, strict=True):
            assert actual.acquire.operation is original.retirement.source.operation
            assert actual.release.operation is original.retirement.consumer.operation
            assert original.retirement.consumer is original.readers[-1]

    binding = next(
        b for b in selection.bindings if b.requirement is credits[0].lifetime.rounds[0].retirement
    )
    for bindings in (
        tuple(b for b in selection.bindings if b is not binding),
        tuple(replace(b) if b is binding else b for b in selection.bindings),
    ):
        with pytest.raises(CompileError, match="original owned requirement binding"):
            replace(selection, bindings=bindings).require_closed()


def test_common_ready_keeps_storage_admission_before_each_actual_writer():
    program = N.from_source(
        """
def grouped(X:N.Tensor((n,4,64),N.f32),
            Out:N.Tensor((n,4,2,64),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(1,kind="vector"):
        a=N.alloc_ub((2,64),N.f32)
        b=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for parent in N.serial(n):
                for child in N.Pipelined((parent//2)%5):
                    with N.Stage(0):
                        with N.CopyGroup():
                            N.copy(X[parent,child,:],a[child%2,:])
                            N.copy(X[parent,child,:],b[child%2,:])
                    with N.Stage(1):
                        N.copy(a[child%2,:],Out[parent,child,0,:])
                        N.copy(b[child%2,:],Out[parent,child,1,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(program)
    assert len(plan.generation.credits) == 4
    assert any(
        b.completion_source.operation is not b.source.operation
        for b in plan.generation.selection.bindings
    )
    for credit in plan.generation.credits:
        for actual, original in zip(credit.rounds, credit.lifetime.rounds, strict=True):
            assert actual.acquire.operation is original.retirement.source.operation
            assert actual.release.operation is original.retirement.consumer.operation
    for n in (0, 3, 10):
        check_compiled_events(plan, {"n": n}, invocations=2)


@pytest.mark.parametrize("reader_index", [0, -1])
def test_a_missing_ready_edge_cannot_drop_a_reader_or_its_round_from_ownership(reader_index):
    memory = native_memory(two_readers())
    builder = _Builder(memory, bind_structured_dispatch(memory), None)
    order = builder.build()
    lifetime = order.cell_lifetimes[0]
    body = lifetime.rounds[0]
    reader = body.readers[reader_index]
    builder.required = [
        n
        for n in builder.required
        if not (
            n.cell == lifetime.cell and n.source == body.retirement.source and n.consumer == reader
        )
    ]
    remaining = infer_cell_lifetimes(builder)
    assert remaining
    assert all(item.cell != lifetime.cell for item in remaining)


def test_returning_credit_after_only_the_first_reader_has_a_real_war_counterexample():
    memory = native_memory(two_readers())
    dependencies = build_access_order(memory)
    program = generate_notifications(dependencies)
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


def test_last_reader_follows_dispatched_stages_instead_of_source_order():
    source = (
        LOCAL_SOURCE.replace(
            "n:N.i64):",
            'Mirror:N.Tensor((n,4,64),N.f32,access="write",write_disjoint=True),n:N.i64):',
        )
        .replace("(2,64)", "(3,64)")
        .replace("child%2", "child%3")
    )
    source = source.replace('with N.Stage(1,name="store"):', 'with N.Stage(2,name="late_store"):')
    source += """
                    with N.Stage(1,name="early_store"):
                        N.copy(scratch[child%3,:],Mirror[parent,child,:])
"""
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    assert len(plan.generation.credits) == 3
    for credit in plan.generation.credits:
        for actual, original in zip(credit.rounds, credit.lifetime.rounds, strict=True):
            assert len(original.readers) == 2
            assert original.retirement.consumer is original.readers[0]
            assert actual.release.operation is original.retirement.consumer.operation
    for n in (0, 3, 10):
        check_compiled_events(plan, {"n": n}, invocations=2)


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

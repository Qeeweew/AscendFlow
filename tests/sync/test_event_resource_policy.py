"""Resource policy and ordinal-domain facts preserve real finite executions."""

from types import SimpleNamespace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.event_coloring import (
    NoEventAssignment,
    color_event_resources,
    exclusive_pressure_banks,
)
from ascendflow.analysis.event_contracts import EventDemand, EventReservation, ResourceWork
from ascendflow.analysis.notification_lifetimes import HandbackRule, Reuse
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.analysis.structured_resources import _mapped_event
from ascendflow.analysis.structured_schedule import ActionGuard
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import Expr
from ascendflow.target.ascend910b import Actor, Engine
from tests.models.structured_events import _Interpret, check_compiled_events


@pytest.mark.parametrize(
    "mechanism,native,composed,rings,policy",
    [
        (NotificationKind.LOCAL_EVENT, (3, 4), (3, 2), (3, 2), "required-queue-handback"),
        (NotificationKind.CROSS_EVENT, (3, 8), (3, 4), (3, 4), "required-queue-handback"),
        (NotificationKind.LOCAL_EVENT, (2, 4), (2, 2), (2, 4), "native-handback"),
        (NotificationKind.LOCAL_EVENT, (2, 5), (2, 5), None, None),
        (NotificationKind.CROSS_EVENT, (2, 8), (2, 8), None, None),
    ],
)
def test_nested_ring_choice_respects_capacity_and_keeps_accepted_preference(
    monkeypatch, mechanism, native, composed, rings, policy
):
    """Selection only: the oracle supplies already established per-clock proofs.

    Actual handback soundness and instruction executions are checked by the
    generation/stream tests; this probe cannot grant a DSL reuse permission.
    """
    guard = ActionGuard()
    windows = tuple(
        SimpleNamespace(loop=name, single=False, rank=lambda name=name: Expr("argument", (name,)))
        for name in ("outer", "inner")
    )
    case = SimpleNamespace(windows=windows, condition=guard)
    channel = SimpleNamespace(mechanism=mechanism)

    def fail(channel, message, **context):
        raise CompileError(message, **context)

    domains = SimpleNamespace(notifications=(), work=ResourceWork(), fail=fail)
    rule = HandbackRule(domains)
    calls = set()

    def proved(channel, case, window, width, *, reentry, queue_composition):
        index = windows.index(window)
        assert reentry == (index == 0)
        key = (index, width, queue_composition)
        assert key not in calls
        calls.add(key)
        minimum = (composed if queue_composition else native)[index]
        return (Reuse(window.loop, width, None, guard),) if width >= minimum else None

    monkeypatch.setattr(rule, "_uses", proved)
    if rings is None:
        with pytest.raises(CompileError, match="consumption before reuse") as error:
            rule.case(channel, case)
        assert error.value.phase == "notification-generation-lifetime"
        return
    lease = rule.case(channel, case)
    assert tuple(width for _, width in lease.rings) == rings
    assert lease.policy == policy
    product = rings[0] * rings[1]
    cross = mechanism is NotificationKind.CROSS_EVENT
    assert lease.width == (1 if cross else product)
    assert lease.backlog == (product if cross else None)
    assert tuple(r.delta for r in lease.reentries) == rings[:1]
    assert tuple(r.delta for r in lease.inside) == rings[1:]
    assert {
        lease.ordinal.evaluate({"outer": outer, "inner": inner})
        for outer in range(rings[0])
        for inner in range(rings[1])
    } == set(range(product))


@pytest.mark.parametrize("ids", [(1, 3), (6, 0, 4), (0, 7, 2, 5), (3, 4, 5)])
def test_finite_mapping_keeps_each_virtual_lane_and_valid_target_id(ids):
    lane = Expr("argument", ("lane",))
    mapped = _mapped_event(ids, lane)
    for number, event in enumerate(ids):
        assert mapped.evaluate({"lane": number}) == event
        assert 0 <= mapped.evaluate({"lane": number}) < 8


def private_bank_problem(*, both_coupled):
    """Finite constraints only; reuse permissions are inputs to this probe."""
    a, b = ((actor, Engine.MTE2, Engine.V) for actor in (Actor.AIV0, Actor.AIV1))
    anchor = arith.ConstantOp.from_int_and_width(0, IndexType())
    free = {a: {6, 7}, b: {5, 7} if both_coupled else {6}}
    reservations = tuple(
        EventReservation(bank, flag, "fixed reservation", anchor)
        for bank in (a, b)
        for flag in range(8)
        if flag not in free[bank]
    )
    banks = (a, a, b, b) if both_coupled else (a, a, b)
    return SimpleNamespace(
        demands=tuple(EventDemand((), 1, (bank,)) for bank in banks),
        reservations=reservations,
        peers=((0, 2), (1, 3)) if both_coupled else ((1, 2),),
        counter_groups=(),
        counter_reuses={},
        relations={},
    ), (a, b)


def test_joint_private_bank_pressure_is_detected_when_individual_counts_fit():
    problem, banks = private_bank_problem(both_coupled=True)
    for bank in banks:
        assert (
            sum(bank in d.banks for d in problem.demands)
            + sum(r.bank == bank for r in problem.reservations)
            == 8
        )
    assert exclusive_pressure_banks(problem.demands, problem.reservations, problem.peers) == set(
        banks
    )
    with pytest.raises(NoEventAssignment):
        color_event_resources(problem)
    # Both complete private pairs have frozen noninterference permissions.
    # Their only common free number is 7; the private banks remain distinct.
    problem.relations = {(0, 1): (), (2, 3): ()}
    ids, _ = color_event_resources(problem)
    assert ids == ((7,),) * 4


def test_joint_exclusive_recoloring_precedes_an_available_sharing_permission():
    problem, _ = private_bank_problem(both_coupled=False)
    problem.relations = {(0, 1): ()}
    ids, _ = color_event_resources(problem)
    assert ids == ((7,), (6,), (6,))


def test_absent_carry_is_proved_before_selecting_an_unavailable_old_metadata_value(monkeypatch):
    source = """
def bounded(X:N.Tensor((n,64),N.f32),Meta:N.Tensor((n,),N.i64),
            Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        row=N.alloc_ub((2,64),N.f32)
        for i in N.Pipelined(N.min(n,1)):
            slot=N.load_index(Meta[i:i+1],minimum=0,maximum=1)
            N.copy(X[i,:],row[slot,:])
            N.copy(row[slot,:],Out[i,:])
"""
    original = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            values[operation.value] = invocation % 2

    monkeypatch.setattr(_Interpret, "native", metadata)
    bounded = synchronize_structure(N.from_source(source, disjoint_external=True))
    unbounded = synchronize_structure(
        N.from_source(source.replace("N.min(n,1)", "n"), disjoint_external=True)
    )
    assert not any(c.absence is not None for c in unbounded.selection.coverage)
    for count in (0, 1, 4):
        check_compiled_events(bounded, {"n": count}, invocations=2)
        check_compiled_events(unbounded, {"n": count}, invocations=2)


def test_metadata_selected_last_is_not_built_for_an_absent_parent_carry():
    from ascendflow.analysis.event_contracts import ResourceWork
    from ascendflow.analysis.event_domains import EventDomains
    from ascendflow.analysis.structured_admission import admit_structured_memory
    from ascendflow.analysis.structured_bounds import admit_memory_bounds
    from ascendflow.analysis.structured_dependencies import build_structured_dependencies
    from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.passes.lower_native import lower_native_structure
    from ascendflow.passes.notification_binding import bind_notifications
    from examples.kda.prefill_prepare import prefill_prepare

    # The absent carry is removed before touching an old invocation's load;
    # active endpoints then consume only their real dominating load ranges.
    structure = lower_native_structure(prefill_prepare.specialize(H=8))
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    admission = admit_structured_memory(structure, memory, bounds)
    dispatch = bind_structured_dispatch(memory)
    dependencies = build_structured_dependencies(memory, admission=admission, dispatch=dispatch)
    selection = bind_notifications(dependencies, dispatch)
    selection.require_closed()
    absences = [c.absence for c in selection.coverage if c.absence is not None]
    assert absences and all(a.answer.impossible and a.premises for a in absences)
    domains = EventDomains(selection, ResourceWork())
    assert any(f.discharged for f in domains._activations.values())
    assert all(not name.captures for name in domains.facts)

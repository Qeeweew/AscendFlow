"""Actual BF16 phases share only after independently established consumption."""

from collections import defaultdict
from itertools import pairwise

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow.analysis.event_coloring import NoEventAssignment, color_event_resources
from ascendflow.analysis.event_contracts import EventIdPools
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.resource_report import event_resource_report
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.generate_notifications import _complete_bindings
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.fixtures.event_phases import binary_phases, binary_phases_mixed, full_bank_phases
from tests.models.structured_events import EventModelError, check_compiled_events
from tests.support.structured import selected

BANK = (Actor.AIV0, Engine.MTE2, Engine.V)
TWO_IDS = EventIdPools(banks=((BANK, (2, 6)),))


def ready(assignment):
    return BANK in assignment.banks


def test_original_actions_independently_prove_each_shared_phase_and_wrap():
    original = synchronize_structure(binary_phases)
    refined = synchronize_structure(binary_phases, id_pools=TWO_IDS)
    assert original.resources.work.phase_occupancies == 0
    assert refined.resources.work.phase_occupancies == 6
    assert refined.resources.work.selected_protocol_option == 1
    assignment = {a.notification.name: a.ids[0] for a in refined.resources.assignments if ready(a)}
    assert set(assignment.values()) == {2, 6}
    original_natives = [n.spec for n in original.executable.natives]
    assert original_natives == [n.spec for n in refined.executable.natives]
    for count in (0, 1, 2, 5):
        # This model contains the original three-ID program. It cannot borrow
        # a relation introduced by either of the proposed shared numbers.
        model = check_compiled_events(original, {"rounds": count}, invocations=2)
        streams = defaultdict(list)
        for use in model.consumptions:
            if use.source.source is Engine.MTE2 and use.source.destination is Engine.V:
                protocol = use.source.occurrence.operation.protocol.data
                assert protocol in assignment
                streams[assignment[protocol]].append(use)
        for uses in streams.values():
            for old, new in pairwise(sorted(uses, key=lambda use: use.source.issue)):
                assert model.happens_before(old.wait.begin, new.source.end)
        check_compiled_events(refined, {"rounds": count}, invocations=2)


def test_phase_projection_preserves_logical_credit_and_original_ownership():
    program = _complete_bindings(selected(binary_phases))
    problem = plan_event_resources(program, id_pools=TWO_IDS)
    with pytest.raises(NoEventAssignment):
        color_event_resources(problem)
    assert len(problem.alternatives) == 1
    phased = problem.alternatives[0]
    assert phased.id_pools is problem.id_pools
    coarse = {n: d for d in problem.demands for n in d.members}
    for demand in phased.demands:
        if demand.phase is not None:
            (member,) = demand.members
            assert demand.credit is coarse[member].credit
            assert member is demand.credit.members[demand.phase]
            assert demand.width == coarse[member].width
            assert demand.banks == coarse[member].banks
    assert phased.leases == problem.leases
    assert phased.selection is problem.selection


def test_phase_projection_keeps_one_aiv_body_and_both_original_private_banks():
    plan = synchronize_structure(binary_phases_mixed, id_pools=TWO_IDS)
    assert plan.resources.work.phase_occupancies == 12
    assert plan.resources.work.peer_equalities >= 6
    peer = (Actor.AIV1, Engine.MTE2, Engine.V)
    first, second = (
        [a.ids for a in plan.resources.assignments if bank in a.banks] for bank in (BANK, peer)
    )
    assert len(first) == len(second) == 6 and first == second
    assert {ids[0] for ids in first} == {2, 6}
    actions = [
        a for a in plan.resources.actions if a.source is Engine.MTE2 and a.destination is Engine.V
    ]
    assert actions and all(a.participants == (Actor.AIV0, Actor.AIV1) for a in actions)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"rounds": count}, invocations=2)


def test_phase_mapping_cannot_borrow_a_removed_physical_return():
    plan = synchronize_structure(binary_phases, id_pools=TWO_IDS)
    check_compiled_events(plan, {"rounds": 2})
    gates = [
        operation
        for operation in plan.module.walk()
        if isinstance(operation, NotificationOp)
        and operation.kind.data in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}
        and operation.source.data == Engine.V.value
        and operation.destination.data == Engine.MTE2.value
    ]
    assert gates
    for operation in gates:
        operation.parent.erase_op(operation)
    with pytest.raises(EventModelError, match="overwritten before actual Wait consumption"):
        check_compiled_events(plan, {"rounds": 2})


def test_nine_whole_credit_conflicts_fit_the_real_eight_id_bank_by_actual_phases():
    plan = synchronize_structure(full_bank_phases)
    assert plan.resources.work.phase_occupancies == 72
    assert plan.resources.work.selected_protocol_option == 1
    assignments = [a for a in plan.resources.assignments if ready(a)]
    assert len(assignments) == 72
    assert len({n for a in assignments for n in a.ids}) <= 8
    report = event_resource_report(plan.resources)
    phases = [r for r in report["occupancies"] if r["phase"] is not None]
    assert len(phases) == 72
    assert len({r["credit"] for r in phases}) == 9
    for count in (0, 1, 2):
        check_compiled_events(plan, {"rounds": count}, invocations=2, max_commands=30000)


def test_publisher_cannot_select_another_phases_id():
    plan = synchronize_structure(binary_phases, id_pools=TWO_IDS)
    publication = next(
        operation
        for operation in plan.module.walk()
        if isinstance(operation, NotificationOp)
        and operation.kind.data == SyncKind.LOCAL_SET.value
        and operation.source.data == Engine.MTE2.value
        and operation.destination.data == Engine.V.value
    )
    # 7 is an actual hardware number; it is not this phase's matching ID.
    wrong_phase = arith.ConstantOp.from_int_and_width(7, IndexType())
    publication.parent.insert_op_before(wrong_phase, publication)
    publication.operands = (wrong_phase.result,)
    with pytest.raises(EventModelError, match="unmatched|unconsumed|missing"):
        check_compiled_events(plan, {"rounds": 2})

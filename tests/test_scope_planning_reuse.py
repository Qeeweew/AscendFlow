"""Reuse owned planning work; final checking still reconstructs actual IR."""

from dataclasses import replace

import pytest

from tests.fixtures.nested import nested_copy, nested_cv
from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_scope_query_planning import query_state_store
from xdsl_npu import scope_sync
from xdsl_npu.budget import limit
from xdsl_npu.ir import BufferOp, CaptureOp, LoopChannelOp, ScopeSyncOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.region_summary import summarize_local_events
from xdsl_npu.scope_events import resource_sites
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_completion import QueryCompletion
from xdsl_npu.scope_query_summary import query_native_sites
from xdsl_npu.scope_sync import CheckedScopes, PlanningScopes


@pytest.fixture(
    scope="module",
    params=(query_state_store, scoped_query_recurrence, nested_copy, nested_cv),
    ids=("query-state", "query-recurrence", "nested-copy", "nested-cv"),
)
def planned(request):
    allocation = {}
    original = scope_sync._allocate_child_resources

    def allocate(program, analyses, access, proof, selected):
        result = original(program, analyses, access, proof, selected)
        allocation.update(
            program=program,
            analyses=analyses,
            access=access,
            before=proof,
            candidates=result[0],
            adapted=result[0] is not proof,
        )
        return result

    with limit(15), pytest.MonkeyPatch.context() as patch:
        module = lower_source(request.param)
        preserved = {
            op: (dict(op.attributes), tuple(op.operands))
            for op in module.walk()
            if isinstance(op, (BufferOp, CaptureOp))
        }
        patch.setattr(scope_sync, "_allocate_child_resources", allocate)
        planning, parent = scope_sync._plan_scope_parent(extract_scopes(module))
    return module, planning, parent, allocation, preserved


def test_planning_record_and_parent_describe_the_actual_allocated_protocols(planned):
    module, planning, parent, _, preserved = planned
    assert isinstance(planning, PlanningScopes) and not isinstance(planning, CheckedScopes)
    assert not hasattr(planning, "query_report")
    assert planning.program.module is module
    assert parent.program is planning.program and parent.access is planning.access
    assert tuple(parent.protocols) == tuple(planning.channels.values())
    actual, _ = scope_sync._read_protocols(planning.program, planning.access, planning.analyses)
    assert actual == planning.channels
    for protocol in planning.channels.values():
        assert bool(protocol.ids) == (protocol.kind != "barrier")
    actual_ops = set(module.walk())
    for op, (attributes, operands) in preserved.items():
        assert op in actual_ops and op.attributes == attributes
        assert tuple(op.operands) == operands
    for leaf in planning.program.leaves.values():
        for op in leaf.stream.walk():
            if isinstance(op, ScopeSyncOp):
                assert op.parent_generation is leaf.call.generation
                assert op.child_generation is leaf.stream.body.block.args[0]


def test_child_width_adaptation_rebuilds_only_actual_resources(planned):
    _, planning, parent, allocation, _ = planned
    adapted = allocation["adapted"]
    assert adapted == (planning.program.root.sym_name.data == "nested_copy")
    if adapted:
        assert parent is not allocation["before"] and parent is not allocation["candidates"]
        assert planning.program is not allocation["program"]
        assert planning.access is not allocation["access"]
        assert planning.analyses is not allocation["analyses"]
    else:
        assert parent is allocation["before"]
        assert planning.program is allocation["program"]
        assert planning.access is allocation["access"]
        assert planning.analyses is allocation["analyses"]
    expected = []
    for uid, leaf in planning.program.leaves.items():
        analysis = planning.analyses[uid]
        actual_ids = {
            op.sym_name.data: tuple(i.data for i in op.event_ids)
            for op in leaf.stream.walk()
            if isinstance(op, LoopChannelOp)
        }
        assert actual_ids == {name: c.ids for name, c in analysis[1].items()}
        for _, variant in planning.analyses.variants(uid):
            for resource in summarize_local_events(variant[0]):
                if resource.domain.contains(variant[0].constant_count):
                    expected.append(
                        (
                            uid,
                            *resource_sites(uid, leaf.actor, leaf.loop.constant_count, resource),
                            resource,
                        )
                    )
    assert parent.resource_pairs == list(dict.fromkeys(expected))


def test_query_summary_sites_are_projected_before_outer_planning(planned):
    _, planning, parent, _, _ = planned
    if not planning.program.invocations:
        assert planning.query_summary is None
        return
    assert planning.query_summary is not None
    sites = query_native_sites(planning.query_summary, planning.access)
    assert sites
    for uid, site in sites:
        assert parent.completion.commands_for(uid, site)
    for name, contract in planning.access.query.contracts.items():
        assert planning.access.origins[name] is contract.actual_root.ref
    # Construction checks actual allocated CV IDs and native event commands.
    QueryCompletion(parent, copies=len(planning.query_summary.parts))


def test_public_autosync_skips_parent_reverification_until_independent_finish(monkeypatch):
    planned, verified, captured = [], [], []
    original_plan = scope_sync._plan_scope_parent
    original_verify = scope_sync._verify_scope_parent

    def plan(program):
        result = original_plan(program)
        planned.append(result)
        return result

    def verify(program):
        result = original_verify(program)
        verified.append(result)
        return result

    monkeypatch.setattr(scope_sync, "_plan_scope_parent", plan)
    monkeypatch.setattr(scope_sync, "_verify_scope_parent", verify)
    with limit(15):
        module = scope_sync.autosync_scopes(
            lower_source(query_state_store), _capture=captured.append
        )
    assert len(planned) == len(verified) == len(captured) == 1
    planning, parent = planned[0]
    report, checked, _ = captured[0]
    assert checked.program.module is module and isinstance(checked, CheckedScopes)
    assert checked.program is not planning.program
    assert checked.access is not planning.access and checked.analyses is not planning.analyses
    assert checked.query_summary is not planning.query_summary
    assert verified[0][2] is not parent
    assert checked.query_report["verified"] is True and report["cross_query_verified"] is True


@pytest.mark.parametrize("damage", ("remove-sync", "wrong-clock"))
def test_final_check_rejects_actual_ir_damage_after_query_planning(monkeypatch, damage):
    from xdsl_npu import scope_query_planning

    original = scope_query_planning.plan_query_memory

    def corrupt(planning, parent):
        result = original(planning, parent)
        sync = next(op for op in planning.program.module.walk() if isinstance(op, ScopeSyncOp))
        if damage == "remove-sync":
            sync.parent.erase_op(sync)
        else:
            sync.operands = [sync.child_generation, sync.child_generation]
        return result

    monkeypatch.setattr(scope_query_planning, "plan_query_memory", corrupt)
    with (
        limit(15),
        pytest.raises(
            CompileError,
            match="(missing or changes an actual native|different parent or child clock)",
        ),
    ):
        scope_sync.autosync_scopes(lower_source(query_state_store))


def test_parent_allocation_cannot_change_protocol_semantics(monkeypatch):
    from xdsl_npu import scope_event_pools

    original = scope_event_pools.allocate_parent_pools

    def alter(protocols, proof, used, **kwargs):
        allocated, banks = original(protocols, proof, used, **kwargs)
        if not allocated:
            return allocated, banks
        return (replace(allocated[0], distance=allocated[0].distance + 1), *allocated[1:]), banks

    monkeypatch.setattr(scope_event_pools, "allocate_parent_pools", alter)
    with (
        limit(15),
        pytest.raises(CompileError, match="changed a planned protocol beyond its event IDs"),
    ):
        scope_sync._plan_scope_parent(extract_scopes(lower_source(query_state_store)))

"""Owned resource/query results compose without repeating their solvers."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.test_scope_query_planning import query_state_store
from xdsl_npu import loop_sync, scope_event_pools, scope_query_verification
from xdsl_npu.budget import limit
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import CaptureOp, QueryChannelOp, ScopeSyncOp, TaskOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_planning import plan_query_memory
from xdsl_npu.scope_query_resources import prepare_binary_resources, verify_query_resources
from xdsl_npu.scope_query_verification import verify_query_invocations
from xdsl_npu.scope_resources import establish_invocation_resources
from xdsl_npu.scope_sync import _plan_scope_parent


@pytest.fixture(scope="module")
def owned():
    with limit(15):
        planning, parent = _plan_scope_parent(extract_scopes(lower_source(query_state_store)))
        resources = getattr(planning, "resource_proof", None)
        if resources is None:
            resources = establish_invocation_resources(planning, parent)
        plan = plan_query_memory(planning, parent, resources=resources)
    return planning, parent, resources, plan


def test_completed_memory_and_resources_are_consumed_without_rebuilding(owned, monkeypatch):
    planning, parent, resources, plan = owned

    def repeated(*args, **kwargs):
        pytest.fail("same-snapshot composition repeated an established solver")

    monkeypatch.setattr(parent, "check_child_resources", repeated)
    monkeypatch.setattr(scope_event_pools, "verify_parent_pools", repeated)
    monkeypatch.setattr(loop_sync, "_lifetime", repeated)
    monkeypatch.setattr(scope_query_verification, "prepare_memory_checks", repeated)
    monkeypatch.setattr(scope_query_verification, "check_memory", repeated)
    monkeypatch.setattr(scope_query_verification, "QueryCompletion", repeated)
    report = verify_query_invocations(planning, parent, resources=resources, query_plan=plan)
    assert report["new_memory_checks"] == 0
    assert report["memory_checks_reused"] == len(plan.checks) > 0
    resource_report = report["resources"]
    assert resource_report["invocation_resources"]["new_invocation_resource_solver_calls"] == 0
    assert resource_report["query_event_reuse_checks_reused"] == len(plan.reuse_proofs) > 0
    assert resource_report["new_completion_checks"] == (
        resource_report["parent_binary_checks"]
        + resource_report["child_binary_checks"]
        + resource_report["mixed_binary_checks"]
        + resource_report["cross_counter_boundary_checks"]
    )


def test_one_actual_invocation_check_feeds_all_query_resource_consumers(owned, monkeypatch):
    planning, parent, _, plan = owned
    calls = {"children": 0, "pools": 0, "counters": 0}

    def counted(key, function):
        def call(*args, **kwargs):
            calls[key] += 1
            return function(*args, **kwargs)

        return call

    monkeypatch.setattr(
        parent, "check_child_resources", counted("children", parent.check_child_resources)
    )
    monkeypatch.setattr(
        scope_event_pools,
        "verify_parent_pools",
        counted("pools", scope_event_pools.verify_parent_pools),
    )
    monkeypatch.setattr(loop_sync, "_lifetime", counted("counters", loop_sync._lifetime))
    resources = establish_invocation_resources(planning, parent)
    prepared = prepare_binary_resources(planning, parent, resources=resources)
    report = verify_query_resources(
        planning, parent, plan.completion, resources=resources, query_plan=plan
    )
    assert calls == {"children": 1, "pools": 1, "counters": 1}
    assert report["parent_resources"] == prepared.parent_resources
    assert report["child_resources"] == prepared.child_resources


@pytest.mark.parametrize("kind", (CaptureOp, ScopeSyncOp, TaskOp))
def test_mutable_ir_cannot_reuse_an_owned_invocation_result(owned, monkeypatch, kind):
    planning, parent, resources, plan = owned
    operation = next(op for op in planning.program.module.walk() if isinstance(op, kind))
    attribute = "nbytes" if kind is CaptureOp else "position" if kind is ScopeSyncOp else "stage"
    monkeypatch.setitem(
        operation.attributes, attribute, IntAttr(operation.attributes[attribute].data + 1)
    )
    with pytest.raises(CompileError, match="invocation resource snapshot changed"):
        verify_query_invocations(planning, parent, resources=resources, query_plan=plan)


def test_query_id_mutation_cannot_consume_an_old_rotation_result(owned, monkeypatch):
    planning, parent, resources, plan = owned
    operation = next(op for op in planning.program.module.walk() if isinstance(op, QueryChannelOp))
    ids = tuple((event.data + 1) % 8 for event in operation.event_ids)
    monkeypatch.setitem(operation.attributes, "event_ids", ArrayAttr(IntAttr(i) for i in ids))
    with pytest.raises(CompileError, match="query memory plan snapshot changed"):
        verify_query_invocations(planning, parent, resources=resources, query_plan=plan)


def test_result_identity_and_completed_coverage_are_required(owned):
    planning, parent, resources, plan = owned
    with pytest.raises(CompileError, match="actual checked parent/protocol"):
        prepare_binary_resources(replace(planning, channels={}), parent, resources=resources)
    with pytest.raises(CompileError, match="actual completed memory result"):
        replace(plan, answers=tuple(False for _ in plan.answers)).check(planning, parent)
    with pytest.raises(TypeError):
        resources.used_banks[("aiv0", "V", "MTE3")] = frozenset()


def test_report_flags_are_not_resource_proofs(owned):
    planning, parent, _, _ = owned
    with pytest.raises(CompileError, match="not a report flag"):
        prepare_binary_resources(planning, parent, resources={"verified": True})

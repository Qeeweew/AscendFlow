"""Query normalization feeds the real parent planner without granting outer safety."""

from dataclasses import replace

import pytest

from tests.fixtures.scoped_queries import scoped_query_recurrence
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import BufferOp, CaptureOp, ScopeChannelOp, ScopeSyncOp, TaskOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scope_event_pools import allocate_parent_pools
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_completion import QueryCompletion, QuerySite
from xdsl_npu.scope_sync import _leaf_banks, _plan_scope_parent, _verify_scope_parent
from xdsl_npu.sync import autosync, verify_plan


@pytest.fixture(scope="module")
def parent():
    with limit(60):
        module = lower_source(scoped_query_recurrence)
        # These actual operations, including pointer expressions, must survive.
        preserved = {
            op: (dict(op.attributes), tuple(op.operands))
            for op in module.walk()
            if isinstance(op, (BufferOp, CaptureOp))
        }
        natives = tuple(op.uid.data for op in module.walk() if isinstance(op, TaskOp))
        _plan_scope_parent(extract_scopes(module))
        report, checked, proof = _verify_scope_parent(extract_scopes(module))
    return module, preserved, natives, report, checked, proof


def test_normalized_parent_retains_actual_allocations_captures_and_natives(parent):
    module, preserved, natives, report, checked, _ = parent
    actual = set(module.walk())
    for op, (attributes, operands) in preserved.items():
        assert op in actual
        assert op.attributes == attributes
        assert tuple(op.operands) == operands
    assert tuple(op.uid.data for op in module.walk() if isinstance(op, TaskOp)) == natives
    assert report["native_templates"] == 16
    assert report["parent_count"] is None
    assert report["parent_bound"] == "kv_tiles"
    mapping = checked.access.query
    assert mapping is not None
    assert mapping.contracts["Score"].kind == "epoch_ring"
    assert mapping.contracts["O"].kind == "query_frame"
    for name, root in checked.access.loop.roots.items():
        assert root is mapping.contracts[name].actual_root
    for (actor, name), storage in mapping.storages.items():
        assert storage.external  # Private layout is never an analysis substitution.
        assert mapping.contracts[name].kind in {"readonly", "query_frame"}


def test_parent_only_proof_cannot_be_published_as_query_verification(parent):
    module, _, _, report, checked, _ = parent
    assert report["cross_query_verified"] is False
    assert "remain unverified" in report["verification"]
    assert checked.program.invocations
    assert checked.query_report is None
    before = dump(module)
    with pytest.raises(
        CompileError, match="missing cross-query memory and event lifetime synchronization:"
    ):
        verify_plan(module)
    with pytest.raises(CompileError, match="unsynchronized physical calls"):
        autosync(module)
    assert dump(module) == before


def test_actual_parent_sync_removal_invalidates_normalized_memory_proof(parent):
    module = parent[0].clone()
    for op in list(module.walk()):
        if isinstance(op, (ScopeChannelOp, ScopeSyncOp)):
            op.parent.erase_op(op)
    with limit(30), pytest.raises(CompileError, match="missing scoped (RAW|WAR|WAW)"):
        _verify_scope_parent(extract_scopes(module))


def test_real_query_ir_exposes_missing_state_read_release(parent):
    _, _, _, _, checked, proof = parent
    uid = "aiv0_call0"
    leaf = checked.program.leaves[uid]
    data = next(iter(leaf.loop.actors.values()))
    tasks = {task.uid: task for task in data.program.tasks}
    endpoints = [site for call, site in checked.access.endpoints.values() if call == uid]
    # The query summary demands the final MTE3 reader even though it has no
    # query-local successor. Its retirement before the next fill is unproved.
    store = next(site for site in endpoints if tasks[site.task[1]].pipe == "MTE3")
    fill = next(site for site in endpoints if tasks[site.task[1]].native == "fill")
    query = QueryCompletion(proof)
    source = QuerySite(0, uid, replace(store, point="end"), Position(True, 0))
    target = QuerySite(1, uid, replace(fill, point="begin"), Position(False, 0))
    assert checked.query_summary is not None
    assert not query.covered(source, target, CountDomain(1, 0, exact=1))


def test_real_query_ir_has_unresolved_child_event_republication(parent):
    proof = parent[-1]
    query = QueryCompletion(proof)
    answers = []
    for uid, consumed, published, _ in proof.resource_pairs:
        source = QuerySite(0, uid, consumed, Position(True, 0))
        target = QuerySite(1, uid, published, Position(False, 0))
        answers.append(query.covered(source, target, CountDomain(1, 0, exact=1)))
    assert answers and not all(answers)


def test_actual_parent_pool_widths_share_queries_across_distinct_banks(parent):
    _, _, _, _, checked, proof = parent
    protocols = tuple(checked.channels.values())
    topology = tuple(proof.graph.nodes), tuple(proof.graph.edges)
    statistics = {}
    allocated, _ = allocate_parent_pools(
        protocols, proof, _leaf_banks(checked.program, checked.analyses), statistics=statistics
    )
    assert allocated == protocols
    banks = {
        (actor, command.task.pipe, proof.events[actor, name, "wait"].task.pipe)
        for (actor, name, action), command in proof.events.items()
        if action == "set" and command.kind == "set"
    }
    assert len(banks) > 1
    assert statistics["self_batches"] == 1
    assert statistics["sharing_queries"] == 0
    assert (tuple(proof.graph.nodes), tuple(proof.graph.edges)) == topology

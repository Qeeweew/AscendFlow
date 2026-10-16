"""Local query releases from the public DSL and actual synchronized IR."""

from __future__ import annotations

from dataclasses import replace

import pytest

from xdsl_npu import language as N
from xdsl_npu.budget import limit
from xdsl_npu.ir import QueryChannelOp, QuerySyncOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_checks import check_memory, prepare_memory_checks
from xdsl_npu.scope_query_completion import QueryCompletion, QueryEvent
from xdsl_npu.scope_query_planning import plan_query_memory
from xdsl_npu.scope_query_protocols import read_query_protocols
from xdsl_npu.scope_sync import _plan_scope_parent, _verify_scope_parent
from xdsl_npu.sync import verify_plan

# Runtime dimension names are interpreted by the public frontend.
# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def query_state_store(
    O: N.Tensor((queries, 2, 8), N.f32, access="write"),
    queries: N.int64(minimum=0),
    count: N.int64(minimum=1),
):
    with N.Kernel(2) as group:
        state = N.alloc_ub((2, 8), N.f32, name="state")
        for query in N.serial(N.ceildiv(N.max(0, queries - group), 2)):
            tile = group + query * 2
            for kv in N.Pipelined(count, stages=1):
                with N.Stage(0, name="update"):
                    for row in N.Pipelined(2, stages=2):
                        with N.Stage(0, name="compute"):
                            if kv == 0:
                                N.fill(state[row, :], 1.0)
                            N.mul(state[row, :], 0.5, out=state[row, :])
                        with N.Stage(1, name="store"):
                            if kv == count - 1:
                                N.copy(state[row, :], O[tile, row, :])


@pytest.fixture(scope="module")
def planned():
    with limit(60):
        module = lower_source(query_state_store)
        _plan_scope_parent(extract_scopes(module))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        before = dump(module)
        result = plan_query_memory(checked, parent)
        # The final proof reconstructs everything from actual modified IR.
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        checks = prepare_memory_checks(checked, parent)
        query = QueryCompletion(parent, copies=len(checked.query_summary.parts))
    return module, before, result, checked, parent, checks, query


def test_actual_local_query_memory_is_rechecked_after_protocol_installation(planned):
    module, before, result, checked, _, checks, query = planned
    assert result.initially_missing > 0
    assert len(result.protocols) == 2 and result.memory_checks == len(checks)
    assert result.new_event_reuse_checks > 0
    assert "npu.query_sync" not in before and "npu.query_sync" in dump(module)
    assert read_query_protocols(checked.program, checked.access) == result.protocols
    assert all(check_memory(checks, query))
    assert sum(isinstance(op, QuerySyncOp) for op in module.walk()) == 2 * len(result.protocols)
    for protocol in result.protocols:
        assert protocol.target.query == 1
        assert protocol.source.parent.end and protocol.source.parent.offset == 0
        assert protocol.domain.minimum == 1 and protocol.domain.exact is None
        assert len(protocol.ids) == 1  # Actual consumption supports one flag.
        actor = protocol.source.site.task[0]
        assert query.covered(
            QueryEvent(1, actor, protocol.name, "wait", protocol.target.parent),
            QueryEvent(1, actor, protocol.name, "set", protocol.source.parent),
            protocol.domain,
        )


def test_public_verifier_rebuilds_memory_resources_and_progress(planned):
    report = verify_plan(planned[0])
    assert report["cross_query_verified"] is True
    assert report["query_verification"]["verified"] is True
    assert report["query_verification"]["memory_checks"] == len(planned[-2])
    assert planned[3].query_report is None  # A parent-only proof grants no gate.


def test_removing_real_query_protocol_recreates_the_uncovered_release(planned):
    module = planned[0].clone()
    for op in tuple(module.walk()):
        if isinstance(op, (QueryChannelOp, QuerySyncOp)):
            op.parent.erase_op(op)
    with limit(30):
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        checks = prepare_memory_checks(checked, parent)
        query = QueryCompletion(parent, copies=len(checked.query_summary.parts))
        assert not all(check_memory(checks, query))


def test_planner_rejects_mismatched_actual_program_without_mutation(planned):
    module, _, _, checked, parent, _, _ = planned
    before = dump(module)
    with pytest.raises(CompileError, match="actual checked invocation proof"):
        plan_query_memory(replace(checked, program=replace(checked.program)), parent)
    assert dump(module) == before

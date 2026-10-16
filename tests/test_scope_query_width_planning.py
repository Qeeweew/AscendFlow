"""Small query windows choose the same actual event width as the full proof."""

from dataclasses import replace

import pytest

from tests.test_scope_query_planning import query_state_store
from tests.test_scope_query_transitions import FiniteQueryTransitions
from xdsl_npu import scope_query_planning
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain
from xdsl_npu.ir import QuerySyncOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_checks import prepare_memory_checks
from xdsl_npu.scope_query_completion import QueryCompletion, QueryEvent
from xdsl_npu.scope_query_protocols import install_query_protocols
from xdsl_npu.scope_sync import _plan_scope_parent, _verify_scope_parent
from xdsl_npu.sync import verify_plan


@pytest.fixture(scope="module")
def parent_module():
    with limit(15):
        module = lower_source(query_state_store)
        _plan_scope_parent(extract_scopes(module))
    return module


def observe(monkeypatch):
    windows, batches = [], []

    class ObservedCompletion(QueryCompletion):
        def __init__(self, parent, *, copies=2):
            windows.append(copies)
            super().__init__(parent, copies=copies)

        def covered_many(self, queries):
            queries = tuple(queries)
            batches.append((self, queries))
            return super().covered_many(queries)

    monkeypatch.setattr(scope_query_planning, "QueryCompletion", ObservedCompletion)
    return windows, batches


@pytest.fixture(scope="module")
def planned(parent_module):
    module = parent_module.clone()
    with limit(15), pytest.MonkeyPatch.context() as patch:
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        checks = prepare_memory_checks(checked, parent)
        windows, batches = observe(patch)
        result = scope_query_planning.plan_query_memory(checked, parent)
    return module, checked, checks, result, windows, batches


def test_actual_width_one_planning_uses_one_small_batch_per_graph_change(planned):
    _, checked, checks, result, windows, batches = planned
    copies = len(checked.query_summary.parts)
    assert result.protocols and all(len(p.ids) == 1 for p in result.protocols)
    assert result.new_event_reuse_checks == len(result.protocols)
    assert windows == [copies] * (1 + len(result.protocols))
    assert len(batches) == len(windows)
    memory = tuple((check.source, check.target, check.domain) for check in checks)
    assert batches[0][1] == memory
    for (_, queries), protocol in zip(batches[1:], result.protocols, strict=True):
        assert queries[1:] == memory  # Preserve every prepared count/span case.
        source, target, domain = queries[0]
        assert isinstance(source, QueryEvent) and isinstance(target, QueryEvent)
        assert source.channel == target.channel == protocol.name
        assert source.query == protocol.target.query and target.query == 1
        assert domain == protocol.domain


def test_final_verifier_independently_rebuilds_actual_memory_and_resources(planned):
    module, _, checks, result, _, _ = planned
    with limit(15):
        report = verify_plan(module)
    actual = report["query_verification"]
    assert actual["verified"] and actual["memory_checks"] == len(checks)
    assert actual["resources"]["query_event_reuse_checks"] == len(result.protocols)
    damaged = module.clone()
    wait = next(
        op for op in damaged.walk() if isinstance(op, QuerySyncOp) and op.action.data == "wait"
    )
    wait.parent.erase_op(wait)
    with limit(15), pytest.raises(CompileError, match="missing or changes an actual native"):
        verify_plan(damaged)


@pytest.fixture(scope="module")
def distance_two(parent_module):
    module = parent_module.clone()
    with limit(15):
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        tasks = {
            (uid, task.uid): task
            for uid, leaf in checked.program.leaves.items()
            for data in leaf.loop.actors.values()
            for task in data.program.tasks
        }
        candidate = next(
            candidate
            for candidate, _ in scope_query_planning._candidates(
                prepare_memory_checks(checked, parent)
            )
            if tasks[candidate.source.call, candidate.source.site.task[1]].pipe == "MTE3"
            and tasks[candidate.target.call, candidate.target.site.task[1]].pipe == "V"
        )
        candidate = replace(
            candidate,
            name="distance_two_release",
            target=replace(candidate.target, query=2),
            ids=(7,),
        )
        install_query_protocols(checked.program, checked.access, (candidate,))
    return parent, candidate


def reuse(candidate, width):
    actor = candidate.source.site.task[0]
    return (
        QueryEvent(2, actor, candidate.name, "wait", candidate.target.parent),
        QueryEvent(width, actor, candidate.name, "set", candidate.source.parent),
        candidate.domain,
    )


@pytest.mark.parametrize("copies,expected_windows", ((3, (3, 9)), (9, (9,))))
def test_width_two_matches_full_window_and_literal_actual_event_oracle(
    distance_two, monkeypatch, copies, expected_windows
):
    parent, candidate = distance_two
    memory = tuple(
        (candidate.source, candidate.target, CountDomain(1, 0, exact=count)) for count in (1, 2, 7)
    )
    with limit(15):
        full = QueryCompletion(parent, copies=9).covered_many(
            reuse(candidate, width) for width in range(1, 9)
        )
        assert full == [False] + [True] * 7
        for count in (1, 2, 7):
            finite = FiniteQueryTransitions(parent, count, (candidate,), copies=9)
            assert [finite.covered(*reuse(candidate, width)[:2]) for width in range(1, 9)] == full
        windows, batches = observe(monkeypatch)
        width, answers, attempted = scope_query_planning._query_width(
            parent, copies, candidate, 8, memory
        )
    assert width == full.index(True) + 1 == 2
    assert answers == [True] * len(memory) and attempted == 8
    assert tuple(windows) == expected_windows
    assert batches[0][1] == (reuse(candidate, 1), *memory)
    assert batches[1][1] == tuple(reuse(candidate, width) for width in range(2, 9))
    assert (batches[0][0] is batches[1][0]) == (len(expected_windows) == 1)


def test_insufficient_actual_width_rejects_without_another_graph(distance_two, monkeypatch):
    parent, candidate = distance_two
    windows, batches = observe(monkeypatch)
    with (
        limit(15),
        pytest.raises(CompileError, match="available event widths cannot prove"),
    ):
        scope_query_planning._query_width(parent, 3, candidate, 1, ())
    assert windows == [3] and len(batches) == 1

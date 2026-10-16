"""Actual query suffixes preserve old facts and exact lexical native cuts."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.test_scope_query_planning import query_state_store
from tests.test_scope_query_transitions import FiniteQueryTransitions
from xdsl_npu import scope_query_planning
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import QueryChannelOp, QuerySyncOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_checks import prepare_memory_checks
from xdsl_npu.scope_query_completion import QueryCompletion, QueryEvent, QuerySite
from xdsl_npu.scope_query_protocols import (
    QueryProtocol,
    install_query_protocols,
    read_query_protocols,
)
from xdsl_npu.scope_sync import _plan_scope_parent, _verify_scope_parent

FIRST, LAST = Position(False, 0), Position(True, 0)
COUNTS = (1, 2, 3, 7)


@pytest.fixture(scope="module")
def parent_module():
    with limit(15):
        module = lower_source(query_state_store)
        _plan_scope_parent(extract_scopes(module))
    return module


@pytest.fixture
def actual_parent(parent_module):
    with limit(15):
        _, checked, parent = _verify_scope_parent(extract_scopes(parent_module.clone()))
        tasks = {
            (uid, task.uid): task
            for uid, leaf in checked.program.leaves.items()
            for data in leaf.loop.actors.values()
            for task in data.program.tasks
        }
        protocol = next(
            candidate
            for candidate, _ in scope_query_planning._candidates(
                prepare_memory_checks(checked, parent)
            )
            if tasks[candidate.source.call, candidate.source.site.task[1]].pipe == "MTE3"
            and tasks[candidate.target.call, candidate.target.site.task[1]].pipe == "V"
        )
    return checked, parent, replace(protocol, name="actual_release", ids=(7,)), tasks


def test_actual_append_refreshes_missing_paths_without_reseeding_or_rebuilding(actual_parent):
    checked, parent, protocol, _ = actual_parent
    with limit(15):
        proof = QueryCompletion(parent, copies=3)
        request = protocol.source, protocol.target, protocol.domain
        known = protocol.source, protocol.source, protocol.domain
        assert proof.covered_many((request, known)) == [False, True]
        flow, topology = proof.flow, proof.topology
        origins = dict(proof.origins)
        questions, cases = proof.queries, proof.cases
        known_result = proof.answers[known]

        install_query_protocols(checked.program, checked.access, (protocol,))
        assert proof.append_protocols() == (protocol,)
        assert proof.flow is flow and proof.topology is topology
        assert proof.origins == origins
        assert proof.answers[known] is known_result
        assert request not in proof.answers  # An old failure is not a permanent fact.

        scans = flow.coverage_requests
        assert proof.covered_many((known, request)) == [True, True]
        assert flow.coverage_requests > scans
        scans = flow.coverage_requests
        assert proof.covered_many((known, request)) == [True, True]
        assert flow.coverage_requests == scans
        assert proof.queries == questions and proof.cases == cases
        assert proof.origins == origins

        fresh = QueryCompletion(parent, copies=3)
        assert proof.topology.families == fresh.topology.families
        assert proof.topology.edges == fresh.topology.edges
        assert proof.query_protocol_edges == fresh.query_protocol_edges
        assert fresh.covered_many((known, request)) == [True, True]
        for count in COUNTS:
            finite = FiniteQueryTransitions(parent, count, (protocol,), copies=3)
            assert finite.covered(*request[:2])


def test_appended_same_native_forwarding_keeps_first_last_equality_domains(actual_parent):
    checked, parent, protocol, tasks = actual_parent
    uid, site = next(
        (uid, site)
        for uid, site in checked.access.endpoints.values()
        if site.task[0] == protocol.source.site.task[0]
        and tasks[uid, site.task[1]].native == "muls"
    )
    target = QuerySite(1, uid, replace(site, point="begin"), LAST)
    tail = QueryProtocol("tail_acquire", protocol.source, target, protocol.domain, (6,))
    head = replace(tail, name="head_acquire", target=replace(target, parent=FIRST), ids=(7,))
    with limit(15):
        proof = QueryCompletion(parent, copies=3)
        install_query_protocols(checked.program, checked.access, (tail,))
        proof.append_protocols()
        # Establish a first-protocol origin before appending its lexical successor.
        tail_wait = QueryEvent(1, site.task[0], tail.name, "wait", LAST)
        assert proof.covered(tail_wait, tail_wait, tail.domain)
        origins = dict(proof.origins)
        install_query_protocols(checked.program, checked.access, (head,))
        proof.append_protocols()
        assert all(proof.origins[key] == bit for key, bit in origins.items())
        requests = tuple(
            (tail_wait, QueryEvent(1, site.task[0], head.name, "wait", FIRST),
             CountDomain(1, 0, exact=count))
            for count in COUNTS
        )
        assert proof.covered_many(requests) == [True, False, False, False]
        fresh = QueryCompletion(parent, copies=3)
        assert proof.topology.families == fresh.topology.families
        assert proof.topology.edges == fresh.topology.edges
        assert fresh.covered_many(requests) == [True, False, False, False]
        for request, count in zip(requests, COUNTS, strict=True):
            finite = FiniteQueryTransitions(parent, count, (tail, head), copies=3)
            assert finite.covered(*request[:2]) == (count == 1)


@pytest.mark.parametrize("damage", ("delete", "ids", "order", "native", "missing_wait"))
def test_append_rejects_edits_to_actual_old_protocol_or_native_inputs(actual_parent, damage):
    checked, parent, protocol, _ = actual_parent
    with limit(15):
        install_query_protocols(checked.program, checked.access, (protocol,))
        proof = QueryCompletion(parent, copies=3)
        assert proof.covered(protocol.source, protocol.target, protocol.domain)
        successor = replace(protocol, name="successor", ids=(6,))
        install_query_protocols(checked.program, checked.access, (successor,))
        if damage == "delete":
            for op in tuple(checked.program.module.walk()):
                if isinstance(op, (QueryChannelOp, QuerySyncOp)) and (
                    op.sym_name.data if isinstance(op, QueryChannelOp) else op.channel.data
                ) == protocol.name:
                    op.parent.erase_op(op)
        elif damage == "ids":
            declaration = next(
                op for op in checked.program.module.walk()
                if isinstance(op, QueryChannelOp) and op.sym_name.data == protocol.name
            )
            declaration.attributes["event_ids"] = ArrayAttr((IntAttr(5),))
        elif damage == "order":
            declarations = {
                op.sym_name.data: op for op in checked.program.module.walk()
                if isinstance(op, QueryChannelOp)
            }
            first, last = declarations[protocol.name], declarations[successor.name]
            block = first.parent
            block.detach_op(last)
            block.insert_op_before(last, first)
            for action in ("set", "wait"):
                syncs = {
                    op.channel.data: op for op in checked.program.module.walk()
                    if isinstance(op, QuerySyncOp) and op.action.data == action
                }
                first, last = syncs[protocol.name], syncs[successor.name]
                block = first.parent
                block.detach_op(last)
                block.insert_op_before(last, first)
            assert read_query_protocols(checked.program, checked.access) == (successor, protocol)
        elif damage == "native":
            call = checked.program.leaves[protocol.source.call].call
            call.attributes["stage"] = IntAttr(call.stage.data + 1)
        else:
            wait = next(
                op for op in checked.program.module.walk()
                if isinstance(op, QuerySyncOp) and op.action.data == "wait"
                and op.channel.data == protocol.name
            )
            wait.parent.erase_op(wait)
        old_nodes, old_edges = len(proof.topology.families), proof.topology.edge_count
        with pytest.raises(CompileError, match="snapshot|prefix|missing or changes"):
            proof.append_protocols()
        assert len(proof.topology.families) == old_nodes
        assert proof.topology.edge_count == old_edges


def test_normal_width_one_planning_builds_one_completion_for_all_actual_insertions(
    actual_parent, monkeypatch
):
    checked, parent, _, _ = actual_parent
    created, appends = [], []

    class ObservedCompletion(QueryCompletion):
        def __init__(self, parent, *, copies=2):
            super().__init__(parent, copies=copies)
            created.append(self)

        def append_protocols(self):
            origins, flow = dict(self.origins), self.flow
            result = super().append_protocols()
            assert self.flow is flow
            assert all(self.origins[key] == bit for key, bit in origins.items())
            appends.append(result)
            return result

    monkeypatch.setattr(scope_query_planning, "QueryCompletion", ObservedCompletion)
    with limit(15):
        result = scope_query_planning.plan_query_memory(checked, parent)
        result.check(checked, parent)
    assert created == [result.completion]
    assert len(result.protocols) == len(appends) == 2
    assert tuple(protocol for addition in appends for protocol in addition) == result.protocols
    assert all(len(protocol.ids) == 1 for protocol in result.protocols)
    assert result.completion.queries == result.memory_checks + len(result.protocols)
    assert all(result.answers)


def test_width_search_rebuilds_only_when_the_existing_window_must_grow(actual_parent, monkeypatch):
    checked, parent, protocol, _ = actual_parent
    protocol = replace(protocol, target=replace(protocol.target, query=2))
    with limit(15):
        proof = QueryCompletion(parent, copies=3)
        memory = ((protocol.source, protocol.target, protocol.domain),)
        assert proof.covered_many(memory) == [False]
        install_query_protocols(checked.program, checked.access, (protocol,))
        windows = []

        class ObservedCompletion(QueryCompletion):
            def __init__(self, parent, *, copies=2):
                windows.append(copies)
                super().__init__(parent, copies=copies)

        monkeypatch.setattr(scope_query_planning, "QueryCompletion", ObservedCompletion)
        width, answers, _, final, _ = scope_query_planning._query_width(
            parent, 3, protocol, 8, memory, completion=proof
        )
    assert width == 2 and answers == (True,)
    assert windows == [9]
    assert final is not proof and final.copies == 9
    assert proof.query_protocols == {protocol.name: protocol}

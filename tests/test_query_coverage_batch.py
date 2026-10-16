"""Query fact-read batches keep full endpoint/count identity and finite answers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.fixtures.guarded import native_guarded_recurrence
from tests.test_scope_query_completion import (
    FiniteQueries,
    candidates,
    checked_parent,
    event_candidates,
)
from xdsl_npu.completion import CountDomain, Position, Site
from xdsl_npu.completion_flow import ALL
from xdsl_npu.scope_query_completion import QueryCompletion, QuerySite
from xdsl_npu.symbolic_completion import Family, SymbolicCompletion

POSITIVE = CountDomain(1, 0, minimum=1)


def _reader_fixture(size, *, linked_copies=(0, 1)):
    """Synthetic cost fixture only; actual IR/BFS checks use a separate fixture below."""
    proof = QueryCompletion.__new__(QueryCompletion)
    proof.copies, proof.size = 2, size + 1
    proof.flow = SymbolicCompletion()
    proof.topology = proof.flow.topology
    proof.parent = SimpleNamespace(graph=SimpleNamespace(maximum=None))
    proof.origins, proof.bindings, proof.answers = {}, {}, {}
    proof.queries = proof.cases = proof.carry_edges = proof.counter_edges = 0
    proof.query_protocols, proof.query_protocol_edges = {}, 0
    bindings, sources, targets = {}, [], []
    for query in range(2):
        for index in range(size):
            node = proof.flow.node(Family(0, (0, index), monotone=True))
            assert node == proof._node(query, index)
            site = Site(("aiv0", f"source{index}"), Position(False, 0), "end")
            endpoint = QuerySite(query, f"source{index}", site, Position(False, 0))
            bindings[endpoint] = ((node, (0, 0), ALL),)
            sources.append(endpoint)
        node = proof.flow.node(Family(0, (0, size - 1)))
        assert node == proof._node(query, size)
        site = Site(("aiv0", "target"), Position(False, 0), "begin")
        endpoint = QuerySite(query, "target", site, Position(False, 0))
        bindings[endpoint] = ((node, (0, 0), ALL),)
        bindings[replace(endpoint, parent=Position(False, 1))] = ((node, (0, 1), ALL),)
        targets.append(endpoint)
        if query in linked_copies:
            for index in range(size):
                proof.flow.edge(proof._node(query, index), node)
    proof._check_snapshot = lambda: None
    proof._bindings = bindings.__getitem__
    return proof, sources, targets


@pytest.mark.parametrize("size", (1, 8, 64))
def test_query_batch_distributes_m_origins_after_one_m_record_target_scan(size):
    proof, sources, targets = _reader_fixture(size)
    requests = [(source, targets[0], POSITIVE) for source in sources[:size]]
    answers = proof.covered_many(iter(requests + requests[::-1]))
    assert answers == [True] * (2 * size)
    assert len(proof.flow.facts[proof._node(0, size)]) == size
    counts = proof.statistics()
    assert counts["coverage_batches"] == 1
    assert counts["coverage_requests"] == counts["coverage_result_items"] == size
    assert counts["coverage_fact_scans"] == size
    assert counts["queries"] == counts["cases"] == counts["origins"] == size


def test_query_groups_keep_different_copy_and_point_identities():
    proof, sources, targets = _reader_fixture(2, linked_copies=(0,))
    requests = [
        (sources[0], targets[0], POSITIVE),
        (sources[0], replace(targets[0], parent=Position(False, 1)), POSITIVE),
        (sources[1], replace(targets[0], parent=Position(False, 1)), POSITIVE),
        (sources[2], targets[1], POSITIVE),
        (sources[0], targets[1], POSITIVE),
    ]
    assert proof.covered_many(requests) == [True, False, True, False, False]
    counts = proof.statistics()
    assert counts["coverage_batches"] == 3  # Copy 0 point 0/1, copy 1 point 0.
    assert counts["coverage_fact_scans"] == 4
    assert len(proof.origins) == 3


def test_query_groups_keep_complete_needed_domain_instead_of_residue_only():
    proof, sources, targets = _reader_fixture(1)
    domains = (
        CountDomain(2, 0, minimum=2, maximum=8),
        CountDomain(2, 0, minimum=4, maximum=6),
        CountDomain(2, 0, minimum=6),
        CountDomain(1, 0, exact=4),
        CountDomain(2, 1, minimum=1, maximum=7),
    )
    requests = [(sources[0], targets[0], domain) for domain in domains]
    results = proof.coverage_many(requests)
    for domain, result in zip(domains, results, strict=True):
        for count in range(12):
            assert any(piece.contains(count) for piece in result) == domain.contains(count)
    assert proof.statistics()["coverage_batches"] == len(domains)
    assert proof.statistics()["coverage_fact_scans"] == len(domains)
    assert len(proof.origins) == 1


def test_later_query_batch_reads_new_origins_from_current_facts():
    proof, sources, targets = _reader_fixture(2)
    assert proof.covered(sources[0], targets[0])
    assert proof.statistics()["coverage_fact_scans"] == 1
    assert proof.covered(sources[1], targets[0])
    counts = proof.statistics()
    assert counts["coverage_fact_scans"] == 3
    assert counts["coverage_batches"] == counts["coverage_requests"] == 2


@pytest.fixture(scope="module")
def actual_parent():
    return checked_parent(native_guarded_recurrence)


def test_actual_ir_batch_answers_match_independent_finite_bfs_and_separate_scalar_flow(
    actual_parent,
):
    available = event_candidates(actual_parent) + candidates(actual_parent)
    selected = available[:: max(1, len(available) // 8)][:8]
    assert len(selected) >= 2
    domains = (
        CountDomain(2, 0, minimum=2, maximum=6),
        CountDomain(2, 1, minimum=1, maximum=5),
        CountDomain(1, 0, exact=2),
    )
    requests = [
        (replace(source, query=copy), replace(target, query=target_copy), domain)
        for source, target in zip(selected, selected[::-1], strict=True)
        for copy, target_copy in ((0, 1), (1, 2), (2, 0))
        for domain in domains
    ]
    requests += [(endpoint, endpoint, POSITIVE) for endpoint in selected]
    requests += requests[:4]
    proof, scalar = (
        QueryCompletion(actual_parent, copies=3),
        QueryCompletion(actual_parent, copies=3),
    )
    assert proof.flow is not scalar.flow and proof.flow.facts is not scalar.flow.facts
    actual = proof.coverage_many(iter(requests))
    assert actual == [scalar.coverage(*request) for request in requests]
    observed = set()
    for count in range(7):
        finite = FiniteQueries(actual_parent, count, copies=3)
        for (source, target, domain), result in zip(requests, actual, strict=True):
            expected = domain.contains(count) and finite.covered(source, target)
            observed.add(expected)
            assert any(piece.contains(count) for piece in result) == expected, (
                count,
                source,
                target,
                domain,
            )
    assert observed == {False, True}

"""Ranks are preserved across graph growth; no old edge is re-solved."""

from types import SimpleNamespace

import pytest

from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_graph import Command, Graph


def node(graph):
    return graph.node(
        Command(len(graph.nodes), "aiv0", SimpleNamespace(stage=0), "task"),
        0,
        "begin",
        guard=(0, 0),
    )


def valid(graph, result):
    stride, rank = result
    assert len(rank) == len(graph.nodes)
    assert all(rank[b] + stride * delta > rank[a] for a, b, delta in graph.edges)


def test_append_only_chain_checks_each_constraint_once():
    graph = Graph(1, None)
    old = node(graph)
    frozen = []
    for _ in range(128):
        new = node(graph)
        graph.edge((old, 0), (new, 0))
        result = graph.potential()
        valid(graph, result)
        frozen.append((result, tuple(result[1])))
        assert graph.potential() is result
        old = new
    assert graph.potential_work["new_edges"] == len(graph.edges)
    assert graph.potential_work["relaxations"] <= 2 * len(graph.edges)
    assert graph.potential_work["scaled"] > 0
    assert all(result[1] == ranks for result, ranks in frozen)


def test_new_constraint_propagates_through_old_successors_and_signed_carries():
    graph = Graph(1, None)
    a, b, c, d = (node(graph) for _ in range(4))
    graph.edge((b, 0), (c, 0))
    graph.edge((c, 0), (d, 0))
    first = graph.potential()
    valid(graph, first)
    graph.edge((a, 2), (b, 0))
    second = graph.potential()
    valid(graph, second)
    assert second[1][d] > first[1][d]
    graph.edge((d, 0), (a, 3))
    valid(graph, graph.potential())
    assert graph.potential_work["new_edges"] == 4


def test_appended_zero_advance_cycle_is_rejected():
    graph = Graph(1, None)
    a, b = node(graph), node(graph)
    graph.edge((a, 0), (b, 0))
    valid(graph, graph.potential())
    graph.edge((b, 0), (a, 0))
    with pytest.raises(CompileError, match="completion cycle"):
        graph.potential()


def test_empty_domain_does_not_invent_execution():
    graph = Graph(1, 0)
    a = node(graph)
    graph.edge((a, 0), (a, 0))
    assert graph.potential()[1] == (0,)

"""Owned activation appends are checked once; external dicts remain mutable inputs."""

import pytest

from tests.test_scope_guarded_completion import command, edge, finite_covered, singleton
from xdsl_npu.completion import CountDomain
from xdsl_npu.loop_graph import Graph
from xdsl_npu.model import CompileError
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion, NodeActivations


def test_published_activation_and_absence_cannot_change_after_import():
    graph = Graph(1, None)
    source, target = (command(graph, index) for index in range(2))
    values = NodeActivations({source[1][0]: singleton()})
    proof = GuardedCompletion(graph, values)
    values[source[1][0]] = singleton()
    assert values.revision == 1
    with pytest.raises(CompileError, match="activation changed"):
        values[source[1][0]] = singleton(1)
    with pytest.raises(CompileError, match="activation changed"):
        del values[source[1][0]]
    with pytest.raises(CompileError, match="activation changed"):
        values[target[1][0]] = singleton()
    proof._sync()
    assert proof.statistics()["activation_checks"] == 1


@pytest.mark.parametrize("size", (8, 64, 512))
def test_unchanged_owned_activations_are_not_rescanned(size):
    graph = Graph(1, None)
    values = NodeActivations()
    for index in range(size):
        node = command(graph, index)[1][0]
        values[node] = singleton()
    proof = GuardedCompletion(graph, values)
    for _ in range(16):
        proof._sync()
    assert proof.statistics()["activation_checks"] == size
    node = command(graph, size)[1][0]
    values[node] = singleton(1)
    proof._sync()
    assert proof.statistics()["activation_checks"] == size + 1


def test_plain_dict_activation_mutation_still_requires_actual_snapshot_validation():
    graph = Graph(1, None)
    node = command(graph, 0)[1][0]
    values = {node: singleton()}
    proof = GuardedCompletion(graph, values)
    for _ in range(8):
        proof._sync()
    assert proof.statistics()["activation_checks"] == 17
    values[node] = singleton(1)
    with pytest.raises(CompileError, match="activation changed"):
        proof._sync()


def test_owned_addition_requires_an_actual_appended_graph_node():
    graph = Graph(1, None)
    command(graph, 0)
    values = NodeActivations()
    proof = GuardedCompletion(graph, values)
    values[len(graph.nodes)] = singleton()
    with pytest.raises(CompileError, match="no actual graph node"):
        proof._sync()


@pytest.mark.parametrize("period", (1, 3))
def test_owned_incremental_activation_paths_match_literal_count_instances(period):
    graph, values = Graph(period, None), NodeActivations()
    source, target = (command(graph, index) for index in range(2))
    proof = GuardedCompletion(graph, values)
    activation = Activation(CountDomain(1, 0, minimum=1))
    for index in range(2, 7):
        intermediate = command(graph, index)
        guard = CountDomain(2, index % 2, minimum=1)
        values.update((node, Activation(guard)) for node in intermediate[1])
        edge(graph, source, intermediate)
        edge(graph, intermediate, target)
        for count in range(1, 9):
            exact = Activation(CountDomain(1, 0, exact=count))
            assert proof.covered(source[0], target[0], 0, exact) == finite_covered(
                graph, values, source, target, 0, exact, count
            )
        if index >= 3:
            assert proof.covered(source[0], target[0], 0, activation)

"""Shared native topology, independent finite paths, and real symbolic costs."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from xdsl_npu.completion_flow import ALL
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.symbolic_completion import CompletionTopology, Family, SymbolicCompletion


def _graph():
    graph = Graph(1, None)
    source, target = [
        Command(i, "aiv0", SimpleNamespace(stage=0), kind)
        for i, kind in enumerate(("task", "task"))
    ]
    a = graph.node(source, 0, "end")
    b = graph.node(target, 0, "begin")
    return graph, source, target, a, b


def _finite_paths(families, arcs, count, terminal, origin):
    """Literal generations and BFS; no production interval/coverage routines."""
    instances = set()
    for node, family in enumerate(families):
        if family.last is not None and not terminal:
            continue
        stop = family.horizon[0] * count + family.horizon[1]
        last = None if family.last is None else family.last[0] * count + family.last[1]
        for generation in range(family.lower, stop + 1):
            if generation % family.period == family.residue and (
                last is None or generation == last
            ):
                instances.add((node, generation))
    following = defaultdict(set)
    for source, target, shift in arcs:
        for node, generation in instances:
            destination = target, generation + shift
            if node == source and destination in instances:
                following[node, generation].add(destination)
    for node, generation in instances:
        family = families[node]
        if family.monotone and (node, generation + family.period) in instances:
            following[node, generation].add((node, generation + family.period))
    reached = {origin} if origin in instances else set()
    pending = list(reached)
    while pending:
        for target in following[pending.pop()]:
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def test_shared_activation_domains_equal_independent_finite_paths():
    families = (
        Family(0, (1, -1), 2, 0),
        Family(0, (1, -1), 2, 0, monotone=True),
        Family(0, (1, -1), 2, 0, last=(1, -1)),
        Family(0, (1, -1), 2, 0),
    )
    arcs = ((0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 2))
    topology = CompletionTopology()
    for family in families:
        topology.node(family)
    for arc in arcs:
        topology.edge(*arc)
    domains = []
    for count in range(14):
        for terminal in (False, True):
            flow = SymbolicCompletion(topology=topology, count=count, terminal=terminal)
            domains.append(flow)
            flow.seed(0, (0, 0), ALL, 1)
            flow.propagate()
            actual = {
                (node, generation)
                for node in range(len(families))
                for generation in range(count)
                if flow.coverage(node, (0, generation), 1, ALL)
            }
            assert actual == _finite_paths(families, arcs, count, terminal, (0, 0))
    assert all(
        flow.edges is topology.edges and flow.families is topology.families for flow in domains
    )
    # A terminal path in one count domain must not publish facts in another.
    assert not domains[2 * 5].coverage(3, (0, 4), 1, ALL)
    assert domains[2 * 5 + 1].coverage(3, (0, 4), 1, ALL)


def test_incremental_native_arcs_update_every_domain_without_cross_domain_facts():
    graph, source, target, a, b = _graph()
    queries = [(source, 0, target, 0), (source, 0, target, 3), (source, 3, target, 3)]
    assert graph.covered_many(queries) == [False, False, False]
    proof = graph.completion()
    domains = dict(proof.flows)
    # A new accepted protocol may add a node and multiple arcs after queries
    # have already propagated. Replaying only the newest domain is unsound.
    capture = graph.node(Command(2, "aiv0", source.task, "set"), 0, "end")
    graph.edge((a, 0), (capture, 0))
    assert graph.covered_many(queries) == [False, False, False]
    graph.edge((capture, 0), (b, 0))
    assert graph.covered_many(queries) == [True, False, True]
    assert all(proof.flows[key] is flow for key, flow in domains.items())
    assert all(flow.edges is proof.topology.edges for flow in domains.values())
    # An individual Set effect is not a cumulative native retirement prefix.
    assert not graph.covered_many([(source, 0, target, 2)])[0]
    with pytest.raises(CompileError, match="cannot mutate"):
        next(iter(domains.values())).edge(a, b)
    independent, other_source, other_target, *_ = _graph()
    assert independent.covered_many([(other_source, 0, other_target, 0)]) == [False]
    assert independent.completion().topology is not proof.topology


def test_query_domain_growth_keeps_one_topology_and_counts_real_facts():
    measurements = []
    for size in (1, 9, 31):
        graph, source, target, a, b = _graph()
        graph.edge((a, 0), (b, 0))
        queries = [(source, 0, target, 0, first) for first in range(size)]
        assert all(graph.covered_many(queries))
        counts = graph.completion_statistics()
        assert counts["query_domains"] == size
        assert counts["topology_instances"] == 1
        assert counts["condition_facts"] == counts["fact_nodes"] == 2 * size
        assert counts["fact_updates"] >= counts["condition_facts"] > 0
        assert counts["fact_transfers"] > 0
        assert counts["translated_domain_states"] == 0  # A different, unused representation.
        measurements.append((counts["topology_nodes"], counts["topology_edges"]))
    assert measurements == [(2, 1)] * 3


def test_new_terminal_path_cannot_contaminate_existing_nonterminal_domains():
    graph, source, target, a, b = _graph()
    terminal = Command(2, "aiv0", source.task, "task", last_gap=0)
    begin = graph.node(terminal, 0, "begin")
    end = graph.node(terminal, 0, "end")
    queries = [
        (source, 0, target, 0),
        (source, 3, target, 3),
        (source, 0, terminal, 0),
        (source, 0, terminal, 3),
        (source, 3, terminal, 3),
    ]
    assert graph.covered_many(queries) == [False] * 5
    graph.edge((a, 0), (begin, 0))
    graph.edge((begin, 0), (end, 0))
    graph.edge((end, 0), (b, 0))
    # The accepted arcs are shared. Last-only activation still belongs to
    # each domain, so ordinary paths cannot borrow the new terminal bridge.
    assert graph.covered_many(queries) == [False, False, True, False, True]


def test_activation_bindings_are_sparse_and_domain_local():
    topology = CompletionTopology()
    source = topology.node(Family(1, (1, -1), 2, 1))
    terminal = topology.node(Family(1, (1, -1), 2, 1, last=(1, -1)))
    for _ in range(32):
        topology.node(Family(0, (1, -1)))
    topology.edge(source, terminal, 4)
    flows = [
        SymbolicCompletion(topology=topology, count=count, terminal=last)
        for count, last in ((6, True), (5, True), (6, False))
    ]
    for flow in flows:
        assert not flow.activation
        flow.seed(source, (0, 1), ALL, 1)
        flow.propagate()
        bindings = dict(flow.activation)
        # Unreachable families allocate neither bound records nor fact maps.
        assert set(bindings) == {source, terminal}
        assert set(flow.facts) <= {source, terminal}
        flow.seed(source, (0, 3), ALL, 2)
        flow.propagate()
        assert all(flow.activation[node] is bound for node, bound in bindings.items())
    assert flows[0].coverage(terminal, (0, 5), 1, ALL)
    assert not flows[1].coverage(terminal, (0, 4), 1, ALL)  # Last has the wrong residue.
    assert not flows[2].coverage(terminal, (0, 5), 1, ALL)  # Last-only work is inactive.
    assert not flows[0].coverage(terminal, (0, 5), 2, ALL)  # The second origin overshoots.


def test_condition_comparisons_and_superseded_records_are_counted():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 9)))
    flow.seed(node, (0, 1), ALL, 1)
    flow.add_fact(node, (0, 0), (0, 2), ALL, 1)
    counts = flow.statistics()
    assert counts["condition_facts"] == 1
    assert counts["peak_condition_facts"] == 1
    assert counts["fact_comparisons"] == 2
    assert counts["fact_updates"] == 3  # Insert, remove the narrower record, insert.
    assert counts["fact_attempts"] == 2


def test_wider_fact_preserves_existing_bits_at_its_exact_condition():
    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 9)))
    flow.seed(node, (0, 4), ALL, 1)
    flow.add_fact(node, (0, 0), (0, 8), ALL, 2)
    flow.add_fact(node, (0, 0), (0, 8), ALL, 1 | 4)
    assert flow.facts[node] == {((0, 0), (0, 8), ALL): 1 | 2 | 4}
    counts = flow.statistics()
    assert counts["condition_facts"] == 1 and counts["peak_condition_facts"] == 2
    # The independent bit 2 skips both scans; the mixed third insert still
    # compares both records twice before superseding the narrower fact.
    assert counts["fact_attempts"] == 3 and counts["fact_comparisons"] == 4
    assert counts["fact_updates"] == 4


def test_public_verifier_rebuilds_after_actual_release_is_removed():
    from xdsl_npu import compile
    from xdsl_npu import language as N
    from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp
    from xdsl_npu.sync import verify_plan

    artifact = compile(
        N.from_source(
            """
            def pipeline(X:N.Tensor((n,16),N.f32), Y:N.Tensor((n,16),N.f32,access="write"), n:N.int64):
                with N.Kernel():
                    x=N.alloc_ub((2,16))
                    y=N.alloc_ub((2,16))
                    for t in N.Pipelined(n,stages=3):
                        with N.Stage(0,name="load"):
                            N.copy(X[t,:],x[t%2,:])
                        with N.Stage(1,name="compute"):
                            N.mul(x[t%2,:],2.0,out=y[t%2,:])
                        with N.Stage(2,name="store"):
                            N.copy(y[t%2,:],Y[t,:])
            """,
            disjoint_external=True,
        )
    )
    counts = artifact.report["completion_statistics"]
    assert counts["query_domains"] > 0 and counts["condition_facts"] > 0
    assert counts["topology_instances"] == 1
    assert "proof_domain_states" not in artifact.report
    plan = artifact.plan
    release = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data > 0
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == release.sym_name:
            op.parent.erase_op(op)
    release.parent.erase_op(release)
    with pytest.raises(CompileError, match="missing loop WAR"):
        verify_plan(plan)

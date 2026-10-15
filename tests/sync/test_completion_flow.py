"""Multi-source fact propagation versus finite reachability on actual edges."""

import random
from collections import defaultdict

import pytest

from xdsl_npu.completion import CountDomain, LeafCompletion
from xdsl_npu.completion_flow import ALL, CompletionFlow


@pytest.mark.parametrize("seed", range(8))
def test_batch_facts_after_each_protocol_match_independent_per_count_bfs(seed):
    rng = random.Random(seed)
    flow = CompletionFlow()
    nodes = [flow.node() for _ in range(7)]
    edges = defaultdict(list)
    for source in nodes[:3]:
        flow.add_fact(source, ALL, 1 << source)
    for _ in range(18):
        source, target = rng.choices(nodes, k=2)
        period = rng.choice((1, 2, 3))
        minimum = rng.randrange(4)
        guard = CountDomain(
            period, rng.randrange(period), minimum=minimum, maximum=rng.choice((None, minimum + 6))
        )
        edges[source].append((target, guard))
        flow.edge(source, target, guard)
        flow.propagate()
        for count in range(16):
            for origin in nodes[:3]:
                seen, work = {origin}, [origin]
                while work:
                    for nxt, active in edges[work.pop()]:
                        if active.contains(count) and nxt not in seen:
                            seen.add(nxt)
                            work.append(nxt)
                actual = {
                    node
                    for node in nodes
                    if any(
                        bits & (1 << origin) and active.contains(count)
                        for active, bits in flow.facts[node].items()
                    )
                }
                assert actual == seen


def test_frame_planning_and_verification_do_not_call_pairwise_leaf_oracle(monkeypatch):
    from examples.state.frame_snapshot import frame_state_sum
    from xdsl_npu import compile, verify_plan
    from xdsl_npu.loop_graph import Graph

    def forbidden(*args, **kwargs):
        raise AssertionError("region composition called a pairwise leaf oracle")

    monkeypatch.setattr(LeafCompletion, "prove", forbidden)
    artifact = compile(frame_state_sum.specialize(B=16, GROUPS=1))
    assert artifact.report["native_templates"] == 7
    assert artifact.report["sequence_native_relation_checks"] == 0
    assert artifact.report["invocation_path_visits"] == 0
    assert artifact.asc.count("for (int64_t") == 2
    # Re-extract the final IR with the old path search disabled entirely.
    # Leaf planning still has legacy pruning; the final local checker does not.
    monkeypatch.setattr(Graph, "path", forbidden)
    assert verify_plan(artifact.plan) == artifact.report

"""Two-copy topology units; these original fixtures have no serial query IR.

The finite oracle instantiates only small test counts. Production queries keep
one symbolic KV N and never instantiate the serial query count or KV extent.
"""

from collections import defaultdict
from dataclasses import replace
from random import Random

import pytest

from tests.fixtures.guarded import native_guarded_recurrence
from tests.fixtures.nested import nested_cv
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.child_completion import ChildPort, FixedChildProjection
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import CallOp, dump
from xdsl_npu.model import CompileError
from xdsl_npu.scope_query_completion import QueryCompletion, QuerySite
from xdsl_npu.scope_sync import ParentCompletion, _scope_period, verify_scopes

FIRST, LAST = Position(False, 0), Position(True, 0)


def checked_parent(function):
    artifact = compile(function, timeout_s=40)
    _, checked = verify_scopes(artifact.plan, _details=True)
    proof = ParentCompletion(
        checked.program,
        checked.analyses,
        checked.access,
        _scope_period(checked.access, checked.channels.values()),
        extra_protocols=checked.channels.values(),
    )
    for protocol in checked.channels.values():
        proof.insert(protocol)
    return proof


@pytest.fixture(
    scope="module", params=(native_guarded_recurrence, nested_cv), ids=("guarded", "mixed")
)
def parent(request):
    return checked_parent(request.param)


def sites(parent):
    return sorted(
        parent.completion.commands.keys() | parent.completion.case_commands.keys(), key=repr
    )


def inputs(projection):
    if isinstance(projection, FixedChildProjection):
        return set(projection.inputs)
    return {port for case in projection.cases for port in case.projection.inputs}


class FiniteQueries:
    """Expand original Graph arcs and actual CallOp physical slots independently."""

    def __init__(self, parent, count, *, copies=2):
        self.parent, self.count = parent, count
        self.following = defaultdict(set)
        self.nodes = set()
        self.closures = {}
        graph, period = parent.graph, parent.period
        for query in range(copies):
            for node, meta in enumerate(graph.nodes):
                activation = parent.node_activations.get(node)
                for generation in range(meta.lower, count - meta.upper):
                    if generation % period != meta.residue:
                        continue
                    if meta.last_gap is not None and generation != count - 1 - meta.last_gap:
                        continue
                    if activation is not None and not (
                        activation.count.contains(count)
                        and generation % activation.period in activation.residues
                        and activation.first.evaluate(count)
                        <= generation
                        <= activation.last.evaluate(count)
                    ):
                        continue
                    self.nodes.add((query, node, generation))
            for source, target, cycles in graph.edges:
                shift = cycles * period + graph.nodes[target].residue - graph.nodes[source].residue
                source_meta = graph.nodes[source]
                for generation in range(source_meta.lower, count - source_meta.upper):
                    self.edge((query, source, generation), (query, target, generation + shift))

        # Reproduce the actor's physical parent loop and source issue order.
        # This deliberately does not use the adapter's boundary expressions.
        completion = parent.completion
        for actor, stream in parent.program.parents.items():
            calls = [op for op in stream.body.block.ops if isinstance(op, CallOp)]
            lexical = {op: index for index, op in enumerate(stream.body.block.ops)}
            calls.sort(key=lambda op: (op.issue_order.data, lexical[op]))
            maximum = max(call.stage.data for call in calls)
            ports = {call.uid.data: inputs(completion.projections[call.uid.data]) for call in calls}
            for port in set.union(*ports.values()):
                if port.actor != actor:
                    continue
                timeline = []
                for step in range(count + maximum):
                    for call in calls:
                        uid = call.uid.data
                        if port not in ports[uid]:
                            continue
                        generation = step - call.stage.data
                        residue = generation % period
                        before = completion.prefixes[uid, actor, port.pipe, port.kind, residue]
                        after = completion.active_ports[uid, replace(port, side="output"), residue]
                        timeline.append(((before, generation), (after, generation)))
                if timeline:
                    target, target_g = timeline[0][0]
                    for source, source_g in timeline[-1]:
                        for query in range(copies - 1):
                            self.edge((query, source, source_g), (query + 1, target, target_g))

    def edge(self, source, target):
        if source in self.nodes and target in self.nodes:
            self.following[source].add(target)

    def endpoint(self, endpoint):
        generation = endpoint.parent.evaluate(self.count)
        key = endpoint.call, endpoint.site, generation % self.parent.period
        completion = self.parent.completion
        nodes = completion.case_sites.get(key)
        if nodes is None:
            nodes = (completion.sites[key],)
        return tuple(
            point for node in nodes if (point := (endpoint.query, node, generation)) in self.nodes
        )

    def reached(self, source):
        if source in self.closures:
            return self.closures[source]
        seen, pending = {source}, [source]
        while pending:
            for target in self.following[pending.pop()]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        self.closures[source] = seen
        return seen

    def covered(self, source, destination):
        sources, targets = self.endpoint(source), self.endpoint(destination)
        return bool(sources and targets) and all(
            target in self.reached(first) for first in sources for target in targets
        )


def candidates(parent, *, query=0):
    return [
        QuerySite(query, uid, site, position)
        for uid, site in sites(parent)
        for position in (FIRST, Position(False, 1), LAST)
    ]


@pytest.fixture(scope="module")
def finite_oracle(parent):
    graphs = {}

    def build(count, *, copies=2):
        key = count, copies
        if key not in graphs:
            graphs[key] = FiniteQueries(parent, count, copies=copies)
        return graphs[key]

    return build


@pytest.mark.parametrize("count", (1, 2, 3, 7))
def test_actual_guarded_and_mixed_graphs_match_independent_two_query_execution(
    parent, finite_oracle, count
):
    proof, finite = QueryCompletion(parent), finite_oracle(count)
    available = candidates(parent)
    rng = Random(9010 + count)
    queries = [
        (
            replace(rng.choice(available), query=rng.randrange(2)),
            replace(rng.choice(available), query=rng.randrange(2)),
        )
        for _ in range(72)
    ]
    # Include identities to distinguish an inactive native from a true path.
    queries.extend((endpoint, endpoint) for endpoint in available[:: max(1, len(available) // 12)])
    for source, destination in queries:
        expected = finite.covered(source, destination)
        assert proof.covered(source, destination, CountDomain(1, 0, exact=count)) == expected, (
            source,
            destination,
            count,
        )


def test_conditional_count_domains_keep_actual_endpoint_existence(parent, finite_oracle):
    proof = QueryCompletion(parent)
    available = candidates(parent)
    domain = CountDomain(2, 1, minimum=1, maximum=7)
    for source in available[:: max(1, len(available) // 9)]:
        target = replace(source, query=1)
        coverage = proof.coverage(source, target, domain)
        for count in (0, 1, 2, 3, 5, 7, 8):
            expected = domain.contains(count) and finite_oracle(count).covered(source, target)
            assert any(piece.contains(count) for piece in coverage) == expected


def test_empty_parent_native_is_uncovered_including_identity(parent):
    endpoint = candidates(parent)[0]
    proof = QueryCompletion(parent)
    for target in (endpoint, replace(endpoint, query=1)):
        assert proof.coverage(endpoint, target, CountDomain(1, 0, exact=0)) == ()
        assert not proof.covered(endpoint, target, CountDomain(1, 0, exact=0))


def test_copy_preserves_families_conditions_and_independent_actor_pipe_state(parent):
    before = dump(parent.program.module)
    original_nodes, original_edges = len(parent.graph.nodes), len(parent.graph.edges)
    proof = QueryCompletion(parent)
    assert dump(parent.program.module) == before
    assert (len(parent.graph.nodes), len(parent.graph.edges)) == (original_nodes, original_edges)
    assert proof.topology.families[: proof.size] == proof.topology.families[proof.size :]
    assert proof.conditions[: proof.size] == proof.conditions[proof.size :]
    cross = [
        (source, target, source_point, target_point)
        for source, edges in enumerate(proof.topology.edges)
        for target, _, source_point, target_point, _ in edges
        if source < proof.size <= target
    ]
    assert cross
    for source, target, source_point, target_point in cross:
        first, last = parent.graph.nodes[source], parent.graph.nodes[target - proof.size]
        assert first.command.actor == last.command.actor
        assert first.point == last.point or first.command.kind == "scope_component"
        assert source_point[0] == 1 and target_point[0] == 0
    assert not any(
        target < proof.size <= source
        for source, edges in enumerate(proof.topology.edges)
        for target, *_ in edges
    )
    if len(parent.program.parents) == 3:
        assert parent.completion.max_stage == {"aic": 0, "aiv0": 1, "aiv1": 1}
        targets = {
            parent.graph.nodes[target - proof.size].command.actor: point
            for _, target, _, point in cross
        }
        assert targets == {"aic": (0, 0), "aiv0": (0, -1), "aiv1": (0, -1)}


def test_old_native_end_does_not_complete_other_pipes_or_actors(parent, finite_oracle):
    proof = QueryCompletion(parent)
    available = candidates(parent)
    count, finite = 3, finite_oracle(3)
    ends = [point for point in available if point.site.point == "end" and point.parent == LAST]
    begins = [point for point in available if point.site.point == "begin" and point.parent == FIRST]
    negatives = 0
    for source in ends[:: max(1, len(ends) // 8)]:
        for target in begins[:: max(1, len(begins) // 8)]:
            target = replace(target, query=1)
            if (
                finite.endpoint(source)
                and finite.endpoint(target)
                and not finite.covered(source, target)
            ):
                assert not proof.covered(source, target, CountDomain(1, 0, exact=count))
                negatives += 1
    assert negatives


def test_child_event_consumption_queries_keep_wait_and_set_identity(parent, finite_oracle):
    proof = QueryCompletion(parent)
    assert parent.resource_pairs
    for uid, wait, send, _ in parent.resource_pairs:
        source, target = QuerySite(0, uid, wait, LAST), QuerySite(1, uid, send, FIRST)
        assert (source.site.point, source.site.action) == ("begin", "wait")
        assert (target.site.point, target.site.action) == ("end", "set")
        for count in (1, 2, 3, 7):
            expected = finite_oracle(count).covered(source, target)
            assert proof.covered(source, target, CountDomain(1, 0, exact=count)) == expected
    # A Set effect and the producer's native end must remain different nodes.
    for uid, _, send, _ in parent.resource_pairs:
        native = replace(send, point="end", channel="", action="")
        if (uid, native) in parent.completion.commands or (
            uid,
            native,
        ) in parent.completion.case_commands:
            assert set(parent.completion.commands_for(uid, send)).isdisjoint(
                parent.completion.commands_for(uid, native)
            )


def test_native_completion_cannot_replace_old_wait_consumption(parent, finite_oracle):
    proof, finite = QueryCompletion(parent), finite_oracle(1)
    differences = 0
    known = parent.completion.commands.keys() | parent.completion.case_commands.keys()
    for uid, wait, send, _ in parent.resource_pairs:
        native = replace(send, channel="", action="")
        if (uid, native) not in known:
            continue
        consumed = QuerySite(0, uid, wait, LAST)
        native_end = QuerySite(0, uid, native, LAST)
        republished = QuerySite(1, uid, send, FIRST)
        if finite.covered(native_end, republished) and not finite.covered(consumed, republished):
            assert proof.covered(native_end, republished, CountDomain(1, 0, exact=1))
            assert not proof.covered(consumed, republished, CountDomain(1, 0, exact=1))
            differences += 1
    assert differences


def test_set_effect_does_not_enter_native_retirement_prefix(parent, finite_oracle):
    proof, finite = QueryCompletion(parent), finite_oracle(1)
    differences = 0
    known = parent.completion.commands.keys() | parent.completion.case_commands.keys()
    for uid, _, send, _ in parent.resource_pairs:
        native = replace(send, channel="", action="")
        retire = replace(native, point="retire")
        if (uid, native) not in known or (uid, retire) not in known:
            continue
        native_end = QuerySite(0, uid, native, LAST)
        set_effect = QuerySite(0, uid, send, LAST)
        following_retire = QuerySite(1, uid, retire, FIRST)
        if finite.covered(native_end, following_retire) and not finite.covered(
            set_effect, following_retire
        ):
            assert proof.covered(native_end, following_retire, CountDomain(1, 0, exact=1))
            assert not proof.covered(set_effect, following_retire, CountDomain(1, 0, exact=1))
            differences += 1
    assert differences


def test_three_static_copies_keep_n1_distance_two_and_actor_identity(parent, finite_oracle):
    proof, finite = QueryCompletion(parent, copies=3), finite_oracle(1, copies=3)
    available = [point for point in candidates(parent) if point.parent == FIRST]
    rng = Random(9431)
    queries = [(rng.choice(available), replace(rng.choice(available), query=2)) for _ in range(96)]
    answers = []
    for source, target in queries:
        expected = finite.covered(source, target)
        assert proof.covered(source, target, CountDomain(1, 0, exact=1)) == expected
        answers.append(expected)
    assert any(answers) and not all(answers)
    assert proof.statistics()["query_copies"] == 3
    assert proof.statistics()["topology_nodes"] == 3 * len(parent.graph.nodes)


def test_unknown_call_native_actor_and_parent_event_are_rejected(parent):
    proof = QueryCompletion(parent)
    endpoint = candidates(parent)[0]
    invalid = (
        replace(endpoint, query=2),
        replace(endpoint, query=True),
        replace(endpoint, call="missing"),
        replace(endpoint, site=replace(endpoint.site, task=("missing", endpoint.site.task[1]))),
        replace(endpoint, site=replace(endpoint.site, position=Position(False, 12345))),
        replace(
            endpoint, site=replace(endpoint.site, channel="unprojected_scope_event", action="set")
        ),
    )
    protocol = next(protocol for protocol in parent.protocols if protocol.sources)
    uid, native = parent.access.endpoints[protocol.sources[0]]
    invalid += (
        QuerySite(0, uid, replace(native, channel=protocol.name, action="set", point="end"), FIRST),
    )
    for source in invalid:
        with pytest.raises(CompileError, match="QuerySite|call|actor|endpoint"):
            proof.coverage(source, endpoint)


@pytest.mark.parametrize("copies", (0, 1, True, 2.0, 10**12))
def test_query_copy_count_is_static_and_bounded(parent, copies):
    with pytest.raises(CompileError, match="static query copies|node budget"):
        QueryCompletion(parent, copies=copies)


def test_final_inactive_call_preserves_the_real_retirement_prefix():
    function = N.from_source(
        native_guarded_recurrence.source.replace(
            "N.Pipelined(n, stages=2)",
            'N.Pipelined(n, stages=2, issue_order={"vector": ("update", "initialize")})',
        ),
        disjoint_external=True,
    )
    parent = checked_parent(function)
    completion = parent.completion
    actor, order = next(iter(completion.call_order.items()))
    first, last = order
    assert parent.program.leaves[first].call.stage.data == 1
    assert parent.program.leaves[last].call.stage.data == 0
    output = completion.active_ports[last, ChildPort(actor, "V", "retire", "output"), 0]
    prefix = completion.prefixes[last, actor, "V", "retire", 0]
    assert parent.guarded()._point_guard(output, (1, 0)) is None
    assert parent.guarded()._point_guard(prefix, (1, 0)).contains(1)
    proof, finite = QueryCompletion(parent), FiniteQueries(parent, 1)
    sources = [
        QuerySite(0, uid, site, LAST)
        for uid, site in sites(parent)
        if uid == first and site.point == "end" and not site.action
    ]
    targets = [
        QuerySite(1, uid, site, FIRST)
        for uid, site in sites(parent)
        if uid == last and site.point == "prefix" and not site.action
    ]
    positives = 0
    for source in sources:
        for target in targets:
            expected = finite.covered(source, target)
            assert proof.covered(source, target, CountDomain(1, 0, exact=1)) == expected
            positives += expected
    assert positives


def test_large_symbolic_counts_do_not_expand_either_clock(parent, monkeypatch):
    def no_expansion(*args, **kwargs):
        pytest.fail("query completion must not instantiate parent or query clocks")

    monkeypatch.setattr(Position, "evaluate", no_expansion)
    monkeypatch.setattr(parent.graph, "domain", no_expansion)
    proof = QueryCompletion(parent)
    endpoint = candidates(parent)[0]
    proof.coverage(
        endpoint, replace(endpoint, query=1), CountDomain(1, 0, minimum=1, maximum=10**12)
    )
    statistics = proof.statistics()
    assert statistics["query_copies"] == 2
    assert statistics["topology_nodes"] == 2 * len(parent.graph.nodes)
    assert statistics["expanded_query_instances"] == 0


def test_query_completion_limits_are_explicit(parent, monkeypatch):
    endpoint = candidates(parent)[0]
    proof = QueryCompletion(parent)
    monkeypatch.setattr("xdsl_npu.scope_query_completion.MAX_QUERY_CASES", 0)
    with pytest.raises(CompileError, match="query budget"):
        proof.coverage(endpoint, endpoint)
    monkeypatch.setattr("xdsl_npu.scope_query_completion.MAX_QUERY_NODES", 0)
    with pytest.raises(CompileError, match="node budget"):
        QueryCompletion(parent)

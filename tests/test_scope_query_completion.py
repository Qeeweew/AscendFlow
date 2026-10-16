"""Two-copy topology units; these original fixtures have no serial query IR.

The finite oracle instantiates only small test counts. Production queries keep
one symbolic KV N and never instantiate the serial query count or KV extent.
"""

from collections import defaultdict
from copy import copy
from dataclasses import replace
from random import Random
from types import SimpleNamespace

import pytest

from tests.fixtures.guarded import native_guarded_recurrence
from tests.fixtures.nested import nested_cv
from tests.models.finite_query_native import ExpandedNativeBodies, input_ports, native_sites
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.child_completion import ChildPort
from xdsl_npu.completion import CountDomain, Position, Site
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import CallOp
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_query_completion import QueryCompletion, QueryEvent, QuerySite
from xdsl_npu.scope_sync import ParentCompletion, ScopeProtocol, _scope_period, verify_scopes

FIRST, LAST = Position(False, 0), Position(True, 0)


def checked_parent(function):
    artifact = compile(function, timeout_s=40)
    _, checked = verify_scopes(artifact.plan, _details=True)
    return completion_from_checked(checked)


def completion_from_checked(checked, protocols=None):
    protocols = tuple(checked.channels.values()) if protocols is None else tuple(protocols)
    proof = ParentCompletion(
        checked.program,
        checked.analyses,
        checked.access,
        _scope_period(checked.access, protocols),
        extra_protocols=protocols,
    )
    for protocol in protocols:
        proof.insert(protocol)
    return proof


@pytest.fixture(
    scope="module", params=(native_guarded_recurrence, nested_cv), ids=("guarded", "mixed")
)
def checked_scopes(request):
    artifact = compile(request.param, timeout_s=40)
    return verify_scopes(artifact.plan, _details=True)[1]


@pytest.fixture(scope="module")
def parent(checked_scopes):
    return completion_from_checked(checked_scopes)


def sites(parent):
    return native_sites(parent)


def inputs(projection):
    return input_ports(projection)


class FiniteQueries:
    """Expand actual native bodies, Graph arcs and CallOp slots independently."""

    def __init__(self, parent, count, *, copies=2):
        self.parent, self.count = parent, count
        self.following = defaultdict(set)
        self.nodes = set()
        self.closures = {}
        self.native = ExpandedNativeBodies(parent, count, copies)
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
                if self.native.native_macro(graph.nodes[source], graph.nodes[target]):
                    continue
                shift = cycles * period + graph.nodes[target].residue - graph.nodes[source].residue
                source_meta = graph.nodes[source]
                for generation in range(source_meta.lower, count - source_meta.upper):
                    self.edge((query, source, generation), (query, target, generation + shift))
        self.parent_nodes = frozenset(self.nodes)
        self.native.attach(self)

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

        # Actual Set effects sharing an exclusive cross counter keep their
        # actor's physical order. Find extrema in the expanded execution,
        # independently of the production count-residue boundary formulas.
        for protocol in parent.protocols:
            if protocol.kind != "cross":
                continue
            for actor, _ in protocol.sources:
                command = parent.events[actor, protocol.name, "set"]
                effects = sorted(
                    (generation + command.task.stage, generation, command.index, node)
                    for query, node, generation in self.parent_nodes
                    if query == 0
                    and graph.nodes[node].command is command
                    and graph.nodes[node].point == "end"
                )
                if effects:
                    _, head, _, first = effects[0]
                    _, tail, _, last = effects[-1]
                    for query in range(copies - 1):
                        self.edge((query, last, tail), (query + 1, first, head))

    def edge(self, source, target):
        if source in self.nodes and target in self.nodes:
            self.following[source].add(target)

    def endpoint(self, endpoint):
        generation = endpoint.parent.evaluate(self.count)
        if isinstance(endpoint, QueryEvent):
            command = self.parent.events[endpoint.actor, endpoint.channel, endpoint.action]
            node = self.parent.graph.ids.get(
                (
                    command.index,
                    generation % self.parent.period,
                    "end" if endpoint.action == "set" else "begin",
                )
            )
            point = endpoint.query, node, generation
            return (point,) if point in self.nodes else ()
        key = endpoint.call, endpoint.site, generation % self.parent.period
        completion = self.parent.completion
        if self.native.cases:
            return self.native.endpoint(endpoint)
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


def event_candidates(parent, *, query=0):
    return [
        QueryEvent(query, actor, protocol.name, action, position)
        for protocol in parent.protocols
        if protocol.kind in {"local", "cross"}
        for owners, action in ((protocol.sources, "set"), (protocol.destinations, "wait"))
        for actor, _ in owners
        for position in (FIRST, Position(False, 1), LAST)
    ]


def independent_parent(parent):
    # The completion graph is immutable in these metadata-corruption tests.
    # Give every mutation its own proof metadata and guarded adapter caches.
    result = copy(parent)
    result.protocols, result.events = list(parent.protocols), dict(parent.events)
    result._guarded = None
    return result


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


def test_actual_native_candidates_survive_empty_materialized_command_tables(parent):
    assert parent.completion.transfer_state is not None
    changed = independent_parent(parent)
    changed.completion = replace(parent.completion, commands={}, case_commands={})
    assert sites(changed) == sites(parent)
    known = set(sites(changed))
    assert all(
        (uid, replace(site, point="begin")) in known
        for uid, site in parent.access.endpoints.values()
    )


def test_finite_oracle_retains_the_original_graph_fixture_path():
    begin = Site(("aiv0", "native"), FIRST, "begin", region="child")
    end = replace(begin, point="end")
    command = SimpleNamespace(index=0)
    nodes = [
        SimpleNamespace(lower=0, upper=0, residue=0, last_gap=None, command=command, point=point)
        for point in ("begin", "end")
    ]
    graph = SimpleNamespace(nodes=nodes, edges={(0, 1, 0)})
    completion = SimpleNamespace(
        transfer_state=None,
        commands={("child", begin): command, ("child", end): command},
        case_commands={},
        case_sites={},
        sites={("child", begin, 0): 0, ("child", end, 0): 1},
    )
    parent = SimpleNamespace(
        graph=graph,
        completion=completion,
        period=1,
        node_activations={},
        protocols=(),
        program=SimpleNamespace(parents={}),
    )
    finite = FiniteQueries(parent, 3)
    first, last = QuerySite(0, "child", begin, FIRST), QuerySite(0, "child", end, FIRST)
    assert set(sites(parent)) == {("child", begin), ("child", end)}
    assert not finite.native.cases
    assert finite.covered(first, last) and not finite.covered(last, first)
    assert not finite.covered(first, replace(last, query=1))


def test_finite_oracle_uses_actual_native_body_without_production_transfer_queries(
    parent, monkeypatch
):
    state = parent.completion.transfer_state
    assert state is not None

    def forbidden(*args, **kwargs):
        pytest.fail("finite native oracle called production transfer reachability")

    for _, value in state.calls.values():
        transfers = (
            tuple(case.transfer for case in value.cases) if hasattr(value, "cases") else (value,)
        )
        for transfer in transfers:
            monkeypatch.setattr(transfer, "covered", forbidden)
            for method in ("covered", "reachable", "predecessors"):
                monkeypatch.setattr(transfer.reach, method, forbidden)
    monkeypatch.setattr(state, "routes", forbidden)
    finite = FiniteQueries(parent, 2)
    assert finite.native.nodes
    available = [point for point in candidates(parent) if finite.endpoint(point)]
    assert available and any(point.site.action for point in available)
    assert all(finite.covered(point, point) for point in available)


def distinct_native_ports(finite):
    completion = finite.parent.completion
    for case in finite.native.cases:
        body = finite.native.bodies[id(case.transfer.view)]
        ports = []
        for local, (node, (_, inner)) in body.view.port_bindings.items():
            port = replace(local, actor=case.transfer.actor)
            key = (case.uid, port, 0) if case.label is None else (case.uid, case.label, port, 0)
            table = completion.active_ports if case.label is None else completion.case_ports
            parent_node = table[key]
            actual = finite.native.point(0, case.uid, case.label, (node, inner), 0)
            if actual in finite.native.nodes:
                ports.append((key, parent_node, (node, inner)))
        for first in ports:
            for last in ports:
                if first[1] != last[1] and not body.reaches(first[2], last[2]):
                    return case, first, last
    raise AssertionError("finite oracle has no distinct native port pair")


def test_finite_oracle_replaces_same_call_macros_with_native_arcs(parent, monkeypatch):
    original = FiniteQueries(parent, 1)
    _, first, last = distinct_native_ports(original)
    # A false projected relation must not become an oracle premise.
    monkeypatch.setattr(parent.graph, "edges", {*parent.graph.edges, (first[1], last[1], 0)})
    changed = FiniteQueries(parent, 1)
    assert dict(changed.following) == dict(original.following)


def test_finite_oracle_independently_rejects_false_native_component_aliases(parent):
    finite = FiniteQueries(parent, 1)
    case, first, last = distinct_native_ports(finite)
    completion = parent.completion
    field = "active_ports" if case.label is None else "case_ports"
    changed_ports = dict(getattr(completion, field))
    changed_ports[last[0]] = first[1]
    changed = independent_parent(parent)
    changed.completion = replace(completion, **{field: changed_ports})
    with pytest.raises(AssertionError, match="merged distinct native points"):
        FiniteQueries(changed, 1)


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
        if first.command.kind == "cross_set":
            assert first.command is last.command
            assert first.point == last.point == "end"
            continue
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
            for source, target, _, point in cross
            if parent.graph.nodes[source].command.kind != "cross_set"
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
    known = set(sites(parent))
    for uid, _, send, _ in parent.resource_pairs:
        native = replace(send, point="end", channel="", action="")
        if (uid, native) in known:
            assert set(parent.completion.commands_for(uid, send)).isdisjoint(
                parent.completion.commands_for(uid, native)
            )


def test_native_completion_cannot_replace_old_wait_consumption(parent, finite_oracle):
    proof, finite = QueryCompletion(parent), finite_oracle(1)
    differences = 0
    known = set(sites(parent))
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
    known = set(sites(parent))
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
    endpoints = candidates(parent)[0], event_candidates(parent)[0]
    # Resolving a typed Site uses its actual fixed child count. Bind that
    # constant inner position before prohibiting evaluation of either runtime
    # clock; the query proof below must remain entirely symbolic in parent N.
    parent.completion.commands_for(endpoints[0].call, endpoints[0].site)

    def no_expansion(*args, **kwargs):
        pytest.fail("query completion must not instantiate parent or query clocks")

    monkeypatch.setattr(Position, "evaluate", no_expansion)
    monkeypatch.setattr(parent.graph, "domain", no_expansion)
    proof = QueryCompletion(parent)
    proof.coverage_many(
        (endpoint, replace(endpoint, query=1), CountDomain(1, 0, minimum=1, maximum=10**12))
        for endpoint in endpoints
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


@pytest.mark.parametrize("copies", (2, 3))
@pytest.mark.parametrize("count", (1, 2, 3, 7))
def test_real_parent_events_and_native_endpoints_match_finite_execution(
    parent, finite_oracle, copies, count
):
    proof, finite = QueryCompletion(parent, copies=copies), finite_oracle(count, copies=copies)
    native, events = candidates(parent), event_candidates(parent)
    assert events
    rng = Random(1639 + 31 * count + copies)
    requests = [
        (
            replace(rng.choice(left), query=rng.randrange(copies)),
            replace(rng.choice(right), query=rng.randrange(copies)),
            CountDomain(1, 0, exact=count),
        )
        for left, right in ((native, events), (events, native), (events, events))
        for _ in range(32)
    ]
    requests.extend((endpoint, endpoint, CountDomain(1, 0, exact=count)) for endpoint in events)
    expected = [finite.covered(source, target) for source, target, _ in requests]
    assert proof.covered_many(iter(requests)) == expected
    assert not all(expected)
    if any(finite.endpoint(endpoint) for endpoint in events):
        assert any(expected)


def test_parent_event_identity_uses_actual_set_effect_and_wait_consumption(parent):
    proof = QueryCompletion(parent)
    for endpoint in event_candidates(parent):
        command = parent.events[endpoint.actor, endpoint.channel, endpoint.action]
        expected_point = "end" if endpoint.action == "set" else "begin"
        for node, _, _ in proof._bindings(endpoint):
            actual = parent.graph.nodes[node % proof.size]
            assert actual.command is command
            assert actual.point == expected_point


def test_parent_event_unknown_action_actor_channel_and_barrier_are_rejected(parent):
    proof, endpoint = QueryCompletion(parent), event_candidates(parent)[0]
    invalid = (
        replace(endpoint, query=-1),
        replace(endpoint, actor="missing"),
        replace(endpoint, channel="missing"),
        replace(endpoint, action="barrier"),
        replace(endpoint, action="end"),
        replace(endpoint, action=[]),
        replace(endpoint, parent=Position(True, 1.0)),
    )
    for protocol in parent.protocols:
        actors = {actor for actor, _ in protocol.sources + protocol.destinations}
        absent = set(parent.access.loop.actors) - actors
        invalid += tuple(
            replace(endpoint, actor=actor, channel=protocol.name) for actor in sorted(absent)
        )
        if protocol.kind in {"local", "cross"}:
            for owners, action in ((protocol.sources, "set"), (protocol.destinations, "wait")):
                invalid += tuple(
                    QueryEvent(0, actor, protocol.name, action, FIRST)
                    for actor in sorted(actors - {actor for actor, _ in owners})
                )
        if protocol.kind == "barrier":
            invalid += (QueryEvent(0, protocol.destinations[0][0], protocol.name, "wait", FIRST),)
    for bad in invalid:
        with pytest.raises(CompileError, match="QueryEvent|event|participant"):
            proof.coverage(bad, endpoint)


def test_parent_event_missing_or_mismatched_actual_command_is_rejected(parent):
    endpoint = event_candidates(parent)[0]
    key = endpoint.actor, endpoint.channel, endpoint.action
    command = parent.events[key]
    for replacement in (
        None,
        replace(command, channel="different"),
        replace(command, kind="task"),
        replace(command, actor="missing"),
        replace(command, task=replace(command.task, uid="different")),
        replace(command, lower=command.lower + 1),
    ):
        changed = independent_parent(parent)
        if replacement is None:
            changed.events.pop(key)
        else:
            changed.events[key] = replacement
        with pytest.raises(CompileError, match="actual command"):
            QueryCompletion(changed).coverage(endpoint, endpoint)


def test_cross_counter_fifo_carries_actual_last_to_first_set_effect(parent, finite_oracle):
    proof = QueryCompletion(parent, copies=3)
    crosses = [p for p in parent.protocols if p.kind == "cross"]
    assert bool(proof.statistics()["cross_counter_fifo_edges"]) == bool(crosses)
    for count in (1, 2, 3, 7):
        finite = finite_oracle(count, copies=3)
        for protocol in crosses:
            for actor, _ in protocol.sources:
                command = parent.events[actor, protocol.name, "set"]
                generations = sorted(
                    generation
                    for query, node, generation in finite.parent_nodes
                    if query == 0
                    and parent.graph.nodes[node].command is command
                    and parent.graph.nodes[node].point == "end"
                )
                if not generations:
                    continue
                source = QueryEvent(
                    0, actor, protocol.name, "set", Position(True, count - 1 - generations[-1])
                )
                target = QueryEvent(2, actor, protocol.name, "set", Position(False, generations[0]))
                assert finite.covered(source, target)
                assert proof.covered(source, target, CountDomain(1, 0, exact=count))


def test_periodic_cross_fifo_keeps_residues_empty_ranges_and_signed_endpoints(checked_scopes):
    original = next((p for p in checked_scopes.channels.values() if p.kind == "cross"), None)
    if original is None:
        return
    for period, residues, distance in ((2, (1,), 0), (3, (0, 2), 2)):
        protocol = replace(original, period=period, residues=residues, distance=distance)
        parent = completion_from_checked(checked_scopes, (protocol,))
        proof = QueryCompletion(parent, copies=3)
        endpoints = [
            QueryEvent(query, actor, protocol.name, action, position)
            for query in range(3)
            for owners, action in ((protocol.sources, "set"), (protocol.destinations, "wait"))
            for actor, _ in owners
            for position in (
                FIRST,
                Position(False, 1),
                LAST,
                Position(True, 1),
                Position(False, -1),
                Position(True, -1),
            )
        ]
        rng = Random(1806 + period)
        pairs = [(rng.choice(endpoints), rng.choice(endpoints)) for _ in range(96)]
        pairs.extend((endpoint, endpoint) for endpoint in endpoints)
        domain = CountDomain(1, 0, minimum=0, maximum=7)
        actual = proof.coverage_many((source, target, domain) for source, target in pairs)
        for count in (0, 1, 2, 3, 7):
            finite = FiniteQueries(parent, count, copies=3)
            for (source, target), result in zip(pairs, actual, strict=True):
                assert any(piece.contains(count) for piece in result) == finite.covered(
                    source, target
                ), (period, residues, distance, count, source, target)
        for count in (1, 2, 3, 7):
            finite = FiniteQueries(parent, count, copies=3)
            for actor, _ in protocol.sources:
                command = parent.events[actor, protocol.name, "set"]
                generations = sorted(
                    generation
                    for query, node, generation in finite.parent_nodes
                    if query == 0
                    and parent.graph.nodes[node].command is command
                    and parent.graph.nodes[node].point == "end"
                )
                if generations:
                    source = QueryEvent(
                        0, actor, protocol.name, "set", Position(False, generations[-1])
                    )
                    target = QueryEvent(
                        2, actor, protocol.name, "set", Position(False, generations[0])
                    )
                    assert finite.covered(source, target)
                    assert proof.covered(source, target, CountDomain(1, 0, exact=count))


def test_cross_set_effect_does_not_complete_native_retirement(parent, finite_oracle):
    crosses = [p for p in parent.protocols if p.kind == "cross"]
    if not crosses:
        return
    proof, finite = QueryCompletion(parent), finite_oracle(1)
    differences = 0
    known = set(sites(parent))
    for protocol in crosses:
        for actor, owner in protocol.sources:
            uid, native = parent.access.endpoints[actor, owner]
            native = replace(native, point="end")
            if (uid, native) not in known:
                continue
            source = QueryEvent(0, actor, protocol.name, "set", LAST)
            native_end = QuerySite(0, uid, native, LAST)
            if not finite.endpoint(source):
                continue
            for target in candidates(parent, query=1):
                if target.site.point != "retire" or target.parent != FIRST:
                    continue
                if finite.covered(native_end, target) and not finite.covered(source, target):
                    assert proof.covered(native_end, target, CountDomain(1, 0, exact=1))
                    assert not proof.covered(source, target, CountDomain(1, 0, exact=1))
                    differences += 1
    assert differences


def test_local_parent_wait_consumption_cannot_be_replaced_by_native_end():
    source = N.from_source(
        """
def parent_local(
    O: N.Tensor((n, 2, 8), N.f32, access="write"), n: N.int64,
):
    with N.Kernel():
        state = N.alloc_ub((2, 2, 8), N.f32)
        for t in N.Pipelined(n, stages=2):
            with N.Stage(0, name="produce"):
                for j in N.Pipelined(2, stages=1):
                    N.fill(state[t % 2, j, :], 1.0)
            with N.Stage(1, name="consume"):
                for j in N.Pipelined(2, stages=1):
                    N.copy(state[t % 2, j, :], O[t, j, :])
""",
        disjoint_external=True,
    )
    parent = checked_parent(source)
    proof, finite = QueryCompletion(parent), FiniteQueries(parent, 1)
    differences = 0
    known = set(sites(parent))
    for protocol in parent.protocols:
        if protocol.kind != "local":
            continue
        actor, owner = protocol.sources[0]
        uid, native = parent.access.endpoints[actor, owner]
        native = replace(native, point="end")
        if (uid, native) not in known:
            continue
        consumed = QueryEvent(0, actor, protocol.name, "wait", LAST)
        native_end = QuerySite(0, uid, native, LAST)
        republished = QueryEvent(1, actor, protocol.name, "set", FIRST)
        if finite.covered(native_end, republished) and not finite.covered(consumed, republished):
            assert proof.covered(native_end, republished, CountDomain(1, 0, exact=1))
            assert not proof.covered(consumed, republished, CountDomain(1, 0, exact=1))
            differences += 1
    assert differences


def test_activated_parent_tail_event_keeps_native_cases_and_actual_consumption(checked_scopes):
    if not any(leaf.cases for leaf in checked_scopes.program.leaves.values()):
        return
    selected = {}
    for owner, (uid, site) in checked_scopes.access.endpoints.items():
        leaf = checked_scopes.program.leaves[uid]
        natives = {
            task.uid: task for data in leaf.loop.actors.values() for task in data.program.tasks
        }
        task = natives[site.task[1]]
        if site.position == FIRST and (task.native == "convert_rn" or task.pipe == "MTE3"):
            selected["set" if task.native == "convert_rn" else "wait"] = owner
    assert set(selected) == {"set", "wait"}
    protocol = ScopeProtocol(
        "actual_tail_parent",
        "local",
        (selected["set"],),
        (selected["wait"],),
        0,
        1,
        (0,),
        (7,),
        activation=Activation(CountDomain(1, 0, minimum=1), LAST, LAST),
    )
    parent = completion_from_checked(checked_scopes, (protocol,))
    proof = QueryCompletion(parent)
    actor = selected["set"][0]
    endpoints = [
        QueryEvent(query, actor, protocol.name, action, position)
        for query in range(2)
        for action in ("set", "wait")
        for position in (FIRST, LAST, Position(False, 1), Position(True, 1))
    ]
    pairs = [(source, target) for source in endpoints for target in endpoints]
    domain = CountDomain(1, 0, minimum=0, maximum=7)
    actual = proof.coverage_many((source, target, domain) for source, target in pairs)
    for count in (0, 1, 2, 3, 7):
        finite = FiniteQueries(parent, count)
        for (source, target), result in zip(pairs, actual, strict=True):
            assert any(piece.contains(count) for piece in result) == finite.covered(source, target)
    uid, native = parent.access.endpoints[selected["set"]]
    native_end = QuerySite(0, uid, replace(native, point="end"), LAST)
    old_wait = QueryEvent(0, actor, protocol.name, "wait", LAST)
    next_set = QueryEvent(1, actor, protocol.name, "set", FIRST)
    finite = FiniteQueries(parent, 1)
    assert finite.covered(native_end, next_set)
    assert not finite.covered(old_wait, next_set)
    assert proof.covered(native_end, next_set, CountDomain(1, 0, exact=1))
    assert not proof.covered(old_wait, next_set, CountDomain(1, 0, exact=1))


def test_unallocated_overlapping_and_activated_cross_counters_are_rejected(parent):
    crosses = [p for p in parent.protocols if p.kind == "cross"]
    if not crosses:
        return
    first = crosses[0]
    replacements = [replace(first, ids=ids) for ids in ((), (8,), (True,), (0, 1))] + [
        replace(first, activation=Activation(CountDomain(1, 0, minimum=1), FIRST, LAST)),
        replace(first, last=True),
        replace(first, carry=True),
    ]
    for replacement in replacements:
        changed = independent_parent(parent)
        changed.protocols[changed.protocols.index(first)] = replacement
        with pytest.raises(CompileError, match="allocated exclusive ID|activation/case"):
            QueryCompletion(changed)
    assert len(crosses) > 1
    changed = independent_parent(parent)
    changed.protocols[changed.protocols.index(crosses[1])] = replace(crosses[1], ids=first.ids)
    with pytest.raises(CompileError, match="allocated exclusive ID"):
        QueryCompletion(changed)


def test_batch_registers_all_origins_before_one_propagation(parent, finite_oracle, monkeypatch):
    proof, finite = QueryCompletion(parent), finite_oracle(1)
    endpoints = [
        endpoint
        for endpoint in event_candidates(parent) + candidates(parent)
        if endpoint.parent == FIRST and finite.endpoint(endpoint)
    ][:8]
    assert len(endpoints) >= 2
    requests = [(endpoint, endpoint, CountDomain(1, 0, exact=1)) for endpoint in endpoints]
    expected_origins = {
        (node, point)
        for endpoint in endpoints
        for node, point, guard in proof._bindings(endpoint)
        if guard.contains(1)
    }
    calls, registered = [], []
    seed, propagate = proof.flow.seed, proof.flow.propagate

    def record_seed(*args):
        # Propagation itself may seed anchored target facts. Record only
        # public source registration before entering that single pass.
        if "propagate" not in calls:
            calls.append("seed")
            registered.append(args)
        return seed(*args)

    def record_propagate():
        assert set(proof.origins) == expected_origins
        calls.append("propagate")
        return propagate()

    monkeypatch.setattr(proof.flow, "seed", record_seed)
    monkeypatch.setattr(proof.flow, "propagate", record_propagate)
    assert proof.covered_many(iter(requests + requests[::-1])) == [True] * (2 * len(requests))
    # Each virtual origin may seed several physical interface vertices, or
    # stay wholly within its native body. Every logical origin must still be
    # registered before the sole propagation pass, with no repeated seed.
    assert calls == ["seed"] * len(registered) + ["propagate"]
    assert len(set(registered)) == len(registered)
    assert len(registered) >= sum(isinstance(node, int) for node, _ in expected_origins)
    assert proof.statistics()["queries"] == len(requests)


def test_batch_order_duplicates_domains_and_endpoint_existence(parent, finite_oracle):
    available = event_candidates(parent) + candidates(parent)
    requests = [
        (endpoint, replace(endpoint, query=1), domain)
        for endpoint in available[:: max(1, len(available) // 9)]
        for domain in (CountDomain(1, 0, exact=0), CountDomain(2, 1, minimum=1, maximum=7))
    ]
    requests += requests[::-1]
    proof, scalar = QueryCompletion(parent), QueryCompletion(parent)
    actual = proof.coverage_many(iter(requests))
    assert actual == [scalar.coverage(*request) for request in requests]
    assert proof.covered_many(requests) == [scalar.covered(*request) for request in requests]
    for (source, target, domain), result in zip(requests, actual, strict=True):
        for count in (0, 1, 2, 3, 5, 7, 8):
            expected = domain.contains(count) and finite_oracle(count).covered(source, target)
            assert any(piece.contains(count) for piece in result) == expected
    assert proof.coverage_many(()) == []
    assert proof.covered_many(()) == []


def test_batch_checks_unknown_endpoints_even_for_empty_required_domain(parent):
    proof, endpoint = QueryCompletion(parent), candidates(parent)[0]
    unknown = replace(endpoint, call="missing")
    for domain in (CountDomain(1, 0, exact=0), CountDomain(1, 0, exact=3)):
        with pytest.raises(CompileError, match="unknown query completion call"):
            proof.coverage_many(((endpoint, endpoint, domain), (unknown, endpoint, domain)))


def test_parent_protocol_metadata_changes_invalidate_batch_snapshot(parent):
    changed = independent_parent(parent)
    proof, endpoint = QueryCompletion(changed), candidates(changed)[0]
    changed.protocols[0] = replace(changed.protocols[0], name="changed_snapshot")
    with pytest.raises(CompileError, match="snapshot changed"):
        proof.covered_many(((endpoint, endpoint, CountDomain(1, 0, exact=1)),))

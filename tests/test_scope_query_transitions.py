"""Actual query Set/Wait transitions against an independent finite event graph."""

from dataclasses import dataclass, replace
from itertools import product

import pytest

from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_scope_query_completion import FiniteQueries
from tests.test_scope_query_planning import query_state_store
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
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
from xdsl_npu.scope_sync import _plan_scope_parent, _scope_access, _verify_scope_parent

FIRST, LAST = Position(False, 0), Position(True, 0)
POSITIVE = CountDomain(1, 0, minimum=1)
COUNTS = (1, 2, 3, 7)


def native(checked, endpoint):
    return next(
        task
        for data in checked.program.leaves[endpoint.call].loop.actors.values()
        for task in data.program.tasks
        if task.uid == endpoint.site.task[1]
    )


class FiniteQueryTransitions(FiniteQueries):
    """Add literal event instructions, without native_order or query.topology.

    The inherited graph expands actual parent/native instructions. New event
    edges below implement issue, FIFO begin, retirement capture and consumption
    directly. Integer KV positions decide whether two conditional insertions
    are the same instruction; no production affine equality helper is used.
    """

    def __init__(self, parent, count, protocols, *, copies=4, outer_count=None):
        super().__init__(parent, count, copies=copies)
        self.protocols = {protocol.name: protocol for protocol in protocols}
        self.events, self.event_begins, self.matches = {}, {}, set()
        outer_count = (
            copies + max((p.target.query for p in protocols), default=0)
            if outer_count is None
            else outer_count
        )
        tasks = {
            owner: task
            for actor, data in parent.access.loop.actors.items()
            for task in data.program.tasks
            for owner in ((actor, task.uid),)
        }
        owners = {
            (uid, replace(site, point="begin")): owner
            for owner, (uid, site) in parent.access.endpoints.items()
        }

        def native_points(endpoint, point):
            return super(FiniteQueryTransitions, self).endpoint(
                replace(endpoint, site=replace(endpoint.site, point=point))
            )

        def parent_points(owner, generation, action, kind):
            result = []
            for protocol in parent.protocols:
                if action == "set":
                    if owner not in protocol.sources or kind != "queue":
                        continue
                    command_action = "set"
                else:
                    if owner not in protocol.destinations:
                        continue
                    command_action = "barrier" if protocol.kind == "barrier" else "wait"
                    if kind == "issue" and protocol.kind != "cross" and tasks[owner].pipe != "S":
                        continue
                command = parent.events[owner[0], protocol.name, command_action]
                node = parent.graph.ids.get((command.index, generation % parent.period, "begin"))
                result.append(node)
            return result

        def connect(sources, target):
            for source in sources:
                self.edge(source, target)

        # Preserve actual declaration/lexical order, but only among events at
        # the same concrete parent generation and native instruction.
        prior = {}
        for protocol in protocols:
            if not protocol.domain.contains(count):
                continue
            distance = protocol.target.query
            for index in range(min(copies, outer_count)):
                for action, endpoint in (("set", protocol.source), ("wait", protocol.target)):
                    if action == "set" and index + distance >= outer_count:
                        continue
                    if action == "wait" and index < distance:
                        continue
                    endpoint = replace(endpoint, query=index)
                    generation = endpoint.parent.evaluate(count)
                    assert native_points(endpoint, "begin"), "query event lost its actual native"
                    owner = owners[endpoint.call, replace(endpoint.site, point="begin")]
                    task = tasks[owner]
                    before_issue = native_points(
                        endpoint,
                        ("end" if task.pipe == "S" else "admit_issue")
                        if action == "set"
                        else "issue",
                    )
                    before_queue = native_points(endpoint, "begin" if action == "set" else "queue")
                    key = index, owner, action, generation
                    before_queue += tuple(prior.get(key, ()))
                    if action == "wait" and task.pipe == "S":
                        before_issue += tuple(prior.get(key, ()))
                    for kind, bindings in (("issue", before_issue), ("queue", before_queue)):
                        extra = tuple(
                            point
                            for node in parent_points(owner, generation, action, kind)
                            if (point := (index, node, generation)) in self.nodes
                        )
                        if kind == "issue":
                            before_issue += extra
                        else:
                            before_queue += extra
                    begin = ("query", index, protocol.name, action, "begin", generation)
                    end = ("query", index, protocol.name, action, "end", generation)
                    self.nodes.add(begin)
                    self.nodes.add(end if action == "set" else begin)
                    connect(before_issue, begin)
                    connect(before_queue, begin)
                    if action == "set":
                        self.edge(begin, end)
                        connect(native_points(endpoint, "prefix"), end)
                        for target in native_points(endpoint, "after_queue"):
                            self.edge(begin, target)
                    else:
                        end = begin  # The exposed Wait point is consumption.
                        for target in native_points(endpoint, "admit_queue"):
                            self.edge(begin, target)
                        if task.pipe == "S":
                            for target in native_points(endpoint, "admit_issue"):
                                self.edge(begin, target)
                    event_key = index, owner[0], protocol.name, action
                    self.events[event_key] = end
                    self.event_begins[event_key] = begin
                    prior.setdefault(key, []).append(begin)
            for index in range(copies - distance):
                source = self.events.get(
                    (index, protocol.source.site.task[0], protocol.name, "set")
                )
                target = self.events.get(
                    (index + distance, protocol.target.site.task[0], protocol.name, "wait")
                )
                if source is not None and target is not None:
                    self.edge(source, target)
                    self.matches.add((protocol.name, index, index + distance))

    def endpoint(self, endpoint):
        if isinstance(endpoint, QueryEvent) and endpoint.channel in self.protocols:
            point = self.events.get(
                (endpoint.query, endpoint.actor, endpoint.channel, endpoint.action)
            )
            return (point,) if point is not None else ()
        return super().endpoint(endpoint)


@dataclass
class Actual:
    checked: object
    parent: object
    protocols: tuple[QueryProtocol, ...]
    query: QueryCompletion
    source: QuerySite
    target: QuerySite
    finite: dict

    def oracle(self, count):
        if count not in self.finite:
            self.finite[count] = FiniteQueryTransitions(self.parent, count, self.protocols)
        return self.finite[count]

    def event(self, index, action, protocol=0):
        protocol = self.protocols[protocol]
        endpoint = protocol.source if action == "set" else protocol.target
        return QueryEvent(index, endpoint.site.task[0], protocol.name, action, endpoint.parent)


@pytest.fixture(scope="module")
def base():
    with limit(40):
        module = lower_source(scoped_query_recurrence)
        _plan_scope_parent(extract_scopes(module))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        check = next(
            check
            for check in prepare_memory_checks(checked, parent)
            if check.source.site.task[0] == "aiv0"
            and check.source.parent == LAST
            and check.target.parent == FIRST
            and native(checked, check.source).pipe == "MTE3"
            and native(checked, check.target).native == "fill"
        )
    return module, checked, parent, check.source, replace(check.target, query=1)


@pytest.fixture(scope="module")
def actual(base):
    cache = {}

    def build(kind):
        if kind in cache:
            return cache[kind]
        module, checked, _, source, target = base
        if kind == "same_native":
            task = next(
                (uid, site)
                for uid, site in checked.access.endpoints.values()
                if site.task[0] == source.site.task[0]
                and native(checked, QuerySite(0, uid, site, FIRST)).native == "add"
            )
            target = QuerySite(1, task[0], replace(task[1], point="begin"), LAST)
            protocols = (
                QueryProtocol("tail_acquire", source, target, POSITIVE, (6,)),
                QueryProtocol(
                    "head_acquire", source, replace(target, parent=FIRST), POSITIVE, (7,)
                ),
            )
        else:
            domain = CountDomain(2, 1, minimum=1) if kind == "odd" else POSITIVE
            distance = 2 if kind == "distance_two" else 1
            target = replace(target, query=distance)
            protocols = (QueryProtocol("query_release", source, target, domain, (7,)),)
        with limit(40):
            program = extract_scopes(module.clone())
            install_query_protocols(program, _scope_access(program), protocols)
            _, checked, parent = _verify_scope_parent(extract_scopes(program.module))
            assert read_query_protocols(checked.program, checked.access) == protocols
            query = QueryCompletion(parent, copies=4)
        result = Actual(checked, parent, protocols, query, source, target, {})
        cache[kind] = result
        return result

    return build


def endpoints(actual):
    result = []
    for index in range(4):
        result.extend(
            (
                replace(actual.source, query=index),
                replace(
                    actual.source, query=index, site=replace(actual.source.site, point="prefix")
                ),
                replace(actual.target, query=index),
            )
        )
        for protocol in range(len(actual.protocols)):
            result.extend(
                (actual.event(index, "set", protocol), actual.event(index, "wait", protocol))
            )
    return tuple(dict.fromkeys(result))


@pytest.mark.parametrize("kind", ("one", "odd", "distance_two", "same_native"))
def test_actual_query_transitions_equal_literal_finite_event_graph(actual, kind):
    actual = actual(kind)
    pairs = tuple(product(endpoints(actual), repeat=2))
    requests = tuple(
        (source, target, CountDomain(1, 0, exact=count))
        for count in COUNTS
        for source, target in pairs
    )
    with limit(40):
        observed = actual.query.covered_many(requests)
        expected = [
            actual.oracle(count).covered(source, target)
            for count in COUNTS
            for source, target in pairs
        ]
    assert observed == expected, next(
        (request, got, want)
        for request, got, want in zip(requests, observed, expected, strict=True)
        if got != want
    )


def test_real_missing_mte3_release_becomes_covered_and_wait_releases_same_id(base, actual):
    _, _, parent, source, target = base
    installed = actual("one")
    requests = tuple((source, target, CountDomain(1, 0, exact=count)) for count in COUNTS)
    with limit(30):
        assert QueryCompletion(parent, copies=4).covered_many(requests) == [False] * len(COUNTS)
        assert installed.query.covered_many(requests) == [True] * len(COUNTS)
        assert installed.query.covered_many(
            (installed.event(1, "wait"), installed.event(1, "set"), CountDomain(1, 0, exact=count))
            for count in COUNTS
        ) == [True] * len(COUNTS)


def test_set_effect_never_becomes_its_native_retirement_capture(actual):
    actual = actual("one")
    prefix = replace(actual.source, site=replace(actual.source.site, point="prefix"))
    requests = tuple(
        query
        for count in COUNTS
        for query in (
            (actual.source, actual.event(0, "set"), CountDomain(1, 0, exact=count)),
            (actual.event(0, "set"), prefix, CountDomain(1, 0, exact=count)),
        )
    )
    assert actual.query.covered_many(requests) == [True, False] * len(COUNTS)


def test_same_native_wait_order_uses_the_real_first_last_equality_domain(actual):
    actual = actual("same_native")
    requests = tuple(
        (actual.event(1, "wait", 0), actual.event(1, "wait", 1), CountDomain(1, 0, exact=count))
        for count in COUNTS
    )
    # Declaration order is tail then head. They are the same physical native
    # only for N=1; a tail acquire must not order an earlier head at larger N.
    assert actual.query.covered_many(requests) == [True, False, False, False]
    assert actual.query.covered_many(
        (actual.event(0, "set", 0), actual.event(0, "set", 1), CountDomain(1, 0, exact=count))
        for count in COUNTS
    ) == [False] * len(COUNTS)


def test_distance_two_has_no_early_wait_or_implicit_single_id_reuse(actual):
    actual = actual("distance_two")
    with pytest.raises(CompileError, match="proof window omits a protocol consumer"):
        QueryCompletion(actual.parent, copies=2)
    query = QueryCompletion(actual.parent, copies=3)
    requests = tuple(
        relation
        for count in COUNTS
        for relation in (
            (actual.source, replace(actual.target, query=1), CountDomain(1, 0, exact=count)),
            (actual.source, actual.target, CountDomain(1, 0, exact=count)),
            (actual.event(0, "wait"), actual.event(0, "set"), CountDomain(1, 0, exact=count)),
            (actual.event(1, "wait"), actual.event(1, "set"), CountDomain(1, 0, exact=count)),
            (actual.event(2, "wait"), actual.event(1, "set"), CountDomain(1, 0, exact=count)),
            (actual.event(2, "wait"), actual.event(2, "set"), CountDomain(1, 0, exact=count)),
        )
    )
    assert query.covered_many(requests) == [False, True, False, False, False, True] * len(COUNTS)


def test_finite_outer_query_bounds_pair_only_existing_consumers(actual):
    actual = actual("distance_two")
    protocol = actual.protocols[0]
    for total in (0, 1, 2, 3, 4, 7):
        finite = FiniteQueryTransitions(
            actual.parent, 1, actual.protocols, copies=max(2, total), outer_count=total
        )
        assert finite.matches == {
            (protocol.name, source, source + 2) for source in range(max(0, total - 2))
        }
        for index in range(max(2, total)):
            assert bool(finite.endpoint(actual.event(index, "set"))) == (index + 2 < total)
            assert bool(finite.endpoint(actual.event(index, "wait"))) == (2 <= index < total)


@pytest.fixture(scope="module", params=(1, 3))
def fixed_kv(request):
    # Only specialize the parent count; native operations and the algorithm
    # are the same public local-query source exercised by the planner.
    annotation = "count: N.int64(minimum=1)"
    assert annotation in query_state_store.source
    definition = replace(
        query_state_store,
        source=query_state_store.source.replace(
            annotation, f"count: N.constexpr = {request.param}"
        ),
    )
    with limit(30):
        module = lower_source(definition)
        _plan_scope_parent(extract_scopes(module))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        endpoints = tuple(
            QuerySite(0, uid, site, LAST) for uid, site in checked.access.endpoints.values()
        )
        source = next(
            replace(endpoint, site=replace(endpoint.site, point="end"))
            for endpoint in endpoints
            if native(checked, endpoint).pipe == "MTE3" and endpoint.site.position.offset == 0
        )
        target = next(
            replace(endpoint, query=1, site=replace(endpoint.site, point="begin"), parent=FIRST)
            for endpoint in endpoints
            if native(checked, endpoint).native == "fill" and endpoint.site.position.offset == 0
        )
        protocol = QueryProtocol("fixed_release", source, target, POSITIVE, (7,))
        install_query_protocols(checked.program, checked.access, (protocol,))
        query = QueryCompletion(parent)
    return request.param, protocol, query


def test_fixed_parent_limits_a_broad_declared_query_count_domain(fixed_kv):
    count, protocol, query = fixed_kv
    assert query.parent.program.count == count
    assert query.query_protocols[protocol.name].domain == POSITIVE
    assert query.query_protocol_edges > 0
    assert query.covered(protocol.source, protocol.target, POSITIVE)
    coverage = query.coverage(protocol.source, protocol.target, POSITIVE)
    assert all(any(domain.contains(n) for domain in coverage) == (n == count) for n in range(8))
    assert not query.covered(protocol.source, protocol.target, CountDomain(1, 0, exact=count + 1))

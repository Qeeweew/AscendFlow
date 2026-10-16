"""Actual resource boundaries checked against independent finite ordinal words."""

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.fixtures.guarded import native_guarded_recurrence
from tests.fixtures.nested import nested_copy
from tests.test_scope_protocol_review import fifo_proof
from tests.test_scope_query_completion import FiniteQueries
from xdsl_npu import compile
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import dump
from xdsl_npu.model import CompileError
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_completion import QueryCompletion, QuerySite
from xdsl_npu.scope_query_resources import (
    _configuration_checks,
    _parent_boundaries,
    check_binary_resources,
    prepare_binary_resources,
    verify_query_resources,
)
from xdsl_npu.scope_sync import CheckedScopes, ScopeProtocol, _analyses, _verify_scope_parent

FIRST, LAST = Position(False, 0), Position(True, 0)


def descriptor(*, period=1, residues=(0,), ids=(7, 2), distance=0, activation=None):
    return ScopeProtocol(
        "source_word",
        "local",
        (("aiv0", "producer"),),
        (("aiv0", "consumer"),),
        distance,
        period,
        residues,
        ids,
        activation=activation,
    )


def word_parent(protocol):
    # Only the mathematical ID-word extractor is under test here. Integration
    # tests below rebuild full actual parent proofs from verified public IR.
    return SimpleNamespace(
        graph=SimpleNamespace(maximum=None),
        events={
            ("aiv0", protocol.name, action): SimpleNamespace(task=SimpleNamespace(pipe=pipe))
            for action, pipe in (("set", "V"), ("wait", "MTE3"))
        },
    )


def ordinal_id(channel, generation):
    # Count actual preceding source epochs, independently of quotient/rank and
    # production's periodic ID-clock construction.
    ordinal = sum(g % channel.period in channel.residues for g in range(generation + 1)) - 1
    return channel.ids[ordinal % len(channel.ids)]


def source_epochs(channel, count):
    activation = channel.activation
    if activation is None:
        first, last = 0, count - 1 - channel.distance
    else:
        if not activation.count.contains(count):
            return []
        first, last = activation.first.evaluate(count), activation.last.evaluate(count)
    return [
        generation
        for generation in range(count)
        if first <= generation <= last
        and generation % channel.period in channel.residues
        and 0 <= generation + channel.distance < count
    ]


@pytest.mark.parametrize(
    "protocol",
    (
        descriptor(ids=(6, 2, 7), distance=2),
        descriptor(period=3, residues=(0, 2)),
        descriptor(
            period=3,
            residues=(0, 2),
            ids=(7, 2, 5),
            activation=Activation(
                CountDomain(2, 1, minimum=5, maximum=19),
                Position(False, 2),
                Position(True, 2),
                3,
                (0, 2),
            ),
        ),
        descriptor(
            ids=(7, 2, 5),
            activation=Activation(CountDomain(1, 0, minimum=1, maximum=13), LAST, LAST),
        ),
        descriptor(
            period=2,
            residues=(1,),
            distance=-1,
            activation=Activation(CountDomain(1, 0, minimum=3), Position(False, 1), LAST, 2, (1,)),
        ),
        descriptor(period=4, residues=(0, 1, 2, 3), ids=(7, 2, 5)),
    ),
)
def test_parent_boundary_domains_match_independent_finite_source_ordinals(protocol):
    boundaries = tuple(_parent_boundaries(word_parent(protocol), protocol))
    for count in range(21):
        generations = defaultdict(list)
        for generation in source_epochs(protocol, count):
            generations[ordinal_id(protocol, generation)].append(generation)
        expected = {
            event_id: (values[-1] + protocol.distance, values[0])
            for event_id, values in generations.items()
        }
        actual = {}
        for boundary in boundaries:
            if boundary.domain.contains(count):
                assert boundary.event_id not in actual
                actual[boundary.event_id] = (
                    boundary.source.parent.evaluate(count),
                    boundary.target.parent.evaluate(count),
                )
                assert boundary.bank == ("aiv0", "V", "MTE3")
                assert boundary.source.action == "wait" and boundary.target.action == "set"
        assert actual == expected, (count, protocol)


def test_activation_first_does_not_restart_event_ordinal():
    protocol = descriptor(
        ids=(7, 2),
        activation=Activation(CountDomain(1, 0, minimum=2), LAST, LAST),
    )
    boundaries = tuple(_parent_boundaries(word_parent(protocol), protocol))
    assert [b.event_id for b in boundaries if b.domain.contains(2)] == [2]
    assert [b.event_id for b in boundaries if b.domain.contains(3)] == [7]


def test_symbolic_resource_boundaries_never_evaluate_or_expand_kv_count(monkeypatch):
    protocol = descriptor(
        period=3,
        residues=(0, 2),
        ids=(7, 2, 5),
        activation=Activation(
            CountDomain(1, 0, minimum=10**12), Position(False, 1), Position(True, 2), 3, (0, 2)
        ),
    )

    def no_expansion(*args, **kwargs):
        pytest.fail("query resource boundaries must not evaluate runtime KV N")

    monkeypatch.setattr(Position, "evaluate", no_expansion)
    boundaries = tuple(_parent_boundaries(word_parent(protocol), protocol))
    assert 0 < len(boundaries) <= 27


@pytest.mark.parametrize(
    ("protocol", "reason"),
    (
        (
            descriptor(
                activation=Activation(CountDomain(1, 0, minimum=3), FIRST, Position(False, 2))
            ),
            "singleton or constant-to-end",
        ),
        (descriptor(period=167, residues=(0,), ids=tuple(range(8))), "ID rotation.*budget"),
        (
            descriptor(
                ids=tuple(range(8)),
                activation=Activation(CountDomain(839, 0, minimum=839)),
            ),
            "count predicate.*budget",
        ),
    ),
)
def test_unsupported_activation_or_coupled_period_is_explicit(protocol, reason):
    with pytest.raises(CompileError, match=reason):
        tuple(_parent_boundaries(word_parent(protocol), protocol))


@pytest.fixture(
    scope="module", params=(nested_copy, native_guarded_recurrence), ids=("plain", "cases")
)
def actual(request):
    artifact = compile(request.param, timeout_s=40)
    _, checked, parent = _verify_scope_parent(extract_scopes(artifact.plan))
    return checked, parent


def actual_word(checked, count):
    """Read verified actual channel declarations; enumerate only small test N."""
    tokens = defaultdict(lambda: {"set": [], "wait": []})
    for protocol in checked.channels.values():
        if protocol.kind != "local":
            continue
        actor = protocol.sources[0][0]
        tasks = {task.uid: task for task in checked.access.loop.actors[actor].program.tasks}
        bank = actor, tasks[protocol.sources[0][1]].pipe, tasks[protocol.destinations[0][1]].pipe
        for generation in source_epochs(protocol, count):
            event_id = ordinal_id(protocol, generation)
            row = tokens[bank, event_id, protocol.name, "parent"]
            row["set"].append((generation, None))
            row["wait"].append((generation + protocol.distance, None))
    for uid, leaf in checked.program.leaves.items():
        variants = dict(checked.analyses.variants(uid))
        for generation in range(count):
            label = (
                "single"
                if count == 1
                else "first"
                if generation == 0
                else "last"
                if generation == count - 1
                else "middle"
            )
            analysis = variants.get(label, variants.get(None))
            if analysis is None:
                continue
            loop, channels = analysis[:2]
            for channel in channels.values():
                assert channel.kind == "local" and not channel.carry
                tasks = {
                    (actor, task.uid): task
                    for actor, data in loop.actors.items()
                    for task in data.program.tasks
                }
                bank = (
                    leaf.actor,
                    tasks[channel.sources[0]].pipe,
                    tasks[channel.destinations[0]].pipe,
                )
                for inner in range(loop.constant_count):
                    if (
                        inner >= loop.constant_count - channel.distance
                        or inner % channel.period not in channel.residues
                        or channel.last
                        and inner != loop.constant_count - 1 - channel.distance
                    ):
                        continue
                    event_id = ordinal_id(channel, inner)
                    row = tokens[bank, event_id, f"{uid}/{channel.name}", "child"]
                    row["set"].append((generation, inner))
                    row["wait"].append((generation, inner + channel.distance))
    return tokens


def endpoint_coordinate(endpoint, count):
    return (
        endpoint.parent.evaluate(count),
        endpoint.site.position.offset if isinstance(endpoint, QuerySite) else None,
    )


def test_actual_all_pool_members_and_child_cases_have_exact_real_boundaries(actual):
    checked, parent = actual
    before = dump(checked.program.module), tuple(parent.graph.nodes), tuple(parent.graph.edges)
    prepared = prepare_binary_resources(checked, parent)
    assert prepared.checks and prepared.parent_resources and prepared.child_resources
    assert prepared.child_status == "complete"
    assert prepared.cross_counter_status == prepared.configuration_status == "absent"
    assert prepared.query_channel_status == "not_checked"
    assert (
        dump(checked.program.module),
        tuple(parent.graph.nodes),
        tuple(parent.graph.edges),
    ) == before
    for count in (0, 1, 2, 3, 7):
        tokens = actual_word(checked, count)
        active = [check for check in prepared.checks if check.domain.contains(count)]
        for check in active:
            source = tokens[check.bank, check.event_id, check.source_owner, check.kind]["wait"]
            target = tokens[check.bank, check.event_id, check.target_owner, check.kind]["set"]
            assert endpoint_coordinate(check.source, count) in source
            assert endpoint_coordinate(check.target, count) in target
        for (bank, event_id, old_owner, kind), old in tokens.items():
            for (other_bank, other_id, new_owner, other_kind), new in tokens.items():
                if (bank, event_id, kind) != (other_bank, other_id, other_kind):
                    continue
                assert any(
                    check.bank == bank
                    and check.event_id == event_id
                    and check.source_owner == old_owner
                    and check.target_owner == new_owner
                    and endpoint_coordinate(check.source, count) == max(old["wait"])
                    and endpoint_coordinate(check.target, count) == min(new["set"])
                    for check in active
                ), (count, bank, event_id, old_owner, new_owner)
    if any(leaf.cases for leaf in checked.program.leaves.values()):
        assert any(
            check.kind == "child" and check.target.parent.end and check.domain.contains(3)
            for check in prepared.checks
        )


def test_actual_resource_coverage_matches_independent_finite_query_graph(actual):
    checked, parent = actual
    prepared = prepare_binary_resources(checked, parent)
    query = QueryCompletion(parent)
    coverage = query.coverage_many((c.source, c.target, c.domain) for c in prepared.checks)
    for count in (1, 2, 3, 7):
        finite = FiniteQueries(parent, count)
        for check, result in zip(prepared.checks, coverage, strict=True):
            expected = check.domain.contains(count) and finite.covered(check.source, check.target)
            assert any(domain.contains(count) for domain in result) == expected


def test_resource_checker_submits_one_complete_batch(actual, monkeypatch):
    checked, parent = actual
    prepared = prepare_binary_resources(checked, parent)
    query = QueryCompletion(parent)
    calls = []

    def covered_many(requests):
        requests = tuple(requests)
        calls.append(requests)
        return [index % 2 == 0 for index in range(len(requests))]

    monkeypatch.setattr(query, "covered_many", covered_many)
    assert check_binary_resources(prepared, query) == tuple(
        index % 2 == 0 for index in range(len(prepared.checks))
    )
    assert calls == [tuple((c.source, c.target, c.domain) for c in prepared.checks)]


def test_shared_pool_tail_absence_keeps_the_active_members_cross_query_self_check():
    parent = fifo_proof(1, 1)
    source, destination, _ = parent.access.endpoints
    active = ScopeProtocol("active", "local", (source,), (destination,), 0, 1, (0,), (0,))
    absent = replace(active, name="absent", distance=2)
    for protocol in (active, absent):
        parent.insert(protocol)
    _, analyses = _analyses(parent.program)
    checked = CheckedScopes(
        parent.program, analyses, parent.access, {p.name: p for p in (active, absent)}
    )
    prepared = prepare_binary_resources(checked, parent)
    assert len(prepared.checks) == 1
    check = prepared.checks[0]
    assert check.source_owner == check.target_owner == "active"
    assert check.domain == CountDomain(1, 0, exact=1)
    assert check.source.action == "wait" and check.target.action == "set"
    query = QueryCompletion(parent)
    assert check_binary_resources(prepared, query) == (False,)
    # A pair-only checker sees no surviving A/B boundary and reports success.
    assert all(
        answer
        for check, answer in zip(prepared.checks, (False,), strict=True)
        if check.source_owner != check.target_owner
    )


def test_preparation_requires_actual_parent_protocol_identity(actual):
    checked, parent = actual
    with pytest.raises(CompileError, match="actual checked parent/protocol"):
        prepare_binary_resources(replace(checked, channels={}), parent)


@pytest.mark.parametrize(
    "budget", ("MAX_BINARY_BOUNDARIES", "MAX_BINARY_CHECKS", "MAX_BINARY_PAIR_WORK")
)
def test_resource_analysis_budgets_are_explicit(actual, monkeypatch, budget):
    checked, parent = actual
    monkeypatch.setattr(f"xdsl_npu.scope_query_resources.{budget}", 0)
    with pytest.raises(CompileError, match="query binary.*budget"):
        prepare_binary_resources(checked, parent)


@pytest.fixture(scope="module")
def local_queries():
    from tests.test_scope_query_planning import query_state_store

    artifact = compile(query_state_store, timeout_s=40)
    _, checked, parent = _verify_scope_parent(extract_scopes(artifact.plan))
    return artifact.plan, checked, parent


def test_final_resources_rebuild_actual_binary_and_query_channel_reuse(local_queries):
    _, checked, parent = local_queries
    query = QueryCompletion(parent, copies=3)
    report = verify_query_resources(checked, parent, query)
    assert report["verified"] and report["required_checks"] == len(report["checks"])
    assert report["child_binary_checks"] and report["query_event_reuse_checks"]
    assert report["maximum_static_queries"] is None
    assert all(row["covered"] for row in report["checks"])
    assert all(
        row["all_ids_by_query_translation"]
        for row in report["checks"]
        if row["kind"] == "query_binary"
    )


def test_final_resource_query_window_covers_the_actual_id_rotation(local_queries):
    from xdsl.dialects.builtin import ArrayAttr, IntAttr

    from xdsl_npu.ir import QueryChannelOp
    from xdsl_npu.scope_query_protocols import read_query_protocols
    from xdsl_npu.scope_query_resources import _query_bank
    from xdsl_npu.scope_sync import _leaf_banks

    module = local_queries[0].clone()
    _, checked, parent = _verify_scope_parent(extract_scopes(module))
    protocol = read_query_protocols(checked.program, checked.access)[0]
    source = checked.program.leaves[protocol.source.call]
    target = checked.program.leaves[protocol.target.call]
    source_task = next(
        task
        for data in source.loop.actors.values()
        for task in data.program.tasks
        if task.uid == protocol.source.site.task[1]
    )
    target_task = next(
        task
        for data in target.loop.actors.values()
        for task in data.program.tasks
        if task.uid == protocol.target.site.task[1]
    )
    bank = source.actor, source_task.pipe, target_task.pipe
    used = _leaf_banks(checked.program, checked.analyses)[bank] | set(protocol.ids)
    for other in read_query_protocols(checked.program, checked.access):
        if _query_bank(checked, other) == bank:
            used.update(other.ids)
    for p in checked.channels.values():
        if p.kind == "local":
            send = parent.events[p.sources[0][0], p.name, "set"]
            wait = parent.events[p.destinations[0][0], p.name, "wait"]
            if (send.actor, send.task.pipe, wait.task.pipe) == bank:
                used.update(p.ids)
    extra = next(event_id for event_id in range(8) if event_id not in used)
    declaration = next(op for op in module.walk() if isinstance(op, QueryChannelOp))
    declaration.attributes["event_ids"] = ArrayAttr(IntAttr(i) for i in (*protocol.ids, extra))
    _, checked, parent = _verify_scope_parent(extract_scopes(module))
    with pytest.raises(CompileError, match="query resource proof window"):
        verify_query_resources(checked, parent, QueryCompletion(parent, copies=2))
    assert verify_query_resources(checked, parent, QueryCompletion(parent, copies=3))["verified"]


def test_final_resource_checker_rejects_two_actual_query_channels_sharing_one_id(local_queries):
    from xdsl_npu.scope_query_protocols import install_query_protocols, read_query_protocols
    from xdsl_npu.scope_sync import _scope_access

    module = local_queries[0].clone()
    program = extract_scopes(module)
    access = _scope_access(program)
    protocol = read_query_protocols(program, access)[0]
    install_query_protocols(program, access, (replace(protocol, name="colliding_query_channel"),))
    _, checked, parent = _verify_scope_parent(extract_scopes(module))
    with pytest.raises(
        CompileError, match="query event ID overlaps child/configuration/parent/query"
    ):
        verify_query_resources(checked, parent, QueryCompletion(parent, copies=3))


def test_final_resource_checker_rejects_missing_actual_query_wait(local_queries):
    from xdsl_npu.ir import QuerySyncOp

    module = local_queries[0].clone()
    wait = next(
        op for op in module.walk() if isinstance(op, QuerySyncOp) and op.action.data == "wait"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(CompileError, match="missing.*actual native Set/Wait"):
        _verify_scope_parent(extract_scopes(module))


def test_final_resource_checker_rejects_a_changed_configuration_descriptor(actual):
    checked, _ = actual
    actor = next(iter(checked.access.loop.actors))
    original = checked.access.loop.actors[actor]
    actors = dict(checked.access.loop.actors)
    actors[actor] = replace(original, program=replace(original.program, config=True))
    access = replace(checked.access, loop=replace(checked.access.loop, actors=actors))
    with pytest.raises(CompileError, match="configuration descriptor differs"):
        _configuration_checks(replace(checked, access=access))

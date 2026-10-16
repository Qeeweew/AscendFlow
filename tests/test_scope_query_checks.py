"""Cross-query memory checks retain real roots, clocks and byte diagnostics."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests import test_scope_query_planning
from tests.fixtures.scoped_queries import scoped_query_recurrence
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import QueryChannelOp, QuerySyncOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.region_summary import Access, ReadSpan
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_checks import check_memory, prepare_memory_checks
from xdsl_npu.scope_query_completion import QueryCompletion, QuerySite
from xdsl_npu.scope_query_protocols import read_query_protocols
from xdsl_npu.scope_query_summary import query_native_sites
from xdsl_npu.scope_sync import (
    ParentCompletion,
    ScopeProtocol,
    _install,
    _plan_scope_parent,
    _verify_scope_parent,
)
from xdsl_npu.sync import verify_plan

FIRST, LAST = Position(False, 0), Position(True, 0)
local_query_plan = test_scope_query_planning.planned


@pytest.fixture(scope="module")
def actual():
    with limit(60):
        module = lower_source(scoped_query_recurrence)
        _plan_scope_parent(extract_scopes(module))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        checks = prepare_memory_checks(checked, parent)
    return checked, parent, checks


def roots(checked, obligations):
    return {
        name
        for obligation in obligations
        for name, root in checked.access.origins.items()
        if obligation.storage is root
    }


def singleton_summary(checked, obligation):
    return replace(checked, query_summary=replace(checked.query_summary, obligations=(obligation,)))


def native(checked, endpoint):
    if hasattr(endpoint, "access"):
        uid, site = checked.access.endpoints[endpoint.access.task]
    else:
        uid, site = endpoint.call, endpoint.site
    return next(
        task
        for data in checked.program.leaves[uid].loop.actors.values()
        for task in data.program.tasks
        if task.uid == site.task[1]
    )


def test_preparation_deduplicates_endpoints_without_losing_byte_obligations(actual):
    checked, _, checks = actual
    obligations = checked.query_summary.obligations
    assert checks and len(checks) < len(obligations)
    assert len({(check.source, check.target, check.domain) for check in checks}) == len(checks)
    assert sorted(map(repr, (o for check in checks for o in check.obligations))) == sorted(
        map(repr, obligations)
    )
    assert all(check.source.query == 0 for check in checks)
    for check in checks:
        assert all(o.domain == check.domain for o in check.obligations)
        assert check.source.site.region == check.source.call
        assert check.target.site.region == check.target.call
        assert check.target.site.point == "begin"
    assert not roots(checked, obligations) & {"Q", "K", "O"}


def test_n_equals_one_ring_reuse_targets_query_two(actual):
    checked, _, checks = actual
    score = [
        check
        for check in checks
        if roots(checked, check.obligations) == {"Score"} and check.domain.contains(1)
    ]
    assert score and all(check.target.query == 2 for check in score)
    assert all(not check.domain.contains(2) for check in score)


def test_missing_final_state_mte3_read_release_is_not_hidden(actual):
    checked, parent, checks = actual
    selected = [
        check
        for check in checks
        if any(name.endswith("/state") for name in roots(checked, check.obligations))
        and check.source.site.point == "end"
        and check.source.parent == LAST
    ]
    assert selected
    with limit(30):
        query = QueryCompletion(parent, copies=len(checked.query_summary.parts))
        answers = check_memory(selected, query)
    tails = []
    for check, answer in zip(selected, answers, strict=True):
        leaf = checked.program.leaves[check.source.call]
        tasks = {
            task.uid: task for data in leaf.loop.actors.values() for task in data.program.tasks
        }
        if tasks[check.source.site.task[1]].pipe == "MTE3":
            tails.append(answer)
            assert check.target.parent == FIRST
            assert check.target.query == 1
    assert tails and not any(tails)


@pytest.mark.parametrize(
    ("step", "first", "last", "domain"),
    [
        (1, FIRST, LAST, CountDomain(2, 1, minimum=3)),
        (2, Position(False, 1), Position(True, 1), CountDomain(2, 1, minimum=5)),
    ],
)
def test_real_span_witnesses_share_one_batch_and_keep_original_predicates(
    actual, monkeypatch, step, first, last, domain
):
    checked, parent, _ = actual
    original = next(
        o
        for o in checked.query_summary.obligations
        if native(checked, o.source).native == "add"
        and o.domain.contains(3)
        and native(checked, o.destination).native != "fill"
    )
    source = replace(
        original.source, access=ReadSpan(original.source.access.task, first, last, step)
    )
    target = replace(
        original.destination, access=ReadSpan(original.destination.access.task, first, last, step)
    )
    obligation = replace(original, source=source, destination=target, domain=domain)
    repeated = replace(obligation, start=obligation.start + 1)
    checked = replace(
        checked,
        query_summary=replace(checked.query_summary, obligations=(obligation, repeated)),
    )
    if parent.period % step:
        # Reproject the same actual IR/protocols at a supported static period;
        # the source module and its planner run are still shared by all tests.
        with limit(30):
            parent = ParentCompletion(
                checked.program,
                checked.analyses,
                checked.access,
                step,
                extra_protocols=checked.channels.values(),
                boundary_sites=query_native_sites(checked.query_summary, checked.access),
            )
            for protocol in checked.channels.values():
                parent.insert(protocol)
    anchored = parent.anchored()
    covered_many, batches = anchored.covered_many, []

    def batch(queries):
        queries = tuple(queries)
        batches.append(queries)
        return covered_many(queries)

    monkeypatch.setattr(anchored, "covered_many", batch)
    with limit(30):
        checks = prepare_memory_checks(checked, parent)
    assert len(batches) == 1 and len(batches[0]) >= 2
    assert all(query[-1].count == domain for query in batches[0])
    assert all(query[-1].first == first and query[-1].last == last for query in batches[0])
    assert all(
        query[-1].period == step and query[-1].residues == (first.offset % step,)
        for query in batches[0]
    )
    assert len(checks) == 1 and checks[0].obligations == (obligation, repeated)
    assert checks[0].source.site.point == "prefix"
    assert checks[0].source.parent == last
    assert checks[0].target.site.point == "begin"
    assert checks[0].target.parent == first


def test_unproved_span_cannot_fall_back_to_last_native_end(actual, monkeypatch):
    checked, parent, _ = actual
    original = next(
        o
        for o in checked.query_summary.obligations
        if native(checked, o.source).native == "muls" and o.domain.contains(3)
    )
    obligation = replace(
        original,
        domain=CountDomain(1, 0, minimum=3),
        source=replace(
            original.source, access=ReadSpan(original.source.access.task, FIRST, LAST, 1)
        ),
    )
    monkeypatch.setattr(parent.anchored(), "covered_many", lambda queries: [False] * len(queries))
    with pytest.raises(CompileError, match=r"\[\d+:\d+\]: unproved query reader span"):
        prepare_memory_checks(singleton_summary(checked, obligation), parent)


def test_span_step_outside_parent_period_has_explicit_diagnostic(actual):
    checked, parent, _ = actual
    original = next(o for o in checked.query_summary.obligations if o.domain.contains(3))
    obligation = replace(
        original,
        domain=CountDomain(1, 0, minimum=3),
        source=replace(
            original.source,
            access=ReadSpan(original.source.access.task, FIRST, LAST, parent.period + 1),
        ),
    )
    with pytest.raises(CompileError, match="span step must divide the parent graph period"):
        prepare_memory_checks(singleton_summary(checked, obligation), parent)


def test_memory_checks_use_one_query_batch_and_validate_original_roots(actual, monkeypatch):
    checked, parent, checks = actual
    query = QueryCompletion(parent, copies=len(checked.query_summary.parts))
    covered_many, batches = query.covered_many, []

    def batch(queries):
        queries = tuple(queries)
        batches.append(queries)
        return covered_many(queries)

    monkeypatch.setattr(query, "covered_many", batch)
    chosen = checks[:2]
    with limit(30):
        answers = check_memory(chosen, query)
    assert len(batches) == 1 and len(answers) == len(chosen)
    assert batches[0] == tuple((c.source, c.target, c.domain) for c in chosen)
    forged = replace(chosen[0].obligations[0], storage=object())
    with pytest.raises(CompileError, match="actual allocation identity"):
        check_memory((replace(chosen[0], obligations=(forged,)),), query)
    assert len(batches) == 1


def test_singleton_span_keeps_n_equals_one_domain_and_actual_inner_position(actual):
    checked, parent, _ = actual
    original = next(o for o in checked.query_summary.obligations if o.domain.contains(1))
    access = original.source.access
    single = ReadSpan(access.task, FIRST, LAST, 1)
    obligation = replace(
        original,
        domain=CountDomain(1, 0, exact=1),
        source=replace(original.source, access=single),
    )
    checks = prepare_memory_checks(singleton_summary(checked, obligation), parent)
    assert len(checks) == 1
    result = checks[0]
    uid, site = checked.access.endpoints[access.task]
    assert result.source.call == uid
    assert result.source.site == replace(site, point="end")
    assert result.source.parent == FIRST
    assert result.domain == obligation.domain


def test_absent_consumer_does_not_pass_by_inactive_case_vacuity(actual):
    checked, parent, _ = actual
    original = checked.query_summary.obligations[0]
    target = replace(original.destination, access=Access(original.destination.access.task, FIRST))
    # A parent native is absent beyond N, even though the projected family is
    # known and an anchored proof by itself would skip the inactive endpoint.
    target = replace(target, access=replace(target.access, position=Position(False, 9)))
    obligation = replace(original, domain=CountDomain(1, 0, exact=1), destination=target)
    with pytest.raises(CompileError, match="escapes|does not exist"):
        prepare_memory_checks(singleton_summary(checked, obligation), parent)


def test_inactive_native_cannot_supply_first_span_consumer(actual):
    checked, parent, _ = actual
    original = next(
        o for o in checked.query_summary.obligations if native(checked, o.source).pipe == "MTE3"
    )
    # The actual store only exists at the last KV position. FIRST is in bounds
    # for N >= 3, but no store begin can serve as that span's consumer witness.
    target = replace(
        original.destination, access=ReadSpan(original.source.access.task, FIRST, LAST, 1)
    )
    obligation = replace(original, domain=CountDomain(1, 0, minimum=3), destination=target)
    with pytest.raises(CompileError, match="endpoint does not exist over its count domain"):
        prepare_memory_checks(singleton_summary(checked, obligation), parent)


def test_unknown_native_is_rejected_before_completion(actual):
    checked, parent, _ = actual
    original = checked.query_summary.obligations[0]
    target = replace(original.destination, access=Access(("aiv0", "not_a_native"), FIRST))
    with pytest.raises(CompileError, match="no actual native endpoint"):
        prepare_memory_checks(
            singleton_summary(checked, replace(original, destination=target)), parent
        )


def test_final_local_query_verification_requires_actual_memory_and_resources(local_query_plan):
    module, _, result, checked, _, checks, query = local_query_plan
    assert result.protocols and all(check_memory(checks, query))
    assert checked.query_report is None  # A memory-stage result grants no public proof.
    with limit(30):
        report, checked = verify_plan(module, _details=True)
    assert report["cross_query_verified"] is True
    assert checked.query_report == report["query_verification"]
    assert checked.query_report["verified"] is True
    assert checked.query_report["resources"]["verified"] is True


@pytest.mark.parametrize("action", ("set", "wait"))
def test_final_verifier_rejects_each_deleted_actual_query_sync(local_query_plan, action):
    module = local_query_plan[0].clone()
    sync = next(
        op for op in module.walk() if isinstance(op, QuerySyncOp) and op.action.data == action
    )
    sync.parent.erase_op(sync)
    before = dump(module)
    with (
        limit(30),
        pytest.raises(CompileError, match="missing or changes an actual native Set/Wait placement"),
    ):
        verify_plan(module)
    assert dump(module) == before


@pytest.mark.parametrize("ids", ((8,), (0, 0)))
def test_final_verifier_rejects_corrupted_actual_query_event_ids(local_query_plan, ids):
    module = local_query_plan[0].clone()
    channel = next(op for op in module.walk() if isinstance(op, QueryChannelOp))
    channel.attributes["event_ids"] = ArrayAttr(IntAttr(value) for value in ids)
    with limit(30), pytest.raises(CompileError, match="distinct IDs"):
        verify_plan(module)


def test_wrong_actual_query_distance_restores_the_adjacent_query_memory_hazard(local_query_plan):
    module = local_query_plan[0].clone()
    for op in module.walk():
        if isinstance(op, QueryChannelOp):
            op.attributes["distance"] = IntAttr(2)
    with limit(30), pytest.raises(CompileError, match="missing cross-query memory"):
        verify_plan(module)


def test_memory_completion_does_not_approve_a_conflicting_parent_event_id(local_query_plan):
    module = local_query_plan[0].clone()
    with limit(30):
        _, checked, _ = _verify_scope_parent(extract_scopes(module))
        protocols = read_query_protocols(checked.program, checked.access)
        query = min(protocols, key=lambda value: value.source.site.position.offset)
        source = next(
            owner
            for owner, (uid, site) in checked.access.endpoints.items()
            if uid == query.source.call and replace(query.source.site, point=site.point) == site
        )
        target = next(
            owner
            for owner, (uid, site) in checked.access.endpoints.items()
            if site.task[0] == source[0]
            and site.position.offset > query.source.site.position.offset
            and native(checked, QuerySite(0, uid, site, LAST)).native == "muls"
        )
        # A last-KV row-zero store may precede the independent row-one V
        # update. This is a real, locally valid parent resource in the exact
        # MTE3->V bank already used by the query release, with the same ID.
        conflict = ScopeProtocol(
            "query_parent_id_conflict",
            "local",
            (source,),
            (target,),
            0,
            1,
            (0,),
            query.ids,
            activation=Activation(CountDomain(1, 0, minimum=1), LAST, LAST),
        )
        _install(checked.program, checked.access, (conflict,))
        _, checked, parent = _verify_scope_parent(extract_scopes(module))
        checks = prepare_memory_checks(checked, parent)
        completion = QueryCompletion(parent, copies=len(checked.query_summary.parts))
        assert all(check_memory(checks, completion))
        with pytest.raises(
            CompileError, match="query event ID overlaps child/configuration/parent/query resource"
        ):
            verify_plan(module)

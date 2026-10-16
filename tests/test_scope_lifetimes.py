"""Shared scratch needs physical lifetime order and fixed logical RAW bindings."""

from dataclasses import replace
from itertools import pairwise

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.fixtures.nested import nested_copy, nested_cv, shared_l0c_qk_pv
from tests.test_definition_lifetimes import (
    _actual_events,
    _actual_natives,
    _c0_cross_call_retirements,
    _logical_model,
    _project_c,
    _pv_recurrence,
    _read_qk_operand_zero,
)
from xdsl_npu import language as N
from xdsl_npu.compilation import compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import CallOp, TaskOp, ViewOp
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_domains import ScopedDependence
from xdsl_npu.scope_lifetimes import plan_lifetimes, propose_lifetimes
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes


@pytest.fixture(scope="module")
def shared():
    source = shared_l0c_qk_pv.to_ir()
    scope = extract_scopes(lower_scopes(source.body.block.first_op))
    access = analyze_scope_access(scope)
    roles = dict(zip(scope.leaves, ("QK", "PV", "FIX"), strict=True))
    return scope, access, roles


@pytest.fixture(scope="module")
def shared_compilation():
    return compile(shared_l0c_qk_pv, timeout_s=25)


def _dispatch_trace(scope, roles, count):
    """Interpret actual CallOp slots directly, without access/completion analyses."""
    parent = scope.parents["aic"]
    calls = [op for op in parent.body.block.ops if isinstance(op, CallOp)]
    ordered = sorted(enumerate(calls), key=lambda item: (item[1].issue_order.data, item[0]))
    return tuple(
        (roles[call.uid.data], step - call.stage.data)
        for step in range(count + 1)
        for _, call in ordered
        if 0 <= step - call.stage.data < count
    )


def _c0_lifetimes(trace):
    """Source-derived value intervals, with real native last-reader milestones.

    These intervals choose the physical occupants. They do not claim that call
    dispatch retires a reader: adjacent intervals still require a proof from the
    first interval's native FIX end to the next interval's native MAD begin.
    PV's seven read/modify/write MADs belong to its initial clear's value, and
    its separate output FIX must keep that same producer binding.
    """
    lifetimes, pending = [], None
    for role, generation in trace:
        if role == "QK":
            assert pending is None
            for inner in (0, 2, 4, 6):
                lifetimes.append(
                    ((role, generation, "MAD", inner), (role, generation, "FIX", inner))
                )
        elif role == "PV":
            assert pending is None
            pending = role, generation, "MAD", 0
        else:
            assert role == "FIX" and pending == ("PV", generation, "MAD", 0)
            lifetimes.append((pending, (role, generation, "FIX", 0)))
            pending = None
    assert pending is None
    return tuple(lifetimes)


def test_qk_and_pv_share_only_the_original_l0c_storage(shared):
    scope, _, _ = shared
    qk, pv, fix = scope.leaves.values()
    roots = [
        {
            capture.root.sym_name.data: capture.root
            for capture in leaf.captures
            if not capture.root.external.data
        }
        for leaf in (qk, pv, fix)
    ]
    assert set(roots[0]) & set(roots[1]) == {"c"}
    assert set(roots[0]) - {"c"} == {"qk_l1", "qk_a", "qk_b"}
    assert set(roots[1]) - {"c"} == {"pv_l1", "pv_a", "pv_b"}
    assert roots[0]["c"] is roots[1]["c"] is roots[2]["c"]
    assert roots[0]["c"].ref.type.bank.data == "L0C"
    assert roots[0]["c"].nbytes.data == 2048


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, ()),
        (1, (("QK", 0), ("PV", 0), ("FIX", 0))),
        (2, (("QK", 0), ("QK", 1), ("PV", 0), ("FIX", 0), ("PV", 1), ("FIX", 1))),
        (
            3,
            (
                ("QK", 0),
                ("QK", 1),
                ("PV", 0),
                ("FIX", 0),
                ("QK", 2),
                ("PV", 1),
                ("FIX", 1),
                ("PV", 2),
                ("FIX", 2),
            ),
        ),
    ],
)
def test_expected_shared_c0_dispatch_trace_includes_startup_and_drain(shared, count, expected):
    scope, _, roles = shared
    assert _dispatch_trace(scope, roles, count) == expected
    lifetimes = _c0_lifetimes(expected)
    assert len(lifetimes) == count * 5
    for generation in range(count - 1):
        assert expected.index(("QK", generation + 1)) < expected.index(("PV", generation))
        assert expected.index(("PV", generation)) < expected.index(("FIX", generation))


def test_expected_c0_retirement_frontiers_include_negative_logical_distance(shared):
    scope, _, roles = shared
    lifetimes = _c0_lifetimes(_dispatch_trace(scope, roles, 3))
    cross_call = tuple(
        (old_reader, new_writer)
        for (_, old_reader), (new_writer, _) in pairwise(lifetimes)
        if old_reader[:2] != new_writer[:2]
    )
    assert cross_call == (
        (("QK", 0, "FIX", 6), ("QK", 1, "MAD", 0)),
        (("QK", 1, "FIX", 6), ("PV", 0, "MAD", 0)),
        (("FIX", 0, "FIX", 0), ("QK", 2, "MAD", 0)),
        (("QK", 2, "FIX", 6), ("PV", 1, "MAD", 0)),
        (("FIX", 1, "FIX", 0), ("PV", 2, "MAD", 0)),
    )


def test_current_logical_frontier_requires_reverse_c0_retirement(shared):
    scope, access, roles = shared
    tasks = {
        (uid, task.uid): task
        for uid, leaf in scope.leaves.items()
        for data in leaf.loop.actors.values()
        for task in data.program.tasks
    }

    def endpoint(key):
        uid, site = access.endpoints[key]
        return roles[uid], tasks[uid, site.task[1]].pipe, site.position.offset

    obligations = {
        (
            dependency.hazard,
            endpoint(dependency.source),
            endpoint(dependency.destination),
            dependency.distance,
        )
        for dependency in access.obligations
        if dependency.storage == "aic/c" and (dependency.start, dependency.stop) == (0, 1024)
    }
    assert ("WAW", ("PV", "M", 7), ("QK", "M", 0), 1) in obligations
    assert ("WAR", ("FIX", "FIX", 0), ("QK", "M", 0), 1) in obligations
    # This logical producer binding remains mandatory after lifetime reordering.
    assert ("RAW", ("PV", "M", 7), ("FIX", "FIX", 0), 0) in obligations


def test_shared_c0_compiles_with_actual_definition_retirement(shared_compilation):
    result = shared_compilation
    assert "scope_access" not in result.asc
    assert result.asc.count("asc_sync_pipe(PIPE_ALL)") == 1


def _expand_retirements(scope, access, dependencies, count, start=0):
    roles = dict(zip(scope.leaves, ("QK", "PV", "FIX"), strict=True))
    natives = {
        (uid, task.uid): task
        for uid, leaf in scope.leaves.items()
        for data in leaf.loop.actors.values()
        for task in data.program.tasks
    }

    def event(key, generation):
        uid, site = access.endpoints[key]
        task = natives[uid, site.task[1]]
        return roles[uid], generation, "MAD" if task.pipe == "M" else "FIX", site.position.offset

    result = set()
    for dep in dependencies:
        if dep.storage != "aic/c" or dep.start != start or dep.hazard != "WAR":
            continue
        for source in range(count):
            target = source + dep.distance
            if not 0 <= target < count:
                continue
            if isinstance(dep, ScopedDependence):
                active = dep.activation
                if not (
                    active.count.contains(count)
                    and active.first.evaluate(count) <= source <= active.last.evaluate(count)
                    and source % active.period in active.residues
                ):
                    continue
            elif target % dep.period not in dep.residues:
                continue
            result.add((event(dep.source, source), event(dep.destination, target)))
    return result


@pytest.mark.parametrize("stages", ((0, 1), (0, 2), (2, 0), (1, 3)))
@pytest.mark.parametrize("qk_first", (False, True))
def test_lifetime_proposal_matches_actual_ref_oracle_across_stage_and_issue_choices(
    shared, stages, qk_first
):
    original, _, _ = shared
    module = original.module.clone()
    scope = extract_scopes(module)
    qk, pv, fix = scope.leaves.values()
    qk.call.attributes["stage"] = IntAttr(stages[0])
    pv.call.attributes["stage"] = fix.call.attributes["stage"] = IntAttr(stages[1])
    for rank, leaf in enumerate((qk, pv, fix) if qk_first else (pv, fix, qk)):
        leaf.call.attributes["issue_order"] = IntAttr(rank)
    scope = extract_scopes(module)
    access = analyze_scope_access(scope, definitions=True)
    plan = plan_lifetimes(scope, access)
    expected_roots = ("aic/c",) if qk_first or stages != (0, 1) else ()
    assert plan.reordered_roots == expected_roots and plan.fallbacks == ()
    for count in (0, 1, 2, 3, 7):
        logical = _project_c(_actual_events(_actual_natives(module, count, physical=False)))
        physical = _project_c(_actual_events(_actual_natives(module, count, physical=True)))
        model = _logical_model(logical)
        assert _expand_retirements(scope, access, plan.dependencies, count) == (
            _c0_cross_call_retirements(physical, model)
        )
        assert _expand_retirements(scope, access, plan.dependencies, count, 1024) == (
            _c0_cross_call_retirements(physical, model, "c1")
        )


def test_lifetime_proposal_preserves_logical_raw_and_external_gm_objects(shared):
    scope, _, _ = shared
    access = analyze_scope_access(scope, definitions=True)
    dependencies = propose_lifetimes(scope, access)
    for original in access.obligations:
        if original.hazard == "RAW" or original.storage != "aic/c":
            assert any(dep is original for dep in dependencies)
    assert (
        sum(
            isinstance(dep, ScopedDependence) and dep.start == 0 and dep.hazard == "WAR"
            for dep in dependencies
        )
        == 5
    )


@pytest.mark.parametrize("fixture", (nested_copy, nested_cv))
def test_unaffected_nested_programs_keep_original_dependency_fast_path(fixture):
    scope = extract_scopes(lower_scopes(fixture.to_ir().body.block.first_op))
    access = analyze_scope_access(scope, definitions=True)
    plan = plan_lifetimes(scope, access)
    assert plan.reordered_roots == plan.fallbacks == ()
    assert plan.visits == 0
    assert len(plan.dependencies) == len(access.obligations)
    assert all(
        left is right for left, right in zip(plan.dependencies, access.obligations, strict=True)
    )


def test_incoming_rw_root_keeps_logical_dependencies_and_explicit_fallback(shared):
    original, _, _ = shared
    module = original.module.clone()
    _pv_recurrence(module)
    scope = extract_scopes(module)
    access = analyze_scope_access(scope, definitions=True)
    plan = plan_lifetimes(scope, access)
    assert plan.reordered_roots == ()
    assert plan.fallbacks == (("aic/c", "incoming RW is a continuation, not a fresh definition"),)
    assert all(
        left is right for left, right in zip(plan.dependencies, access.obligations, strict=True)
    )


@pytest.mark.parametrize("mutation", (_pv_recurrence, _read_qk_operand_zero))
def test_true_shared_operand_or_recurrence_conflicts_remain_rejected(shared, mutation):
    scope, _, _ = shared
    module = scope.module.clone()
    mutation(module)
    with pytest.raises(CompileError, match="cycle|unsatisfied|uncovered|cannot cover"):
        compile(module, timeout_s=25)


def test_lifetime_count_guards_do_not_iterate_large_parent_counts(shared, monkeypatch):
    import xdsl_npu.scope_lifetimes as lifetimes

    scope, _, _ = shared
    access = analyze_scope_access(scope, definitions=True)
    original_range = range

    def bounded_range(*args):
        assert all(abs(value) < 1000 for value in args), "enumerated parent N"
        return original_range(*args)

    monkeypatch.setattr(lifetimes, "range", bounded_range, raising=False)
    huge = plan_lifetimes(replace(scope, count=1 << 60), access)
    small = plan_lifetimes(replace(scope, count=3), access)
    assert huge.reordered_roots == small.reordered_roots == ("aic/c",)
    assert huge.visits == small.visits
    for dep in huge.dependencies:
        if isinstance(dep, ScopedDependence):
            assert dep.activation.count.contains(1 << 60)


def test_lifetime_budget_and_unsupported_translation_do_not_drop_old_edges(shared, monkeypatch):
    import xdsl_npu.scope_lifetimes as lifetimes

    scope, _, _ = shared
    access = analyze_scope_access(scope, definitions=True)
    key = next(key for key in access.capture_offsets if key[1] == "aic/c")
    offset = access.capture_offsets[key]
    offsets = {**access.capture_offsets, key: (offset[0], ((0, 32, 0, 1),), (), 0, 0)}
    plan = plan_lifetimes(scope, replace(access, capture_offsets=offsets))
    assert plan.reordered_roots == ()
    assert "stationary" in plan.fallbacks[0][1]
    assert all(
        left is right for left, right in zip(plan.dependencies, access.obligations, strict=True)
    )
    monkeypatch.setattr(lifetimes, "MAX_LIFETIME_VISITS", 1)
    with pytest.raises(CompileError, match="scope lifetime analysis visit budget"):
        plan_lifetimes(scope, access)


@pytest.mark.parametrize("translation", ("core_stride", "core_terms"))
def test_child_native_core_translation_cannot_hide_uninitialized_bytes(translation):
    source = """
def nested(Score:N.Tensor((2,2,32),N.f32,access="write"),
           Y:N.Tensor((2,n,16),N.f32,access="write"),n:N.int64):
    with N.Kernel(2) as group:
        producer=N.alloc_ub((16,),name="producer")
        consumer=N.alloc_ub((16,),name="consumer")
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="produce"):
                for j in N.Pipelined(1,stages=1):
                    N.fill(producer,1.0)
                    N.copy(producer,Score[group,t%2,:16])
            with N.Stage(1,name="consume"):
                for j in N.Pipelined(1,stages=1):
                    N.copy(Score[group,t%2,:16],consumer)
                    N.copy(consumer,Y[group,t,:])
"""
    definition = N.from_source(source, disjoint_external=True)
    physical = lower_scopes(definition.to_ir().body.block.first_op)
    scope = extract_scopes(physical)
    _, reader = scope.leaves.values()
    capture = next(cap for cap in reader.captures if cap.root.sym_name.data == "Score")
    load = next(op for op in reader.stream.walk() if isinstance(op, TaskOp))
    view = load.refs[0].owner
    assert isinstance(view, ViewOp) and view.source is capture.op.ref
    view.attributes[translation] = (
        IntAttr(64)
        if translation == "core_stride"
        else ArrayAttr([ArrayAttr([IntAttr(64), IntAttr(1), IntAttr(0)])])
    )
    # Independently enumerate the actual source's writes. The added native
    # translation moves group 1's read from [256:320] into unwritten [320:384],
    # wholly inside Score. A summary of only core 0 would falsely bind a RAW.
    written = {
        byte
        for group in range(2)
        for slot in range(2)
        for byte in range(group * 256 + slot * 128, group * 256 + slot * 128 + 64)
    }
    assert not written.intersection(range(320, 384))
    assert capture.offset.evaluate({"$core": 1, "$parent0": 0, "n": 2}) == 256
    with pytest.raises(CompileError, match="child.*capture|ancestor.*capture"):
        compile(physical, timeout_s=25)

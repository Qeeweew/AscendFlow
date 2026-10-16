"""Query boundary occupants follow actual dispatch and logical native bindings."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr

from tests.fixtures.attention_target import attention_target
from tests.fixtures.nested import shared_l0c_qk_pv
from tests.test_definition_lifetimes import (
    _actual_calls,
    _actual_events,
    _actual_natives,
    _logical_model,
    _pv_recurrence,
)
from xdsl_npu.budget import limit
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_lifetimes import plan_lifetimes
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_sync import _scope_access


@pytest.fixture(scope="module")
def shared_template():
    return lower_source(shared_l0c_qk_pv)


def configuration(template, stages=(0, 1), qk_first=True):
    module = template.clone()
    program = extract_scopes(module)
    qk, pv, fix = program.leaves.values()
    qk.call.attributes["stage"] = IntAttr(stages[0])
    pv.call.attributes["stage"] = fix.call.attributes["stage"] = IntAttr(stages[1])
    for rank, leaf in enumerate((qk, pv, fix) if qk_first else (pv, fix, qk)):
        leaf.call.attributes["issue_order"] = IntAttr(rank)
    program = extract_scopes(module)
    access = analyze_scope_access(program, definitions=True)
    before = dump(module)
    plan = plan_lifetimes(program, access)
    assert dump(module) == before
    return module, program, access, plan


@pytest.fixture(
    scope="module",
    params=[
        (stages, first) for stages in ((0, 1), (0, 2), (2, 0), (1, 3)) for first in (False, True)
    ],
    ids=lambda value: f"stages{value[0]}-qk_first{value[1]}",
)
def scheduled(request, shared_template):
    stages, first = request.param
    return configuration(shared_template, stages, first)


@pytest.fixture(scope="module")
def default(shared_template):
    return configuration(shared_template)


def c_events(module, count, *, physical):
    """Use the existing actual Capture/View/Task interpreter, retaining native IDs."""
    result = []
    for event in _actual_events(_actual_natives(module, count, physical=physical)):
        reads, writes = (
            tuple(cell for cell in cells if cell[0].ref.type.bank.data == "L0C")
            for cells in (event.reads, event.writes)
        )
        if reads or writes:
            result.append(replace(event, reads=reads, writes=writes))
    return tuple(result)


def boundary_event(program, access, endpoint, parent, count):
    uid, site = access.endpoints[endpoint]
    leaf = program.leaves[uid]
    _, _, roles = _actual_calls(program.module)
    native = next(
        op for op in leaf.stream.walk() if isinstance(op, TaskOp) and op.uid.data == site.task[1]
    )
    return (
        roles[leaf.call],
        parent.evaluate(count),
        native.native.data,
        site.position.evaluate(leaf.loop.constant_count),
        native.uid.data,
    )


def assert_boundary_oracle(module, program, access, boundaries, count):
    logical = c_events(module, count, physical=False)
    physical = c_events(module, count, physical=True)
    _, owners, last_writes, pending_readers = _logical_model(logical)
    actual_cells = {cell for event in physical for cell in (*event.reads, *event.writes)}
    assert {(cell.start, cell.stop) for cell in boundaries} == {
        (lo, hi) for _, lo, hi in actual_cells
    }
    for boundary in boundaries:
        assert boundary.domain.contains(count) and not boundary.domain.contains(0)
        assert boundary.reads_before_write == ()
        cell = next(cell for cell in actual_cells if cell[1:] == (boundary.start, boundary.stop))
        # Logical native modes decide which writes create fresh definitions.
        # Their order comes from the actual physical call and native dispatch.
        fresh = [
            event.uid
            for event in physical
            if cell in event.writes and owners[event.uid, cell] == event.uid
        ]
        head, tail = fresh[0], fresh[-1]
        assert (
            boundary_event(
                program, access, boundary.first_write.task, boundary.first_write.position, count
            )
            == head
        )
        assert (
            boundary_event(
                program, access, boundary.last_write.task, boundary.last_write.position, count
            )
            == last_writes[tail, cell]
        )
        actual_readers = set()
        for span in boundary.pending_reads:
            assert span.first == span.last and span.step == 1
            actual_readers.add(boundary_event(program, access, span.task, span.first, count))
        assert actual_readers == set(pending_readers[tail, cell])


@pytest.mark.parametrize("count", (1, 2, 3, 7))
def test_boundary_native_identities_match_actual_dispatch_oracle(scheduled, count):
    module, program, access, plan = scheduled
    if not plan.reordered_roots:
        assert plan.boundaries == () and plan.fallbacks == ()
        assert all(
            left is right for left, right in zip(plan.dependencies, access.obligations, strict=True)
        )
        return
    assert plan.reordered_roots == ("aic/c",) and plan.fallbacks == ()
    assert len(plan.boundaries) == 2
    assert_boundary_oracle(module, program, access, plan.boundaries, count)


def test_separate_fix_reader_stays_bound_to_the_tail_pv_definition(default):
    module, program, access, plan = default
    lower = next(cell for cell in plan.boundaries if cell.start == 0)
    pv = tuple(program.leaves.values())[1]
    local_cells = [
        cell
        for cell in access.summaries[pv.call.uid.data].cells
        if access.capture_roots[pv.call.uid.data, cell.storage] == "aic/c"
    ]
    assert local_cells and all(cell.pending_reads == () for cell in local_cells)
    assert len(lower.pending_reads) == 1
    actual = boundary_event(
        program, access, lower.pending_reads[0].task, lower.pending_reads[0].first, 1
    )
    assert actual[:4] == ("FIX", 0, "fix", 0)
    assert_boundary_oracle(module, program, access, plan.boundaries, 1)


def test_n1_keeps_clear_last_write_and_fix_native_identities_distinct(default):
    _, program, access, plan = default
    lower, upper = sorted(plan.boundaries, key=lambda cell: cell.start)
    assert boundary_event(program, access, lower.first_write.task, lower.first_write.position, 1)[
        :4
    ] == ("QK", 0, "mad", 0)
    assert boundary_event(program, access, lower.last_write.task, lower.last_write.position, 1)[
        :4
    ] == ("PV", 0, "mad_first", 7)
    assert boundary_event(program, access, upper.first_write.task, upper.first_write.position, 1)[
        :4
    ] == ("QK", 0, "mad", 1)
    assert boundary_event(program, access, upper.last_write.task, upper.last_write.position, 1)[
        :4
    ] == ("QK", 0, "mad", 7)


def test_incoming_rw_fallback_exports_no_fresh_boundary(shared_template):
    module = shared_template.clone()
    _pv_recurrence(module)
    program = extract_scopes(module)
    access = analyze_scope_access(program, definitions=True)
    plan = plan_lifetimes(program, access)
    assert plan.reordered_roots == plan.boundaries == ()
    assert plan.fallbacks == (("aic/c", "incoming RW is a continuation, not a fresh definition"),)
    assert all(
        left is right for left, right in zip(plan.dependencies, access.obligations, strict=True)
    )


def test_late_upper_cell_fallback_cannot_publish_lower_cell_boundary(shared_template):
    module = shared_template.clone()
    program = extract_scopes(module)
    fix = tuple(program.leaves.values())[-1]
    captured = next(cap for cap in fix.captures if cap.root.sym_name.data == "c")
    native = next(op for op in fix.stream.walk() if isinstance(op, TaskOp))
    captured.op.attributes["nbytes"] = IntAttr(2048)
    native.refs[0].owner.attributes["offset"] = IntAttr(1024)
    program = extract_scopes(module)
    access = analyze_scope_access(program, definitions=True)
    assert len(plan_lifetimes(program, access).boundaries) == 2
    qk = next(iter(program.leaves.values()))
    earlier_upper_write = next(
        key
        for key, (uid, site) in access.endpoints.items()
        if uid == qk.call.uid.data and site.position.offset == 5 and "mad" in site.task[1]
    )
    # Corrupt only the upper cell's incoming producer binding. The lower cell
    # remains valid and is processed first; unsupported-root fallback is atomic.
    obligations = tuple(
        replace(dep, source=earlier_upper_write)
        if dep.storage == "aic/c"
        and dep.start == 1024
        and dep.hazard == "RAW"
        and access.endpoints[dep.destination][0] == fix.call.uid.data
        else dep
        for dep in access.obligations
    )
    assert any(
        left is not right for left, right in zip(obligations, access.obligations, strict=True)
    )
    plan = plan_lifetimes(program, replace(access, obligations=obligations))
    assert plan.reordered_roots == plan.boundaries == ()
    assert plan.fallbacks == (
        ("aic/c", "incoming read does not bind one exported logical version"),
    )
    assert all(left is right for left, right in zip(plan.dependencies, obligations, strict=True))


def test_real_attention_access_exports_both_original_c0_plane_boundaries():
    with limit(30):
        module = lower_source(attention_target())
        program = extract_scopes(module)
        before = dump(module)
        access = _scope_access(program)
    assert dump(module) == before
    root = next(
        cap.root
        for leaf in program.leaves.values()
        for cap in leaf.captures
        if cap.root.ref.type.bank.data == "L0C"
    )
    assert root.nbytes.data == 131072
    name = "aic/" + root.sym_name.data
    assert access.lifetime_report["reordered_roots"] == [name]
    cells = sorted(access.lifetime_boundaries, key=lambda cell: cell.start)
    assert [(cell.storage, cell.start, cell.stop) for cell in cells] == [
        (name, 0, 65536),
        (name, 65536, 131072),
    ]

    def native(at):
        uid, site = access.endpoints[at.task]
        leaf = program.leaves[uid]
        task = next(
            op
            for op in leaf.stream.walk()
            if isinstance(op, TaskOp) and op.uid.data == site.task[1]
        )
        return uid, task.native.data, site.position.evaluate(leaf.loop.constant_count)

    lower, upper = cells
    assert native(lower.first_write) == ("aic_call1", "mad", 0)
    assert native(lower.last_write) == ("aic_call2", "mad_first", 7)
    assert native(upper.first_write) == ("aic_call1", "mad", 1)
    assert native(upper.last_write) == ("aic_call1", "mad", 7)
    assert len(lower.pending_reads) == len(upper.pending_reads) == 1
    assert native(lower.pending_reads[0]) == ("aic_call3", "fix", 0)
    assert native(upper.pending_reads[0]) == ("aic_call1", "fix", 7)

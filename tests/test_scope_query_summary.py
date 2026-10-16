"""Query reuse against real Capture/View/Task byte events, without summaries."""

from dataclasses import dataclass
from itertools import product

import pytest

from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_index_frontend import evaluate as evaluate_ssa
from xdsl_npu import language as N
from xdsl_npu.ir import CaptureOp, CoreIdOp, ScopeGuardOp, TaskOp, ViewOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.region_summary import ReadSpan
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_summary import summarize_queries
from xdsl_npu.scope_sync import _scope_access

CARRIED = """
def carried(State:N.Tensor((GROUPS,8),N.f32,access="read_write"),
            O:N.Tensor((total_queries,8),N.f32,access="write"),
            total_queries:N.int64,kv_tiles:N.int64(minimum=1),
            GROUPS:N.constexpr=2):
    with N.Kernel(GROUPS) as group:
        x=N.alloc_ub((8,),N.f32,name="x")
        for query in N.serial(N.ceildiv(N.max(0,total_queries-group),GROUPS)):
            flat=group+query*GROUPS
            for kv in N.Pipelined(kv_tiles,stages=1):
                for inner in N.Pipelined(1,stages=3):
                    with N.Stage(0,name="load"):
                        N.copy(State[group,:],x)
                    with N.Stage(1,name="update"):
                        N.mul(x,0.5,out=x)
                    with N.Stage(2,name="store"):
                        N.copy(x,State[group,:])
                        if kv == kv_tiles - 1:
                            N.copy(x,O[flat,:])
"""


@pytest.fixture(scope="module")
def recurrence():
    program = extract_scopes(lower_source(scoped_query_recurrence))
    access = _scope_access(program)
    assert not access.lifetime_boundaries
    return program, access, summarize_queries(program, access)


@pytest.fixture(scope="module")
def carried():
    program = extract_scopes(lower_source(N.from_source(CARRIED, disjoint_external=True)))
    access = _scope_access(program)
    assert not access.lifetime_boundaries
    return program, access, summarize_queries(program, access)


@dataclass(frozen=True)
class NativeEvent:
    query: int
    actor: str
    call: str
    native: str
    inner: int
    kv: int


def _modes(task, inner):
    """Independent effects for the actual recipes exercised by these sources."""
    name = task.native.data
    if name in {"copy", "copy_rows", "nd2nz", "load2d_a", "load2d_b_nt", "fix", "muls"}:
        return "R", "W"
    if name == "fill":
        return ("W",)
    if name == "add":
        return "R", "R", "W"
    if name == "mad":
        return "R", "R", "W" if task.params.data[3].data else "RW"
    if name == "mad_first":
        return "R", "R", "W" if inner == 0 else "RW"
    raise AssertionError(f"unsupported actual oracle recipe {name}")


def _view(value, inner):
    """Read actual sparse rows and nested offsets, without Ref.segments."""
    last = value.owner
    assert isinstance(last, ViewOp)
    rows = 1 if last.rows is None else last.rows.data
    pitch = 0 if last.row_stride is None else last.row_stride.data
    start = 0
    while isinstance(value.owner, ViewOp):
        view = value.owner
        coordinate = inner // (1 if view.divisor is None else view.divisor.data)
        if view.modulus.data:
            coordinate %= view.modulus.data
        start += view.offset.data + view.stride.data * coordinate
        for term in view.index_terms or ():
            stride, divisor, modulus = (part.data for part in term)
            coordinate = inner // divisor
            start += stride * (coordinate % modulus if modulus else coordinate)
        assert not view.core_stride.data and not view.core_terms
        assert view.count_offset is None or view.count_offset.data == 0
        assert view.count_core_stride is None or view.count_core_stride.data == 0
        value = view.source
    assert isinstance(value.owner, CaptureOp)
    intervals = tuple(
        (start + row * pitch, start + row * pitch + last.nbytes.data) for row in range(rows)
    )
    assert all(0 <= lo < hi <= value.owner.nbytes.data for lo, hi in intervals)
    return value.owner, intervals


def _active(task, leaf, kv, count):
    parent = task.parent_op()
    while parent is not leaf.stream:
        if isinstance(parent, ScopeGuardOp):
            assert parent.kind.data in {"first", "last"}
            if kv != (0 if parent.kind.data == "first" else count - 1):
                return False
        parent = parent.parent_op()
    return True


def _templates(program):
    """Logical native order from actual IR, with both native and KV clocks."""
    result = []
    for uid, leaf in program.leaves.items():
        natives = [task for task in leaf.stream.walk() if isinstance(task, TaskOp)]
        for inner in range(leaf.stream.count.data):
            for index, task in enumerate(natives):
                order = (
                    leaf.call.logical_phase.data,
                    inner,
                    0 if task.logical_phase is None else task.logical_phase.data,
                    index,
                    leaf.actor,
                )
                memory = tuple(
                    (*_view(value, inner), mode)
                    for value, mode in zip(task.refs, _modes(task, inner), strict=True)
                )
                result.append((order, uid, leaf, task, inner, memory))
    return sorted(result, key=lambda entry: entry[0])


def _bindings(program, leaf, count, query, kv):
    parameters = {"total_queries": program.groups * 4, "kv_tiles": count}
    return {op.value: parameters[op.sym_name.data] for op in program.parameters} | {
        program.invocations[leaf.actor].op.body.block.args[0]: query,
        program.parents[leaf.actor].body.block.args[0]: kv,
        next(
            op.value
            for op in program.kernels[leaf.actor].body.block.ops
            if isinstance(op, CoreIdOp)
        ): 0,
    }


def _literal(program, count, *, queries):
    """Full byte scan across actual queries; no production frontier input."""
    state, edges = {}, set()
    templates = _templates(program)
    captures = {cap.op: cap.root for leaf in program.leaves.values() for cap in leaf.captures}
    for query in range(queries):
        for kv in range(count):
            for _, uid, leaf, task, inner, memory in templates:
                if not _active(task, leaf, kv, count):
                    continue
                event = NativeEvent(query, leaf.actor, uid, task.uid.data, inner, kv)
                bindings = _bindings(program, leaf, count, query, kv)
                touched = {}
                for capture, intervals, mode in memory:
                    base = evaluate_ssa(capture.offset, bindings)
                    root = captures[capture]
                    for lo, hi in intervals:
                        for byte in range(base + lo, base + hi):
                            touched.setdefault((root.ref, byte), set()).update(mode)
                for key, modes in touched.items():
                    writer, pending = state.setdefault(key, (None, set()))

                    def add(hazard, source, *, key=key, destination=event):
                        if source is not None and source.query == 0 < destination.query:
                            edges.add((*key, hazard, source, destination))

                    if "R" in modes:
                        assert writer is not None or key[0].owner.initialized.data
                        add("RAW", writer)
                        pending.add(event)
                    if "W" in modes:
                        add("WAW", writer)
                        for reader in pending:
                            add("WAR", reader)
                        writer, pending = event, set()
                    state[key] = writer, pending
    return edges


def _event(endpoint, program, access, kv):
    uid, site = access.endpoints[endpoint.access.task]
    leaf = program.leaves[uid]
    inner = site.position.evaluate(leaf.loop.constant_count)
    return NativeEvent(int(endpoint.region), leaf.actor, uid, site.task[1], inner, kv)


def _events(endpoint, program, access, count):
    at = endpoint.access
    local = count if endpoint.constant_count is None else endpoint.constant_count
    if isinstance(at, ReadSpan):
        positions = range(at.first.evaluate(local), at.last.evaluate(local) + 1, at.step)
    else:
        positions = (at.position.evaluate(local),)
    return tuple(_event(endpoint, program, access, kv) for kv in positions)


def _covers_actual_bytes(program, event, count, root, mode, start, stop):
    leaf = program.leaves[event.call]
    task = next(
        task
        for task in leaf.stream.walk()
        if isinstance(task, TaskOp) and task.uid.data == event.native
    )
    assert _active(task, leaf, event.kv, count)
    bindings = _bindings(program, leaf, count, event.query, event.kv)
    for value, actual in zip(task.refs, _modes(task, event.inner), strict=True):
        if mode not in actual:
            continue
        cap, intervals = _view(value, event.inner)
        if cap.source is not root:
            continue
        base = evaluate_ssa(cap.offset, bindings)
        if any(base + lo <= start < stop <= base + hi for lo, hi in intervals):
            return True
    return False


def _expanded(summary, program, access, count):
    result = set()
    for obligation in summary.obligations:
        if not obligation.domain.contains(count):
            continue
        assert obligation.source.region == "0"
        for source, target in product(
            _events(obligation.source, program, access, count),
            _events(obligation.destination, program, access, count),
        ):
            assert 0 <= source.kv < count and 0 <= target.kv < count
            for byte in range(obligation.start, obligation.stop):
                result.add((obligation.storage, byte, obligation.hazard, source, target))
    return result


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_recurrence_query_zero_byte_hazards_equal_actual_native_scan(recurrence, count):
    program, access, summary = recurrence
    assert len(summary.parts) == 3
    assert _expanded(summary, program, access, count) == _literal(program, count, queries=3)


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_readwrite_gm_query_raw_war_waw_equal_actual_native_scan(carried, count):
    program, access, summary = carried
    actual = _expanded(summary, program, access, count)
    assert actual == _literal(program, count, queries=2)
    if count:
        assert {edge[2] for edge in actual} == {"RAW", "WAR", "WAW"}
        assert any(edge[0].owner.sym_name.data == "State" and edge[2] == "RAW" for edge in actual)


def test_one_kv_ring_leaves_the_other_slot_untouched_until_query_two(recurrence):
    program, access, summary = recurrence
    score = [
        edge
        for edge in _expanded(summary, program, access, 1)
        if edge[0].owner.sym_name.data == "Score"
    ]
    assert score and {target.query for *_, target in score} == {2}
    assert {edge[2] for edge in score} == {"WAR", "WAW"}


@pytest.mark.parametrize("count", (1, 2, 3, 7))
def test_ring_phase_and_sparse_native_columns_retain_both_vector_actor_readers(recurrence, count):
    program, access, summary = recurrence
    score = [
        edge
        for edge in _expanded(summary, program, access, count)
        if edge[0].owner.sym_name.data == "Score"
    ]
    assert {source.actor for _, _, hazard, source, _ in score if hazard == "WAR"} == {
        "aiv0",
        "aiv1",
    }
    for _, byte, hazard, source, target in score:
        assert target.actor == "aic"
        column = byte % 128 // 64
        assert target.inner == column
        if hazard == "WAW":
            assert source.inner == column
        else:
            lane = 0 if source.actor == "aiv0" else 1
            row = byte % 2048 // 128
            assert lane * 8 + source.inner * 4 <= row < lane * 8 + source.inner * 4 + 4
        assert (source.query * count + source.kv) % 2 == (target.query * count + target.kv) % 2


def test_last_only_mte3_state_reader_precedes_next_query_first_fill_on_each_aiv(recurrence):
    program, access, summary = recurrence
    for count in (1, 2, 3, 7):
        edges = [
            edge
            for edge in _expanded(summary, program, access, count)
            if edge[0].owner.sym_name.data == "state" and edge[2] == "WAR"
        ]
        assert {source.actor for _, _, _, source, _ in edges} == {"aiv0", "aiv1"}
        for _, _, _, source, target in edges:
            assert source.actor == target.actor
            assert source.query == 0 and source.kv == count - 1
            assert target.query == 1 and target.kv == 0
            assert source.native.endswith("store_4_copy")
            assert target.native.endswith("update_1_fill")
            assert source.inner == target.inner


def test_readonly_inputs_and_disjoint_output_frames_have_no_cross_query_hazard(recurrence):
    program, access, summary = recurrence
    excluded = {
        cap.root.ref
        for leaf in program.leaves.values()
        for cap in leaf.captures
        if cap.root.sym_name.data in {"Q", "K", "O"}
    }
    assert not any(obligation.storage in excluded for obligation in summary.obligations)
    for count in (1, 3):
        assert not any(edge[0] in excluded for edge in _literal(program, count, queries=3))
    for root, _, _, source, target in _expanded(summary, program, access, 3):
        if not root.owner.external.data:
            assert source.actor == target.actor


def test_large_symbolic_query_summary_does_not_expand_kv_or_query_counts(recurrence, monkeypatch):
    import xdsl_npu.region_sequence as sequence
    import xdsl_npu.scope_boundary_summary as boundary
    import xdsl_npu.scope_guarded_access as guarded
    import xdsl_npu.scope_query_summary as query

    program, access, original = recurrence
    before = dump(program.module)

    def static_range(*args):
        result = range(*args)
        assert len(result) <= 64, "expanded a runtime KV/query count"
        return result

    for module in (sequence, boundary, guarded, query):
        monkeypatch.setattr(module, "range", static_range, raising=False)
    rebuilt = query.summarize_queries(program, access)
    assert rebuilt == original
    assert dump(program.module) == before
    for count in (3,):
        assert _expanded(rebuilt, program, access, count) == _literal(program, count, queries=3)
    for count in (10**12, (1 << 60) + 1):
        active = [
            obligation for obligation in rebuilt.obligations if obligation.domain.contains(count)
        ]
        assert active and {obligation.destination.region for obligation in active} == {"1"}
        for obligation in active:
            for endpoint, mode in (
                (obligation.source, "R" if obligation.hazard == "WAR" else "W"),
                (obligation.destination, "R" if obligation.hazard == "RAW" else "W"),
            ):
                at = endpoint.access
                first, last = (
                    (at.first, at.last) if isinstance(at, ReadSpan) else (at.position, at.position)
                )
                lo, hi = first.evaluate(count), last.evaluate(count)
                assert 0 <= lo <= hi < count
                samples = {lo, hi}
                if isinstance(at, ReadSpan) and lo < hi:
                    assert (hi - lo) % at.step == 0
                    samples.update((lo + at.step, hi - at.step))
                for kv in samples:
                    assert _covers_actual_bytes(
                        program,
                        _event(endpoint, program, access, kv),
                        count,
                        obligation.storage,
                        mode,
                        obligation.start,
                        obligation.stop,
                    )

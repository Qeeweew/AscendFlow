"""Shared scratch needs physical lifetime order and fixed logical RAW bindings."""

from itertools import pairwise

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.fixtures.nested import shared_l0c_qk_pv
from xdsl_npu import language as N
from xdsl_npu.compilation import compile
from xdsl_npu.ir import CallOp, TaskOp, ViewOp
from xdsl_npu.model import CompileError
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes


@pytest.fixture(scope="module")
def shared():
    source = shared_l0c_qk_pv.to_ir()
    scope = extract_scopes(lower_scopes(source.body.block.first_op))
    access = analyze_scope_access(scope)
    roles = dict(zip(scope.leaves, ("QK", "PV", "FIX"), strict=True))
    return scope, access, roles


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


def test_current_compile_rejects_shared_c0_completion_cycle():
    with pytest.raises(
        CompileError,
        match="^symbolic fixed schedule has a completion cycle or non-well-founded carry$",
    ):
        compile(shared_l0c_qk_pv, timeout_s=25)


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

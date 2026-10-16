"""Collected lexical work drives admitted AST leaves without changing output."""

import pytest
from xdsl.dialects.builtin import IntAttr

from examples.gemm.batched import batched_matmul
from examples.gemm.hierarchical import hierarchical_gemm
from examples.state.grouped import grouped_state_sum
from examples.state.recurrence import state_sum
from examples.vector.scale import scale
from tests.fixtures.cv import cv_source
from tests.fixtures.nested import nested_copy
from xdsl_npu import language as N
from xdsl_npu import lowering, source_scopes
from xdsl_npu.compilation_facts import PhysicalFacts
from xdsl_npu.ir import CallOp, StreamOp, TaskOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.source_ir import CopyOp, ElementOp, MatmulOp


def test_actual_ast_prefix_loop_suffix_use_one_scope_collection_and_no_legacy_walker(monkeypatch):
    source = state_sum.specialize(B=16, GROUPS=1).to_ir()
    collect = source_scopes.collect_source_scopes
    emit = lowering._Actor.emit
    collected, leaves = [], []

    def capture_scope(program):
        scope = collect(program)
        collected.append(scope)
        return scope

    def capture_records(actor):
        leaves.append(tuple(actor.records))
        return emit(actor)

    monkeypatch.setattr(source_scopes, "collect_source_scopes", capture_scope)
    monkeypatch.setattr(lowering._Actor, "emit", capture_records)
    before = dump(source)
    lower_source(source)
    assert dump(source) == before
    assert len(collected) == 1 and len(leaves) == 3
    assert [record.op for records in leaves for record in records] == [
        op for op in source.walk() if isinstance(op, (CopyOp, ElementOp))
    ]
    assert all(record.context is not None for records in leaves for record in records)
    assert leaves[0][0].context.clock is leaves[-1][0].context.clock is None
    assert len(leaves[1][0].context.clocks) == 1
    assert leaves[1][0].context.clock.coordinate is collected[0].items[1].clock.coordinate
    assert not hasattr(lowering, "_legacy_work")


@pytest.mark.parametrize("case", ["vector", "cube", "serial", "grouped", "mixed", "nested"])
def test_source_metadata_does_not_select_a_different_native_lowering(case):
    source = (
        scale.to_ir()
        if case == "vector"
        else batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1).to_ir()
        if case == "cube"
        else hierarchical_gemm.specialize(M=16, N=16, BK=16, GROUPS=1).to_ir()
        if case == "serial"
        else grouped_state_sum.specialize(B=16, GROUPS=1).to_ir()
        if case == "grouped"
        else cv_source(2)
        if case == "mixed"
        else nested_copy.to_ir()
    )
    imported = source.clone()
    imported.body.block.first_op.attributes.pop("frontend")
    assert dump(lower_source(source)) == dump(lower_source(imported))


@pytest.mark.parametrize("count", (0, 1, 1_000_000))
def test_imported_source_retains_fixed_children_without_a_frontend_tag(count):
    source = N.from_source(f"""
        def nested(n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for t in N.Pipelined(n, stages=1):
                    for part in N.Pipelined({count}, stages=1):
                        N.fill(x[part % 2, :], 0.0)
    """).to_ir()
    source.body.block.first_op.attributes.pop("frontend")
    physical = lower_source(source)
    (call,) = [op for op in physical.walk() if isinstance(op, CallOp)]
    (child,) = [op for op in call.body.block.ops if isinstance(op, StreamOp)]
    assert child.count.data == count
    assert sum(isinstance(op, TaskOp) for op in child.walk()) == 1


def test_unmarked_source_keeps_the_l1_machine_recipe():
    source = batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1).to_ir()
    source.body.block.first_op.attributes.pop("frontend")
    matmul = next(op for op in source.walk() if isinstance(op, MatmulOp))
    a, b = [
        op
        for op in source.walk()
        if isinstance(op, CopyOp) and op.destination.type.memory.data == "L1"
    ]
    matmul.operands = (a.destination, b.destination, *matmul.operands[2:4], a.value, b.value)
    matmul.attributes["transpose_b"] = IntAttr(1)
    physical = lower_source(source)
    natives = [op.native.data for op in physical.walk() if isinstance(op, TaskOp)]
    assert natives.count("load2d_a") == natives.count("load2d_b_nt") == 2
    assert natives.count("mad") == 1


@pytest.mark.parametrize("organization", ("loop", "sequence", "nested"))
def test_source_value_checks_handoff_the_same_native_dependencies(organization, monkeypatch):
    from xdsl_npu import loop_model

    definition = N.from_source("""
        def values(X: N.Tensor((n, 64), N.f32), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                y = N.alloc_ub((64,))
                for t in N.Pipelined(n, stages=1):
                    value = N.copy(X[t, :], x)
                    N.mul(value, 2.0, out=y)
    """)
    if organization == "nested":
        definition = nested_copy
    elif organization == "sequence":
        definition = N.from_source(
            """
            def sequence(X: N.Tensor((n, 64), N.f32),
                         Y: N.Tensor((64,), N.f32, access="write"), n: N.int64):
                with N.Kernel():
                    x = N.alloc_ub((64,))
                    y = N.alloc_ub((64,))
                    N.fill(y, 0.0)
                    for t in N.Pipelined(n, stages=1):
                        value = N.copy(X[t, :], x)
                        N.mul(value, 2.0, out=y)
                    N.copy(y, Y)
        """,
            disjoint_external=True,
        )
    dependencies = loop_model.loop_dependencies
    established = []

    def record(loop, *, live_in=()):
        result = dependencies(loop, live_in=live_in)
        established.append((loop, frozenset(live_in), result))
        return result

    monkeypatch.setattr(loop_model, "loop_dependencies", record)
    facts = PhysicalFacts()
    physical = lower_source(definition.to_ir(), _facts=facts)
    assert established
    builds = facts.statistics()["dependency_builds"]

    def no_rebuild(*args, **kwargs):
        pytest.fail("unchanged source effects were rebuilt instead of handed to the planner")

    monkeypatch.setattr(loop_model, "loop_dependencies", no_rebuild)
    for loop, live_in, expected in established:
        assert facts.loop_dependencies(loop, live_in=live_in) is expected
    assert facts.statistics()["dependency_builds"] == builds
    assert physical is not None


def test_source_value_handoff_keeps_stale_borrow_rejection():
    source = N.from_source("""
        def stale(X: N.Tensor((n, 64), N.f32), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                y = N.alloc_ub((64,))
                for task in N.Pipelined(n, stages=1):
                    old = N.copy(X[task, :], x)
                    N.mul(old, 2.0, out=x)
                    N.add(old, 1.0, out=y)
    """).to_ir()
    with pytest.raises(CompileError, match="overwritten"):
        lower_source(source, _facts=PhysicalFacts())


@pytest.mark.parametrize("marked", (False, True))
def test_effect_free_invalid_guard_is_still_refused_before_native_construction(monkeypatch, marked):
    source = N.from_source("""
        def empty_guard(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(n,stages=1):
                    if task == 0:
                        pass
                    N.fill(x,0.0)
    """).to_ir()
    if not marked:
        source.body.block.first_op.attributes.pop("frontend")

    def no_native(*args, **kwargs):
        raise AssertionError("unsupported guard reached native construction")

    monkeypatch.setattr(lowering._Actor, "emit", no_native)
    with pytest.raises(CompileError, match="generation == its pipeline bound - 1"):
        lower_source(source)

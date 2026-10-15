"""Collected lexical work drives admitted AST leaves without changing output."""

import pytest

from examples.gemm.batched import batched_matmul
from examples.gemm.hierarchical import hierarchical_gemm
from examples.state.grouped import grouped_state_sum
from examples.state.recurrence import state_sum
from examples.vector.scale import scale
from tests.fixtures.cv import cv_source
from xdsl_npu import language as N
from xdsl_npu import lowering, region_lowering
from xdsl_npu.ir import dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.source_ir import CopyOp, ElementOp


def test_actual_ast_prefix_loop_suffix_use_one_scope_collection_and_no_legacy_walker(monkeypatch):
    source = state_sum.specialize(B=16, GROUPS=1).to_ir()
    collect = region_lowering.collect_source_scopes
    emit = lowering._Actor.emit
    collected, leaves = [], []

    def capture_scope(program):
        scope = collect(program)
        collected.append(scope)
        return scope

    def no_legacy(*args, **kwargs):
        raise AssertionError("AST lowering entered the legacy source walker")

    def capture_records(actor):
        leaves.append(tuple(actor.records))
        return emit(actor)

    monkeypatch.setattr(region_lowering, "collect_source_scopes", capture_scope)
    monkeypatch.setattr(lowering, "_legacy_work", no_legacy)
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


@pytest.mark.parametrize("case", ["vector", "cube", "serial", "grouped", "mixed"])
def test_scope_ast_adapter_preserves_existing_physical_ir_exactly(case):
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
    )
    legacy = source.clone()
    legacy.body.block.first_op.attributes.pop("frontend")
    assert dump(lower_source(source)) == dump(lower_source(legacy))


def test_effect_free_invalid_guard_is_still_refused_before_native_construction(monkeypatch):
    source = N.from_source("""
        def empty_guard(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(n,stages=1):
                    if task == 0:
                        pass
                    N.fill(x,0.0)
    """).to_ir()

    def no_native(*args, **kwargs):
        raise AssertionError("unsupported guard reached native construction")

    monkeypatch.setattr(lowering._Actor, "emit", no_native)
    with pytest.raises(CompileError, match="generation == its pipeline bound - 1"):
        lower_source(source)

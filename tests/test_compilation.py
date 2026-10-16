from dataclasses import FrozenInstanceError

import pytest
from xdsl.dialects.builtin import FloatAttr, f32

from tests.fixtures.cv import cv_source, vector_source
from tests.report_contract import semantic_report
from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.compilation import Compilation, compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp, TaskOp
from xdsl_npu.lowering import lower_source


@pytest.mark.parametrize("factory", [vector_source, cv_source])
def test_frozen_compile_matches_independent_public_physical_path(factory, tmp_path):
    source = factory(2)
    result = compile(source)
    physical = lower_source(source)
    plan = autosync(physical)
    assert result.asc == emit_asc(plan)
    assert semantic_report(result.report) == semantic_report(verify_plan(plan))
    assert result.report["verification_mode"] == "normal"
    assert result.schedule
    assert result.timings["total_s"] > 0
    result.write(tmp_path)
    assert not (tmp_path / "witnesses.json").exists()
    assert (tmp_path / "after.xdsl").read_text() == result.plan_ir
    assert (tmp_path / "source.xdsl").read_text() == result.source_ir


def test_cached_code_owns_no_externally_mutable_ir_or_verification_flag():
    physical = lower_source(vector_source(2))
    result = Compilation(physical)
    expected = result.asc
    compute = next(
        op for op in physical.walk() if isinstance(op, TaskOp) and op.native.data == "muls"
    )
    compute.attributes["scalar"] = FloatAttr(3.0, f32)
    assert result.asc == expected
    assert compile(physical).asc != expected
    report = result.report
    report["channels"] = -1
    assert result.report["channels"] >= 0
    with pytest.raises(FrozenInstanceError):
        result.asc = "unsafe"
    with pytest.raises(TypeError):
        Compilation(physical, verified=True)


def test_an_edited_exported_plan_is_rechecked_and_cannot_reuse_the_artifact_proof():
    result = compile(cv_source(1))
    plan = result.plan
    channel = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.kind.data == "cross"
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError):
        emit_asc(plan)
    with pytest.raises(CompileError):
        compile(plan, synchronize=False)
    # The export is a copy; the frozen result still has its original complete plan.
    assert emit_asc(result.plan) == result.asc


def test_expert_synchronization_has_the_same_final_check():
    plan = autosync(lower_source(vector_source(2)))
    result = compile(plan, synchronize=False, host_launcher=False)
    assert result.asc == emit_asc(plan, host_launcher=False)
    with pytest.raises(CompileError, match="missing"):
        compile(lower_source(vector_source(2)), synchronize=False)


def test_owned_normal_path_reuses_facts_and_deep_is_explicit(monkeypatch):
    import xdsl_npu.compilation as pipeline
    import xdsl_npu.loop_sync as leaf
    from examples.vector.scale import scale

    original = pipeline.verify_plan
    checks = []

    def check(*args, **kwargs):
        checks.append(True)
        return original(*args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("normal compilation rebuilt an independent leaf proof")

    monkeypatch.setattr(pipeline, "verify_plan", check)
    with monkeypatch.context() as normal_context:
        normal_context.setattr(leaf, "_verify_leaf", unexpected)
        normal = compile(scale)
    assert not checks
    assert normal.timings["final_check_s"] == normal.timings["debug_check_s"] == 0
    assert "npu.loop_channel" not in normal.before_ir
    assert "npu.loop_channel" in normal.plan_ir
    deep = compile(scale, debug_verify=True)
    assert len(checks) == 1
    assert normal.asc == deep.asc
    assert semantic_report(normal.report) == semantic_report(deep.report)
    assert deep.timings["debug_check_s"] > 0

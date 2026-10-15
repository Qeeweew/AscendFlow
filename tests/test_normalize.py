"""Typed reduction, Scalar retirement and expert issue-order regressions."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr

from examples.normalization.rms_norm import rms_normalize
from xdsl_npu import compile, verify_plan
from xdsl_npu.frontend import from_source
from xdsl_npu.ir import LoopChannelOp, LoopSyncOp, TaskOp
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.model import CompileError
from xdsl_npu.source_ir import PipelineOp, ReduceOp, ScalarReadOp


@pytest.fixture(scope="module")
def artifact():
    return compile(rms_normalize)


def test_reduction_keeps_numeric_order_and_exact_scratch_footprint(artifact):
    source = rms_normalize.to_ir()
    reduction = next(op for op in source.walk() if isinstance(op, ReduceOp))
    assert reduction.order.data == "r64_then_native_merge"
    assert tuple(n.data for n in reduction.scratch.type.shape) == (64,)
    loop = extract_loop(artifact.plan)
    tasks = loop.actors["aiv0"].program.tasks
    assert [task.native for task in tasks] == [
        "copy",
        "mul",
        "reduce_sum64",
        "merge_sum",
        "muls",
        "adds",
        "sqrt",
        "fill",
        "div",
        "scalar_load",
        "scale_value",
        "copy",
    ]
    merge = next(task for task in tasks if task.native == "merge_sum")
    capture = next(task for task in tasks if task.native == "scalar_load")
    assert merge.refs[-1].nbytes == capture.refs[0].nbytes == 4
    assert merge.refs[0].nbytes == 256
    assert artifact.report["scalar_capture_queries"] == artifact.report["protocol_period"]
    assert "rsqrt(" not in artifact.asc


def test_statistics_reuse_depends_on_scalar_read_not_normalization_completion(artifact):
    loop = extract_loop(artifact.plan)
    tasks = {task.uid: task for task in loop.actors["aiv0"].program.tasks}
    wars = [
        dep
        for dep in loop_dependencies(loop)
        if dep.hazard == "WAR" and dep.storage.endswith("statistics") and dep.distance == 1
    ]
    assert wars and all(tasks[dep.source[1]].native == "scalar_load" for dep in wars)
    scalar = next(task for task in tasks.values() if task.native == "scalar_load")
    scale = next(task for task in tasks.values() if task.native == "scale_value")
    assert scale.scalar_source == scalar.uid
    assert all(ref.storage.name != "statistics" for ref in scale.refs)


def test_expert_store_dispatch_precedes_current_scalar_gate(artifact):
    code = artifact.asc
    assert (
        code.index("// aiv0/store_") < code.index("// aiv0/load_") < code.index("// aiv0/compute_")
    )
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert code.count("asc_copy_gm2ub(") == code.count("asc_copy_ub2gm(") == 1


def test_capture_cannot_move_after_its_consumer(artifact):
    plan = artifact.plan
    tasks = [op for op in plan.walk() if isinstance(op, TaskOp)]
    capture = next(op for op in tasks if op.native.data == "scalar_load")
    consume = next(op for op in tasks if op.native.data == "scale_value")
    capture.attributes["issue_order"] = IntAttr(consume.issue_order.data + 1)
    with pytest.raises(CompileError, match="Scalar capture must precede"):
        verify_plan(plan)


def test_capture_cannot_silently_read_another_generation(artifact):
    plan = artifact.plan
    consume = next(
        op for op in plan.walk() if isinstance(op, TaskOp) and op.native.data == "scale_value"
    )
    consume.attributes["stage"] = IntAttr(2)
    with pytest.raises(CompileError, match="delayed captures require"):
        verify_plan(plan)


def test_scalar_use_def_does_not_replace_vector_completion(artifact):
    plan = artifact.plan
    scalar = next(
        op for op in plan.walk() if isinstance(op, TaskOp) and op.native.data == "scalar_load"
    )
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopChannelOp) and scalar.uid.data in str(op.destinations)
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing loop"):
        verify_plan(plan)


@pytest.mark.parametrize(
    "labels", [(), ("load", "compute"), ("load", "compute", "store", "unknown")]
)
def test_expert_policy_cannot_silently_drop_stage_constraints(labels):
    source = rms_normalize.to_ir()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    pipeline.attributes["issue_order"] = DictionaryAttr(
        {"vector": ArrayAttr(StringAttr(label) for label in labels)}
    )
    with pytest.raises(CompileError, match="every active Stage"):
        compile(source)


def test_reduction_scratch_alias_cannot_be_hidden_by_sync():
    source = rms_normalize.source.replace("scratch=partial", "scratch=y[slot, :64]")
    with pytest.raises(CompileError, match="shifted in-place Vector alias"):
        compile(from_source(source, disjoint_external=True))


def test_scalar_read_is_an_effectful_source_op():
    source = rms_normalize.to_ir()
    assert len([op for op in source.walk() if isinstance(op, ScalarReadOp)]) == 1


def test_runtime_work_counts_real_invocation_without_expanding_rows(artifact):
    from xdsl_npu.performance import runtime_work

    for rows in (0, 1, 256, 1 << 20):
        work = runtime_work(artifact.plan, {"rows": rows})
        assert work["transfer_requested_bytes"]["GM->UB"] == 32 * rows * 4096 * 4
        assert work["transfer_requested_bytes"]["UB->GM"] == 32 * rows * 4096 * 4
        assert work["native_invocations"]["scalar_load"] == 32 * rows
        assert work["terminal_drains"] == 32


def test_cv_actor_issue_policies_keep_the_shared_multistage_model():
    from examples.legacy.multistage import cv_source
    from xdsl_npu import language as N

    source = cv_source(N.dynamic("tasks"))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    pipeline.attributes["issue_order"] = DictionaryAttr(
        {
            "cube": ArrayAttr(StringAttr(name) for name in ("qk", "pv")),
            "vector": ArrayAttr(StringAttr(name) for name in ("output", "vector")),
        }
    )
    result = compile(source)
    assert result.asc.index("// aic/qk_") < result.asc.index("// aic/pv_")
    assert result.asc.index("// aiv0/output_") < result.asc.index("// aiv0/vector_")
    assert result.report["cross_counter_ids"]

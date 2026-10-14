import pytest
from multistage import cv_source, vector_source
from xdsl.utils.exceptions import VerifyException

from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu import language as N
from xdsl_npu.check import bounded_check
from xdsl_npu.ir import CrossChannelOp, CrossSyncOp, dump, parse
from xdsl_npu.lowering import lower_source
from xdsl_npu.mixed import extract_mixed
from xdsl_npu.model import CompileError
from xdsl_npu.source_ir import AllocationOp, DomainOp, PipelineOp, StageOp, ViewOp


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_vector_stages_trace_once_and_have_one_task_coordinate(count):
    source = parse(dump(vector_source(count)))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert pipeline.count.data == count
    assert [op.offset.data for op in stages] == [0, 1, 2]
    assert len(pipeline.body.block.args) == 1
    assert {
        op.versions.data
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "UB"
    } == {2}


def test_cv_regions_share_stage_one_without_merging_their_semantics():
    source = parse(dump(cv_source()))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert [op.offset.data for op in stages] == [0, 1, 1, 2]
    assert [op.label.data for op in stages] == ["qk", "vector", "pv", "output"]
    assert [op.kind.data for op in source.walk() if isinstance(op, DomainOp)] == [
        "cube",
        "vector",
        "cube",
        "vector",
    ]
    assert all(
        pipeline.body.block.args[0] in op.indices
        for op in source.walk()
        if isinstance(op, ViewOp)
        and op.source.owner.sym_name.data in {"scores", "probabilities", "pv"}
    )
    assert "npu.task" not in dump(source)


def test_dynamic_python_if_is_rejected_instead_of_selecting_one_branch():
    with pytest.raises(TypeError, match="Python if"):

        @N.prim_func
        def invalid(X: N.Tensor((2, 64), N.f32)):
            with N.Kernel():
                for task in N.Pipelined(2, stages=1):
                    if task == 0:
                        N.copy(X[task, :], N.alloc_ub((64,)))


def test_stage_depth_does_not_silently_change_buffer_versions():
    source = vector_source()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    from xdsl.dialects.builtin import IntAttr

    pipeline.attributes["stages"] = IntAttr(2)
    with pytest.raises(VerifyException, match="stage horizon"):
        source.verify()


def test_typed_copy_requires_an_explicit_rounding_conversion():
    with pytest.raises(CompileError, match="explicit cast"):

        @N.prim_func
        def invalid(X: N.Tensor((1, 64), N.f32)):
            with N.Kernel():
                output = N.alloc_ub((64,), N.f16)
                for task in N.Pipelined(1, stages=1):
                    with N.Stage(0):
                        N.copy(X[task, :], output)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_source_vector_closes_through_physical_sync_and_asc(count):
    plan = autosync(lower_source(parse(dump(vector_source(count)))))
    assert verify_plan(plan)["tasks"] == 3 * count
    if count <= 2:
        assert bounded_check(plan).exhaustive
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    if count:
        assert "step - 2" in code


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_source_cv_preserves_shared_stage_and_actual_handoff(count):
    plan = autosync(lower_source(parse(dump(cv_source(count)))))
    report = verify_plan(plan)
    assert report["epochs"] == count
    mixed = extract_mixed(plan)
    tasks = {t.uid: t for t in mixed.programs["aic"].tasks}
    qk = next(t for t in tasks.values() if t.uid.startswith("qk_") and t.native == "mad")
    pv = next(t for t in tasks.values() if t.uid.startswith("pv_") and t.native == "mad")
    assert qk.stage == 0 and pv.stage == 1 and qk.logical_phase < pv.logical_phase
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 3
    if count:
        assert code.index("// qk_") < code.index("// pv_")
        assert set(report["cross_counter_ids"]) == {
            "scores_ready",
            "probabilities_ready",
            "pv_ready",
        }
        channel = next(
            op
            for op in plan.walk()
            if isinstance(op, CrossChannelOp) and op.sym_name.data == "probabilities_ready"
        )
        for op in list(plan.walk()):
            if isinstance(op, CrossSyncOp) and op.channel.data == channel.sym_name.data:
                op.parent.erase_op(op)
        channel.parent.erase_op(channel)
        with pytest.raises(CompileError, match="missing mixed (RAW|WAR)"):
            verify_plan(plan)


def test_a_logical_value_is_not_a_snapshot_of_overwritten_storage():
    @N.prim_func(disjoint_external=True)
    def stale(X: N.Tensor((1, 64), N.f32), Y: N.Tensor((1, 64), N.f32, access="write")):
        with N.Kernel():
            x, y = N.alloc_ub((64,)), N.alloc_ub((64,))
            for task in N.Pipelined(1, stages=1):
                with N.Stage(0):
                    old = N.copy(X[task, :], x)
                    N.mul(old, 2.0, out=x)
                    result = N.add(old, 1.0, out=y)
                    N.copy(result, Y[task, :])

    with pytest.raises(CompileError, match="was overwritten"):
        lower_source(stale)


def test_nested_pipeline_result_preserves_the_producer_descriptor_and_slot():
    @N.prim_func(disjoint_external=True)
    def nested(X: N.Tensor((2, 3, 64), N.f32), Y: N.Tensor((2, 64), N.f32, access="write")):
        with N.Kernel():
            x = N.alloc_ub((64,), versions=2, name="x")
            y = N.alloc_ub((64,), versions=2, name="y")
            for task in N.Pipelined(2, stages=2):
                with N.Stage(0, name="inner"):
                    for part in N.Pipelined(3, stages=2):
                        with N.Stage(0):
                            value = N.copy(X[task, part, :], x)
                        with N.Stage(1):
                            result = N.mul(value, 2.0, out=y)
                with N.Stage(1, name="store"):
                    N.copy(result, Y[task, :])

    from xdsl_npu.model import extract

    physical = lower_source(parse(dump(nested)))
    program = extract(physical)
    store = next(t for t in program.tasks if t.uid.startswith("store_"))
    # Inner final part=2 maps to slot 0 for EVERY outer task. Re-evaluating
    # default versions against the consumer's outer coordinate would read slot 1.
    assert [store.refs[0].address(g) for g in range(2)] == [0, 0]
    plan = autosync(physical)
    assert verify_plan(plan)["tasks"] == 14
    assert emit_asc(plan).count("asc_sync_pipe(PIPE_ALL)") == 1


def test_two_logical_values_may_share_one_root_with_disjoint_input_views():
    @N.prim_func(disjoint_external=True)
    def slices(X: N.Tensor((1, 128), N.f32), Y: N.Tensor((1, 64), N.f32, access="write")):
        with N.Kernel():
            x, y = N.alloc_ub((128,)), N.alloc_ub((64,))
            for task in N.Pipelined(1, stages=1):
                left = N.copy(X[task, :64], x[:64])
                right = N.copy(X[task, 64:], x[64:])
                result = N.add(left, right, out=y)
                N.copy(result, Y[task, :])

    assert verify_plan(autosync(lower_source(slices)))["tasks"] == 4


def test_source_axis_bounds_are_checked_before_flattening_addresses():
    with pytest.raises(CompileError, match="logical axis"):

        @N.prim_func
        def invalid(X: N.Tensor((2, 64), N.f32)):
            with N.Kernel():
                x = N.alloc_ub((64,))
                for task in N.Pipelined(1, stages=1):
                    # Byte addresses fit the tensor, but row 0's second axis does not.
                    N.copy(X[task, 64:128], x)


def test_region_yield_types_are_verified_after_source_roundtrip():
    from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

    from xdsl_npu.source_ir import TileType

    source = parse(dump(vector_source()))
    stage = next(op for op in source.walk() if isinstance(op, StageOp))
    stage.results[0]._type = TileType(ArrayAttr([IntAttr(32)]), StringAttr("f32"))
    with pytest.raises(VerifyException, match="yield types"):
        source.verify()


@pytest.mark.parametrize("count,segments", [(0, 3), (1, 1), (2, 3), (4, 3)])
def test_segmented_cv_uses_inner_prefetch_and_complete_outer_handoffs(count, segments):
    from multistage import segmented_cv_source

    physical = lower_source(parse(dump(segmented_cv_source(count, segments))))
    plan = autosync(physical)
    report = verify_plan(plan)
    mixed = extract_mixed(plan)
    aic = mixed.programs["aic"]
    qk = [t for t in aic.tasks if t.native == "mad" and t.stage == 0]
    pv = [t for t in aic.tasks if t.native == "mad" and t.stage == 1]
    assert len(qk) == len(pv) == segments
    assert [t.params[-1] for t in pv] == [1] + [0] * (segments - 1)
    assert report["epochs"] == count
    # Same CV stage horizon regardless of the inner software stage count.
    assert max(t.stage for p in mixed.programs.values() for t in p.tasks) == 2
    if segments > 1:
        prefetch = [t for t in aic.tasks if t.native == "nd2nz" and t.stage == 0][1:]
        assert prefetch[1].issue_order < qk[0].issue_order
    code = emit_asc(plan)
    if count:
        # ALL new QK contributions must dispatch before the old P-ready gate.
        gate = next(
            line
            for line in code.splitlines()
            if "asc_sync_block_wait" in line and "; // probabilities_ready:" in line
        )
        assert code.index("// " + qk[-1].uid) < code.index(gate)


def test_unimplemented_runtime_initialization_is_rejected_explicitly():
    @N.prim_func(disjoint_external=True)
    def invalid(A: N.Tensor((2, 16, 16), N.f16), B: N.Tensor((2, 16, 16), N.f16)):
        with N.Kernel(kind="cube"):
            a, b, c = N.alloc_l1((16, 16)), N.alloc_l1((16, 16)), N.alloc_l0c((16, 16))
            for task in N.Pipelined(2, stages=1):
                N.copy(A[task, :, :], a)
                N.copy(B[task, :, :], b)
                N.gemm(a, b, c, initialize=(task == 0))

    with pytest.raises(CompileError, match="runtime matmul initialization"):
        lower_source(invalid)


def test_all_segment_contributions_precede_ready_and_an_early_ready_is_rejected():
    from multistage import segmented_cv_source
    from xdsl_npu.ir import TaskOp
    from xdsl_npu.mixed import mixed_execution, mixed_graph
    from xdsl_npu.model import Instance

    plan = autosync(lower_source(segmented_cv_source(1, 3, 1)))
    mixed, channels, commands = mixed_execution(plan)
    graph, _, end, sends, *_ = mixed_graph(mixed, channels, commands)
    exports = [
        (i, t)
        for i, t in enumerate(mixed.programs["aic"].tasks)
        if t.native == "fix" and t.stage == 0
    ]
    assert len(exports) == 3
    assert all(
        graph.precedes(end[("aic", Instance(0, i))], sends[("aic", "scores_ready", 0)])
        for i, _ in exports
    )
    actor = mixed.actors["aic"]
    ready = next(
        op
        for op in actor.walk()
        if isinstance(op, CrossSyncOp)
        and op.action.data == "set"
        and op.channel.data == "scores_ready"
    )
    first = next(
        op for op in actor.walk() if isinstance(op, TaskOp) and op.uid.data == exports[0][1].uid
    )
    block = ready.parent
    block.detach_op(ready)
    block.insert_op_after(ready, first)
    with pytest.raises(CompileError, match="missing mixed RAW.*scores"):
        verify_plan(plan)


def test_incompatible_domain_is_rejected_without_silently_dropping_work():
    with pytest.raises(VerifyException, match="kernel kind"):

        @N.prim_func
        def invalid(X: N.Tensor((1, 64), N.f32)):
            with N.Kernel(kind="vector"):
                x = N.alloc_ub((64,))
                for task in N.Pipelined(1, stages=1):
                    with N.Cube():
                        N.copy(X[task, :], x)

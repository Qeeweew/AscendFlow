"""AST source semantics and physical mixed-protocol regression boundaries."""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from tests.fixtures.cv import cv_source, vector_source
from xdsl_npu import language as N
from xdsl_npu.check import bounded_check
from xdsl_npu.codegen import emit_asc
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import CrossChannelOp, CrossSyncOp
from xdsl_npu.dialects.source import AllocationOp, DomainOp, PipelineOp, StageOp, ViewOp
from xdsl_npu.dialects.types import TileType
from xdsl_npu.frontend import FrontendError
from xdsl_npu.lowering import lower_source
from xdsl_npu.mixed import extract_mixed
from xdsl_npu.sync import autosync, verify_plan


def diagnostic_physical(module):
    """Explicit finite diagnostic; ordinary AST compilation stays symbolic."""
    return lower_source(module, symbolic=False)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_ast_stages_retain_one_task_coordinate_and_explicit_slot_axes(count):
    source = parse(dump(vector_source(count)))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert pipeline.count.data == count
    assert [op.offset.data for op in stages] == [0, 1, 2]
    assert len(pipeline.body.block.args) == 1
    allocations = [
        op
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "UB"
    ]
    assert all(op.versions.data == 1 and op.ring.data == 0 for op in allocations)
    assert {tuple(n.data for n in op.ref.type.shape) for op in allocations} == {(2, 64)}


def test_ast_cv_regions_share_stage_one_without_merging_their_semantics():
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

    def uses_coordinate(value):
        return value is pipeline.body.block.args[0] or any(
            uses_coordinate(operand) for operand in getattr(value.owner, "operands", ())
        )

    assert all(
        any(uses_coordinate(origin) for origin in op.origins)
        for op in source.walk()
        if isinstance(op, ViewOp)
        and op.source.owner.sym_name.data in {"scores", "probabilities", "pv"}
    )
    assert "npu.task" not in dump(source)


def test_device_if_retains_both_sides_instead_of_selecting_a_python_branch():
    definition = N.from_source("""
        def branch(X:N.Tensor((2,64),N.f32)):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(2,stages=1):
                    if task == 0:
                        N.copy(X[task,:],x)
                    else:
                        N.fill(x,0.0)
    """)
    source = definition.to_ir()
    assert sum(isinstance(op, scf.IfOp) for op in source.walk()) == 1
    assert sum(isinstance(op, arith.CmpiOp) for op in source.walk()) == 1
    with pytest.raises(CompileError, match="structured scf.if synchronization"):
        lower_source(source)


def test_stage_depth_does_not_silently_change_explicit_buffer_slots():
    source = vector_source()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    pipeline.attributes["stages"] = IntAttr(2)
    with pytest.raises(VerifyException, match="stage horizon"):
        source.verify()


def test_typed_copy_requires_an_explicit_rounding_conversion():
    definition = N.from_source("""
        def invalid(X:N.Tensor((1,64),N.f32)):
            with N.Kernel():
                output=N.alloc_ub((64,),N.f16)
                for task in N.Pipelined(1,stages=1):
                    N.copy(X[task,:],output)
    """)
    with pytest.raises(FrontendError, match="explicit cast"):
        definition.to_ir()


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_ast_vector_physical_plan_closes_through_sync_and_asc(count):
    plan = autosync(diagnostic_physical(parse(dump(vector_source(count)))))
    assert verify_plan(plan)["tasks"] == 3 * count
    if count <= 2:
        assert bounded_check(plan).exhaustive
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    if count:
        assert "step - 2" in code


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_ast_local_cv_physical_plan_preserves_handoff_and_rejects_missing_ready(count):
    plan = autosync(diagnostic_physical(parse(dump(cv_source(count)))))
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
    definition = N.from_source(
        """
        def stale(X:N.Tensor((1,64),N.f32),Y:N.Tensor((1,64),N.f32,access="write")):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    old=N.copy(X[task,:],x)
                    N.mul(old,2.0,out=x)
                    result=N.add(old,1.0,out=y)
                    N.copy(result,Y[task,:])
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="was overwritten"):
        lower_source(definition.to_ir())


def test_nested_pipeline_keeps_child_clock_for_borrowed_tile():
    definition = N.from_source(
        """
        def nested(X:N.Tensor((2,3,64),N.f32),Y:N.Tensor((2,64),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((2,64),name="x")
                y=N.alloc_ub((2,64),name="y")
                for task in N.Pipelined(2,stages=2):
                    with N.Stage(0,name="inner"):
                        for part in N.Pipelined(3,stages=2):
                            with N.Stage(0):
                                value=N.copy(X[task,part,:],x[part%2,:])
                            with N.Stage(1):
                                result=N.mul(value,2.0,out=y[part%2,:])
                    with N.Stage(1,name="store"):
                        N.copy(y[0,:],Y[task,:])
        """,
        disjoint_external=True,
    )
    from xdsl_npu.dialects.physical import CallOp, StreamOp
    from xdsl_npu.dialects.physical import ViewOp as PhysicalViewOp

    physical = lower_source(definition.to_ir())
    calls = [op for op in physical.walk() if isinstance(op, CallOp)]
    assert len(calls) == 2
    child = next(op for op in calls[0].body.block.ops if isinstance(op, StreamOp))
    assert child.count.data == 3
    views = [op for op in child.walk() if isinstance(op, PhysicalViewOp)]
    assert views
    assert all(op.generation is child.body.block.args[0] for op in views)
    assert all(op.generation is not calls[0].generation for op in views)


def test_two_logical_values_may_share_one_root_with_disjoint_input_views():
    definition = N.from_source(
        """
        def slices(X:N.Tensor((1,128),N.f32),Y:N.Tensor((1,64),N.f32,access="write")):
            with N.Kernel():
                x,y=N.alloc_ub((128,)),N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    left=N.copy(X[task,:64],x[:64])
                    right=N.copy(X[task,64:],x[64:])
                    result=N.add(left,right,out=y)
                    N.copy(result,Y[task,:])
        """,
        disjoint_external=True,
    )
    assert verify_plan(autosync(diagnostic_physical(definition.to_ir())))["tasks"] == 4


def test_source_axis_bounds_are_checked_before_flattening_addresses():
    definition = N.from_source("""
        def invalid(X:N.Tensor((2,64),N.f32)):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    N.copy(X[task,64:128],x)
    """)
    with pytest.raises(FrontendError, match="logical axis"):
        definition.to_ir()


def test_region_yield_types_are_verified_after_source_roundtrip():
    source = parse(dump(vector_source()))
    stage = next(op for op in source.walk() if isinstance(op, StageOp))
    stage.results[0]._type = TileType(ArrayAttr([IntAttr(32)]), StringAttr("f32"))
    with pytest.raises(VerifyException, match="yield types"):
        source.verify()


@pytest.mark.parametrize("count,segments", [(0, 3), (1, 1), (2, 3), (4, 3)])
def test_internal_cv_fixture_keeps_prefetch_and_complete_outer_handoffs(count, segments):
    from tests.fixtures.cv import cv_program

    plan = autosync(parse(dump(cv_program(count=count, segments=segments, groups=2))))
    report = verify_plan(plan)
    mixed = extract_mixed(plan)
    aic = mixed.programs["aic"]
    qk = [t for t in aic.tasks if t.native == "mad" and t.stage == 0]
    pv = [t for t in aic.tasks if t.native == "mad" and t.stage == 1]
    assert len(qk) == len(pv) == segments
    assert [t.params[-1] for t in pv] == [1] + [0] * (segments - 1)
    assert report["epochs"] == count
    assert max(t.stage for p in mixed.programs.values() for t in p.tasks) == 2
    if segments > 1:
        prefetch = [t for t in aic.tasks if t.native == "nd2nz" and t.stage == 0][1:]
        assert prefetch[1].issue_order < qk[0].issue_order
    code = emit_asc(plan)
    if count:
        gate = next(
            line
            for line in code.splitlines()
            if "asc_sync_block_wait" in line and "; // probabilities_ready:" in line
        )
        assert code.index("// " + qk[-1].uid) < code.index(gate)


def test_internal_cv_fixture_rejects_ready_before_all_segment_contributions():
    from tests.fixtures.cv import cv_program
    from xdsl_npu.dialects.physical import TaskOp
    from xdsl_npu.mixed import mixed_execution, mixed_graph
    from xdsl_npu.model import Instance

    plan = autosync(cv_program(count=1, segments=3, groups=1))
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
    definition = N.from_source("""
        def invalid(X:N.Tensor((1,64),N.f32)):
            with N.Kernel(kind="vector"):
                x=N.alloc_ub((64,))
                for task in N.Pipelined(1,stages=1):
                    with N.Cube():
                        N.copy(X[task,:],x)
    """)
    with pytest.raises(VerifyException, match="kernel kind"):
        definition.to_ir()

"""Explicit block8 trees preserve exact native effects and synchronization."""

import pytest
from xdsl.utils.exceptions import VerifyException

from xdsl_npu import compile, emit_asc, verify_plan
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import LoopSyncOp, TaskOp
from xdsl_npu.dialects.source import ReduceOp
from xdsl_npu.loop_model import extract_loop


def definition(kind="sum", rows=8, columns=1024, workspace=1152):
    return N.from_source(
        f"""
        def block8(n: N.int64):
            with N.Kernel(kind="vector"):
                x = N.alloc_ub(({rows}, {columns}), N.f32)
                workspace = N.alloc_ub((2, {workspace}), N.f32)
                statistic = N.alloc_ub(({rows},), N.f32)
                for i in N.Pipelined(n, stages=1):
                    values = N.fill(x, 1.0)
                    reduced = N.reduce_{kind}(values, axis=1, out=statistic,
                        scratch=workspace[i % 2, :], order="block8_then_packed_merge")
                    N.mul(reduced, 2.0, out=statistic)
        """,
        disjoint_external=True,
    )


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize(
    "rows,columns,widths",
    [
        (1, 64, (8,)),
        (255, 64, (8,)),
        (3, 192, (24,)),
        (5, 512, (64,)),
        (8, 1024, (128, 16)),
        (3, 1536, (192, 24)),
        (1, 4096, (512, 64)),
    ],
)
def test_block8_tree_reaches_verified_asc_with_disjoint_exact_planes(kind, rows, columns, widths):
    words = rows * sum(widths)
    artifact = compile(definition(kind, rows, columns, words), timeout_s=10)
    source = parse(artifact.source_ir)
    reduction = next(op for op in source.walk() if isinstance(op, ReduceOp))
    assert tuple(n.data for n in reduction.scratch.type.shape) == (words,)
    loop = extract_loop(artifact.plan)
    tasks = loop.actors["aiv0"].program.tasks
    stages = [task for task in tasks if task.native == f"reduce_{kind}8"]
    merge = next(task for task in tasks if task.native == f"merge_{kind}_packed")
    assert len(stages) == len(widths)
    byte_offset = 0
    for level, (task, width) in enumerate(zip(stages, widths)):
        read, write = task.refs
        assert task.modes == ("R", "W")
        assert read.total_nbytes == rows * (columns if level == 0 else widths[level - 1]) * 4
        assert write.rows == rows and write.nbytes == width * 4
        assert write.row_stride == width * 4 and write.contiguous
        assert write.total_nbytes == rows * width * 4
        assert write.storage.nbytes == words * 2 * 4
        assert write.segments(0) == tuple(
            (byte_offset + row * width * 4, byte_offset + (row + 1) * width * 4)
            for row in range(rows)
        )
        assert write.segments(1) == tuple(
            (words * 4 + start, words * 4 + stop) for start, stop in write.segments(0)
        )
        if level:
            assert read == stages[level - 1].refs[-1]
            assert read.segments(0)[-1][1] <= write.segments(0)[0][0]
        byte_offset += rows * width * 4
    assert byte_offset == words * 4
    assert merge.refs[0] == stages[-1].refs[-1]
    assert merge.refs[-1].total_nbytes == rows * 4
    assert merge.params == (rows, widths[-1])
    native = f"asc_datablock_reduce_{kind}("
    assert artifact.asc.count(native) == len(widths)
    merge_native = (
        "asc_repeat_reduce_sum(" if kind == "sum" else "asc_repeat_reduce_max_only_value("
    )
    assert artifact.asc.count(merge_native) == 1
    calls = [line for line in artifact.asc.splitlines() if native in line]
    for task, call in zip(stages, calls):
        assert task.refs[0].storage.name in call and task.refs[-1].storage.name in call
        assert f"uint32_t({task.refs[0].total_nbytes // 4})" in call
    assert "asc_set_vector_mask(0, uint64_t(" + str((1 << widths[-1]) - 1) + "))" in artifact.asc
    assert verify_plan(parse(dump(artifact.plan)))["runtime_length"] == "n"


@pytest.mark.parametrize("columns", [576, 704, 960, 1088, 1984, 4032])
def test_block8_order_rejects_unaligned_derived_stages_without_fallback(columns):
    with pytest.raises(VerifyException, match="block8.*whole 64-element repeats"):
        definition(columns=columns).to_ir()


@pytest.mark.parametrize("words", [1144, 1160])
def test_block8_workspace_extent_must_be_exact(words):
    with pytest.raises(VerifyException, match="exact partial/statistic shapes"):
        definition(workspace=words).to_ir()


def test_block8_workspace_must_be_one_dimensional():
    source = definition().source.replace("scratch=workspace[i % 2, :]", "scratch=x[:, :144]")
    with pytest.raises(VerifyException, match="exact partial/statistic shapes"):
        N.from_source(source, disjoint_external=True).to_ir()


@pytest.mark.parametrize("destination", [False, True])
def test_block8_does_not_relax_native_in_place_overlap_rules(destination):
    source = definition(rows=1, workspace=144).source
    if destination:
        source = source.replace("out=statistic,", "out=workspace[i % 2, 128:129],")
    else:
        source = source.replace("scratch=workspace[i % 2, :]", "scratch=x[0, :144]")
    with pytest.raises(CompileError, match="in-place Vector (alias|overlap)"):
        compile(N.from_source(source, disjoint_external=True), timeout_s=10)


def test_block8_tracks_the_input_reader_and_final_value_producer():
    source = definition().source.replace(
        "reduced = N.reduce_sum", "N.fill(x, 0.0)\n            reduced = N.reduce_sum"
    )
    with pytest.raises(CompileError, match="overwritten before.*partial"):
        compile(N.from_source(source, disjoint_external=True), timeout_s=10)


def test_block8_stage_readiness_cannot_be_deleted_from_the_final_plan():
    artifact = compile(definition(), timeout_s=10)
    plan = artifact.plan
    second = next(
        op for op in plan.walk() if isinstance(op, TaskOp) and op.uid.data.endswith("_block8_1")
    )
    barrier = second.prev_op
    assert isinstance(barrier, LoopSyncOp) and barrier.action.data == "barrier"
    barrier.parent.erase_op(barrier)
    with pytest.raises(CompileError, match="missing loop RAW"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="missing loop RAW"):
        emit_asc(plan)

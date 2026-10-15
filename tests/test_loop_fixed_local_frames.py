"""Fixed affine local frames preserve real pitched rows and allocation bounds."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from xdsl_npu import language as N
from xdsl_npu.ir import BufferOp, RuntimeSizeOp, StreamOp, TaskOp, ViewOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError

SOURCE = """
def local_frames(
    X:N.Tensor((COUNT,ROWS,8),N.f32),
    O:N.Tensor((COUNT,ROWS,8),N.f32,access="write"),
):
    with N.Kernel():
        local=N.alloc_ub((COUNT,ROWS,16),N.f32,name="local")
        for tile in N.Pipelined(COUNT,stages=2):
            with N.Stage(0,name="load"):
                value=N.copy(X[tile,:,:],local[tile,:,:8])
            with N.Stage(1,name="store"):
                N.copy(value,O[tile,:,:])
"""


def _module(*, count=4, rows=2):
    source = N.from_source(SOURCE, constants={"COUNT": count, "ROWS": rows}, disjoint_external=True)
    return lower_source(source)


def _local_views(module):
    native_refs = {value for op in module.walk() if isinstance(op, TaskOp) for value in op.refs}
    return [
        op
        for op in module.walk()
        if isinstance(op, ViewOp)
        and isinstance(op.source.owner, BufferOp)
        and op.source.owner.sym_name.data == "local"
        and op.ref in native_refs
    ]


@pytest.mark.parametrize("count,rows", ((1, 2), (4, 2), (4, 3)))
def test_public_fixed_pipeline_accepts_positive_stride_and_pitched_local_rows(count, rows):
    module = _module(count=count, rows=rows)
    loop = extract_loop(module)
    frame = rows * 64
    assert loop.constant_count == count
    assert loop.frames["aiv0/local"] == frame
    assert len([op for op in module.walk() if isinstance(op, TaskOp)]) == 2
    refs = {
        ref
        for data in loop.actors.values()
        for task in data.program.tasks
        for ref in task.refs
        if ref.storage.name == "local"
    }
    assert refs
    for ref in refs:
        assert ref.segments(0) == tuple((row * 64, row * 64 + 32) for row in range(rows))
        assert ref.segments(count - 1) == tuple(
            ((count - 1) * frame + row * 64, (count - 1) * frame + row * 64 + 32)
            for row in range(rows)
        )


def test_actual_runtime_extent_cannot_reuse_the_fixed_local_frame_contract():
    source = N.from_source("""
def bounded_local(n:N.int64):
    with N.Kernel():
        local=N.alloc_ub((4,2,8),N.f32,name="local")
        for tile in N.Pipelined(4,stages=1):
            N.fill(local[tile,:,:],0.0)
""")
    # A public runtime source is rejected even earlier by logical axis bounds.
    # Change the real normalized clock to exercise physical extraction too.
    module = lower_source(source)
    stream = next(op for op in module.walk() if isinstance(op, StreamOp))
    size = next(op for op in module.walk() if isinstance(op, RuntimeSizeOp))
    del stream.attributes["count"]
    stream.operands = (size.value,)
    with pytest.raises(CompileError, match="local lifetime is unbounded; declare a ring"):
        extract_loop(module)


@pytest.mark.parametrize("stride", (0, 256))
def test_actual_views_of_one_local_root_must_share_the_same_positive_drift(stride):
    module = _module()
    views = _local_views(module)
    assert len(views) == 2
    assert all(tuple(term.data for term in op.index_terms.data[0]) == (128, 1, 0) for op in views)
    views[-1].attributes["index_terms"] = ArrayAttr(
        [ArrayAttr([IntAttr(stride), IntAttr(1), IntAttr(0)])]
    )
    with pytest.raises(CompileError, match="one uniform positive byte stride"):
        extract_loop(module)


def test_access_into_the_following_frame_is_rejected_inside_the_allocation():
    module = _module()
    stream = next(op for op in module.walk() if isinstance(op, StreamOp))
    # Three executed frames leave an entire spare frame in the allocation.
    # The shifted second DMA row is in bounds but belongs to the next task.
    stream.attributes["count"] = IntAttr(3)
    view = _local_views(module)[0]
    assert view.rows.data == 2 and view.row_stride.data == 64 and view.nbytes.data == 32
    assert view.source.owner.nbytes.data == 512
    view.attributes["offset"] = IntAttr(64)
    assert 2 * 128 + 64 + 64 + 32 < 512
    with pytest.raises(CompileError, match="access crosses its task frame"):
        extract_loop(module)


def test_final_real_frame_must_fit_the_actual_local_allocation():
    module = _module()
    root = _local_views(module)[0].source.owner
    root.attributes["nbytes"] = IntAttr(3 * 128)
    with pytest.raises(CompileError, match="affine static bound exceeds allocation"):
        extract_loop(module)

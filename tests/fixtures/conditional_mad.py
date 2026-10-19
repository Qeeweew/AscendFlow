"""Actual MAD source setup and admission, never an independent oracle."""

from textwrap import indent

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from ascendflow import language as N
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure


def _structure(body=None, *, ring=False):
    destination = "c0[k%2,:,:]" if ring else "c0"
    if body is None:
        body = f"N.matmul(aa,bb,accum={destination},initialize=(k==0))"
    shape = "(2,16,16)" if ring else "(16,16)"
    source = f"""
def reduction(A:N.Tensor((n,16,16),N.f16),B:N.Tensor((n,16,16),N.f16),n:N.int64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c({shape},N.f32)
        for k in N.Pipelined(n,stages=3):
            with N.Stage(0):
                av=N.copy(A[k,:,:],a1)
                bv=N.copy(B[k,:,:],b1)
            with N.Stage(1):
                aa=N.copy(av,a0)
                bb=N.copy(bv,b0,transpose=True)
            with N.Stage(2):
{indent(body, " " * 16)}
"""
    return lower_native_structure(N.from_source(source, disjoint_external=True))


def _retarget(structure, period):
    """Construct an actual cycle primitive for its native-contract unit test."""
    task = next(
        op
        for op in structure.module.walk()
        if isinstance(op, TaskOp) and op.native.data in {"mad", "mad_first"}
    )
    task.native = StringAttr("mad_cycle")
    if len(task.refs) == 3:
        stage = task.parent_op()
        while not isinstance(stage, PipelineStageOp):
            stage = stage.parent_op()
        task.operands = (*task.refs, stage.body.block.args[0])
    task.params = ArrayAttr(
        IntAttr(value) for value in (*tuple(p.data for p in task.params)[:3], period)
    )
    return task


def _admit(structure):
    memory = collect_native_memory(structure.module)
    return admit_structured_memory(structure, memory, admit_memory_bounds(memory))

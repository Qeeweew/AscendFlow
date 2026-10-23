"""Common readiness is explicit source scheduling, with per-buffer retirement."""

import pytest
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import NotificationOp
from ascendflow.target.ascend910b import SyncKind
from tests.models.structured_events import EventModelError, check_structured_events


def copies(grouped=False, reverse=False):
    lines = [
        'def copies(X:N.Tensor((9,64),N.bf16),Y:N.Tensor((9,64),N.bf16,access="write")):',
        "    with N.Kernel():",
        "        x=N.alloc_ub((9,64),N.bf16)",
    ]
    if grouped:
        lines.append('        with N.CopyGroup(name="loads"):')
    indent = "            " if grouped else "        "
    lines += [indent + f"N.copy(X[{i},:],x[{i},:])" for i in range(9)]
    lines += [
        f"        N.copy(x[{i},:],Y[{i},:])" for i in (reversed(range(9)) if reverse else range(9))
    ]
    return N.from_source("\n".join(lines), disjoint_external=True)


def test_common_group_and_reverse_acquire_fit_without_changing_independent_forward_semantics():
    with pytest.raises(CompileError):
        compile(copies())
    for program in (copies(reverse=True), copies(grouped=True)):
        artifact = compile(program)
        original = collect_native_memory(parse(artifact.before_ir))
        final = collect_native_memory(artifact.plan)
        check_structured_events(final, logical_reference=original, invocations=2)


@pytest.mark.parametrize(
    "body",
    [
        "for i in N.serial(n):\n                N.copy(X,x)",
        "if n>0:\n                N.copy(X,x)",
        "with N.CopyGroup():\n                N.copy(X,x)",
        "N.fill(x,1.0)",
    ],
)
def test_group_rejects_runtime_control_nesting_and_noncopy_effects(body):
    program = N.from_source(f"""
def invalid(X:N.Tensor((64,),N.f32),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        with N.CopyGroup():
            {body}
""")
    with pytest.raises((CompileError, VerifyException), match="CopyGroup"):
        compile(program)


def test_group_rejects_multiple_real_source_queues():
    program = N.from_source("""
def mixed(X:N.Tensor((64,),N.bf16),Y:N.Tensor((64,),N.bf16,access="write")):
    with N.Kernel():
        x=N.alloc_ub((64,),N.bf16)
        y=N.alloc_ub((64,),N.bf16)
        with N.CopyGroup():
            N.copy(X,x)
            N.copy(x,y)
""")
    with pytest.raises(CompileError, match="CopyGroup.*queue"):
        compile(program)


def test_common_readiness_in_runtime_rings_preserves_each_old_reader():
    program = N.from_source(
        """
def rings(X:N.Tensor((n,2,64),N.bf16),Y:N.Tensor((n,2,64),N.bf16,access="write"),n:N.i64):
    with N.Kernel():
        a=N.alloc_ub((64,),N.bf16,slots=2)
        b=N.alloc_ub((64,),N.bf16,slots=3)
        for t in N.Pipelined(n):
            with N.Stage(0):
                with N.CopyGroup():
                    av=N.copy(X[t,0,:],a.at(t))
                    bv=N.copy(X[t,1,:],b.at(t))
            with N.Stage(1):
                N.copy(av,Y[t,0,:])
            with N.Stage(2):
                N.copy(bv,Y[t,1,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    original = collect_native_memory(parse(artifact.before_ir))
    final = collect_native_memory(artifact.plan)
    for count in (0, 1, 2, 3, 7):
        check_structured_events(final, {"n": count}, logical_reference=original, invocations=2)


def test_grouped_partial_writes_do_not_define_unwritten_payload():
    program = N.from_source(
        """
def partial(X:N.Tensor((64,),N.bf16),Y:N.Tensor((128,),N.bf16,access="write")):
    with N.Kernel():
        x=N.alloc_ub((128,),N.bf16)
        with N.CopyGroup():
            N.copy(X,x[0:64])
        N.copy(x,Y)
""",
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized|defined"):
        compile(program)


def test_independent_model_detects_common_ready_loss_even_for_an_unread_member():
    program = N.from_source(
        """
def common(X:N.Tensor((2,64),N.bf16),Y:N.Tensor((64,),N.bf16,access="write")):
    with N.Kernel():
        x=N.alloc_ub((2,64),N.bf16)
        with N.CopyGroup():
            N.copy(X[0,:],x[0,:])
            N.copy(X[1,:],x[1,:])
        N.copy(x[0,:],Y)
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    original = collect_native_memory(parse(artifact.before_ir))
    final = collect_native_memory(artifact.plan)
    check_structured_events(final, logical_reference=original)
    publish = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp) and op.kind.data == SyncKind.LOCAL_SET.value
    )
    copies = [
        op for op in publish.parent.ops if isinstance(op, TaskOp) and op.native.data == "copy"
    ]
    publish.parent.detach_op(publish)
    copies[0].parent.insert_op_after(publish, copies[0])
    with pytest.raises(EventModelError, match="CopyGroup-ready"):
        check_structured_events(final, logical_reference=original)


def test_mixed_common_publication_preserves_both_aiv_receivers_and_one_body():
    program = N.from_source(
        """
def mixed(A:N.Tensor((16,16),N.bf16),B:N.Tensor((16,16),N.bf16),
          W:N.Tensor((2,2,16,16),N.f32,access="write"),
          Y:N.Tensor((n,2,16,16),N.f32,access="write"),n:N.i64):
    with N.Kernel(kind="mixed"):
        a1=N.alloc_l1((16,16),N.bf16)
        b1=N.alloc_l1((16,16),N.bf16)
        a0=N.alloc_l0a((16,16),N.bf16)
        b0=N.alloc_l0b((16,16),N.bf16)
        c0=N.alloc_l0c((16,16))
        x=N.alloc_ub((8,16))
        workspace=N.slotted_view(W)
        for t in N.Dataflow(n):
            with N.Cube(),N.Stage(0):
                N.copy(A,a1)
                N.copy(B,b1)
                N.copy(a1,a0)
                N.copy(b1,b0)
                N.matmul(a0,b0,accum=c0)
                with N.CopyGroup(name="exports"):
                    N.copy(c0,workspace.at(t)[0,:,:])
                    N.copy(c0,workspace.at(t)[1,:,:])
            with N.Vector() as sid,N.Stage(1):
                N.copy(workspace.at(t)[0,sid*8:sid*8+8,:],x)
                N.copy(x,Y[t,0,sid*8:sid*8+8,:])
                N.copy(workspace.at(t)[1,sid*8:sid*8+8,:],x)
                N.copy(x,Y[t,1,sid*8:sid*8+8,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    original = collect_native_memory(parse(artifact.before_ir))
    final = collect_native_memory(artifact.plan)
    # Separate launches retire through the runtime stream contract. This
    # model exercises cross-actor reuse within one actual kernel invocation.
    for count in (0, 1, 3, 7):
        check_structured_events(final, {"n": count}, logical_reference=original)

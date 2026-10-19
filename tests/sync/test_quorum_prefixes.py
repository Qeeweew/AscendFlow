"""Real per-actor capture/acquire prefixes retain two AIV contributions."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import EventModelError, check_structured_events


def test_eight_local_loads_keep_one_ordered_prefix_and_exact_ring_lifetimes():
    program = N.from_source(
        """
def local_prefix(X:N.Tensor((n,8,64),N.f32),
                 Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        inputs=N.alloc_ub((8,2,64),N.f32)
        total=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for i in N.Pipelined(n,stages=3):
                with N.Stage(0,name="load"):
                    for panel in N.unroll(8):
                        N.copy(X[i,panel,:],inputs[panel,i%2,:])
                with N.Stage(1,name="sum"):
                    N.copy(inputs[0,i%2,:],total[i%2,:])
                    for panel in N.unroll(1,8):
                        N.add(total[i%2,:],inputs[panel,i%2,:],out=total[i%2,:])
                with N.Stage(2,name="store"):
                    N.copy(total[i%2,:],Out[i,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for count in (0, 1, 2, 5):
        check_structured_events(final, {"n": count}, logical_reference=original)


def test_two_gm_handoffs_use_one_complete_quorum_prefix():
    program = N.from_source(
        """
def paired(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
           Work:N.Tensor((2,2,16,16),N.f32,access="write"),
           Out:N.Tensor((n,16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="mixed"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        x=N.alloc_ub((8,16),N.f32)
        y=N.alloc_ub((8,16),N.f32)
        for i in N.Pipelined(n,stages=2):
            with N.Stage(0,name="cube"),N.Cube():
                N.copy(A,a1)
                N.copy(B,b1)
                N.copy(a1,a0)
                N.copy(b1,b0)
                N.matmul(a0,b0,accum=c0,initialize=True)
                N.copy(c0,Work[i%2,0,:,:])
                N.copy(c0,Work[i%2,1,:,:])
            with N.Stage(1,name="vector"),N.Vector() as lane:
                N.copy(Work[i%2,0,lane*8:lane*8+8,:],x)
                N.copy(Work[i%2,1,lane*8:lane*8+8,:],y)
                N.add(x,y,out=x)
                N.copy(x,Out[i,lane*8:lane*8+8,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    assert artifact.report["analysis_work"]["resources"]["allocated_cross_ids"] <= 4
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for count in (0, 1, 2, 5):
        check_structured_events(final, {"n": count}, logical_reference=original)
    release = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.CROSS_SET.value
        and op.source.data == Engine.MTE2.value
    )
    release.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
    with pytest.raises(EventModelError, match="quorum|contribution"):
        check_structured_events(final, {"n": 5}, logical_reference=original)

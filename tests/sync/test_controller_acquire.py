"""Cross waits acquire a producer prefix on the whole receiver controller."""

from ascendflow import language as N
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine
from tests.models.structured_events import check_compiled_events


def test_cross_acquire_also_orders_later_fix_overwrite():
    source = N.from_source(
        """
        def roundtrip(A:N.Tensor((32,16),N.f32), B:N.Tensor((16,16),N.f32),
                      Shared:N.Tensor((32,16),N.f32,access="write")):
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((16,16),N.f32)
                a1=N.alloc_l1((32,16),N.f32)
                b1=N.alloc_l1((16,16),N.f32)
                a0=N.alloc_l0a((32,16),N.f32)
                b0=N.alloc_l0b((16,16),N.f32)
                c0=N.alloc_l0c((32,16),N.f32)
                with N.Vector() as lane:
                    N.copy(A[lane*16:lane*16+16,:],x)
                    N.copy(x,Shared[lane*16:lane*16+16,:])
                with N.Cube():
                    N.copy(Shared,a1)
                    N.copy(B,b1)
                    aa=N.copy(a1,a0)
                    bb=N.copy(b1,b0)
                    N.matmul(aa,bb,accum=c0)
                    N.copy(c0,Shared)
                with N.Vector() as lane:
                    N.copy(Shared[lane*16:lane*16+16,:],x)
    """,
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    crosses = [
        n for n in plan.selection.notifications if n.mechanism is NotificationKind.CROSS_EVENT
    ]
    assert any(
        n.source_engine is Engine.MTE3 and n.destination_engine is Engine.MTE2 for n in crosses
    )
    assert not any(
        n.source_engine is Engine.MTE3 and n.destination_engine is Engine.FIX for n in crosses
    )
    check_compiled_events(plan, invocations=2)


def test_local_queue_acquire_does_not_order_a_different_receiver_pipe():
    source = N.from_source(
        """
        def local(X:N.Tensor((64,),N.f32),Y:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((64,),N.f32)
                y=N.alloc_ub((64,),N.f32)
                N.copy(X,x)
                N.mul(x,2.0,out=y)
                N.copy(x,Y)
    """,
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    destinations = {
        n.destination_engine
        for n in plan.selection.notifications
        if n.mechanism is NotificationKind.LOCAL_EVENT and n.source_engine is Engine.MTE2
    }
    assert {Engine.V, Engine.MTE3} <= destinations
    check_compiled_events(plan, invocations=2)

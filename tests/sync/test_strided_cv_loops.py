"""Nested C/V streams hand off full strided invocation prefixes."""

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def cohort_source(pack):
    return N.from_source(
        f"""
def cohort(X:N.Tensor((n,{pack},16,16),N.f16),B:N.Tensor((16,16),N.f16),
           Work:N.Tensor((2,{pack},16,16),N.f16,access="write"),
           Score:N.Tensor((2,{pack},16,16),N.f32,access="write"),
           Out:N.Tensor((n,{pack},16,16),N.f32,access="write",write_disjoint=True),
           n:N.int64):
    with N.Kernel(1,kind="mixed"):
        x=N.alloc_ub((8,16),N.f16)
        y=N.alloc_ub((8,16),N.f32)
        a1=N.alloc_l1((2,16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        with N.Cube():
            N.copy(B,b1)
        for parent in N.Dataflow(n,issue_order={{"vector":("store","prepare")}}):
            with N.Vector() as lane, N.Stage(0,name="prepare"):
                for item in N.serial({pack}):
                    N.copy(X[parent,item,lane*8:lane*8+8,:],x)
                    N.copy(x,Work[parent%2,item,lane*8:lane*8+8,:])
            with N.Cube(), N.Stage(1,name="compute"):
                for item in N.Pipelined({pack}):
                    with N.Stage(0):
                        N.copy(Work[parent%2,item,:,:],a1[item%2,:,:])
                    with N.Stage(1):
                        aa=N.copy(a1[item%2,:,:],a0)
                        bb=N.copy(b1,b0)
                        N.matmul(aa,bb,accum=c0,initialize=True)
                        N.copy(c0,Score[parent%2,item,:,:])
            with N.Vector() as lane, N.Stage(2,name="store"):
                for item in N.serial({pack}):
                    N.copy(Score[parent%2,item,lane*8:lane*8+8,:],y)
                    N.copy(y,Out[parent,item,lane*8:lane*8+8,:])
""",
        disjoint_external=True,
    )


def test_cohort_size_does_not_allocate_one_cross_event_per_item():
    small = synchronize_structure(cohort_source(3))
    large = synchronize_structure(cohort_source(17))
    assert large.resources.work.allocated_cross_ids == small.resources.work.allocated_cross_ids
    assert large.resources.work.allocated_cross_ids <= 8
    for count in (0, 1, 2, 4):
        check_compiled_events(small, {"n": count}, invocations=2)

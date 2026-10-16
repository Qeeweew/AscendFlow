"""One symbolic AIV body with two participant address interpretations.

Dispatch interpretation does not synchronize AIC/AIV or prove device results.
"""

import pytest

from tests.control.test_recursive_native import dispatch
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import BufferOp, MixedOp, TaskOp
from xdsl_npu.dialects.source import DomainOp
from xdsl_npu.dialects.structured import ActorDomainOp, PipelineLoopOp
from xdsl_npu.passes.lower_native import lower_native_structure

SHARED = """
def shared(A:N.Tensor((GROUPS,n,32,16),N.f16),
           B:N.Tensor((GROUPS,n,16,16),N.f16),
           Scratch:N.Tensor((GROUPS,2,32,16),N.f32,access="write"),
           Y:N.Tensor((GROUPS,n,32,16),N.f32,access="write"),
           n:N.int64,GROUPS:N.constexpr=3):
    with N.Kernel(GROUPS,kind="mixed") as group:
        a1=N.alloc_l1((32,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((32,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((32,16),N.f32)
        x=N.alloc_ub((16,16),N.f32,name="x")
        y=N.alloc_ub((16,16),N.f32,name="y")
        for t in N.Pipelined(n,stages=2,
                 issue_order={"cube":("produce",),"vector":("consume",)}):
            with N.Stage(0,name="produce"),N.Cube():
                av=N.copy(A[group,t,:,:],a1)
                bv=N.copy(B[group,t,:,:],b1)
                aa=N.copy(av,a0)
                bb=N.copy(bv,b0)
                acc=N.matmul(aa,bb,accum=c0,initialize=True)
                N.copy(acc,Scratch[group,t%2,:,:])
            with N.Stage(1,name="consume"),N.Vector() as sid:
                tile=N.copy(Scratch[group,t%2,sid*16:(sid+1)*16,:],x)
                value=N.mul(tile,2.0,out=y)
                N.copy(value,Y[group,t,sid*16:(sid+1)*16,:])
"""


def test_mixed_ir_has_one_vector_body_one_sid_and_shared_gm_root():
    plan = lower_native_structure(N.from_source(SHARED, disjoint_external=True))
    assert isinstance(plan.module.body.block.first_op, MixedOp)
    domains = tuple(op for op in plan.module.walk() if isinstance(op, ActorDomainOp))
    assert [domain.kind.data for domain in domains] == ["cube", "vector"]
    assert sum(isinstance(op, TaskOp) for op in domains[1].walk()) == 3
    assert plan.work["native_templates"] == 9  # vector tasks are not doubled
    assert len(plan.loops) == 1 and len(plan.domains) == 2
    assert all(isinstance(op, DomainOp) for op in plan.domains)
    shared = next(
        op.ref
        for op in plan.module.walk()
        if isinstance(op, BufferOp) and op.sym_name.data == "Scratch"
    )
    cube_fix = next(
        op for op in domains[0].walk() if isinstance(op, TaskOp) and op.native.data == "fix"
    )
    vector_copy = next(
        op for op in domains[1].walk() if isinstance(op, TaskOp) and op.native.data == "copy"
    )
    assert cube_fix.refs[1].owner.source is vector_copy.refs[0].owner.source is shared
    assert plan.task_scopes[vector_copy].operation is domains[1]
    assert dump(parse(dump(plan.module))) == dump(plan.module)


@pytest.mark.parametrize("count,group", [(0, 0), (1, 0), (3, 2)])
def test_same_vector_template_selects_two_gm_halves_and_keeps_private_ub_offsets(count, group):
    plan = lower_native_structure(N.from_source(SHARED, disjoint_external=True))
    for sid in (0, 1):
        _, trace = dispatch(plan, n=count, group=group, actor="vector", sid=sid)
        expected = []
        for t in range(count):
            source = ((group * 2 + t % 2) * 32 + sid * 16) * 16 * 4
            output = ((group * count + t) * 32 + sid * 16) * 16 * 4
            expected.extend(
                [
                    ("copy", 0.0, (("Scratch", source), ("x", 0))),
                    ("muls", 2.0, (("x", 0), ("y", 0))),
                    ("copy", 0.0, (("y", 0), ("Y", output))),
                ]
            )
        assert trace == expected


def test_actor_specific_issue_orders_remain_on_one_loop_template():
    plan = lower_native_structure(N.from_source(SHARED, disjoint_external=True))
    loop = next(op for op in plan.module.walk() if isinstance(op, PipelineLoopOp))
    assert {
        family: tuple(label.data for label in labels)
        for family, labels in loop.issue_order.data.items()
    } == {
        "cube": ("produce",),
        "vector": ("consume",),
    }


def test_unscoped_mixed_native_is_rejected_instead_of_assuming_a_lane():
    source = N.from_source("""
        def ambiguous():
            with N.Kernel(kind="mixed"):
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
    """)
    with pytest.raises(CompileError, match="explicit Cube or Vector domain"):
        lower_native_structure(source)

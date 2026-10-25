"""Per-read cache aliases preserve physical identity, versions and ownership."""

import pytest

from ascendflow import compile
from ascendflow import language as T
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.physical import BufferOp
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def program(layout="linear", bypass="i%2==0"):
    return T.from_source(
        f"""
def selected(X:T.Tensor((2,32,32),T.f16,layout="{layout}"),
             Y:T.Tensor((n,32,32),T.f32,access="write",write_disjoint=True),n:T.i64):
    with T.Kernel(1,kind="cube"):
        a1=T.alloc_l1((32,32),T.f16,slots=2)
        b1=T.alloc_l1((32,32),T.f16,slots=2)
        a0=T.alloc_l0a((32,32),T.f16)
        b0=T.alloc_l0b((32,32),T.f16)
        c0=T.alloc_l0c((32,32),T.f32)
        for i in T.Pipelined(n):
            with T.Stage(0), T.CopyGroup():
                a=T.copy(X[i%2,:,:],a1.at(i),cache_bypass={bypass})
                b=T.copy(X[i%2,:,:],b1.at(i))
            with T.Stage(1):
                aa=T.copy(a,a0)
                bb=T.copy(b,b0)
                c=T.matmul(aa,bb,accum=c0,initialize=True)
                T.copy(c,Y[i,:,:])
""",
        disjoint_external=True,
    )


@pytest.mark.parametrize("layout", ("linear", "NZ"))
def test_a_cache_selected_copy_uses_the_same_real_allocation_and_events(layout):
    plan = synchronize_structure(program(layout))
    ordinary = synchronize_structure(program(layout, "False"))
    assert len(plan.admission.memory.templates) == len(ordinary.admission.memory.templates)
    assert [(a.kind, a.source, a.destination) for a in plan.resources.actions] == [
        (a.kind, a.source, a.destination) for a in ordinary.resources.actions
    ]
    root = next(
        op
        for op in plan.structure.module.walk()
        if isinstance(op, BufferOp) and op.sym_name.data == "X"
    )
    reads = [
        a for t in plan.admission.memory.templates for a in t.accesses if a.address.root is root.ref
    ]
    assert len(reads) == 2
    assert sum(a.address.operation.cache_bypass is not None for a in reads) == 1
    result = compile(program(layout))
    assert result.report["abi"]["device_cache_aliases"] == ["X"]
    assert len(result.report["abi"]["tensors"]) == 2
    assert "uintptr_t(arg_X) + npu_l2_offset" in result.asc
    assert "? npu_buffer_0_bypass : npu_buffer_0" in result.asc
    for n in (0, 1, 4):
        check_compiled_events(plan, {"n": n}, invocations=2)


@pytest.mark.parametrize("bypass", ("True", "False"))
def test_static_cache_selection_needs_no_runtime_pointer_branch(bypass):
    result = compile(program(bypass=bypass))
    assert bool(result.report["abi"].get("device_cache_aliases")) == (bypass == "True")
    if bypass == "False":
        assert "rtGetL2CacheOffset" not in result.asc


def test_a_cache_read_policy_cannot_be_applied_to_a_local_copy():
    source = program().source.replace("aa=T.copy(a,a0)", "aa=T.copy(a,a0,cache_bypass=True)")
    with pytest.raises(CompileError, match="external read-only GM source"):
        synchronize_structure(T.from_source(source, disjoint_external=True))


def test_private_aivs_share_code_and_select_the_same_gm_allocation_independently():
    source = T.from_source(
        """
def lanes(X:T.Tensor((64,),T.f32),
          Y:T.Tensor((64,),T.f32,access="write",write_disjoint=True)):
    with T.Kernel(1,kind="mixed"):
        tile=T.alloc_ub((32,),T.f32)
        with T.Vector() as lane:
            value=T.copy(X[lane*32:(lane+1)*32],tile,cache_bypass=lane==0)
            T.copy(value,Y[lane*32:(lane+1)*32])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    result = compile(source)
    assert result.report["abi"]["device_cache_aliases"] == ["X"]
    assert result.asc.count("asc_copy_gm2ub(") == 1
    check_compiled_events(plan, invocations=2)

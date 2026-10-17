"""Native reader retirement and source-value bindings on structured dependencies."""

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Actor, Engine, Milestone
from tests.analysis.test_structured_dependencies import materialized_edges


def dependencies(source):
    structure = lower_native_structure(N.from_source(source, disjoint_external=True))
    result = build_structured_dependencies(collect_native_memory(structure.module))
    result.require_closed()
    return result


def test_l1_reuse_retires_at_actual_load2d_reader():
    result = dependencies("""
        def early(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
                  Y:N.Tensor((16,16),N.f32,access="write")):
            with N.Kernel(kind="cube"):
                a,b=N.alloc_l1((16,16),N.f16),N.alloc_l1((16,16),N.f16)
                aa,bb=N.alloc_l0a((16,16),N.f16),N.alloc_l0b((16,16),N.f16)
                c=N.alloc_l0c((16,16),N.f32)
                N.copy(A,a)
                N.copy(B,b)
                N.copy(a,aa)
                N.copy(b,bb)
                N.matmul(aa,bb,accum=c,initialize=True)
                N.copy(c,Y)
                N.copy(A,a)
    """)
    templates = result.memory.templates
    root = templates[0].accesses[1].address.root
    reader = next(t.operation for t in templates if t.contract.native == "load2d_a")
    overwrite = templates[-1].operation
    release = [n for n in result.needs if n.kind is HazardKind.WAR and n.cell.root is root]
    assert len(release) == 1
    assert release[0].source.operation is reader
    assert release[0].consumer.operation is overwrite
    assert release[0].source_cut.milestone is Milestone.NATIVE_END
    assert release[0].consumer_cut.milestone is Milestone.ADMISSION
    assert {Engine.M, Engine.FIX} <= {t.contract.engine for t in templates}


def test_shared_gm_reuse_retires_both_aiv_readers_of_one_template():
    result = dependencies("""
        def readers(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
                    Scratch:N.Tensor((16,16),N.f32,access="read_write")):
            with N.Kernel(kind="mixed"):
                a,b=N.alloc_l1((16,16),N.f16),N.alloc_l1((16,16),N.f16)
                aa,bb=N.alloc_l0a((16,16),N.f16),N.alloc_l0b((16,16),N.f16)
                c=N.alloc_l0c((16,16),N.f32)
                x=N.alloc_ub((16,16),N.f32)
                with N.Cube():
                    N.copy(A,a)
                    N.copy(B,b)
                    N.copy(a,aa)
                    N.copy(b,bb)
                    N.matmul(aa,bb,accum=c,initialize=True)
                    N.copy(c,Scratch)
                with N.Vector():
                    N.copy(Scratch,x)
                with N.Cube():
                    N.matmul(aa,bb,accum=c,initialize=False)
                    N.copy(c,Scratch)
    """)
    first, overwrite = (t for t in result.memory.templates if t.contract.native == "fix")
    root = first.accesses[1].address.root
    release = [
        n
        for n in result.needs
        if n.kind is HazardKind.WAR
        and n.cell.root is root
        and n.consumer.operation is overwrite.operation
    ]
    assert len(release) == 2
    assert {n.source.participant for n in release} == {Actor.AIV0, Actor.AIV1}
    assert release[0].source.operation is release[1].source.operation
    reader = next(t for t in result.memory.templates if t.operation is release[0].source.operation)
    assert reader.contract.engine is Engine.MTE2
    assert all(n.source_cut.milestone is Milestone.NATIVE_END for n in release)
    assert all(n.consumer_cut.milestone is Milestone.ADMISSION for n in release)


def test_reverse_issue_order_preserves_each_read_source_value():
    result = dependencies("""
        def logical():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for g in N.Pipelined(4,stages=3,
                    issue_order={"vector":("overwrite","read","initialize")}):
                    with N.Stage(0,name="initialize"):
                        N.fill(x,0.0)
                    with N.Stage(1,name="read"):
                        N.mul(x,2.0,out=y)
                    with N.Stage(2,name="overwrite"):
                        N.fill(x,1.0)
    """)
    initialize, read, _ = (t.operation for t in result.memory.templates)
    loop = next(iter(result.memory.control.loops.values())).operation
    raw = {
        edge
        for edge in materialized_edges(result, {"$group": 0, "$sid": 0})
        if edge[0] is HazardKind.RAW
    }
    assert raw == {
        (
            HazardKind.RAW,
            (initialize, Actor.AIV0, ((loop, generation),)),
            (read, Actor.AIV0, ((loop, generation),)),
        )
        for generation in range(4)
    }

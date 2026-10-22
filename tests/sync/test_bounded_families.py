"""Bounded read unions retain actual family lengths and selected parent clocks."""

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


def check(source, values):
    result = compile(N.from_source(source, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for arguments in values:
        check_structured_events(memory, arguments, logical_reference=original)
    return memory


def test_reader_prefix_releases_after_actual_last_member_before_overwrite():
    memory = check(
        """
    def prefix(Work:N.Tensor((4,64),N.f32,access="write"),
               Out:N.Tensor((4,64),N.f32,access="write"),n:N.i64(minimum=2)):
        with N.Kernel():
            init=N.alloc_ub((4,64))
            x=N.alloc_ub((2,64))
            N.fill(init,1.0)
            N.copy(init,Work)
            for j in N.Pipelined(N.min(4,n)):
                with N.Stage(0):
                    N.copy(Work[j,:],x[j%2,:])
                with N.Stage(1):
                    N.copy(x[j%2,:],Out[j,:])
            N.fill(init,2.0)
            N.copy(init,Work)
""",
        ({"n": n} for n in (2, 3, 4, 9)),
    )
    reads = [
        a
        for t in memory.templates
        for a in t.accesses
        if a.mode.reads and a.address.root.owner.sym_name.data == "Work"
    ]
    assert len(reads) == 1 and reads[0].address.geometry.total_nbytes == 256


def test_child_last_uses_the_selected_parent_header_not_an_unbound_parent():
    check(
        """
    def selected(X:N.Tensor((4,64),N.f32),Y:N.Tensor((2,64),N.f32,access="write"),
                 n:N.i64(minimum=1)):
        with N.Kernel():
            x=N.alloc_ub((2,64))
            for parent in N.serial(n):
                for child in N.Pipelined(N.min(4,N.max(2,parent+2))):
                    with N.Stage(0):
                        N.copy(X[child,:],x[child%2,:])
                    with N.Stage(1):
                        N.mul(x[child%2,:],2.0,out=x[child%2,:])
            N.fill(x,7.0)
            N.copy(x,Y)
""",
        ({"n": n} for n in (1, 2, 3, 5)),
    )


def test_entry_publication_resolves_computed_last_after_a_selected_parent():
    check(
        """
    def selected(X:N.Tensor((4,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),
                 n:N.i64(minimum=1)):
        with N.Kernel():
            x=N.alloc_ub((64,))
            one=N.alloc_ub((64,))
            N.fill(one,1.0)
            for parent in N.serial(n):
                count=N.min(4,N.max(2,N.ceildiv((parent+1)*2,3)))
                for child in N.Pipelined(count):
                    with N.Stage(0):
                        N.copy(X[child,:],x)
                    with N.Stage(1):
                        N.mul(x,2.0,out=x)
                        if child==count-1:
                            N.mul(x,one,out=x)
                            N.copy(x,Y[parent,:])
""",
        ({"n": n} for n in (1, 2, 3, 5)),
    )

"""Structured storage selects actual bytes; borrowed slices retain their writer."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


def check(program, counts=(0, 1, 2, 7)):
    artifact = compile(program)
    before = collect_native_memory(parse(artifact.before_ir))
    after = collect_native_memory(artifact.plan)
    for count in counts:
        check_structured_events(after, {"n": count}, logical_reference=before, invocations=2)
    return artifact


def test_independent_slots_keep_raw_aliases_and_runtime_addresses():
    program = N.from_source(
        """
def rings(X:N.Tensor((n,64),N.bf16),Y:N.Tensor((n,64),N.bf16,access="write"),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,),N.bf16,slots=3)
        y=N.alloc_ub((64,),N.bf16,slots=5)
        for t in N.Pipelined(n,name="tiles"):
            with N.Stage(0,name="load"):
                value=N.copy(X[t,:],x.at(t))
            with N.Stage(1,name="copy"):
                N.copy(value,y.at(t))
            with N.Stage(2,name="store"):
                N.copy(y.storage[t%5,:],Y[t,:])
""",
        disjoint_external=True,
    )
    artifact = check(program)
    assert "npu_source.storage_select" in artifact.source_ir


def test_slotted_workspace_view_keeps_original_root_and_nonzero_origin():
    program = N.from_source(
        """
def slots(X:N.Tensor((n,64),N.bf16),W:N.Tensor((2,5,64),N.bf16,access="read_write"),
          Y:N.Tensor((n,64),N.bf16,access="write"),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,),N.bf16)
        ring=N.slotted_view(W[1,1:4,:])
        for t in N.Pipelined(n):
            with N.Stage(0):
                N.copy(X[t,:],x)
                N.copy(x,ring.at(t))
            with N.Stage(1):
                N.copy(W[1,1+t%3,:],x)
                N.copy(x,Y[t,:])
""",
        disjoint_external=True,
    )
    check(program)


@pytest.mark.parametrize("slots", ["0", "-1", "True", "1.5", "n"])
def test_slot_depth_requires_positive_static_integer(slots):
    program = N.from_source(f"""
def invalid(n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,),slots={slots})
""")
    with pytest.raises(CompileError, match="slots.*positive constexpr integer"):
        program.to_ir()


def test_physical_slot_does_not_wrap_and_storage_has_no_event_capacity_limit():
    program = N.from_source("""
def physical():
    with N.Kernel():
        x=N.alloc_ub((64,),slots=9)
        N.fill(x.slot(8),1.0)
""")
    compile(program)
    invalid = N.from_source("""
def invalid():
    with N.Kernel():
        x=N.alloc_ub((64,),slots=2)
        N.fill(x.slot(2),1.0)
""")
    with pytest.raises(CompileError, match="exceeds"):
        invalid.to_ir()


def test_borrowed_tile_slice_preserves_writer_and_rejects_raw_overwrite():
    def program(overwrite):
        return N.from_source(
            f"""
def sliced(X:N.Tensor((128,),N.bf16),Y:N.Tensor((64,),N.bf16,access="write")):
    with N.Kernel():
        x=N.alloc_ub((128,),N.bf16,slots=2)
        value=N.copy(X,x.at(0))
        {"N.copy(X,x.storage[0,:])" if overwrite else "pass"}
        N.copy(value[64:128],Y)
""",
            disjoint_external=True,
        )

    check(program(False), (0,))
    with pytest.raises(CompileError, match="overwrit|borrowed|logical Tile"):
        compile(program(True))


def test_slot_selection_keeps_illegal_original_expression():
    program = N.from_source("""
def invalid():
    with N.Kernel():
        x=N.alloc_ub((64,),slots=2)
        N.fill(x.at((1//0)*0),1.0)
""")
    with pytest.raises(CompileError):
        program.to_ir()


def test_packed_predicate_slots_keep_descriptor_storage_and_writer_together():
    program = N.from_source(
        """
def masks(X:N.Tensor((n,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,),slots=3)
        y=N.alloc_ub((64,),slots=2)
        temporary=N.alloc_ub((64,))
        packed=N.alloc_mask((64,),slots=3)
        for t in N.Pipelined(n):
            with N.Stage(0):
                value=N.copy(X[t,:],x.at(t))
                predicate=N.compare(value,0.0,out=packed.at(t))
            with N.Stage(1):
                N.fill(y.at(t),0.0)
                N.mul(value,2.0,out=y.at(t),mask=predicate,scratch=temporary)
            with N.Stage(2):
                N.copy(y.at(t),Y[t,:])
""",
        disjoint_external=True,
    )
    check(program)


def test_nonleading_slot_axis_retains_real_stride():
    program = N.from_source(
        """
def columns(X:N.Tensor((2,3,64),N.bf16),Y:N.Tensor((2,64),N.bf16,access="write")):
    with N.Kernel():
        x=N.alloc_ub((2,64),N.bf16)
        ring=N.slotted_view(X,axis=1)
        N.copy(ring.slot(2),x)
        N.copy(x,Y)
""",
        disjoint_external=True,
    )
    check(program, (0,))

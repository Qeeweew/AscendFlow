"""Address admission consumes dominating predicates and correlated metadata."""

import pytest

from ascendflow import compile
from ascendflow import language as N


def test_window_metadata_relationship_survives_original_unsigned_page_arithmetic():
    source = N.from_source(
        """
def window(X:N.Tensor((16,16,64),N.f32), L:N.Tensor((1,),N.i32),
           W:N.Tensor((1,),N.i32), O:N.Tensor((16,64),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,))
        length=N.load_index(L,minimum=0,maximum=256)
        visible=N.load_index(W,minimum=0,maximum=N.min(16,length))
        for i in N.serial(16):
            if i < visible:
                position=length-visible+i
                N.copy(X[position//16,position%16,:],row)
                N.copy(row,O[i,:])
""",
        disjoint_external=True,
    )
    result = compile(source)
    assert "volatile int32_t" in result.asc


def test_else_region_keeps_its_own_signed_comparison_for_page_bounds():
    from ascendflow.analysis.structured_bounds import admit_memory_bounds
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.passes.lower_native import lower_native_structure

    source = N.from_source(
        """
def selected(X:N.Tensor((2,8,64),N.f32), O:N.Tensor((32,64),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,))
        for i in N.serial(32):
            N.fill(row,0.0)
            if i < 16:
                N.fill(row,0.0)
            else:
                position=i-16
                N.copy(X[position//8,position%8,:],row)
            N.copy(row,O[i,:])
""",
        disjoint_external=True,
    )
    # Test source evaluation and address admission here. Contracting this
    # conditional read family into precise storage cells is a separate pass.
    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    assert admit_memory_bounds(memory).checked_accesses > 0


def test_only_guarded_rows_are_read_without_a_universal_axis_proof():
    from ascendflow.analysis.structured_memory import collect_native_memory
    from tests.models.structured_events import check_structured_events

    source = N.from_source(
        """
def guarded(X:N.Tensor((8,64),N.f32), O:N.Tensor((16,64),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,))
        for i in N.serial(16):
            N.fill(row,0.0)
            if i < 8:
                N.copy(X[i,:],row)
            N.copy(row,O[i,:])
""",
        disjoint_external=True,
    )
    result = compile(source)
    trace = check_structured_events(collect_native_memory(result.plan))
    assert [
        (s.start, s.end) for c in trace.natives for s in c.spans if s.root.sym_name.data == "X"
    ] == [(i * 256, (i + 1) * 256) for i in range(8)]


@pytest.mark.parametrize("dtype,width", [("i32", 4), ("i64", 8)])
def test_metadata_has_actual_storage_width_and_signed_widening(dtype, width):
    from ascendflow.analysis.structured_memory import collect_native_memory

    source = N.from_source(
        """
def sentinel(X:N.Tensor((4,64),N.f32), I:N.Tensor((1,),N.DTYPE),
             O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,))
        index=N.load_index(I,minimum=0,maximum=3)
        N.copy(X[index,:],row)
        N.copy(row,O)
""".replace("DTYPE", dtype),
        disjoint_external=True,
    )
    result = compile(source)
    memory = collect_native_memory(result.plan)
    loads = [t for t in memory.templates if t.spec.native == "index_load"]
    assert len(loads) == 1
    assert loads[0].accesses[0].address.geometry.total_nbytes == width
    assert "int64_t(*((__gm__ volatile int" + str(width * 8) + "_t*)" in result.asc

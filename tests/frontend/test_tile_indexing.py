"""Runtime Tile rows and slices are ordinary views with writer provenance."""

import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


@pytest.mark.parametrize("sliced", [False, True])
def test_runtime_tile_row_selection_does_not_evaluate_a_device_predicate_in_python(sliced):
    shape = "1,64" if sliced else "64,"
    selection = "value[which:which+1,:]" if sliced else "value[which,:]"
    source = N.from_source(
        f"""
def select(A:N.Tensor((2,64),N.f32),O:N.Tensor(({shape}),N.f32,access="write"),which:N.i64):
    with N.Kernel(kind="vector"):
        x=N.alloc_ub((2,64),N.f32)
        y=N.alloc_ub(({shape}),N.f32)
        value=N.copy(A,x)
        result=N.mul({selection},2.0,out=y)
        N.copy(result,O)
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    assert plan.admission.values
    for which in (0, 1):
        check_compiled_events(plan, {"which": which}, invocations=2)


@pytest.mark.parametrize("rows", [False, True])
def test_borrowed_broadcast_uses_the_selected_vector_slice(rows):
    selection = "values[which*8:which*8+8,None]" if rows else "values[None,which*8:which*8+64]"
    source = N.from_source(
        f"""
def broadcast(A:N.Tensor((8,64),N.f32),V:N.Tensor((128,),N.f32),
              O:N.Tensor((8,64),N.f32,access="write"),which:N.i64):
    with N.Kernel(kind="vector"):
        x=N.alloc_ub((8,64),N.f32)
        v=N.alloc_ub((128,),N.f32,name="vectors")
        y=N.alloc_ub((8,64),N.f32)
        scratch=N.alloc_ub((8,8),N.f32)
        matrix=N.copy(A,x)
        values=N.copy(V,v)
        result=N.mul(matrix,{selection},out=y,scratch=scratch if {rows} else None)
        N.copy(result,O)
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for which in (0, 8):
        trace = check_compiled_events(plan, {"which": which}, invocations=2)
        # Each invocation reads one selected, aligned vector. Reading the
        # full parent would either mismatch the native shape or its bytes.
        reads = [
            span
            for command in trace.natives
            for span in command.spans
            if span.root.sym_name.data == "vectors" and span.mode == "R"
        ]
        assert reads
        width = 8 if rows else 64
        assert all(
            span.start == which * 32 and span.end == which * 32 + width * 4 for span in reads
        )

"""A complete metadata Tile retains its selected writer at indirect reads."""

import pytest

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def metadata(*, mixed=False, partial=False, overwrite=False):
    return N.from_source(
        f"""
def metadata(A:N.Tensor((32,),N.i32), B:N.Tensor((32,),N.i32), index:N.i64):
    with N.Kernel(kind={"'mixed'" if mixed else "'vector'"}):
        x=N.alloc_ub((32,),N.i32)
        with N.Vector():
            N.copy(B,x)
            value=N.copy(A[:16] if {partial} else A, x[:16] if {partial} else x)
            {"N.copy(B,x)" if overwrite else "pass"}
            selected=N.load_index(value[index:index+1])
""",
        disjoint_external=True,
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_full_metadata_value_supports_caller_owned_indirect_access(mixed):
    plan = synchronize_structure(metadata(mixed=mixed))
    assert plan.admission.values
    for index in (0, 7, 16, 31):
        check_compiled_events(plan, {"index": index}, invocations=2)


def test_sibling_initialization_cannot_expand_the_selected_partial_writer():
    with pytest.raises(CompileError, match="logical Tile selection"):
        synchronize_structure(metadata(partial=True))


def test_later_full_write_cannot_replace_the_selected_metadata_writer():
    with pytest.raises(CompileError, match="overwrite"):
        synchronize_structure(metadata(overwrite=True))


@pytest.mark.parametrize("mixed", [False, True])
def test_indirect_metadata_in_one_runtime_slot_keeps_its_actual_plane(mixed):
    program = N.from_source(
        f"""
def metadata(A:N.Tensor((32,),N.i32), index:N.i64, count:N.i64):
    with N.Kernel(kind={"'mixed'" if mixed else "'vector'"}):
        x=N.alloc_ub((32,),N.i32,slots=2)
        for t in N.Dataflow(count,issue_order={{"vector":("load","read")}}):
            with N.Vector(),N.Stage(0,name="load"):
                value=N.copy(A,x.at(t))
            with N.Vector(),N.Stage(1,name="read"):
                selected=N.load_index(value[index:index+1])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(program)
    for count in (0, 1, 2, 5):
        for index in (0, 16, 31):
            check_compiled_events(plan, {"index": index, "count": count}, invocations=2)

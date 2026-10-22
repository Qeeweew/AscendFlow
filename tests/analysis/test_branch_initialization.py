"""Pointwise branch definitions precede loop-range initialization summaries."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.passes.lower_native import lower_native_structure
from tests.models.structured_events import check_structured_events

SOURCE = """
def initialize(
    X:N.Tensor((4,64),N.f32),
    Flags:N.Tensor((4,),N.i32),
    Count:N.Tensor((1,),N.i32),
    Y:N.Tensor((4,64),N.f32,access="write"),
):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((4,64),N.f32)
        count=N.load_index(Count[0:1],minimum=0,maximum=4)
        for row in N.serial(4):
            if row<count:
                selected=N.load_index(Flags[row:row+1],minimum=0,maximum=3)
                if selected<2:
                    N.copy(X[selected,:],x[row,:])
                else:
                    N.fill(x[row,:],2.0)
            else:
                N.fill(x[row,:],0.0)
        N.copy(x,Y)
"""


def admit(source):
    structure = lower_native_structure(N.from_source(source, disjoint_external=True))
    memory = collect_native_memory(structure.module)
    return admit_structured_memory(structure, memory, admit_memory_bounds(memory))


def test_nested_complete_writes_initialize_each_row_before_whole_tile_read():
    admission = admit(SOURCE)
    assert admission.work["expanded_iterations"] == 0


def test_three_levels_join_the_actual_complete_branch_tree():
    source = SOURCE.replace(
        "                    N.copy(X[selected,:],x[row,:])",
        "                    if selected%2==0:\n"
        "                        N.copy(X[selected,:],x[row,:])\n"
        "                    else:\n"
        "                        N.fill(x[row,:],1.0)",
    )
    admit(source)


def test_invariant_condition_can_select_different_complete_child_loops():
    source = """
def whole_children(X:N.Tensor((4,64),N.f32), Count:N.Tensor((1,),N.i32),
                   Y:N.Tensor((4,64),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((4,64),N.f32)
        count=N.load_index(Count[0:1],minimum=0,maximum=4)
        if count<2:
            for row in N.serial(4):
                N.copy(X[row,:],x[row,:])
        else:
            for pair in N.serial(2):
                N.fill(x[pair*2:pair*2+2,:],2.0)
        N.copy(x,Y)
"""
    admit(source)


@pytest.mark.parametrize("change", ("missing_inner", "missing_outer", "wrong_row", "half_row", "rmw"))
def test_incomplete_or_reading_writers_cannot_initialize_the_whole_tile(change):
    before, after = {
        "missing_inner": ("                    N.fill(x[row,:],2.0)", "                    pass"),
        "missing_outer": ("                N.fill(x[row,:],0.0)", "                pass"),
        "wrong_row": ("N.fill(x[row,:],2.0)", "N.fill(x[(row+1)%4,:],2.0)"),
        "half_row": ("N.fill(x[row,:],0.0)", "N.fill(x[row,0:32],0.0)"),
        "rmw": ("N.fill(x[row,:],2.0)", "N.add(x[row,:],1.0,out=x[row,:])"),
    }[change]
    assert before in SOURCE
    with pytest.raises(CompileError, match="initialized writer coverage"):
        admit(SOURCE.replace(before, after))


@pytest.mark.parametrize("minimum", (0, 4))
def test_complete_branches_do_not_make_an_optional_writer_loop_execute(minimum):
    source = SOURCE.replace(
        '    Y:N.Tensor((4,64),N.f32,access="write"),',
        '    Y:N.Tensor((4,64),N.f32,access="write"),\n'
        f'    n:N.i64(minimum={minimum}),',
    ).replace('N.serial(4)', 'N.serial(N.min(n,4))')
    if minimum == 4:
        admit(source)
    else:
        with pytest.raises(CompileError, match="initialized writer coverage"):
            admit(source)


def test_different_metadata_definitions_cannot_supply_opposite_branch_leaves():
    source = """
def unrelated(X:N.Tensor((4,64),N.f32),
              Left:N.Tensor((4,),N.i32), Right:N.Tensor((4,),N.i32),
              Y:N.Tensor((4,64),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((4,64),N.f32)
        for row in N.serial(4):
            left=N.load_index(Left[row:row+1],minimum=0,maximum=3)
            right=N.load_index(Right[row:row+1],minimum=0,maximum=3)
            if left<2:
                N.copy(X[row,:],x[row,:])
            if right>=2:
                N.fill(x[row,:],0.0)
        N.copy(x,Y)
"""
    # left=3 and right=0 leave that row undefined.
    with pytest.raises(CompileError, match="initialized writer coverage"):
        admit(source)


def test_shared_gm_cannot_join_complementary_metadata_from_different_aivs():
    source = """
def peers(Flags:N.Tensor((2,),N.i32),
          Scratch:N.Tensor((192,),N.f32,access="write"),
          Y:N.Tensor((128,),N.f32,access="write")):
    with N.Kernel(1,kind="mixed"):
        x=N.alloc_ub((64,),N.f32)
        with N.Vector() as sid:
            flag=N.load_index(Flags[sid:sid+1],minimum=0,maximum=3)
            N.fill(x,1.0)
            if flag<2:
                N.copy(x,Scratch[sid*64:sid*64+64])
            else:
                start=(2-2*sid)*64
                N.copy(x,Scratch[start:start+64])
            N.copy(Scratch[0:64],x)
            N.copy(x,Y[sid*64:sid*64+64])
"""
    # Flags=(3,0) writes disjoint rows at 128 and 64. Neither participant
    # defines [0:64], although their unbound printed guards look opposite.
    with pytest.raises(CompileError, match="initialized writer coverage"):
        admit(source)


def test_actual_nested_pipeline_keeps_all_conditional_writer_completions(monkeypatch):
    from tests.models.structured_events import _Interpret

    original = _Interpret.native
    inputs = {"count": 0, "flags": (0, 0, 0, 0)}

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            values[operation.value] = (
                inputs["count"] if root.sym_name.data == "Count"
                else inputs["flags"][offset // 4]
            )

    monkeypatch.setattr(_Interpret, "native", metadata)
    result = compile(N.from_source(SOURCE, disjoint_external=True), timeout_s=300)
    memory = collect_native_memory(result.plan)
    reference = collect_native_memory(parse(result.before_ir))
    for count in range(5):
        for flags in ((0, 0, 0, 0), (3, 3, 3, 3), (0, 1, 2, 3), (2, 0, 2, 0)):
            inputs.update(count=count, flags=flags)
            check_structured_events(memory, logical_reference=reference, invocations=2)

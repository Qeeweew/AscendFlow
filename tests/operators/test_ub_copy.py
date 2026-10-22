"""Current source UB copy rejects exact, shifted and changing aliases."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError


@pytest.mark.parametrize("destination", ("0:16", "8:24"))
def test_exact_and_shifted_overlaps_are_refused_without_memmove_assumptions(destination):
    text = f"""
def overlap(n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((32,),N.f32)
        for tile in N.Pipelined(n):
            N.fill(x,1.0)
            N.copy(x[0:16],x[{destination}])
"""
    with pytest.raises(CompileError, match="UB-to-UB copy requires provably disjoint"):
        compile(N.from_source(text))


def test_generation_dependent_aliases_are_not_assumed_disjoint():
    text = """
def changing(n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((3,8),N.f32)
        for tile in N.Pipelined(n):
            N.fill(x,1.0)
            N.copy(x[tile%2,:],x[tile%3,:])
"""
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        compile(N.from_source(text))


@pytest.mark.parametrize("destination", ("0:16", "8:24"))
def test_pitched_row_copy_does_not_authorize_overlapping_columns(destination):
    text = f"""
def overlap_rows():
    with N.Kernel():
        x=N.alloc_ub((16,32),N.f32)
        N.fill(x,1.0)
        N.copy(x[:,:16],x[:,{destination}])
"""
    with pytest.raises(CompileError, match="UB-to-UB copy requires provably disjoint"):
        compile(N.from_source(text))


def test_pitched_row_copy_keeps_exact_rectangles_and_a_closed_vector_mask():
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from ascendflow.dialects.physical import TaskOp
    from ascendflow.target.packets import MaskStep, audited_packet
    from tests.models.structured_events import check_structured_events

    text = """
def copy_rows():
    with N.Kernel():
        x=N.alloc_ub((16,32),N.f32)
        y=N.alloc_ub((16,64),N.f32)
        z=N.alloc_ub((16,64),N.f32)
        N.fill(x,1.0)
        N.fill(y,2.0)
        N.copy(x[:,16:32],y[:,32:48])
        N.copy(y,z)
"""
    result = compile(N.from_source(text))
    tasks = [op for op in result.plan.walk() if isinstance(op, TaskOp)]
    assert any(task.native.data == "vector_copy_rows" for task in tasks)
    packet = audited_packet("vector_copy_rows")
    assert packet.closed_mask and MaskStep.SET_COLUMNS in packet.mask_steps
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    execution = check_structured_events(memory, logical_reference=original)
    assert execution.bytes_read and execution.bytes_written


@pytest.mark.parametrize("source,destination", [("4:20", "16:32"), ("0:16", "4:20")])
def test_pitched_vector_copy_preserves_native_origin_alignment_checks(source, destination):
    text = f"""
def unaligned_rows():
    with N.Kernel():
        x=N.alloc_ub((16,32),N.f32)
        y=N.alloc_ub((16,32),N.f32)
        N.fill(x,1.0)
        N.fill(y,0.0)
        N.copy(x[:,{source}],y[:,{destination}])
"""
    with pytest.raises(CompileError, match="32-byte alignment"):
        compile(N.from_source(text))


def test_pitched_vector_copy_rejects_transpose_instead_of_copying_the_wrong_axes():
    text = """
def transposed_rows():
    with N.Kernel():
        x=N.alloc_ub((16,32),N.f32)
        y=N.alloc_ub((16,64),N.f32)
        N.fill(x,1.0)
        N.copy(x[:,:16],y[:,:16],transpose=True)
"""
    with pytest.raises(CompileError, match="transposed copy requires"):
        compile(N.from_source(text))

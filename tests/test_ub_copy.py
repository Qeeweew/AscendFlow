"""Current source UB copy rejects exact, shifted and changing aliases."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError


@pytest.mark.parametrize("destination", ("0:16", "8:24"))
def test_exact_and_shifted_overlaps_are_refused_without_memmove_assumptions(destination):
    text = f"""
def overlap(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((32,),N.f32)
        for tile in N.Pipelined(n,stages=1):
            N.fill(x,1.0)
            N.copy(x[0:16],x[{destination}])
"""
    with pytest.raises(CompileError, match="UB-to-UB copy requires provably disjoint"):
        compile(N.from_source(text))


def test_generation_dependent_aliases_are_not_assumed_disjoint():
    text = """
def changing(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((3,8),N.f32)
        for tile in N.Pipelined(n,stages=1):
            N.fill(x,1.0)
            N.copy(x[tile%2,:],x[tile%3,:])
"""
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        compile(N.from_source(text))

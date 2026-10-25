"""Partial results borrow preserved entry bytes at their actual update cut."""

import pytest

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def overlay(*, initialized=True, overwrite=False, branch=False):
    initial = "N.copy(A,a1.at(0))" if initialized else "pass"
    if branch:
        initial = (
            "if n>8:\n            N.copy(A,a1.at(0))\n        else:\n            N.copy(B,a1.at(0))"
        )
    return N.from_source(
        f"""
def overlay(A:N.Tensor((16,16),N.bf16),B:N.Tensor((16,16),N.bf16),
            O:N.Tensor((16,16),N.f32,access="write"),n:N.i64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.bf16,slots=2)
        b1=N.alloc_l1((16,16),N.bf16)
        a0=N.alloc_l0a((16,16),N.bf16)
        b0=N.alloc_l0b((16,16),N.bf16)
        c0=N.alloc_l0c((16,16))
        {initial}
        value=N.copy(B,a1.at(0),valid_rows=N.min(n,16))
        {"N.copy(A,a1.storage[0,:,:])" if overwrite else "pass"}
        N.copy(value,a0)
        N.copy(B,b1)
        N.copy(b1,b0)
        N.matmul(a0,b0,accum=c0)
        N.copy(c0,O)
""",
        disjoint_external=True,
    )


@pytest.mark.parametrize("branch", [False, True])
def test_partial_value_inherits_defined_entry_region_at_the_update(branch):
    plan = synchronize_structure(overlay(branch=branch))
    for rows in (0, 1, 7, 8, 9, 16):
        check_compiled_events(plan, {"n": rows}, invocations=2)


def test_partial_value_does_not_inherit_a_later_definition():
    with pytest.raises(CompileError, match="initialized|logical Tile"):
        synchronize_structure(overlay(initialized=False, overwrite=True))


def test_raw_overwrite_after_the_partial_value_still_invalidates_it():
    with pytest.raises(CompileError, match="overwrite"):
        synchronize_structure(overlay(overwrite=True))


def test_partial_value_snapshot_composes_across_runtime_stages_and_independent_slots():
    program = N.from_source(
        """
def staged(A:N.Tensor((n,16,16),N.bf16),B:N.Tensor((n,16,16),N.bf16),
           O:N.Tensor((n,16,16),N.f32,access="write"),n:N.i64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.bf16,slots=3)
        b1=N.alloc_l1((16,16),N.bf16,slots=2)
        a0=N.alloc_l0a((16,16),N.bf16,slots=2)
        b0=N.alloc_l0b((16,16),N.bf16,slots=2)
        c0=N.alloc_l0c((16,16))
        for t in N.Pipelined(n):
            with N.Stage(0):
                N.copy(A[t,:,:],a1.at(t))
            with N.Stage(1):
                with N.CopyGroup():
                    value=N.copy(B[t,:,:],a1.at(t),valid_rows=t%17)
                    N.copy(B[t,:,:],b1.at(t))
            with N.Stage(2):
                N.copy(value,a0.at(t))
                N.copy(b1.at(t),b0.at(t))
                N.matmul(a0.at(t),b0.at(t),accum=c0)
                N.copy(c0,O[t,:,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(program)
    assert plan.admission.work["value_snapshot_queries"] > 0
    for count in (0, 1, 2, 3, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)

"""A real unconditional read retires branch writers; optional reads do not."""

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def program(optional):
    body = (
        """if t % 2 == 0:
                    N.copy(x,Y[t,:])"""
        if optional
        else "N.copy(x,Y[t,:])"
    )
    return N.from_source(
        f"""
def branches(X:N.Tensor((n,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),
             n:N.i64, split:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for t in N.Pipelined(n):
            if t<split:
                N.copy(X[t,:],x)
            else:
                N.fill(x,0.0)
            {body}
""",
        disjoint_external=True,
    )


@pytest.mark.parametrize("optional", [False, True])
def test_branch_writer_retirement_requires_a_real_unconditional_reader(optional):
    plan = synchronize_structure(program(optional))
    ub_waw = [
        need
        for need in plan.dependencies.needs
        if need.kind is HazardKind.WAW and need.cell.root.type.bank.data == "UB"
    ]
    assert ub_waw
    required = {id(need) for need in plan.dependencies.required}
    assert any(id(need) in required for need in ub_waw) == optional
    for count in (0, 1, 2, 5):
        for split in (0, 1, count):
            checked = check_compiled_events(plan, {"n": count, "split": split}, invocations=2)
            assert checked is not None


@pytest.mark.parametrize("condition", ["t<split or t%2==0", "t>=split and t%2==0"])
def test_short_circuit_branch_and_common_reader_keep_real_buffer_reuse(condition):
    source = N.from_source(
        f"""
def short_circuit(X:N.Tensor((n,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),
                  n:N.i64, split:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for t in N.Pipelined(n):
            if {condition}:
                N.copy(X[t,:],x)
            else:
                N.fill(x,0.0)
            N.copy(x,Y[t,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for count in (0, 1, 2, 7):
        for split in (0, 1, count):
            check_compiled_events(plan, {"n": count, "split": split}, invocations=2)

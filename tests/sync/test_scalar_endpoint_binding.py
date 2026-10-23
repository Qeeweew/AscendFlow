"""Notifications retain actual dominating short-circuit scalar results."""

import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


@pytest.mark.parametrize("form", ["short_circuit", "if_elif"])
def test_equivalent_conditional_copies_keep_actual_scalar_bindings(form):
    work = "N.copy(X,x)\nN.add(x,1.0,out=x)\nN.copy(x,Y[i,:])\n"
    body = "".join("                " + line for line in work.splitlines(keepends=True))
    branch = "            if i%2==0 or i==n-1:\n" + body
    if form == "if_elif":
        branch = "            if i%2==0:\n" + body + "            elif i==n-1:\n" + body
    source = (
        """
def condition(X:N.Tensor((64,),N.f32),
              Y:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),
              n:N.i64(minimum=0)):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((64,),N.f32)
        for i in N.serial(n):
"""
        + branch
    )
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    for count in (0, 1, 3, 4):
        check_compiled_events(plan, {"n": count})

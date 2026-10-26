"""A nonadjacent release must not share an inner loop's binary credit."""
import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events

SOURCE = '''
def nested(X:N.Tensor((n,4,64),N.bf16),
           Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),
           n:N.i64,Initial:N.i64):
    with N.Kernel(1,kind="vector"):
        raw=N.alloc_ub((64,),N.bf16)
        value=N.alloc_ub((64,),N.f32)
        total=N.alloc_ub((64,),N.f32)
        for row in N.serial(n):
            N.copy(X[row,0,:],raw)
            N.cast(raw,out=total)
            if Initial==0:
                for part in N.serial(1,4):
                    N.copy(X[row,part,:],raw)
                    N.cast(raw,out=value)
                    N.add(total,value,out=total)
            N.copy(total,Out[row,:])
'''


@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("children", [0, 1, 3])
def test_outer_release_cannot_be_consumed_by_an_inner_wait(conditional, children):
    source = SOURCE.replace("N.serial(1,4)", f"N.serial(1,{children + 1})")
    if not conditional:
        source = source.replace("            if Initial==0:\n", "")
        source = "\n".join(line[4:] if line.startswith("                ") else line
                           for line in source.splitlines())
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    for count in (0, 1, 2, 4):
        for initial in (0, 1):
            check_compiled_events(plan, {"n": count, "Initial": initial}, invocations=2)

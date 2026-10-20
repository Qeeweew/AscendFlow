"""Logical domains, actor-local policy, and transparent inline inheritance."""

import pytest

from ascendflow import language as N
from ascendflow.dialects.source import DomainOp, IterationOp, StageOp
from ascendflow.frontend import FrontendError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


@N.inline
def import_and_cast(X, x, y, g):
    for child in N.Pipelined(1):
        with N.Stage(0, name="load"):
            value = N.copy(X[g, :, :], x)
        with N.Stage(1, name="compute"):
            N.cast(value, out=y)


def test_labels_are_local_to_actual_actor_and_domain_after_inline():
    source = N.from_source(
        """
def scopes(X:N.Tensor((n,16,16),N.f16),n:N.int64):
    with N.Kernel(kind="mixed"):
        a=N.alloc_l1((16,16))
        b=N.alloc_l0a((16,16))
        x,y=N.alloc_ub((16,16),N.f16),N.alloc_ub((16,16),N.f32)
        for g in N.Dataflow(n,issue_order={"cube":("load","compute"),"vector":("load",)}):
            with N.Cube(),N.Stage(0,name="load"):
                N.copy(X[g,:,:],a)
            with N.Cube(),N.Stage(1,name="compute"):
                N.copy(a,b)
            with N.Vector(),N.Stage(0,name="load"):
                import_and_cast(X,x,y,g)
""",
        functions={"import_and_cast": import_and_cast},
        disjoint_external=True,
    )
    ir = source.to_ir()
    loops = [op for op in ir.walk() if isinstance(op, IterationOp)]
    assert len(loops) == 2 and loops[0] is not loops[1]
    assert "stages" not in loops[0].attributes
    stages = [op for op in ir.walk() if isinstance(op, StageOp)]
    assert [stage.family.data for stage in stages] == ["cube", "cube", "vector", "vector", "vector"]
    assert isinstance(stages[0].parent_op(), DomainOp)
    result = synchronize_structure(source)
    for count in (0, 1, 2, 4):
        checked = check_compiled_events(result, {"n": count})
        assert bool(checked.natives) == bool(count)


@pytest.mark.parametrize(
    "loop,scopes,message",
    [
        ("N.Pipelined(n)", "with N.Vector():", "actor-local"),
        ("N.Dataflow(n)", "with N.Stage(0),N.Vector():", "enclosing Cube or Vector"),
        ("N.Dataflow(n)", "with N.Stage(0):", "enclosing Cube or Vector"),
        ("N.Dataflow(n,stages=2)", "with N.Vector():", "one count"),
    ],
)
def test_mixed_domain_rejects_ambiguous_actor_or_removed_stage_declaration(loop, scopes, message):
    source = N.from_source(f"""
def invalid(n:N.int64):
    with N.Kernel(kind="mixed"):
        x=N.alloc_ub((64,))
        for g in {loop}:
            {scopes}
                N.fill(x,1.0)
""")
    with pytest.raises(FrontendError, match=message):
        source.to_ir()


@pytest.mark.parametrize(
    "policy", ["stages=2", 'order="source"', 'issue_order={"vector":("work",)}']
)
def test_local_pipeline_accepts_only_local_policy(policy):
    source = N.from_source(f"""
def invalid(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for g in N.Pipelined(n,{policy}):
            with N.Stage(0,name="work"):
                N.fill(x,1.0)
""")
    with pytest.raises(FrontendError, match="one count|tuple of local"):
        source.to_ir()

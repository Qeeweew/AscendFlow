"""Actual source clocks and actor participation, without native expansion."""

import pytest
from xdsl.dialects import scf
from xdsl.ir import Block, Operation

from tests.fixtures.attention_target import attention_target
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.source import CopyOp, ElementOp, MatmulOp, PipelineOp, ProgramOp, ReduceOp
from xdsl_npu.source_scopes import Scope, Work, collect_source_scopes

MIXED_CHILDREN = """
    def nested(X: N.Tensor((n,16,16),N.f16), n: N.int64,
               PARTS: N.constexpr=2):
        with N.Kernel(kind="mixed"):
            q=N.alloc_l1((16,16))
            x=N.alloc_ub((64,))
            for task in N.Pipelined(n,stages=2):
                with N.Stage(0,name="body"):
                    with N.Vector():
                        N.fill(x,1.0)
                    with N.Cube():
                        N.copy(X[task,:,:],q)
                    for part in N.Pipelined(PARTS,stages=1):
                        with N.Cube():
                            N.copy(X[task,:,:],q)
                    with N.Vector():
                        N.fill(x,2.0)
                    for shared in N.Pipelined(3,stages=2):
                        with N.Stage(0,name="cube"),N.Cube():
                            N.copy(X[task,:,:],q)
                        with N.Stage(1,name="vector"),N.Vector():
                            N.fill(x,3.0)
                    with N.Cube():
                        N.copy(X[task,:,:],q)
                with N.Stage(1,name="after"),N.Vector():
                    N.fill(x,4.0)
"""


def program(module):
    return next(op for op in module.walk() if isinstance(op, ProgramOp))


def coordinates(value, seen=None):
    """Inspect address dependencies independently from collected scope clocks."""
    seen = set() if seen is None else seen
    if value in seen:
        return set()
    seen.add(value)
    if isinstance(value.owner, Block):
        return {value}
    return set().union(*(coordinates(operand, seen) for operand in value.owner.operands))


def test_ordinary_effect_order_and_child_actor_subsets_are_preserved():
    module = N.from_source(MIXED_CHILDREN).to_ir()
    root = collect_source_scopes(program(module))
    (parent,) = root.items
    assert isinstance(parent, Scope)
    before_v, before_c, cube_child, between, shared_child, after_c, after_v = parent.items
    assert all(isinstance(item, Work) for item in (before_v, before_c, between, after_c, after_v))
    assert isinstance(cube_child, Scope) and isinstance(shared_child, Scope)
    assert cube_child.actors == {"cube"}
    assert shared_child.actors == parent.actors == {"cube", "vector"}
    assert parent.items_for("vector") == (before_v, between, shared_child, after_v)
    assert parent.items_for("cube") == (before_c, cube_child, shared_child, after_c)
    assert tuple(item.context.path for item in parent.items) == tuple(
        sorted(item.context.path for item in parent.items)
    )
    assert [item.op.scalar.value.data for item in parent.walk_work("vector")] == [1, 2, 3, 4]
    assert before_v.clock.stage.label.data == "body" and before_v.clock.offset == 0
    assert after_v.clock.stage.label.data == "after" and after_v.clock.offset == 1
    assert cube_child.parent_clocks[-1].coordinate is parent.clock.coordinate
    assert cube_child.parent_clocks[-1].stage is before_v.clock.stage
    assert shared_child.context.actor is None
    assert all(
        work.context.owner is work.context.domain.body.block.args[0] for work in root.walk_work()
    )
    actual = [op for op in module.walk() if isinstance(op, (CopyOp, ElementOp, MatmulOp, ReduceOp))]
    assert [work.op for work in root.walk_work()] == actual


@pytest.mark.parametrize("parts", [0, 1, 1_000_000])
def test_fixed_child_trip_count_does_not_copy_source_work(parts):
    module = N.from_source(MIXED_CHILDREN).specialize(PARTS=parts).to_ir()
    root = collect_source_scopes(program(module))
    (parent,) = root.items
    cube_child = next(item for item in parent.items if isinstance(item, Scope))
    assert cube_child.clock.bound == parts
    assert cube_child.clock.coordinate is cube_child.op.body.block.args[0]
    assert len(tuple(cube_child.walk_work())) == 1
    assert len(tuple(root.walk_work())) == 8


def test_guard_branches_and_parent_stage_are_captured_on_child_entry():
    module = N.from_source("""
        def guarded(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(n,stages=2):
                    with N.Stage(0,name="before"):
                        if task == 0:
                            N.fill(x,0.0)
                            for part in N.Pipelined(3,stages=2):
                                with N.Stage(1,name="child"):
                                    N.fill(x,1.0)
                        else:
                            N.fill(x,2.0)
                    with N.Stage(1,name="after"):
                        N.fill(x,3.0)
    """).to_ir()
    (parent,) = collect_source_scopes(program(module)).items
    before, child, otherwise, after = parent.items
    branch = next(op for op in module.walk() if isinstance(op, scf.IfOp))
    assert before.context.guards[0].op is branch
    assert before.context.guards[0].taken
    assert child.context.guards == before.context.guards
    assert otherwise.context.guards[0].op is branch
    assert not otherwise.context.guards[0].taken
    assert not after.context.guards
    (child_work,) = child.items
    assert child_work.context.guards[0].condition is branch.cond
    assert child.parent_clocks[-1].coordinate is parent.clock.coordinate
    assert child.parent_clocks[-1].offset == 0
    assert child_work.parent_clocks == child.parent_clocks
    assert child_work.clock.coordinate is child.clock.coordinate
    assert child_work.clock.offset == 1
    assert after.clock.coordinate is parent.clock.coordinate and after.clock.offset == 1


@pytest.mark.parametrize("continuous", [False, True])
def test_real_attention_scope_clocks_survive_constant_and_parent_independent_addresses(continuous):
    module = attention_target(continuous=continuous).to_ir()
    root = collect_source_scopes(program(module))
    scopes = tuple(root.walk_scopes())
    pipelines = [scope for scope in scopes if isinstance(scope.op, PipelineOp)]
    assert len(pipelines) == 5
    assert len([scope for scope in scopes if isinstance(scope.op, scf.ForOp)]) == int(
        not continuous
    )
    assert len(tuple(root.walk_work())) == 37
    assert len({id(work.op) for work in root.walk_work()}) == 37
    assert any(
        isinstance(work.op, ElementOp)
        and work.op.kind.data == "fill"
        and work.op.scalar.value.data == float("-inf")
        for work in root.walk_work()
    )
    outer = pipelines[0]
    assert outer.clock.bound is outer.op.limit
    assert root.actors == {"cube", "vector"}
    assert [scope.actors for scope in pipelines[1:]] == [{"cube"}, {"vector"}, {"cube"}, {"vector"}]
    outputs = {
        work.op.destination.owner.source.owner.sym_name.data: work
        for work in root.walk_work()
        if isinstance(work.op, CopyOp)
        and work.op.destination.owner.source.owner.sym_name.data in {"Score", "P", "PV", "O"}
    }
    score = outputs["Score"]
    assert score.parent_clocks[-1].coordinate is outer.clock.coordinate
    assert score.parent_clocks[-1].stage.label.data == "qk"
    assert score.parent_clocks[-1].offset == 0
    assert score.clock.stage.label.data == "export_score" and score.clock.offset == 3
    assert score.clock.op is not outer.op
    probability = outputs["P"]
    assert probability.parent_clocks[-1].coordinate is outer.clock.coordinate
    assert (probability.parent_clocks[-1].offset, probability.clock.offset) == (1, 2)
    output = outputs["O"]
    assert output.parent_clocks[-1].coordinate is outer.clock.coordinate
    assert (output.parent_clocks[-1].offset, output.clock.offset) == (2, 2)
    assert outer.clock.coordinate in coordinates(output.context.guards[-1].condition)
    address_clocks = set().union(
        *(coordinates(origin) for origin in output.op.destination.owner.origins)
    )
    assert (outer.clock.coordinate in address_clocks) == continuous
    pv = outputs["PV"]
    assert not set().union(*(coordinates(origin) for origin in pv.op.source.owner.origins))
    assert pv.clock.coordinate is outer.clock.coordinate
    assert pv.clock.stage.label.data == "pv" and pv.clock.offset == 1
    position = outer.items.index(pv)
    child = outer.items[position - 1]
    assert isinstance(child, Scope) and child.clock.bound == 8
    assert child.parent_clocks[-1].coordinate is outer.clock.coordinate
    mad = next(work for work in child.walk_work() if isinstance(work.op, MatmulOp))
    assert mad.op.accumulator.owner.source is pv.op.source.owner.source
    assert mad.clock.coordinate is child.clock.coordinate


def test_unknown_source_effect_cannot_disappear_from_collection():
    class UnexpectedEffect(Operation):
        name = "unexpected.effect"

    module = N.from_source("""
        def unknown():
            with N.Kernel():
                pass
    """).to_ir()
    source = program(module)
    source.body.block.insert_op_before(UnexpectedEffect.create(), source.body.block.last_op)
    with pytest.raises(CompileError, match="unexpected.effect; effects cannot be ignored"):
        collect_source_scopes(source)

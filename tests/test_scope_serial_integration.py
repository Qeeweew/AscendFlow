"""Three real clocks reach physical IR; cross-query proof is still mandatory."""

from collections import Counter

import pytest
from xdsl.dialects import scf

from tests.fixtures.attention_target import attention_target
from tests.fixtures.scoped_queries import scoped_query_recurrence
from xdsl_npu import compile
from xdsl_npu.budget import limit
from xdsl_npu.ir import FrameOp, ScopeGuardOp, TaskOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.sync import autosync, verify_plan


@pytest.fixture(scope="module", params=["recurrence", "attention"])
def physical(request):
    source = scoped_query_recurrence if request.param == "recurrence" else attention_target()
    with limit(40):
        module = lower_source(source)
        program = extract_scopes(module)
    return request.param, module, program


def test_actual_query_kv_child_clocks_and_guards_survive_physical_lowering(physical):
    name, module, program = physical
    assert set(program.invocations) == {"aic", "aiv0", "aiv1"}
    assert not any(isinstance(op, FrameOp) for op in module.walk())
    for actor, invocation in program.invocations.items():
        query = invocation.op
        assert isinstance(query, scf.ForOp)
        assert program.parents[actor].parent is query.body.block
        assert invocation.cyclic is not None
        for leaf in program.leaves.values():
            if leaf.actor != actor:
                continue
            assert leaf.call.generation is program.parents[actor].body.block.args[0]
            assert leaf.stream.limit is None and leaf.stream.count is not None
            assert leaf.call.generation is not query.body.block.args[0]
            assert leaf.stream.body.block.args[0] is not leaf.call.generation
            for op in leaf.stream.walk():
                if isinstance(op, ScopeGuardOp):
                    assert op.parent_generation is leaf.call.generation
                    assert op.child_generation is leaf.stream.body.block.args[0]
    natives = Counter(op.native.data for op in module.walk() if isinstance(op, TaskOp))
    if name == "attention":
        assert natives["mad"] + natives["mad_first"] == 2
        assert natives["exp"] == 4  # One source pair on each physical AIV.
        assert {op.kind.data for op in module.walk() if isinstance(op, ScopeGuardOp)} == {
            "first",
            "last",
        }
    else:
        assert sum(natives.values()) == 16


def test_physical_query_scope_cannot_bypass_cross_query_verification(physical):
    _, module, _ = physical
    before = dump(module)
    diagnostic = "cross-query memory and event lifetime synchronization"
    for action in (
        autosync,
        verify_plan,
        lambda value: compile(value, synchronize=False, timeout_s=40),
    ):
        with pytest.raises(CompileError, match=diagnostic):
            action(module)
        assert dump(module) == before

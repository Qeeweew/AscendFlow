"""Three real clocks reach physical IR; cross-query proof is still mandatory."""

from collections import Counter

import pytest
from xdsl.dialects import scf

from tests.fixtures.attention_target import attention_target
from tests.fixtures.scoped_queries import scoped_query_recurrence
from xdsl_npu import compile
from xdsl_npu.budget import limit
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import FrameOp, ScopeGuardOp, TaskOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.sync import autosync, verify_plan


@pytest.fixture(scope="module")
def recurrence():
    with limit(40):
        return lower_source(scoped_query_recurrence)


@pytest.fixture(scope="module", params=["recurrence", "attention"])
def physical(request, recurrence):
    with limit(40):
        module = recurrence if request.param == "recurrence" else lower_source(attention_target())
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


def test_unsynchronized_query_scope_rejects_real_missing_child_sync(physical):
    name, module, _ = physical
    before = dump(module)
    with limit(30), pytest.raises(CompileError, match="missing loop (RAW|WAR|WAW)"):
        verify_plan(module)
    assert dump(module) == before
    if name == "recurrence":
        with pytest.raises(CompileError, match="missing loop (RAW|WAR|WAW)"):
            compile(module, synchronize=False, timeout_s=30)
        assert dump(module) == before


def test_recurrence_autosync_rejects_its_exhausted_actual_binary_event_bank(recurrence):
    before = dump(recurrence)
    with (
        limit(40),
        pytest.raises(
            CompileError,
            match=r"\('aic', 'MTE1', 'MTE2'\): query release has no free binary event ID",
        ),
    ):
        autosync(recurrence)
    assert dump(recurrence) == before

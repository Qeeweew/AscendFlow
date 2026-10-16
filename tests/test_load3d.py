"""Load3D selects identical physical effects and checks invariant register state."""

import pytest
from xdsl.dialects.builtin import IntAttr

from examples.gemm.batched import batched_matmul
from xdsl_npu import compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import Load3DConfigOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.sync import verify_plan


@pytest.fixture(scope="module")
def artifacts():
    source = batched_matmul.specialize(TRANSPOSE_B=False)
    return compile(source), compile(source, load3d=True)


def test_load3d_keeps_native_accesses_slots_and_issue_order(artifacts):
    old, new = artifacts
    a, b = (extract_loop(x.plan).actors["aic"].program for x in (old, new))
    assert a.storages == b.storages
    assert [(t.uid, t.refs, t.stage, t.issue_order) for t in a.tasks] == [
        (t.uid, t.refs, t.stage, t.issue_order) for t in b.tasks
    ]
    assert {t.native for t in b.tasks} >= {"load3d_a", "load3d_b_nn"}
    assert new.asc.count("asc_set_l13d_fmatrix(") == 1
    assert new.asc.count("PIPE_S, PIPE_MTE1") == 2


@pytest.mark.parametrize("mutation", ["missing", "wrong", "duplicate"])
def test_missing_or_changed_load3d_register_contract_is_rejected(artifacts, mutation):
    plan = artifacts[1].plan
    op = next(op for op in plan.walk() if isinstance(op, Load3DConfigOp))
    if mutation == "missing":
        op.parent.erase_op(op)
    elif mutation == "wrong":
        op.attributes["a_width"] = IntAttr(16)
    else:
        op.parent.insert_op_before(op.clone(), op)
    with pytest.raises(CompileError, match="Load3D"):
        verify_plan(plan)

"""Owned serial plans retain completion facts; imports still check actual IR."""

import pytest
from xdsl.dialects.builtin import IntAttr

from examples.gemm.hierarchical import hierarchical_gemm
from examples.state.grouped import grouped_state_sum
from examples.state.recurrence import state_sum
from xdsl_npu import compile, invocation_sync, loop_graph, sequence_calls, sequence_sync
from xdsl_npu.ir import RegionSyncOp, TaskOp
from xdsl_npu.model import CompileError
from xdsl_npu.region_completion import RegionCompletion


def test_sequence_planning_composes_one_native_transfer_proof_without_final_rebuild(monkeypatch):
    proofs = []

    class ObservedCompletion(RegionCompletion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            proofs.append(self)

    def repeated(*args, **kwargs):
        pytest.fail("normal sequence compilation rebuilt an established proof")

    monkeypatch.setattr(sequence_sync, "RegionCompletion", ObservedCompletion)
    monkeypatch.setattr(sequence_calls, "RegionCompletion", ObservedCompletion)
    monkeypatch.setattr(sequence_sync, "verify_sequence", repeated)
    monkeypatch.setattr(sequence_sync, "_verify_leaf", repeated)
    monkeypatch.setattr(sequence_calls, "verify_calls", repeated)
    artifact = compile(grouped_state_sum.specialize(B=16, GROUPS=1), timeout_s=10)
    assert len(proofs) == 1 and proofs[0].calls == (0, 1)
    assert artifact.report["sequence_channels"] > 0
    assert artifact.report["invocation_memory_obligations"] > 0
    assert artifact.report["invocation_event_domains"] > 0
    assert (
        artifact.report["verification_origin"] == "composed owned sequence transformation contracts"
    )


def test_invocation_planning_retains_its_child_graph_and_completed_results(monkeypatch):
    proofs, completed = [], []
    original_proof = invocation_sync._proof
    original_complete = invocation_sync.InvocationPlan.complete

    def counted(*args, **kwargs):
        result = original_proof(*args, **kwargs)
        proofs.append(result)
        return result

    def complete(plan, loop, channels, graph, streams):
        assert plan.graph is graph
        completed.append(plan)
        return original_complete(plan, loop, channels, graph, streams)

    def repeated(*args, **kwargs):
        pytest.fail("normal invocation compilation rebuilt its established child or parent proof")

    monkeypatch.setattr(invocation_sync, "_proof", counted)
    monkeypatch.setattr(invocation_sync, "verify_invocation", repeated)
    monkeypatch.setattr(invocation_sync.InvocationPlan, "complete", complete)
    # _plan_leaf owns its already imported completion_graph function. A second
    # invocation-local lazy import must never reconstruct that same graph.
    monkeypatch.setattr(loop_graph, "completion_graph", repeated)
    artifact = compile(hierarchical_gemm.specialize(M=16, N=16, BK=16, GROUPS=2), timeout_s=10)
    assert len(proofs) == len(completed) == 1
    assert completed[0].proof is proofs[0]
    assert completed[0].memory_coverage and completed[0].resource_coverage
    assert artifact.report["invocation_memory_obligations"] == len(completed[0].memory)


@pytest.mark.parametrize("damage", ("remove-wait", "native-stage"))
def test_sequence_materialization_checks_actual_mapping_and_native_preservation(
    monkeypatch, damage
):
    install = sequence_sync._install

    def corrupt(sequence, bridges):
        install(sequence, bridges)
        if damage == "remove-wait":
            operation = next(
                op
                for op in sequence.module.walk()
                if isinstance(op, RegionSyncOp) and op.action.data == "wait"
            )
            operation.parent.erase_op(operation)
        else:
            operation = next(op for op in sequence.module.walk() if isinstance(op, TaskOp))
            operation.attributes["stage"] = IntAttr(operation.stage.data + 1)

    monkeypatch.setattr(sequence_sync, "_install", corrupt)
    with pytest.raises(CompileError, match="(unpaired or placed|admitted native)"):
        compile(state_sum.specialize(B=16, GROUPS=1), timeout_s=10)


def test_invocation_completion_rejects_actual_native_damage_without_rebuilding(monkeypatch):
    plan_invocation = invocation_sync.plan_invocation

    def corrupt(loop, **kwargs):
        result = plan_invocation(loop, **kwargs)
        operation = next(op for op in loop.op.walk() if isinstance(op, TaskOp))
        operation.attributes["stage"] = IntAttr(operation.stage.data + 1)
        return result

    monkeypatch.setattr(invocation_sync, "plan_invocation", corrupt)
    with pytest.raises(CompileError, match="owned native/protocol snapshot"):
        compile(hierarchical_gemm.specialize(M=16, N=16, BK=16, GROUPS=2), timeout_s=10)

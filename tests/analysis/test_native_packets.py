"""CANN issue packets retain configuration effects without completion claims."""

from dataclasses import replace

import pytest

from tests.fixtures.pipelines import vector_pipeline
from tests.legacy.passes.insert_sync import synchronize_local
from tests.legacy.passes.local_sync import select_local_protocols
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis.frontier import Clock
from xdsl_npu.analysis.native import NativeContract, extract_native_tasks
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.target.ascend910b import Engine
from xdsl_npu.target.packets import MaskStep, NativePacket, audited_packet


@pytest.mark.parametrize(
    "native", ["muls", "add", "convert_rn", "vector_copy", "reduce_sum64", "stat_fill", "mul_rows"]
)
def test_count_calls_are_closed_and_restore_only_normal_mode(native):
    packet = audited_packet(native)
    assert packet is not None and packet.closed_mask and packet.each_count_call
    assert packet.mask_steps == (
        MaskStep.SET_COUNTER,
        MaskStep.SET_COUNT,
        MaskStep.VECTOR_READ,
        MaskStep.RESTORE_NORMAL,
    )
    assert packet.restores_normal_mode
    assert packet.clobbers == {"VECTOR_MASK_MODE", "VECTOR_MASK"}


@pytest.mark.parametrize("native", ["copy", "brcb8", "scalar_load", "fix", "load3d_a"])
def test_mask_free_packets_do_not_invent_mask_consumers(native):
    packet = audited_packet(native)
    assert packet is not None and packet.closed_mask
    assert packet.mask_steps == () and not packet.clobbers


def test_old_contract_constructor_and_replace_derive_the_correct_packet_once():
    original = NativeContract("muls", Engine.V, ("R", "W"), (), 2.0)
    assert original.packet is not None and original.packet.each_count_call
    assert original.packet is audited_packet("muls")
    assert not original.implicit_completion
    changed = replace(original, native="merge_sum_packed", params=(2, 3))
    assert changed.packet is not original.packet
    assert changed.packet.mask_steps == (
        MaskStep.SET_NORMAL,
        MaskStep.SET_PARTIALS,
        MaskStep.VECTOR_READ,
    )
    assert not changed.packet.restores_normal_mode
    assert audited_packet("unmodeled_future_native") is None


def test_binding_preserves_admitted_packet_and_actual_task_identity():
    loop = extract_loop(vector_pipeline())
    data = next(iter(loop.actors.values()))
    tasks = extract_native_tasks(loop, clock=Clock("packets"))
    for decoded, bound in zip(data.owned_effects.tasks, tasks, strict=True):
        assert bound.operation is decoded.operation
        assert bound.packet is decoded.packet is decoded.contract.packet
        assert bound.accesses is decoded.accesses


@pytest.mark.parametrize(
    "packet",
    [
        None,
        NativePacket("muls"),
        NativePacket("muls", (MaskStep.VECTOR_READ,)),
        replace(audited_packet("muls")),
    ],
)
def test_local_admission_rejects_missing_or_incoming_mask_packets(packet):
    loop = extract_loop(vector_pipeline())
    task = next(t for t in extract_native_tasks(loop, clock=Clock("missing")) if t.native == "muls")
    with pytest.raises(CompileError, match="audited closed actor issue packet"):
        select_local_protocols((replace(task, packet=packet),), ())


def test_masked_partial_recipe_reaches_real_local_protocol_selection():
    kernel = Kernel("closed_partial_packet")
    partials = kernel.buffer("partials", "UB", 64)
    output = kernel.buffer("statistics", "UB", 32)
    kernel.loop(4, symbolic=True)
    kernel.task("init", "fill", kernel.view(partials, 64))
    kernel.task(
        "merge",
        "merge_sum_packed",
        kernel.view(partials, 12, rows=2, row_stride=32),
        kernel.view(output, 8),
        params=(2, 3),
    )
    plan = synchronize_local(kernel.finish())
    merge = next(task for task in plan.selection.tasks if task.native == "merge_sum_packed")
    assert merge.packet.closed_mask and not merge.registers
    assert merge.accesses[0].ref.segments(0) == ((0, 12), (32, 44))
    assert not merge.implicit_completion
    assert any(protocol.destination is merge for protocol in plan.selection.protocols)

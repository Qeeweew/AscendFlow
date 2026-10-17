"""CANN issue packets retain configuration effects without completion claims."""

from dataclasses import replace

import pytest

from ascendflow.analysis import native
from ascendflow.analysis.native import NativeContract
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Engine
from ascendflow.target.packets import MaskStep, NativePacket, audited_packet
from examples.vector.scale import scale


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


def test_contract_constructor_and_replace_derive_the_correct_packet_once():
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


@pytest.mark.parametrize(
    "packet", [None, NativePacket("muls"), NativePacket("muls", (MaskStep.VECTOR_READ,))]
)
def test_native_admission_rejects_missing_or_incoming_mask_packets(monkeypatch, packet):
    rule = native.audited_packet
    monkeypatch.setattr(
        native, "audited_packet", lambda name: packet if name == "muls" else rule(name)
    )
    with pytest.raises(CompileError, match="audited closed instruction packet"):
        collect_native_memory(lower_native_structure(scale).module)

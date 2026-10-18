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
    "native",
    [
        "muls",
        "add",
        "convert_rn",
        "vector_copy",
        "reduce_sum64",
        "stat_fill",
        "mul_rows",
        "mask_and",
        "mask_zero",
        "mask_not",
    ],
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


@pytest.mark.parametrize(
    "native",
    [
        "copy",
        "brcb8",
        "scalar_load",
        "fix",
        "load3d_a",
        "nd2nz_pad",
        "compare_gt",
        "compare_scalar_eq",
    ],
)
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


def test_masked_packet_needs_local_mode_and_repeat_bitmap():
    packet = audited_packet("mask_fill")
    assert packet.closed_mask and packet.each_mask_call
    assert packet.mask_steps == (
        MaskStep.SET_NORMAL,
        MaskStep.SET_COORD,
        MaskStep.VECTOR_READ,
    )
    assert packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}
    assert not packet.restores_normal_mode
    changed = replace(packet, mask_steps=(MaskStep.SET_COORD, MaskStep.VECTOR_READ))
    assert not changed.closed_mask
    mode_only = replace(packet, mask_steps=(MaskStep.SET_NORMAL, MaskStep.VECTOR_READ))
    assert not mode_only.closed_mask


def test_each_mask_replay_path_is_a_closed_packet():
    # The emitted affine replay chooses at runtime between an unproven
    # per-chunk coordinate bitmap, a full-tile run that sets one all-ones
    # bitmap per bounded repeat group, and a fully masked tile that issues only
    # the local normal-mode setup. The recorded native contract is the same for
    # all three, and every path closes on its own.
    packet = audited_packet("mask_fill")
    assert packet.closed_mask and packet.each_mask_call
    assert packet.alternate_mask_steps == (
        (MaskStep.SET_NORMAL, MaskStep.SET_FULL, MaskStep.VECTOR_READ),
        (MaskStep.SET_NORMAL,),
    )
    assert packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}
    # Each alternate path is closed on its own: the full replay may not read
    # before setting its all-ones bitmap.
    unset_full = replace(
        packet,
        alternate_mask_steps=((MaskStep.SET_NORMAL, MaskStep.VECTOR_READ), (MaskStep.SET_NORMAL,)),
    )
    assert not unset_full.closed_mask
    global_mode = replace(
        packet,
        alternate_mask_steps=((MaskStep.VECTOR_READ,), (MaskStep.SET_NORMAL,)),
    )
    assert not global_mode.closed_mask


def test_select_packet_requires_local_full_mask_and_cmpmask():
    packet = audited_packet("select_masked")
    assert packet.closed_mask and not packet.each_mask_call
    assert packet.internal_fences == ("V",)
    assert "CMPMASK" in packet.clobbers
    assert MaskStep.SELECT_READ in packet.mask_steps
    no_cmp = replace(
        packet,
        mask_steps=(MaskStep.SET_NORMAL, MaskStep.SET_FULL, MaskStep.SELECT_READ),
    )
    assert not no_cmp.closed_mask
    no_full = replace(
        packet,
        mask_steps=(MaskStep.SET_NORMAL, MaskStep.SET_CMPMASK, MaskStep.SELECT_READ),
    )
    assert not no_full.closed_mask


def test_each_conditional_mask_path_must_close_without_another_paths_setup():
    packet = audited_packet("nd2nz_pad")
    assert packet.closed_mask
    changed = replace(packet, alternate_mask_steps=((MaskStep.VECTOR_READ,),))
    assert not changed.closed_mask

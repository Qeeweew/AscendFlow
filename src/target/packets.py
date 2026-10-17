"""Audited CANN actor issue packets, without hardware latch or retirement facts.

Each packet stays inside one actual TaskOp. Its mask steps describe the
retained CANN call order; they are not independently movable instructions.
The execution engines of the mask setters remain deliberately unspecified.
"""

from dataclasses import dataclass
from enum import Enum


class MaskStep(str, Enum):
    SET_COUNTER = "set-mode-counter"
    SET_NORMAL = "set-mode-normal"
    SET_COUNT = "set-mask-from-count-argument"
    SET_FULL = "set-mask-full-f32"
    SET_PARTIALS = "set-mask-from-partials-parameter"
    VECTOR_READ = "vector-read-mode-and-mask"
    RESTORE_NORMAL = "restore-mode-normal"


@dataclass(frozen=True, slots=True)
class NativePacket:
    native: str
    mask_steps: tuple[MaskStep, ...] = ()
    # A row loop may call a count C API repeatedly. Each such call contains
    # this complete sequence, rather than one setup surrounding all rows.
    each_count_call: bool = False
    internal_fences: tuple[str, ...] = ()

    @property
    def closed_mask(self):
        mode = mask = False
        for step in self.mask_steps:
            if step in {MaskStep.SET_COUNTER, MaskStep.SET_NORMAL, MaskStep.RESTORE_NORMAL}:
                mode = True
            elif step in {MaskStep.SET_COUNT, MaskStep.SET_FULL, MaskStep.SET_PARTIALS}:
                mask = True
            elif step is MaskStep.VECTOR_READ and not (mode and mask):
                return False
        return True

    @property
    def clobbers(self):
        # Restore-to-normal is a defined write, not preservation of an
        # arbitrary incoming mode. None of the current packets saves mask.
        result = set()
        for step in self.mask_steps:
            if step in {MaskStep.SET_COUNTER, MaskStep.SET_NORMAL, MaskStep.RESTORE_NORMAL}:
                result.add("VECTOR_MASK_MODE")
            elif step in {MaskStep.SET_COUNT, MaskStep.SET_FULL, MaskStep.SET_PARTIALS}:
                result.add("VECTOR_MASK")
        return frozenset(result)

    @property
    def restores_normal_mode(self):
        return MaskStep.RESTORE_NORMAL in self.mask_steps


_COUNT = frozenset(
    {
        "convert_f32",
        "convert_rn",
        "muls",
        "adds",
        "scale_value",
        "sub_value",
        "add",
        "mul",
        "sub",
        "div",
        "max",
        "exp",
        "rsqrt",
        "sqrt",
        "vector_copy",
        "fill",
        "fill_neg_inf",
        "reduce_rows64",
        "reduce_sum8",
        "reduce_max8",
        "reduce_sum64",
        "reduce_max64",
        "merge_sum",
        "merge_max",
        "stat_muls",
        "stat_adds",
        "stat_rsqrt",
        "stat_sqrt",
        "mul_rows",
        "sub_rows",
        "add_rows",
        "div_rows",
        "mul_gamma",
        "add_bias",
        "stat_fill",
        "stat_div",
    }
)
_FULL = frozenset(
    {
        "broadcast_rows64",
        "mul_row_broadcast",
        "sub_row_broadcast",
        "div_row_broadcast",
        "mul_columns64",
        "fma_columns64",
        "reduce_tile_sum64",
        "reduce_tile_max64",
    }
)
_PARTIAL = frozenset({"merge_sum_packed", "merge_max_packed"})
_MASK_FREE = frozenset(
    {
        "copy",
        "copy_rows",
        "scalar_load",
        "brcb8",
        "nd2nz",
        "load2d_a",
        "load2d_b_nt",
        "load2d_b_nn",
        "load3d_a",
        "load3d_b_nn",
        "mad",
        "mad_cycle",
        "mad_first",
        "mad16",
        "fix",
        "fix16",
    }
)


def _packet(native: str) -> NativePacket | None:
    if native == "nd2nz_pad":
        # L1 fill and ND2NZ share MTE2 but can overlap execution. The fill
        # must retire before the valid values overwrite its zero padding.
        return NativePacket(native, internal_fences=("MTE2",))
    if native in _COUNT:
        return NativePacket(
            native,
            (
                MaskStep.SET_COUNTER,
                MaskStep.SET_COUNT,
                MaskStep.VECTOR_READ,
                MaskStep.RESTORE_NORMAL,
            ),
            each_count_call=True,
        )
    if native in _FULL | _PARTIAL:
        return NativePacket(
            native,
            (
                MaskStep.SET_NORMAL,
                MaskStep.SET_FULL if native in _FULL else MaskStep.SET_PARTIALS,
                MaskStep.VECTOR_READ,
            ),
        )
    if native in _MASK_FREE:
        return NativePacket(native)
    return None


_AUDITED = {
    native: _packet(native) for native in _COUNT | _FULL | _PARTIAL | _MASK_FREE | {"nd2nz_pad"}
}


def audited_packet(native: str) -> NativePacket | None:
    """Return the target's fixed descriptor, without decoding native geometry.

    A hand-built descriptor is not an audit. Stable target-table identity lets
    admission retain that distinction without an external verified attribute.
    """
    return _AUDITED.get(native)

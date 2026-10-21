"""Audited CANN actor issue packets, without hardware latch or retirement facts.

Each packet stays inside one actual TaskOp. Its mask steps describe the
retained CANN call order; they are not independently movable instructions.
The execution engines of the mask setters remain deliberately unspecified.

The packet state below distinguishes the actual VECTOR_MASK mode, what the
configured bits mean (a repeat bitmap, a Counter element count, or a full
mask) and whether a selection CMPMASK was established in this packet. It is a
path-local description of one audited call sequence, not a new latch model:
``count_end`` only restores normal mode and restores no bit values.
"""

from dataclasses import dataclass
from enum import Enum


class MaskStep(str, Enum):
    SET_COUNTER = "set-mode-counter"
    SET_NORMAL = "set-mode-normal"
    SET_COUNT = "set-mask-from-count-argument"
    SET_FULL = "set-mask-full-f32"
    SET_PARTIALS = "set-mask-from-partials-parameter"
    SET_COLUMNS = "set-mask-from-min64-columns-parameter"
    SET_COORD = "set-mask-bits-from-lane-predicate"
    SET_CMPMASK = "set-selection-cmpmask-from-packed-predicate"
    VECTOR_READ = "vector-read-mode-and-mask"
    SELECT_READ = "vector-select-reads-full-mask-and-cmpmask"
    RESTORE_NORMAL = "restore-mode-normal"


@dataclass(frozen=True)
class _MaskState:
    mode: str | None = None
    mask: str | None = None
    cmp: bool = False


@dataclass(frozen=True, slots=True)
class NativePacket:
    native: str
    mask_steps: tuple[MaskStep, ...] = ()
    # A row loop may call a count C API repeatedly. Each such call contains
    # this complete sequence, rather than one setup surrounding all rows.
    each_count_call: bool = False
    # A masked native configures a fresh mode and repeat bitmap for every
    # chunk it issues; one bitmap never covers a tile with changing lanes.
    each_mask_call: bool = False
    internal_fences: tuple[str, ...] = ()
    # An audited conditional packet must close every path independently.
    alternate_mask_steps: tuple[tuple[MaskStep, ...], ...] = ()

    @property
    def closed_mask(self):
        return all(
            self._closed_mask(steps) for steps in (self.mask_steps, *self.alternate_mask_steps)
        )

    @staticmethod
    def _closed_mask(steps):
        state = _MaskState()
        for step in steps:
            if step is MaskStep.SET_COUNTER:
                state = _MaskState(mode="counter", mask=state.mask, cmp=state.cmp)
            elif step in {MaskStep.SET_NORMAL, MaskStep.RESTORE_NORMAL}:
                state = _MaskState(mode="normal", mask=state.mask, cmp=state.cmp)
            elif step is MaskStep.SET_COUNT:
                state = _MaskState(state.mode, "count", state.cmp)
            elif step is MaskStep.SET_FULL:
                state = _MaskState(state.mode, "full", state.cmp)
            elif step is MaskStep.SET_PARTIALS:
                state = _MaskState(state.mode, "partials", state.cmp)
            elif step is MaskStep.SET_COLUMNS:
                state = _MaskState(state.mode, "columns", state.cmp)
            elif step is MaskStep.SET_COORD:
                state = _MaskState(state.mode, "coordinate", state.cmp)
            elif step is MaskStep.SET_CMPMASK:
                state = _MaskState(state.mode, state.mask, True)
            elif step is MaskStep.VECTOR_READ:
                if state.mode is None or state.mask is None:
                    return False
            elif step is MaskStep.SELECT_READ and (
                state.mode != "normal" or state.mask != "full" or not state.cmp
            ):
                return False
        return True

    @property
    def clobbers(self):
        # Restore-to-normal is a defined write, not preservation of an
        # arbitrary incoming mode. No current packet saves mask or CMPMASK.
        result = set()
        for step in (
            step for steps in (self.mask_steps, *self.alternate_mask_steps) for step in steps
        ):
            if step in {MaskStep.SET_COUNTER, MaskStep.SET_NORMAL, MaskStep.RESTORE_NORMAL}:
                result.add("VECTOR_MASK_MODE")
            elif step in {
                MaskStep.SET_COUNT,
                MaskStep.SET_FULL,
                MaskStep.SET_PARTIALS,
                MaskStep.SET_COLUMNS,
                MaskStep.SET_COORD,
            }:
                result.add("VECTOR_MASK")
            elif step is MaskStep.SET_CMPMASK:
                result.add("CMPMASK")
        return frozenset(result)

    @property
    def restores_normal_mode(self):
        return all(
            MaskStep.RESTORE_NORMAL in steps
            for steps in (self.mask_steps, *self.alternate_mask_steps)
        )


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
        "mask_and",
        "mask_or",
        "mask_not",
        "mask_zero",
    }
)
_FULL = frozenset(
    {
        "add_strided",
        "mul_columns",
        "mul_outer",
        "add_columns",
        "broadcast_columns",
        "broadcast_rows",
        "broadcast_rows64",
        "mul_columns64",
        "fma_columns64",
        "reduce_tile_sum64",
        "reduce_tile_max64",
    }
)
_COLUMNS = frozenset(
    {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast", "vector_copy_rows"}
)
_MASKED = frozenset(
    {
        "mask_fill",
        "mask_muls",
        "mask_adds",
        "mask_add",
        "mask_mul",
        "mask_sub",
        "mask_div",
        "mask_max",
        "mask_exp",
        "mask_rsqrt",
        "mask_sqrt",
    }
)
_SELECT = frozenset({"select_masked", "select_masked_update"})
_PARTIAL = frozenset({"merge_sum_packed", "merge_max_packed"})
_MASK_FREE = frozenset(
    {
        "zero_l1",
        "copy",
        "copy_bytes",
        "copy_rows",
        "copy_rows_prefix",
        "scalar_load",
        "scalar_store",
        "index_load",
        "brcb8",
        "nd2nz",
        "nd2nz_pad",
        "nd2nz_rows",
        "nd2nz_overlay",
        "load2d_a",
        "load2d_a_t",
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
        "compare_eq",
        "compare_ne",
        "compare_lt",
        "compare_le",
        "compare_gt",
        "compare_ge",
        "compare_scalar_eq",
        "compare_scalar_ne",
        "compare_scalar_lt",
        "compare_scalar_le",
        "compare_scalar_gt",
        "compare_scalar_ge",
        "mask_bits",
    }
)


def _packet(native: str) -> NativePacket | None:
    if native in _MASKED:
        # Normal mode plus a fresh repeat bitmap selected from the lane
        # predicate for each emitted chunk. Full chunks reuse one all-ones
        # bitmap through a repeat count; no zero bitmap is ever issued.
        # The admitted whole-tile coverage chooses one closed path: `mixed`
        # replays the per-chunk coordinate bitmap, `full` sets one all-ones
        # bitmap per bounded repeat group, and `empty` issues no element call
        # (its declared RW footprint still requires an initialized
        # destination).
        return NativePacket(
            native,
            (MaskStep.SET_NORMAL, MaskStep.SET_COORD, MaskStep.VECTOR_READ),
            each_mask_call=True,
            alternate_mask_steps=(
                (MaskStep.SET_NORMAL, MaskStep.SET_FULL, MaskStep.VECTOR_READ),
                (MaskStep.SET_NORMAL,),
            ),
        )
    if native in _SELECT:
        # CMPMASK is loaded from packed UB and the audited local PIPE_V
        # barrier separates that setup from vsel. CMPMASK is a distinct state
        # from VECTOR_MASK_MODE/VECTOR_MASK.
        return NativePacket(
            native,
            (
                MaskStep.SET_NORMAL,
                MaskStep.SET_FULL,
                MaskStep.SET_CMPMASK,
                MaskStep.SELECT_READ,
            ),
            internal_fences=("V",),
        )
    if native in {"nd2nz_pad", "nd2nz_rows"}:
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
    if native in _FULL | _PARTIAL | _COLUMNS:
        return NativePacket(
            native,
            (
                MaskStep.SET_NORMAL,
                MaskStep.SET_FULL
                if native in _FULL
                else (MaskStep.SET_COLUMNS if native in _COLUMNS else MaskStep.SET_PARTIALS),
                MaskStep.VECTOR_READ,
            ),
        )
    if native in _MASK_FREE:
        return NativePacket(native)
    return None


_AUDITED = {
    native: _packet(native)
    for native in _COUNT | _FULL | _COLUMNS | _MASKED | _SELECT | _PARTIAL | _MASK_FREE
}


def audited_packet(native: str) -> NativePacket | None:
    """Return the target's fixed descriptor, without decoding native geometry.

    A hand-built descriptor is not an audit. Stable target-table identity lets
    admission retain that distinction without an external verified attribute.
    """
    return _AUDITED.get(native)

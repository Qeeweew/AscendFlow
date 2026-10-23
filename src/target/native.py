"""Trusted dav-2201 native footprints and machine resource contracts.

The contracts consume static footprint descriptions. They do not extract
programs, enumerate finite instances, or plan synchronization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..diagnostics import CompileError, require
from .ascend910b import Engine
from .layout import NZLayout
from .packets import NativePacket, audited_packet

CAPACITY = {
    "UB": 192 * 1024,
    "L1": 512 * 1024,
    "L0A": 64 * 1024,
    "L0B": 64 * 1024,
    "L0C": 128 * 1024,
}


ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2, "u8": 1, "i32": 4, "i64": 8}


def nz_columns(bank, dtype):
    """L1 NZ uses one 32-byte C0; FP32 L0C keeps its 16-column fractal."""
    return 32 // ITEMSIZE[dtype] if bank == "L1" else 16


_CMP_KINDS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_MASKED_UNARY = frozenset({"mask_muls", "mask_adds", "mask_exp", "mask_rsqrt", "mask_sqrt"})
_MASKED_ELEMENTS = frozenset(
    {
        "mask_fill",
        *_MASKED_UNARY,
        "mask_add",
        "mask_mul",
        "mask_sub",
        "mask_div",
        "mask_max",
    }
)


def _chunks(params, message):
    require(
        len(params) == 1 and params[0] > 0 and params[0] <= 65535,
        f"{message} needs a bounded 64-lane chunk count",
    )
    return params[0]


@dataclass(frozen=True, slots=True)
class NativeGeometry:
    """Static native footprint, without addresses or completion knowledge.

    ``nbytes`` is the complete static width of each row. Bounds, owners,
    aliasing and retirement
    require actual address/effect facts supplied by the consuming analysis.
    Rows have positive widths and counts, and do not overlap each other.
    """

    bank: str
    dtype: str
    layout: str
    nbytes: int
    rows: int = 1
    row_stride: int = 0
    matrix_shape: tuple[int, int] | None = None

    def __post_init__(self):
        if self.matrix_shape is not None:
            require(
                self.bank == "L1" and self.layout == "NZ" and self.dtype in {"f16", "bf16", "f32"},
                "physical matrix shape is an L1 NZ allocation descriptor",
            )
            NZLayout(self.matrix_shape, ITEMSIZE[self.dtype])
        require(
            self.nbytes > 0 and self.rows > 0,
            "native geometry requires positive row widths and row counts",
        )
        require(
            self.row_stride >= 0 and (self.rows == 1 or self.row_stride >= self.nbytes),
            "native geometry requires nonnegative, non-overlapping row strides",
        )

    @property
    def contiguous(self) -> bool:
        return self.rows == 1 or self.row_stride == self.nbytes

    @property
    def total_nbytes(self) -> int:
        return self.nbytes * self.rows

    @property
    def leading_rows(self) -> int:
        require(
            self.bank == "L1" and self.layout == "NZ" and self.matrix_shape is not None,
            "L1 native requires its allocation's physical leading dimension",
        )
        return self.matrix_shape[0]


# Current supported MAD clear-cycle policy, independent of finite expansion.
# This is an implementation range, not a hardware resource or instruction limit.
MAX_MAD_CYCLE = 4096


@dataclass(frozen=True, slots=True)
class NativeReadReset:
    """An operand's old value is ignored at an instruction-ordinal reset.

    ``period=None`` resets only at ordinal zero. A positive period resets at
    every zero residue. The instruction still writes its complete admitted
    operand footprint; this rule does not narrow its physical access modes.
    """

    operand: int
    period: int | None

    def __post_init__(self):
        require(type(self.operand) is int and self.operand >= 0, "invalid native reset operand")
        require(
            self.period is None or type(self.period) is int and self.period > 0,
            "native reset period must be positive",
        )


def native_read_resets(native: str, params: tuple[int, ...]) -> tuple[NativeReadReset, ...]:
    """Conditional numerical reads of admitted primitives, before loop binding."""
    if native == "mad_first":
        require(len(params) == 3, "MAD first reset requires exact dimensions")
        return (NativeReadReset(2, None),)
    if native == "mad_cycle":
        require(
            len(params) == 4 and 1 <= params[3] <= MAX_MAD_CYCLE,
            "MAD cycle reset requires a supported positive period",
        )
        return (NativeReadReset(2, params[3]),)
    return ()


def _implicit_completion(native, engine, params):
    if native in {"scalar_store", "set_fmatrix_a", "set_fmatrix_b", "set_load3d", "set_fp32"}:
        # Ordered Scalar issue does not publish the UB write to Vector.
        return False
    if engine is Engine.S:
        return True
    if engine is not Engine.M:
        return False
    m, n = (16, 16) if native == "mad16" else params[:2]
    # AtlasA2 CATLASS TileMmad fences short instructions: <10 output
    # fractals cannot supply the prior blanket M completion guarantee.
    return (m // 16) * (n // 16) >= 10


@dataclass(frozen=True, slots=True)
class RegisterEffect:
    """An actor-private hardware state access, distinct from tensor bytes."""

    register: str
    mode: str
    value: int


def register_effects(native, operands, params):
    if native in {"set_fmatrix_a", "set_fmatrix_b"}:
        return (
            RegisterEffect("FMATRIX_A" if native.endswith("_a") else "FMATRIX_B", "W", params[0]),
        )
    common = (("L3D_RPT", 1 << 16), ("L3D_PADDING", 0), ("L3D_SIZE", 0))
    if native == "set_load3d":
        return tuple(RegisterEffect(name, "W", value) for name, value in common)
    if native == "set_fp32":
        return (RegisterEffect("FP32_MODE", "W", 1),)
    if native in {"load3d_a", "load3d_b_nn"}:
        return (
            RegisterEffect(
                "FMATRIX_A" if native == "load3d_a" else "FMATRIX_B", "R", operands[0].leading_rows
            ),
            *(RegisterEffect(name, "R", value) for name, value in common),
        )
    if native in {"mad", "mad_first", "mad_cycle"} and operands[0].dtype == "f32":
        return (RegisterEffect("FP32_MODE", "R", 1),)
    return ()


@dataclass(frozen=True, slots=True)
class NativeSpec:
    """Static instruction facts shared by logical analysis and final emission.

    Operand order is the native's leading reference-operand order. Addresses,
    scalar SSA values/literals and loop-bound read predicates remain in their
    actual IR or analysis; this record contains no execution coordinates.
    """

    native: str
    engine: Engine
    modes: tuple[str, ...]
    params: tuple[int, ...]
    operands: tuple[NativeGeometry, ...]
    packet: NativePacket | None = field(init=False)
    read_resets: tuple[NativeReadReset, ...] = field(init=False)
    implicit_completion: bool = field(init=False)
    registers: tuple[RegisterEffect, ...] = field(init=False)

    def __post_init__(self):
        require(len(self.operands) == len(self.modes), "native operand/effect counts differ")
        object.__setattr__(self, "packet", audited_packet(self.native))
        resets = native_read_resets(self.native, self.params)
        require(
            all(
                reset.operand < len(self.modes) and self.modes[reset.operand] == "RW"
                for reset in resets
            ),
            "conditional native reset needs a complete read/write operand",
        )
        object.__setattr__(self, "read_resets", resets)
        object.__setattr__(
            self, "registers", register_effects(self.native, self.operands, self.params)
        )
        object.__setattr__(
            self, "implicit_completion", _implicit_completion(self.native, self.engine, self.params)
        )


def native_contract(
    native: str,
    refs: tuple[NativeGeometry, ...],
    params: tuple[int, ...],
    *,
    disjoint: Callable[[], bool] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Validate a physical native footprint and return its engine and access modes.

    Bounds, ownership, and completion coverage remain obligations of the
    consuming analysis; an admitted native does not supply those proofs.
    UB copy needs an explicit disjointness premise from the consuming analysis.
    """
    require(
        all(isinstance(ref, NativeGeometry) for ref in refs),
        "native contract needs static NativeGeometry operands",
    )
    if native in {"set_fmatrix_a", "set_fmatrix_b", "set_load3d", "set_fp32"}:
        require(not refs, "register setters have no tensor operands")
        if native in {"set_fmatrix_a", "set_fmatrix_b"}:
            require(
                len(params) == 1 and 1 <= params[0] <= 65535,
                "FMATRIX needs a positive uint16 width",
            )
        else:
            require(not params, "fixed register setup has no parameters")
        return "S", ()
    if native in {"load3d_a", "load3d_b_nn"}:
        if refs and refs[0].dtype == "f32":
            require(len(params) == 2, "FP32 Load3D requires matrix dimensions")
            rows, cols = params
            source = refs[0]
            source_matches = source.contiguous or (
                native in {"load3d_a", "load3d_b_nn"}
                and source.nbytes == rows * 8 * 4
                and source.rows == cols // 8
                and source.row_stride % (16 * 8 * 4) == 0
                and source.nbytes <= source.row_stride <= 256 * 8 * 4
            )
            require(
                all(16 <= d <= 256 and d % 16 == 0 for d in params)
                and len(refs) == 2
                and refs[1].dtype == "f32"
                and (refs[0].bank, refs[1].bank) == ("L1", "L0A" if native == "load3d_a" else "L0B")
                and (refs[0].layout, refs[1].layout)
                == ("NZ", "ZZ" if native == "load3d_a" else "ZN")
                and refs[0].total_nbytes == refs[1].total_nbytes == rows * cols * 4
                and source_matches
                and refs[1].contiguous,
                "FP32 Load3D requires an aligned NZ source and compact f32 destination plane",
            )
            return "MTE1", ("R", "W")
        # Complete b16 matrix, unit filters/strides, zero padding. Layout and
        # byte footprints are identical to the corresponding audited Load2D.
        return native_contract(
            "load2d_a" if native == "load3d_a" else "load2d_b_nn",
            refs,
            params,
            disjoint=disjoint,
        )
    banks = tuple(r.bank for r in refs)
    dtypes = tuple(r.dtype for r in refs)
    lengths = tuple(r.total_nbytes for r in refs)
    layouts = tuple(r.layout for r in refs)
    if native in {"copy", "copy_bytes"}:
        require(all(r.contiguous for r in refs), "DMA recipe requires contiguous physical views")
        require(len(refs) == 2 and lengths[0] == lengths[1], "copy needs equal native footprints")
        require(dtypes[0] == dtypes[1], "copy cannot silently convert dtype")
        routes = {
            ("GM", "UB"): "MTE2",
            ("UB", "GM"): "MTE3",
            ("GM", "L1"): "MTE2",
            ("L1", "L0A"): "MTE1",
            ("L1", "L0B"): "MTE1",
            ("UB", "UB"): "V",
        }
        require(banks in routes, f"unsupported dav-2201 DMA route {banks}; no hidden GM staging")
        require(not params, "copy has no unchecked native parameters")
        require(
            native != "copy_bytes" or banks in {("GM", "UB"), ("UB", "GM")},
            "byte-addressed DMA needs a GM/UB route",
        )
        if banks[0] == "L1":
            expected = ("NZ", "ZZ" if banks[1] == "L0A" else "ZN")
            require(
                dtypes[0] == dtypes[1] in {"f16", "bf16"}
                and lengths == (512, 512)
                and layouts == expected,
                "L1->L0 requires one complete b16 16x16 fractal",
            )
        else:
            require(layouts[0] == layouts[1], "raw DMA must preserve physical layout")
            require(
                lengths[0] % 32 == 0 and 0 < lengths[0] <= 65535 * 32,
                "raw DMA footprint must be positive, 32-byte aligned and fit burst length",
            )
        if banks == ("UB", "UB"):
            # CANN 9.1/2201 asc_copy_ub2ub uses copy_ubuf_to_ubuf on PIPE_V.
            # Its size is bytes, divided by 32 into a uint16_t burst length.
            require(
                disjoint is not None,
                "UB-to-UB copy requires provably disjoint source and destination ranges; "
                "geometry inputs need an explicit alias proof",
            )
            require(callable(disjoint), "UB-copy disjointness premise must be callable")
            separated = disjoint()
            require(
                separated is True,
                "UB-to-UB copy requires provably disjoint source and destination ranges; "
                "overlapping or changing aliases are unsupported",
            )
        return routes[banks], ("R", "W")
    if native == "vector_copy_rows":
        require(
            len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes[0] == dtypes[1]
            and layouts == ("linear", "linear")
            and refs[0].rows == refs[1].rows
            and refs[0].nbytes == refs[1].nbytes
            and 1 <= refs[0].rows <= 255
            and refs[0].nbytes % 32 == 0
            and 32 <= refs[0].nbytes <= 256
            and all(r.row_stride % 32 == 0 for r in refs)
            and all(r.rows == 1 or 1 <= r.row_stride // 32 <= 255 for r in refs)
            and not params,
            "Vector row copy needs equal aligned UB rows, at most 64 words and bounded repeat strides",
        )
        require(
            callable(disjoint),
            "UB-to-UB copy requires provably disjoint source and destination ranges",
        )
        require(
            disjoint() is True,
            "UB-to-UB copy requires provably disjoint source and destination ranges; overlapping or changing aliases are unsupported",
        )
        return "V", ("R", "W")
    if native == "copy_rows":
        require(
            len(refs) == 2
            and banks in {("GM", "UB"), ("UB", "GM")}
            and dtypes[0] == dtypes[1]
            and layouts == ("linear", "linear")
            and refs[0].rows == refs[1].rows
            and refs[0].nbytes == refs[1].nbytes
            and 1 <= refs[0].rows <= 4095
            and refs[0].nbytes % 32 == 0
            and 0 < refs[0].nbytes <= 65535 * 32
            and all(r.row_stride % 32 == 0 for r in refs)
            and all(r.rows == 1 or 0 <= (r.row_stride - r.nbytes) // 32 <= 65535 for r in refs)
            and not params,
            "row DMA requires equal, aligned physical rows in GM and UB",
        )
        return ("MTE2" if banks[0] == "GM" else "MTE3"), ("R", "W")
    if native in {"convert_f32", "convert_rn"}:
        require(
            len(refs) == 2
            and banks == ("UB", "UB")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and not params
            and lengths[0] // ITEMSIZE[dtypes[0]] == lengths[1] // ITEMSIZE[dtypes[1]] > 0
            and (
                dtypes[0] in {"f16", "bf16"} and dtypes[1] == "f32"
                if native == "convert_f32"
                else dtypes[0] == "f32" and dtypes[1] in {"f16", "bf16"}
            ),
            "conversion needs equal element counts in separate contiguous b16/f32 UB views",
        )
        # Different-width overlap cannot be repaired by an event or by native
        # in-place conversion assumptions. Origins are rechecked at extraction.
        return "V", ("R", "W")
    if native == "add_strided":
        require(
            len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and not params
            and 1 <= refs[0].rows <= 255
            and refs[0].nbytes % 256 == 0
            and all(r.rows == refs[0].rows and r.nbytes == refs[0].nbytes for r in refs)
            and all(r.row_stride % 32 == 0 and 1 <= r.row_stride // 32 <= 255 for r in refs),
            "strided add needs equal full-lane f32 UB rows and bounded repeat strides",
        )
        return "V", ("R", "R", "W")
    if native in {
        "muls",
        "adds",
        "add",
        "mul",
        "sub",
        "div",
        "max",
        "exp",
        "rsqrt",
        "sqrt",
        "scale_value",
        "sub_value",
    }:
        arity = 3 if native in {"add", "mul", "sub", "div", "max"} else 2
        require(
            len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("f32",) * arity
            and layouts == ("linear",) * arity
            and len(set(lengths)) == 1
            and lengths[0] > 0
            and lengths[0] % 4 == 0,
            "Vector recipe requires equal contiguous f32 UB footprints",
        )
        require(all(r.contiguous for r in refs), "count-mode Vector requires contiguous views")
        require(not params, "Vector recipe has no unchecked native parameters")
        return "V", ("R",) * (arity - 1) + ("W",)
    if native == "scalar_load":
        require(
            len(refs) == 1
            and banks[0] in {"UB", "GM"}
            and dtypes[0] in {"f32", "f16", "bf16"}
            and (banks[0] == "GM" or dtypes == ("f32",))
            and layouts == ("linear",)
            and lengths == (ITEMSIZE[dtypes[0]],)
            and not params,
            "Scalar load requires one f32 UB or floating GM element",
        )
        return "S", ("R",)
    if native == "scalar_store":
        require(
            len(refs) == 1
            and banks == ("UB",)
            and dtypes == ("f32",)
            and layouts == ("linear",)
            and lengths == (4,)
            and not params,
            "Scalar store requires one f32 UB element",
        )
        return "S", ("W",)
    if native == "index_load":
        require(
            len(refs) == 1
            and banks[0] in {"GM", "UB"}
            and dtypes[0] in {"i32", "i64"}
            and layouts == ("linear",)
            and lengths == (ITEMSIZE[dtypes[0]],)
            and not params,
            "index load requires exactly one GM/UB int32/int64",
        )
        return "S", ("R",)
    if native in {"mul_columns", "add_columns", "broadcast_columns", "broadcast_rows"}:
        require(len(params) == 2, "broadcast needs (rows, columns)")
        rows, cols = params
        binary = native in {"mul_columns", "add_columns"}
        arity = 3 if binary else 2
        source_bytes = rows * 32 if native == "broadcast_rows" else cols * 4
        require(
            1 <= rows <= 255
            and 64 <= cols <= 1984
            and cols % 64 == 0
            and len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("f32",) * arity
            and layouts == ("linear",) * arity
            and all(r.contiguous for r in refs)
            and lengths
            == (
                (rows * cols * 4, cols * 4, rows * cols * 4)
                if binary
                else (source_bytes, rows * cols * 4)
            ),
            "broadcast needs exact contiguous f32 matrix/vector footprints",
        )
        return "V", ("R",) * (arity - 1) + ("W",)
    if native == "vector_copy":
        require(
            len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths[0] == lengths[1] > 0
            and lengths[0] % 4 == 0
            and not params,
            "bitwise Vector copy needs equal contiguous f32 UB footprints",
        )
        return "V", ("R", "W")
    if native in _MASKED_ELEMENTS:
        arity = 1 if native == "mask_fill" else 2 if native in _MASKED_UNARY else 3
        require(
            len(refs) == arity
            and not params
            and banks == ("UB",) * arity
            and dtypes == ("f32",) * arity
            and layouts == ("linear",) * arity
            and all(r.contiguous for r in refs)
            and len(set(lengths)) == 1
            and lengths[0] > 0
            and lengths[0] % 256 == 0,
            "masked Vector work needs equal contiguous f32 UB tiles of whole 64-lane blocks",
        )
        # Masked lanes keep the old destination value: a real read/write
        # update, never a full initialization of the destination footprint.
        return "V", ("RW",) if arity == 1 else ("R",) * (arity - 1) + ("RW",)
    if native.startswith("compare_"):
        chunks = _chunks(params, "packed data comparison")
        scalar = native.startswith("compare_scalar_")
        kind = native[len("compare_scalar_") if scalar else len("compare_") :]
        require(kind in _CMP_KINDS, "packed data comparison needs a supported kind")
        arity = 2 if scalar else 3
        require(
            len(refs) == arity
            and banks == ("UB",) * arity
            and layouts == ("linear",) * arity
            and (dtypes[0] == "f32" if scalar else dtypes[:2] == ("f32", "f32"))
            and dtypes[-1] == "u8"
            and all(r.contiguous for r in refs[:-1])
            and all(r.total_nbytes == chunks * 256 for r in refs[:-1])
            # The destination is a conservative whole-block preserve update:
            # vcmpv writes the 64-bit lane word and preserves the other 24
            # padding bytes, which the separately declared `mask_zero` Vector
            # task has already defined. `RW` states "old whole-block state is
            # read and the block stays defined", not a hardware padding read,
            # and it never creates a fresh full-initialization fact on its own.
            and refs[-1].contiguous
            and lengths[-1] == chunks * 32,
            "packed data comparison needs whole f32 chunks and 32-byte predicate blocks",
        )
        return "V", ("R",) * (arity - 1) + ("RW",)
    if native == "mask_zero":
        require(
            len(params) == 1 and params[0] > 0 and params[0] % 16 == 0,
            "packed predicate zeroing needs a whole uint16 count of 32-byte blocks",
        )
        require(
            len(refs) == 1
            and banks == ("UB",)
            and dtypes == ("u8",)
            and layouts == ("linear",)
            and refs[0].contiguous
            and lengths == (params[0] * 2,),
            "packed predicate zeroing needs its complete aligned byte block",
        )
        # A count-mode Vector Dup defines the whole packed block, including
        # the padding bytes no packed mask consumer interprets as lanes.
        return "V", ("W",)
    if native == "mask_bits":
        chunks = _chunks(params, "coordinate predicate materialization")
        require(
            len(refs) == 1
            and banks == ("UB",)
            and dtypes == ("u8",)
            and layouts == ("linear",)
            and refs[0].contiguous
            and lengths == (chunks * 32,),
            "predicate materialization writes exact 32-byte packed blocks",
        )
        # Scalar stores build each packed word; no Vector execute-lane mask is
        # involved. The bytes are a real UB write for later synchronization.
        return "S", ("W",)
    if native in {"mask_and", "mask_or", "mask_not"}:
        require(
            len(params) == 1 and params[0] > 0,
            "packed predicate combination needs its uint16 word count",
        )
        arity = 2 if native == "mask_not" else 3
        require(
            len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("u8",) * arity
            and layouts == ("linear",) * arity
            and all(r.contiguous for r in refs)
            and len(set(lengths)) == 1
            and lengths[0] == params[0] * 2
            and lengths[0] % 32 == 0,
            "packed predicate combination needs equal aligned byte blocks",
        )
        return "V", ("R",) * (arity - 1) + ("W",)
    if native in {"select_masked", "select_masked_update"}:
        chunks = _chunks(params, "packed predicate selection")
        update = native == "select_masked_update"
        arity = 3 if update else 4
        floats = ("f32",) * (arity - 1)
        require(
            len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("u8", *floats)
            and layouts == ("linear",) * arity
            and refs[0].contiguous
            and lengths[0] == chunks * 32
            and all(r.contiguous and r.total_nbytes == chunks * 256 for r in refs[1:]),
            "predicate selection needs packed blocks and whole f32 chunks",
        )
        return "V", ("R", "R", "RW") if update else ("R", "R", "R", "W")
    if native == "zero_l1":
        require(
            len(refs) == 1
            and banks == ("L1",)
            and layouts == ("NZ",)
            and dtypes[0] in {"f16", "bf16", "f32"}
            and refs[0].contiguous
            and 0 < lengths[0] <= CAPACITY["L1"]
            and lengths[0] % 32 == 0
            and not params,
            "L1 zero fill needs a contiguous NZ plane in physical L1 capacity",
        )
        return "MTE2", ("W",)
    if native in {"fill", "fill_neg_inf"}:
        require(
            len(refs) == 1
            and banks == ("UB",)
            and dtypes in {("f32",), ("f16",), ("bf16",)}
            and layouts == ("linear",)
            and refs[0].contiguous
            and lengths[0] > 0
            and lengths[0] % ITEMSIZE[dtypes[0]] == 0
            and not params,
            "Vector fill needs a contiguous f32/f16/bf16 UB destination",
        )
        return "V", ("W",)
    if native == "mul_outer":
        require(len(params) == 2, "outer product needs (rows,columns)")
        rows, cols = params
        require(
            8 <= rows <= 248
            and rows % 8 == 0
            and 64 <= cols <= 1984
            and cols % 64 == 0
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.contiguous for r in refs)
            and lengths == (cols * 4, rows * 32, rows * cols * 4),
            "outer product needs exact f32 column and BRCB row footprints",
        )
        return "V", ("R", "R", "W")
    if native in {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast"}:
        require(len(params) == 2, "row broadcast arithmetic needs (rows,columns)")
        rows, cols = params
        require(
            1 <= rows <= 255
            and 8 <= cols <= 1024
            and (cols % 64 == 0 or cols < 64 and cols % 8 == 0)
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.contiguous for r in refs)
            and lengths == (rows * cols * 4, rows * 32, rows * cols * 4),
            "row broadcast arithmetic needs complete f32 rows and eight identical words per row",
        )
        return "V", ("R", "R", "W")
    if native in {"merge_sum_packed", "merge_max_packed"}:
        require(len(params) == 2, "packed partial reduction needs (rows,partials)")
        rows, partials = params
        require(
            1 <= rows <= 255
            and 1 <= partials <= 64
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and refs[0].rows == rows
            and refs[0].nbytes == partials * 4
            and refs[0].row_stride % 32 == 0
            and partials * 4 <= refs[0].row_stride <= 255 * 32
            and refs[1].contiguous
            and lengths[1] == rows * 4,
            "packed partial reduction reads masked words in aligned rows and writes compact statistics",
        )
        return "V", ("R", "W")
    if native in {"mul_columns64", "fma_columns64"}:
        require(len(params) == 1, "column operation needs its physical row count")
        rows = params[0]
        require(
            1 <= rows <= 255
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.contiguous for r in refs)
            and lengths == (rows * 256, 256, rows * 256),
            "column operation requires complete contiguous f32 rows of 64 elements",
        )
        return "V", ("R", "R", "RW" if native == "fma_columns64" else "W")
    if native == "reduce_rows64":
        require(len(params) == 1, "packed row reduction needs its physical row count")
        rows = params[0]
        require(
            1 <= rows <= 255
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths == (rows * 256, rows * 4),
            "packed reduction requires f32 rows of 64 and one contiguous result per row",
        )
        return "V", ("R", "W")
    if native == "brcb8":
        require(len(params) == 1, "BRCB needs the source word count")
        rows = params[0]
        require(
            8 <= rows <= 255 * 8
            and rows % 8 == 0
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths == (rows * 4, rows * 32),
            "BRCB replicates each of a multiple of eight f32 words to one 32-byte block",
        )
        return "V", ("R", "W")
    if native == "broadcast_rows64":
        require(len(params) == 1, "row broadcast needs its physical row count")
        rows = params[0]
        require(
            1 <= rows <= 255
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths == (rows * 32, rows * 256),
            "row broadcast copies each 32-byte block to a complete f32 row of 64",
        )
        return "V", ("R", "W")
    if native in {"mul_rows", "sub_rows", "add_rows", "div_rows"}:
        require(len(params) == 2, "row binary recipe needs (rows, columns)")
        rows, cols = params
        require(
            1 <= rows <= 255
            and cols > 0
            and cols % 8 == 0
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.rows == rows and r.nbytes == cols * 4 for r in refs),
            "row binary requires bounded equal f32 rows with aligned origins",
        )
        return "V", ("R", "R", "W")
    if native in {"reduce_tile_sum64", "reduce_tile_max64"}:
        require(len(params) == 2, "plane reduction needs (rows,columns)")
        rows, cols = params
        require(
            8 <= rows <= 128
            and rows % 8 == 0
            and cols in {64, 128}
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths == (rows * cols * 4, rows * (cols // 64) * 4),
            "plane reduction writes one compact row-statistic plane for each aligned 64-column chunk",
        )
        return "V", ("R", "W")
    if native in {"reduce_max8", "reduce_sum8"}:
        require(
            len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.contiguous for r in refs)
            and lengths[0] >= 256
            and lengths[0] % 256 == 0
            and lengths[0] == 8 * lengths[1]
            and not params,
            f"datablock {'max' if native == 'reduce_max8' else 'sum'} needs contiguous f32 UB, whole 64-element repeats, and one result per eight inputs",
        )
        return "V", ("R", "W")
    if native in {"reduce_sum64", "reduce_max64"}:
        require(len(params) == 2, "native reduction needs (rows, columns)")
        rows, cols = params
        require(
            1 <= rows <= 255 and 64 <= cols <= 4096 and cols % 64 == 0,
            "native row reduction requires positive rows and 64-aligned columns <=4096",
        )
        require(
            len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and refs[0].contiguous
            and lengths[0] == rows * cols * 4
            and refs[1].rows == rows
            and refs[1].nbytes == cols // 64 * 4,
            "reduction partials must expose their exact row footprint",
        )
        return "V", ("R", "W")
    if native in {"merge_sum", "merge_max"}:
        require(len(params) == 2, "partial reduction needs (rows, partials)")
        rows, partials = params
        require(
            1 <= rows <= 255
            and 1 <= partials <= 64
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and refs[0].rows == refs[1].rows == rows
            and refs[0].nbytes == partials * 4
            and refs[1].nbytes == 4,
            "partial reduction must preserve rows and exact compact results",
        )
        return "V", ("R", "W")
    if native in {"stat_muls", "stat_adds", "stat_rsqrt", "stat_sqrt"}:
        require(
            len(params) == 1
            and 1 <= params[0] <= 255
            and len(refs) == 2
            and banks == ("UB", "UB")
            and dtypes == ("f32", "f32")
            and layouts == ("linear", "linear")
            and all(r.rows == params[0] and r.nbytes == 4 for r in refs),
            "statistics operation needs one aligned f32 result per row",
        )
        return "V", ("R", "W")
    if native in {"mul_gamma", "add_bias"}:
        require(len(refs) == 3, "Gamma multiplication needs matrix, Gamma and output")
        require(len(params) == 2, "Gamma multiplication needs (rows, columns)")
        rows, cols = params
        require(
            1 <= rows <= 255
            and cols > 0
            and cols % 8 == 0
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.contiguous for r in refs)
            and lengths == (rows * cols * 4, cols * 4, rows * cols * 4),
            "column broadcast requires a resident column vector and contiguous rows",
        )
        return "V", ("R", "R", "W")
    if native in {"stat_fill", "stat_div"}:
        arity = 1 if native == "stat_fill" else 3
        require(
            len(params) == 1
            and 1 <= params[0] <= 255
            and len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("f32",) * arity
            and layouts == ("linear",) * arity
            and all(r.rows == params[0] and r.nbytes == 4 for r in refs),
            "statistics recipe needs aligned f32 words per row",
        )
        return "V", ("W",) if arity == 1 else ("R", "R", "W")
    if native == "copy_rows_prefix":
        require(tuple(r.bank for r in refs) == ("UB", "GM"), "runtime row output requires UB→GM")
        return native_contract("copy_rows", refs, params)
    if native == "nd2nz_rows":
        require(len(params) == 4, "runtime ND2NZ needs capacity and resident dimensions")
        m, n, pm, pn = params
        itemsize = ITEMSIZE[dtypes[0]] if dtypes else 0
        require(
            1 <= m <= pm <= 256
            and 1 <= n <= pn <= 256
            and pm % 16 == pn % 16 == 0
            and banks == ("GM", "L1")
            and layouts == ("linear", "NZ")
            and dtypes[0] == dtypes[1] in {"f16", "bf16", "f32"}
            and refs[0].rows == m
            and refs[0].nbytes == n * itemsize
            and refs[0].row_stride % itemsize == 0
            and refs[0].row_stride >= n * itemsize
            and refs[1].contiguous
            and (refs[1].matrix_shape is None or refs[1].leading_rows == pm)
            and lengths == (m * n * itemsize, pm * pn * itemsize),
            "runtime ND2NZ requires bounded GM rows and a fully zero-defined NZ destination",
        )
        return "MTE2", ("R", "W")
    if native == "nd2nz_overlay":
        require(
            len(params) == 3, "ND2NZ overlay needs row capacity, columns and resident row stride"
        )
        m, n, resident = params
        itemsize = ITEMSIZE[dtypes[0]] if dtypes else 0
        c0 = 32 // itemsize if itemsize else 0
        require(
            1 <= m <= resident <= 256
            and resident % 16 == 0
            and banks == ("GM", "L1")
            and layouts == ("linear", "NZ")
            and dtypes[0] == dtypes[1] in {"f16", "bf16", "f32"}
            and c0 <= n <= 256
            and n % c0 == 0
            and refs[0].rows == m
            and refs[0].nbytes == n * itemsize
            and refs[0].row_stride >= n * itemsize
            and refs[0].row_stride % itemsize == 0
            and refs[1].nbytes == m * 32
            and refs[1].rows == n // c0
            and refs[1].row_stride == resident * 32
            and (refs[1].matrix_shape is None or refs[1].leading_rows == resident),
            "ND2NZ overlay requires exact GM rows and C0-aligned NZ row prefixes",
        )
        return "MTE2", ("R", "W")
    if native == "nd2nz_pad":
        require(len(params) == 4, "padded ND2NZ requires valid and resident dimensions")
        m, n, pm, pn = params
        itemsize = ITEMSIZE[dtypes[0]] if dtypes else 0
        require(
            1 <= m <= pm <= 256
            and 1 <= n <= pn <= 256
            and pm % 16 == pn % 16 == 0
            and banks == ("GM", "L1")
            and dtypes[0] == dtypes[1] in {"f16", "bf16", "f32"}
            and layouts == ("linear", "NZ")
            and refs[0].nbytes == n * itemsize
            and refs[0].rows == m
            and refs[0].row_stride % itemsize == 0
            and n <= refs[0].row_stride // itemsize
            and refs[1].contiguous
            and (refs[1].matrix_shape is None or refs[1].leading_rows == pm)
            and lengths == (m * n * itemsize, pm * pn * itemsize),
            "padded ND2NZ needs exact GM rows and a complete NZ destination",
        )
        return "MTE2", ("R", "W")
    if native in {"nd2nz", "load2d_a", "load2d_a_t", "load2d_b_nt", "load2d_b_nn"}:
        require(len(params) == 2, "matrix DMA requires (rows, columns)")
        rows, cols = params
        require(
            (1 if native == "nd2nz" else 16) <= rows <= 256
            and 16 <= cols <= (65520 if native == "nd2nz" else 256)
            and (native == "nd2nz" or rows % 16 == 0)
            and cols % 16 == 0,
            "matrix DMA requires complete b16 fractals and bounded native dimensions",
        )
        require(
            len(refs) == 2
            and dtypes[0]
            == dtypes[1]
            in ({"f16", "bf16", "f32"} if native in {"nd2nz", "load2d_b_nt"} else {"f16", "bf16"})
            and lengths == (rows * cols * ITEMSIZE[dtypes[0]],) * 2,
            "matrix DMA must expose the full source and destination tile",
        )
        if native == "nd2nz":
            require(
                banks == ("GM", "L1")
                and layouts == ("linear", "NZ")
                and refs[0].rows == rows
                and refs[0].nbytes == cols * ITEMSIZE[dtypes[0]]
                and refs[0].row_stride % ITEMSIZE[dtypes[0]] == 0
                and cols <= refs[0].row_stride // ITEMSIZE[dtypes[0]]
                and (
                    refs[1].contiguous
                    and rows % 16 == 0
                    or refs[1].nbytes == rows * nz_columns("L1", dtypes[1]) * ITEMSIZE[dtypes[1]]
                    and refs[1].rows == cols // nz_columns("L1", dtypes[1])
                    and refs[1].row_stride % 32 == 0
                    and refs[1].row_stride >= refs[1].nbytes
                ),
                "ND2NZ needs precise GM rows, legal leading dimension and compact NZ output",
            )
            return "MTE2", ("R", "W")
        destination = "L0A" if native in {"load2d_a", "load2d_a_t"} else "L0B"
        layout = "ZZ" if native in {"load2d_a", "load2d_a_t"} else "ZN"
        source = refs[0]
        source_matches = source.contiguous or (
            native in {"load2d_a", "load2d_b_nn"}
            and source.nbytes == rows * 16 * 2
            and source.rows == cols // 16
            and source.row_stride % 512 == 0
        )
        require(
            banks == ("L1", destination)
            and layouts == ("NZ", layout)
            and source_matches
            and refs[1].contiguous
            and (native == "load2d_a" or rows * cols * ITEMSIZE[dtypes[0]] // 512 <= 255),
            "Load2D requires complete compact fractals and a legal repeat count",
        )
        return "MTE1", ("R", "W")
    if native in {"mad", "mad_cycle", "mad_first"}:
        require(
            len(params) == (3 if native == "mad_first" else 4),
            "MAD requires exact dimensions and initialization policy",
        )
        m, n, k = params[:3]
        clear = params[3] if native != "mad_first" else 0
        require(
            all(16 <= dim <= 256 and dim % 16 == 0 for dim in (m, n, k))
            and (
                native == "mad_first"
                or (clear in {0, 1} if native == "mad" else 1 <= clear <= MAX_MAD_CYCLE)
            )
            and banks == ("L0A", "L0B", "L0C")
            and dtypes[0] == dtypes[1] in {"f16", "bf16", "f32"}
            and dtypes[2] == "f32"
            and layouts == ("ZZ", "ZN", "NZ")
            and lengths == (m * k * ITEMSIZE[dtypes[0]], n * k * ITEMSIZE[dtypes[0]], m * n * 4)
            and all(r.contiguous for r in refs),
            "MAD requires exact b16 operand tiles and an f32 accumulator, without padding",
        )
        return "M", ("R", "R", "W" if native == "mad" and clear else "RW")
    if native == "fix":
        require(len(refs) == 2, "Fixpipe requires a source and destination")
        require(len(params) in {2, 3}, "Fixpipe requires (M,N[,source_M])")
        m, n = params[:2]
        source_m = params[2] if len(params) == 3 else m
        itemsize = ITEMSIZE.get(dtypes[1], 0) if len(dtypes) == 2 else 0
        source_matches = (
            refs[0].nbytes == m * 16 * 4
            and refs[0].rows == (n + 15) // 16
            and refs[0].row_stride == source_m * 16 * 4
            if len(params) == 3
            else refs[0].contiguous and lengths[0] == m * n * 4
        )
        require(
            1 <= m <= source_m <= 256
            and 1 <= n <= 256
            and source_m % 16 == 0
            and (len(params) == 3 or m % 16 == n % 16 == 0)
            and banks == ("L0C", "GM")
            and dtypes[0] == "f32"
            and dtypes[1] in {"f32", "f16", "bf16"}
            and layouts == ("NZ", "linear")
            and source_matches
            and refs[1].rows == m
            and refs[1].nbytes == n * itemsize
            and refs[1].row_stride % itemsize == 0
            and 1 <= refs[1].row_stride // itemsize <= 4294967295,
            "Fixpipe requires exact f32 NZ reads and f32/f16/bf16 ND destination rows",
        )
        return "FIX", ("R", "W")
    if native == "mad16":
        require(
            banks == ("L0A", "L0B", "L0C")
            and dtypes == ("f16", "f16", "f32")
            and lengths == (512, 512, 1024)
            and layouts == ("ZZ", "ZN", "NZ")
            and params in {(0,), (1,)},
            "MAD v0.1 requires f16 16x16x16, f32 accumulation",
        )
        return "M", ("R", "R", "W" if params[0] else "RW")
    if native == "fix16":
        require(
            banks == ("L0C", "GM")
            and dtypes == ("f32", "f32")
            and lengths == (1024, 1024)
            and layouts == ("NZ", "linear")
            and not params,
            "Fixpipe v0.1 requires a complete f32 16x16 NZ->ND transfer",
        )
        return "FIX", ("R", "W")
    raise CompileError(
        f"unmodeled native operation {native!r}: effects/completion contract required"
    )

"""Trusted dav-2201 native contracts and exact finite physical-access analysis.

No analysis reads a compiler certificate as a premise. Layout changes are legal
only at a named native recipe; a subview/recast preserves allocation identity.
The first version deliberately bounds finite expansion, and emits compact loops.
"""

import re
import struct
from dataclasses import dataclass, replace
from functools import cached_property
from math import isfinite, isinf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scalar import Expr

from xdsl.dialects.builtin import IndexType, ModuleOp, f32
from xdsl.ir import SSAValue

from .ir import (
    BufferOp,
    ChannelOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    Load3DConfigOp,
    RefType,
    StreamOp,
    SyncOp,
    TaskOp,
    ViewOp,
)

CAPACITY = {
    "UB": 192 * 1024,
    "L1": 512 * 1024,
    "L0A": 64 * 1024,
    "L0B": 64 * 1024,
    "L0C": 128 * 1024,
}
ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2}
IMPLICIT_COMPLETION = {"S"}  # Unconditional; short Cube instructions need separate treatment.
MAX_INSTANCES = 4096


class CompileError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CompileError(message)


def round_f32(value, *, allow_infinity=False):
    """The IR literal and emitted C++ float must have the same value."""
    if allow_infinity and isinf(value):
        return float(value)
    require(isfinite(value), "Vector scalar must be finite")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise CompileError("Vector scalar is outside finite f32 range") from error
    require(isfinite(result), "Vector scalar is outside finite f32 range")
    return result


def task_scalar(op):
    """One scalar contract for finite and parametric physical task parsing."""
    require(op.scalar.type == f32, "native Vector scalar literals must be f32")
    return round_f32(op.scalar.value.data, allow_infinity=op.native.data == "fill")


def optional_int(attr, default):
    return default if attr is None else attr.data


def ranges_cover(parent, child):
    parent = sorted(parent)
    index = 0
    for start, stop in sorted(child):
        while index < len(parent) and parent[index][1] <= start:
            index += 1
        cursor = start
        while cursor < stop:
            if index == len(parent) or parent[index][0] > cursor:
                return False
            cursor = max(cursor, parent[index][1])
            if cursor < stop:
                index += 1
    return True


def ranges_overlap(left, right):
    """Linear comparison of two sorted interval unions, including their holes."""
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if a < d and c < b:
            return True
        if b <= c:
            i += 1
        else:
            j += 1
    return False


@dataclass(frozen=True)
class Storage:
    name: str
    bank: str
    nbytes: int
    dtype: str
    layout: str
    external: bool
    initialized: bool
    count_scale: int = 0
    extent: "Expr | None" = None


@dataclass(frozen=True)
class Ref:
    storage: Storage
    terms: tuple[tuple[int, int, int, int], ...]  # offset, byte stride, modulus, divisor
    nbytes: int
    dtype: str
    layout: str
    tail_nbytes: int = 0
    tail_generation: int = -1
    rows: int = 1
    row_stride: int = 0
    core_terms: tuple[tuple[int, int, int], ...] = ()  # byte stride, divisor, modulus
    count_offset: int = 0
    count_core_stride: int = 0
    prefix_extent: "Expr | None" = None
    prefix_elements: int = 0

    def actual_segments(self, g, parameters, core=0):
        """Concrete footprint for an explicitly supplied runtime invocation."""
        if self.prefix_extent is None:
            require(
                not self.count_offset and not self.count_core_stride,
                "count-translated concrete footprint requires an explicit loop-bound binding",
            )
            return self.segments(g, core)
        valid = min(
            self.prefix_elements, self.prefix_extent.evaluate(parameters) - g * self.prefix_elements
        )
        if valid <= 0:
            return ()
        start = self.address(g, core)
        return ((start, start + valid * ITEMSIZE[self.dtype]),)

    def address(self, g: int, core=0) -> int:
        return sum(o + s * ((g // d) % m if m else g // d) for o, s, m, d in self.terms) + sum(
            s * ((core // d) % m if m else core // d) for s, d, m in self.core_terms
        )

    def interval(self, g: int, core=0) -> tuple[int, int]:
        parts = self.segments(g, core)
        return parts[0][0], parts[-1][1]

    def segments(self, g: int, core=0):
        start = self.address(g, core)
        return tuple(
            (start + r * self.row_stride, start + r * self.row_stride + self.length(g))
            for r in range(self.rows)
        )

    @property
    def total_nbytes(self):
        return self.nbytes * self.rows

    @property
    def contiguous(self):
        return self.rows == 1 or self.row_stride == self.nbytes

    def length(self, g: int) -> int:
        return self.tail_nbytes if self.tail_nbytes and g == self.tail_generation else self.nbytes


@dataclass(frozen=True)
class Task:
    uid: str
    native: str
    pipe: str
    stage: int
    refs: tuple[Ref, ...]
    modes: tuple[str, ...]
    scalar: float
    params: tuple[int, ...]
    lower: int = 0
    upper: int = -1
    issue_order: int = 0
    scalar_source: str | None = None
    period: int = 1
    phase: int = 0
    logical_phase: int = 0
    phase_width: int = 1
    last: bool = False

    def active(self, g):
        return (
            self.lower <= g < self.upper
            and self.phase <= g % self.period < self.phase + self.phase_width
        )

    def modes_for(self, g):
        if self.native == "mad_first":
            return ("R", "R", "W" if g == 0 else "RW")
        if self.native == "mad_cycle":
            return ("R", "R", "W" if g % self.params[3] == 0 else "RW")
        return self.modes

    @property
    def implicit_completion(self):
        if self.pipe == "S":
            return True
        if self.pipe != "M":
            return False
        m, n = (16, 16) if self.native == "mad16" else self.params[:2]
        # AtlasA2 CATLASS TileMmad fences short instructions: <10 output
        # fractals cannot supply the prior blanket M completion guarantee.
        return (m // 16) * (n // 16) >= 10


@dataclass(frozen=True, order=True)
class Instance:
    generation: int
    task: int


@dataclass(frozen=True)
class Obligation:
    source: Instance
    destination: Instance
    hazard: str
    storage: str
    start: int
    stop: int
    milestone: str


@dataclass(frozen=True)
class Program:
    name: str
    core: str
    count: int
    storages: tuple[Storage, ...]
    tasks: tuple[Task, ...]
    config: bool
    blocks: int = 1
    epoch_divisor: int = 1
    load3d: tuple[int, int] | None = None

    @cached_property
    def _semantic_instances(self):
        instances = [
            Instance(g, t)
            for g in range(self.count)
            for t, task in enumerate(self.tasks)
            if task.active(g)
        ]
        return tuple(
            sorted(
                instances,
                key=lambda i: (
                    i.generation // self.epoch_divisor,
                    self.tasks[i.task].logical_phase,
                    i.generation,
                    i.task,
                ),
            ),
        )

    def instances(self):
        return self._semantic_instances

    def issued(self):
        return sorted(
            self.instances(),
            key=lambda i: (
                i.generation + self.tasks[i.task].stage,
                self.tasks[i.task].issue_order,
                i.task,
            ),
        )


def _disjoint_ub_copy(source: Ref, destination: Ref):
    """Prove disjoint contiguous operands without a memmove assumption.

    Different physical UB allocations are separate. For two views of one
    allocation, equal translations leave a constant relative displacement;
    this remains valid for every generation, owner and runtime count. Other
    aliases require a stronger native overlap contract and are rejected.
    """
    if source.storage != destination.storage:
        return True

    def translation(ref):
        generation, core = {}, {}
        for _, stride, modulus, divisor in ref.terms:
            key = modulus, divisor
            generation[key] = generation.get(key, 0) + stride
        for stride, divisor, modulus in ref.core_terms:
            key = modulus, divisor
            core[key] = core.get(key, 0) + stride
        return (
            {key: stride for key, stride in generation.items() if stride},
            {key: stride for key, stride in core.items() if stride},
            ref.count_offset,
            ref.count_core_stride,
        )

    displacement = sum(term[0] for term in destination.terms) - sum(
        term[0] for term in source.terms
    )
    return (
        translation(source) == translation(destination) and abs(displacement) >= source.total_nbytes
    )


def native_contract(native: str, refs: tuple[Ref, ...], params: tuple[int, ...]):
    if native in {"load3d_a", "load3d_b_nn"}:
        # Complete b16 matrix, unit filters/strides, zero padding. Layout and
        # byte footprints are identical to the corresponding audited Load2D.
        return native_contract("load2d_a" if native == "load3d_a" else "load2d_b_nn", refs, params)
    banks = tuple(r.storage.bank for r in refs)
    dtypes = tuple(r.dtype for r in refs)
    lengths = tuple(r.total_nbytes for r in refs)
    layouts = tuple(r.layout for r in refs)
    if native == "copy":
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
                all(ref.prefix_extent is None and not ref.tail_nbytes for ref in refs),
                "UB-to-UB copy requires fixed complete 32-byte blocks",
            )
            require(
                _disjoint_ub_copy(*refs),
                "UB-to-UB copy requires provably disjoint source and destination ranges; "
                "overlapping or changing aliases are unsupported",
            )
        return routes[banks], ("R", "W")
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
            and not any(r.tail_nbytes for r in refs)
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
            and all(r.contiguous and not r.tail_nbytes for r in refs)
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
            banks == ("UB",)
            and dtypes == ("f32",)
            and layouts == ("linear",)
            and lengths == (4,)
            and not params,
            "Scalar load requires one f32 UB word",
        )
        return "S", ("R",)
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
    if native in {"fill", "fill_neg_inf"}:
        require(
            len(refs) == 1
            and banks == ("UB",)
            and dtypes == ("f32",)
            and layouts == ("linear",)
            and refs[0].contiguous
            and lengths[0] > 0
            and lengths[0] % 4 == 0
            and not params,
            "Vector fill needs a contiguous f32 UB destination",
        )
        return "V", ("W",)
    if native in {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast"}:
        require(len(params) == 2, "row broadcast arithmetic needs (rows,columns)")
        rows, cols = params
        require(
            1 <= rows <= 255
            and 64 <= cols <= 1024
            and cols % 64 == 0
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
            and all(r.contiguous and not r.tail_nbytes for r in refs)
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
            and all(r.contiguous and not r.tail_nbytes for r in refs)
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
            and all(r.contiguous and not r.tail_nbytes for r in refs)
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
            and all(r.contiguous and not r.tail_nbytes for r in refs)
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
    if native in {"nd2nz", "load2d_a", "load2d_b_nt", "load2d_b_nn"}:
        require(len(params) == 2, "matrix DMA requires (rows, columns)")
        rows, cols = params
        require(
            16 <= rows <= 256
            and 16 <= cols <= (65520 if native == "nd2nz" else 256)
            and rows % 16 == cols % 16 == 0,
            "matrix DMA requires complete b16 fractals and bounded native dimensions",
        )
        require(
            len(refs) == 2
            and dtypes[0] == dtypes[1] in {"f16", "bf16"}
            and lengths == (rows * cols * 2,) * 2,
            "matrix DMA must expose the full source and destination tile",
        )
        if native == "nd2nz":
            require(
                banks == ("GM", "L1")
                and layouts == ("linear", "NZ")
                and refs[0].rows == rows
                and refs[0].nbytes == cols * 2
                and refs[0].row_stride % 2 == 0
                and cols <= refs[0].row_stride // 2 <= 65535
                and refs[1].contiguous,
                "ND2NZ needs precise GM rows, legal leading dimension and compact NZ output",
            )
            return "MTE2", ("R", "W")
        destination = "L0A" if native == "load2d_a" else "L0B"
        layout = "ZZ" if native == "load2d_a" else "ZN"
        require(
            banks == ("L1", destination)
            and layouts == ("NZ", layout)
            and all(r.contiguous for r in refs)
            and (native == "load2d_a" or rows * cols // 256 <= 255),
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
                or (clear in {0, 1} if native == "mad" else 1 <= clear <= MAX_INSTANCES)
            )
            and banks == ("L0A", "L0B", "L0C")
            and dtypes[0] == dtypes[1] in {"f16", "bf16"}
            and dtypes[2] == "f32"
            and layouts == ("ZZ", "ZN", "NZ")
            and lengths == (m * k * 2, n * k * 2, m * n * 4)
            and all(r.contiguous for r in refs),
            "MAD requires exact b16 operand tiles and an f32 accumulator, without padding",
        )
        return "M", ("R", "R", "W" if native == "mad" and clear else "RW")
    if native == "fix":
        require(len(params) == 2, "Fixpipe requires (M,N)")
        m, n = params
        require(
            16 <= m <= 256
            and 16 <= n <= 256
            and m % 16 == n % 16 == 0
            and banks == ("L0C", "GM")
            and dtypes == ("f32", "f32")
            and layouts == ("NZ", "linear")
            and lengths == (m * n * 4,) * 2
            and refs[0].contiguous
            and refs[1].rows == m
            and refs[1].nbytes == n * 4
            and refs[1].row_stride % 4 == 0,
            "Fixpipe requires the exact f32 NZ tile and ND destination rows",
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


def extract(module: ModuleOp) -> Program:
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "one kernel per compilation is required")
    kernel = module.body.block.first_op
    require(isinstance(kernel, KernelOp), "expected npu.kernel")
    require(kernel.target.data == "dav-2201", "only Ascend 910B / dav-2201 is supported")
    require(kernel.core.data in {"aic", "aiv"}, "cross AIC/AIV ownership is not implemented")
    blocks = optional_int(kernel.blocks, 1)
    require(1 <= blocks <= 1024, "independent core grid must be static in [1,1024]")
    require(re.fullmatch(r"[A-Za-z_]\w*", kernel.sym_name.data), "invalid kernel identifier")
    streams = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
    require(
        len(streams) == 1, "one finite stream is required; dynamic/nested streams are unsupported"
    )
    stream = streams[0]
    require(
        stream.limit is None and stream.symbolic is None,
        "symbolic loop must use the loop checker; finite extraction is only for explicit diagnostics",
    )
    count = stream.count.data
    require(
        isinstance(count, int) and 0 <= count <= MAX_INSTANCES,
        f"task count must be a nonnegative static integer <= {MAX_INSTANCES}",
    )
    require(
        len(stream.body.block.args) == 1 and isinstance(stream.body.block.args[0].type, IndexType),
        "stream requires exactly one logical generation argument",
    )
    roots: dict[SSAValue, Storage] = {}
    names = set()
    config = False
    load3d = None
    seen_stream = False
    for op in kernel.body.block.ops:
        if isinstance(op, StreamOp):
            seen_stream = True
        elif isinstance(op, BufferOp):
            require(
                op.extent_count is None and optional_int(op.count_scale, 0) == 0,
                "runtime buffer extent cannot be ignored by finite extraction",
            )
            require(not seen_stream, "storage declarations must precede the stream")
            typ = op.ref.type
            require(isinstance(typ, RefType), "invalid buffer type")
            name, bank, size = op.sym_name.data, typ.bank.data, op.nbytes.data
            require(
                re.fullmatch(r"[A-Za-z_]\w*", name) and name not in names,
                f"duplicate or invalid storage identifier {name!r}",
            )
            names.add(name)
            require(
                name
                not in {
                    "g",
                    "step",
                    "core_id",
                    "stream",
                    "npu_row",
                    "npu_chunk",
                    "npu_fa",
                    "npu_fb",
                    "npu_rpt",
                }
                and not name.startswith("npu_value_"),
                "storage name conflicts with internal ASC variables",
            )
            require(
                bank in {*CAPACITY, "GM"} and typ.dtype.data in ITEMSIZE, "unsupported storage type"
            )
            require(
                op.external.data in {0, 1} and op.initialized.data in {0, 1},
                "storage contracts must be boolean",
            )
            require(
                (bank == "GM") == bool(op.external.data),
                "GM is external; local banks are allocated",
            )
            require(
                size >= 0 and (bank == "GM" or size % 32 == 0), "invalid allocation size/alignment"
            )
            require(
                not op.initialized.data or bank == "GM", "local initialization needs a real writer"
            )
            roots[op.ref] = Storage(
                name,
                bank,
                size,
                typ.dtype.data,
                typ.layout.data,
                bool(op.external.data),
                bool(op.initialized.data),
            )
        elif isinstance(op, FixConfigOp):
            require(
                not seen_stream and not config and kernel.core.data == "aic",
                "FIX configuration must be unique, unconditional and before the stream",
            )
            config = True
        elif isinstance(op, Load3DConfigOp):
            require(
                not seen_stream and load3d is None and kernel.core.data == "aic",
                "Load3D configuration must be unique, unconditional and before the stream",
            )
            load3d = op.a_width.data, op.b_width.data
            require(
                all(16 <= w <= 256 and w % 16 == 0 for w in load3d),
                "invalid invariant Load3D fmatrix width",
            )
        elif not isinstance(op, (ChannelOp, EndOp)):
            raise CompileError(f"unsupported kernel operation {op.name}; cannot ignore effects")
    require(kernel.disjoint_external.data in {0, 1}, "noalias contract must be boolean")
    require(
        sum(s.external for s in roots.values()) <= 1 or kernel.disjoint_external.data,
        "external pointer alias is unknown; declare disjoint_external or use views of one root",
    )
    _, peaks = storage_layout(tuple(roots.values()))
    for bank, capacity in CAPACITY.items():
        usage = peaks.get(bank, 0)
        require(usage <= capacity, f"{bank} capacity exceeded: {usage} > {capacity} bytes")
        require(
            not usage or (kernel.core.data == "aiv") == (bank == "UB"),
            f"{bank} storage has incompatible {kernel.core.data} owner",
        )
    views: dict[SSAValue, Ref] = {}
    scalar_values: dict[SSAValue, str] = {}
    tasks = []
    uids = set()
    # These caches belong to this extraction of actual IR. Ref is immutable;
    # equal refs have exactly the same finite addresses, extents, and holes.
    ref_pool = {}
    addresses = {}
    translations = {}
    alignments = {}

    def finite_addresses(ref):
        key = id(ref)
        if key not in addresses:
            addresses[key] = tuple(ref.address(g) for g in range(max(count, 1)))
        return addresses[key]

    def core_translations(ref):
        key = id(ref)
        if key not in translations:
            base = ref.address(0)
            translations[key] = tuple(
                ref.address(0, core) - base for core in range(blocks if ref.storage.external else 1)
            )
        return translations[key]

    def aligned_origins(ref, alignment):
        key = id(ref), alignment
        if key not in alignments:
            alignments[key] = (
                all(address % alignment == 0 for address in finite_addresses(ref))
                and all(shift % alignment == 0 for shift in core_translations(ref))
                and (ref.rows == 1 or ref.row_stride % alignment == 0)
            )
        return alignments[key]

    for op in stream.body.block.ops:
        if isinstance(op, ViewOp):
            require(
                op.extent_count is None
                and optional_int(op.count_offset, 0) == 0
                and optional_int(op.count_core_stride, 0) == 0,
                "runtime view translation cannot be ignored by finite extraction",
            )
            require(
                optional_int(op.prefix_elements, 0) == 0,
                "runtime valid prefix requires the symbolic access checker",
            )
            require(
                op.generation is stream.body.block.args[0], "view uses a different generation clock"
            )
            require(op.source in roots or op.source in views, "unknown view provenance")
            typ = op.ref.type
            require(
                isinstance(typ, RefType) and typ.dtype.data in ITEMSIZE, "unsupported view dtype"
            )
            o, s, m, length = op.offset.data, op.stride.data, op.modulus.data, op.nbytes.data
            divisor = optional_int(op.divisor, 1)
            require(
                o >= 0 and s >= 0 and m >= 0 and length > 0 and 1 <= divisor <= MAX_INSTANCES,
                "invalid view address/footprint/divisor",
            )
            if op.source in roots:
                root, terms = roots[op.source], ()
                bounds = lambda g, core, size=root.nbytes: ((0, size),)
                core_terms = ()
            else:
                parent = views[op.source]
                root, terms = parent.storage, parent.terms
                bounds = parent.segments
                core_terms = parent.core_terms
            require(typ.bank.data == root.bank, "a view cannot change physical bank/owner")
            tail = op.tail_nbytes.data
            require(
                0 <= tail <= length and tail % ITEMSIZE[typ.dtype.data] == 0,
                "tail footprint must fit the view and its dtype",
            )
            rows, row_stride = optional_int(op.rows, 1), optional_int(op.row_stride, 0)
            core_stride = optional_int(op.core_stride, 0)
            core_divisor = optional_int(op.core_divisor, 1)
            core_modulus = optional_int(op.core_modulus, 0)
            require(
                1 <= rows <= MAX_INSTANCES
                and row_stride >= 0
                and (rows == 1 or row_stride >= length),
                "strided view rows must not overlap",
            )
            require(
                core_stride >= 0
                and (root.external or core_stride == 0)
                and 1 <= core_divisor <= 1024
                and 0 <= core_modulus <= 1024,
                "local physical banks are core-private, not indexed by another core",
            )
            if core_stride:
                core_terms += ((core_stride, core_divisor, core_modulus),)
            extra_terms = []
            for term in () if op.index_terms is None else op.index_terms:
                require(len(term) == 3, "view index terms require stride/divisor/modulus")
                coefficient, scale, modulus = (value.data for value in term)
                require(
                    coefficient >= 0
                    and 1 <= scale <= MAX_INSTANCES
                    and 0 <= modulus <= MAX_INSTANCES,
                    "invalid additional generation coordinate",
                )
                extra_terms.append((0, coefficient, modulus, scale))
            for term in () if op.core_terms is None else op.core_terms:
                require(len(term) == 3, "view core terms require stride/divisor/modulus")
                coefficient, scale, modulus = (value.data for value in term)
                require(
                    coefficient >= 0
                    and (root.external or coefficient == 0)
                    and 1 <= scale <= 1024
                    and 0 <= modulus <= 1024,
                    "invalid additional core coordinate",
                )
                core_terms += ((coefficient, scale, modulus),)
            ref = Ref(
                root,
                terms + ((o, s, m, divisor),) + tuple(extra_terms),
                length,
                typ.dtype.data,
                typ.layout.data,
                tail,
                count - 1,
                rows,
                row_stride,
                core_terms,
            )
            ref = ref_pool.setdefault(ref, ref)
            core_count = blocks if root.external else 1
            generations = range(max(count, 1))
            if op.source in roots:
                # Root allocations have no holes. Core/generation terms are
                # additive and independent, so exact finite maxima suffice.
                values = finite_addresses(ref)
                shifts = core_translations(ref)
                last_g = max(generations, key=lambda g: values[g] + ref.length(g))
                last_core = max(range(core_count), key=shifts.__getitem__)
                stop = (
                    values[last_g]
                    + shifts[last_core]
                    + (rows - 1) * row_stride
                    + ref.length(last_g)
                )
                require(
                    count == 0 and root.external or stop <= root.nbytes,
                    f"view {root.name} generation {last_g}, core {last_core} end {stop} exceeds its parent",
                )
                itemsize = ITEMSIZE[ref.dtype]
                require(
                    aligned_origins(ref, itemsize) and length % itemsize == 0,
                    f"view {root.name} is not aligned for its element type",
                )
                views[op.ref] = ref
                continue
            # With no new core term, a nested view and its already checked
            # parent share a translation. Sparse-row containment is invariant.
            check_cores = (
                range(core_count) if core_stride or op.core_terms is not None else range(1)
            )
            require(
                all(shift % ITEMSIZE[ref.dtype] == 0 for shift in core_translations(ref)),
                f"view {root.name} core translation is not aligned for its element type",
            )
            for core in check_cores:
                for g in range(max(count, 1)):
                    parts = ref.segments(g, core)
                    require(
                        (count == 0 and root.external) or ranges_cover(bounds(g, core), parts),
                        f"view {root.name} generation {g}, core {core} {parts} exceeds its parent",
                    )
                    require(
                        all(a % ITEMSIZE[ref.dtype] == 0 for a, _ in parts)
                        and length % ITEMSIZE[ref.dtype] == 0,
                        f"view {root.name} is not aligned for its element type",
                    )
            views[op.ref] = ref
        elif isinstance(op, TaskOp):
            uid, native, stage = op.uid.data, op.native.data, op.stage.data
            require(
                uid not in uids and re.fullmatch(r"[A-Za-z_]\w*", uid), "duplicate/invalid task ID"
            )
            uids.add(uid)
            require(0 <= stage <= 32, "stage must be a static offset in [0,32]")
            operands = tuple(op.refs)
            scalar_source = None
            if native in {"scale_value", "sub_value"}:
                require(
                    len(operands) == 3 and operands[-1] in scalar_values,
                    f"{native} needs a preceding synchronous Scalar f32 result",
                )
                scalar_source = scalar_values[operands[-1]]
                operands = operands[:-1]
            require(
                all(v in views for v in operands),
                "native task operands must be explicit bounded views",
            )
            refs = tuple(views[v] for v in operands)
            params = tuple(p.data for p in op.params)
            pipe, modes = native_contract(native, refs, params)
            require(
                (op.value is not None) == (native == "scalar_load"),
                "only synchronous Scalar load may return a numerical SSA value",
            )
            if op.value is not None:
                scalar_values[op.value] = uid
            for ref in refs:
                alignment = 4 if pipe == "S" else 32
                require(
                    aligned_origins(ref, alignment),
                    f"{uid}: {pipe} operand origin requires {alignment}-byte alignment",
                )
            if any(r.tail_nbytes for r in refs):
                require(
                    pipe in {"V", "MTE2", "MTE3"} and kernel.core.data == "aiv",
                    "Cube/Fixpipe tail recipes are not implemented",
                )
                tail_refs = tuple(
                    replace(r, nbytes=r.length(count - 1), tail_nbytes=0) for r in refs
                )
                native_contract(native, tail_refs, params)
            if pipe in {"M", "MTE1", "FIX"}:
                for ref in refs:
                    if ref.storage.bank != "GM":
                        require(
                            all(address % 512 == 0 for address in finite_addresses(ref)),
                            f"{uid}: Cube/Fixpipe local origin must be 512-byte aligned",
                        )
            require(
                (kernel.core.data == "aiv")
                == (
                    pipe in {"V", "MTE3", "S"} or (pipe == "MTE2" and refs[-1].storage.bank == "UB")
                ),
                "native task owner mismatch",
            )
            require(
                native not in {"fix16", "fix"} or config,
                "Fixpipe needs explicit invariant ND_PARA initialization",
            )
            scalar = task_scalar(op)
            lower, upper = optional_int(op.lower, 0), optional_int(op.upper, -1)
            upper = count if upper == -1 else min(upper, count)
            require(
                0 <= lower <= upper <= count,
                "task activation must be a bounded contiguous interval",
            )
            period, phase = optional_int(op.period, 1), optional_int(op.phase, 0)
            phase_width = optional_int(op.phase_width, 1)
            logical_phase = optional_int(op.logical_phase, 0)
            require(0 <= logical_phase <= 255, "logical phase must be in [0,255]")
            require(
                1 <= period <= MAX_INSTANCES
                and 0 <= phase < period
                and 1 <= phase_width <= period - phase,
                "static periodic task activation requires a nonempty phase interval inside period<=4096",
            )
            issue_order = optional_int(op.issue_order, len(tasks))
            require(
                0 <= issue_order <= 65535, "fixed issue order must be a bounded nonnegative rank"
            )
            task = Task(
                uid,
                native,
                pipe,
                stage,
                refs,
                modes,
                scalar,
                params,
                lower,
                upper,
                issue_order,
                scalar_source,
                period,
                phase,
                logical_phase,
                phase_width,
            )
            if scalar_source is not None:
                provider = next(t for t in tasks if t.uid == scalar_source)
                require(
                    provider.stage == stage
                    and provider.lower <= lower
                    and provider.upper >= upper
                    and all(provider.active(g) for g in range(lower, upper) if task.active(g))
                    and (provider.issue_order, tasks.index(provider)) < (issue_order, len(tasks)),
                    "Scalar arguments must be captured earlier in the same physical step; "
                    "delayed descriptors need explicit storage",
                )
            # Native lane overlap must itself be legal; no event can repair it.
            if pipe == "V":
                for r in refs[:-1]:
                    dst = refs[-1]
                    if r.storage != dst.storage or r is dst:
                        continue
                    for g in range(count):
                        old, new = r.segments(g), dst.segments(g)
                        require(
                            old == new or not ranges_overlap(old, new),
                            f"{uid}: shifted in-place Vector overlap is unsupported",
                        )
            tasks.append(task)
        elif not isinstance(op, (SyncOp, EndOp)):
            raise CompileError(
                f"unsupported stream operation {op.name}; dynamic control needs a contract"
            )
    require(tasks, "stream must contain native tasks")
    check_load3d(tasks, load3d)
    require(
        sum(sum(t.active(g) for g in range(count)) for t in tasks) <= MAX_INSTANCES,
        f"finite verifier budget exceeded ({MAX_INSTANCES} task instances); no unchecked fallback",
    )
    epoch_divisor = optional_int(kernel.epoch_divisor, 1)
    require(
        type(epoch_divisor) is int
        and 1 <= epoch_divisor <= MAX_INSTANCES
        and count % epoch_divisor == 0,
        "native generation clock must form complete static logical epochs",
    )
    program = Program(
        kernel.sym_name.data,
        kernel.core.data,
        count,
        tuple(roots.values()),
        tuple(tasks),
        config,
        blocks,
        epoch_divisor,
        load3d,
    )
    verify_independent_owners(program)
    return program


def check_load3d(tasks, config):
    """Only the explicit root initializer may write the Load3D register state.

    All other admitted natives preserve it; opaque/external calls are rejected
    by native_contract. The initializer's local S->MTE1 pair is fully consumed
    before stream dispatch, so event ID0 has no lifetime inside any loop.
    """
    for task in tasks:
        if task.native in {"load3d_a", "load3d_b_nn"}:
            index = 0 if task.native == "load3d_a" else 1
            require(
                config is not None and config[index] == task.params[0],
                "Load3D needs a preceding matching invariant FMATRIX configuration",
            )


def verify_independent_owners(program):
    if program.blocks == 1:
        return
    instances = program.instances()
    active_tasks = {i.task for i in instances}
    written_roots = {
        ref.storage.name
        for i in active_tasks
        for ref, mode in zip(program.tasks[i].refs, program.tasks[i].modes)
        if ref.storage.external and "W" in mode
    }
    refs = {
        ref
        for i in active_tasks
        for ref in program.tasks[i].refs
        if ref.storage.name in written_roots
    }
    effects = [
        (ref.storage.name, lo, hi, mode)
        for inst in instances
        for ref, mode in zip(
            program.tasks[inst.task].refs, program.tasks[inst.task].modes_for(inst.generation)
        )
        if ref.storage.name in written_roots
        for lo, hi in ref.segments(inst.generation)
    ]
    verify_translated_owners(
        refs,
        effects,
        program.blocks,
        translation_error="views of a written GM root must share one core translation; core-dependent internal aliases need per-actor synchronization",
        overlap_error="cross-core ownership overlap; handoff/global synchronization is not implemented",
    )


def verify_translated_owners(refs, effects, blocks, *, translation_error, overlap_error):
    """Check every core's additive translation and all exact R/W interval unions.

    Materialize real group-0 effects including holes. Check EVERY written-root
    reference's translation for EVERY core, then sweep ALL translated R/W
    intervals. No generation/core sample or bounding-box alias assumption.
    """
    import heapq

    base = {}
    for root, lo, hi, mode in effects:
        for access in ("R", "W"):
            if access in mode:
                base.setdefault((root, access), []).append((lo, hi))
    unions = {}
    for key, parts in base.items():
        merged = []
        for lo, hi in sorted(parts):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        unions[key] = merged
    translated = {}
    for core in range(blocks):
        shifts = {}
        for ref in refs:
            shifts.setdefault(ref.storage.name, set()).add(ref.address(0, core) - ref.address(0))
        for root, values in shifts.items():
            require(len(values) == 1, f"core {core}, {root}: {translation_error}")
            shift = next(iter(values))
            for access in ("R", "W"):
                translated.setdefault(root, []).extend(
                    (lo + shift, hi + shift, access, core)
                    for lo, hi in unions.get((root, access), ())
                )
    for root, parts in translated.items():
        active = {"R": {}, "W": {}}
        deadlines = []
        for lo, hi, access, core in sorted(parts):
            while deadlines and deadlines[0][0] <= lo:
                end, kind, owner = heapq.heappop(deadlines)
                if active[kind].get(owner) == end:
                    del active[kind][owner]
            conflicts = ("R", "W") if access == "W" else ("W",)
            require(
                all(len(active[kind]) == int(core in active[kind]) for kind in conflicts),
                f"{root}[{lo}:{hi}], core {core}: {overlap_error}",
            )
            active[access][core] = hi
            heapq.heappush(deadlines, (hi, access, core))


def storage_layout(storages):
    offsets, peaks = {}, {}
    for s in storages:
        if s.external:
            continue
        alignment = 32 if s.bank == "UB" else 512
        offset = (peaks.get(s.bank, 0) + alignment - 1) // alignment * alignment
        offsets[s.name] = offset
        peaks[s.bank] = offset + s.nbytes
    return offsets, peaks


def dependencies(program: Program, *, _imports=()) -> tuple[Obligation, ...]:
    """Interval frontier in semantic generation/source order, independent of issue stages.

    Each byte segment has a last writer and ALL unretired readers. Reading one
    subview cannot kill another reader. RW processes the read before the write.
    """
    cuts = {s.name: {0, s.nbytes} for s in program.storages}
    effects = []
    for instance in program.instances():
        task = program.tasks[instance.task]
        accesses = []
        for ref, mode in zip(task.refs, task.modes_for(instance.generation)):
            for start, stop in ref.segments(instance.generation):
                cuts[ref.storage.name].update((start, stop))
                accesses.append((ref.storage.name, start, stop, mode))
        effects.append((instance, accesses))
    points = {name: tuple(sorted(values)) for name, values in cuts.items()}
    positions = {
        name: {point: i for i, point in enumerate(values)} for name, values in points.items()
    }
    # Dense cells index exact cut intervals, including every hole. This avoids
    # re-allocating/hashing a (root, start, stop) key for every native access.
    writers = {name: [None] * (len(values) - 1) for name, values in points.items()}
    readers = {name: [None] * (len(values) - 1) for name, values in points.items()}
    require(set(_imports) <= {s.name for s in program.storages if s.external}, "invalid imports")
    initialized = {s.name: s.initialized or s.name in _imports for s in program.storages}
    edges = {}

    def edge(src, dst, kind, root, a, b):
        if src is None or src == dst:
            return
        spans = edges.setdefault((src, dst, kind, root), [])
        if spans and spans[-1][0] <= a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(b, spans[-1][1]))
        else:
            spans.append((a, b))

    for instance, footprints in effects:
        task = program.tasks[instance.task]
        # Process ALL reads before any write, even with multiple aliased refs.
        for root, start, stop, mode in footprints:
            if "R" not in mode:
                continue
            boundaries, index = points[root], positions[root]
            previous, pending = writers[root], readers[root]
            for cell in range(index[start], index[stop]):
                a, b = boundaries[cell], boundaries[cell + 1]
                require(
                    previous[cell] is not None or initialized[root],
                    f"{task.uid}[{instance.generation}] reads undefined {root}[{a}:{b}]",
                )
                edge(previous[cell], instance, "RAW", root, a, b)
                if pending[cell] is None:
                    pending[cell] = set()
                pending[cell].add(instance)
        for root, start, stop, mode in footprints:
            if "W" not in mode:
                continue
            boundaries, index = points[root], positions[root]
            previous, pending = writers[root], readers[root]
            for cell in range(index[start], index[stop]):
                a, b = boundaries[cell], boundaries[cell + 1]
                edge(previous[cell], instance, "WAW", root, a, b)
                for src in pending[cell] or ():
                    edge(src, instance, "WAR", root, a, b)
                pending[cell] = None
                previous[cell] = instance
    obligations = []
    for (src, dst, kind, root), spans in edges.items():
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        obligations.extend(
            Obligation(
                src, dst, kind, root, a, b, "read_retired" if kind == "WAR" else "write_done"
            )
            for a, b in merged
        )
    return tuple(
        sorted(
            obligations,
            key=lambda o: (
                o.destination.generation,
                o.destination.task,
                o.source.generation,
                o.source.task,
                o.hazard,
                o.storage,
                o.start,
            ),
        )
    )

"""Public AST declarations; device operations never execute as Python tracing.

Data and parameter annotations are ordinary immutable descriptors. The parser
resolves operation names to internal builders without invoking these declarations.
"""

from dataclasses import dataclass
from enum import StrEnum

from ..diagnostics import CompileError, require
from .contracts import IntegerContract


class DType(StrEnum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I64 = "i64"


f32, f16, bf16 = DType.F32, DType.F16, DType.BF16
i64 = DType.I64


class _ParameterType(StrEnum):
    INT64 = "int64"
    CONSTEXPR = "constexpr"

    def __call__(self, *, minimum=0, multiple_of=1):
        require(self is _ParameterType.INT64, "only int64 accepts runtime input contracts")
        return IntegerContract(minimum, multiple_of)


int64, constexpr = _ParameterType.INT64, _ParameterType.CONSTEXPR


@dataclass(frozen=True)
class Tensor:
    """External tensor permissions and source memory contracts.

    write_disjoint declares pairwise disjoint writes within one kernel launch;
    it does not describe scratch slots, input/output aliasing or later launches.
    """

    shape: tuple[int, ...]
    dtype: DType | str
    access: str = "read"
    cache: str = "normal"
    write_disjoint: bool = False


def _ast_only():
    raise CompileError("device syntax requires prim_func or from_source AST capture")


def alloc_ub(shape, dtype=f32, *, name=None):
    _ast_only()


def alloc_mask(shape, *, name=None):
    """Explicit packed predicate storage: 32 bytes per whole 64-lane block."""
    _ast_only()


def alloc_l1(shape, dtype=f16, *, name=None):
    _ast_only()


def alloc_l0a(shape, dtype=f16, *, name=None):
    _ast_only()


def alloc_l0b(shape, dtype=f16, *, name=None):
    _ast_only()


def alloc_l0c(shape, dtype=f32, *, name=None):
    _ast_only()


class Kernel:
    def __init__(self, groups=1, *, kind="vector"):
        _ast_only()


class Stage:
    def __init__(self, offset, *, name=None):
        _ast_only()


class Cube:
    def __init__(self):
        _ast_only()


class Vector(Cube):
    def __init__(self):
        _ast_only()


class Pipelined:
    def __init__(self, count, *, stages, order="source", issue_order=None):
        _ast_only()


def copy(source, destination, *, valid=None, transpose=False, pad_value=None):
    """Copy tiles; GM→L1 can zero-pad and L0C→GM converts to the output dtype.

    ``pad_value=0`` explicitly defines the complete destination NZ plane.
    A smaller GM destination exports the top-left accumulator rectangle.
    """
    _ast_only()


def arange(extent):
    """A logical lane-coordinate vector of one static tile axis.

    The coordinate values are exact integers 0..extent-1, not a stored index
    table. Comparisons and ``& | ~`` form predicates; every other use is a
    source diagnostic.
    """
    _ast_only()


def compare(source, threshold, *, kind="gt", out):
    """Pack an f32 per-lane comparison into explicit predicate storage.

    ``kind`` is one of eq/ne/lt/le/gt/ge. ``threshold`` is a scalar literal or
    a same-shape f32 UB tile. The comparison is one actual read/compute; its
    predicate binds this result and is never recomputed after ``source``
    changes.
    """
    _ast_only()


def materialize(mask, *, out):
    """Write a symbolic coordinate predicate into explicit packed storage."""
    _ast_only()


def mask_and(lhs, rhs, *, out):
    _ast_only()


def mask_or(lhs, rhs, *, out):
    _ast_only()


def mask_not(value, *, out):
    _ast_only()


def where(mask, x, y, *, out):
    """Select two legal, already-defined f32 UB branches per predicate element."""
    _ast_only()


def mul(lhs, rhs, *, out, valid=None, scratch=None, mask=None):
    _ast_only()


def add(lhs, rhs, *, out, mask=None, scratch=None):
    _ast_only()


def sub(lhs, rhs, *, out, scratch=None, mask=None):
    _ast_only()


def maximum(lhs, rhs, *, out, mask=None, scratch=None):
    _ast_only()


def exp(source, *, out, mask=None, scratch=None):
    _ast_only()


def sqrt(source, *, out, mask=None, scratch=None):
    _ast_only()


def rsqrt(source, *, out, mask=None, scratch=None):
    _ast_only()


def div(lhs, rhs, *, out, scratch=None, mask=None):
    _ast_only()


def load_scalar(source):
    _ast_only()


def load_index(source, *, minimum=0, maximum):
    """Read one immutable GM int64 with an inclusive input value contract.

    Bounds describe the caller's metadata, and may use preceding index loads.
    They are not runtime clamps or promises of unique ownership.
    """
    _ast_only()


def broadcast(source, *, out, scratch=None):
    """Materialize an explicitly expanded row/column vector into an f32 tile."""
    _ast_only()


def reduce_sum(source, *, axis, out, scratch, order):
    """Reduce f32 rows using an explicit tree and caller-owned scratch.

    ``block8_then_packed_merge`` uses a one-dimensional workspace containing
    consecutive, disjoint block8 output planes, then writes compact row sums.
    Every stage requires complete 64-element repeats and aligned partial rows.
    This order changes the FP32 sum tree; it does not promise a speedup.
    """
    _ast_only()


def reduce_max(source, *, axis, out, scratch, order):
    """Reduce f32 rows with the same explicit scratch/tree contract as reduce_sum."""
    _ast_only()


def cast(source, *, out, mask=None, scratch=None):
    _ast_only()


def fill(destination, value, *, mask=None, scratch=None):
    """Fill a complete f32 UB tile, or its predicate-selected lanes only.

    ``mask=False`` lanes keep their old destination value, so masked fills and
    masked arithmetic require an already initialized destination. Coordinate
    masks need no scratch; materialized data predicates need an explicit
    same-shape scratch tile for the value.
    """
    _ast_only()


def matmul(a, b, *, accum, initialize=True):
    """Native L0A[M,K] × L0B[K,N] → L0C[M,N], without hidden operands.

    L1→L0 copies and every buffer slot are explicit. For stored Bᵀ[N,K],
    use copy(..., transpose=True) into an L0B[K,N] tile before this operation.
    """
    _ast_only()


def serial(*bounds):
    _ast_only()


def unroll(*bounds):
    _ast_only()


def tile_ranges(extent, width):
    """Specialize at most two (base, count, valid) ranges, omitting empty ones.

    Whole tiles form one range; a clipped final tile forms the other. All
    three fields are constexpr. Iterate count with serial or Pipelined to
    retain device loops instead of expanding individual tiles.
    """
    _ast_only()


def ceildiv(lhs, rhs):
    _ast_only()


def min(lhs, rhs):
    _ast_only()


def max(lhs, rhs):
    _ast_only()

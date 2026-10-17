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


f32, f16, bf16 = DType.F32, DType.F16, DType.BF16


class _ParameterType(StrEnum):
    INT64 = "int64"
    CONSTEXPR = "constexpr"

    def __call__(self, *, minimum=0, multiple_of=1):
        require(self is _ParameterType.INT64, "only int64 accepts runtime input contracts")
        return IntegerContract(minimum, multiple_of)


int64, constexpr = _ParameterType.INT64, _ParameterType.CONSTEXPR


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    dtype: DType | str
    access: str = "read"


def _ast_only():
    raise CompileError("device syntax requires prim_func or from_source AST capture")


def alloc_ub(shape, dtype=f32, *, name=None):
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


def copy(source, destination, *, valid=None, transpose=False):
    _ast_only()


def mul(lhs, rhs, *, out, valid=None, scratch=None):
    _ast_only()


def add(lhs, rhs, *, out):
    _ast_only()


def sub(lhs, rhs, *, out, scratch=None):
    _ast_only()


def maximum(lhs, rhs, *, out):
    _ast_only()


def exp(source, *, out):
    _ast_only()


def sqrt(source, *, out):
    _ast_only()


def rsqrt(source, *, out):
    _ast_only()


def div(lhs, rhs, *, out, scratch=None):
    _ast_only()


def load_scalar(source):
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


def cast(source, *, out):
    _ast_only()


def fill(destination, value):
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


def ceildiv(lhs, rhs):
    _ast_only()


def min(lhs, rhs):
    _ast_only()


def max(lhs, rhs):
    _ast_only()

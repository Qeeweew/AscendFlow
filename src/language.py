"""Public typed AST syntax and source capture for NPU device programs."""

import sys as _sys

from .frontend.contracts import dynamic
from .frontend.parser import capture as _capture
from .frontend.parser import from_source
from .frontend.syntax import (
    Cube,
    DType,
    Kernel,
    Pipelined,
    Stage,
    Tensor,
    Vector,
    add,
    alloc_l0a,
    alloc_l0b,
    alloc_l0c,
    alloc_l1,
    alloc_mask,
    alloc_ub,
    arange,
    bf16,
    cast,
    ceildiv,
    compare,
    constexpr,
    copy,
    div,
    exp,
    f16,
    f32,
    fill,
    int64,
    load_scalar,
    mask_and,
    mask_not,
    mask_or,
    materialize,
    matmul,
    max,
    maximum,
    min,
    mul,
    reduce_max,
    reduce_sum,
    rsqrt,
    serial,
    sqrt,
    sub,
    tile_ranges,
    unroll,
    where,
)


def prim_func(function=None, *, disjoint_external=False):
    """Capture a device function without executing its body."""

    def decorate(func):
        return _capture(func, namespace=_sys.modules[__name__], disjoint_external=disjoint_external)

    return decorate if function is None else decorate(function)


def inline(function):
    """Capture a lexically scoped DSL helper; calls inline its AST, not Python.

    Helpers accept DSL values, may return a scalar/view/tuple with one final
    return, and share the caller's normal lowering and expansion budget.
    """
    return _capture(function, namespace=_sys.modules[__name__], inline=True)


__all__ = [
    "Cube",
    "DType",
    "Kernel",
    "Pipelined",
    "Stage",
    "Tensor",
    "Vector",
    "add",
    "alloc_l0a",
    "alloc_l0b",
    "alloc_l0c",
    "alloc_l1",
    "alloc_mask",
    "alloc_ub",
    "arange",
    "bf16",
    "cast",
    "ceildiv",
    "compare",
    "constexpr",
    "copy",
    "div",
    "dynamic",
    "exp",
    "f16",
    "f32",
    "fill",
    "from_source",
    "inline",
    "int64",
    "load_scalar",
    "mask_and",
    "mask_not",
    "mask_or",
    "materialize",
    "matmul",
    "max",
    "maximum",
    "min",
    "mul",
    "prim_func",
    "reduce_max",
    "reduce_sum",
    "rsqrt",
    "serial",
    "sqrt",
    "sub",
    "tile_ranges",
    "unroll",
    "where",
]

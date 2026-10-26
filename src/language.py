"""Public typed AST syntax and source capture for NPU device programs."""

import sys as _sys

from .frontend.contracts import dynamic
from .frontend.parser import capture as _capture
from .frontend.parser import from_source
from .frontend.syntax import (
    CopyGroup,
    Cube,
    Dataflow,
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
    alloc_workspace,
    arange,
    bf16,
    broadcast,
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
    get_core_num,
    i32,
    i64,
    load_index,
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
    slotted_view,
    sqrt,
    store_index,
    store_scalar,
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
    "CopyGroup",
    "Cube",
    "DType",
    "Dataflow",
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
    "alloc_workspace",
    "arange",
    "bf16",
    "broadcast",
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
    "get_core_num",
    "i32",
    "i64",
    "inline",
    "load_index",
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
    "slotted_view",
    "sqrt",
    "store_index",
    "store_scalar",
    "sub",
    "tile_ranges",
    "unroll",
    "where",
]

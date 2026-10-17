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
    alloc_ub,
    bf16,
    cast,
    ceildiv,
    constexpr,
    copy,
    div,
    exp,
    f16,
    f32,
    fill,
    int64,
    load_scalar,
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
    unroll,
)


def prim_func(function=None, *, disjoint_external=False):
    """Capture a device function without executing its body."""

    def decorate(func):
        return _capture(func, namespace=_sys.modules[__name__], disjoint_external=disjoint_external)

    return decorate if function is None else decorate(function)


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
    "alloc_ub",
    "bf16",
    "cast",
    "ceildiv",
    "constexpr",
    "copy",
    "div",
    "dynamic",
    "exp",
    "f16",
    "f32",
    "fill",
    "from_source",
    "int64",
    "load_scalar",
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
    "unroll",
]

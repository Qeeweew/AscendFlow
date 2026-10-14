"""Tile-style typed source construction with explicit software stage regions.

Python iterates a symbolic pipeline body once. Static Python policy remains
ordinary Python; execution coordinates become xDSL region arguments.
"""

import inspect
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from math import prod

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IndexType, IntAttr, ModuleOp, StringAttr, i1
from xdsl.dialects.builtin import f32 as ir_f32
from xdsl.ir import Block, OpResult, Region, SSAValue, SSAValues

from .model import ITEMSIZE, require, round_f32
from .source_ir import (
    AllocationOp,
    CopyOp,
    DomainOp,
    ElementOp,
    MatmulOp,
    PipelineOp,
    PredicateOp,
    ProgramOp,
    SizeOp,
    StageOp,
    TensorType,
    TileType,
    ViewOp,
    YieldOp,
)


class DType(StrEnum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"


f32, f16, bf16 = DType.F32, DType.F16, DType.BF16


def _ints(values):
    return ArrayAttr(IntAttr(v) for v in values)


def _type(shape, dtype, memory, layout):
    return TensorType(
        _ints(-1 if isinstance(n, _Dimension) else n for n in shape),
        StringAttr(DType(dtype).value),
        StringAttr(memory),
        StringAttr(layout),
    )


@dataclass(frozen=True)
class _Dimension:
    name: str
    multiple: int = 1
    scale: int = 1

    def __mul__(self, factor):
        require(
            type(factor) is int and factor >= 0,
            "one runtime extent times a static integer required",
        )
        return 0 if factor == 0 else replace(self, scale=self.scale * factor)

    __rmul__ = __mul__

    def __bool__(self):
        raise TypeError("runtime lengths cannot control Python branches")


def dynamic(name, *, multiple_of=1):
    """Uniform runtime length. Local/native tile dimensions remain static."""
    require(type(name) is str and name.isidentifier(), "runtime size needs an identifier")
    require(type(multiple_of) is int and multiple_of > 0, "size multiple must be positive")
    return _Dimension(name, multiple_of)


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    dtype: DType | str
    access: str = "read"


@dataclass(frozen=True)
class _Predicate:
    lhs: object
    comparison: str
    rhs: object

    def __bool__(self):
        raise TypeError(
            "device comparisons cannot control Python if; predicate regions are required"
        )


@dataclass(frozen=True)
class _Index:
    # Each term is coefficient * ((coordinate // divisor) % modulus).
    terms: tuple[tuple[SSAValue, int, int, int], ...] = ()
    constant: int = 0

    def __add__(self, other):
        other = _index(other)
        return _Index(self.terms + other.terms, self.constant + other.constant)

    __radd__ = __add__

    def __mul__(self, factor):
        require(type(factor) is int or isinstance(factor, _Dimension), "unsupported index product")
        if isinstance(factor, _Dimension):
            constant = _index(self.constant * factor) if self.constant else _Index()
            return constant + _Index(tuple((v, factor * c, d, m) for v, c, d, m in self.terms))
        return _Index(
            tuple((v, c * factor, d, m) for v, c, d, m in self.terms), self.constant * factor
        )

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + _index(other) * -1

    def __rsub__(self, other):
        return _index(other) - self

    def __floordiv__(self, divisor):
        require(
            type(divisor) is int and divisor > 0 and not self.constant and len(self.terms) == 1,
            "division requires one unshifted coordinate and a positive static divisor",
        )
        value, coefficient, old_divisor, modulus = self.terms[0]
        require(
            coefficient == 1 and modulus == 0, "composed nonlinear index division is unsupported"
        )
        return _Index(((value, 1, old_divisor * divisor, 0),))

    def __mod__(self, modulus):
        require(
            type(modulus) is int and modulus > 0 and not self.constant and len(self.terms) == 1,
            "modulo requires one unshifted coordinate and a positive static modulus",
        )
        value, coefficient, divisor, old_modulus = self.terms[0]
        require(
            coefficient == 1 and old_modulus == 0, "composed nonlinear index modulo is unsupported"
        )
        return _Index(((value, 1, divisor, modulus),))

    def __bool__(self):
        raise TypeError("device coordinates cannot control Python branches")

    def __eq__(self, other):
        return _Predicate(self, "eq", other)

    def __lt__(self, other):
        return _Predicate(self, "lt", other)

    def __le__(self, other):
        return _Predicate(self, "le", other)


def _index(value):
    if isinstance(value, _Index):
        return value
    if isinstance(value, _Dimension):
        return _Index(((_trace().size(value), value.scale, 1, 0),))
    require(type(value) is int, "tensor indices must be static integers or pipeline coordinates")
    return _Index(constant=value)


def _coordinate(value):
    return _Index(((value, 1, 1, 0),))


_current = ContextVar("xdsl_npu_source", default=None)


class _Trace:
    def __init__(self, name, disjoint):
        block = Block(arg_types=[IndexType()])
        self.op = ProgramOp.create(
            attributes={
                "sym_name": StringAttr(name),
                "kind": StringAttr("vector"),
                "groups": IntAttr(1),
                "disjoint_external": IntAttr(int(disjoint)),
            },
            regions=[Region(block)],
        )
        self.blocks = [block]
        self.values = [[]]
        self.aliases = {}
        self.pipelines = []
        self.kernel_entered = False
        self.counter = 0
        self.sizes = {}

    def size(self, dimension):
        if dimension.name not in self.sizes:
            op = SizeOp.create(
                result_types=[IndexType()],
                attributes={
                    "sym_name": StringAttr(dimension.name),
                    "multiple": IntAttr(dimension.multiple),
                },
            )
            block = self.op.body.block
            if block.first_op is None:
                block.add_op(op)
            else:
                block.insert_op_before(op, block.first_op)
            self.sizes[dimension.name] = op
        op = self.sizes[dimension.name]
        require(op.multiple.data == dimension.multiple, "inconsistent contracts for runtime size")
        return op.value

    def add(self, op):
        self.blocks[-1].add_op(op)
        return op

    def name(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def logical(self, value):
        while value in self.aliases:
            value = self.aliases[value]
        return value

    def record(self, value):
        self.values[-1].append(value)

    def resolve_index(self, index):
        """An exported tile keeps its producer's descriptor and storage version."""
        result = _Index(constant=index.constant)
        for value, coefficient, divisor, modulus in index.terms:
            if isinstance(value.owner, SizeOp):
                result += _Index(((value, coefficient, divisor, modulus),))
                continue
            if value.owner in self.blocks:
                result += _Index(((value, coefficient, divisor, modulus),))
                continue
            region = value.owner.parent_op()
            require(
                isinstance(region, PipelineOp),
                "execution-owner coordinate escaped its source scope",
            )
            require(region.count.data > 0, "a zero-trip pipeline has no exported tile")
            literal = (region.count.data - 1) // divisor
            result += coefficient * (literal % modulus if modulus else literal)
        return result

    def enter(self, op):
        self.add(op)
        self.blocks.append(op.body.block)
        self.values.append([])

    def leave(self, op):
        values = self.values.pop()
        self.add(YieldOp.build(operands=[values]))
        self.blocks.pop()
        op.results = SSAValues(OpResult(value.type, op, i) for i, value in enumerate(values))
        for value, result in zip(values, op.results):
            self.aliases[value] = result
        self.values[-1].extend(op.results)


def _trace():
    trace = _current.get()
    require(trace is not None, "source operations require @prim_func tracing")
    return trace


@dataclass(frozen=True)
class _Buffer:
    op: AllocationOp
    shape: tuple[int, ...]
    dtype: DType
    memory: str
    layout: str
    versions: int
    ring: bool

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        generation = None
        if self.ring:
            require(
                indices and isinstance(indices[0], _Index),
                "task ring indexing starts with a task coordinate",
            )
            generation, indices = indices[0], indices[1:]
        strides = tuple(
            prod(self.shape[i + 1 :]) * ITEMSIZE[self.dtype.value] for i in range(len(self.shape))
        )
        view = _View(self, self.shape, strides, _Index(), generation)
        return view[indices] if indices else view

    def _full(self):
        strides = tuple(
            prod(self.shape[i + 1 :]) * ITEMSIZE[self.dtype.value] for i in range(len(self.shape))
        )
        return _View(self, self.shape, strides, _Index())


@dataclass(frozen=True)
class _View:
    buffer: _Buffer
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: _Index
    generation: _Index | None = None

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        require(len(indices) <= len(self.shape), "too many tensor axes")
        indices += (slice(None),) * (len(self.shape) - len(indices))
        shape, strides, offset = [], [], self.offset
        for size, stride, index in zip(self.shape, self.strides, indices):
            if isinstance(index, slice):
                require(index.step in {None, 1}, "strided source slices are not implemented")
                start = _index(0 if index.start is None else index.start)
                stop = _index(size if index.stop is None else index.stop)
                require(start.terms == stop.terms, "slice width must be a static integer")
                width = stop.constant - start.constant
                require(width > 0, "empty or negative source slice")
                if isinstance(size, _Dimension):
                    _runtime_axis(start, size, width)
                else:
                    lo, hi = _index_bounds(start)
                    require(0 <= lo and hi + width <= size, "source slice exceeds its logical axis")
                offset += start * stride
                shape.append(width)
                strides.append(stride)
            else:
                index = _index(index)
                if isinstance(size, _Dimension):
                    _runtime_axis(index, size, 1)
                else:
                    lo, hi = _index_bounds(index)
                    require(0 <= lo and hi < size, "source index exceeds its logical axis")
                offset += index * stride
        require(shape, "Scalar accesses need an explicit Scalar operation")
        return _View(self.buffer, tuple(shape), tuple(strides), offset, self.generation)

    def bound(self):
        trace = _trace()
        generation = self.generation
        if generation is None and self.buffer.versions > 1:
            require(trace.pipelines, "versioned views require a Pipelined task region")
            generation = _coordinate(trace.pipelines[-1].body.block.args[0])
        return replace(
            self,
            offset=trace.resolve_index(self.offset),
            generation=None if generation is None else trace.resolve_index(generation),
        )

    def emit(self):
        trace, buffer = _trace(), self.buffer
        require(trace.pipelines, "executable views require a Pipelined task region")
        bound = self.bound()
        offset = bound.offset
        if buffer.versions > 1:
            offset += _version_index(bound.generation, buffer.versions) * (
                prod(buffer.shape) * ITEMSIZE[buffer.dtype.value]
            )
        if buffer.ring:
            require(buffer.memory == "GM", "task workspace rings reside in GM")
            offset += _coordinate(trace.op.body.block.args[0]) * (
                buffer.versions * prod(buffer.shape) * ITEMSIZE[buffer.dtype.value]
            )
        variables = list(dict.fromkeys(v for v, _, _, _ in offset.terms))
        dynamic_terms, static_terms = [], []
        for variable, coefficient, divisor, modulus in offset.terms:
            if isinstance(coefficient, _Dimension):
                count = trace.size(coefficient)
                if count not in variables:
                    variables.append(count)
                dynamic_terms.append(
                    _ints(
                        (
                            variables.index(variable),
                            variables.index(count),
                            coefficient.scale,
                            divisor,
                            modulus,
                        )
                    )
                )
            else:
                static_terms.append(
                    _ints((variables.index(variable), coefficient, divisor, modulus))
                )
        terms = ArrayAttr(static_terms)
        op = ViewOp.build(
            operands=[buffer.op.ref, variables],
            result_types=[_type(self.shape, buffer.dtype, buffer.memory, buffer.layout)],
            attributes={
                "offset": IntAttr(offset.constant),
                "terms": terms,
                "strides": _ints(self.strides),
                "dynamic_terms": ArrayAttr(dynamic_terms),
            },
        )
        trace.add(op)
        return op.ref


@dataclass(frozen=True)
class _Value:
    view: _View
    value: SSAValue

    @property
    def shape(self):
        return self.view.shape


def _view(value):
    if isinstance(value, _Value):
        return value.view
    if isinstance(value, _Buffer):
        return value._full()
    require(isinstance(value, _View), "expected a typed tensor, view, or tile value")
    return value


def _index_bounds(index):
    index, trace = _trace().resolve_index(index), _trace()
    lo = hi = index.constant
    for value, coefficient, divisor, modulus in index.terms:
        region = value.owner.parent_op()
        if isinstance(region, ProgramOp):
            count = region.groups.data
        elif isinstance(region, PipelineOp):
            if region.limit is not None:
                require(modulus > 0, "runtime index on a static axis needs a bounded mapping")
                count = divisor * modulus
            else:
                count = max(1, region.count.data)
        else:
            require(isinstance(region, DomainOp), "unknown source index scope")
            count = 2 if region.kind.data == "vector" and trace.op.kind.data == "mixed" else 1
        maximum = (count - 1) // divisor
        if modulus:
            maximum = min(modulus - 1, maximum)
        endpoints = 0, coefficient * maximum
        lo += min(endpoints)
        hi += max(endpoints)
    return lo, hi


def _runtime_axis(index, dimension, width):
    trace = _trace()
    require(
        dimension.scale == 1 and width == 1, "runtime axis currently requires one task per index"
    )
    if not index.terms and index.constant == 0:
        return
    require(
        len(index.terms) == 1 and index.constant == 0,
        "runtime axis requires its logical task coordinate",
    )
    variable, coefficient, divisor, modulus = index.terms[0]
    pipeline = variable.owner.parent_op()
    require(
        isinstance(pipeline, PipelineOp)
        and pipeline.limit is trace.size(dimension)
        and (coefficient, divisor, modulus) == (1, 1, 0),
        "runtime axis must use the corresponding pipeline limit and coordinate",
    )


def _version_index(index, versions):
    if not index.terms:
        return _Index(constant=index.constant % versions)
    return index % versions


def _predicate(value):
    if type(value) is bool:
        value = _Predicate(int(value), "eq", 1)
    require(isinstance(value, _Predicate), "initialization requires a boolean or task comparison")
    lhs, rhs = (_trace().resolve_index(_index(v)) for v in (value.lhs, value.rhs))
    variables = list(dict.fromkeys(v for expr in (lhs, rhs) for v, _, _, _ in expr.terms))
    op = PredicateOp.build(
        operands=[variables],
        result_types=[i1],
        attributes={
            "lhs_terms": ArrayAttr(
                _ints((variables.index(v), c, d, m)) for v, c, d, m in lhs.terms
            ),
            "rhs_terms": ArrayAttr(
                _ints((variables.index(v), c, d, m)) for v, c, d, m in rhs.terms
            ),
            "lhs_offset": IntAttr(lhs.constant),
            "rhs_offset": IntAttr(rhs.constant),
            "comparison": StringAttr(value.comparison),
        },
    )
    _trace().add(op)
    return op.value


def _tile(view):
    return TileType(_ints(view.shape), StringAttr(view.buffer.dtype.value))


def _deps(values):
    return [_trace().logical(v.value) for v in values if isinstance(v, _Value)]


def _dependency_inputs(values):
    return _ints(i for i, value in enumerate(values) if isinstance(value, _Value))


def _allocate(
    shape, dtype, memory, *, versions=1, layout="linear", name=None, access="local", ring=False
):
    trace = _trace()
    shape = tuple(shape)
    require(
        shape and all((type(n) is int and n > 0) or isinstance(n, _Dimension) for n in shape),
        "positive static or explicit runtime allocation dimensions required",
    )
    axes = [i for i, n in enumerate(shape) if isinstance(n, _Dimension)]
    require(not axes or memory == "GM", "local native tile shapes must be static")
    require(
        len(axes) <= 1 and all(shape[i].scale == 1 for i in axes),
        "one unscaled runtime tensor axis currently required",
    )
    dtype = DType(dtype)
    name = name or trace.name(memory.lower())
    op = AllocationOp.create(
        operands=[trace.size(shape[i]) for i in axes],
        result_types=[_type(shape, dtype, memory, layout)],
        attributes={
            "sym_name": StringAttr(name),
            "versions": IntAttr(versions),
            "access": StringAttr(access),
            "ring": IntAttr(int(ring)),
            "dynamic_axes": _ints(axes),
        },
    )
    trace.add(op)
    op.ref.name_hint = name
    return _Buffer(op, shape, dtype, memory, layout, versions, ring)


def alloc_ub(shape, dtype=f32, *, versions=1, name=None):
    return _allocate(shape, dtype, "UB", versions=versions, name=name)


def alloc_l1(shape, dtype=f16, *, versions=1, name=None):
    return _allocate(shape, dtype, "L1", versions=versions, layout="NZ", name=name)


def alloc_l0c(shape, dtype=f32, *, versions=1, name=None):
    return _allocate(shape, dtype, "L0C", versions=versions, layout="NZ", name=name)


def alloc_workspace(shape, dtype=f32, *, versions, name=None):
    return _allocate(shape, dtype, "GM", versions=versions, name=name, ring=True)


class Kernel:
    def __init__(self, groups=1, *, kind="vector"):
        self.groups, self.kind = groups, kind

    def __enter__(self):
        trace = _trace()
        require(not trace.kernel_entered, "one Kernel per source function")
        require(self.kind in {"vector", "cube", "mixed"}, "unknown execution group kind")
        trace.kernel_entered = True
        trace.op.attributes.update(groups=IntAttr(self.groups), kind=StringAttr(self.kind))
        return _coordinate(trace.op.body.block.args[0])

    def __exit__(self, *args):
        pass


class _RegionScope:
    def __enter__(self):
        trace = _trace()
        trace.enter(self.op)

    def __exit__(self, *args):
        trace = _trace()
        trace.leave(self.op)


class Stage(_RegionScope):
    def __init__(self, offset, *, name=None):
        trace = _trace()
        require(trace.pipelines, "Stage requires an enclosing Pipelined")
        require(
            type(offset) is int and 0 <= offset < trace.pipelines[-1].stages.data,
            "Stage exceeds this pipeline's horizon",
        )
        self.op = StageOp.create(
            attributes={
                "offset": IntAttr(offset),
                "label": StringAttr(name or trace.name("stage")),
            },
            regions=[Region(Block())],
        )


class Cube(_RegionScope):
    def __init__(self):
        self.op = DomainOp.create(
            attributes={"kind": StringAttr("cube")},
            regions=[Region(Block(arg_types=[IndexType()]))],
        )

    def __enter__(self):
        super().__enter__()
        return _coordinate(self.op.body.block.args[0])


class Vector(Cube):
    def __init__(self):
        super().__init__()
        self.op.attributes["kind"] = StringAttr("vector")


class Pipelined:
    def __init__(self, count, *, stages, order="source"):
        trace = _trace()
        require(
            (type(count) is int and 0 <= count <= (1 << 31) - 1) or isinstance(count, _Dimension),
            "pipeline limit must be an integer or explicit runtime size",
        )
        if isinstance(count, _Dimension):
            require(count.scale == 1, "runtime pipeline limit must be an unscaled size")
        require(
            order == "source",
            "automatic schedule is not implemented; use explicit Stage/source order",
        )
        self.op = PipelineOp.create(
            operands=[trace.size(count)] if isinstance(count, _Dimension) else [],
            attributes={
                "uid": StringAttr(trace.name("pipeline")),
                **({} if isinstance(count, _Dimension) else {"count": IntAttr(count)}),
                "stages": IntAttr(stages),
                "order": StringAttr(order),
            },
            regions=[Region(Block(arg_types=[IndexType()]))],
        )

    def __iter__(self):
        trace = _trace()
        trace.enter(self.op)
        trace.pipelines.append(self.op)
        try:
            yield _coordinate(self.op.body.block.args[0])
        finally:
            trace.pipelines.pop()
            trace.leave(self.op)


def prim_func(function=None, *, disjoint_external=False):
    def decorate(func):
        trace = _Trace(func.__name__, disjoint_external)
        token = _current.set(trace)
        try:
            arguments = []
            for name, parameter in inspect.signature(func).parameters.items():
                typ = parameter.annotation
                require(
                    isinstance(typ, Tensor),
                    f"{name}: expected a Tensor shape/dtype/access annotation",
                )
                arguments.append(
                    _allocate(typ.shape, typ.dtype, "GM", name=name, access=typ.access)
                )
            func(*arguments)
            require(trace.kernel_entered, "source function must contain Kernel")
            trace.add(YieldOp.build(operands=[[]]))
            module = ModuleOp([trace.op])
            module.verify()
            return module
        finally:
            _current.reset(token)

    return decorate if function is None else decorate(function)


def copy(source, destination):
    src, dst = _view(source), _view(destination).bound()
    require(dst.buffer.op.access.data != "read", "copy cannot write a read-only tensor")
    require(
        src.shape == dst.shape and src.buffer.dtype == dst.buffer.dtype,
        "copy requires identical typed extents; use explicit cast",
    )
    op = CopyOp.build(
        operands=[src.emit(), dst.emit(), _deps([source])],
        result_types=[_tile(dst)],
        attributes={"dependency_inputs": _dependency_inputs([source])},
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def _element(kind, values, out, scalar=None):
    dst, inputs = _view(out).bound(), [_view(value) for value in values]
    require(dst.buffer.op.access.data != "read", "arithmetic cannot write a read-only tensor")
    require(
        all(v.shape == dst.shape for v in inputs), "elementwise operands require equal tile shape"
    )
    if kind != "cast":
        require(
            dst.buffer.dtype == f32 and all(v.buffer.dtype == f32 for v in inputs),
            "arithmetic currently requires f32; use explicit cast",
        )
    attrs = {"kind": StringAttr(kind), "dependency_inputs": _dependency_inputs(values)}
    if scalar is not None:
        attrs["scalar"] = FloatAttr(round_f32(scalar), ir_f32)
    op = ElementOp.build(
        operands=[[v.emit() for v in inputs], dst.emit(), _deps(values)],
        result_types=[_tile(dst)],
        attributes=attrs,
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def mul(lhs, rhs, *, out):
    return (
        _element("mul_scalar", [lhs], out, rhs)
        if isinstance(rhs, (float, int))
        else _element("mul", [lhs, rhs], out)
    )


def add(lhs, rhs, *, out):
    return (
        _element("add_scalar", [lhs], out, rhs)
        if isinstance(rhs, (float, int))
        else _element("add", [lhs, rhs], out)
    )


def sub(lhs, rhs, *, out):
    return _element("sub", [lhs, rhs], out)


def exp(source, *, out):
    return _element("exp", [source], out)


def cast(source, *, out):
    return _element("cast", [source], out)


def fill(destination, value):
    return _element("fill", [], destination, value)


def gemm(a, b, accumulator, *, transpose_b=False, initialize=True):
    av, bv, cv = _view(a), _view(b), _view(accumulator).bound()
    require(all(len(v.shape) == 2 for v in (av, bv, cv)), "gemm requires matrix tiles")
    m, k = av.shape
    n, bk = bv.shape if transpose_b else (bv.shape[1], bv.shape[0])
    require(k == bk and cv.shape == (m, n), "gemm matrix extents do not agree")
    require(
        av.buffer.dtype == bv.buffer.dtype
        and av.buffer.dtype in {f16, bf16}
        and cv.buffer.dtype == f32,
        "gemm requires b16 operands and f32 accumulator",
    )
    require(type(transpose_b) is bool, "explicit boolean transpose policy required")
    initialization = _predicate(initialize)
    inputs = [a, b] if initialize is True else [a, b, accumulator]
    op = MatmulOp.build(
        operands=[av.emit(), bv.emit(), cv.emit(), initialization, _deps(inputs)],
        result_types=[_tile(cv)],
        attributes={
            "transpose_b": IntAttr(int(transpose_b)),
            "dependency_inputs": _dependency_inputs(inputs),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(cv, op.value)

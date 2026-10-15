"""Tile-style AST declarations and internal typed operation builders.

prim_func captures source without executing the device function. The builders
below are used by the AST parser; they do not provide an executable tracer.
"""

from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from math import prod

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    StringAttr,
    i1,
)
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
    ReduceOp,
    ScalarReadOp,
    ScaleOp,
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


class _ParameterType(StrEnum):
    INT64 = "int64"
    CONSTEXPR = "constexpr"

    def __call__(self, *, minimum=0, multiple_of=1):
        require(self is _ParameterType.INT64, "only int64 accepts runtime input contracts")
        return _IntegerParameter(minimum, multiple_of)


@dataclass(frozen=True)
class _IntegerParameter:
    minimum: int = 0
    multiple_of: int = 1

    def __post_init__(self):
        require(
            type(self.minimum) is int and 0 <= self.minimum <= (1 << 63) - 1,
            "int64 minimum must be a nonnegative int64 integer",
        )
        require(
            type(self.multiple_of) is int and 0 < self.multiple_of <= (1 << 63) - 1,
            "int64 multiple_of must be a positive int64 integer",
        )
        first = ((self.minimum + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        require(first <= (1 << 63) - 1, "int64 input contract has no representable value")


int64, constexpr = _ParameterType.INT64, _ParameterType.CONSTEXPR


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
    minimum: int = 0

    def __mul__(self, factor):
        if isinstance(factor, (_Dimension, _ExtentProduct)):
            return _extent_product(self, factor)
        require(
            type(factor) is int and factor >= 0,
            "one runtime extent times a static integer required",
        )
        return 0 if factor == 0 else replace(self, scale=self.scale * factor)

    __rmul__ = __mul__

    def __bool__(self):
        raise TypeError("runtime lengths cannot control Python branches")


@dataclass(frozen=True)
class _ExtentProduct:
    """Uniform row-major stride, with the static byte coefficient kept separate.

    This is a source address expression, not a flattened loop coordinate. Its
    factors become a shared arith.muli DAG in the program's parameter scope.
    """

    factors: tuple[_Dimension, ...]
    scale: int = 1

    def __mul__(self, factor):
        if type(factor) is int:
            return 0 if factor == 0 else replace(self, scale=self.scale * factor)
        require(isinstance(factor, (_Dimension, _ExtentProduct)), "unsupported extent product")
        return _extent_product(self, factor)

    __rmul__ = __mul__


def _extent_product(left, right):
    factors = tuple(
        replace(dimension, scale=1)
        for part in (left, right)
        for dimension in (part.factors if isinstance(part, _ExtentProduct) else (part,))
    )
    require(len(factors) <= 8, "tensor address exceeds eight dynamic stride factors")
    return _ExtentProduct(tuple(sorted(factors, key=lambda d: d.name)), left.scale * right.scale)


def dynamic(name, *, minimum=0, multiple_of=1):
    """Uniform runtime length. Local/native tile dimensions remain static."""
    require(type(name) is str and name.isidentifier(), "runtime size needs an identifier")
    contract = _IntegerParameter(minimum, multiple_of)
    return _Dimension(name, contract.multiple_of, minimum=contract.minimum)


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
        require(
            type(factor) is int or isinstance(factor, (_Dimension, _ExtentProduct)),
            "unsupported index product",
        )
        if isinstance(factor, (_Dimension, _ExtentProduct)):
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
    if isinstance(value, (_Dimension, _ExtentProduct)):
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
        self.products = {}
        self.location = None

    def size(self, dimension):
        if isinstance(dimension, _ExtentProduct):
            from xdsl.dialects import arith

            factors = dimension.factors
            if factors not in self.products:
                lhs = self.size(factors[0])
                for end in range(2, len(factors) + 1):
                    key = factors[:end]
                    if key not in self.products:
                        op = arith.MuliOp(lhs, self.size(factors[end - 1]))
                        if self.location is not None:
                            op.location = self.location
                        previous = (
                            list(self.products.values())[-1].owner
                            if self.products
                            else list(self.sizes.values())[-1]
                        )
                        self.op.body.block.insert_op_after(op, previous)
                        self.products[key] = op.result
                    lhs = self.products[key]
            return self.products[factors]
        if dimension.name not in self.sizes:
            op = SizeOp.create(
                result_types=[IndexType()],
                attributes={
                    "sym_name": StringAttr(dimension.name),
                    "minimum": IntAttr(dimension.minimum),
                    "multiple": IntAttr(dimension.multiple),
                },
            )
            if self.location is not None:
                op.location = self.location
            block = self.op.body.block
            sizes = [old for old in block.ops if isinstance(old, SizeOp)]
            if sizes:
                block.insert_op_after(op, sizes[-1])
            elif block.first_op is None:
                block.add_op(op)
            else:
                block.insert_op_before(op, block.first_op)
            self.sizes[dimension.name] = op
        op = self.sizes[dimension.name]
        require(
            (op.minimum.data, op.multiple.data) == (dimension.minimum, dimension.multiple),
            "inconsistent contracts for runtime size",
        )
        return op.value

    def add(self, op):
        if self.location is not None:
            op.location = self.location
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
            if isinstance(value.owner, SizeOp) or value in self.products.values():
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
    require(trace is not None, "source operations require DSL construction")
    return trace


@dataclass(frozen=True)
class _Buffer:
    op: AllocationOp
    shape: tuple[int, ...]
    dtype: DType
    memory: str
    layout: str

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        strides = tuple(
            prod(self.shape[i + 1 :]) * ITEMSIZE[self.dtype.value] for i in range(len(self.shape))
        )
        view = _View(self, self.shape, strides, _Index())
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

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if any(index is None for index in indices):
            require(
                indices[-1] is None and sum(index is None for index in indices) == 1,
                "broadcast supports only one trailing new axis",
            )
            source = self[indices[:-1]]
            require(len(source.shape) == 1, "row broadcast requires one packed vector")
            return _RowBroadcast(source)
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
        return _View(self.buffer, tuple(shape), tuple(strides), offset)

    def bound(self):
        trace = _trace()
        require(
            self.buffer.op.parent in trace.blocks,
            f"allocation {self.buffer.op.sym_name.data} escaped its source region; "
            "declare shared storage in an enclosing scope",
        )
        return replace(self, offset=trace.resolve_index(self.offset))

    def emit(self, valid=None):
        trace, buffer = _trace(), self.buffer
        bound = self.bound()
        offset = bound.offset
        variables = list(dict.fromkeys(v for v, _, _, _ in offset.terms))
        dynamic_terms, static_terms = [], []
        for variable, coefficient, divisor, modulus in offset.terms:
            if isinstance(coefficient, (_Dimension, _ExtentProduct)):
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
        require(
            all(type(stride) is int for stride in self.strides),
            "a native tile needs static inner strides; select runtime axes before the tile",
        )
        valid_attrs = {}
        if valid is not None:
            require(
                isinstance(valid, SSAValue) and isinstance(valid.type, IndexType),
                "valid requires an explicit device index expression",
            )
            if valid not in variables:
                variables.append(valid)
            valid_attrs["valid_index"] = IntAttr(variables.index(valid))
        op = ViewOp.build(
            operands=[buffer.op.ref, variables],
            result_types=[_type(self.shape, buffer.dtype, buffer.memory, buffer.layout)],
            attributes={
                "offset": IntAttr(offset.constant),
                "terms": terms,
                "strides": _ints(self.strides),
                "dynamic_terms": ArrayAttr(dynamic_terms),
                **valid_attrs,
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

    def __getitem__(self, indices):
        require(
            isinstance(indices, tuple)
            and len(indices) == 2
            and isinstance(indices[0], slice)
            and indices[0] == slice(None)
            and indices[1] is None
            and len(self.shape) == 1,
            "tile values support only [:, None]; slice the physical view explicitly",
        )
        return _RowBroadcast(self)


@dataclass(frozen=True)
class _RowBroadcast:
    source: object


def _view(value):
    if isinstance(value, _Value):
        return value.view
    if isinstance(value, _Buffer):
        return value._full()
    require(isinstance(value, _View), "expected a typed tensor, view, or tile value")
    return value


def _index_bounds(index):
    from xdsl.dialects import arith, scf

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
        elif isinstance(region, scf.ForOp):
            lower = region.lb.owner
            require(
                isinstance(lower, arith.ConstantOp) and lower.value.value.data == 0,
                "serial view bounds currently require a zero lower bound",
            )
            upper = region.ub.owner
            if isinstance(upper, arith.ConstantOp):
                count = max(1, upper.value.value.data)
            else:
                require(modulus > 0, "runtime serial index on a static axis needs modulo")
                count = divisor * modulus
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
    if trace.op.attributes.get("frontend") == StringAttr("ast") and index.terms:
        from .scalar import bounded_by, from_ssa, literal

        require(
            len(index.terms) == 1 and index.constant == 0,
            "runtime tile view needs one unshifted logical coordinate",
        )
        variable, coefficient, divisor, modulus = index.terms[0]
        pipeline = variable.owner.parent_op()
        from xdsl.dialects import arith, scf

        if isinstance(pipeline, scf.ForOp):
            require(
                isinstance(pipeline.lb.owner, arith.ConstantOp)
                and pipeline.lb.owner.value.value.data == 0
                and isinstance(pipeline.step.owner, arith.ConstantOp)
                and pipeline.step.owner.value.value.data == 1,
                "runtime tensor axis requires a zero-based unit-step serial coordinate",
            )
            limit = pipeline.ub
        else:
            require(
                isinstance(pipeline, PipelineOp) and pipeline.limit is not None,
                "runtime tile view requires its pipeline bound",
            )
            limit = pipeline.limit
        bound = from_ssa(limit)
        extent = from_ssa(trace.size(dimension))
        if (coefficient, divisor, modulus) == (width, 1, 0) and (
            bound == extent
            and width == 1
            or bound.kind == "ceildiv"
            and bound.args[1] == literal(width)
            and bounded_by(bound.args[0], extent)
        ):
            return
        require(False, "runtime tile view is not bounded by its tensor extent")
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


def _predicate(value):
    if isinstance(value, SSAValue) and value.type == i1:
        return value
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


def _allocate(shape, dtype, memory, *, layout="linear", name=None, access="local"):
    trace = _trace()
    shape = tuple(shape)
    require(
        shape and all((type(n) is int and n > 0) or isinstance(n, _Dimension) for n in shape),
        "positive static or explicit runtime allocation dimensions required",
    )
    axes = [i for i, n in enumerate(shape) if isinstance(n, _Dimension)]
    require(not axes or memory == "GM", "local native tile shapes must be static")
    require(
        len(axes) <= 8 and all(shape[i].scale == 1 for i in axes),
        "at most eight unscaled runtime tensor axes required",
    )
    dtype = DType(dtype)
    name = name or trace.name(memory.lower())
    op = AllocationOp.create(
        operands=[trace.size(shape[i]) for i in axes],
        result_types=[_type(shape, dtype, memory, layout)],
        attributes={
            "sym_name": StringAttr(name),
            "versions": IntAttr(1),
            "access": StringAttr(access),
            "ring": IntAttr(0),
            "dynamic_axes": _ints(axes),
        },
    )
    trace.add(op)
    op.ref.name_hint = name
    return _Buffer(op, shape, dtype, memory, layout)


def alloc_ub(shape, dtype=f32, *, name=None):
    return _allocate(shape, dtype, "UB", name=name)


def alloc_l1(shape, dtype=f16, *, name=None):
    return _allocate(shape, dtype, "L1", layout="NZ", name=name)


def alloc_l0a(shape, dtype=f16, *, name=None):
    return _allocate(shape, dtype, "L0A", layout="ZZ", name=name)


def alloc_l0b(shape, dtype=f16, *, name=None):
    return _allocate(shape, dtype, "L0B", layout="ZN", name=name)


def alloc_l0c(shape, dtype=f32, *, name=None):
    return _allocate(shape, dtype, "L0C", layout="NZ", name=name)


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
    def __init__(self, count, *, stages, order="source", issue_order=None):
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
                **_issue_order(issue_order),
            },
            regions=[Region(Block(arg_types=[IndexType()]))],
        )


def _issue_order(order):
    if order is None:
        return {}
    require(
        type(order) is dict
        and all(
            type(key) is str
            and type(labels) is tuple
            and all(type(label) is str for label in labels)
            for key, labels in order.items()
        ),
        "issue_order requires a literal actor-to-stage-label mapping",
    )
    return {
        "issue_order": DictionaryAttr(
            {key: ArrayAttr(StringAttr(label) for label in labels) for key, labels in order.items()}
        )
    }


def prim_func(function=None, *, disjoint_external=False):
    """Capture a device function without executing its body."""
    from .frontend import capture

    def decorate(func):
        return capture(func, disjoint_external=disjoint_external)

    return decorate if function is None else decorate(function)


def from_source(source, *, constants=None, filename="<dsl>", disjoint_external=False):
    from .frontend import from_source as parse_source

    return parse_source(
        source, constants=constants, filename=filename, disjoint_external=disjoint_external
    )


def copy(source, destination, *, valid=None, transpose=False):
    src, dst = _view(source), _view(destination).bound()
    require(dst.buffer.op.access.data != "read", "copy cannot write a read-only tensor")
    require(type(transpose) is bool, "copy transpose policy must be constexpr boolean")
    if transpose:
        require(
            (src.buffer.memory, dst.buffer.memory) == ("L1", "L0B")
            and len(src.shape) == 2
            and valid is None,
            "transposed copy currently requires a complete L1→L0B matrix",
        )
    require(
        (src.shape[::-1] if transpose else src.shape) == dst.shape
        and src.buffer.dtype == dst.buffer.dtype,
        "copy requires identical typed extents; use explicit cast",
    )
    op = CopyOp.build(
        operands=[src.emit(valid), dst.emit(valid), _deps([source])],
        result_types=[_tile(dst)],
        attributes={
            "dependency_inputs": _dependency_inputs([source]),
            **({"transpose": IntAttr(1)} if transpose else {}),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def _element(kind, values, out, scalar=None, valid=None, scratch=None):
    dst, inputs = _view(out).bound(), [_view(value) for value in values]
    require(dst.buffer.op.access.data != "read", "arithmetic cannot write a read-only tensor")
    if not kind.endswith("_rows"):
        require(
            all(v.shape == dst.shape for v in inputs),
            "elementwise operands require equal tile shape",
        )
    temporary = None if scratch is None else _view(scratch).bound()
    require(
        temporary is None or temporary.buffer.op.access.data != "read",
        "broadcast scratch must be writable",
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
        operands=[
            [v.emit(valid) for v in inputs],
            dst.emit(valid),
            _deps(values),
            None if temporary is None else temporary.emit(),
        ],
        result_types=[_tile(dst)],
        attributes=attrs,
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def _rows(kind, lhs, rhs, out, scratch):
    require(scratch is not None, "row broadcast needs explicit scratch=(rows,8) UB storage")
    return _element(kind + "_rows", [lhs, rhs.source], out, scratch=scratch)


def mul(lhs, rhs, *, out, valid=None, scratch=None):
    if isinstance(rhs, _RowBroadcast):
        require(valid is None, "masked row broadcast is not implemented")
        return _rows("mul", lhs, rhs, out, scratch)
    require(scratch is None, "scratch requires a row broadcast operand")
    if isinstance(rhs, _Scalar):
        require(valid is None, "masked captured-scalar multiplication is not implemented")
        src, dst = _view(lhs), _view(out).bound()
        require(dst.buffer.op.access.data != "read", "multiplication writes a read-only view")
        op = ScaleOp.build(
            operands=[src.emit(), dst.emit(), rhs.value, _deps([lhs])],
            result_types=[_tile(dst)],
            attributes={"dependency_inputs": _dependency_inputs([lhs])},
        )
        _trace().add(op)
        _trace().record(op.value)
        return _Value(dst, op.value)
    return (
        _element("mul_scalar", [lhs], out, rhs, valid)
        if isinstance(rhs, (float, int))
        else _element("mul", [lhs, rhs], out, valid=valid)
    )


def add(lhs, rhs, *, out):
    return (
        _element("add_scalar", [lhs], out, rhs)
        if isinstance(rhs, (float, int))
        else _element("add", [lhs, rhs], out)
    )


def sub(lhs, rhs, *, out, scratch=None):
    if isinstance(rhs, _RowBroadcast):
        return _rows("sub", lhs, rhs, out, scratch)
    require(scratch is None, "scratch requires a row broadcast operand")
    return _element("sub", [lhs, rhs], out)


def maximum(lhs, rhs, *, out):
    return _element("maximum", [lhs, rhs], out)


def exp(source, *, out):
    return _element("exp", [source], out)


def sqrt(source, *, out):
    return _element("sqrt", [source], out)


def rsqrt(source, *, out):
    return _element("rsqrt", [source], out)


def div(lhs, rhs, *, out, scratch=None):
    if isinstance(rhs, _RowBroadcast):
        return _rows("div", lhs, rhs, out, scratch)
    require(scratch is None, "scratch requires a row broadcast operand")
    return _element("div", [lhs, rhs], out)


@dataclass(frozen=True)
class _Scalar:
    value: SSAValue


def load_scalar(source):
    src = _view(source)
    op = ScalarReadOp.build(
        operands=[src.emit(), _deps([source])],
        result_types=[ir_f32],
        attributes={"dependency_inputs": _dependency_inputs([source])},
    )
    _trace().add(op)
    return _Scalar(op.value)


def _reduce(kind, source, *, axis, out, scratch, order):
    src, dst, temporary = _view(source), _view(out).bound(), _view(scratch).bound()
    require(
        all(v.buffer.op.access.data != "read" for v in (dst, temporary)),
        "reduction output and scratch must be writable",
    )
    require(type(axis) is int, "reduction axis is a constexpr integer")
    if axis < 0:
        axis += len(src.shape)
    op = ReduceOp.build(
        operands=[src.emit(), temporary.emit(), dst.emit(), _deps([source])],
        result_types=[_tile(dst)],
        attributes={
            "kind": StringAttr(kind),
            "order": StringAttr(order),
            "axis": IntAttr(axis),
            "dependency_inputs": _dependency_inputs([source]),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def reduce_sum(source, *, axis, out, scratch, order):
    return _reduce("sum", source, axis=axis, out=out, scratch=scratch, order=order)


def reduce_max(source, *, axis, out, scratch, order):
    return _reduce("max", source, axis=axis, out=out, scratch=scratch, order=order)


def cast(source, *, out):
    return _element("cast", [source], out)


def fill(destination, value):
    return _element("fill", [], destination, value)


def matmul(a, b, *, accum, initialize=True):
    """Native L0A[M,K] × L0B[K,N] → L0C[M,N], without hidden operands.

    L1→L0 copies and every buffer slot are explicit. For stored Bᵀ[N,K],
    use copy(..., transpose=True) into an L0B[K,N] tile before this operation.
    """
    require(
        (_view(a).buffer.memory, _view(b).buffer.memory, _view(accum).buffer.memory)
        == ("L0A", "L0B", "L0C"),
        "matmul requires explicit L0A, L0B and L0C tiles; copy L1 operands first",
    )
    return _matmul(a, b, accum, initialize=initialize)


def _matmul(a, b, accumulator, *, initialize):
    av, bv, cv = _view(a), _view(b), _view(accumulator).bound()
    require(all(len(v.shape) == 2 for v in (av, bv, cv)), "matmul requires matrix tiles")
    m, k = av.shape
    bk, n = bv.shape
    require(k == bk and cv.shape == (m, n), "matmul matrix extents do not agree")
    require(
        av.buffer.dtype == bv.buffer.dtype
        and av.buffer.dtype in {f16, bf16}
        and cv.buffer.dtype == f32,
        "matmul requires b16 operands and f32 accumulator",
    )
    initialization = _predicate(initialize)
    inputs = [a, b] if initialize is True else [a, b, accumulator]
    op = MatmulOp.build(
        operands=[av.emit(), bv.emit(), cv.emit(), initialization, _deps(inputs)],
        result_types=[_tile(cv)],
        attributes={
            "transpose_b": IntAttr(0),
            "dependency_inputs": _dependency_inputs(inputs),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(cv, op.value)

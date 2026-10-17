"""Internal typed source-IR construction used only by the restricted AST parser."""

from contextvars import ContextVar
from dataclasses import dataclass, replace

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

from ..diagnostics import require
from ..dialects.source import (
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
    ViewOp,
    YieldOp,
)
from ..dialects.types import TensorType, TileType
from ..target.scalars import round_f32
from .contracts import _Dimension, _ExtentProduct
from .syntax import DType, bf16, f16, f32


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
class _Predicate:
    lhs: object
    comparison: str
    rhs: object

    def __bool__(self):
        raise TypeError(
            "device comparisons cannot control Python if; predicate regions are required"
        )


@dataclass(frozen=True, eq=False)
class _Index:
    """An index SSA expression with a separate static slice displacement."""

    value: SSAValue | None = None
    constant: int = 0

    def __add__(self, other):
        other = _index(other)
        value = self.value if other.value is None else other.value
        if self.value is not None and other.value is not None:
            from xdsl.dialects import arith

            value = _trace().index_op(arith.AddiOp, self.value, other.value)
        return _Index(value, self.constant + other.constant)

    __radd__ = __add__

    def __mul__(self, factor):
        from xdsl.dialects import arith

        other = _index(factor)
        if self.value is None:
            if other.value is None:
                return _Index(constant=self.constant * other.constant)
            return other * self.constant
        if other.value is None:
            if other.constant == 0:
                return _Index()
            if other.constant == 1:
                return self
            return _Index(
                _trace().index_op(arith.MuliOp, self.value, other.constant),
                self.constant * other.constant,
            )
        require(
            _trace().uniform(self.value) or _trace().uniform(other.value),
            "index multiplication requires a uniform parameter factor",
        )
        return _Index(_trace().index_op(arith.MuliOp, self, other))

    __rmul__ = __mul__

    def __sub__(self, other):
        from xdsl.dialects import arith

        other = _index(other)
        if self.value is other.value:
            return _Index(constant=self.constant - other.constant)
        if other.value is None:
            return _Index(self.value, self.constant - other.constant)
        if self.value is None:
            value = _trace().index_op(arith.SubiOp, 0, other.value)
        else:
            value = _trace().index_op(arith.SubiOp, self.value, other.value)
        return _Index(value, self.constant - other.constant)

    def __rsub__(self, other):
        return _index(other) - self

    def __floordiv__(self, divisor):
        from xdsl.dialects import arith

        return self._divide(divisor, arith.DivUIOp)

    def __mod__(self, modulus):
        from xdsl.dialects import arith

        return self._divide(modulus, arith.RemUIOp)

    def _divide(self, divisor, operation):
        from xdsl.dialects import arith

        divisor = _index(divisor)
        if divisor.value is None:
            require(divisor.constant > 0, "index divisor must be positive")
            if self.value is None:
                require(self.constant >= 0, "unsigned index division needs a nonnegative numerator")
                return _Index(
                    constant=(
                        self.constant // divisor.constant
                        if operation is arith.DivUIOp
                        else self.constant % divisor.constant
                    )
                )
        else:
            require(_trace().uniform(divisor.value), "index divisor must be a uniform parameter")
        # Positive runtime divisors and nonnegative numerators require lowering
        # proofs over the actual active coordinate domain.
        return _Index(_trace().index_op(operation, self, divisor))

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
        return _Index(_trace().size(value)) * value.scale
    if isinstance(value, SSAValue):
        require(isinstance(value.type, IndexType), "expected an index scalar")
        return _Index(value)
    require(
        type(value) is int, "tensor indices must be static integers or device index expressions"
    )
    return _Index(constant=value)


def _coordinate(value):
    return _Index(value)


def _index_arithmetic():
    from xdsl.dialects import arith

    return (
        arith.ConstantOp,
        arith.AddiOp,
        arith.SubiOp,
        arith.MuliOp,
        arith.DivUIOp,
        arith.RemUIOp,
        arith.MinSIOp,
        arith.MaxSIOp,
        arith.CeilDivUIOp,
        arith.CmpiOp,
    )


_current = ContextVar("ascendflow_source", default=None)


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
        self.index_ops = {}
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
                        self.products[key] = self.index_op(
                            arith.MuliOp, lhs, self.size(factors[end - 1])
                        )
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

    def scalar(self, value):
        from xdsl.dialects import arith

        index = self.resolve_index(_index(value))
        if index.value is None:
            return self._pure(arith.ConstantOp.from_int_and_width(index.constant, IndexType()))
        if index.constant == 0:
            return index.value
        return self.index_op(arith.AddiOp, index.value, index.constant)

    def index_op(self, operation, *operands, comparison=None):
        values = tuple(self.scalar(value) for value in operands)
        op = operation(*values) if comparison is None else operation(*values, comparison)
        return self._pure(op)

    def _pure(self, op):
        """Place pure arithmetic where all its operands first become available."""
        scopes = []
        for value in op.operands:
            block = value.owner if isinstance(value.owner, Block) else value.owner.parent
            require(block in self.blocks, "index expression escaped its source scope")
            scopes.append(self.blocks.index(block))
        depth = max(scopes, default=0)
        block = self.blocks[depth]
        key = (
            block,
            type(op),
            tuple(op.operands),
            frozenset(op.attributes.items()),
            frozenset(op.properties.items()),
        )
        if key in self.index_ops:
            op.drop_all_references()
            return self.index_ops[key]
        if self.location is not None:
            op.location = self.location
        if depth + 1 < len(self.blocks):
            anchor = self.blocks[depth + 1].parent_op()
            require(anchor is not None, "index scope is detached")
            block.insert_op_before(op, anchor)
        else:
            block.add_op(op)
        self.index_ops[key] = op.results[0]
        return op.results[0]

    def uniform(self, value):
        """Whether a value depends only on declared uniform scalar parameters."""
        seen = set()

        def visit(current, depth):
            if current in seen:
                return True
            if depth > 128 or len(seen) >= 4096:
                return False
            seen.add(current)
            owner = current.owner
            if isinstance(owner, SizeOp):
                return True
            if not isinstance(owner, _index_arithmetic()):
                return False
            return all(visit(operand, depth + 1) for operand in owner.operands)

        return visit(value, 0)

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
        resolved = {}

        def visit(value, depth=0):
            require(depth <= 128 and len(resolved) < 4096, "index expression exceeds SSA budget")
            if value in resolved:
                return resolved[value]
            owner = value.owner
            block = owner if isinstance(owner, Block) else owner.parent
            if block in self.blocks:
                resolved[value] = value
                return value
            if isinstance(owner, Block):
                region = owner.parent_op()
                require(
                    isinstance(region, PipelineOp),
                    "execution-owner coordinate escaped its source scope",
                )
                require(
                    region.count is not None,
                    "dynamic child coordinate cannot escape without a last-iteration proof",
                )
                require(region.count.data > 0, "a zero-trip pipeline has no exported tile")
                result = self.scalar(region.count.data - 1)
            else:
                require(
                    isinstance(owner, _index_arithmetic()),
                    "index expression escaped its source scope",
                )
                operands = tuple(visit(operand, depth + 1) for operand in owner.operands)
                result = self._pure(
                    type(owner).create(
                        operands=operands,
                        result_types=owner.result_types,
                        attributes=owner.attributes.copy(),
                        properties=owner.properties.copy(),
                    )
                )
            resolved[value] = result
            return result

        return _Index(None if index.value is None else visit(index.value), index.constant)

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
        return self._full()[indices]

    def _full(self):
        return _View(
            self, tuple(_Index() for _ in self.shape), self.shape, tuple(range(len(self.shape)))
        )


@dataclass(frozen=True)
class _View:
    buffer: _Buffer
    origins: tuple[_Index, ...]
    widths: tuple[int | _Dimension, ...]
    axes: tuple[int, ...]

    @property
    def shape(self):
        return tuple(self.widths[axis] for axis in self.axes)

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
        require(len(indices) <= len(self.axes), "too many tensor axes")
        indices += (slice(None),) * (len(self.axes) - len(indices))
        origins, widths, axes = list(self.origins), list(self.widths), []
        for axis, index in zip(self.axes, indices):
            size = self.widths[axis]
            restricted = not (
                self.origins[axis].value is None
                and self.origins[axis].constant == 0
                and size == self.buffer.shape[axis]
            )
            if isinstance(index, slice):
                require(index.step in {None, 1}, "strided source slices are not implemented")
                if index.start is None and index.stop is None:
                    axes.append(axis)
                    continue
                start = _index(0 if index.start is None else index.start)
                stop = _index(size if index.stop is None else index.stop)
                difference = stop - start
                require(difference.value is None, "slice width must be a static integer")
                width = difference.constant
                require(width > 0, "empty or negative source slice")
                axes.append(axis)
            else:
                start, width = _index(index), 1
            if start.value is None:
                require(start.constant >= 0, "source index must be nonnegative")
                if type(size) is int:
                    require(
                        start.constant + width <= size,
                        "source slice exceeds its logical axis",
                    )
            else:
                require(
                    not restricted,
                    "dynamic relative indexing of a restricted view needs a range proof",
                )
            require(
                not restricted
                or self.origins[axis].value is None
                or start.value is None
                and start.constant == 0
                and width == size,
                "narrowing a view with a dynamic origin needs a parent range proof",
            )
            origins[axis] += start
            widths[axis] = width
        require(axes, "Scalar accesses need an explicit Scalar operation")
        return _View(self.buffer, tuple(origins), tuple(widths), tuple(axes))

    def bound(self):
        trace = _trace()
        require(
            self.buffer.op.parent in trace.blocks,
            f"allocation {self.buffer.op.sym_name.data} escaped its source region; "
            "declare shared storage in an enclosing scope",
        )
        return replace(self, origins=tuple(trace.resolve_index(origin) for origin in self.origins))

    def emit(self, valid=None):
        trace, buffer = _trace(), self.buffer
        bound = self.bound()
        require(
            all(type(width) is int and width > 0 for width in bound.widths),
            "a native tile needs static widths; select runtime axes before the tile",
        )
        op = ViewOp.build(
            operands=[
                buffer.op.ref,
                [trace.scalar(origin) for origin in bound.origins],
                None if valid is None else trace.scalar(valid),
            ],
            result_types=[_type(bound.shape, buffer.dtype, buffer.memory, buffer.layout)],
            attributes={"widths": _ints(bound.widths), "axes": _ints(bound.axes)},
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


def _predicate(value):
    from xdsl.dialects import arith

    if isinstance(value, SSAValue) and value.type == i1:
        return value
    trace = _trace()
    if type(value) is bool:
        op = PredicateOp.build(
            operands=[[]],
            result_types=[i1],
            attributes={
                "lhs_terms": ArrayAttr([]),
                "rhs_terms": ArrayAttr([]),
                "lhs_offset": IntAttr(int(value)),
                "rhs_offset": IntAttr(1),
                "comparison": StringAttr("eq"),
            },
        )
        trace.add(op)
        return op.value
    require(isinstance(value, _Predicate), "initialization requires a boolean or task comparison")
    comparison = {"eq": "eq", "lt": "slt", "le": "sle"}[value.comparison]
    return trace.index_op(arith.CmpiOp, value.lhs, value.rhs, comparison=comparison)


def _tile(view):
    return TileType(_ints(view.shape), StringAttr(view.buffer.dtype.value))


def _deps(values):
    return [_trace().logical(v.value) for v in values if isinstance(v, _Value)]


def _dependency_inputs(values):
    return _ints(i for i, value in enumerate(values) if isinstance(value, _Value))


def _allocate(shape, dtype, memory, *, layout="linear", name=None, access="local", cache="normal"):
    trace = _trace()
    shape = tuple(shape)
    require(
        shape and all((type(n) is int and n > 0) or isinstance(n, _Dimension) for n in shape),
        "positive static or explicit runtime allocation dimensions required",
    )
    axes = [i for i, n in enumerate(shape) if isinstance(n, _Dimension)]
    require(not axes or memory == "GM", "local native tile shapes must be static")
    require(
        type(cache) is str
        and cache in {"normal", "bypass"}
        and (cache == "normal" or memory == "GM"),
        "cache policy must be normal or GM bypass",
    )
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
            **({"cache": StringAttr(cache)} if cache != "normal" else {}),
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
            (type(count) is int and 0 <= count <= (1 << 63) - 1) or isinstance(count, _Dimension),
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
        attrs["scalar"] = FloatAttr(round_f32(scalar, allow_infinity=kind == "fill"), ir_f32)
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
    """Reduce f32 rows using an explicit tree and caller-owned scratch.

    ``block8_then_packed_merge`` uses a one-dimensional workspace containing
    consecutive, disjoint block8 output planes, then writes compact row sums.
    Every stage requires complete 64-element repeats and aligned partial rows.
    This order changes the FP32 sum tree; it does not promise a speedup.
    """
    return _reduce("sum", source, axis=axis, out=out, scratch=scratch, order=order)


def reduce_max(source, *, axis, out, scratch, order):
    """Reduce f32 rows with the same explicit scratch/tree contract as reduce_sum."""
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

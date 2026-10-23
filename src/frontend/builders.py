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
    CompareOp,
    CopyGroupOp,
    CopyOp,
    DomainOp,
    ElementOp,
    IndexReadOp,
    IterationOp,
    MaskOp,
    MaterializeMaskOp,
    MatmulOp,
    PredicateOp,
    ProgramOp,
    ReduceOp,
    ScalarReadOp,
    ScalarWriteOp,
    ScaleOp,
    SizeOp,
    StageOp,
    StorageSelectOp,
    TileSliceOp,
    ViewOp,
    YieldOp,
    copy_compatible,
)
from ..dialects.types import TensorType, TileType
from ..mask import (
    ADD,
    AND,
    BROADCAST,
    CMP,
    CMP_KINDS,
    CONST,
    DIVC,
    LANE,
    MODC,
    MULC,
    NEG,
    NOT,
    OR,
    SCALAR,
    SUB,
    Node,
    Program,
    append_boolean,
    broadcast_shapes,
    rebase,
    shift,
    validate,
)
from ..target.scalars import INTEGER_OPERATIONS, round_f32
from .contracts import _Dimension, _ExtentProduct
from .syntax import DType, bf16, f16, f32, i32, i64


def _ints(values):
    return ArrayAttr(IntAttr(v) for v in values)


def _type(shape, dtype, memory, layout):
    return TensorType(
        _ints(-1 if isinstance(n, _Dimension) else n for n in shape),
        StringAttr(dtype.value if isinstance(dtype, DType) else dtype),
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
        if isinstance(other, _Coord):
            return other._binary(self, ADD)
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
        return _Index(_trace().index_op(arith.MuliOp, self, other))

    __rmul__ = __mul__

    def __sub__(self, other):
        from xdsl.dialects import arith

        if isinstance(other, _Coord):
            return _coord(self)._binary(other, SUB)
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
        # Actual source evaluations require a nonnegative dividend and a
        # positive divisor. Dynamic definedness belongs to the input contract.
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
        self.iterations = []
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
        """Keep original arithmetic in its evaluation scope until admission.

        Operand availability alone does not authorize speculative division or
        overflowing arithmetic before a branch or an empty loop. Constants
        may be shared at the root; later scalar passes hoist admitted work.
        """
        from xdsl.dialects import arith

        for value in op.operands:
            block = value.owner if isinstance(value.owner, Block) else value.owner.parent
            require(block in self.blocks, "index expression escaped its source scope")
        depth = 0 if isinstance(op, arith.ConstantOp) else len(self.blocks) - 1
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
                    isinstance(region, IterationOp),
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
                    isinstance(owner, (*INTEGER_OPERATIONS, StorageSelectOp)),
                    "index expression escaped its source scope",
                )
                operands = (
                    (owner.root, visit(owner.selector, depth + 1))
                    if isinstance(owner, StorageSelectOp)
                    else tuple(visit(operand, depth + 1) for operand in owner.operands)
                )
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
            if len(indices) == 2 and indices[0] is None:
                source = self[indices[1]]
                require(len(source.shape) == 1, "column broadcast requires one vector")
                return _ColumnBroadcast(source)
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
        if isinstance(indices, tuple) and indices == (None, slice(None)):
            require(len(self.shape) == 1, "column broadcast requires one vector")
            return _ColumnBroadcast(self)
        if not (
            isinstance(indices, tuple)
            and len(indices) == 2
            and isinstance(indices[0], slice)
            and indices[0] == slice(None)
            and indices[1] is None
            and len(self.shape) == 1
        ):
            view = self.view[indices]
            if isinstance(view, (_ColumnBroadcast, _RowBroadcast)):
                return type(view)(self)
            op = TileSliceOp.build(
                operands=[_trace().logical(self.value), view.emit()],
                result_types=[_tile(view)],
            )
            _trace().add(op)
            _trace().record(op.value)
            return _Value(view, op.value)
        return _RowBroadcast(self)


@dataclass(frozen=True)
class _SlottedView:
    storage: _View
    axis: int

    @property
    def depth(self):
        return self.storage.shape[self.axis]

    @property
    def shape(self):
        return self.storage.shape[: self.axis] + self.storage.shape[self.axis + 1 :]

    def select(self, selector, *, physical=False):
        original = _index(selector)
        if original.value is None:
            require(
                original.constant >= 0 and (not physical or original.constant < self.depth),
                "storage selection exceeds its physical slot axis",
            )
        trace = _trace()
        op = StorageSelectOp.build(
            operands=[self.storage.buffer.op.ref, trace.scalar(original)],
            attributes={
                "axis": IntAttr(self.storage.axes[self.axis]),
                "depth": IntAttr(self.depth),
                "mode": StringAttr("physical" if physical else "generation"),
            },
            result_types=[IndexType()],
        )
        trace.add(op)
        indices = [slice(None)] * len(self.storage.axes)
        indices[self.axis] = _Index(op.index)
        return self.storage[tuple(indices)]


@dataclass(frozen=True)
class _SlottedMask:
    slots: _SlottedView
    shape: tuple[int, ...]

    @property
    def storage(self):
        return self.slots.storage

    def select(self, selector, *, physical=False):
        return _MaskBuffer(self.slots.select(selector, physical=physical), self.shape)


def slotted_view(view, *, axis=0):
    view = _view(view)
    require(
        type(axis) is int and 0 <= axis < len(view.axes),
        "slotted_view axis must name a retained static axis",
    )
    require(
        len(view.axes) > 1 and type(view.shape[axis]) is int and view.shape[axis] > 0,
        "slotted_view needs a positive static slot axis and nonempty payload",
    )
    return _SlottedView(view, axis)


@dataclass(frozen=True)
class _RowBroadcast:
    source: object


@dataclass(frozen=True)
class _ColumnBroadcast:
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


def _mask_type(shape):
    from ..dialects.types import MaskType

    return MaskType(_ints(shape))


@dataclass(frozen=True)
class _MaskBuffer:
    """Explicit packed predicate storage; one 32-byte block per 64 lanes."""

    buffer: _Buffer | _View
    shape: tuple[int, ...]

    def __post_init__(self):
        require(self.view.buffer.dtype == "u8", "predicate storage must use packed u8 bytes")
        require(
            self.view.shape == (self.chunks * 32,),
            "predicate storage needs exact 32-byte blocks",
        )

    @property
    def chunks(self):
        from math import prod

        return prod(self.shape) // 64

    @property
    def view(self):
        return self.buffer if isinstance(self.buffer, _View) else self.buffer._full()


@dataclass(frozen=True)
class _MaskStorage:
    view: _View
    chunks: int
    tile: _Value


@dataclass(frozen=True, eq=False)
class _Mask:
    """A logical predicate value with exactly one actual producer."""

    shape: tuple[int, ...]
    _value: SSAValue
    storage: _MaskStorage | None = None

    @property
    def materialized(self):
        return self.storage is not None

    @property
    def value(self):
        return _trace().logical(self._value) if self.materialized else self._value

    def __and__(self, other):
        return _mask_binary(self, other, "and")

    def __or__(self, other):
        return _mask_binary(self, other, "or")

    def __invert__(self):
        require(
            not self.materialized,
            "predicate negation needs explicit storage: use N.mask_not(mask, out=...)",
        )
        op = self.value.owner
        require(isinstance(op, MaskOp), "symbolic predicate lost its source program")
        program = op.program()
        nodes, shapes = append_boolean(
            list(program.nodes), list(program.node_shapes), NOT, [len(program.nodes) - 1]
        )
        return _emit_mask(program.shape, nodes, shapes, tuple(op.scalars))


def _emit_mask(shape, nodes, shapes, scalars):
    program = Program(tuple(shape), tuple(shapes), tuple(nodes))
    validate(program.nodes, program.node_shapes, program.shape)
    trace = _trace()
    op = MaskOp.build(
        operands=[list(scalars)],
        result_types=[_mask_type(program.shape)],
        attributes=program.encode(),
    )
    trace.add(op)
    return _Mask(program.shape, op.value)


def _merge_coords(left, right):
    """Shared scalar operand list plus rebased node lists."""
    scalars = list(left.scalars)
    index = {value: position for position, value in enumerate(scalars)}
    right_map = []
    for value in right.scalars:
        if value not in index:
            index[value] = len(scalars)
            scalars.append(value)
        right_map.append(index[value])
    left_map = [index[value] for value in left.scalars]
    return (
        rebase(left.nodes, left_map),
        rebase(right.nodes, right_map),
        tuple(scalars),
        index,
    )


def _coord(value):
    if isinstance(value, _Coord):
        return value
    if isinstance(value, (_Dimension, _ExtentProduct, _Index)) or (
        type(value) is int and not isinstance(value, bool)
    ):
        return _Coord.from_index(_index(value))
    require(False, "coordinate expressions require N.arange, integer constants or index scalars")
    raise AssertionError("unreachable")


@dataclass(frozen=True, eq=False)
class _Coord:
    """Exact logical lane coordinates with broadcasting arithmetic."""

    shape: tuple[int, ...]
    nodes: tuple[Node, ...]
    shapes: tuple[tuple[int, ...], ...]
    scalars: tuple[SSAValue, ...]

    @classmethod
    def from_index(cls, index):
        nodes, shapes, scalars = [], [], []

        def constant(value):
            nodes.append(Node(CONST, (int(value),)))
            shapes.append(())
            return len(nodes) - 1

        def scalar(value):
            if value not in scalars:
                scalars.append(value)
            nodes.append(Node(SCALAR, (scalars.index(value),)))
            shapes.append(())
            return len(nodes) - 1

        roots = []
        if index.constant:
            roots.append(constant(index.constant))
        if index.value is not None:
            roots.append(scalar(index.value))
        if not roots:
            roots.append(constant(0))
        while len(roots) > 1:
            right = roots.pop()
            left = roots.pop()
            nodes.append(Node(ADD, (left, right)))
            shapes.append(())
            roots.append(len(nodes) - 1)
        return cls((), tuple(nodes), tuple(shapes), tuple(scalars))

    @classmethod
    def arange(cls, extent):
        require(type(extent) is int and 0 < extent <= (1 << 22), "N.arange needs a bounded integer")
        return cls((extent,), (Node(LANE, (0,)),), ((extent,),), ())

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        require(len(self.shape) == 1, "coordinate new axes apply to one-dimensional N.arange")
        if indices == (slice(None),):
            return self
        patterns = ((slice(None), None), (None, slice(None)))
        require(
            any(
                len(indices) == len(pattern)
                and all(
                    (item is None and expected is None)
                    or (item == expected and expected is not None)
                    for item, expected in zip(indices, pattern, strict=True)
                )
                for pattern in patterns
            ),
            "logical lane coordinates support only [:, None] and [None, :] new axes",
        )
        inserted = 0 if indices[0] is None else 1
        axis = 1 - inserted
        new_shape = (1, self.shape[0]) if inserted == 0 else (self.shape[0], 1)
        nodes, shapes = [], []
        for node, stored in zip(self.nodes, self.shapes, strict=True):
            if node.op == LANE:
                node, stored = Node(LANE, (axis,)), new_shape
            elif stored:
                stored = (1, stored[0]) if inserted == 0 else (stored[0], 1)
            nodes.append(node)
            shapes.append(stored)
        return _Coord(new_shape, tuple(nodes), tuple(shapes), self.scalars)

    def _binary(self, other, op):
        other = _coord(other)
        shape = broadcast_shapes(self.shape, other.shape)
        require(shape is not None, "coordinate shapes do not broadcast")
        left_nodes, right_nodes, scalars, _ = _merge_coords(self, other)
        right_nodes = shift(right_nodes, len(left_nodes))
        nodes = (*left_nodes, *right_nodes)
        shapes = (*self.shapes, *other.shapes, shape)
        nodes = (*nodes, Node(op, (len(left_nodes) - 1, len(nodes) - 1)))
        return _Coord(shape, nodes, shapes, scalars)

    def _constant(self, other, op):
        other = _coord(other)
        require(
            not other.scalars and other.shape == (),
            "coordinate scaling and division require an integer constant",
        )
        factor = _fold_constant(other)
        nodes = (*self.nodes, Node(op, (len(self.nodes) - 1, factor)))
        return _Coord(self.shape, nodes, (*self.shapes, self.shape), self.scalars)

    def __add__(self, other):
        return self._binary(other, ADD)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, SUB)

    def __rsub__(self, other):
        return _coord(other)._binary(self, SUB)

    def __mul__(self, other):
        if isinstance(other, int) and not isinstance(other, bool):
            return self._constant(other, MULC)
        require(False, "coordinate multiplication currently requires an integer constant")

    __rmul__ = __mul__

    def __floordiv__(self, other):
        require(type(other) is int and other > 0, "coordinate division needs a positive constant")
        return self._constant(other, DIVC)

    def __mod__(self, other):
        require(type(other) is int and other > 0, "coordinate modulo needs a positive constant")
        return self._constant(other, MODC)

    def __neg__(self):
        return _Coord(
            self.shape,
            (*self.nodes, Node(NEG, (len(self.nodes) - 1,))),
            (*self.shapes, self.shape),
            self.scalars,
        )

    def _compare(self, other, kind):
        other = _coord(other)
        shape = broadcast_shapes(self.shape, other.shape)
        require(shape is not None, "compared coordinate shapes do not broadcast")
        left_nodes, right_nodes, scalars, _ = _merge_coords(self, other)
        right_nodes = shift(right_nodes, len(left_nodes))
        nodes = (*left_nodes, *right_nodes)
        shapes = (*self.shapes, *other.shapes, shape)
        code = CMP_KINDS.index(kind)
        nodes = (*nodes, Node(CMP, (len(left_nodes) - 1, len(nodes) - 1, code)))
        return _emit_mask(shape, nodes, shapes, scalars)

    def __eq__(self, other):
        return self._compare(other, "eq")

    def __ne__(self, other):
        return self._compare(other, "ne")

    def __lt__(self, other):
        return self._compare(other, "lt")

    def __le__(self, other):
        return self._compare(other, "le")

    def __gt__(self, other):
        return self._compare(other, "gt")

    def __ge__(self, other):
        return self._compare(other, "ge")

    def __bool__(self):
        raise TypeError("device lane coordinates cannot control Python branches")


def _fold_constant(coord):
    """Evaluate one scalar-only coordinate expression at construction time."""
    require(not coord.scalars and coord.shape == (), "expected a coordinate constant")
    values = []
    for node in coord.nodes:
        op, args = node.op, node.args
        if op == CONST:
            values.append(args[0])
        elif op == ADD:
            values.append(values[args[0]] + values[args[1]])
        elif op == SUB:
            values.append(values[args[0]] - values[args[1]])
        elif op == MULC:
            values.append(values[args[0]] * args[1])
        elif op == DIVC:
            values.append(values[args[0]] // args[1])
        elif op == MODC:
            values.append(values[args[0]] % args[1])
        elif op == NEG:
            values.append(-values[args[0]])
        else:
            raise AssertionError("not a scalar coordinate expression")
    return values[-1]


def _merge_programs(left, right):
    left_nodes = left.value.owner.program().nodes
    right_nodes = right.value.owner.program().nodes
    left_scalars, right_scalars = left.value.owner.scalars, right.value.owner.scalars
    scalars = list(left_scalars)
    index = {value: position for position, value in enumerate(scalars)}
    right_map = []
    for value in right_scalars:
        if value not in index:
            index[value] = len(scalars)
            scalars.append(value)
        right_map.append(index[value])
    left_map = [index[value] for value in left_scalars]
    return (
        rebase(left_nodes, left_map),
        rebase(right_nodes, right_map),
        tuple(scalars),
    )


def _mask_binary(left, right, kind):
    require(isinstance(right, _Mask), "predicate combination requires two predicate values")
    require(
        not left.materialized and not right.materialized,
        "materialized predicate combination needs explicit storage: "
        "N.mask_and/mask_or(left, right, out=...)",
    )
    shape = broadcast_shapes(left.shape, right.shape)
    require(shape is not None, "combined predicates do not broadcast")
    require(
        isinstance(left.value.owner, MaskOp) and isinstance(right.value.owner, MaskOp),
        "symbolic predicate lost its source program",
    )
    left_program, right_program = left.value.owner.program(), right.value.owner.program()
    left_nodes, right_nodes, scalars = _merge_programs(left, right)
    right_nodes = shift(right_nodes, len(left_nodes))
    nodes = (*left_nodes, *right_nodes)
    shapes = (*left_program.node_shapes, *right_program.node_shapes)
    nodes, shapes = append_boolean(
        list(nodes),
        list(shapes),
        AND if kind == "and" else OR,
        [len(left_nodes) - 1, len(nodes) - 1],
    )
    return _emit_mask(shape, nodes, shapes, scalars)


def _deps(values):
    return [_trace().logical(v.value) for v in values if isinstance(v, _Value)]


def _dependency_inputs(values):
    return _ints(i for i, value in enumerate(values) if isinstance(value, _Value))


def _allocate(
    shape,
    dtype,
    memory,
    *,
    layout="linear",
    name=None,
    access="local",
    cache="normal",
    write_disjoint=False,
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
        type(cache) is str
        and cache in {"normal", "bypass"}
        and (cache == "normal" or memory == "GM"),
        "cache policy must be normal or GM bypass",
    )
    require(
        type(write_disjoint) is bool
        and (not write_disjoint or memory == "GM" and access in {"write", "read_write"}),
        "write_disjoint must be boolean and requires a writable external tensor",
    )
    require(
        len(axes) <= 8 and all(shape[i].scale == 1 for i in axes),
        "at most eight unscaled runtime tensor axes required",
    )
    try:
        dtype = DType(dtype)
    except ValueError:
        # Internal packed predicate storage uses raw u8 bytes, never f32.
        require(dtype == "u8", "unknown native element type")
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
            **({"write_disjoint": IntAttr(1)} if write_disjoint else {}),
        },
    )
    trace.add(op)
    op.ref.name_hint = name
    return _Buffer(op, shape, dtype, memory, layout)


def _local_storage(shape, dtype, memory, *, slots, name, layout="linear"):
    require(
        slots is None or type(slots) is int and slots > 0,
        "slots must be a positive constexpr integer",
    )
    shape = tuple(shape)
    buffer = _allocate(
        shape if slots is None else (slots, *shape), dtype, memory, layout=layout, name=name
    )
    return buffer if slots is None else slotted_view(buffer)


def alloc_ub(shape, dtype=f32, *, slots=None, name=None):
    return _local_storage(shape, dtype, "UB", slots=slots, name=name)


def alloc_l1(shape, dtype=f16, *, slots=None, name=None):
    return _local_storage(shape, dtype, "L1", layout="NZ", slots=slots, name=name)


def alloc_l0a(shape, dtype=f16, *, slots=None, name=None):
    return _local_storage(shape, dtype, "L0A", layout="ZZ", slots=slots, name=name)


def alloc_l0b(shape, dtype=f16, *, slots=None, name=None):
    return _local_storage(shape, dtype, "L0B", layout="ZN", slots=slots, name=name)


def alloc_l0c(shape, dtype=f32, *, slots=None, name=None):
    return _local_storage(shape, dtype, "L0C", layout="NZ", slots=slots, name=name)


class Kernel:
    """Launch independent SPMD groups; mappings obey the source parallel contract."""

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
        require(trace.iterations, "Stage requires an enclosing Dataflow or Pipelined")
        require(
            type(offset) is int and 0 <= offset <= 32,
            "Stage offset must be a constexpr integer in [0,32]",
        )
        parent = trace.blocks[-1].parent_op()
        while isinstance(parent, DomainOp):
            parent = parent.parent_op()
        require(
            parent is trace.iterations[-1],
            "Stage must belong to its iteration through actor scopes",
        )
        family = _actor_family(trace)
        require(
            family is not None,
            "mixed Stage requires an enclosing Cube or Vector scope",
        )
        require(name is None or type(name) is str and bool(name), "Stage name must be nonempty")
        self.op = StageOp.create(
            attributes={
                "offset": IntAttr(offset),
                "label": StringAttr(name or ""),
                "family": StringAttr(family),
            },
            regions=[Region(Block())],
        )


class CopyGroup(_RegionScope):
    def __init__(self, *, name=None):
        trace = _trace()
        require(_actor_family(trace) is not None, "CopyGroup requires one actual actor")
        require(name is None or type(name) is str and bool(name), "CopyGroup name must be nonempty")
        self.op = CopyGroupOp.create(
            attributes={"label": StringAttr(name or "")}, regions=[Region(Block())]
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


def _actor_family(trace):
    for block in reversed(trace.blocks):
        owner = block.parent_op()
        if isinstance(owner, DomainOp):
            return owner.kind.data
    return None if trace.op.kind.data == "mixed" else trace.op.kind.data


def _iteration(count, issue_order, name):
    trace = _trace()
    require(name is None or type(name) is str and bool(name), "iteration name must be nonempty")
    if type(count) is int:
        require(
            0 <= count <= (1 << 63) - 1,
            "iteration count must be a nonnegative int64",
        )
    return IterationOp.create(
        operands=[] if type(count) is int else [trace.scalar(count)],
        attributes={
            "uid": StringAttr(trace.name("iteration")),
            **({"label": StringAttr(name)} if name is not None else {}),
            **({"count": IntAttr(count)} if type(count) is int else {}),
            **_issue_order(issue_order),
        },
        regions=[Region(Block(arg_types=[IndexType()]))],
    )


class Dataflow:
    def __init__(self, count, *, name=None, issue_order=None):
        self.op = _iteration(count, issue_order, name)


class Pipelined:
    def __init__(self, count, *, name=None, issue_order=None):
        family = _actor_family(_trace())
        require(family is not None, "Pipelined is actor-local; use Dataflow for mixed iterations")
        require(
            issue_order is None
            or type(issue_order) is tuple
            and all(type(name) is str for name in issue_order),
            "Pipelined issue_order must be a tuple of local stage names",
        )
        self.op = _iteration(count, None if issue_order is None else {family: issue_order}, name)


def _issue_order(order):
    if order is None:
        return {}
    require(
        type(order) is dict
        and all(
            type(key) is str
            and type(labels) is tuple
            and all(type(label) is str and bool(label) for label in labels)
            for key, labels in order.items()
        ),
        "issue_order requires a literal actor-to-stage-label mapping",
    )
    return {
        "issue_order": DictionaryAttr(
            {key: ArrayAttr(StringAttr(label) for label in labels) for key, labels in order.items()}
        )
    }


def copy(source, destination, *, valid=None, valid_rows=None, transpose=False, pad_value=None):
    src, dst = _view(source), _view(destination).bound()
    require(dst.buffer.op.access.data != "read", "copy cannot write a read-only tensor")
    require(type(transpose) is bool, "copy transpose policy must be constexpr boolean")
    require(
        pad_value is None or type(pad_value) in {int, float} and pad_value == 0,
        "copy padding currently requires pad_value=0",
    )
    if valid_rows is not None:
        require(
            valid is None
            and not transpose
            and len(src.shape) == len(dst.shape) == 2
            and (
                (src.buffer.memory, dst.buffer.memory) == ("GM", "L1")
                or (src.buffer.memory, dst.buffer.memory) == ("UB", "GM")
                and pad_value is None
            ),
            "valid_rows requires a GM→L1 or UB→GM matrix copy",
        )
    if transpose:
        require(
            src.buffer.memory == "L1"
            and dst.buffer.memory in {"L0A", "L0B"}
            and len(src.shape) == 2
            and valid is None,
            "transposed copy requires a complete L1→L0A/L0B matrix",
        )
    require(
        copy_compatible(
            _type(src.shape, src.buffer.dtype, src.buffer.memory, src.buffer.layout),
            _type(dst.shape, dst.buffer.dtype, dst.buffer.memory, dst.buffer.layout),
            transpose=transpose,
            zero_pad=pad_value is not None,
        ),
        "copy requires compatible extents/types; use explicit cast or pad_value=0",
    )
    op = CopyOp.build(
        operands=[
            src.emit(valid_rows if valid_rows is not None else valid),
            dst.emit(
                None
                if valid_rows is not None and pad_value is not None
                else valid_rows
                if valid_rows is not None
                else valid
            ),
            _deps([source]),
        ],
        result_types=[_tile(dst)],
        attributes={
            "dependency_inputs": _dependency_inputs([source]),
            **({"transpose": IntAttr(1)} if transpose else {}),
            **({"zero_pad": IntAttr(1)} if pad_value is not None else {}),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def _mask_operands(mask, destination_shape, scratch, values):
    """Resolve one explicit predicate operand into storage and dependence."""
    if mask is None:
        return None, None, None, []
    require(isinstance(mask, _Mask), "mask= requires an N.arange comparison or data predicate")
    if tuple(mask.shape) != tuple(destination_shape):
        require(
            broadcast_shapes(tuple(mask.shape), tuple(destination_shape))
            == tuple(destination_shape),
            "masked update needs a predicate broadcastable to the destination",
        )
        require(
            not mask.materialized,
            "materialized predicate storage must match the destination lane domain",
        )
        producer = mask.value.owner
        require(isinstance(producer, MaskOp), "symbolic predicate lost its program")
        program = producer.program()
        mask = _emit_mask(
            tuple(destination_shape),
            (*program.nodes, Node(BROADCAST, (len(program.nodes) - 1,))),
            (*program.node_shapes, tuple(destination_shape)),
            tuple(producer.scalars),
        )
    if not mask.materialized:
        return mask.value, None, None, []
    storage = mask.storage
    require(scratch is not None, "data-predicate updates need explicit same-shape scratch")
    dependency = _trace().logical(storage.tile.value)
    return (
        mask.value,
        storage.view.emit(),
        dependency,
        [(len(values), dependency)],
    )


def _element(kind, values, out, scalar=None, valid=None, scratch=None, mask=None):
    dst, inputs = _view(out).bound(), [_view(value) for value in values]
    require(dst.buffer.op.access.data != "read", "arithmetic cannot write a read-only tensor")
    if not kind.endswith(("_rows", "_columns", "_outer")):
        require(
            all(v.shape == dst.shape for v in inputs),
            "elementwise operands require equal tile shape",
        )
    if kind == "fill" and mask is None:
        require(
            dst.buffer.dtype in {f32, f16, bf16},
            "fill requires f32, f16 or bf16",
        )
    elif kind != "cast":
        require(
            dst.buffer.dtype == f32 and all(v.buffer.dtype == f32 for v in inputs),
            "arithmetic currently requires f32; use explicit cast",
        )
    if mask is not None:
        require(valid is None, "masked updates cannot combine a valid prefix")
        require(
            not kind.endswith("_rows") and kind != "cast",
            ("masked row broadcast and conversion are not implemented"),
        )
    mask_value, mask_storage, mask_dependency, extra = _mask_operands(
        mask, dst.shape, scratch, values
    )
    temporary = None if scratch is None else _view(scratch).bound()
    if temporary is not None and not kind.endswith(("_rows", "_outer")):
        require(
            temporary.buffer.op.access.data != "read" and temporary.shape == dst.shape,
            "scratch must be a writable same-shape tile",
        )
        require(
            temporary.buffer.dtype == f32 and temporary.buffer.memory == "UB",
            "scratch must be an f32 UB tile",
        )
        if mask is not None and mask.materialized:
            require(
                temporary.buffer.op is not dst.buffer.op,
                "data-predicate scratch must not alias the preserve-update destination",
            )
    attrs = {
        "kind": StringAttr(kind),
        "dependency_inputs": _ints(
            (
                *(value.data for value in _dependency_inputs(values)),
                *(index for index, _ in extra),
            )
        ),
    }
    if scalar is not None:
        attrs["scalar"] = FloatAttr(round_f32(scalar, allow_infinity=kind == "fill"), ir_f32)
    op = ElementOp.build(
        operands=[
            [v.emit(valid) for v in inputs],
            dst.emit(valid),
            [*_deps(values), *(dependency for _, dependency in extra)],
            None if temporary is None else temporary.emit(),
            mask_value,
            mask_storage,
            mask_dependency,
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


def mul(lhs, rhs, *, out, valid=None, scratch=None, mask=None):
    if isinstance(lhs, _RowBroadcast) and isinstance(rhs, _ColumnBroadcast):
        lhs, rhs = rhs, lhs
    if isinstance(lhs, _ColumnBroadcast) and isinstance(rhs, _RowBroadcast):
        require(valid is None and mask is None, "masked outer products are not implemented")
        require(scratch is not None, "outer product needs explicit (rows,8) UB scratch")
        return _element("mul_outer", [lhs.source, rhs.source], out, scratch=scratch)
    if isinstance(rhs, _ColumnBroadcast):
        require(
            valid is None and mask is None and scratch is None,
            "column broadcast takes no mask or scratch",
        )
        return _element("mul_columns", [lhs, rhs.source], out)
    if isinstance(rhs, _RowBroadcast):
        require(valid is None and mask is None, "masked row broadcast is not implemented")
        return _rows("mul", lhs, rhs, out, scratch)
    if not (mask is not None and mask.materialized):
        require(scratch is None, "scratch requires a data predicate or a row broadcast")
    if isinstance(rhs, _Scalar):
        require(valid is None and mask is None, "masked captured scalars are not implemented")
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
        _element("mul_scalar", [lhs], out, rhs, valid, mask=mask, scratch=scratch)
        if isinstance(rhs, (float, int))
        else _element("mul", [lhs, rhs], out, valid=valid, mask=mask, scratch=scratch)
    )


def add(lhs, rhs, *, out, mask=None, scratch=None):
    if isinstance(rhs, _ColumnBroadcast):
        require(mask is None and scratch is None, "column broadcast takes no mask or scratch")
        return _element("add_columns", [lhs, rhs.source], out)
    return (
        _element("add_scalar", [lhs], out, rhs, mask=mask)
        if isinstance(rhs, (float, int))
        else _element("add", [lhs, rhs], out, mask=mask, scratch=scratch)
    )


def sub(lhs, rhs, *, out, scratch=None, mask=None):
    if isinstance(rhs, _RowBroadcast):
        require(mask is None, "masked row broadcast is not implemented")
        return _rows("sub", lhs, rhs, out, scratch)
    if not (mask is not None and mask.materialized):
        require(scratch is None, "scratch requires a data predicate or a row broadcast")
    return _element("sub", [lhs, rhs], out, scratch=scratch, mask=mask)


def maximum(lhs, rhs, *, out, mask=None, scratch=None):
    return _element("maximum", [lhs, rhs], out, mask=mask, scratch=scratch)


def exp(source, *, out, mask=None, scratch=None):
    return _element("exp", [source], out, mask=mask, scratch=scratch)


def sqrt(source, *, out, mask=None, scratch=None):
    return _element("sqrt", [source], out, mask=mask, scratch=scratch)


def rsqrt(source, *, out, mask=None, scratch=None):
    return _element("rsqrt", [source], out, mask=mask, scratch=scratch)


def div(lhs, rhs, *, out, scratch=None, mask=None):
    if isinstance(rhs, _RowBroadcast):
        require(mask is None, "masked row broadcast is not implemented")
        return _rows("div", lhs, rhs, out, scratch)
    if not (mask is not None and mask.materialized):
        require(scratch is None, "scratch requires a data predicate or a row broadcast")
    return _element("div", [lhs, rhs], out, scratch=scratch, mask=mask)


def arange(extent):
    """One logical lane-coordinate vector; exact integer, never a float table."""
    return _Coord.arange(extent)


def alloc_mask(shape, *, slots=None, name=None):
    """Explicit packed predicate storage: 32 bytes per whole 64-lane block."""
    from math import prod

    shape = tuple(shape)
    require(
        shape and all(type(size) is int and 0 < size <= (1 << 22) for size in shape),
        "alloc_mask needs positive bounded static lane extents",
    )
    lanes = prod(shape)
    require(lanes % 64 == 0, "packed predicates need whole 64-lane blocks")
    chunks = lanes // 64
    buffer = _local_storage(
        (chunks * 32,), "u8", "UB", slots=slots, name=name or _trace().name("mask")
    )
    return (
        _SlottedMask(buffer, shape)
        if isinstance(buffer, _SlottedView)
        else _MaskBuffer(buffer, shape)
    )


def compare(source, threshold, *, kind="gt", out):
    from math import prod

    require(kind in CMP_KINDS, "compare requires one of eq/ne/lt/le/gt/ge")
    require(isinstance(out, _MaskBuffer), "compare requires explicit alloc_mask storage")
    src = _view(source)
    require(
        tuple(out.shape) == tuple(src.shape),
        "data comparison predicate must match the compared tile",
    )
    lanes = prod(src.shape)
    require(lanes % 64 == 0, "packed data comparison needs whole 64-lane blocks")
    chunks = lanes // 64
    scalar = None
    other = None
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        scalar = round_f32(threshold)
    else:
        other = _view(threshold)
        require(
            other.shape == src.shape,
            "data comparison operands require equal tile shapes",
        )
    from ..dialects.types import MaskType

    op = CompareOp.build(
        operands=[src.emit(), None if other is None else other.emit(), out.view.emit()],
        result_types=[
            TileType(_ints((chunks * 32,)), StringAttr("u8")),
            MaskType(_ints(src.shape)),
        ],
        attributes={
            "kind": StringAttr(kind),
            **({"scalar": FloatAttr(scalar, ir_f32)} if scalar is not None else {}),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    storage = _MaskStorage(out.view, chunks, _Value(out.view, op.value))
    _trace().record(op.mask)
    return _Mask(tuple(src.shape), op.mask, storage)


def materialize(mask, *, out):
    require(
        isinstance(mask, _Mask) and not mask.materialized,
        "only symbolic predicates need materialization",
    )
    require(isinstance(out, _MaskBuffer), "materialize requires explicit alloc_mask storage")
    require(
        tuple(out.shape) == tuple(mask.shape),
        "predicate materialization needs matching lane storage",
    )
    from ..dialects.types import MaskType

    op = MaterializeMaskOp.build(
        operands=[mask.value, out.view.emit()],
        result_types=[
            TileType(_ints((out.chunks * 32,)), StringAttr("u8")),
            MaskType(_ints(mask.shape)),
        ],
    )
    _trace().add(op)
    _trace().record(op.value)
    storage = _MaskStorage(out.view, out.chunks, _Value(out.view, op.value))
    _trace().record(op.result)
    return _Mask(mask.shape, op.result, storage)


def _require_mask_out(out):
    require(
        isinstance(out, _MaskBuffer), "masked boolean combination needs explicit alloc_mask out"
    )


def mask_and(lhs, rhs, *, out):
    return _mask_combine(lhs, rhs, "and", out)


def mask_or(lhs, rhs, *, out):
    return _mask_combine(lhs, rhs, "or", out)


def mask_not(value, *, out):
    require(isinstance(value, _Mask), "mask_not requires a predicate value")
    _require_mask_out(out)
    if not value.materialized:
        require(
            tuple(value.shape) == tuple(out.shape),
            "predicate negation needs matching lane storage",
        )
        return materialize(~value, out=out)
    return _mask_combine(value, None, "not", out)


def _mask_combine(lhs, rhs, kind, out):
    from ..dialects.types import MaskType

    require(isinstance(lhs, _Mask), "mask combination needs predicates")
    _require_mask_out(out)
    unary = kind == "not"
    if unary:
        require(rhs is None, "predicate negation takes one operand")
    else:
        require(isinstance(rhs, _Mask), "mask combination needs two predicates")
        require(
            lhs.shape == rhs.shape and lhs.shape == out.shape,
            "masked boolean combination requires equal predicate shapes",
        )
    if not unary and not lhs.materialized and not rhs.materialized:
        combined = _mask_binary(lhs, rhs, kind)
        return materialize(combined, out=out)
    require(
        lhs.materialized and (unary or rhs.materialized),
        ("mixed coordinate/data predicate combination needs an explicit N.materialize first"),
    )
    require(
        tuple(lhs.shape) == tuple(out.shape),
        "masked boolean combination requires equal predicate shapes",
    )
    sources = (lhs.storage,) if unary else (lhs.storage, rhs.storage)
    require(
        all(storage.view.buffer.op is not out.buffer.op for storage in sources),
        "masked boolean combination output needs distinct predicate storage",
    )
    from ..dialects.source import MaskCombineOp as Combine

    dependencies = [_trace().logical(storage.tile.value) for storage in sources]
    rhs_mask = None if unary else rhs.value
    rhs_view = None if unary else rhs.storage.view.emit()
    op = Combine.build(
        operands=[
            lhs.value,
            rhs_mask,
            lhs.storage.view.emit(),
            rhs_view,
            out.view.emit(),
            dependencies,
        ],
        result_types=[
            TileType(_ints((out.chunks * 32,)), StringAttr("u8")),
            MaskType(_ints(out.shape)),
        ],
        attributes={
            "kind": StringAttr(kind),
            "dependency_inputs": _ints(range(len(sources))),
        },
    )
    _trace().add(op)
    _trace().record(op.value)
    storage = _MaskStorage(out.view, out.chunks, _Value(out.view, op.value))
    _trace().record(op.result)
    return _Mask(out.shape, op.result, storage)


def where(mask, x, y, *, out):
    require(
        isinstance(mask, _Mask) and mask.materialized,
        ("where requires a materialized data predicate; coordinate predicates use mask= updates"),
    )
    xv, yv, dst = _view(x), _view(y), _view(out).bound()
    require(
        xv.shape == yv.shape == dst.shape == mask.shape,
        "where requires equal predicate and branch shapes",
    )
    require(
        all(view.buffer.memory == "UB" and view.buffer.dtype == f32 for view in (xv, yv, dst)),
        "where currently requires f32 UB branches",
    )
    require(dst.buffer.op.access.data != "read", "where writes a read-only tensor")
    from ..dialects.source import WhereOp as Select

    dependency = _trace().logical(mask.storage.tile.value)
    op = Select.build(
        operands=[
            mask.value,
            mask.storage.view.emit(),
            dependency,
            xv.emit(),
            yv.emit(),
            dst.emit(),
            [dependency],
        ],
        result_types=[_tile(dst)],
        attributes={"dependency_inputs": _ints((0,))},
    )
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def cast(source, *, out, mask=None, scratch=None):
    require(scratch is None, "conversion keeps no scratch tile")
    return _element("cast", [source], out, mask=mask)


def fill(destination, value, *, mask=None, scratch=None):
    return _element("fill", [], destination, value, mask=mask, scratch=scratch)


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


def store_scalar(value, *, out):
    require(isinstance(value, _Scalar), "store_scalar requires a loaded f32 scalar")
    dst = _view(out).bound()
    require(dst.buffer.op.access.data != "read", "store_scalar requires writable storage")
    op = ScalarWriteOp.build(operands=[value.value, dst.emit()], result_types=[_tile(dst)])
    _trace().add(op)
    _trace().record(op.value)
    return _Value(dst, op.value)


def load_index(source, *, minimum=None, maximum=None):
    src = _view(source)
    require(
        src.buffer.memory == "UB" or src.buffer.op.access.data == "read",
        "GM index metadata must be immutable during the kernel",
    )
    require(src.buffer.dtype in {i32, i64}, "index metadata needs int32 or int64")
    bits = 32 if src.buffer.dtype == i32 else 64
    minimum = -(1 << (bits - 1)) if minimum is None else minimum
    maximum = (1 << (bits - 1)) - 1 if maximum is None else maximum
    op = IndexReadOp.build(
        operands=[src.emit(), _trace().scalar(minimum), _trace().scalar(maximum), _deps([source])],
        result_types=[IndexType()],
        attributes={"dependency_inputs": _dependency_inputs([source])},
    )
    _trace().add(op)
    return _Index(op.value)


def broadcast(source, *, out, scratch=None):
    require(
        isinstance(source, (_ColumnBroadcast, _RowBroadcast)),
        "broadcast requires [None,:] or [:,None]",
    )
    kind = "broadcast_columns" if isinstance(source, _ColumnBroadcast) else "broadcast_rows"
    return _element(kind, [source.source], out, scratch=scratch)


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
        and av.buffer.dtype in {f16, bf16, f32}
        and cv.buffer.dtype == f32,
        "matmul requires equal floating operands and f32 accumulator",
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

"""SSA address construction, separate from future bounds/ownership admission."""

from math import prod

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr
from xdsl.ir import Block

from xdsl_npu import indexing
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.physical import BufferOp
from xdsl_npu.dialects.source import AllocationOp, PipelineOp, SizeOp, ViewOp
from xdsl_npu.dialects.types import TensorType, ref_type
from xdsl_npu.passes.native_addresses import lower_address, subview
from xdsl_npu.target.native import ITEMSIZE


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def _allocation(shape, *, dimensions=(), bank="UB", dtype="f32", layout="linear"):
    return AllocationOp.build(
        operands=[dimensions],
        attributes={
            "sym_name": StringAttr("data"),
            "versions": IntAttr(1),
            "access": StringAttr("local" if bank != "GM" else "read"),
            "ring": IntAttr(0),
            "dynamic_axes": _ints(i for i, size in enumerate(shape) if size == -1),
        },
        result_types=[
            TensorType(_ints(shape), StringAttr(dtype), StringAttr(bank), StringAttr(layout))
        ],
    )


def _index(block, value):
    operation = arith.ConstantOp.from_int_and_width(value, IndexType())
    block.add_op(operation)
    return operation.result


def _view(block, allocation, widths, axes, origins=None, *, valid=None):
    if origins is None:
        origins = tuple(_index(block, 0) for _ in widths)
    typ = allocation.ref.type
    operation = ViewOp.build(
        operands=[allocation.ref, origins, valid],
        attributes={"widths": _ints(widths), "axes": _ints(axes)},
        result_types=[
            TensorType(_ints(widths[axis] for axis in axes), typ.dtype, typ.memory, typ.layout)
        ],
    )
    block.add_op(operation)
    return operation


def _buffer(block, allocation):
    typ = allocation.ref.type
    size = prod(max(1, dim.data) for dim in typ.shape) * ITEMSIZE[typ.dtype.data]
    operation = BufferOp.build(
        operands=[None],
        attributes={
            "sym_name": allocation.sym_name,
            "nbytes": IntAttr(size),
            "external": IntAttr(int(typ.memory.data == "GM")),
            "initialized": IntAttr(int(typ.memory.data == "GM")),
        },
        result_types=[ref_type(typ.memory.data, typ.dtype.data, typ.layout.data)],
    )
    block.add_op(operation)
    return operation


def _materializer(block, bindings=()):
    memo = dict(bindings)

    def materialize(value):
        if value not in memo:
            owner = value.owner
            assert isinstance(
                owner,
                (
                    arith.ConstantOp,
                    arith.AddiOp,
                    arith.SubiOp,
                    arith.MuliOp,
                    arith.DivUIOp,
                    arith.RemUIOp,
                ),
            )
            operation = owner.clone({operand: materialize(operand) for operand in owner.operands})
            block.add_op(operation)
            memo.update(zip(owner.results, operation.results, strict=True))
        return memo[value]

    return materialize


def _lower(view, *, matrix=False, bindings=(), output=None, buffer=None):
    block = Block() if output is None else output
    root = _buffer(block, view.source.owner) if buffer is None else buffer
    value = lower_address(
        view.ref,
        materialize=_materializer(block, bindings),
        roots={view.source.owner: root.ref},
        append=block.add_op,
        matrix=matrix,
    )
    value.owner.verify()
    return value.owner, block, root


def _evaluate(value, bindings=None):
    """Independent integer interpretation, without compiler scalar helpers."""
    bindings = {} if bindings is None else bindings
    if value in bindings:
        return bindings[value]
    owner = value.owner
    if isinstance(owner, arith.ConstantOp):
        return owner.value.value.data
    lhs, rhs = (_evaluate(operand, bindings) for operand in owner.operands)
    if isinstance(owner, arith.AddiOp):
        return lhs + rhs
    if isinstance(owner, arith.SubiOp):
        return lhs - rhs
    if isinstance(owner, arith.MuliOp):
        return lhs * rhs
    if isinstance(owner, arith.DivUIOp):
        return lhs // rhs
    if isinstance(owner, arith.RemUIOp):
        return lhs % rhs
    raise AssertionError(owner.name)


def _leaves(value):
    owner = value.owner
    if isinstance(owner, Block):
        return {value}
    return set().union(*(_leaves(operand) for operand in owner.operands))


def test_multiple_ancestor_coordinates_and_runtime_extent_stay_in_one_ssa_address(monkeypatch):
    source = N.from_source("""
        def nested(X: N.Tensor((n,m,64),N.f32), n:N.int64(minimum=1), m:N.int64(minimum=1)):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for outer in N.serial(n):
                    for inner in N.Pipelined(m,stages=1):
                        N.copy(X[outer,inner,:],x)
    """).to_ir()
    serial = next(op for op in source.walk() if isinstance(op, scf.ForOp))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    sizes = {op.sym_name.data: op.value for op in source.walk() if isinstance(op, SizeOp)}
    view = next(
        op for op in source.walk() if isinstance(op, ViewOp) and op.source.type.memory.data == "GM"
    )
    original = (sizes["n"], sizes["m"], serial.body.block.args[0], pipeline.body.block.args[0])
    output = Block(arg_types=[IndexType()] * 4)
    bindings = tuple(zip(original, output.args, strict=True))

    def no_flatten(*args, **kwargs):
        raise AssertionError(
            "address construction flattened multiple clocks through physical_terms"
        )

    monkeypatch.setattr(indexing, "physical_terms", no_flatten)
    address, _, root = _lower(view, bindings=bindings, output=output)
    assert address.source is root.ref
    assert tuple(address.origins[:2]) == tuple(output.args[2:])
    assert tuple(address.extents[:2]) == tuple(output.args[:2])
    assert tuple(width.data for width in address.widths) == (1, 1, 64)
    assert tuple(axis.data for axis in address.axes) == (2,)
    assert _leaves(address.offset) == {output.args[1], output.args[2], output.args[3]}
    for n, m, outer, inner in ((2, 3, 0, 1), (7, 5, 3, 4), (1024, 16, 1000, 15)):
        runtime = dict(zip(output.args, (n, m, outer, inner), strict=True))
        assert _evaluate(address.offset, runtime) == (outer * m + inner) * 64 * 4


def test_whole_source_expression_and_per_call_lexical_bindings_are_preserved():
    source = N.from_source("""
        def indexed(n:N.int64(minimum=1),m:N.int64(minimum=1)):
            with N.Kernel():
                x=N.alloc_ub((2,64))
                for outer in N.serial(n):
                    for inner in N.Pipelined(m,stages=1):
                        slot=((outer*m+inner+3)//m)%2
                        N.fill(x[slot,:],0.0)
    """).to_ir()
    serial = next(op for op in source.walk() if isinstance(op, scf.ForOp))
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    m = next(op.value for op in source.walk() if isinstance(op, SizeOp) and op.sym_name.data == "m")
    view = next(op for op in source.walk() if isinstance(op, ViewOp))
    output = Block(arg_types=[IndexType()] * 4)
    root = _buffer(output, view.source.owner)
    common = [(m, output.args[0]), (serial.body.block.args[0], output.args[1])]
    addresses = [
        _lower(
            view,
            bindings=[*common, (pipeline.body.block.args[0], output.args[index])],
            output=output,
            buffer=root,
        )[0]
        for index in (2, 3)
    ]
    assert all(address.source is root.ref for address in addresses)
    assert addresses[0].offset is not addresses[1].offset
    for length, parent, first, second in ((2, 3, 0, 1), (7, 5, 1, 6)):
        runtime = dict(zip(output.args, (length, parent, first, second), strict=True))
        expected = [
            ((parent * length + inner + 3) // length) % 2 * 256 for inner in (first, second)
        ]
        assert [_evaluate(address.offset, runtime) for address in addresses] == expected


def test_two_static_views_share_one_root_without_sharing_their_offsets():
    source, output = Block(), Block()
    allocation = _allocation((16,))
    views = [_view(source, allocation, (8,), (0,), (_index(source, start),)) for start in (0, 8)]
    root = _buffer(output, allocation)
    addresses = [_lower(view, output=output, buffer=root)[0] for view in views]
    assert all(address.source is root.ref for address in addresses)
    assert [_evaluate(address.offset) for address in addresses] == [0, 32]


@pytest.mark.parametrize(
    "widths,matrix,geometry",
    [
        ((2, 16), False, (128, 1, 0)),
        ((2, 16), True, (64, 2, 64)),
        ((2, 8), False, (32, 2, 64)),
        ((2, 8), True, (32, 2, 64)),
    ],
)
def test_linear_matrix_geometry_preserves_real_row_pitch(widths, matrix, geometry):
    source = Block()
    view = _view(source, _allocation((4, 16)), widths, (0, 1))
    address, _, _ = _lower(view, matrix=matrix)
    assert (address.nbytes.data, address.rows.data, address.row_stride.data) == geometry


@pytest.mark.parametrize("layout,bank", [("NZ", "L1"), ("ZZ", "L0A"), ("ZN", "L0B")])
def test_complete_fractal_plane_uses_only_its_leading_slot_coordinate(layout, bank):
    source = Block(arg_types=[IndexType()])
    allocation = _allocation((3, 16, 16), dtype="f16", bank=bank, layout=layout)
    view = _view(
        source,
        allocation,
        (1, 16, 16),
        (1, 2),
        (source.args[0], _index(source, 0), _index(source, 0)),
    )
    output = Block(arg_types=[IndexType()])
    address, _, _ = _lower(
        view, matrix=True, bindings=[(source.args[0], output.args[0])], output=output
    )
    assert (address.nbytes.data, address.rows.data, address.row_stride.data) == (512, 1, 0)
    assert [_evaluate(address.offset, {output.args[0]: slot}) for slot in range(3)] == [
        0,
        512,
        1024,
    ]


def test_nested_subviews_keep_the_original_root_and_axis_metadata():
    source = Block(arg_types=[IndexType()])
    view = _view(
        source,
        _allocation((2, 4, 16)),
        (1, 4, 16),
        (1, 2),
        (source.args[0], _index(source, 0), _index(source, 0)),
    )
    output = Block(arg_types=[IndexType()])
    address, _, root = _lower(view, bindings=[(source.args[0], output.args[0])], output=output)
    first = subview(address.ref, 32, offset=16, rows=2, row_stride=64, append=output.add_op).owner
    second = subview(first.ref, 16, offset=8, rows=2, row_stride=64, append=output.add_op).owner
    first.verify()
    second.verify()
    assert first.source is second.source is root.ref
    assert second.source is not first.ref
    assert tuple(second.origins) == tuple(address.origins)
    assert tuple(second.extents) == tuple(address.extents)
    assert second.widths is address.widths and second.axes is address.axes
    assert first.displacement.data == 16 and second.displacement.data == 24
    assert _evaluate(second.offset, {output.args[0]: 1}) == 256 + 24


def test_direct_buffer_subview_has_no_invented_source_axis_contract():
    block = Block()
    root = _buffer(block, _allocation((32,)))
    address = subview(root.ref, 32, offset=64, append=block.add_op).owner
    address.verify()
    assert address.source is root.ref and _evaluate(address.offset) == 64
    assert not address.origins and not address.extents and address.widths is None


def test_fractal_subview_preserves_a_complete_plane_and_refuses_partial_payload():
    source = Block()
    view = _view(
        source,
        _allocation((16, 16), bank="L1", dtype="f16", layout="NZ"),
        (16, 16),
        (0, 1),
    )
    address, block, root = _lower(view)
    whole = subview(address.ref, 512, append=block.add_op).owner
    whole.verify()
    assert whole.source is root.ref and whole.displacement.data == 0
    with pytest.raises(CompileError, match="fractal subview must retain the complete matrix plane"):
        subview(address.ref, 256, append=block.add_op)


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("prefix", "valid prefix"),
        ("fractal", "complete matrix plane"),
        ("pitch", "row pitch must be static"),
        ("columns", "columns must be contiguous"),
    ],
)
def test_unrepresented_address_geometry_fails_explicitly(case, diagnostic):
    source = Block(arg_types=[IndexType()])
    if case == "prefix":
        view = _view(source, _allocation((64,)), (64,), (0,), valid=source.args[0])
    elif case == "fractal":
        view = _view(
            source, _allocation((16, 16), bank="L1", dtype="f16", layout="NZ"), (8, 16), (0, 1)
        )
    elif case == "pitch":
        allocation = _allocation((4, -1), dimensions=[source.args[0]], bank="GM")
        view = _view(source, allocation, (2, 8), (0, 1))
    else:
        view = _view(source, _allocation((4, 4)), (2, 1), (0,))
    with pytest.raises(CompileError, match=diagnostic) as failure:
        _lower(view)
    assert failure.value.kind is ErrorKind.UNSUPPORTED
    assert failure.value.phase == "native-address"


@pytest.mark.parametrize("origin,width", [(-1, 8), (12, 8), (0, 17)])
def test_obvious_constant_axis_errors_are_rejected(origin, width):
    source = Block()
    view = _view(source, _allocation((16,)), (width,), (0,), (_index(source, origin),))
    with pytest.raises(CompileError, match="static view"):
        _lower(view)


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"offset": -1}, "offset must be nonnegative"),
        ({"offset": 120}, "exceeds its allocation"),
        ({"rows": 2, "row_stride": 16}, "rows must not overlap"),
    ],
)
def test_obvious_static_subview_errors_are_rejected(kwargs, diagnostic):
    block = Block()
    root = _buffer(block, _allocation((32,)))
    with pytest.raises(CompileError, match=diagnostic):
        subview(root.ref, 32, append=block.add_op, **kwargs)


def test_subview_cannot_turn_a_known_parent_row_gap_into_payload():
    source = Block()
    view = _view(source, _allocation((4, 16)), (2, 8), (0, 1))
    address, block, _ = _lower(view)
    with pytest.raises(CompileError, match="parent row gap"):
        subview(address.ref, 16, offset=24, append=block.add_op)


def test_subview_rejects_an_address_chain_in_place_of_the_actual_buffer_root():
    source = Block()
    view = _view(source, _allocation((32,)), (32,), (0,))
    address, block, _ = _lower(view)
    forged = subview(address.ref, 64, append=block.add_op).owner
    forged.operands = (address.ref, *forged.operands[1:])
    with pytest.raises(CompileError, match="original BufferOp allocation root"):
        subview(forged.ref, 32, append=block.add_op)


def test_dynamic_axis_metadata_is_retained_without_claiming_bounds_admission():
    source = Block(arg_types=[IndexType()] * 2)
    allocation = _allocation((-1, 64), dimensions=[source.args[0]], bank="GM")
    view = _view(source, allocation, (1, 64), (1,), (source.args[1], _index(source, 0)))
    output = Block(arg_types=[IndexType()] * 2)
    address, _, _ = _lower(view, bindings=zip(source.args, output.args, strict=True), output=output)
    assert address.extents[0] is output.args[0] and address.origins[0] is output.args[1]
    # Construction does not prove g<N. Later admission must reject this actual
    # invocation; preserving its two distinct SSA inputs makes that possible.
    values = {output.args[0]: 1, output.args[1]: 7}
    assert _evaluate(address.offset, values) == 7 * 256
    assert _evaluate(address.extents[0], values) == 1

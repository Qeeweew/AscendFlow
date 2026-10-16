"""Retained hierarchical source coordinates and independent row-major addresses."""

from xdsl.dialects import arith, scf

from examples.gemm.hierarchical import hierarchical_gemm
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.source import AllocationOp, PipelineOp, ProgramOp, SizeOp, ViewOp


def _value(value, bindings):
    if value in bindings:
        return bindings[value]
    op = value.owner
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    lhs, rhs = (_value(operand, bindings) for operand in op.operands)
    if isinstance(op, arith.AddiOp):
        return lhs + rhs
    if isinstance(op, arith.SubiOp):
        return lhs - rhs
    if isinstance(op, arith.MuliOp):
        return lhs * rhs
    if isinstance(op, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
        assert lhs >= 0 and rhs > 0
        if isinstance(op, arith.RemUIOp):
            return lhs % rhs
        return (lhs + rhs - 1) // rhs if isinstance(op, arith.CeilDivUIOp) else lhs // rhs
    if isinstance(op, arith.MinSIOp):
        return min(lhs, rhs)
    if isinstance(op, arith.MaxSIOp):
        return max(lhs, rhs)
    raise AssertionError(op.name)


def _address(view, bindings):
    # Independently derive row-major byte pitches from the allocation's axes.
    # No production normalization, address, or ownership proof is called.
    allocation = view.source.owner
    assert isinstance(allocation, AllocationOp)
    extents = [dimension.data for dimension in view.source.type.shape]
    for axis, dimension in zip(allocation.dynamic_axes, allocation.dimensions):
        extents[axis.data] = _value(dimension, bindings)
    linear = 0
    for origin, extent in zip(view.origins, extents):
        linear = linear * extent + _value(origin, bindings)
    return linear * {"f16": 2, "bf16": 2, "f32": 4}[view.source.type.dtype.data]


def test_parent_and_child_coordinates_and_axis_extents_are_retained_in_ir():
    module = parse(dump(hierarchical_gemm.to_ir()))
    program = module.body.block.first_op
    assert isinstance(program, ProgramOp)
    parents = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    children = [op for op in module.walk() if isinstance(op, PipelineOp)]
    assert len(parents) == len(children) == 1
    parent, child = parents[0], children[0]
    assert child.parent_op() is parent
    sizes = {op.sym_name.data: op.value for op in module.walk() if isinstance(op, SizeOp)}
    assert parent.ub is sizes["outputs"] and child.limit is sizes["panels"]
    views = [op for op in module.walk() if isinstance(op, ViewOp)]
    external = {
        op.source.owner.sym_name.data: op
        for op in views
        if isinstance(op.source.owner, AllocationOp) and op.source.type.memory.data == "GM"
    }
    coordinates = (program.body.block.args[0], parent.body.block.args[0], child.body.block.args[0])
    for name in ("A", "B"):
        view = external[name]
        assert tuple(view.origins[:3]) == coordinates
        assert tuple(axis.data for axis in view.axes) == (3, 4)
        assert tuple(view.source.owner.dimensions) == (sizes["outputs"], sizes["panels"])
    assert tuple(external["C"].origins[:2]) == coordinates[:2]
    assert tuple(axis.data for axis in external["C"].axes) == (2, 3)
    for outputs, panels in ((1, 1), (2, 7), (17, 1024), (1 << 20, (1 << 20) + 1)):
        for group, output, k in ((0, 0, 0), (19, outputs - 1, panels - 1)):
            bindings = {
                sizes["outputs"]: outputs,
                sizes["panels"]: panels,
                program.body.block.args[0]: group,
                parent.body.block.args[0]: output,
                child.body.block.args[0]: k,
            }
            linear = (group * outputs + output) * panels + k
            assert _address(external["A"], bindings) == linear * 32 * 64 * 2
            assert _address(external["B"], bindings) == linear * 64 * 64 * 2
            assert _address(external["C"], bindings) == (group * outputs + output) * 32 * 64 * 4

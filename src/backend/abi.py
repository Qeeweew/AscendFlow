"""Describe the actual launch ABI using preserved source tensor contracts."""

from xdsl.dialects import arith

from ..diagnostics import require
from ..dialects.source import AllocationOp, SizeOp


def _dimension(value):
    owner = value.owner
    if isinstance(owner, SizeOp):
        return owner.sym_name.data
    if isinstance(owner, arith.ConstantOp):
        return owner.value.value.data
    require(isinstance(owner, arith.MuliOp), "host tensor axes require uniform size products")
    return {"mul": [_dimension(owner.lhs), _dimension(owner.rhs)]}


def launch_abi(source, externals, parameters, workspace):
    program = source.body.block.first_op
    allocations = {
        op.sym_name.data: op
        for op in program.walk()
        if isinstance(op, AllocationOp) and op.access.data != "local"
    }
    tensors = []
    for buffer in externals:
        op = allocations[buffer.sym_name.data]
        dimensions = iter(op.dimensions)
        tensors.append(
            {
                "name": buffer.sym_name.data,
                "dtype": op.ref.type.dtype.data,
                "shape": [
                    n.data if n.data > 0 else _dimension(next(dimensions))
                    for n in op.ref.type.shape
                ],
                "layout": op.ref.type.layout.data,
                "access": op.access.data,
            }
        )
    resources = program.device_resources
    return {
        "version": 1,
        "name": program.sym_name.data,
        "target": "dav-2201",
        "kind": program.kind.data,
        "groups": program.groups.data,
        "disjoint_external": bool(program.disjoint_external.data),
        "device_resources": None
        if resources is None
        else {k: v.data for k, v in resources.data.items()},
        "tensors": tensors,
        "parameters": [
            {"name": op.sym_name.data, "minimum": op.minimum.data, "multiple": op.multiple.data}
            for op in parameters
        ],
        "workspace": bool(workspace.slices),
    }

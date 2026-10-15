"""Select complete dav-2201 Load3D matrix loads with invariant register state.

This pass changes no storage, access range, numerical operation or issue stage.
If a bank needs different FMatrix widths it retains the original Load2D recipe;
it never reconfigures registers across an outstanding asynchronous load.
"""

from xdsl.dialects.builtin import IntAttr, StringAttr

from .ir import KernelOp, Load3DConfigOp, TaskOp
from .model import require


def select_load3d(module):
    result = module.clone()
    for kernel in result.walk():
        if not isinstance(kernel, KernelOp) or kernel.core.data != "aic":
            continue
        require(
            not any(isinstance(op, Load3DConfigOp) for op in kernel.walk()),
            "Load3D selection requires an unselected kernel",
        )
        selected, widths = [], []
        for source, destination in (("load2d_a", "load3d_a"), ("load2d_b_nn", "load3d_b_nn")):
            tasks = [
                op for op in kernel.walk() if isinstance(op, TaskOp) and op.native.data == source
            ]
            values = {op.params.data[0].data for op in tasks}
            width = next(iter(values)) if len(values) == 1 else None
            widths.append(16 if width is None else width)
            if width is not None:
                selected.extend((op, destination) for op in tasks)
        if selected:
            kernel.body.block.insert_op_before(
                Load3DConfigOp.create(
                    attributes={"a_width": IntAttr(widths[0]), "b_width": IntAttr(widths[1])}
                ),
                kernel.body.block.first_op,
            )
            for op, destination in selected:
                op.attributes["native"] = StringAttr(destination)
    return result

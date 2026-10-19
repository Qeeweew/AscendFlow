"""Pure native formatting with actual structured SSA address bindings.

Every pointer comes from the actual AddressOp offset SSA in its recursive
scope. Retained facts supply admitted geometry only; their proof expressions
never execute.
"""

from dataclasses import dataclass

from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr

from ..diagnostics import require
from ..dialects.structured import AddressOp
from ..mask import Program
from ..target.native import NativeSpec
from .native import CTYPE, QUALIFIER, native_line


@dataclass(frozen=True, slots=True)
class PrintTask:
    spec: NativeSpec
    scalar: float
    uid: str
    scalar_source: str | None
    operands: tuple[str, ...]
    mask: object | None
    mask_word: str | None
    mask_repeat: str | None
    mask_iteration: str | None


def format_native(template, *, render, roots, task_names, generation):
    def pointer(index, *, extra=""):
        geometry = template.spec.operands[index]
        # Geometry is shared; pointer identity and offset are final SSA values.
        operation = template.operation.refs[index].owner
        require(isinstance(operation, AddressOp), "final native operand needs an SSA address")
        offset = render(operation.offset)
        if extra:
            owner = operation.offset.owner
            zero = (
                isinstance(owner, arith.ConstantOp)
                and isinstance(owner.value, IntegerAttr)
                and owner.value.value.data == 0
            )
            offset = extra if zero else f"({offset} + {extra})"
        return f"({QUALIFIER[geometry.bank]} {CTYPE[geometry.dtype]}*)({roots[operation.source]} + ({offset}))"

    operation = template.operation
    atoms = tuple(operation.refs[len(template.spec.operands) :])
    scalar_source = (
        atoms[0].owner
        if template.spec.native in {"scale_value", "sub_value", "scalar_store"}
        else None
    )
    program = None
    if "nodes" in operation.attributes:
        program = Program.decode(
            {name: operation.attributes[name] for name in ("shape", "node_shapes", "nodes")}
        )
    task = PrintTask(
        template.spec,
        operation.scalar.value.data,
        task_names[template.operation],
        None if scalar_source is None else task_names[scalar_source],
        tuple(render(value) for value in atoms),
        program,
        None if "mask_word" not in operation.attributes else operation.attributes["mask_word"].data,
        None
        if "mask_repeat" not in operation.attributes
        else operation.attributes["mask_repeat"].data,
        None
        if "mask_iteration" not in operation.attributes
        else operation.attributes["mask_iteration"].data,
    )
    return native_line(task, pointer_renderer=pointer, generation=generation)

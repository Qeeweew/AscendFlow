"""Pure native formatting with actual structured SSA address bindings.

Every pointer comes from the actual AddressOp offset SSA in its recursive
scope. Retained facts supply admitted geometry only; their proof expressions
never execute.
"""

from dataclasses import dataclass

from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr

from ..analysis.structured_addresses import AddressFacts
from ..diagnostics import require
from ..dialects.structured import AddressOp
from ..mask import Program
from ..target.native import NativeGeometry
from .native import CTYPE, QUALIFIER, native_line


@dataclass(frozen=True, slots=True)
class PrintStorage:
    bank: str
    name: str


@dataclass(frozen=True, slots=True)
class PrintOperand:
    address: AddressFacts
    operation: AddressOp
    storage: PrintStorage

    @property
    def geometry(self) -> NativeGeometry:
        return self.address.geometry

    @property
    def dtype(self):
        return self.geometry.dtype

    @property
    def nbytes(self):
        return self.geometry.nbytes

    @property
    def total_nbytes(self):
        return self.geometry.total_nbytes

    @property
    def rows(self):
        return self.geometry.rows

    @property
    def row_stride(self):
        return self.geometry.row_stride

    @property
    def contiguous(self):
        return self.geometry.contiguous


@dataclass(frozen=True, slots=True)
class PrintTask:
    native: str
    refs: tuple[PrintOperand, ...]
    params: tuple[int, ...]
    scalar: float
    uid: str
    scalar_source: str | None
    operands: tuple[str, ...]
    mask: object | None
    mask_word: str | None
    mask_repeat: str | None
    mask_iteration: str | None


def format_native(template, *, render, roots, task_names, generation):
    def pointer(ref, *, extra=""):
        operation = ref.operation
        offset = render(operation.offset)
        if extra:
            owner = operation.offset.owner
            zero = (
                isinstance(owner, arith.ConstantOp)
                and isinstance(owner.value, IntegerAttr)
                and owner.value.value.data == 0
            )
            offset = extra if zero else f"({offset} + {extra})"
        return f"({QUALIFIER[ref.geometry.bank]} {CTYPE[ref.dtype]}*)({roots[operation.source]} + ({offset}))"

    operands = []
    for access in template.accesses:
        # The admitted geometry survives controlled scalar rewrites. Executable
        # pointer identity and offset always come from the final native operand.
        operation = template.operation.refs[access.operand_index].owner
        require(isinstance(operation, AddressOp), "final native operand needs an SSA address")
        operands.append(
            PrintOperand(
                access.address,
                operation,
                PrintStorage(access.address.geometry.bank, roots[operation.source]),
            )
        )
    operation = template.operation
    count = operation.attributes.get("address_count")
    count = len(operation.refs) if count is None else count.data
    atoms = tuple(operation.refs[count:])
    scalar_source = (
        atoms[0].owner
        if operation.native.data in {"scale_value", "sub_value", "scalar_store"}
        else None
    )
    program = None
    if "nodes" in operation.attributes:
        program = Program.decode(
            {name: operation.attributes[name] for name in ("shape", "node_shapes", "nodes")}
        )
    task = PrintTask(
        operation.native.data,
        tuple(operands),
        tuple(value.data for value in operation.params),
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

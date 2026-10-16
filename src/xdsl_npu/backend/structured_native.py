"""Pure native formatting with actual structured SSA address bindings.

The operand adapter carries no legacy Ref, generation terms, or alias proof.
Only the existing native formatter is reused; every pointer comes directly
from the actual AddressOp offset SSA in its recursive scope. Retained facts
supply admitted geometry only; their proof expressions never execute.
"""

from dataclasses import dataclass

from ..analysis.structured_addresses import AddressFacts
from ..diagnostics import require
from ..dialects.structured import AddressOp
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

    @property
    def prefix_extent(self):
        return self.geometry.prefix_extent

    @property
    def tail_nbytes(self):
        return self.geometry.tail_nbytes


@dataclass(frozen=True, slots=True)
class PrintTask:
    native: str
    refs: tuple[PrintOperand, ...]
    params: tuple[int, ...]
    scalar: float
    uid: str
    scalar_source: str | None


def format_native(template, *, render, roots, task_names, generation):
    require(
        all(
            access.address.geometry.prefix_extent is None
            and not access.address.geometry.tail_nbytes
            for access in template.accesses
        ),
        "structured native geometry must carry its complete static footprint",
    )

    def pointer(ref, g="g", *, core="0", extra="", scalar=None):
        operation = ref.operation
        offset = render(operation.offset)
        if extra:
            offset = f"({offset} + {extra})"
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
    scalar_source = (
        operation.refs[-1].owner if operation.native.data in {"scale_value", "sub_value"} else None
    )
    task = PrintTask(
        operation.native.data,
        tuple(operands),
        tuple(value.data for value in operation.params),
        operation.scalar.value.data,
        task_names[template.operation],
        None if scalar_source is None else task_names[scalar_source],
    )
    return native_line(task, "npu_unused_count", pointer_renderer=pointer, generation=generation)

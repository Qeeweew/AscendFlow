"""One packed GM arena; logical roots and their byte effects stay independent."""

from dataclasses import dataclass

from xdsl.dialects import arith

from ..diagnostics import require
from ..dialects.physical import BufferOp, RuntimeSizeOp

ALIGNMENT = 512
MAX_BYTES = (1 << 63) - 1


def _factors(value):
    owner = value.owner
    if isinstance(owner, RuntimeSizeOp):
        return 1, (owner.sym_name.data,)
    if isinstance(owner, arith.ConstantOp):
        return owner.value.value.data, ()
    require(isinstance(owner, arith.MuliOp), "workspace extents require uniform size parameters")
    left, lhs = _factors(owner.lhs)
    right, rhs = _factors(owner.rhs)
    return left * right, (*lhs, *rhs)


@dataclass(frozen=True)
class WorkspaceSlice:
    buffer: BufferOp
    coefficient: int
    dimensions: tuple[str, ...]

    @property
    def size_code(self):
        return " * ".join(
            (f"{self.coefficient}ULL", *(f"uint64_t(npu_size_{n})" for n in self.dimensions))
        )


class WorkspaceLayout:
    def __init__(self, buffers):
        slices = []
        for buffer in buffers:
            if buffer.external.data or buffer.ref.type.bank.data != "GM":
                continue
            coefficient, dimensions = (
                (1, ()) if buffer.extent_count is None else _factors(buffer.extent_count)
            )
            coefficient *= (
                buffer.nbytes.data if buffer.extent_count is None else buffer.count_scale.data
            )
            require(0 <= coefficient <= MAX_BYTES - 511, "workspace byte extent exceeds int64")
            slices.append(WorkspaceSlice(buffer, coefficient, dimensions))
        self.slices = tuple(slices)
        self.static_bytes = (
            None
            if any(s.dimensions for s in slices)
            else sum((s.coefficient + 511) // 512 * 512 for s in slices)
        )
        require(
            self.static_bytes is None or self.static_bytes <= MAX_BYTES,
            "packed workspace byte extent exceeds int64",
        )

    @property
    def metadata(self):
        return {
            "alignment": ALIGNMENT,
            "static_bytes": self.static_bytes,
            "allocations": [
                {
                    "name": s.buffer.sym_name.data,
                    "byte_coefficient": s.coefficient,
                    "dimensions": list(s.dimensions),
                }
                for s in self.slices
            ],
        }

    def bindings(self, roots):
        """Use the same rounded sizes as the checked host query."""
        lines, offset = [], "0"
        for i, item in enumerate(self.slices):
            lines.append(f"  __gm__ uint8_t* {roots[item.buffer.ref]} = npu_workspace + {offset};")
            if i + 1 < len(self.slices):
                next_offset = f"npu_workspace_offset_{i + 1}"
                lines.append(
                    f"  const uint64_t {next_offset} = {offset} + "
                    f"(({item.size_code} + 511ULL) / 512ULL) * 512ULL;"
                )
                offset = next_offset
        return lines

    def query(self, name, parameters):
        """Checked sizing is outside the repeated asynchronous launch path."""
        signature = ["uint64_t* npu_bytes"] + [
            f"int64_t npu_size_{op.sym_name.data}" for op in parameters
        ]
        lines = [
            "",
            f'extern "C" int workspace_size_{name}({", ".join(signature)}) {{',
            "  if (npu_bytes == nullptr) return -1;",
            "  *npu_bytes = 0;",
        ]
        for dimension in dict.fromkeys(n for s in self.slices for n in s.dimensions):
            lines.append(f"  if (npu_size_{dimension} < 0) return -1;")
        lines.append("  uint64_t npu_total = 0;")
        for i, item in enumerate(self.slices):
            size = f"npu_bytes_{i}"
            lines.append(f"  uint64_t {size} = {item.coefficient}ULL;")
            if item.dimensions:
                empty = " || ".join(f"npu_size_{n} == 0" for n in dict.fromkeys(item.dimensions))
                lines.append(f"  if ({empty}) {{ {size} = 0; }} else {{")
                for dimension in item.dimensions:
                    lines.extend(
                        [
                            f"    if ({size} > uint64_t(INT64_MAX) / uint64_t(npu_size_{dimension})) return -1;",
                            f"    {size} *= uint64_t(npu_size_{dimension});",
                        ]
                    )
                lines.append("  }")
            lines.extend(
                [
                    f"  if ({size} > uint64_t(INT64_MAX) - 511ULL) return -1;",
                    f"  {size} = (({size} + 511ULL) / 512ULL) * 512ULL;",
                    f"  if (npu_total > uint64_t(INT64_MAX) - {size}) return -1;",
                    f"  npu_total += {size};",
                ]
            )
        lines.extend(["  *npu_bytes = npu_total;", "  return 0;", "}"])
        return lines

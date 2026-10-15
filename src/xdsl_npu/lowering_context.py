"""One physical allocation/uniform binding environment per source actor.

Region lowering borrows this environment. It never creates a second physical
allocation for the same source identity or joins separately constructed kernels.
"""

from math import prod

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr

from .builder import Kernel
from .ir import FrameOp
from .model import ITEMSIZE, require
from .scalar import ARITHMETIC, expression, from_ssa, literal
from .source_ir import MatmulOp, SizeOp


class ActorBindings:
    def __init__(self, program, family, allocations, *, invocation=None):
        self.program, self.family, self.invocation = program, family, invocation
        self.kernel = Kernel(
            program.sym_name.data,
            "aic" if family == "cube" else "aiv",
            blocks=program.groups.data,
            disjoint_external=bool(program.disjoint_external.data),
        )
        self.uniform = {}
        for op in program.body.block.ops:
            if isinstance(op, SizeOp):
                self.uniform[op.value] = self.kernel.runtime_size(
                    op.sym_name.data, multiple=op.multiple.data
                )
            elif isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
                cloned = op.clone(self.uniform)
                self.kernel.block.add_op(cloned)
                self.uniform.update(zip(op.results, cloned.results))
        self.scalar_cache = {}
        self.uniform_by_expression = {
            from_ssa(source, memo=self.scalar_cache): target
            for source, target in self.uniform.items()
        }
        self.allocation_order = {op: i for i, op in enumerate(allocations)}
        self.externals = {op for op in allocations if op.ref.type.memory.data == "GM"}
        self.roots, self.frame_specs, self.operand_buffers = {}, {}, {}
        self.configured = False
        self.execution_anchor = None

    def mark_region(self, region):
        if self.execution_anchor is None:
            self.execution_anchor = self.kernel.invocation or region

    def _place_declaration(self, op):
        """Late first-use declarations dominate every physical execution region."""
        before = self.execution_anchor
        if before is not None:
            self.kernel.block.detach_op(op)
            self.kernel.block.insert_op_before(op, before)

    def buffer(self, *args, **kwargs):
        value = self.kernel.buffer(*args, **kwargs)
        self._place_declaration(value.owner)
        return value

    def uniform_product(self, factors):
        result = literal(1)
        for factor in factors:
            result = expression("mul", result, factor)
        if result not in self.uniform_by_expression:
            if result.constant is not None:
                op = arith.ConstantOp.from_int_and_width(result.constant, IndexType())
            else:
                require(result.kind == "mul", "unbound uniform frame expression")
                op = arith.MuliOp(*(self.uniform_product((arg,)) for arg in result.args))
            self.kernel.block.add_op(op)
            self._place_declaration(op)
            self.uniform_by_expression[result] = op.results[0]
        return self.uniform_by_expression[result]

    def bind(self, used, records, pipeline):
        program, invocation = self.program, self.invocation
        missing = used - self.roots.keys()
        if not self.configured:
            missing |= self.externals
        require(missing <= self.allocation_order.keys(), "allocation escaped its source actor")
        for allocation in sorted(missing, key=self.allocation_order.__getitem__):
            typ = allocation.ref.type
            dimensions = tuple(n.data for n in typ.shape)
            size = (
                prod(n if n != -1 else 1 for n in dimensions)
                * ITEMSIZE[typ.dtype.data]
                * allocation.versions.data
            )
            dynamic = bool(allocation.dimensions)
            require(
                not dynamic
                or (
                    all(d in self.uniform for d in allocation.dimensions)
                    and (len(allocation.dimensions) == 1 or invocation is not None)
                ),
                "GM extent requires a declared uniform scalar expression",
            )
            if allocation.ring.data:
                size *= program.groups.data
            external = typ.memory.data == "GM"
            if not external:
                size = (size + 31) // 32 * 32
            self.roots[allocation] = self.buffer(
                allocation.sym_name.data,
                typ.memory.data,
                0 if dynamic else size,
                typ.dtype.data,
                layout=typ.layout.data,
                external=external,
                initialized=allocation.access.data in {"read", "read_write"},
                extent_count=self.uniform_product(tuple(from_ssa(d) for d in allocation.dimensions))
                if dynamic
                else None,
                count_scale=size if dynamic else 0,
            )
            if invocation is not None and external:
                require(
                    len(dimensions) >= 3, "invocation GM shape needs group/invocation/frame axes"
                )
                values = dict(zip((d.data for d in allocation.dynamic_axes), allocation.dimensions))
                parent_extent = from_ssa(values[1]) if 1 in values else literal(dimensions[1])
                require(
                    dimensions[0] == program.groups.data
                    and parent_extent == from_ssa(invocation.ub)
                    and all(i in {1, 2} for i in values),
                    "invocation tensors need leading [group, invocation] axes and bounded frame dimensions",
                )
                extent = from_ssa(values[2]) if 2 in values else literal(1)
                require(
                    2 not in values
                    or pipeline.limit is not None
                    and extent == from_ssa(pipeline.limit),
                    "dynamic frame axis must match the child pipeline bound",
                )
                scale = prod(n for n in dimensions[2:] if n != -1) * ITEMSIZE[typ.dtype.data]
                self.frame_specs[allocation] = extent, scale
        for record in records:
            op = record.op
            if not isinstance(op, MatmulOp) or op.a.type.memory.data != "L1":
                continue
            require(self.family == "cube", "matmul requires a Cube execution domain")
            if op not in self.operand_buffers:
                dtype = op.a.type.dtype.data
                m, k = (n.data for n in op.a.type.shape)
                n = op.b.type.shape.data[0 if op.transpose_b.data else 1].data
                name = f"operand_{len(self.operand_buffers)}"
                self.operand_buffers[op] = (
                    self.buffer(name + "_a", "L0A", m * k * ITEMSIZE[dtype], dtype, layout="ZZ"),
                    self.buffer(name + "_b", "L0B", n * k * ITEMSIZE[dtype], dtype, layout="ZN"),
                )
        if not self.configured:
            if self.family == "cube":
                self.kernel.init_fixpipe()
            self.configured = True
        if invocation is not None and self.kernel.invocation is None:
            frame_values = {
                a: self.uniform_product((extent,)) for a, (extent, _) in self.frame_specs.items()
            }
            parent = self.kernel.invocations(self.uniform[invocation.ub])
            for allocation, (_, scale) in self.frame_specs.items():
                root = self.roots[allocation]
                frame = FrameOp.create(
                    operands=[root, parent.body.block.args[0], frame_values[allocation]],
                    attributes={"scale": IntAttr(scale)},
                    result_types=[root.type],
                )
                parent.body.block.add_op(frame)
                self.roots[allocation] = frame.ref

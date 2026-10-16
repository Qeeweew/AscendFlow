"""One physical allocation/uniform binding environment per source actor.

Region lowering borrows this environment. It never creates a second physical
allocation for the same source identity or joins separately constructed kernels.
"""

from math import prod

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntAttr

from ._physical_builder import Kernel
from .diagnostics import require
from .dialects.physical import FrameOp
from .dialects.source import AllocationOp, MatmulOp, PipelineOp, SizeOp
from .scalar import ARITHMETIC, expression, facts_from_sizes, from_ssa, literal, uniform
from .target.native import ITEMSIZE


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
        self.facts = facts_from_sizes(program.body.block.ops)
        needed = set()

        def need(value):
            if value in needed:
                return
            needed.add(value)
            if isinstance(value.owner, arith.ConstantOp) or type(value.owner) in ARITHMETIC:
                for operand in value.owner.operands:
                    need(operand)

        for child in program.walk():
            if isinstance(child, AllocationOp):
                for dimension in child.dimensions:
                    need(dimension)
            elif isinstance(child, PipelineOp) and child.limit is not None:
                need(child.limit)
            elif isinstance(child, scf.ForOp):
                need(child.ub)
        for op in program.body.block.ops:
            if isinstance(op, SizeOp):
                self.uniform[op.value] = self.kernel.runtime_size(
                    op.sym_name.data, minimum=op.minimum.data, multiple=op.multiple.data
                )
            elif isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
                if op.results[0] not in needed:
                    continue
                if not all(value in self.uniform for value in op.operands):
                    # Owner-dependent descriptors remain source SSA until an
                    # admitted iteration domain gives them physical meaning.
                    continue
                if not uniform(from_ssa(op.results[0], facts=self.facts), self.facts):
                    # Signed predicate intermediates (e.g. n-1) are not loop
                    # bounds or extents. Their source SSA remains available to
                    # the predicate lowering; do not emit unused declarations.
                    continue
                cloned = op.clone(self.uniform)
                self.kernel.block.add_op(cloned)
                self.uniform.update(zip(op.results, cloned.results))
        self.scalar_cache = {}
        self.uniform_by_expression = {
            from_ssa(source, memo=self.scalar_cache, facts=self.facts): target
            for source, target in self.uniform.items()
        }
        self.allocation_order = {op: i for i, op in enumerate(allocations)}
        self.externals = {op for op in allocations if op.ref.type.memory.data == "GM"}
        self.roots, self.frame_specs, self.operand_buffers = {}, {}, {}
        # These domains belong to actual source block arguments. A retained
        # serial owner's bound is not a uniform value or a leaf's generation.
        self.owner_domains = {}
        self.scoped_serial_bounds = {}
        self.configured = False
        self.execution_anchor = None

    def runtime_bound(self, pipeline):
        """Admit a cyclic owner domain from actual SSA, without a trusted flag."""
        from .dialects.physical import CyclicCountOp
        from .indexing import equivalent

        if pipeline.limit in self.uniform:
            return self.uniform[pipeline.limit]
        owner = self.program.body.block.args[0]
        bound = from_ssa(pipeline.limit, coordinates={owner: "$core"}, facts=self.facts)
        require(
            self.invocation is None
            and self.program.kind.data != "mixed"
            and bound.kind == "ceildiv"
            and bound.args[1] == literal(self.program.groups.data)
            and bound.args[0].kind == "max"
            and literal(0) in bound.args[0].args,
            "pipeline bound requires a uniform count or a proved cyclic owner domain",
        )
        remaining = next(arg for arg in bound.args[0].args if arg != literal(0))
        from .scalar import Expr

        total = expression("add", remaining, Expr("argument", ("$core",)))
        candidates = [
            (expr, value)
            for expr, value in self.uniform_by_expression.items()
            if uniform(expr, self.facts) and equivalent(expr, total, self.facts)
        ]
        require(candidates, "cyclic count must retain its declared uniform total")
        # Prefer the original total SSA, including its parameter contracts.
        total_expr, total_value = candidates[0]
        count = CyclicCountOp.create(operands=[total_value], result_types=[IndexType()])
        self.kernel.block.add_op(count)
        self._place_declaration(count)
        self.uniform[pipeline.limit] = count.count
        self.cyclic_total = total_expr
        return count.count

    def scoped_serial_bound(self, serial):
        """Bind one actual serial owner without the legacy invocation frames.

        Only a declared uniform bound or the existing cyclic owner formula is
        admitted. Keep owner-dependent counts out of ``uniform``: the query
        clock and the inner KV clock have independent bounds and identities.
        """
        from .cyclic import CyclicDomain
        from .dialects.physical import CyclicCountOp
        from .indexing import equivalent
        from .scalar import Expr

        require(
            isinstance(serial, scf.ForOp)
            and serial.parent_op() is self.program
            and self.invocation is None,
            "scoped serial bound needs its actual root query loop",
        )
        if serial in self.scoped_serial_bounds:
            return self.scoped_serial_bounds[serial]
        if serial.ub in self.uniform:
            value = self.uniform[serial.ub]
        else:
            owner = self.program.body.block.args[0]
            core = Expr("argument", ("$core",))
            bound = from_ssa(serial.ub, coordinates={owner: "$core"}, facts=self.facts)
            require(
                bound.kind == "ceildiv"
                and bound.args[1] == literal(self.program.groups.data)
                and bound.args[0].kind == "max"
                and literal(0) in bound.args[0].args,
                "scoped serial bound requires a uniform count or a proved cyclic owner domain",
            )
            remaining = next(arg for arg in bound.args[0].args if arg != literal(0))
            total = expression("add", remaining, core)
            candidates = [
                (expr, value)
                for expr, value in self.uniform_by_expression.items()
                if uniform(expr, self.facts) and equivalent(expr, total, self.facts)
            ]
            require(candidates, "scoped cyclic count must retain its declared uniform total")
            total_expr, total_value = candidates[0]
            count = CyclicCountOp.create(operands=[total_value], result_types=[IndexType()])
            self.kernel.block.add_op(count)
            self._place_declaration(count)
            value = count.count
            self.owner_domains[serial.body.block.args[0]] = CyclicDomain(
                total_expr, self.program.groups.data
            )
        self.scoped_serial_bounds[serial] = value
        return value

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
                not dynamic or (all(d in self.uniform for d in allocation.dimensions)),
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
                extent_count=self.uniform_product(
                    tuple(from_ssa(d, facts=self.facts) for d in allocation.dimensions)
                )
                if dynamic
                else None,
                count_scale=size if dynamic else 0,
            )
            if invocation is not None and external:
                values = dict(zip((d.data for d in allocation.dynamic_axes), allocation.dimensions))
                parent_extent = (
                    from_ssa(values[1], facts=self.facts)
                    if 1 in values
                    else literal(dimensions[1])
                    if len(dimensions) > 1
                    else None
                )
                if not (
                    len(dimensions) >= 3
                    and dimensions[0] == program.groups.data
                    and parent_extent == from_ssa(invocation.ub, facts=self.facts)
                    and all(i in {1, 2} for i in values)
                ):
                    # Unpartitioned inputs keep their real group coordinates.
                    # Physical extraction checks EVERY native effect is a read;
                    # initialized/access annotations alone do not prove that.
                    continue
                extent = from_ssa(values[2], facts=self.facts) if 2 in values else literal(1)
                require(
                    2 not in values
                    or pipeline.limit is not None
                    and extent == from_ssa(pipeline.limit, facts=self.facts),
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

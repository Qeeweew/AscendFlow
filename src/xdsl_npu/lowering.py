"""Lower stage/domain source regions to exact dav-2201 physical work.

The outer software task clock stays intact. Finite nested pipelines expand
their native operations in semantic order, with independent inner issue keys.
No events, completion certificates, or operator-name cases are supplied here.
"""

import re
from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr
from xdsl.ir import Block, Region

from .builder import Kernel
from .ir import BoundaryOp, EndOp
from .mixed import compose, extract_mixed, global_dependencies
from .model import ITEMSIZE, Instance, dependencies, extract, ranges_cover, require
from .scalar import ARITHMETIC, expression, from_ssa, literal, prefix
from .source_ir import (
    AllocationOp,
    CopyOp,
    DomainOp,
    ElementOp,
    MatmulOp,
    PipelineOp,
    PredicateOp,
    ProgramOp,
    ReduceOp,
    ScalarReadOp,
    ScaleOp,
    SizeOp,
    StageOp,
    ViewOp,
    YieldOp,
)


def shape(value):
    return tuple(n.data for n in value.type.shape)


@dataclass(frozen=True)
class Work:
    op: object
    stage: int
    phase: int
    label: str
    order: tuple[int, ...]
    constants: tuple
    boundary: object = None


def _last_condition(op, pipeline):
    require(
        not op.results
        and not list(op.false_region.block.ops)[:-1]
        and isinstance(op.cond.owner, arith.CmpiOp)
        and op.cond.owner.predicate.value.data == 0,
        "structured scf.if synchronization currently needs a last-generation equality with no else/results",
    )
    coordinate = pipeline.body.block.args[0]
    pair = [from_ssa(value, coordinates={coordinate: "$g"}) for value in op.cond.owner.operands]
    bound = from_ssa(pipeline.limit) if pipeline.limit is not None else literal(pipeline.count.data)
    require(
        set(pair)
        == {
            from_ssa(coordinate, coordinates={coordinate: "$g"}),
            expression("sub", bound, literal(1)),
        },
        "structured scf.if synchronization requires generation == its pipeline bound - 1",
    )


def _work(program, pipeline, family, owner):
    records = []
    phases = {op: i for i, op in enumerate(pipeline.body.block.ops) if isinstance(op, StageOp)}
    default_domain = None if program.kind.data == "mixed" else program.kind.data
    issue_order = () if pipeline.issue_order is None else pipeline.issue_order.data.get(family, ())
    pinned = pipeline.issue_order is not None and family in pipeline.issue_order.data
    ranks = {label.data: i for i, label in enumerate(issue_order)}

    def walk(block, clock, bindings, stage, phase, label, domain, order, boundary=None):
        for position, op in enumerate(block.ops):
            path = (*order, position)
            if isinstance(op, StageOp):
                require(boundary is None, "place last-generation conditions inside their Stage")
                if clock is pipeline:
                    walk(
                        op.body.block,
                        clock,
                        bindings,
                        op.offset.data,
                        phases[op],
                        op.label.data,
                        domain,
                        (ranks.get(op.label.data, -1), *path) if ranks else path,
                    )
                else:
                    # The final component identifies this inner iteration's
                    # issue time. Stage shifts ONLY its own pipeline clock.
                    parent = order[:-1]
                    time = bindings[clock.body.block.args[0]] + op.offset.data
                    walk(
                        op.body.block,
                        clock,
                        bindings,
                        stage,
                        phase,
                        label,
                        domain,
                        (*parent, time, position),
                    )
            elif isinstance(op, DomainOp):
                selected = op.kind.data
                if selected != family:
                    continue
                local = {**bindings, op.body.block.args[0]: owner}
                walk(op.body.block, clock, local, stage, phase, label, selected, path, boundary)
            elif isinstance(op, scf.IfOp):
                require(
                    boundary is None and clock is pipeline,
                    "nested boundary predicates need hierarchical analysis",
                )
                _last_condition(op, pipeline)
                walk(op.true_region.block, clock, bindings, stage, phase, label, domain, path, op)
            elif isinstance(op, PipelineOp):
                require(boundary is None, "a loop inside a boundary requires hierarchical region analysis")
                require(op.issue_order is None, "nested issue policies need hierarchical lowering")
                require(
                    op.limit is None,
                    "nested runtime pipelines need hierarchical loop synchronization; "
                    "only static native tile subdivisions are currently supported",
                )
                require(
                    op.count.data <= 64,
                    "native tile subdivision exceeds 64 templates; "
                    "an input-length loop must remain the outer runtime pipeline",
                )
                for iteration in range(op.count.data):
                    local = {**bindings, op.body.block.args[0]: iteration}
                    walk(op.body.block, op, local, stage, phase, label, domain, (*path, iteration))
            elif (
                isinstance(
                    op,
                    (
                        AllocationOp,
                        ViewOp,
                        PredicateOp,
                        YieldOp,
                        arith.ConstantOp,
                        arith.CmpiOp,
                        scf.YieldOp,
                    ),
                )
                or type(op) in ARITHMETIC
            ):
                continue
            elif isinstance(op, (CopyOp, ElementOp, MatmulOp, ReduceOp, ScalarReadOp, ScaleOp)):
                require(domain is not None, "mixed work requires a Cube or Vector execution scope")
                require(domain == family, "source operation has incompatible execution owner")
                records.append(
                    Work(op, stage, phase, label, path, tuple(bindings.items()), boundary)
                )
            else:
                raise TypeError(
                    f"unsupported source operation {op.name}; effects cannot be ignored"
                )

    walk(pipeline.body.block, pipeline, {}, 0, 0, "pipeline", default_domain, ())
    first_boundary = next(
        (i for i, record in enumerate(records) if record.boundary is not None), len(records)
    )
    require(
        all(r.boundary is not None for r in records[first_boundary:]),
        "last-generation regions must form a terminal logical effect suffix",
    )
    if first_boundary < len(records):
        require(
            program.kind.data != "mixed", "mixed boundary protocols need collective region analysis"
        )
    if pinned:
        require(
            set(ranks) == {record.label for record in records},
            f"{family}: issue_order must name every active Stage exactly once",
        )
        labels = [op.label.data for op in pipeline.body.block.ops if isinstance(op, StageOp)]
        require(len(labels) == len(set(labels)), "issue_order requires unambiguous Stage names")
    return records


def _refs(op):
    if isinstance(op, (CopyOp, ScaleOp)):
        return op.source, op.destination
    if isinstance(op, ScalarReadOp):
        return (op.source,)
    if isinstance(op, ReduceOp):
        return op.source, op.scratch, op.destination
    if isinstance(op, ElementOp):
        return (*op.inputs, op.destination)
    return op.a, op.b, op.accumulator


def _root(value):
    owner = value.owner
    require(isinstance(owner, ViewOp), "source work requires explicit typed views")
    root = owner.source.owner
    require(isinstance(root, AllocationOp), "view must retain one allocation identity")
    return root


def _initialize(op, record, coordinate=None):
    predicate = op.initialize.owner
    if isinstance(predicate, arith.CmpiOp):
        values = [from_ssa(value, coordinates={coordinate: "$g"}) for value in predicate.operands]
        require(
            predicate.predicate.value.data == 0
            and set(values) == {literal(0), from_ssa(coordinate, coordinates={coordinate: "$g"})},
            "runtime matmul initialization requires its own generation == 0",
        )
        require(record.boundary is None, "a terminal MAD cannot initialize an earlier recurrence")
        return "first"
    require(isinstance(predicate, PredicateOp), "matmul initialization needs a typed predicate")
    bindings = dict(record.constants)

    def evaluate(terms, offset):
        value = offset.data
        for term in terms:
            index, coefficient, divisor, modulus = (n.data for n in term)
            variable = predicate.indices[index]
            require(
                variable in bindings,
                "runtime matmul initialization is not implemented; "
                "use an inner static pipeline comparison",
            )
            literal = bindings[variable] // divisor
            value += coefficient * (literal % modulus if modulus else literal)
        return value

    lhs, rhs = (
        evaluate(predicate.lhs_terms, predicate.lhs_offset),
        evaluate(predicate.rhs_terms, predicate.rhs_offset),
    )
    return int({"eq": lhs == rhs, "lt": lhs < rhs, "le": lhs <= rhs}[predicate.comparison.data])


class _Actor:
    def __init__(self, program, pipeline, actor, family, owner, allocations, symbolic=False):
        self.program, self.pipeline = program, pipeline
        self.records = _work(program, pipeline, family, owner)
        require(self.records, f"{actor}: execution domain has no real work")
        self.kernel = Kernel(
            program.sym_name.data,
            "aic" if family == "cube" else "aiv",
            blocks=program.groups.data,
            disjoint_external=bool(program.disjoint_external.data),
        )
        self.runtime = None
        self.uniform = {}
        for scalar_op in program.body.block.ops:
            if isinstance(scalar_op, SizeOp):
                self.uniform[scalar_op.value] = self.kernel.runtime_size(
                    scalar_op.sym_name.data, multiple=scalar_op.multiple.data
                )
            elif isinstance(scalar_op, arith.ConstantOp) or type(scalar_op) in ARITHMETIC:
                cloned = scalar_op.clone(self.uniform)
                self.kernel.block.add_op(cloned)
                self.uniform.update(zip(scalar_op.results, cloned.results))
        if pipeline.limit is not None:
            require(
                pipeline.limit in self.uniform, "pipeline bound must be uniform at region entry"
            )
            self.runtime = self.uniform[pipeline.limit]
        self.scalar_cache = {}
        self.uniform_by_expression = {
            from_ssa(source, memo=self.scalar_cache): target
            for source, target in self.uniform.items()
        }
        self.roots = {}
        used = {_root(ref) for record in self.records for ref in _refs(record.op)}
        for allocation in allocations:
            typ = allocation.ref.type
            if allocation not in used and typ.memory.data != "GM":
                continue
            dimensions = shape(allocation.ref)
            size = (
                prod(n if n != -1 else 1 for n in dimensions)
                * ITEMSIZE[typ.dtype.data]
                * allocation.versions.data
            )
            dynamic = bool(allocation.dimensions)
            require(
                not dynamic
                or (len(allocation.dimensions) == 1 and allocation.dimensions[0] in self.uniform),
                "GM extent requires a declared uniform scalar expression",
            )
            if allocation.ring.data:
                size *= program.groups.data
            external = typ.memory.data == "GM"
            if not external:
                size = (size + 31) // 32 * 32
            self.roots[allocation] = self.kernel.buffer(
                allocation.sym_name.data,
                typ.memory.data,
                0 if dynamic else size,
                typ.dtype.data,
                layout=typ.layout.data,
                external=external,
                initialized=allocation.access.data in {"read", "read_write"},
                extent_count=self.uniform[allocation.dimensions[0]] if dynamic else None,
                count_scale=size if dynamic else 0,
            )
        self.operand_buffers = {}
        for record in self.records:
            op = record.op
            if (
                isinstance(op, MatmulOp)
                and op.a.type.memory.data == "L1"
                and op not in self.operand_buffers
            ):
                require(family == "cube", "matmul requires a Cube execution domain")
                dtype = op.a.type.dtype.data
                m, k = shape(op.a)
                n = shape(op.b)[0 if op.transpose_b.data else 1]
                name = f"operand_{len(self.operand_buffers)}"
                self.operand_buffers[op] = (
                    self.kernel.buffer(
                        name + "_a", "L0A", m * k * ITEMSIZE[dtype], dtype, layout="ZZ"
                    ),
                    self.kernel.buffer(
                        name + "_b", "L0B", n * k * ITEMSIZE[dtype], dtype, layout="ZN"
                    ),
                )
        if family == "cube":
            self.kernel.init_fixpipe()
        self.kernel.loop(
            self.runtime if self.runtime is not None else pipeline.count.data, symbolic=symbolic
        )
        self.schedule = []
        self.definitions = {}
        self.readers = {}
        self.scalar_values = {}

    def view(self, value, record, *, matrix=False):
        view, allocation = value.owner, _root(value)
        typ = value.type
        require(
            (typ.memory, typ.dtype, typ.layout)
            == (allocation.ref.type.memory, allocation.ref.type.dtype, allocation.ref.type.layout),
            "a source view cannot silently change memory, dtype, or physical layout",
        )
        dimensions = shape(value)
        strides = tuple(v.data for v in view.strides)
        require(
            len(dimensions) in {1, 2} and len(strides) == len(dimensions),
            "native tile rank must be one or two",
        )
        require(strides[-1] == ITEMSIZE[typ.dtype.data], "native tile columns must be contiguous")
        bindings = dict(record.constants)
        offset = view.offset.data
        generation, core = [], []
        count_offset, count_core_stride = 0, 0
        for term in view.terms:
            require(len(term) == 4, "source index term must have index/coefficient/divisor/modulus")
            index, coefficient, divisor, modulus = (v.data for v in term)
            require(
                0 <= index < len(view.indices) and divisor > 0 and modulus >= 0,
                "invalid source coordinate",
            )
            variable = view.indices[index]
            if variable in bindings:
                value = bindings[variable] // divisor
                offset += coefficient * (value % modulus if modulus else value)
            elif variable is self.pipeline.body.block.args[0]:
                generation.append((coefficient, divisor, modulus))
            elif variable is self.program.body.block.args[0]:
                core.append((coefficient, divisor, modulus))
            else:
                raise ValueError("coordinate escaped its task or execution-domain scope")
        for term in () if view.dynamic_terms is None else view.dynamic_terms:
            require(
                len(term) == 5,
                "dynamic coordinate needs variable/count/coefficient/divisor/modulus",
            )
            variable_index, count_index, coefficient, divisor, modulus = (v.data for v in term)
            require(
                0 <= variable_index < len(view.indices)
                and 0 <= count_index < len(view.indices)
                and divisor > 0
                and modulus >= 0,
                "invalid dynamic view term",
            )
            variable, count = view.indices[variable_index], view.indices[count_index]
            require(
                count is self.pipeline.limit and self.runtime is not None,
                "address uses a different runtime count",
            )
            if variable is self.program.body.block.args[0]:
                require(
                    divisor == 1 and modulus == 0, "runtime group translation needs an affine owner"
                )
                count_core_stride += coefficient
            elif variable in bindings:
                literal = bindings[variable] // divisor
                count_offset += coefficient * (literal % modulus if modulus else literal)
            else:
                require(False, "runtime task stride needs a segmented ownership contract")
        contiguous = len(dimensions) == 1 or strides[0] == dimensions[1] * strides[-1]
        if typ.layout.data != "linear":
            # Leading axes select complete packed planes. Row-major byte
            # slicing INSIDE a fractal plane is not a valid layout mapping.
            plane = shape(allocation.ref)[-2:]
            plane_bytes = prod(plane) * ITEMSIZE[typ.dtype.data]
            require(
                contiguous
                and len(dimensions) == 2
                and dimensions == plane
                and offset % plane_bytes == 0
                and all(coefficient % plane_bytes == 0 for coefficient, _, _ in generation + core)
                and not (count_offset or count_core_stride),
                "fractal view must select a complete matrix plane; "
                "partial-plane slices need an explicit layout mapping",
            )
        if len(dimensions) == 2 and typ.layout.data == "linear" and (matrix or not contiguous):
            rows, columns = dimensions
            width, pitch = columns * strides[-1], strides[0]
        else:
            rows, width, pitch = 1, prod(dimensions) * ITEMSIZE[typ.dtype.data], 0
        prefix_elements, prefix_extent = 0, None
        if view.valid_index is not None:
            require(
                self.pipeline.limit is not None and rows == 1 and len(dimensions) == 1,
                "valid prefix currently requires a one-dimensional runtime pipeline tile",
            )
            bound = from_ssa(self.pipeline.limit, memo=self.scalar_cache)
            prefix_elements = dimensions[0]
            require(
                bound.kind == "ceildiv" and bound.args[1].constant == prefix_elements,
                "valid prefix tile must match the pipeline ceildiv divisor",
            )
            valid = from_ssa(
                view.indices[view.valid_index.data],
                coordinates={self.pipeline.body.block.args[0]: "$g"},
                memo=self.scalar_cache,
            )
            require(
                valid == prefix(bound.args[0], prefix_elements),
                "valid must be min(tile, logical_extent - task * tile)",
            )
            require(
                not (count_offset or count_core_stride),
                "prefix and count-relative ownership cannot be combined yet",
            )
            require(
                bound.args[0] in self.uniform_by_expression,
                "valid prefix extent must be available at region entry",
            )
            prefix_extent = self.uniform_by_expression[bound.args[0]]
        return self.kernel.view(
            self.roots[allocation],
            width,
            offset=offset,
            rows=rows,
            row_stride=pitch,
            index_terms=generation,
            core_terms=core,
            extent_count=prefix_extent
            if prefix_extent is not None
            else self.runtime
            if count_offset or count_core_stride
            else None,
            count_offset=count_offset,
            count_core_stride=count_core_stride,
            prefix_elements=prefix_elements,
        )

    def emit(self):
        ranks = {
            id(record): rank
            for rank, record in enumerate(sorted(self.records, key=lambda r: r.order))
        }
        stream_block = self.kernel.task_block
        boundaries = {}
        for number, record in enumerate(self.records):
            if record.boundary is not None:
                if record.boundary not in boundaries:
                    block = Block()
                    guard = BoundaryOp.create(
                        operands=[stream_block.args[0]],
                        attributes={"kind": StringAttr("last")},
                        regions=[Region(block)],
                    )
                    stream_block.add_op(guard)
                    boundaries[record.boundary] = block
                self.kernel.task_block = boundaries[record.boundary]
            else:
                self.kernel.task_block = stream_block
            op = record.op
            stem = record.label if re.fullmatch(r"[A-Za-z_]\w*", record.label) else "region"
            uid = f"{stem}_{number}"
            order = ranks[id(record)] * 4

            def task(
                suffix,
                native,
                *refs,
                params=(),
                scalar=0.0,
                order_delta=0,
                uid=uid,
                record=record,
                order=order,
            ):
                operation = self.kernel.task(
                    uid + suffix,
                    native,
                    *refs,
                    params=params,
                    scalar=scalar,
                    stage=record.stage,
                    issue_order=order + order_delta,
                    logical_phase=record.phase,
                )
                operation.attributes["source_region"] = StringAttr(record.label)
                self.schedule.append(
                    DictionaryAttr(
                        {
                            "task": StringAttr(uid + suffix),
                            "region": StringAttr(record.label),
                            "stage": IntAttr(record.stage),
                            "phase": IntAttr(record.phase),
                            "issue_order": IntAttr(order + order_delta),
                        }
                    )
                )
                return operation

            if isinstance(op, CopyOp):
                require(
                    _root(op.destination).access.data != "read", "copy writes a read-only tensor"
                )
                require(
                    (
                        shape(op.source)[::-1]
                        if op.transpose is not None and op.transpose.data
                        else shape(op.source)
                    )
                    == shape(op.destination)
                    and op.source.type.dtype == op.destination.type.dtype,
                    "copy must preserve element type/extent; use an explicit cast",
                )
                banks = op.source.type.memory.data, op.destination.type.memory.data
                if banks == ("GM", "L1"):
                    require(len(shape(op.source)) == 2, "ND→NZ copy requires a matrix")
                    task(
                        "_copy",
                        "nd2nz",
                        self.view(op.source, record, matrix=True),
                        self.view(op.destination, record),
                        params=shape(op.source),
                    )
                elif banks in {("L1", "L0A"), ("L1", "L0B")}:
                    transpose = op.transpose is not None and op.transpose.data
                    task(
                        "_copy",
                        "load2d_a"
                        if banks[1] == "L0A"
                        else "load2d_b_nt"
                        if transpose
                        else "load2d_b_nn",
                        self.view(op.source, record),
                        self.view(op.destination, record),
                        params=shape(op.source),
                    )
                elif banks == ("L0C", "GM"):
                    task(
                        "_export",
                        "fix",
                        self.view(op.source, record),
                        self.view(op.destination, record, matrix=True),
                        params=shape(op.source),
                    )
                else:
                    source, destination = (
                        self.view(op.source, record),
                        self.view(op.destination, record),
                    )
                    dimensions = shape(op.source)
                    padded = len(dimensions) == 2 and any(
                        ref.owner.rows.data > 1 for ref in (source, destination)
                    )
                    if padded:
                        # Preserve row footprints even if the other side is compact.
                        source, destination = (
                            self.view(op.source, record, matrix=True),
                            self.view(op.destination, record, matrix=True),
                        )
                    task("_copy", "copy_rows" if padded else "copy", source, destination)
            elif isinstance(op, ElementOp):
                require(
                    _root(op.destination).access.data != "read",
                    "arithmetic writes a read-only tensor",
                )
                native = {
                    "mul_scalar": "muls",
                    "add_scalar": "adds",
                    "fill": "fill",
                    "mul": "mul",
                    "add": "add",
                    "sub": "sub",
                    "exp": "exp",
                    "sqrt": "sqrt",
                    "rsqrt": "rsqrt",
                    "div": "div",
                }.get(op.kind.data)
                if op.kind.data == "cast":
                    require(len(op.inputs) == 1, "cast takes one source tile")
                    source_dtype, destination_dtype = (
                        op.inputs[0].type.dtype.data,
                        op.destination.type.dtype.data,
                    )
                    require(
                        source_dtype != destination_dtype,
                        "identity cast needs no conversion operation",
                    )
                    native = "convert_f32" if destination_dtype == "f32" else "convert_rn"
                require(native is not None, f"unsupported typed element operation {op.kind.data}")
                refs = [self.view(value, record) for value in (*op.inputs, op.destination)]
                task(
                    "_" + op.kind.data,
                    native,
                    *refs,
                    scalar=0.0 if op.scalar is None else op.scalar.value.data,
                )
            elif isinstance(op, ReduceOp):
                require(
                    _root(op.scratch).access.data != "read"
                    and _root(op.destination).access.data != "read",
                    "reduction writes a read-only buffer",
                )
                rows, cols = (
                    (1, shape(op.source)[0]) if len(shape(op.source)) == 1 else shape(op.source)
                )
                source, temporary, destination = (
                    self.view(value, record, matrix=True)
                    for value in (op.source, op.scratch, op.destination)
                )
                task(
                    "_partial",
                    "reduce_" + op.kind.data + "64",
                    source,
                    temporary,
                    params=(rows, cols),
                )
                task(
                    "_merge",
                    "merge_" + op.kind.data,
                    temporary,
                    destination,
                    params=(rows, cols // 64),
                    order_delta=1,
                )
            elif isinstance(op, ScalarReadOp):
                captured = task("_capture", "scalar_load", self.view(op.source, record))
                self.scalar_values[op.value] = captured.value
            elif isinstance(op, ScaleOp):
                require(
                    op.factor in self.scalar_values and op.factor.owner.parent is op.parent,
                    "captured scalar must be read earlier in the same source scope",
                )
                require(
                    _root(op.destination).access.data != "read", "scale writes a read-only buffer"
                )
                task(
                    "_scale",
                    "scale_value",
                    self.view(op.source, record),
                    self.view(op.destination, record),
                    self.scalar_values[op.factor],
                )
            elif isinstance(op, MatmulOp):
                initialization = _initialize(op, record, self.pipeline.body.block.args[0])
                m, k = shape(op.a)
                n, bk = shape(op.b) if op.transpose_b.data else (shape(op.b)[1], shape(op.b)[0])
                require(k == bk and shape(op.accumulator) == (m, n), "gemm shapes disagree")
                if op.a.type.memory.data == "L1":
                    a_buffer, b_buffer = self.operand_buffers[op]
                    dtype = op.a.type.dtype.data
                    a = self.kernel.view(a_buffer, m * k * ITEMSIZE[dtype])
                    b = self.kernel.view(b_buffer, n * k * ITEMSIZE[dtype])
                    task("_operand_a", "load2d_a", self.view(op.a, record), a, params=(m, k))
                    task(
                        "_operand_b",
                        "load2d_b_nt" if op.transpose_b.data else "load2d_b_nn",
                        self.view(op.b, record),
                        b,
                        params=shape(op.b),
                        order_delta=1,
                    )
                else:
                    a, b = self.view(op.a, record), self.view(op.b, record)
                task(
                    "_mad",
                    "mad_first" if initialization == "first" else "mad",
                    a,
                    b,
                    self.view(op.accumulator, record),
                    params=(m, n, k) if initialization == "first" else (m, n, k, initialization),
                    order_delta=2 if op.a.type.memory.data == "L1" else 0,
                )
            if isinstance(op, CopyOp):
                suffix = "_export" if op.source.type.memory.data == "L0C" else "_copy"
                self.definitions[number] = uid + suffix
                self.readers[number, 0] = uid + suffix
            elif isinstance(op, ElementOp):
                self.definitions[number] = uid + "_" + op.kind.data
                for i in range(len(op.inputs)):
                    self.readers[number, i] = self.definitions[number]
            elif isinstance(op, ReduceOp):
                self.definitions[number] = uid + "_merge"
                self.readers[number, 0] = uid + "_partial"
            elif isinstance(op, ScalarReadOp):
                self.definitions[number] = self.readers[number, 0] = uid + "_capture"
            elif isinstance(op, ScaleOp):
                self.definitions[number] = self.readers[number, 0] = uid + "_scale"
            else:
                self.definitions[number] = uid + "_mad"
                explicit = op.a.type.memory.data == "L0A"
                self.readers[number, 0] = uid + ("_mad" if explicit else "_operand_a")
                self.readers[number, 1] = uid + ("_mad" if explicit else "_operand_b")
                if initialization != 1:
                    self.readers[number, 2] = uid + "_mad"
        for block in boundaries.values():
            block.add_op(EndOp())
        self.kernel.task_block = stream_block
        module = self.kernel.finish()
        module.body.block.first_op.attributes["source_schedule"] = ArrayAttr(self.schedule)
        return module


def _tile_producer(value, outer):
    final_bindings = {}
    while isinstance(value.owner, (StageOp, DomainOp, PipelineOp)):
        region = value.owner
        if isinstance(region, PipelineOp) and region is not outer:
            require(region.count.data > 0, "a zero-trip inner pipeline has no tile result")
            final_bindings[region.body.block.args[0]] = region.count.data - 1
        value = region.body.block.last_op.values[value.index]
    require(
        isinstance(value.owner, (CopyOp, ElementOp, MatmulOp, ReduceOp, ScaleOp)),
        "unknown logical tile producer",
    )
    return value.owner, final_bindings


def _check_values(physical, actors, outer):
    """Check borrowed value identity independently of synchronization.

    Actual native RAW writers must be the declared numerical producer.
    Reading a stale tile is rejected; there is no hidden snapshot copy.
    """
    from .loop_model import is_symbolic

    if is_symbolic(physical):
        return _check_loop_values(physical, actors, outer)
    if len(actors) == 3:
        mixed = extract_mixed(physical)
        programs = mixed.programs
        raw = [
            (src, dst, o.storage, o.start, o.stop)
            for src, dst, o in global_dependencies(mixed)
            if o.hazard == "RAW"
        ]
    else:
        actor = next(iter(actors))
        program = extract(physical)
        programs = {actor: program}
        raw = [
            ((actor, o.source), (actor, o.destination), o.storage, o.start, o.stop)
            for o in dependencies(program)
            if o.hazard == "RAW"
        ]
    writers = {}
    for source, destination, root, lo, hi in raw:
        writers.setdefault((destination, root), []).append((source, lo, hi))
    indices = {
        actor: {task.uid: i for i, task in enumerate(p.tasks)} for actor, p in programs.items()
    }
    for actor, lowering in actors.items():
        for number, record in enumerate(lowering.records):
            refs = _refs(record.op)
            for value, input_index in zip(record.op.dependencies, record.op.dependency_inputs):
                producer, final = _tile_producer(value, outer)
                root = _root(_refs(producer)[-1])
                private = root.ref.type.memory.data != "GM"
                consumer_bindings = {**dict(record.constants), **final}
                candidates = []
                for source_actor, source_lowering in actors.items():
                    if private and source_actor != actor:
                        continue
                    for source_number, source_record in enumerate(source_lowering.records):
                        if source_record.op is producer and all(
                            consumer_bindings.get(var, literal) == literal
                            for var, literal in source_record.constants
                        ):
                            candidates.append(
                                (source_actor, source_lowering.definitions[source_number])
                            )
                require(
                    len(candidates) == 1,
                    "logical tile producer has ambiguous task/domain ownership",
                )
                source_actor, source_uid = candidates[0]
                root_name = root.sym_name.data
                if private and len(actors) == 3:
                    root_name = f"{actor}/{root_name}"
                read_index = input_index.data
                require(
                    (number, read_index) in lowering.readers and _root(refs[read_index]) is root,
                    "logical tile result does not match a consumed storage view",
                )
                read_uid = lowering.readers[number, read_index]
                task_index = indices[actor][read_uid]
                read_task = programs[actor].tasks[task_index]
                native_index = (
                    0
                    if isinstance(record.op, MatmulOp)
                    and record.op.a.type.memory.data == "L1"
                    and read_index < 2
                    else read_index
                )
                ref = read_task.refs[native_index]
                for generation in range(programs[actor].count):
                    expected = (
                        source_actor,
                        Instance(generation, indices[source_actor][source_uid]),
                    )
                    consumer = (actor, Instance(generation, task_index))
                    produced = [
                        (lo, hi)
                        for writer, lo, hi in writers.get((consumer, root_name), ())
                        if writer == expected
                    ]
                    require(
                        ranges_cover(produced, ref.segments(generation)),
                        f"logical tile from {source_uid} was overwritten before {read_uid}; "
                        "use the current buffer or a distinct declared destination",
                    )


def _check_loop_values(physical, actors, outer):
    from .loop_model import extract_loop, loop_dependencies

    loop = extract_loop(physical)
    raw = [o for o in loop_dependencies(loop) if o.hazard == "RAW"]
    for actor, lowering in actors.items():
        owner = actor if len(actors) == 3 else ("aic" if actor == "cube" else "aiv0")
        tasks = {t.uid: t for t in loop.actors[owner].program.tasks}
        for number, record in enumerate(lowering.records):
            for value, input_index in zip(record.op.dependencies, record.op.dependency_inputs):
                producer, final = _tile_producer(value, outer)
                root = _root(_refs(producer)[-1])
                private = root.ref.type.memory.data != "GM"
                bindings = {**dict(record.constants), **final}
                candidates = []
                for source_actor, source_lowering in actors.items():
                    if private and source_actor != actor:
                        continue
                    for source_number, source_record in enumerate(source_lowering.records):
                        if source_record.op is producer and all(
                            bindings.get(var, literal) == literal
                            for var, literal in source_record.constants
                        ):
                            source_owner = (
                                source_actor
                                if len(actors) == 3
                                else ("aic" if source_actor == "cube" else "aiv0")
                            )
                            candidates.append(
                                (source_owner, source_lowering.definitions[source_number])
                            )
                require(len(candidates) == 1, "logical tile producer has ambiguous ownership")
                index = input_index.data
                read_uid = lowering.readers[number, index]
                task = tasks[read_uid]
                native_index = (
                    0
                    if isinstance(record.op, MatmulOp)
                    and record.op.a.type.memory.data == "L1"
                    and index < 2
                    else index
                )
                ref = task.refs[native_index]
                root_name = loop.root(owner, ref)
                for r in range(loop.storage_periods[root_name]):
                    produced = [
                        (o.start, o.stop)
                        for o in raw
                        if o.source == candidates[0]
                        and o.destination == (owner, read_uid)
                        and o.distance == 0
                        and r % o.period in o.residues
                        and o.storage == root_name
                    ]
                    frame = loop.frames.get(root_name, 0)
                    needed = [(lo - r * frame, hi - r * frame) for lo, hi in ref.segments(r)]
                    require(
                        ranges_cover(produced, needed),
                        f"logical tile from {candidates[0][1]} was overwritten before {read_uid}",
                    )


def lower_source(module, *, symbolic=True):
    from .frontend import PrimFunction

    if isinstance(module, PrimFunction):
        module = module.to_ir()
    module.verify()
    program = module.body.block.first_op
    require(
        isinstance(program, ProgramOp) and len(list(module.body.block.ops)) == 1,
        "expected one typed source program",
    )
    if program.attributes.get("frontend") == StringAttr("ast"):
        for operation in program.walk():
            if isinstance(operation, MatmulOp):
                require(
                    operation.a.type.memory.data == "L0A",
                    "AST matmul requires explicit L0 operands; implicit L1 recipes are legacy",
                )
            require(
                not isinstance(operation, scf.ForOp),
                f"{operation.location}: structured {operation.name} synchronization is not "
                "implemented; AST control flow is never silently expanded or discarded",
            )
            if isinstance(operation, PipelineOp):
                require(
                    operation.parent_op() is program,
                    "nested AST pipelines require hierarchical synchronization; "
                    "constant bounds do not authorize implicit unrolling",
                )
    pipelines = [op for op in program.body.block.ops if isinstance(op, PipelineOp)]
    require(len(pipelines) == 1, "source requires one continuous outer task stream")
    require(
        all(
            isinstance(op, (AllocationOp, SizeOp, PipelineOp, YieldOp, arith.ConstantOp))
            or type(op) in ARITHMETIC
            for op in program.body.block.ops
        ),
        "executable source work must belong to a task pipeline",
    )
    allocations = [op for op in program.walk() if isinstance(op, AllocationOp)]
    pipeline = pipelines[0]
    require(
        symbolic or not any(isinstance(op, scf.IfOp) for op in pipeline.walk()),
        "boundary regions require the symbolic compiler; no finite fallback",
    )
    if program.kind.data == "mixed":
        actors = {
            actor: _Actor(program, pipeline, actor, family, owner, allocations, symbolic)
            for actor, family, owner in [
                ("aic", "cube", 0),
                ("aiv0", "vector", 0),
                ("aiv1", "vector", 1),
            ]
        }
        modules = [actor.emit() for actor in actors.values()]
        physical = compose(
            program.sym_name.data, *modules, disjoint_external=bool(program.disjoint_external.data)
        )
    else:
        actors = {
            program.kind.data: _Actor(
                program, pipeline, program.kind.data, program.kind.data, 0, allocations, symbolic
            )
        }
        physical = next(iter(actors.values())).emit()
    _check_values(physical, actors, pipeline)
    return physical

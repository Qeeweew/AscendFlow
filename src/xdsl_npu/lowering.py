"""Lower stage/domain source regions to exact dav-2201 physical work.

The outer software task clock stays intact. Finite nested pipelines expand
their native operations in semantic order, with independent inner issue keys.
No events, completion certificates, or operator-name cases are supplied here.
"""

import re
from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr

from .builder import Kernel
from .mixed import compose, extract_mixed, global_dependencies
from .model import ITEMSIZE, Instance, dependencies, extract, ranges_cover, require
from .source_ir import (
    AllocationOp,
    CopyOp,
    DomainOp,
    ElementOp,
    MatmulOp,
    PipelineOp,
    PredicateOp,
    ProgramOp,
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


def _work(program, pipeline, family, owner):
    records = []
    phases = {op: i for i, op in enumerate(pipeline.body.block.ops) if isinstance(op, StageOp)}
    default_domain = None if program.kind.data == "mixed" else program.kind.data

    def walk(block, clock, bindings, stage, phase, label, domain, order):
        for position, op in enumerate(block.ops):
            path = (*order, position)
            if isinstance(op, StageOp):
                if clock is pipeline:
                    walk(
                        op.body.block,
                        clock,
                        bindings,
                        op.offset.data,
                        phases[op],
                        op.label.data,
                        domain,
                        path,
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
                walk(op.body.block, clock, local, stage, phase, label, selected, path)
            elif isinstance(op, PipelineOp):
                for iteration in range(op.count.data):
                    local = {**bindings, op.body.block.args[0]: iteration}
                    walk(op.body.block, op, local, stage, phase, label, domain, (*path, iteration))
            elif isinstance(op, (AllocationOp, ViewOp, PredicateOp, YieldOp)):
                continue
            elif isinstance(op, (CopyOp, ElementOp, MatmulOp)):
                require(domain is not None, "mixed work requires a Cube or Vector execution scope")
                require(domain == family, "source operation has incompatible execution owner")
                records.append(Work(op, stage, phase, label, path, tuple(bindings.items())))
            else:
                raise TypeError(
                    f"unsupported source operation {op.name}; effects cannot be ignored"
                )

    walk(pipeline.body.block, pipeline, {}, 0, 0, "pipeline", default_domain, ())
    return records


def _refs(op):
    if isinstance(op, CopyOp):
        return op.source, op.destination
    if isinstance(op, ElementOp):
        return (*op.inputs, op.destination)
    return op.a, op.b, op.accumulator


def _root(value):
    owner = value.owner
    require(isinstance(owner, ViewOp), "source work requires explicit typed views")
    root = owner.source.owner
    require(isinstance(root, AllocationOp), "view must retain one allocation identity")
    return root


def _initialize(op, record):
    predicate = op.initialize.owner
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
    def __init__(self, program, pipeline, actor, family, owner, allocations):
        self.program, self.pipeline = program, pipeline
        self.records = _work(program, pipeline, family, owner)
        require(self.records, f"{actor}: execution domain has no real work")
        self.kernel = Kernel(
            program.sym_name.data,
            "aic" if family == "cube" else "aiv",
            blocks=program.groups.data,
            disjoint_external=bool(program.disjoint_external.data),
        )
        self.roots = {}
        used = {_root(ref) for record in self.records for ref in _refs(record.op)}
        for allocation in allocations:
            typ = allocation.ref.type
            if allocation not in used and typ.memory.data != "GM":
                continue
            size = prod(shape(allocation.ref)) * ITEMSIZE[typ.dtype.data] * allocation.versions.data
            if allocation.ring.data:
                size *= program.groups.data
            external = typ.memory.data == "GM"
            if not external:
                size = (size + 31) // 32 * 32
            self.roots[allocation] = self.kernel.buffer(
                allocation.sym_name.data,
                typ.memory.data,
                size,
                typ.dtype.data,
                layout=typ.layout.data,
                external=external,
                initialized=allocation.access.data in {"read", "read_write"},
            )
        self.operand_buffers = {}
        for record in self.records:
            op = record.op
            if isinstance(op, MatmulOp) and op not in self.operand_buffers:
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
        self.kernel.loop(pipeline.count.data)
        self.schedule = []
        self.definitions = {}
        self.readers = {}

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
        contiguous = len(dimensions) == 1 or strides[0] == dimensions[1] * strides[-1]
        if typ.layout.data != "linear":
            require(
                contiguous and dimensions == shape(allocation.ref),
                "fractal subviews need an explicit layout conversion contract",
            )
        if len(dimensions) == 2 and typ.layout.data == "linear" and (matrix or not contiguous):
            rows, columns = dimensions
            width, pitch = columns * strides[-1], strides[0]
        else:
            rows, width, pitch = 1, prod(dimensions) * ITEMSIZE[typ.dtype.data], 0
        return self.kernel.view(
            self.roots[allocation],
            width,
            offset=offset,
            rows=rows,
            row_stride=pitch,
            index_terms=generation,
            core_terms=core,
        )

    def emit(self):
        ranks = {
            id(record): rank
            for rank, record in enumerate(sorted(self.records, key=lambda r: r.order))
        }
        for number, record in enumerate(self.records):
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

            if isinstance(op, CopyOp):
                require(
                    _root(op.destination).access.data != "read", "copy writes a read-only tensor"
                )
                require(
                    shape(op.source) == shape(op.destination)
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
            elif isinstance(op, MatmulOp):
                initialization = _initialize(op, record)
                require(
                    op.a.type.memory.data == op.b.type.memory.data == "L1"
                    and op.accumulator.type.memory.data == "L0C",
                    "gemm requires declared L1 inputs and L0C accumulator",
                )
                m, k = shape(op.a)
                n, bk = shape(op.b) if op.transpose_b.data else (shape(op.b)[1], shape(op.b)[0])
                require(k == bk and shape(op.accumulator) == (m, n), "gemm shapes disagree")
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
                task(
                    "_mad",
                    "mad",
                    a,
                    b,
                    self.view(op.accumulator, record),
                    params=(m, n, k, initialization),
                    order_delta=2,
                )
            if isinstance(op, CopyOp):
                suffix = "_export" if op.source.type.memory.data == "L0C" else "_copy"
                self.definitions[number] = uid + suffix
                self.readers[number, 0] = uid + suffix
            elif isinstance(op, ElementOp):
                self.definitions[number] = uid + "_" + op.kind.data
                for i in range(len(op.inputs)):
                    self.readers[number, i] = self.definitions[number]
            else:
                self.definitions[number] = uid + "_mad"
                self.readers[number, 0] = uid + "_operand_a"
                self.readers[number, 1] = uid + "_operand_b"
                if not initialization:
                    self.readers[number, 2] = uid + "_mad"
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
    require(isinstance(value.owner, (CopyOp, ElementOp, MatmulOp)), "unknown logical tile producer")
    return value.owner, final_bindings


def _check_values(physical, actors, outer):
    """Check borrowed value identity independently of synchronization.

    Actual native RAW writers must be the declared numerical producer.
    Reading a stale tile is rejected; there is no hidden snapshot copy.
    """
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
                    0 if isinstance(record.op, MatmulOp) and read_index < 2 else read_index
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


def lower_source(module):
    module.verify()
    program = module.body.block.first_op
    require(
        isinstance(program, ProgramOp) and len(list(module.body.block.ops)) == 1,
        "expected one typed source program",
    )
    pipelines = [op for op in program.body.block.ops if isinstance(op, PipelineOp)]
    require(len(pipelines) == 1, "source requires one continuous outer task stream")
    require(
        all(isinstance(op, (AllocationOp, PipelineOp, YieldOp)) for op in program.body.block.ops),
        "executable source work must belong to a task pipeline",
    )
    allocations = [op for op in program.walk() if isinstance(op, AllocationOp)]
    pipeline = pipelines[0]
    if program.kind.data == "mixed":
        actors = {
            actor: _Actor(program, pipeline, actor, family, owner, allocations)
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
                program, pipeline, program.kind.data, program.kind.data, 0, allocations
            )
        }
        physical = next(iter(actors.values())).emit()
    _check_values(physical, actors, pipeline)
    return physical

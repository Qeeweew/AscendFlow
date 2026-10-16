"""Lower stage/domain source regions to exact dav-2201 physical work.

The AST path consumes collected lexical scopes without expanding child clocks.
The explicit non-AST fixture path retains finite subdivisions. No events,
completion certificates, or operator-name cases are supplied here.
"""

import re
from collections import ChainMap
from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import DictionaryAttr, IntAttr, StringAttr
from xdsl.ir import Block, Region

from .ir import BoundaryOp, EndOp, ScopeGuardOp
from .lowering_context import ActorBindings
from .mixed import extract_mixed, global_dependencies
from .model import ITEMSIZE, Instance, dependencies, extract, ranges_cover, require
from .scalar import ARITHMETIC, Expr, expression, from_ssa, literal, prefix
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
    StageOp,
    ViewOp,
    YieldOp,
    block8_reduction_widths,
)
from .source_scopes import Context


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
    context: Context | None = None
    parent_guard: str | None = None


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


def _check_work_order(records, program, pipeline, family, pinned, ranks):
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


def _work(
    program,
    pipeline,
    family,
    owner,
    *,
    operations=None,
    scoped=None,
    scope=None,
    nested=False,
    parent_guards=None,
):
    if program.attributes.get("frontend") != StringAttr("ast"):
        return _legacy_work(program, pipeline, family, owner, operations=operations)
    require(
        scoped is not None and scope is not None,
        "AST native records need their collected source scope",
    )
    require(
        operations is not None or scope.op is pipeline,
        "AST native records changed their source pipeline",
    )
    phases = {op: i for i, op in enumerate(pipeline.body.block.ops) if isinstance(op, StageOp)}
    issue_order = () if pipeline.issue_order is None else pipeline.issue_order.data.get(family, ())
    ranks = {label.data: i for i, label in enumerate(issue_order)}
    pinned = (
        operations is None
        and pipeline.issue_order is not None
        and family in pipeline.issue_order.data
    )
    records = []
    for item in scoped:
        if item.actor != family:
            continue
        clock = item.clock
        if operations is None:
            require(
                clock is not None and clock.op is pipeline,
                "nested AST pipelines require hierarchical synchronization; "
                "constant bounds do not authorize implicit unrolling",
            )
            stage = clock.stage
        else:
            require(
                clock is None or isinstance(clock.op, scf.ForOp) or nested and clock.op is pipeline,
                "ordinary source work cannot borrow a child clock",
            )
            stage = None
        guards = item.context.guards
        require(len(guards) <= 1, "nested boundary predicates need hierarchical analysis")
        require(
            not guards or guards[0].taken,
            "structured scf.if synchronization currently needs a last-generation equality with no else/results",
        )
        parent_guard = (
            None if not guards or parent_guards is None else parent_guards.get(guards[0].op)
        )
        boundary = guards[0].op if guards and parent_guard is None else None
        if boundary is not None:
            _last_condition(boundary, pipeline)
        label = "pipeline" if stage is None else stage.label.data
        entry = scope.context.path
        require(
            item.context.path[: len(entry)] == entry,
            "AST native work escaped its collected source scope",
        )
        order = item.context.path[len(entry) :]
        if stage is not None and ranks:
            order = (ranks.get(label, -1), *order)
        constants = () if item.context.owner is None else ((item.context.owner, owner),)
        records.append(
            Work(
                item.op,
                0 if stage is None else stage.offset.data,
                0 if stage is None else phases[stage],
                label,
                order,
                constants,
                boundary,
                item.context,
                parent_guard,
            )
        )
    return _check_work_order(records, program, pipeline, family, pinned, ranks)


def _legacy_work(program, pipeline, family, owner, *, operations=None):
    records = []
    phases = {op: i for i, op in enumerate(pipeline.body.block.ops) if isinstance(op, StageOp)}
    default_domain = None if program.kind.data == "mixed" else program.kind.data
    issue_order = () if pipeline.issue_order is None else pipeline.issue_order.data.get(family, ())
    pinned = (
        operations is None
        and pipeline.issue_order is not None
        and family in pipeline.issue_order.data
    )
    ranks = {label.data: i for i, label in enumerate(issue_order)}

    def walk(block, clock, bindings, stage, phase, label, domain, order, boundary=None):
        for position, op in enumerate(block if isinstance(block, tuple) else block.ops):
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
                require(
                    boundary is None,
                    "a loop inside a boundary requires hierarchical region analysis",
                )
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

    walk(
        pipeline.body.block if operations is None else tuple(operations),
        pipeline,
        {},
        0,
        0,
        "pipeline",
        default_domain,
        (),
    )
    return _check_work_order(records, program, pipeline, family, pinned, ranks)


def _refs(op):
    if isinstance(op, (CopyOp, ScaleOp)):
        return op.source, op.destination
    if isinstance(op, ScalarReadOp):
        return (op.source,)
    if isinstance(op, ReduceOp):
        return op.source, op.scratch, op.destination
    if isinstance(op, ElementOp):
        return (*op.inputs, *((op.scratch,) if op.scratch is not None else ()), op.destination)
    return op.a, op.b, op.accumulator


def _native_input_index(op, index):
    if isinstance(op, MatmulOp) and op.a.type.memory.data == "L1" and index < 2:
        return 0  # Each input is read by a separate Load2D.
    if isinstance(op, ElementOp) and op.kind.data.endswith("_rows") and index == 1:
        return 0  # The row statistic's last reader is BRCB, not the arithmetic.
    return index


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
    def __init__(
        self,
        program,
        pipeline,
        actor,
        family,
        owner,
        allocations,
        symbolic=False,
        invocation=None,
        records=None,
        once=False,
        region_name=None,
        bindings=None,
        ambient_frames=(),
        capture=None,
        parent_generation=None,
    ):
        self.program, self.pipeline = program, pipeline
        self.invocation = invocation
        self.ambient_frames, self.capture = tuple(ambient_frames), capture
        self.parent_generation = parent_generation
        self.records = _work(program, pipeline, family, owner) if records is None else records
        self.once, self.region_name = once, region_name
        require(
            all(
                record.parent_guard is None
                or record.parent_guard in {"first", "last"}
                and parent_generation is not None
                and record.boundary is None
                for record in self.records
            ),
            "native parent guard needs its actual parent generation and a separate child clock",
        )
        require(
            not once
            or all(record.boundary is None and record.stage == 0 for record in self.records),
            "ordinary source work cannot borrow a child stage or boundary predicate",
        )
        require(self.records, f"{actor}: execution domain has no real work")
        self.bindings = bindings or ActorBindings(
            program, family, allocations, invocation=invocation
        )
        require(
            self.bindings.program is program
            and self.bindings.family == family
            and self.bindings.invocation is invocation,
            "region binding scope changed",
        )
        used = {_root(ref) for record in self.records for ref in _refs(record.op)}
        self.bindings.bind(used, self.records, pipeline)
        self.kernel = self.bindings.kernel
        self.uniform = self.bindings.uniform
        self.uniform_by_expression = self.bindings.uniform_by_expression
        self.roots, self.frame_specs = self.bindings.roots, self.bindings.frame_specs
        self.operand_buffers = self.bindings.operand_buffers
        # A coordinate-dependent expression may only be cached within this region.
        self.scalar_cache = ChainMap({}, self.bindings.scalar_cache)
        self.runtime = None
        if pipeline.limit is not None:
            self.runtime = self.bindings.runtime_bound(pipeline)
        if self.kernel.stream is not None:
            require(region_name is not None, "another region needs a lexical identity")
            self.kernel.region(
                region_name,
                None
                if once
                else (self.runtime if self.runtime is not None else pipeline.count.data),
            )
        elif once:
            self.kernel.once(region_name)
        else:
            self.kernel.loop(
                self.runtime if self.runtime is not None else pipeline.count.data, symbolic=symbolic
            )
            if region_name is not None:
                self.kernel.stream.attributes["uid"] = StringAttr(region_name)
        self.region = self.kernel.stream
        self.bindings.mark_region(self.region)
        self.schedule = []
        self.definitions = {}
        self.readers = {}
        self.scalar_values = {}

    def view(self, value, record, *, matrix=False):
        view, allocation = value.owner, _root(value)
        from .indexing import (
            arguments,
            check_axis,
            constant_ratio,
            equivalent,
            normalize,
            physical_terms,
        )

        facts = self.bindings.facts
        typ = value.type
        require(
            (typ.memory, typ.dtype, typ.layout)
            == (allocation.ref.type.memory, allocation.ref.type.dtype, allocation.ref.type.layout),
            "a source view cannot silently change memory, dtype, or physical layout",
        )
        dimensions = shape(value)
        require(len(dimensions) in {1, 2}, "native tile rank must be one or two")
        coordinate = self.pipeline.body.block.args[0]
        owner = self.program.body.block.args[0]
        coordinates = {coordinate: "$g", owner: "$core"}
        coordinates.update(
            (frame.coordinate, f"$parent{i}") for i, frame in enumerate(self.ambient_frames)
        )
        if self.invocation is not None:
            coordinates[self.invocation.body.block.args[0]] = "$call"
        # Each statically selected execution lane has its own physical actor.
        memo = {variable: literal(number) for variable, number in record.constants}

        def scalar(ssa):
            return from_ssa(ssa, coordinates=coordinates, memo=memo, facts=facts)

        origins = [scalar(origin) for origin in view.origins]
        require(
            not self.once or all("$g" not in arguments(origin) for origin in origins),
            "ordinary work cannot use a child generation outside its region",
        )
        dynamic = dict(zip((v.data for v in allocation.dynamic_axes), allocation.dimensions))
        extents = [
            scalar(dynamic[axis]) if axis in dynamic else literal(size)
            for axis, size in enumerate(shape(allocation.ref))
        ]
        byte_strides, stride = [], literal(ITEMSIZE[typ.dtype.data])
        for extent in reversed(extents):
            byte_strides.insert(0, stride)
            stride = expression("mul", extent, stride)
        strides = tuple(normalize(byte_strides[axis.data], facts).constant for axis in view.axes)
        require(all(stride is not None for stride in strides), "native row pitch must be static")
        require(strides[-1] == ITEMSIZE[typ.dtype.data], "native tile columns must be contiguous")
        bound = (
            scalar(self.pipeline.limit)
            if self.pipeline.limit is not None
            else literal(self.pipeline.count.data)
        )
        g, core = Expr("argument", ("$g",)), Expr("argument", ("$core",))
        domains = [
            (g, literal(0), expression("sub", bound, literal(1))),
            (core, literal(0), literal(self.program.groups.data - 1)),
        ]
        for i, frame in enumerate(self.ambient_frames):
            extent = literal(frame.bound) if isinstance(frame.bound, int) else scalar(frame.bound)
            domains.append(
                (
                    Expr("argument", (f"$parent{i}",)),
                    literal(0),
                    expression("sub", extent, literal(1)),
                )
            )
        total = getattr(self.bindings, "cyclic_total", None)
        if total is not None:
            flat = expression("add", core, expression("mul", g, literal(self.program.groups.data)))
            domains.insert(0, (flat, literal(0), expression("sub", total, literal(1))))
        for source_coordinate, owner_domain in self.bindings.owner_domains.items():
            if source_coordinate not in coordinates:
                continue
            actual = Expr("argument", (coordinates[source_coordinate],))
            flat = expression("add", core, expression("mul", actual, literal(owner_domain.groups)))
            domains.insert(0, (flat, literal(0), expression("sub", owner_domain.total, literal(1))))
        if self.invocation is not None:
            domains.append(
                (
                    Expr("argument", ("$call",)),
                    literal(0),
                    expression("sub", scalar(self.invocation.ub), literal(1)),
                )
            )
        for origin, width, extent in zip(origins, view.widths, extents):
            if bound == literal(0) and not self.once:
                continue  # No active generation can access this axis.
            if (
                view.valid is not None
                and allocation.ref.type.memory.data == "GM"
                and len(extents) == 1
            ):
                # Exact aligned prefix geometry is proved below and independently
                # revalidated on the physical native footprints.
                continue
            check_axis(origin, width.data, extent, domains, facts)
        address = literal(0)
        for origin, stride in zip(origins, byte_strides):
            address = expression("add", address, expression("mul", origin, stride))
        root = self.roots[allocation]
        if self.capture is not None:
            from .scope_address import split_offset

            parent, address = split_offset(
                address,
                child_clock=None if self.once else "$g",
                captured_variables={
                    "$core",
                    *(f"$parent{i}" for i in range(len(self.ambient_frames))),
                },
                facts=facts,
            )
            root = self.capture(allocation, parent)
        terms = physical_terms(address, facts)
        if allocation in self.frame_specs:
            extent, scale = self.frame_specs[allocation]
            frame_stride = expression("mul", extent, literal(scale))
            expected = {
                Expr("argument", ("$call",)): frame_stride,
                core: expression("mul", scalar(self.invocation.ub), frame_stride),
            }
            for variable, translation in expected.items():
                require(
                    equivalent(terms.pop(variable, literal(0)), translation, facts),
                    "GM view must select its actual owner/invocation frame",
                )
        offset, count_offset, count_core_stride = 0, 0, 0
        generation, core_terms = [], []
        for variable, coefficient in terms.items():
            if variable is None:
                if coefficient.constant is not None:
                    offset += coefficient.constant
                else:
                    ratio = constant_ratio(coefficient, bound, facts)
                    require(
                        ratio is not None, "uniform address offset needs its loop count contract"
                    )
                    count_offset += ratio
                continue
            divisor, modulus = 1, 0
            if variable.kind == "rem":
                variable, radix = variable.args
                require(radix.constant is not None, "runtime radix cannot become a storage period")
                modulus = radix.constant
            if variable.kind == "floordiv":
                variable, radix = variable.args
                require(
                    radix.constant is not None,
                    "runtime task quotient needs a segmented access contract",
                )
                divisor = radix.constant
            require(variable in {g, core}, "coordinate escaped its task or execution-domain scope")
            if coefficient.constant is not None:
                (generation if variable == g else core_terms).append(
                    (coefficient.constant, divisor, modulus)
                )
            else:
                ratio = constant_ratio(coefficient, bound, facts)
                require(
                    variable == core and divisor == 1 and modulus == 0 and ratio is not None,
                    "runtime task stride needs a segmented ownership contract",
                )
                count_core_stride += ratio
        core = core_terms
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
        if view.valid is not None:
            require(
                self.pipeline.limit is not None and rows == 1 and len(dimensions) == 1,
                "valid prefix currently requires a one-dimensional runtime pipeline tile",
            )
            bound = from_ssa(self.pipeline.limit, memo=self.scalar_cache, facts=facts)
            prefix_elements = dimensions[0]
            require(
                bound.kind == "ceildiv" and bound.args[1].constant == prefix_elements,
                "valid prefix tile must match the pipeline ceildiv divisor",
            )
            valid = from_ssa(
                view.valid,
                coordinates={self.pipeline.body.block.args[0]: "$g"},
                memo=self.scalar_cache,
                facts=facts,
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
            root,
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
        parent_blocks = []
        for number, record in enumerate(self.records):
            if record.parent_guard is not None:
                block = Block()
                guard = ScopeGuardOp.create(
                    operands=[self.parent_generation, stream_block.args[0]],
                    attributes={"kind": StringAttr(record.parent_guard)},
                    regions=[Region(block)],
                )
                stream_block.add_op(guard)
                parent_blocks.append(block)
                self.kernel.task_block = block
            elif record.boundary is not None:
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
            uid = ("" if self.region_name is None else self.region_name + "_") + f"{stem}_{number}"
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
                    "maximum": "max",
                }.get(op.kind.data)
                if op.kind.data.endswith("_rows"):
                    require(op.scratch is not None, "row broadcast requires scratch")
                    rows, cols = shape(op.destination)
                    source, statistic, destination = (
                        self.view(value, record) for value in (*op.inputs, op.destination)
                    )
                    temporary = self.view(op.scratch, record)
                    task("_broadcast", "brcb8", statistic, temporary, params=(rows,))
                    task(
                        "_" + op.kind.data,
                        op.kind.data.removesuffix("_rows") + "_row_broadcast",
                        source,
                        temporary,
                        destination,
                        params=(rows, cols),
                        order_delta=1,
                    )
                    self.definitions[number] = uid + "_" + op.kind.data
                    self.readers[number, 0] = self.definitions[number]
                    self.readers[number, 1] = uid + "_broadcast"
                    continue
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
                block8 = op.order.data == "block8_then_packed_merge"
                if block8:
                    workspace, byte_offset = temporary, 0
                    widths = block8_reduction_widths(cols)
                    for level, columns in enumerate(widths):
                        temporary = self.kernel.view(
                            workspace,
                            columns * 4,
                            offset=byte_offset,
                            rows=rows,
                            row_stride=columns * 4,
                        )
                        task(
                            "_partial" if level == 0 else f"_block8_{level}",
                            "reduce_" + op.kind.data + "8",
                            source,
                            temporary,
                            order_delta=level,
                        )
                        source = temporary
                        byte_offset += rows * columns * 4
                    partials, merge_order = widths[-1], len(widths)
                else:
                    task(
                        "_partial",
                        "reduce_" + op.kind.data + "64",
                        source,
                        temporary,
                        params=(rows, cols),
                    )
                    partials, merge_order = cols // 64, 1
                task(
                    "_merge",
                    "merge_"
                    + op.kind.data
                    + ("_packed" if block8 or op.order.data == "r64_then_packed_merge" else ""),
                    temporary,
                    destination,
                    params=(rows, partials),
                    order_delta=merge_order,
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
        for block in parent_blocks:
            block.add_op(EndOp())
        self.kernel.task_block = stream_block
        self.kernel.finish_region()


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
                native_index = _native_input_index(record.op, read_index)
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


def _check_loop_values(physical, actors, outer, *, extracted=None, live_in=()):
    from .loop_model import extract_loop, loop_dependencies

    loop = extract_loop(physical) if extracted is None else extracted
    if not any(
        record.op.dependencies for lowering in actors.values() for record in lowering.records
    ):
        return
    raw = [o for o in loop_dependencies(loop, live_in=live_in) if o.hazard == "RAW"]
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
                native_index = _native_input_index(record.op, index)
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
    from .region_lowering import lower_regions

    if symbolic and program.attributes.get("frontend") == StringAttr("ast"):
        from .scope_lowering import has_nested_pipelines, lower_scopes

        if has_nested_pipelines(program) and any(
            isinstance(op, (PipelineOp, scf.ForOp)) for op in program.body.block.ops
        ):
            return lower_scopes(program)

    return lower_regions(program, symbolic=symbolic)

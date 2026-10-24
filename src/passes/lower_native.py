"""Recursive source-to-native structure, before access/completion admission.

Serial loops keep SCF's scalar carry semantics. Only explicit pipelines use
physical stage coordinates; arithmetic is rebound in each stage's lexical
scope. No source body is copied per iteration or selected by an operator name.

This intermediate result is deliberately not a synchronized/executable plan.
It retains all logical Tile obligations for the subsequent access analysis;
building native recipes or exiting a region never establishes completion.
"""

from __future__ import annotations

from collections import ChainMap, Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, StringAttr, f32
from xdsl.ir import Block, Operation, Region, SSAValue

from ..analysis.value_bindings import LastTile, TileDefinition, TileUse, ValueScope, bind_values
from ..diagnostics import ErrorKind, require
from ..dialects.physical import (
    BufferOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    MixedOp,
    RuntimeSizeOp,
    TaskOp,
    task_attrs,
)
from ..dialects.source import (
    AllocationOp,
    CompareOp,
    CopyGroupOp,
    CopyOp,
    DomainOp,
    ElementOp,
    IndexReadOp,
    IterationOp,
    MaskCombineOp,
    MaskOp,
    MaterializeMaskOp,
    MatmulOp,
    PredicateOp,
    ProgramOp,
    ReduceOp,
    ScalarReadOp,
    ScalarWriteOp,
    ScaleOp,
    SizeOp,
    StageOp,
    StorageSelectOp,
    TileSliceOp,
    ViewOp,
    WhereOp,
    YieldOp,
)
from ..dialects.structured import (
    ActorDomainOp,
    AddressOp,
    PipelineLoopOp,
    PipelineStageOp,
    ReadyGroupOp,
)
from ..dialects.types import TaskType, TileType, ref_type
from ..indexing import multiple_of, validate_static_arithmetic
from ..scalar import facts_from_sizes, from_ssa
from ..target.native import ITEMSIZE
from ..target.scalars import INTEGER_OPERATIONS
from . import native_recipes as recipes
from .native_addresses import lower_address, subview


def _unsupported(condition, message, operation=None):
    require(
        condition,
        message,
        kind=ErrorKind.UNSUPPORTED,
        phase="recursive native lowering",
        location=None if operation is None else str(operation.location),
    )


@dataclass(frozen=True)
class NativeStructure:
    """Owned intermediate IR and unresolved value/access admission inputs.

    This type has no code generation method. In particular, ``module.verify``
    establishes only xDSL structure, not bounds, alias, ownership, stale-Tile
    legality, asynchronous completion, or event resource safety.
    """

    source: ModuleOp
    module: ModuleOp
    recipes: Mapping[Operation, recipes.RecipeResult]
    loops: Mapping[Operation, Operation]
    values: Mapping[SSAValue, TileDefinition | LastTile]
    uses: tuple[TileUse, ...]
    task_scopes: Mapping[TaskOp, ValueScope]
    work: Mapping[str, int]
    ready_groups: Mapping[CopyGroupOp, ReadyGroupOp]


_INHERIT = object()


@dataclass
class _Scope:
    block: Block
    values: ChainMap
    execution: ValueScope
    # Pure results are scoped to this insertion block. Reusing a parent's pure
    # cache across a stage would retain that stage's g in an address like g%2.
    pure: dict[SSAValue, SSAValue] = field(default_factory=dict)
    stage: int = 0
    pipeline: IterationOp | None = None
    family: str | None = None

    def child(
        self,
        block,
        bindings=(),
        *,
        stage=None,
        pipeline=_INHERIT,
        family=_INHERIT,
        control=None,
        branch=None,
    ):
        return _Scope(
            block,
            self.values.new_child(dict(bindings)),
            self.execution if control is None else ValueScope(self.execution, control, branch),
            stage=self.stage if stage is None else stage,
            pipeline=self.pipeline if pipeline is _INHERIT else pipeline,
            family=self.family if family is _INHERIT else family,
        )

    def constant(self, value):
        op = arith.ConstantOp.from_int_and_width(value, IndexType())
        self.block.add_op(op)
        return op.result


class _Lowerer:
    def __init__(self, source, program):
        self.source, self.program = source, program
        self.roots = {}
        self.index_facts = None
        self.results = {}
        self.loops = {}
        self.family_work = Counter()
        self.task_scopes = {}
        self.needs_fix = False
        self.needs_fp32 = any(
            isinstance(op, MatmulOp) and op.a.type.dtype.data == "f32" for op in source.walk()
        )
        self.load3d_a = (
            self.needs_fp32
            or any(recipes.strided_b_width(op) is not None for op in source.walk())
            or any(
                recipes.compact_a_width(op) is not None and op.source.type.dtype.data == "f32"
                for op in source.walk()
            )
        )
        self.register_uses = []
        self.ready_groups = {}
        self.work = Counter(source_operations=0, native_templates=0, expanded_iterations=0)

    def scalar(self, value, scope):
        if value in scope.values:
            return scope.values[value]
        if value in scope.pure:
            return scope.pure[value]
        owner = value.owner
        if isinstance(owner, StorageSelectOp):
            selector = self.scalar(owner.selector, scope)
            if owner.mode.data == "generation":
                operation = arith.RemUIOp(selector, scope.constant(owner.depth.data))
                scope.block.add_op(operation)
                selector = operation.result
            scope.pure[value] = selector
            return selector
        _unsupported(
            isinstance(owner, (arith.ConstantOp, arith.CmpiOp, PredicateOp))
            or type(owner) in INTEGER_OPERATIONS,
            "scalar value must retain its lexical binding; asynchronous values cannot be cloned",
            owner if isinstance(owner, Operation) else None,
        )
        if isinstance(owner, PredicateOp):
            result = self.predicate(owner, scope)
        else:
            cloned = owner.clone(
                {operand: self.scalar(operand, scope) for operand in owner.operands}
            )
            scope.block.add_op(cloned)
            self.work["scalar_templates"] += 1
            result = cloned.results[value.index]
        scope.pure[value] = result
        return result

    def predicate(self, op, scope):
        def side(terms, offset):
            result = scope.constant(offset.data)
            for term in terms:
                index, coefficient, divisor, modulus = (value.data for value in term)
                value = self.scalar(op.indices[index], scope)
                if divisor != 1:
                    div = arith.DivUIOp(value, scope.constant(divisor))
                    scope.block.add_op(div)
                    value = div.result
                if modulus:
                    rem = arith.RemUIOp(value, scope.constant(modulus))
                    scope.block.add_op(rem)
                    value = rem.result
                mul = arith.MuliOp(value, scope.constant(coefficient))
                scope.block.add_op(mul)
                add = arith.AddiOp(result, mul.result)
                scope.block.add_op(add)
                result = add.result
            return result

        lhs, rhs = side(op.lhs_terms, op.lhs_offset), side(op.rhs_terms, op.rhs_offset)
        result = arith.CmpiOp(lhs, rhs, {"eq": "eq", "lt": "slt", "le": "sle"}[op.comparison.data])
        scope.block.add_op(result)
        return result.result

    def allocation(self, op, scope):
        _unsupported(
            op.parent_op() is self.program, "allocations must be declared at kernel entry", op
        )
        _unsupported(
            op.versions.data == 1 and op.ring.data == 0, "use explicit physical slot axes", op
        )
        dimensions = [self.scalar(value, scope) for value in op.dimensions]
        extent = None
        for dimension in dimensions:
            if extent is None:
                extent = dimension
            else:
                mul = arith.MuliOp(extent, dimension)
                scope.block.add_op(mul)
                extent = mul.result
        size = (
            prod(n.data for n in op.ref.type.shape if n.data > 0) * ITEMSIZE[op.ref.type.dtype.data]
        )
        attrs = {
            "sym_name": op.sym_name,
            "nbytes": IntAttr(size),
            "external": IntAttr(int(op.ref.type.memory.data == "GM")),
            "initialized": IntAttr(int(op.access.data in {"read", "read_write"})),
            "access": op.access,
        }
        if op.ref.type.memory.data == "L1" and op.ref.type.layout.data == "NZ":
            attrs["matrix_shape"] = ArrayAttr(IntAttr(n.data) for n in op.ref.type.shape)
        if op.write_disjoint is not None:
            attrs["write_disjoint"] = op.write_disjoint
        if op.cache is not None:
            attrs["cache"] = op.cache
        if extent is not None:
            attrs["count_scale"] = IntAttr(size)
        buffer = BufferOp.build(
            operands=[extent],
            attributes=attrs,
            result_types=[
                ref_type(op.ref.type.memory.data, op.ref.type.dtype.data, op.ref.type.layout.data)
            ],
        )
        scope.block.add_op(buffer)
        self.roots[op] = buffer.ref

    def primitive(self, op, scope):
        _unsupported(
            scope.family is not None,
            "mixed native work needs an explicit Cube or Vector domain",
            op,
        )
        number = len(self.results)

        def emit(
            suffix,
            native,
            *refs,
            params=(),
            scalar=0.0,
            order_delta=0,
            operands=(),
            mask=None,
            mask_word=None,
            mask_repeat=None,
            mask_iteration=None,
            mask_layout=None,
        ):
            self.needs_fix |= native in {"fix", "fix16"}
            scalar_operands = tuple(operands)
            if native in {"mad_first", "mad_cycle"}:
                require(scope.pipeline is not None, "native reset needs its logical ordinal")
                scalar_operands = (
                    *scalar_operands,
                    scope.values[scope.pipeline.body.block.args[0]],
                )
            attributes = {
                **task_attrs(f"native_{number}{suffix}", native, scope.stage, params, scalar),
                "address_count": IntAttr(len(refs)),
                **({} if mask is None else mask.encode()),
                **({} if mask_word is None else {"mask_word": StringAttr(mask_word)}),
                **({} if mask_repeat is None else {"mask_repeat": StringAttr(mask_repeat)}),
                **(
                    {} if mask_iteration is None else {"mask_iteration": StringAttr(mask_iteration)}
                ),
                **({} if mask_layout is None else {"mask_layout": StringAttr(mask_layout)}),
            }
            task = TaskOp.build(
                operands=[(*refs, *scalar_operands)],
                attributes=attributes,
                result_types=[
                    TaskType(),
                    f32
                    if native == "scalar_load"
                    else IndexType()
                    if native == "index_load"
                    else None,
                ],
            )
            task.location = op.location
            scope.block.add_op(task)
            self.task_scopes[task] = scope.execution
            if native in {"load3d_a", "load3d_b_nn"}:
                address = refs[0].owner
                while isinstance(address, AddressOp):
                    address = address.source.owner
                matrix = address.matrix_shape
                self.register_uses.append(
                    (task, scope, "a" if native == "load3d_a" else "b", matrix.data[-2].data)
                )
            self.work["native_templates"] += 1
            self.family_work[scope.family] += 1
            return task

        def view(value, *, matrix=False):
            return lower_address(
                value,
                materialize=lambda value: self.scalar(value, scope),
                roots=self.roots,
                append=scope.block.add_op,
                matrix=matrix,
            )

        if self.index_facts is None:
            root = scope.block.parent_op()
            while root.parent_op() is not None and not isinstance(root, (KernelOp, MixedOp)):
                root = root.parent_op()
            self.index_facts = facts_from_sizes(root.body.block.ops)

        context = recipes.RecipeContext(
            view,
            lambda source, nbytes, **kwargs: subview(
                source, nbytes, append=scope.block.add_op, **kwargs
            ),
            emit,
            operand=lambda value: self.scalar(value, scope),
            aligned=lambda address, alignment: multiple_of(
                from_ssa(address.owner.offset), alignment, self.index_facts
            ),
            load3d_a=self.load3d_a,
        )
        if isinstance(op, CopyOp):
            result = recipes.lower_copy(op, context)
        elif isinstance(op, ElementOp):
            result = recipes.lower_element(op, context)
        elif isinstance(op, ReduceOp):
            result = recipes.lower_reduce(op, context)
        elif isinstance(op, ScalarReadOp):
            result = recipes.lower_scalar_read(op, context)
            scope.values[op.value] = result.scalar_value
        elif isinstance(op, ScalarWriteOp):
            result = recipes.lower_scalar_write(op, context, value=self.scalar(op.source, scope))
        elif isinstance(op, IndexReadOp):
            result = recipes.lower_index_read(
                op,
                context,
                minimum=self.scalar(op.minimum, scope),
                maximum=self.scalar(op.maximum, scope),
            )
            scope.values[op.value] = result.scalar_value
        elif isinstance(op, ScaleOp):
            result = recipes.lower_scale(op, context, factor=self.scalar(op.factor, scope))
        elif isinstance(op, CompareOp):
            result = recipes.lower_compare(op, context)
        elif isinstance(op, MaterializeMaskOp):
            result = recipes.lower_materialize_mask(op, context)
        elif isinstance(op, MaskCombineOp):
            result = recipes.lower_mask_combine(op, context)
        elif isinstance(op, WhereOp):
            result = recipes.lower_where(op, context)
        else:
            result = recipes.lower_matmul(
                op, context, initialization=self.initialization(op, scope)
            )
        self.results[op] = result

    def initialization(self, op, scope):
        predicate = op.initialize.owner
        if isinstance(predicate, PredicateOp) and not predicate.indices:
            lhs, rhs = predicate.lhs_offset.data, predicate.rhs_offset.data
            return int(
                {"eq": lhs == rhs, "lt": lhs < rhs, "le": lhs <= rhs}[predicate.comparison.data]
            )
        if isinstance(predicate, arith.CmpiOp) and scope.pipeline is not None:
            values = predicate.operands
            coordinate = scope.pipeline.body.block.args[0]
            other = next((value for value in values if value is not coordinate), None)
            if (
                predicate.predicate.value.data == 0
                and coordinate in values
                and other is not None
                and isinstance(other.owner, arith.ConstantOp)
                and other.owner.value.value.data == 0
            ):
                return "first"
        _unsupported(
            False,
            "matmul initialization needs a constant or its own pipeline's first logical iteration",
            op,
        )

    def block(self, source, scope):
        for op in source.ops:
            self.operation(op, scope)

    def operation(self, op, scope):
        self.work["source_operations"] += 1
        if isinstance(op, SizeOp):
            size = RuntimeSizeOp.build(attributes=dict(op.attributes), result_types=[IndexType()])
            scope.block.add_op(size)
            scope.values[op.value] = size.value
        elif isinstance(op, AllocationOp):
            self.allocation(op, scope)
        elif isinstance(op, CopyGroupOp):
            before = set(scope.block.ops)
            self.block(op.body.block, scope)
            members = tuple(
                operation.task
                for operation in scope.block.ops
                if isinstance(operation, TaskOp) and operation not in before
            )
            _unsupported(bool(members), "CopyGroup has no native copy members", op)
            group = ReadyGroupOp.build(operands=[members], attributes={"label": op.label})
            group.location = op.location
            scope.block.add_op(group)
            self.ready_groups[op] = group
        elif isinstance(
            op,
            (
                CopyOp,
                ElementOp,
                ReduceOp,
                ScalarReadOp,
                ScalarWriteOp,
                IndexReadOp,
                ScaleOp,
                CompareOp,
                MaterializeMaskOp,
                MaskCombineOp,
                WhereOp,
                MatmulOp,
            ),
        ):
            self.primitive(op, scope)
        elif isinstance(op, MaskOp):
            # A symbolic predicate owns no native until it is materialized or
            # consumed by a picked masked recipe.
            op.program()
        elif isinstance(op, IterationOp):
            self.pipeline(op, scope)
        elif isinstance(op, scf.ForOp):
            self.serial(op, scope)
        elif isinstance(op, scf.IfOp):
            self.conditional(op, scope)
        elif isinstance(op, DomainOp):
            if self.program.kind.data == "mixed":
                block = Block(arg_types=[IndexType()])
                domain = ActorDomainOp.build(attributes={"kind": op.kind}, regions=[Region(block)])
                domain.location = op.location
                scope.block.add_op(domain)
                nested = scope.child(
                    block,
                    [(op.body.block.args[0], block.args[0])],
                    family=op.kind.data,
                    control=domain,
                )
                self.block(op.body.block, nested)
                block.add_op(EndOp())
            else:
                _unsupported(
                    op.kind == self.program.kind,
                    "execution domain belongs to another actor family",
                    op,
                )
                # A single family has one lane; group is a different coordinate.
                nested = scope.child(scope.block, [(op.body.block.args[0], scope.constant(0))])
                self.block(op.body.block, nested)
        elif isinstance(op, scf.YieldOp):
            _unsupported(
                all(not isinstance(value.type, TileType) for value in op.arguments),
                "SCF Tile carry needs an explicit value selection contract",
                op,
            )
            scope.block.add_op(scf.YieldOp(*(self.scalar(value, scope) for value in op.arguments)))
        elif (
            isinstance(
                op,
                (
                    YieldOp,
                    ViewOp,
                    PredicateOp,
                    StorageSelectOp,
                    TileSliceOp,
                    arith.ConstantOp,
                    arith.CmpiOp,
                ),
            )
            or type(op) in INTEGER_OPERATIONS
        ):
            # Pure scalar work is materialized at its actual use, with this
            # stage's coordinate. Tile yields are retained in the value DAG.
            return
        else:
            _unsupported(
                False, f"unsupported source operation {op.name}; effects cannot be discarded", op
            )

    def serial(self, op, scope):
        _unsupported(
            all(not isinstance(value.type, TileType) for value in op.iter_args),
            "SCF Tile carry needs an explicit value selection contract",
            op,
        )
        body = Block(arg_types=[arg.type for arg in op.body.block.args])
        loop = scf.ForOp(
            self.scalar(op.lb, scope),
            self.scalar(op.ub, scope),
            self.scalar(op.step, scope),
            [self.scalar(value, scope) for value in op.iter_args],
            body,
        )
        loop.location = op.location
        loop.attributes["npu.loop_id"] = StringAttr(f"loop{len(self.loops)}")
        scope.block.add_op(loop)
        self.loops[op] = loop
        self.block(
            op.body.block,
            scope.child(
                body,
                zip(op.body.block.args, body.args, strict=True),
                stage=0,
                pipeline=None,
                control=loop,
            ),
        )
        scope.values.update(zip(op.results, loop.results, strict=True))

    def conditional(self, op, scope):
        _unsupported(
            all(not isinstance(value.type, TileType) for value in op.results),
            "SCF Tile merge needs an explicit value selection contract",
            op,
        )
        blocks = [Block(), Block()]
        conditional = scf.IfOp(
            self.scalar(op.cond, scope),
            [v.type for v in op.results],
            *(Region(block) for block in blocks),
        )
        conditional.location = op.location
        scope.block.add_op(conditional)
        for branch, (region, block) in enumerate(
            zip((op.true_region, op.false_region), blocks, strict=True)
        ):
            if region.blocks:
                self.block(region.block, scope.child(block, control=conditional, branch=branch))
            else:
                block.add_op(scf.YieldOp())
        scope.values.update(zip(op.results, conditional.results, strict=True))

    def pipeline(self, op, scope):
        body = Block()
        loop = PipelineLoopOp.build(
            operands=[
                self.scalar(op.limit, scope)
                if op.limit is not None
                else scope.constant(op.count.data),
            ],
            attributes={
                "issue_order": op.issue_order,
            },
            regions=[Region(body)],
        )
        loop.location = op.location
        loop.attributes["npu.loop_id"] = StringAttr(f"loop{len(self.loops)}")
        scope.block.add_op(loop)
        self.loops[op] = loop
        stages, stage_families = [], {}
        loop_scope = scope.child(body, control=loop)

        def stage(operations, offset, label, actor=None):
            before = dict(self.family_work)
            block = Block(arg_types=[IndexType()])
            physical = PipelineStageOp.build(
                attributes={
                    "offset": IntAttr(offset),
                    "label": StringAttr(label),
                    **(
                        {"family": actor.kind}
                        if actor is not None
                        else {"family": StringAttr(scope.family)}
                        if scope.family is not None
                        else {}
                    ),
                },
                regions=[Region(block)],
            )
            body.add_op(physical)
            nested = loop_scope.child(
                block,
                [(op.body.block.args[0], block.args[0])],
                stage=offset,
                pipeline=op,
                control=physical,
            )
            mixed_actor = actor is not None and self.program.kind.data == "mixed"
            if mixed_actor:
                actor_body = Block(arg_types=[IndexType()])
                domain = ActorDomainOp.build(
                    attributes={"kind": actor.kind}, regions=[Region(actor_body)]
                )
                domain.location = actor.location
                block.add_op(domain)
                nested = nested.child(
                    actor_body,
                    [(actor.body.block.args[0], actor_body.args[0])],
                    family=actor.kind.data,
                    control=domain,
                )
            elif actor is not None:
                _unsupported(
                    actor.kind == self.program.kind,
                    "execution domain belongs to another actor family",
                    actor,
                )
                nested = nested.child(block, [(actor.body.block.args[0], nested.constant(0))])
            for operation in operations:
                self.operation(operation, nested)
            if mixed_actor:
                actor_body.add_op(EndOp())
            block.add_op(EndOp())
            stages.append(physical)
            stage_families[physical] = frozenset(
                family
                for family, count in self.family_work.items()
                if count > before.get(family, 0)
            )

        descriptors = (
            ViewOp,
            YieldOp,
            PredicateOp,
            StorageSelectOp,
            TileSliceOp,
            arith.ConstantOp,
            arith.CmpiOp,
        )

        def collect(source, actor=None):
            ordinary = []

            def flush():
                if ordinary:
                    stage(ordinary, 0, "", actor)
                    ordinary.clear()

            for child in source.ops:
                if isinstance(child, StageOp):
                    flush()
                    self.work["source_operations"] += 1
                    stage(child.body.block.ops, child.offset.data, child.label.data, actor)
                elif isinstance(child, DomainOp):
                    flush()
                    self.work["source_operations"] += 1
                    _unsupported(actor is None, "actor domains cannot nest", child)
                    collect(child.body.block, child)
                elif isinstance(child, descriptors) or type(child) in INTEGER_OPERATIONS:
                    self.work["source_operations"] += 1
                else:
                    ordinary.append(child)
            flush()

        collect(op.body.block)
        if op.issue_order is not None:
            for family, labels in op.issue_order.data.items():
                names = [label.data for label in labels]
                actual = [s.label.data for s in stages if family in stage_families[s]]
                _unsupported(
                    len(set(actual)) == len(actual) and set(names) == set(actual),
                    f"{family}: issue_order must name every participating Stage exactly once",
                    op,
                )
            # The body retains logical access order for dependence analysis.
            # Physical issue order belongs to this explicit per-family table,
            # including single-actor loops. Reordering the body itself would
            # turn an expert export-first dispatch into a different algorithm.
        body.add_op(EndOp())


def lower_native_structure(source) -> NativeStructure:
    """Own a source module and retain its actual recursive control and natives.

    Explicit intermediate entry while the recursive access and synchronization
    consumers establish their own access and completion contracts.
    """
    from ..frontend import PrimFunction

    if isinstance(source, PrimFunction):
        # The known parser returns fresh, structurally checked IR directly
        # into this compilation. It has no external owner to preserve.
        source = PrimFunction.to_ir(source)
    else:
        require(isinstance(source, ModuleOp), "native lowering requires typed source IR")
        source = source.clone()
        source.verify()
    program = source.body.block.first_op
    require(
        isinstance(program, ProgramOp) and len(source.body.block.ops) == 1,
        "expected one typed source program",
    )
    validate_static_arithmetic(program)
    body = Block()
    attrs = {
        "sym_name": program.sym_name,
        "target": StringAttr("dav-2201"),
        "disjoint_external": program.disjoint_external,
        "blocks": program.groups,
    }
    mixed = program.kind.data == "mixed"
    if not mixed:
        attrs["core"] = StringAttr("aic" if program.kind.data == "cube" else "aiv")
    kernel = (MixedOp if mixed else KernelOp).build(
        attributes=attrs,
        regions=[Region(body)],
    )
    module = ModuleOp([kernel])
    core = CoreIdOp.build(result_types=[IndexType()])
    body.add_op(core)
    scope = _Scope(
        body,
        ChainMap({program.body.block.args[0]: core.value}),
        ValueScope(None, kernel),
        family=None if mixed else program.kind.data,
    )
    lowerer = _Lowerer(source, program)
    lowerer.block(program.body.block, scope)
    if lowerer.needs_fix:
        # One actual invariant register initializer dominates every Fixpipe
        # reader, even when a containing runtime loop has zero iterations.
        # Its native packet includes the target S -> FIX readiness handshake.
        body.insert_op_before(FixConfigOp(), core)
    setup = []

    def register_task(native, params, task_scope, stage=0):
        task = TaskOp.build(
            operands=[()],
            attributes={
                **task_attrs(f"register_{len(lowerer.task_scopes)}", native, stage, params),
                "address_count": IntAttr(0),
            },
            result_types=[TaskType(), None],
        )
        lowerer.task_scopes[task] = task_scope
        lowerer.work["native_templates"] += 1
        return task

    if lowerer.needs_fp32:
        setup.append(register_task("set_fp32", (), scope.execution))
    if lowerer.register_uses:
        setup.append(register_task("set_load3d", (), scope.execution))
        widths = {}
        for _, _, side, width in lowerer.register_uses:
            widths.setdefault(side, set()).add(width)
        for side, values in widths.items():
            if len(values) == 1:
                setup.append(
                    register_task(f"set_fmatrix_{side}", (next(iter(values)),), scope.execution)
                )
        for task, use_scope, side, width in lowerer.register_uses:
            if len(widths[side]) > 1:
                setter = register_task(
                    f"set_fmatrix_{side}", (width,), use_scope.execution, use_scope.stage
                )
                task.parent.insert_op_before(setter, task)
    if setup:
        # Generated ready guards can depend on runtime sizes and the group
        # coordinate. Keep their real SSA definitions before the setters;
        # the invariant Fixpipe packet itself has no such dependency.
        setup_before = next(
            operation
            for operation in body.ops
            if not isinstance(
                operation,
                (BufferOp, CoreIdOp, RuntimeSizeOp, FixConfigOp, *tuple(arith.Arith.operations)),
            )
        )
        if mixed:
            config_body = Block(arg_types=[IndexType()])
            config_body.add_ops((*setup, EndOp()))
            domain = ActorDomainOp.build(
                attributes={"kind": StringAttr("cube")}, regions=[Region(config_body)]
            )
            body.insert_op_before(domain, setup_before)
        else:
            body.insert_ops_before(setup, setup_before)
    body.add_op(EndOp())
    values, uses = bind_values(source, lowerer.results, lowerer.loops)
    # The constructors preserve structure from the admitted source. Developer
    # round-trip regressions verify the transformation; normal compilation
    # need not repeat a whole-module structural check here. Native/access
    # admission still establishes its own new physical premises downstream.
    lowerer.work["loop_templates"] = len(lowerer.loops)
    lowerer.work["value_bindings"] = len(values)
    lowerer.work["value_uses"] = len(uses)
    return NativeStructure(
        source,
        module,
        MappingProxyType(lowerer.results),
        MappingProxyType(lowerer.loops),
        MappingProxyType(values),
        uses,
        MappingProxyType(lowerer.task_scopes),
        MappingProxyType(dict(lowerer.work)),
        MappingProxyType(lowerer.ready_groups),
    )

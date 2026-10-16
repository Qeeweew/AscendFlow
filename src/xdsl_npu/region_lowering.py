"""One source-region admission and actor construction path.

These immutable handles refer to the original xDSL regions; they are not a
second program or a flattened execution history. Present admission is one
local serial parent with ordinary work and direct pipelines, or one mixed
pipeline. Nested calls use the same collected work and actor bindings.
"""

from dataclasses import dataclass, replace

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from .lowering_context import ActorBindings
from .model import require
from .scalar import ARITHMETIC, from_ssa, literal
from .source_ir import AllocationOp, DomainOp, MatmulOp, PipelineOp, SizeOp, ViewOp, YieldOp
from .source_scopes import Scope, collect_source_scopes
from .source_scopes import Work as ScopedWork


@dataclass(frozen=True)
class SourceRegion:
    pipeline: PipelineOp
    operations: tuple | None = None
    work: tuple[ScopedWork, ...] | None = None
    scope: Scope | None = None


@dataclass(frozen=True)
class SourceBody:
    invocation: scf.ForOp | None
    regions: tuple[SourceRegion, ...]
    allocations: tuple[AllocationOp, ...]
    scope: Scope | None


def _has_work(op):
    if isinstance(op, DomainOp):
        return any(_has_work(child) for child in op.body.block.ops)
    return not (
        isinstance(op, (AllocationOp, SizeOp, ViewOp, YieldOp, scf.YieldOp, arith.ConstantOp))
        or type(op) in ARITHMETIC
    )


def source_body(program, *, symbolic, _scope=None):
    """Admit the implemented control subset before lowering any native work."""
    parents = [op for op in program.body.block.ops if isinstance(op, scf.ForOp)]
    require(
        len(parents) <= 1 and (not parents or symbolic and program.kind.data != "mixed"),
        "structured invocation lowering currently requires one local actor and parent",
    )
    invocation = parents[0] if parents else None
    scope = invocation or program
    if invocation is not None:
        require(
            not invocation.iter_args
            and not invocation.results
            and from_ssa(invocation.lb) == literal(0)
            and from_ssa(invocation.step) == literal(1),
            "structured scf.for synchronization requires zero-based unit-step control with no scalar carry",
        )
        require(
            all(op is invocation or not _has_work(op) for op in program.body.block.ops),
            "work outside a serial sequence requires an enclosing region contract",
        )
    pipelines = [op for op in scope.body.block.ops if isinstance(op, PipelineOp)]
    require(pipelines, "source requires a pipeline with a declared iteration domain")
    regions, pending = [], []

    def flush():
        if any(_has_work(op) for op in pending):
            regions.append(SourceRegion(pipelines[0], tuple(pending)))
        pending.clear()

    for op in scope.body.block.ops:
        if isinstance(op, PipelineOp):
            flush()
            regions.append(SourceRegion(op))
        elif not isinstance(op, SizeOp):
            pending.append(op)
    flush()
    sequential = len(regions) > 1
    require(not sequential or symbolic, "sequential source regions require retained control")
    require(
        not sequential or program.kind.data != "mixed",
        "mixed sequence ownership is not yet implemented",
    )
    ast = program.attributes.get("frontend") == StringAttr("ast")
    allocations, conditions = [], []
    for op in program.walk():
        if isinstance(op, AllocationOp):
            allocations.append(op)
        if ast and isinstance(op, MatmulOp):
            require(
                op.a.type.memory.data == "L0A",
                "AST matmul requires explicit L0 operands; implicit L1 recipes are legacy",
            )
        if isinstance(op, scf.ForOp):
            require(
                op is invocation,
                f"{op.location}: structured scf.for synchronization needs hierarchical region analysis",
            )
        if isinstance(op, PipelineOp):
            require(
                op.parent_op() is scope,
                "nested AST pipelines require hierarchical synchronization; "
                "constant bounds do not authorize implicit unrolling",
            )
        require(
            symbolic or not isinstance(op, scf.IfOp),
            "boundary regions require the symbolic compiler; no finite fallback",
        )
        if isinstance(op, scf.IfOp):
            conditions.append(op)
    # Admit the entire control tree before interpreting a predicate in the
    # current one-clock subset. A nested clock must not lose its ancestor facts
    # and be misdiagnosed as invalid scalar arithmetic.
    for op in conditions:
        from .lowering import _last_condition

        parent = op.parent_op()
        while parent is not None and not isinstance(parent, PipelineOp):
            require(
                not isinstance(parent, scf.IfOp),
                "nested boundary predicates need hierarchical analysis",
            )
            parent = parent.parent_op()
        require(
            parent is not None,
            "ordinary source work cannot borrow a child stage or boundary predicate",
        )
        _last_condition(op, parent)
    source_scope = collect_source_scopes(program) if _scope is None else _scope
    require(source_scope.op is program, "source collection belongs to another program")
    by_scope = {item.op: item for item in source_scope.walk_scopes()}
    direct_work = {
        op: tuple(item for item in tree.items if isinstance(item, ScopedWork))
        for op, tree in by_scope.items()
    }
    owner_scope = by_scope[scope]
    owner_ops = tuple(scope.body.block.ops)
    prefix = len(owner_scope.context.path)
    by_operation = {}
    for item in direct_work[scope]:
        op = owner_ops[item.context.path[prefix]]
        by_operation.setdefault(op, []).append(item)
    regions = [
        replace(
            region,
            scope=by_scope[region.pipeline] if region.operations is None else owner_scope,
            work=direct_work[region.pipeline]
            if region.operations is None
            else tuple(item for op in region.operations for item in by_operation.get(op, ())),
        )
        for region in regions
    ]
    return SourceBody(invocation, tuple(regions), tuple(allocations), source_scope)


def lower_regions(program, *, symbolic, _scope=None, _facts=None):
    from .lowering import _Actor, _check_loop_values, _check_values, _work
    from .mixed import compose
    from .sequence_model import extract_sequence

    body = source_body(program, symbolic=symbolic, _scope=_scope)
    from .indexing import validate_source_arithmetic

    validate_source_arithmetic(program)
    multiple = len(body.regions) > 1
    domains = (
        (("aic", "cube", 0), ("aiv0", "vector", 0), ("aiv1", "vector", 1))
        if program.kind.data == "mixed"
        else ((program.kind.data, program.kind.data, 0),)
    )
    modules, actors = [], {}
    for name, family, owner in domains:
        bindings = ActorBindings(program, family, body.allocations, invocation=body.invocation)
        leaves = []
        for region in body.regions:
            records = _work(
                program,
                region.pipeline,
                family,
                owner,
                operations=region.operations,
                scoped=region.work,
                scope=region.scope,
            )
            if not records and multiple:
                continue
            leaf = _Actor(
                program,
                region.pipeline,
                name,
                family,
                owner,
                body.allocations,
                symbolic=symbolic,
                invocation=body.invocation,
                records=records,
                once=region.operations is not None,
                region_name=f"r{len(leaves)}" if multiple else None,
                bindings=bindings,
            )
            leaf.emit()
            leaves.append(leaf)
        require(leaves, "sequence contains no actual source work")
        module = bindings.kernel.finish()
        bindings.kernel.op.attributes["source_schedule"] = ArrayAttr(
            item for leaf in leaves for item in leaf.schedule
        )
        if multiple:
            # Existing physical representations retain their proof adapters.
            # Both use this same source admission and native construction path.
            sequence = (
                extract_sequence(module) if _facts is None else _facts.extract_sequence(module)
            )
            for leaf in leaves:
                _check_loop_values(
                    module,
                    {name: leaf},
                    leaf.pipeline,
                    extracted=sequence.regions[leaf.region_name],
                    live_in=sequence.live_in[leaf.region_name],
                    _facts=_facts,
                )
        else:
            actors[name] = leaves[0]
        modules.append(module)
    physical = (
        compose(
            program.sym_name.data, *modules, disjoint_external=bool(program.disjoint_external.data)
        )
        if program.kind.data == "mixed"
        else modules[0]
    )
    if not multiple:
        _check_values(physical, actors, body.regions[0].pipeline, _facts=_facts)
    return physical

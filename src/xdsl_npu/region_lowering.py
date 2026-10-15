"""One source-region admission and actor construction path.

These immutable handles refer to the original xDSL regions; they are not a
second program or a flattened execution history. Present admission is one
local serial parent with ordinary work and direct pipelines, or one mixed
pipeline. The explicit legacy path alone admits finite nested subdivisions.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from .lowering_context import ActorBindings
from .model import require
from .scalar import ARITHMETIC, from_ssa, literal
from .source_ir import AllocationOp, DomainOp, MatmulOp, PipelineOp, SizeOp, ViewOp, YieldOp


@dataclass(frozen=True)
class SourceRegion:
    pipeline: PipelineOp
    operations: tuple | None = None


@dataclass(frozen=True)
class SourceBody:
    invocation: scf.ForOp | None
    regions: tuple[SourceRegion, ...]
    allocations: tuple[AllocationOp, ...]


def _has_work(op):
    if isinstance(op, DomainOp):
        return any(_has_work(child) for child in op.body.block.ops)
    return not (
        isinstance(op, (AllocationOp, SizeOp, ViewOp, YieldOp, scf.YieldOp, arith.ConstantOp))
        or type(op) in ARITHMETIC
    )


def source_body(program, *, symbolic):
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
    allocations = []
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
        if (ast or sequential) and isinstance(op, PipelineOp):
            require(
                op.parent_op() is scope,
                "nested AST pipelines require hierarchical synchronization; "
                "constant bounds do not authorize implicit unrolling",
            )
        require(
            symbolic or not isinstance(op, scf.IfOp),
            "boundary regions require the symbolic compiler; no finite fallback",
        )
    return SourceBody(invocation, tuple(regions), tuple(allocations))


def lower_regions(program, *, symbolic):
    from .lowering import _Actor, _check_loop_values, _check_values, _work
    from .mixed import compose
    from .sequence_model import extract_sequence

    body = source_body(program, symbolic=symbolic)
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
            records = _work(program, region.pipeline, family, owner, operations=region.operations)
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
            sequence = extract_sequence(module)
            for leaf in leaves:
                _check_loop_values(
                    module,
                    {name: leaf},
                    leaf.pipeline,
                    extracted=sequence.regions[leaf.region_name],
                    live_in=sequence.live_in[leaf.region_name],
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
        _check_values(physical, actors, body.regions[0].pipeline)
    return physical

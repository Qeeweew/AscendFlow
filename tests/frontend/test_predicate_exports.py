"""Packed predicates cross source scopes with their exact writer and storage."""

import pytest
from xdsl.utils.exceptions import VerifyException

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.source import DomainOp, ElementOp, IterationOp, MaterializeMaskOp
from ascendflow.dialects.types import MaskType
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


@pytest.mark.parametrize("kind", ["coordinate", "compare", "combine"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_exported_predicate_roundtrips_and_keeps_its_actual_packed_reads(kind, count):
    construction = {
        "coordinate": "condition=N.materialize(N.arange(64)<32,out=storage)",
        "compare": "condition=N.compare(x,0.0,kind='gt',out=storage)",
        "combine": """left=N.materialize(N.arange(64)<32,out=storage)
            right=N.compare(x,0.0,kind='gt',out=other_storage)
            condition=N.mask_or(left,right,out=combined)""",
    }[kind]
    construction = "\n                    ".join(line.strip() for line in construction.splitlines())
    definition = N.from_source(f"""
        def exported():
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,),N.f32)
                scratch=N.alloc_ub((64,),N.f32)
                storage=N.alloc_mask((64,))
                other_storage=N.alloc_mask((64,))
                combined=N.alloc_mask((64,))
                with N.Vector():
                    N.fill(x,1.0)
                    {construction}
                for task in N.Dataflow({count}):
                    with N.Vector(),N.Stage(0):
                        N.fill(x,0.0,mask=condition,scratch=scratch)
    """)
    source = parse(dump(definition.to_ir()))
    source.verify()
    plan = synchronize_structure(source)
    check_compiled_events(plan, invocations=2)


def test_materialized_predicate_crosses_stage_results_and_unused_iteration_results():
    source = N.from_source("""
        def staged():
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,),N.f32)
                scratch=N.alloc_ub((64,),N.f32)
                storage=N.alloc_mask((64,))
                for task in N.Dataflow(3):
                    with N.Vector(),N.Stage(0,name="predicate"):
                        N.fill(x,1.0)
                        condition=N.materialize(N.arange(64)<32,out=storage)
                    with N.Vector(),N.Stage(0,name="consume"):
                        N.fill(x,0.0,mask=condition,scratch=scratch)
    """)
    plan = synchronize_structure(parse(dump(source.to_ir())))
    check_compiled_events(plan, invocations=2)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_iteration_predicate_export_keeps_last_writer_and_nonempty_obligation(count):
    definition = N.from_source(f"""
        def last_predicate():
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,),N.f32)
                scratch=N.alloc_ub((64,),N.f32)
                storage=N.alloc_mask((64,))
                with N.Vector():
                    N.fill(x,1.0)
                    condition=N.materialize(N.arange(64)<32,out=storage)
                for task in N.Dataflow({count}):
                    with N.Vector(),N.Stage(0):
                        inner=N.materialize(N.arange(64)>32,out=storage)
                with N.Vector():
                    N.fill(x,0.0,mask=condition,scratch=scratch)
    """)
    source = parse(dump(definition.to_ir()))
    # The AST deliberately does not export loop-local names. Explicit source
    # IR may select the yielded last value; retain that distinct contract.
    loop = next(op for op in source.walk() if isinstance(op, IterationOp))
    descriptor = next(value for value in loop.values if isinstance(value.type, MaskType))
    dependency = next(value for value in loop.values if not isinstance(value.type, MaskType))
    masked = next(op for op in source.walk() if isinstance(op, ElementOp) and op.mask is not None)
    replacements = {masked.mask: descriptor, masked.mask_dependency: dependency}
    masked.operands = [replacements.get(value, value) for value in masked.operands]
    source.verify()
    if count == 0:
        with pytest.raises(CompileError):
            synchronize_structure(source)
    else:
        check_compiled_events(synchronize_structure(source), invocations=2)


@pytest.mark.parametrize("same_storage", [False, True])
def test_exported_predicate_rejects_substituting_its_writer(same_storage):
    destination = "storage" if same_storage else "other_storage"
    definition = N.from_source(f"""
        def forged():
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,),N.f32)
                scratch=N.alloc_ub((64,),N.f32)
                storage=N.alloc_mask((64,))
                other_storage=N.alloc_mask((64,))
                with N.Vector():
                    condition=N.materialize(N.arange(64)<32,out=storage)
                    other=N.materialize(N.arange(64)>32,out={destination})
                with N.Vector():
                    N.fill(x,1.0)
                    N.fill(x,0.0,mask=condition,scratch=scratch)
    """)
    source = parse(dump(definition.to_ir()))
    masked = next(op for op in source.walk() if isinstance(op, ElementOp) and op.mask is not None)
    scope = masked.mask_dependency.owner
    assert isinstance(scope, DomainOp)
    original = scope.body.block.last_op.values[masked.mask_dependency.index]
    other = next(
        op.value
        for op in scope.walk()
        if isinstance(op, MaterializeMaskOp) and op.value is not original
    )
    values = list(scope.body.block.last_op.values)
    values[masked.mask_dependency.index] = other
    scope.body.block.last_op.operands = values
    message = "same writer" if same_storage else "storage view does not belong to its writer"
    with pytest.raises(VerifyException, match=message):
        source.verify()

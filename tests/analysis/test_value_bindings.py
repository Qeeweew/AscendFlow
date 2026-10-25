"""Pure provenance contracts using actual source and native-operation anchors.

Recipes below are mock associations to real TaskOps, not native/access/value
admission. These tests establish collection/sharing and intentionally retain
undefined or stale obligations for the later admission phase.
"""

from dataclasses import FrozenInstanceError, replace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp
from xdsl.ir import Block, Region

from ascendflow import language as N
from ascendflow.analysis import value_bindings
from ascendflow.analysis.value_bindings import (
    LastTile,
    TileDefinition,
    ValueScope,
    bind_values,
)
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.physical import BufferOp, EndOp, TaskOp, task_attrs
from ascendflow.dialects.source import CopyOp, DomainOp, ElementOp, IterationOp, StageOp
from ascendflow.dialects.structured import PipelineLoopOp
from ascendflow.dialects.types import TaskType, ref_type
from ascendflow.passes.native_recipes import NativeRead, RecipeResult

STAGED = """
def staged(X:N.Tensor((3,64),N.f32)):
    with N.Kernel():
        x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
        for g in N.Pipelined(3):
            with N.Vector(), N.Stage(0):
                a=N.copy(X[g,:],x)
            with N.Vector(), N.Stage(1):
                b=N.mul(a,2.0,out=y)
"""


def source(text):
    return N.from_source(text).to_ir()


def anchors(module):
    """Mock recipe results have real writer/reader TaskOp and ref positions."""
    from xdsl.dialects.builtin import StringAttr

    buffers = tuple(
        BufferOp.create(
            result_types=[ref_type("UB", "f32")],
            attributes={
                "sym_name": StringAttr(f"b{i}"),
                "nbytes": IntAttr(256),
                "external": IntAttr(0),
                "initialized": IntAttr(1),
            },
        )
        for i in range(3)
    )
    native_module = ModuleOp(buffers)
    views = tuple(buffer.ref for buffer in buffers)
    recipes = {}
    for operation in module.walk():
        if not isinstance(operation, (CopyOp, ElementOp)):
            continue
        task = TaskOp.create(
            operands=views,
            result_types=[TaskType()],
            attributes=task_attrs(f"anchor_{len(recipes)}", "add", 0),
        )
        native_module.body.block.add_op(task)
        count = 1 if isinstance(operation, CopyOp) else len(operation.inputs)
        recipes[operation] = RecipeResult(
            (task,), task, 2, tuple(NativeRead(i, task, i) for i in range(count))
        )
    loops = {}
    for operation in module.walk():
        if not isinstance(operation, IterationOp):
            continue
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        count = arith.ConstantOp.from_int_and_width(operation.count.data, IndexType())
        one = arith.ConstantOp.from_int_and_width(1, IndexType())
        loop = PipelineLoopOp.build(
            operands=[count.result],
            regions=[Region(Block([EndOp()]))],
        )
        ModuleOp([zero, count, one, loop])
        loops[operation] = loop
    return recipes, loops


def test_stage_domain_yields_forward_one_definition_and_keep_real_source_use():
    module = source(STAGED)
    recipes, loops = anchors(module)
    values, uses = bind_values(module, recipes, loops)
    writer = next(op for op in recipes if isinstance(op, CopyOp))
    reader = next(op for op in recipes if isinstance(op, ElementOp))
    binding = values[writer.value]
    assert isinstance(binding, TileDefinition)
    assert binding.operation is recipes[writer].definition and binding.operand == 2
    stage, domain = writer.parent_op(), writer.parent_op().parent_op()
    assert isinstance(domain, DomainOp) and isinstance(stage, StageOp)
    assert values[domain.values[0]] is values[stage.values[0]] is binding
    (use,) = uses
    assert use.expected is binding
    assert use.source is reader.dependencies[0] is domain.values[0]
    assert use.source_input == 0
    assert use.reader is recipes[reader].definition and use.operand == 0
    pipeline = next(op for op in module.walk() if isinstance(op, IterationOp))
    assert values[pipeline.values[0]].loop is loops[pipeline]
    assert values[pipeline.values[0]].value is binding


def test_multiple_inputs_and_native_readers_keep_distinct_use_contracts():
    module = source("""
        def inputs(X:N.Tensor((1,64),N.f32)):
            with N.Kernel():
                x,y,z=N.alloc_ub((64,)),N.alloc_ub((64,)),N.alloc_ub((64,))
                for g in N.Pipelined(1):
                    left=N.copy(X[0,:],x)
                    right=N.copy(X[0,:],y)
                    N.add(left,right,out=z)
""")
    recipes, loops = anchors(module)
    consumer = next(op for op in recipes if isinstance(op, ElementOp))
    recipe = recipes[consumer]
    other = next(result.definition for op, result in recipes.items() if isinstance(op, CopyOp))
    recipes[consumer] = replace(
        recipe,
        readers=(*recipe.readers, NativeRead(0, other, 1)),
    )
    values, uses = bind_values(module, recipes, loops)
    assert [(u.source_input, u.reader, u.operand) for u in uses] == [
        (0, recipe.definition, 0),
        (0, other, 1),
        (1, recipe.definition, 1),
    ]
    assert uses[0].expected is uses[1].expected is values[consumer.dependencies[0]]
    assert uses[2].expected is values[consumer.dependencies[1]]
    assert uses[0].expected is not uses[2].expected


@pytest.mark.parametrize("count", [0, 1, 1_000_000])
def test_nested_last_selections_preserve_each_actual_loop_without_readiness(count):
    module = source(f"""
        def nested():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in N.Pipelined(3):
                    for h in N.Pipelined({count}):
                        N.fill(x,0.0)
""")
    recipes, loops = anchors(module)
    values, uses = bind_values(module, recipes, loops)
    outer, inner = tuple(loops)
    result = values[outer.values[0]]
    assert isinstance(result, LastTile) and result.loop is loops[outer]
    assert isinstance(result.value, LastTile) and result.value.loop is loops[inner]
    definition = next(iter(recipes)).value
    assert result.value.value is values[definition]
    assert result.value is values[inner.values[0]]
    assert not uses and len(values) == 3  # no runtime count expansion


def test_real_use_of_empty_pipeline_result_remains_an_unresolved_obligation():
    module = source("""
        def empty(X:N.Tensor((1,64),N.f32),Y:N.Tensor((1,64),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in N.Pipelined(0):
                    N.copy(X[0,:],x)
                for h in N.Pipelined(1):
                    N.copy(x,Y[0,:])
""")
    first, second = tuple(op for op in module.walk() if isinstance(op, IterationOp))
    consumer = next(op for op in second.walk() if isinstance(op, CopyOp))
    replacement = CopyOp.build(
        operands=[consumer.source, consumer.destination, [first.values[0]], None],
        result_types=[consumer.value.type],
        attributes={"dependency_inputs": ArrayAttr([IntAttr(0)])},
    )
    consumer.parent.insert_op_before(replacement, consumer)
    consumer.value.replace_all_uses_with(replacement.value)
    consumer.parent.erase_op(consumer)
    module.verify()
    recipes, loops = anchors(module)
    values, uses = bind_values(module, recipes, loops)
    (use,) = uses
    assert isinstance(use.expected, LastTile)
    assert use.expected is values[first.values[0]]
    assert use.expected.loop is loops[first]
    assert use.source is first.values[0] and use.source_input == 0
    assert first.count.data == 0  # collection did not invent a fallback definition


def test_stale_tile_is_not_rebound_to_the_later_storage_writer():
    module = source("""
        def stale(X:N.Tensor((1,64),N.f32)):
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for g in N.Pipelined(1):
                    old=N.copy(X[0,:],x)
                    changed=N.mul(old,2.0,out=x)
                    N.add(old,1.0,out=y)
""")
    recipes, loops = anchors(module)
    values, uses = bind_values(module, recipes, loops)
    original = next(op for op in recipes if isinstance(op, CopyOp))
    overwrite = next(op for op in recipes if isinstance(op, ElementOp))
    assert len(uses) == 2
    assert uses[0].expected is uses[1].expected is values[original.value]
    assert values[overwrite.value] is not uses[1].expected


def test_memo_constructs_each_definition_and_pipeline_selection_once(monkeypatch):
    module = source(STAGED)
    recipes, loops = anchors(module)
    definitions, selections = [], []
    definition_type, last_type = TileDefinition, LastTile

    def definition(*args):
        result = definition_type(*args)
        definitions.append(result)
        return result

    def last(*args):
        result = last_type(*args)
        selections.append(result)
        return result

    monkeypatch.setattr(value_bindings, "TileDefinition", definition)
    monkeypatch.setattr(value_bindings, "LastTile", last)
    values, uses = bind_values(module, recipes, loops)
    assert len(definitions) == len(recipes) == 2
    assert len(selections) == sum(len(loop.values) for loop in loops) == 2
    assert len(values) > len(definitions) + len(selections) and len(uses) == 1


@pytest.mark.parametrize("missing", ["writer", "reader", "loop"])
def test_missing_real_anchors_are_diagnosed_without_a_completion_query(missing):
    module = source(STAGED)
    recipes, loops = anchors(module)
    writer = next(op for op in recipes if isinstance(op, CopyOp))
    reader = next(op for op in recipes if isinstance(op, ElementOp))
    if missing == "writer":
        recipes[writer] = replace(recipes[writer], definition_ref=None)
    elif missing == "reader":
        recipes[reader] = replace(recipes[reader], readers=())
    else:
        loops.clear()
    with pytest.raises(CompileError, match="logical Tile"):
        bind_values(module, recipes, loops)


def test_cyclic_yield_provenance_has_an_explicit_diagnostic():
    module = source(STAGED)
    recipes, loops = anchors(module)
    stage = next(op for op in module.walk() if isinstance(op, StageOp))
    stage.body.block.last_op.operands = [stage.values[0]]
    with pytest.raises(CompileError, match="cyclic logical Tile yield provenance") as error:
        bind_values(module, recipes, loops)
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_value_scope_is_a_shared_immutable_actual_scope_link():
    module = source(STAGED)
    stage = next(op for op in module.walk() if isinstance(op, StageOp))
    parent = ValueScope(None, stage.parent_op())
    left, right = ValueScope(parent, stage, 0), ValueScope(parent, stage, 1)
    assert left.parent is right.parent is parent
    assert left.operation is right.operation is stage
    assert left.branch == 0 and right.branch == 1
    assert ValueScope(parent, stage, 0) is not left
    with pytest.raises(FrozenInstanceError):
        left.parent = None

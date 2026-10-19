"""Actual DSL -> retained memory -> selected/leased physical instructions."""

from collections import Counter

import pytest

from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import (
    ActionGuard,
    Comparison,
    Placement,
    SyncAction,
    SyncKind,
)
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import Expr, expression, literal
from ascendflow.target.ascend910b import Actor, Engine
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale


@pytest.fixture(scope="module", params=[scale, batched_matmul])
def selected(request):
    # Consumers below only read this owned result; parsed snapshots are fresh.
    # Runtime-count cases must not each rebuild the same symbolic program.
    return synchronize_structure(request.param)


def test_actual_local_pipeline_has_one_owned_structure(selected):
    selected.selection.require_closed()
    assert selected.selection.dispatch.memory is selected.admission.memory
    assert len(selected.selection.coverage) == len(selected.dependencies.required)
    assert selected.resources.selection is selected.selection
    assert selected.resources.leases
    assert selected.resources.order.root.points
    assert {action.kind for action in selected.resources.actions} == {
        SyncKind.LOCAL_SET,
        SyncKind.LOCAL_WAIT,
    }


def test_actual_selected_actions_materialize_as_guarded_ssa_ir(selected):
    from ascendflow.dialects.io import dump, parse
    from ascendflow.dialects.structured import NotificationOp

    memory = selected.admission.memory
    # Logical analysis retains its module; terminal native correspondence is
    # established once before scalar/control rewriting of the terminal module.
    original_operations = tuple(native.operation for native in selected.executable.natives)
    module = selected.module
    assert memory.module is selected.structure.module and module is not memory.module
    assert all(
        old.operation is not new.operation and old.operation.uid == new.operation.uid
        for old, new in zip(memory.templates, selected.executable.natives, strict=True)
    )
    module.verify()
    text = dump(module)
    parse(text).verify()
    assert (
        0
        < len([op for op in module.walk() if isinstance(op, NotificationOp)])
        <= len(selected.resources.actions)
    )
    assert "npu.notification" in text
    assert "npu.notification" not in selected.before_ir
    parse(selected.before_ir).verify()
    assert dump(memory.module) != selected.before_ir
    retained = set(original_operations)
    assert {op for op in module.walk() if op in retained} == retained
    assert not any(op.name == "npu.pipeline_loop" for op in module.walk())
    assert any("npu.pipeline_clock" in op.attributes for op in module.walk())


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 12])
def test_emitted_domains_have_exactly_paired_ids_at_start_and_end(selected, count):
    logical = next(iter(selected.admission.memory.control.loops.values()))
    events = {SyncKind.LOCAL_SET: Counter(), SyncKind.LOCAL_WAIT: Counter()}
    for action in selected.resources.actions:
        for generation in range(count):
            values = {"tiles": count, logical.ordinal.args[0]: generation}
            if action.guard.evaluate(values):
                flag = action.event.evaluate(values)
                assert 0 <= flag < 8
                for protocol in action.protocols:
                    events[action.kind][protocol, flag] += 1
    assert events[SyncKind.LOCAL_SET] == events[SyncKind.LOCAL_WAIT]


def test_physical_order_is_projected_without_reordering_the_memory_templates():
    structure = lower_native_structure(batched_matmul)
    memory = collect_native_memory(structure.module)
    dispatch = bind_structured_dispatch(memory)
    body = next(body for body in dispatch.bodies.values() if body)
    assert memory.templates[0].spec.native == "nd2nz"
    assert body[0].native == "fix"  # The expert export-first issue policy.
    assert body[0].template is memory.templates[-1]
    assert len(body) == len(memory.templates) == 6


def test_guarded_last_id_does_not_evaluate_empty_loop_remainder():
    structure = lower_native_structure(scale)
    task = collect_native_memory(structure.module).templates[0].operation
    count = Expr("argument", ("N",))
    last_id = expression("rem", expression("sub", count, literal(1)), literal(2))
    action = SyncAction(
        task,
        Placement.AFTER,
        (Actor.AIV0,),
        SyncKind.LOCAL_SET,
        Engine.MTE2,
        Engine.V,
        last_id,
        ActionGuard((Comparison(count, ">", literal(0)),)),
    )
    assert action.active_event({"N": 0}) is None
    assert action.active_event({"N": 1}) == 0
    assert action.active_event({"N": 2}) == 1


@pytest.mark.parametrize("entry", [scale, batched_matmul])
def test_public_source_compile_uses_the_recursive_owned_path(entry, monkeypatch):
    from ascendflow import compile

    result = compile(entry, debug_verify=True)
    assert result.report["pipeline"] == "structured-regions"
    assert "npu.notification" not in result.before_ir
    assert "npu.notification" in result.plan_ir
    assert "PIPE_ALL" not in result.asc
    assert (
        result.report["analysis_work"]["emission"]["native_prints"]
        == result.report["native_templates"]
    )
    assert result.explain()
    result.plan.verify()

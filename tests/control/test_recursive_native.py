"""Native control/address semantics, before memory/completion admission.

The shared models/native_control interpreter executes scalar control and records native
dispatch. It is not an asynchronous hardware oracle or a synchronization proof.
"""

import pytest
from xdsl.dialects import scf

from ascendflow import language as N
from ascendflow.analysis.value_bindings import LastTile, TileDefinition
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.physical import (
    TaskOp,
)
from ascendflow.dialects.source import CopyOp, IterationOp
from ascendflow.dialects.structured import PipelineLoopOp, PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.control import NESTED
from tests.models.native_control import dispatch


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7])
def test_real_scale_rebinds_shared_slot_expression_per_stage(count):
    plan = lower_native_structure(scale.specialize(B=64))
    assert len(plan.recipes) == 3 and plan.work["expanded_iterations"] == 0
    _, trace = dispatch(plan, tiles=count)
    expected = []
    for t in range(count):
        slot = (t % 2) * 256
        expected.extend(
            [
                (t, 0, ("copy", 0.0, (("X", t * 256), ("inputs", slot)))),
                (t + 1, 1, ("muls", 2.0, (("inputs", slot), ("outputs", slot)))),
                (t + 2, 2, ("copy", 0.0, (("outputs", slot), ("Y", t * 256)))),
            ]
        )
    assert trace == [entry[2] for entry in sorted(expected)]
    assert dump(parse(dump(plan.module))) == dump(plan.module)


@pytest.mark.parametrize("n,k", [(0, 0), (1, 3), (2, 0), (2, 1), (5, 3)])
def test_recursive_native_control_keeps_ordinary_work_and_all_ancestor_addresses(n, k):
    plan = lower_native_structure(N.from_source(NESTED))
    _, trace = dispatch(plan, n=n, k=k)
    expected = [("fill", 0.0, (("x", 0),))]
    for i in range(1, n, 2):
        expected.append(("fill", 1.0, (("x", 0),)))
        for j in range(k):
            expected.append(("fill", 2.0, (("x", 0),)))
            for t in range(3):
                expected.append(
                    ("copy", 0.0, (("X", ((i * k + j) * 3 + t) * 256), ("x", (t % 2) * 256)))
                )
            expected.append(("fill", 3.0, (("x", 0),)))
        expected.append(("fill", 4.0, (("x", 0),)))
    expected.append(("fill", 5.0, (("x", 0),)))
    assert trace == expected
    assert len(plan.loops) == 3 and len(plan.recipes) == 7
    assert sum(isinstance(op, TaskOp) for op in plan.module.walk()) == 7


def test_large_count_retains_identical_native_and_control_work():
    plans = [
        lower_native_structure(N.from_source(NESTED).specialize(PARTS=count))
        for count in (1024, 1_048_576)
    ]
    assert plans[0].work == plans[1].work
    assert len(tuple(plans[0].module.walk())) == len(tuple(plans[1].module.walk()))


@pytest.mark.parametrize("n", [0, 1, 3, 8])
def test_serial_carry_and_if_results_keep_zero_iteration_initial_value(n):
    source = N.from_source("""
        def carry(n:N.int64):
            with N.Kernel():
                total=7
                for i in N.serial(1,n,2):
                    if i<3:
                        total=total+i
                    else:
                        total=total+2
                for j in N.Pipelined(total):
                    pass
""")
    plan = lower_native_structure(source)
    values, _ = dispatch(plan, n=n)
    serial = next(op for op in plan.module.walk() if isinstance(op, scf.ForOp))
    assert values[serial.results[0]] == 7 + sum(i if i < 3 else 2 for i in range(1, n, 2))
    assert len(serial.iter_args) == len(serial.results) == 1


def test_cube_keeps_expert_issue_order_and_actual_tile_writers_readers():
    plan = lower_native_structure(batched_matmul)
    loop = next(op for op in plan.module.walk() if isinstance(op, PipelineLoopOp))
    assert [
        stage.label.data for stage in loop.body.block.ops if isinstance(stage, PipelineStageOp)
    ] == ["load", "operands", "compute", "export"]
    assert [name.data for name in loop.issue_order.data["cube"]] == [
        "export",
        "load",
        "operands",
        "compute",
    ]
    assert len(plan.recipes) == 6
    assert all(use.expected.operation in tuple(plan.module.walk()) for use in plan.uses)
    assert all(isinstance(use.expected, TileDefinition) for use in plan.uses)
    assert len(plan.uses) == 5
    assert not any(op.name.startswith("npu_source.") for op in plan.module.walk())
    _, trace = dispatch(plan, tiles=5)
    instructions = (
        (0, 1, ("nd2nz", "nd2nz")),
        (1, 2, ("load2d_a", "load2d_b_nt")),
        (2, 3, ("mad",)),
        (3, 0, ("fix",)),
    )
    expected = sorted(
        (g + stage, order, inside, name)
        for g in range(5)
        for stage, order, names in instructions
        for inside, name in enumerate(names)
    )
    assert [item[0] for item in trace] == [item[3] for item in expected]


def test_empty_pipeline_exports_last_selection_without_inventing_completion():
    source = N.from_source("""
        def empty():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for t in N.Pipelined(0):
                    value=N.fill(x,1.0)
""")
    plan = lower_native_structure(source)
    pipeline = next(op for op in plan.source.walk() if isinstance(op, IterationOp))
    assert pipeline.values
    assert all(isinstance(plan.values[value], LastTile) for value in pipeline.values)
    assert not plan.uses
    assert dispatch(plan)[1] == []


def test_stale_tile_remains_an_explicit_unresolved_writer_requirement():
    source = N.from_source("""
        def stale(X:N.Tensor((n,64),N.f32),n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                y=N.alloc_ub((64,))
                for t in N.Pipelined(n):
                    old=N.copy(X[t,:],x)
                    N.mul(old,2.0,out=x)
                    N.add(old,1.0,out=y)
""")
    plan = lower_native_structure(source)
    copy = next(recipe for op, recipe in plan.recipes.items() if isinstance(op, CopyOp))
    assert len(plan.uses) == 2
    assert all(use.expected.operation is copy.definition for use in plan.uses)
    assert not hasattr(plan, "asc")


def test_cross_stage_scalar_capture_is_not_silently_cloned_or_treated_as_complete():
    source = N.from_source("""
        def scalar(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                y=N.alloc_ub((64,))
                for t in N.Pipelined(n):
                    with N.Stage(0,name="read"):
                        factor=N.load_scalar(x[0:1])
                    with N.Stage(1,name="use"):
                        N.mul(x,factor,out=y)
""")
    source.to_ir().verify()
    with pytest.raises(CompileError, match="asynchronous values cannot be cloned") as error:
        lower_native_structure(source)
    assert error.value.kind == ErrorKind.UNSUPPORTED

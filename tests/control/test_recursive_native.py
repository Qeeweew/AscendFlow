"""Native control/address semantics, before memory/completion admission.

The finite interpreter below executes only scalar control and records native
dispatch. It is not an asynchronous hardware oracle or a synchronization proof.
"""

import operator

import pytest
from xdsl.dialects import arith, scf

from ascendflow import language as N
from ascendflow.analysis.value_bindings import LastTile, TileDefinition
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.physical import (
    BufferOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    RuntimeSizeOp,
    TaskOp,
)
from ascendflow.dialects.source import CopyOp, PipelineOp
from ascendflow.dialects.structured import ActorDomainOp, AddressOp, PipelineLoopOp, PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale


def interpret(block, values, arguments, trace):
    """Independent scalar evaluation; plain clock enumeration only in tests."""
    for op in block.ops:
        if isinstance(op, FixConfigOp):
            result = ()  # Register setup is outside this scalar/address oracle.
        elif isinstance(op, RuntimeSizeOp):
            result = (arguments[op.sym_name.data],)
        elif isinstance(op, CoreIdOp):
            result = (arguments.get("group", 0),)
        elif isinstance(op, BufferOp):
            result = (op,)
        elif isinstance(op, arith.ConstantOp):
            result = (op.value.value.data,)
        elif isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp)):
            a, b = (values[operand] for operand in op.operands)
            result = (
                {
                    arith.AddiOp: operator.add,
                    arith.SubiOp: operator.sub,
                    arith.MuliOp: operator.mul,
                    arith.RemUIOp: operator.mod,
                }[type(op)](a, b),
            )
        elif isinstance(op, arith.CmpiOp):
            a, b = (values[operand] for operand in op.operands)
            result = (
                {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[
                    op.predicate.value.data
                ],
            )
        elif isinstance(op, scf.ForOp):
            state = tuple(values[value] for value in op.iter_args)
            for iv in range(values[op.lb], values[op.ub], values[op.step]):
                local = values | dict(zip(op.body.block.args, (iv, *state), strict=True))
                state = interpret(op.body.block, local, arguments, trace)
            result = state
        elif isinstance(op, scf.IfOp):
            region = op.true_region if values[op.cond] else op.false_region
            result = interpret(region.block, dict(values), arguments, trace)
        elif isinstance(op, scf.YieldOp):
            return tuple(values[value] for value in op.arguments)
        elif isinstance(op, PipelineLoopOp):
            lower, upper, stride = (values[value] for value in op.operands)
            iterations = list(range(lower, upper, stride))
            stages = [stage for stage in op.body.block.ops if isinstance(stage, PipelineStageOp)]
            if (
                op.issue_order is not None
                and (policy := op.issue_order.data.get(arguments.get("actor"))) is not None
            ):
                by_name = {stage.label.data: stage for stage in stages}
                stages = [by_name[label.data] for label in policy]
            for clock in range(len(iterations) + op.schedule.horizon.data):
                for stage in stages:
                    g = clock - stage.offset.data
                    if 0 <= g < len(iterations):
                        local = values | dict(
                            zip(stage.body.block.args, (g, iterations[g]), strict=True)
                        )
                        interpret(stage.body.block, local, arguments, trace)
            result = ()
        elif isinstance(op, ActorDomainOp):
            if op.kind.data == arguments.get("actor"):
                owner = arguments.get("sid", 0) if op.kind.data == "vector" else 0
                interpret(op.body.block, values | {op.body.block.args[0]: owner}, arguments, trace)
            result = ()
        elif isinstance(op, AddressOp):
            result = ((values[op.source].sym_name.data, values[op.offset]),)
        elif isinstance(op, TaskOp):
            refs = tuple(values[value] for value in op.refs)
            trace.append((op.native.data, op.scalar.value.data, refs))
            result = (op,)
        elif isinstance(op, EndOp):
            return ()
        else:
            raise TypeError(f"unsupported oracle instruction {op.name}")
        values.update(zip(op.results, result, strict=True))
    return ()


def dispatch(plan, **arguments):
    values, trace = {}, []
    root = plan.module.body.block.first_op
    if "actor" not in arguments and hasattr(root, "core"):
        arguments["actor"] = "cube" if root.core.data == "aic" else "vector"
    interpret(root.body.block, values, arguments, trace)
    return values, trace


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


NESTED = """
def nested(X: N.Tensor((n,k,PARTS,64),N.f32), n: N.int64, k: N.int64,
           PARTS: N.constexpr=3):
    with N.Kernel():
        x=N.alloc_ub((2,64),name="x")
        N.fill(x[0,:],0.0)
        for i in N.serial(1,n,2):
            N.fill(x[0,:],1.0)
            for j in N.Pipelined(k,stages=3):
                with N.Stage(2,name="children"):
                    N.fill(x[0,:],2.0)
                    for t in N.Pipelined(PARTS,stages=2):
                        with N.Stage(1,name="load"):
                            N.copy(X[i,j,t,:],x[t%2,:])
                    N.fill(x[0,:],3.0)
            N.fill(x[0,:],4.0)
        N.fill(x[0,:],5.0)
"""


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
                for j in N.Pipelined(total,stages=1):
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
                for t in N.Pipelined(0,stages=1):
                    value=N.fill(x,1.0)
    """)
    plan = lower_native_structure(source)
    pipeline = next(op for op in plan.source.walk() if isinstance(op, PipelineOp))
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
                for t in N.Pipelined(n,stages=1):
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
                for t in N.Pipelined(n,stages=2):
                    with N.Stage(0,name="read"):
                        factor=N.load_scalar(x[0:1])
                    with N.Stage(1,name="use"):
                        N.mul(x,factor,out=y)
    """)
    source.to_ir().verify()
    with pytest.raises(CompileError, match="asynchronous values cannot be cloned") as error:
        lower_native_structure(source)
    assert error.value.kind == ErrorKind.UNSUPPORTED

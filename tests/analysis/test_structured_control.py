"""Actual stage/serial/participant bindings used by structured address facts."""

import pytest
from xdsl.dialects import scf

from ascendflow import language as N
from ascendflow.analysis.structured_addresses import StructuredAddresses
from ascendflow.analysis.structured_control import _division_iteration_range, build_control_facts
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import ActorDomainOp, AddressOp, PipelineLoopOp, PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr, Fact, expression, literal
from ascendflow.target.ascend910b import Actor
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.control.test_shared_aiv import SHARED


@pytest.mark.parametrize(
    "source", [scale, batched_matmul, N.from_source(SHARED, disjoint_external=True)]
)
def test_real_addresses_use_their_actual_scope_ranges_once(source):
    plan = lower_native_structure(source)
    before = tuple(plan.module.walk())
    control = build_control_facts(plan.module)
    addresses = StructuredAddresses(control.environment)
    for op in before:
        if not isinstance(op, AddressOp):
            continue
        facts = addresses.decode(op.ref)
        addresses.check_axes(facts, control.frames[op].domains())
        assert addresses.decode(op.ref) is facts
        assert facts.root is op.source
    assert control.operations_visited == len(before) - 1  # module is an ownership container
    assert addresses.decode_count == sum(isinstance(op, AddressOp) for op in before)
    assert tuple(plan.module.walk()) == before


def test_same_loop_stages_share_logical_ordinal_without_claiming_same_dispatch_step():
    plan = lower_native_structure(scale)
    control = build_control_facts(plan.module)
    loop = next(op for op in plan.module.walk() if isinstance(op, PipelineLoopOp))
    logical = control.loops[loop]
    for stage in loop.body.block.ops:
        if isinstance(stage, PipelineStageOp):
            assert control.environment.expression(stage.body.block.args[0]) is logical.ordinal
            assert len(stage.body.block.args) == 1
    assert [s.offset.data for s in loop.body.block.ops if isinstance(s, PipelineStageOp)] == [
        0,
        1,
        2,
    ]


def test_distinct_loop_operations_never_share_a_clock_by_label_or_bound():
    source = N.from_source("""
        def sibling(n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for a in N.Pipelined(n,stages=1):
                    N.fill(x,1.0)
                for b in N.Pipelined(n,stages=1):
                    N.fill(x,2.0)
    """)
    plan = lower_native_structure(source)
    first, second = build_control_facts(plan.module).loops.values()
    assert first.operation is not second.operation
    assert first.ordinal != second.ordinal
    assert first.count == second.count


@pytest.mark.parametrize("lower,upper,step", [(3, 10, 2), (-2, 4, 3), (3, 0, 2)])
def test_nonzero_serial_induction_and_empty_count_remain_exact(lower, upper, step):
    source = N.from_source(f"""
        def serial():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial({lower},{upper},{step}):
                    N.fill(x,1.0)
    """)
    plan = lower_native_structure(source)
    control = build_control_facts(plan.module)
    (loop,) = control.loops.values()
    assert isinstance(loop.operation, scf.ForOp)
    assert loop.count.constant == len(range(lower, upper, step))
    assert [
        loop.induction.evaluate({loop.ordinal.args[0]: g}) for g in range(loop.count.constant)
    ] == list(range(lower, upper, step))
    task = next(op for op in plan.module.walk() if isinstance(op, TaskOp))
    assert control.frames[task].statically_empty == (lower >= upper)


def test_shared_vector_domain_has_two_participants_but_one_symbolic_sid():
    source = N.from_source("""
        def lanes(X:N.Tensor((2,64),N.f32)):
            with N.Kernel(kind="mixed"):
                x=N.alloc_ub((64,))
                with N.Vector() as left:
                    N.copy(X[left,:],x)
                with N.Vector() as right:
                    N.copy(X[right,:],x)
    """)
    plan = lower_native_structure(source)
    control = build_control_facts(plan.module)
    domains = [op for op in plan.module.walk() if isinstance(op, ActorDomainOp)]
    assert len(domains) == 2
    assert domains[0].body.block.args[0] is not domains[1].body.block.args[0]
    for domain in domains:
        assert control.environment.expression(domain.body.block.args[0]) is control.sid
        assert control.frames[domain].participants == (Actor.AIV0, Actor.AIV1)
    assert control.sid != control.group


def test_scalar_carry_retains_unknown_range_instead_of_an_assumed_nonnegative_value():
    source = N.from_source("""
        def carry(n:N.int64):
            with N.Kernel():
                count=0
                for i in N.serial(n):
                    count=count-1
                for j in N.Pipelined(count,stages=1):
                    pass
    """)
    control = build_control_facts(lower_native_structure(source).module)
    serial, pipeline = control.loops.values()
    carry = control.environment.expression(serial.operation.results[0])
    assert carry.kind == "argument" and carry.args[0] not in control.environment.facts
    assert pipeline.upper is carry


@pytest.mark.parametrize("division", ["ceildiv", "floordiv"])
def test_divided_loop_bound_keeps_joint_ancestor_bias_and_exact_active_range(division):
    n, owner, induction = (Expr("argument", (name,)) for name in ("n", "$owner", "$iv"))
    facts = {"n": Fact(), "$owner": Fact(maximum=6)}
    for divisor in (2, 3, 7):
        upper = expression(
            division, expression("max", literal(0), expression("sub", n, owner)), literal(divisor)
        )
        ((joint, low, high),) = _division_iteration_range(
            literal(0), upper, induction, [(owner, literal(0), literal(6))], facts
        )
        for size in range(19):
            for lane in range(7):
                values = {"n": size, "$owner": lane}
                for index in range(upper.evaluate(values)):
                    value = joint.evaluate(values | {"$iv": index})
                    assert low.evaluate(values) <= value <= high.evaluate(values)
                    assert value == lane + divisor * index
    assert _division_iteration_range(literal(-1), upper, induction, [], facts) == ()


@pytest.mark.parametrize("lower_bias,lower_divisor", [(1, 3), (2, 3), (1, 2)])
def test_divided_lower_bound_preserves_only_the_matching_joint_coordinate(
    lower_bias, lower_divisor
):
    end, owner, induction = (Expr("argument", (name,)) for name in ("end", "$owner", "$iv"))
    facts = {"begin": Fact(), "end": Fact(), "$owner": Fact(maximum=6)}
    for start in range(12):
        lower = expression(
            "ceildiv",
            expression(
                "max",
                literal(0),
                expression("sub", literal(start), expression("mul", literal(lower_bias), owner)),
            ),
            literal(lower_divisor),
        )
        upper = expression(
            "ceildiv", expression("max", literal(0), expression("sub", end, owner)), literal(3)
        )
        ((joint, low, high),) = _division_iteration_range(
            lower, upper, induction, [(owner, literal(0), literal(6))], facts
        )
        for stop in range(start, 19):
            for lane in range(7):
                values = {"begin": start, "end": stop, "$owner": lane}
                for index in range(lower.evaluate(values), upper.evaluate(values)):
                    actual = joint.evaluate(values | {"$iv": index})
                    assert low.evaluate(values) <= actual <= high.evaluate(values)
                    if lower_bias == 1 and lower_divisor == 3:
                        assert low.evaluate(values) == start


def test_complete_tiles_use_floor_bound_without_requiring_length_multiple():
    from ascendflow.analysis.structured_bounds import admit_memory_bounds
    from ascendflow.analysis.structured_memory import collect_native_memory

    source = N.from_source("""
        def full_tiles(X:N.Tensor((n,),N.f32),n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(n//64,stages=1):
                    N.copy(X[i*64:(i+1)*64],x)
    """)
    memory = collect_native_memory(lower_native_structure(source).module)
    bounds = admit_memory_bounds(memory)
    assert bounds.checked_accesses == 2

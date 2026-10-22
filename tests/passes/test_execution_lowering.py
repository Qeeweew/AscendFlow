"""Terminal clock lowering preserves actual instructions and int64 boundaries."""

import itertools

from xdsl.dialects import arith, scf

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.local_schedule import bind_local_schedules
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.physical import BufferOp, CoreIdOp, EndOp, RuntimeSizeOp, TaskOp
from ascendflow.dialects.structured import ActorDomainOp, AddressOp, PipelineLoopOp
from ascendflow.passes.lower_execution import lower_execution
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.simplify_scalars import ScalarFacts, simplify_scalars
from ascendflow.target.ascend910b import Actor
from tests.fixtures.nested import nested_cv
from tests.models.structured_events import _Interpret, check_structured_events
from tests.models.terminal_trace import trace

NESTED = """
def clocks(n:N.i64,m:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
        for i in N.Pipelined(n,issue_order=("late","early")):
            with N.Stage(0,name="early"):
                N.fill(x,1.0)
            with N.Stage(3,name="late"):
                for j in N.Pipelined(m):
                    with N.Stage(2,name="inner"):
                        N.fill(x,2.0)
"""


def test_mixed_actors_keep_independent_horizons_and_nested_instruction_generations():
    memory = collect_native_memory(lower_native_structure(nested_cv).module)
    original = dump(memory.module)
    schedules = bind_local_schedules(memory)
    outer = next(op for op in memory.module.walk() if isinstance(op, PipelineLoopOp))
    assert schedules[outer, "cube"].horizon == 0
    assert schedules[outer, "vector"].horizon == 1
    native_ids = {id(template.operation) for template in memory.templates}
    facts = ScalarFacts.from_control(memory.control)
    lower_execution(memory.module, memory.templates, facts, schedules)
    simplify_scalars(memory.module, facts=facts)
    memory.module.verify()
    root = memory.module.body.block.first_op
    clocks = {
        domain.kind.data: next(op for op in domain.body.block.ops if isinstance(op, scf.ForOp))
        for domain in root.body.block.ops
        if isinstance(domain, ActorDomainOp)
    }
    assert {family: loop.lb.owner.value.value.data for family, loop in clocks.items()} == {
        "cube": 0,
        "vector": -1,
    }
    assert {id(op) for op in memory.module.walk() if isinstance(op, TaskOp)} == native_ids
    for count, actor in itertools.product((0, 1, 2, 3, 5), tuple(Actor)):
        assert trace(memory.module, {"n": count}, actor) == trace(
            parse(original), {"n": count}, actor
        )


def test_nested_short_empty_and_sparse_offsets_preserve_the_actor_instruction_trace():
    memory = collect_native_memory(lower_native_structure(N.from_source(NESTED)).module)
    original = dump(memory.module)
    native_ids = {id(template.operation) for template in memory.templates}
    facts = ScalarFacts.from_control(memory.control)
    lower_execution(memory.module, memory.templates, facts, bind_local_schedules(memory))
    simplify_scalars(memory.module, facts=facts)
    memory.module.verify()
    assert not any(isinstance(op, PipelineLoopOp) for op in memory.module.walk())
    assert {id(op) for op in memory.module.walk() if isinstance(op, TaskOp)} == native_ids
    assert sum("npu.pipeline_clock" in op.attributes for op in memory.module.walk()) == 2
    for n, m in itertools.product(range(5), range(4)):
        assert trace(memory.module, {"n": n, "m": m}) == trace(parse(original), {"n": n, "m": m})


def test_unbounded_count_clock_has_defined_arithmetic_at_int64_edges():
    memory = collect_native_memory(
        lower_native_structure(
            N.from_source("""
        def edges(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in N.Pipelined(n):
                    with N.Stage(0,name="early"):
                        if g==0:
                            N.fill(x,1.0)
                    with N.Stage(3,name="late"):
                        N.fill(x,2.0)
""")
        ).module
    )
    facts = ScalarFacts.from_control(memory.control)
    lower_execution(memory.module, memory.templates, facts, bind_local_schedules(memory))
    simplify_scalars(memory.module, facts=facts)
    memory.module.verify()
    root = memory.module.body.block.first_op
    loop = next(op for op in memory.module.walk() if isinstance(op, scf.ForOp))
    assert loop.lb.owner.value.value.data == -3
    maximum = (1 << 63) - 1

    def execute(block, values, calls):
        for op in block.ops:
            if isinstance(op, arith.ConstantOp):
                values[op.result] = op.value.value.data
            elif isinstance(op, RuntimeSizeOp):
                pass
            elif isinstance(op, CoreIdOp):
                values[op.value] = 0
            elif isinstance(op, ActorDomainOp):
                execute(op.body.block, values | {op.body.block.args[0]: 0}, calls)
            elif isinstance(op, scf.IfOp):
                region = op.true_region if values[op.cond] else op.false_region
                if region.blocks:
                    execute(region.block, dict(values), calls)
            elif isinstance(op, TaskOp):
                calls.append(op.scalar.value.data)
            elif isinstance(op, (BufferOp, AddressOp, EndOp, scf.YieldOp, scf.ForOp)):
                continue
            else:
                _Interpret.arithmetic(op, values)
                assert all(-(1 << 63) <= values[v] <= maximum for v in op.results)

    for count in (0, 1, 2, maximum - 1, maximum):
        values = {
            next(op for op in memory.module.walk() if isinstance(op, RuntimeSizeOp)).value: count
        }
        execute(root.body.block, values, [])
        for tick in sorted({-3, -2, -1, 0, count - 2, count - 1}):
            if not -3 <= tick < count:
                continue
            calls = []
            execute(loop.body.block, values | {loop.body.block.args[0]: tick}, calls)
            expected = []
            if tick + 3 == 0 and tick + 3 < count:
                expected.append(1.0)
            if 0 <= tick < count:
                expected.append(2.0)
            assert calls == expected


def test_public_static_int64_count_does_not_require_an_overflowing_count_plus_horizon():
    maximum = (1 << 63) - 1
    source = N.from_source(f"""
        def large():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in N.Pipelined({maximum}):
                    with N.Stage(3,name="work"):
                        N.fill(x,1.0)
""")
    compiled = compile(source)
    assert compiled.report["native_templates"] == 1
    assert compiled.report["analysis_work"]["emission"]["native_prints"] == 1
    loop = next(op for op in compiled.plan.walk() if isinstance(op, scf.ForOp))
    assert loop.lb.owner.value.value.data == -3
    assert loop.ub.owner.value.value.data == maximum


def test_cyclic_owner_range_simplifies_a_constant_quotient_on_the_public_path():
    source = N.from_source(
        """
        def grouped(X:N.Tensor((n,64),N.f32),n:N.i64(minimum=1)):
            with N.Kernel(7) as group:
                x=N.alloc_ub((64,))
                for local in N.serial(N.ceildiv(N.max(0,n-group),7)):
                    index=group+7*local
                    plane=index//n
                    for g in N.Pipelined(1):
                        N.copy(X[index+plane,:],x)
""",
        disjoint_external=True,
    )
    compiled = compile(source)
    original = collect_native_memory(parse(compiled.before_ir))
    final = collect_native_memory(compiled.plan)
    assert not any(isinstance(op, arith.DivUIOp) for op in final.module.walk())
    for n in (1, 6, 7, 8, 17):
        check_structured_events(final, {"n": n}, logical_reference=original, groups=tuple(range(7)))


def test_joint_quotient_bound_retains_a_uniform_divisor_with_rational_coefficients():
    source = N.from_source(
        """
        def grouped_radix(X:N.Tensor((n,64),N.f32),
                          n:N.i64(minimum=1,multiple_of=64)):
            with N.Kernel(7) as group:
                x=N.alloc_ub((64,))
                tiles=n//64
                for local in N.serial(N.ceildiv(N.max(0,8*tiles-group),7)):
                    index=group+7*local
                    head=(index//tiles)%8
                    N.copy(X[head*tiles+index%tiles,:],x)
    """,
        disjoint_external=True,
    )
    compiled = compile(source)
    original = collect_native_memory(parse(compiled.before_ir))
    final = collect_native_memory(compiled.plan)
    assert not any(
        isinstance(op, arith.RemUIOp)
        and isinstance(op.rhs.owner, arith.ConstantOp)
        and op.rhs.owner.value.value.data == 8
        for op in final.module.walk()
    )
    for n in (64, 128, 192):
        check_structured_events(final, {"n": n}, logical_reference=original, groups=tuple(range(7)))

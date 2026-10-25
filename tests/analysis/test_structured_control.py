"""Actual stage/serial/participant bindings used by structured address facts."""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.smt import IntegerQueryService, negate
from ascendflow.analysis.structured_addresses import StructuredAddresses, check_static_axes
from ascendflow.analysis.structured_control import _division_iteration_range, build_control_facts
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import ActorDomainOp, AddressOp, PipelineLoopOp, PipelineStageOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import Expr, Fact, expression, literal
from ascendflow.symbols import Symbol, SymbolKind
from ascendflow.target.ascend910b import Actor
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.control import SHARED


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
        check_static_axes(facts)
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
        def sibling(n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for a in N.Pipelined(n):
                    N.fill(x,1.0)
                for b in N.Pipelined(n):
                    N.fill(x,2.0)
""")
    plan = lower_native_structure(source)
    first, second = build_control_facts(plan.module).loops.values()
    assert first.operation is not second.operation
    assert first.ordinal != second.ordinal
    assert first.count == second.count


@pytest.mark.parametrize("index", ["t//2 if t%2==0 else (t+1)//2", "0 if t==0 else 16//t"])
def test_pure_integer_selection_keeps_value_across_actual_stage_copies(index):
    from ascendflow.scalar import substitute

    source = N.from_source(
        f"""
        def selected(X:N.Tensor((64,64),N.f32),Y:N.Tensor((64,64),N.f32,access="write"),n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,),slots=2)
                for t in N.Pipelined(n):
                    item={index}
                    with N.Stage(0):
                        N.copy(X[item,:],x.at(t))
                    with N.Stage(1):
                        N.copy(x.at(t),Y[item,:])
    """,
        disjoint_external=True,
    )
    structure = lower_native_structure(source)
    control = build_control_facts(structure.module)
    copies = [op for op in structure.module.walk() if isinstance(op, scf.IfOp) and op.results]
    assert len(copies) == 2
    first, second = (control.environment.expression(op.results[0]) for op in copies)
    assert first.args[0].kind is second.args[0].kind is SymbolKind.SELECTION
    assert first.args[0].origin is not second.args[0].origin
    queries = IntegerQueryService(control.environment.facts)
    assert queries.implies(Comparison(first, "==", second)).proved
    (loop,) = control.loops.values()
    later = substitute(second, {loop.ordinal: expression("add", loop.ordinal, literal(1))})
    assert queries.satisfies((Comparison(first, "!=", later),)).satisfied
    assert not control.invariant_over(first, loop.operation)
    # The partial division stays inside its selected region in both copies.
    if "16//t" in index:
        assert queries.implies(
            Comparison(first, "==", literal(0)), (Comparison(loop.ordinal, "==", literal(0)),)
        ).proved
        divisions = [op for op in structure.module.walk() if isinstance(op, arith.DivUIOp)]
        assert len(divisions) == 2 and all(op.parent_op() in copies for op in divisions)


def test_effectful_integer_selection_stays_an_actual_unknown_snapshot():
    source = N.from_source(
        """
        def effect(Meta:N.Tensor((n,),N.i64),n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for t in N.serial(n):
                    item=N.load_index(Meta[t:t+1]) if t%2==0 else 0
                    if item==0:
                        N.fill(x,1.0)
    """,
        disjoint_external=True,
    )
    structure = lower_native_structure(source)
    control = build_control_facts(structure.module)
    selected = next(op for op in structure.module.walk() if isinstance(op, scf.IfOp) and op.results)
    result = control.environment.expression(selected.results[0])
    assert result.args[0].kind is SymbolKind.CARRY


def test_scalar_prefix_keeps_load_and_branch_results_after_their_real_definitions():
    source = N.from_source(
        """
        def prefixes(Meta:N.Tensor((1,),N.i64),n:N.i64):
            with N.Kernel(kind="vector"):
                x=N.alloc_ub((64,),N.f32)
                value=N.load_index(Meta[:1],minimum=0,maximum=1)
                if value==0 or n==0:
                    N.fill(x,1.0)
    """,
        disjoint_external=True,
    )
    structure = lower_native_structure(source)
    control = build_control_facts(structure.module)
    loaded = next(
        op
        for op in structure.module.walk()
        if isinstance(op, TaskOp) and op.native.data == "index_load"
    )
    branch = next(op for op in structure.module.walk() if isinstance(op, scf.IfOp) and op.results)
    for operation, result in ((loaded, loaded.value), (branch, branch.results[0])):
        assert result not in control.dominating_scalars(operation.parent, operation)
        after = control.dominating_scalars(operation.parent, operation.next_op)
        assert result in after
        assert control.dominating_arguments(operation.parent, operation.next_op) == frozenset(
            name for value in after for name in control.environment.expression(value).variables
        )
    anchor = loaded.next_op
    before = control.dominating_scalars(loaded.parent, anchor)
    inserted = arith.ConstantOp.from_int_and_width(123, IndexType())
    loaded.parent.insert_op_before(inserted, anchor)
    assert control.dominating_scalars(loaded.parent, inserted) == before
    assert inserted.result not in control.dominating_scalars(loaded.parent, anchor)


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
        def carry(n:N.i64):
            with N.Kernel():
                count=0
                for i in N.serial(n):
                    count=count-1
                for j in N.Pipelined(count):
                    pass
""")
    control = build_control_facts(lower_native_structure(source).module)
    serial, pipeline = control.loops.values()
    carry = control.environment.expression(serial.operation.results[0])
    assert carry.kind == "argument" and carry.args[0] not in control.environment.facts
    assert pipeline.upper is carry


@pytest.mark.parametrize(
    "condition",
    [
        "n<split or n%2==0",
        "n>=split and n%2==0",
        "(n<split or n==0) and (split==0 or n%2==1)",
        "n==0 or split//n>0",
    ],
)
def test_boolean_results_keep_exact_short_circuit_integer_semantics(condition):
    source = N.from_source(f"""
        def booleans(n:N.i64,split:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                if {condition}:
                    N.fill(x,1.0)
    """)
    structure = lower_native_structure(source)
    control = build_control_facts(structure.module)
    branch = next(
        op for op in structure.module.walk() if isinstance(op, scf.IfOp) and not op.results
    )
    actual = Comparison(control.environment.expression(branch.cond), "!=", literal(0))
    queries = IntegerQueryService(control.environment.facts)
    inputs = {str(name): Expr("argument", (name,)) for name in control.environment.facts}
    for n in range(4):
        for split in range(4):
            values = {"n": n, "split": split}
            premises = tuple(
                Comparison(inputs[name], "==", literal(v)) for name, v in values.items()
            )
            expected = eval(condition, {"__builtins__": {}}, values)
            assert queries.implies(actual if expected else negate(actual), premises).proved


def test_boolean_invariance_retains_each_actual_loop_capture():
    source = N.from_source("""
        def booleans(n:N.i64,split:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for t in N.Pipelined(n):
                    if t<split or n==0:
                        N.fill(x,1.0)
    """)
    control = build_control_facts(lower_native_structure(source).module)
    (loop,) = control.loops.values()
    branch = next(op for op in control.branches if not op[0].results)[0]
    assert not control.invariant_over(control.environment.expression(branch.cond), loop.operation)


@pytest.mark.parametrize("division", ["ceildiv", "floordiv"])
def test_divided_loop_bound_keeps_joint_ancestor_bias_and_exact_active_range(division):
    n = Expr("argument", ("n",))
    owner = Expr("argument", (Symbol("$owner", SymbolKind.GROUP),))
    induction = Expr("argument", (Symbol("$iv", SymbolKind.ORDINAL),))
    facts = {"n": Fact(), owner.args[0]: Fact(maximum=6)}
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
    end = Expr("argument", ("end",))
    owner = Expr("argument", (Symbol("$owner", SymbolKind.GROUP),))
    induction = Expr("argument", (Symbol("$iv", SymbolKind.ORDINAL),))
    facts = {"begin": Fact(), "end": Fact(), owner.args[0]: Fact(maximum=6)}
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
        def full_tiles(X:N.Tensor((n,),N.f32),n:N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(n//64):
                    N.copy(X[i*64:(i+1)*64],x)
""")
    memory = collect_native_memory(lower_native_structure(source).module)
    bounds = admit_memory_bounds(memory)
    assert bounds.checked_accesses == 2

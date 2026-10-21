"""Scalar rewrites preserve the physical control and asynchronous program."""

import pytest
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    ModuleOp,
    StringAttr,
    UnrealizedConversionCastOp,
    VectorType,
    f32,
    i1,
    i32,
    i64,
)
from xdsl.ir import Block, Region, SSAValue

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump
from ascendflow.dialects.physical import RuntimeSizeOp, TaskOp
from ascendflow.dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.simplify_scalars import simplify_scalars

_ARITH_OPERATIONS = tuple(arith.Arith.operations)
_MAX_I64 = (1 << 63) - 1


def _constant(block, value, typ=None):
    operation = arith.ConstantOp.from_int_and_width(value, IndexType() if typ is None else typ)
    block.add_op(operation)
    return operation


def _function(block, *results):
    block.add_op(func.ReturnOp(*results))
    function = func.FuncOp.from_region(
        "scalar",
        [argument.type for argument in block.args],
        [SSAValue.get(result).type for result in results],
        Region(block),
    )
    return ModuleOp([function])


def _control_identity(module):
    return tuple(
        (operation, tuple(operation.regions), tuple(b for r in operation.regions for b in r.blocks))
        for operation in module.walk()
        if not isinstance(operation, _ARITH_OPERATIONS)
    )


def test_stock_integer_identities_constants_and_select_replace_ssa_uses():
    block = Block(arg_types=[IndexType()])
    value = block.args[0]
    zero, one, two, three = (_constant(block, i) for i in (0, 1, 2, 3))
    add_zero = arith.AddiOp(value, zero)
    mul_one = arith.MuliOp(add_zero, one)
    folded = arith.AddiOp(two, three)
    result = arith.AddiOp(mul_one, folded)
    comparison = arith.CmpiOp(value, value, "eq")
    selection = arith.SelectOp(comparison, result, zero)
    unused_non_arith = UnrealizedConversionCastOp.get([value], [IndexType()])
    block.add_ops([add_zero, mul_one, folded, result, comparison, selection, unused_non_arith])
    module = _function(block, selection)
    control = _control_identity(module)

    assert simplify_scalars(module) is None
    module.verify()
    returned = block.last_op.arguments[0]
    assert returned is result.result
    assert result.lhs is value
    assert isinstance(result.rhs.owner, arith.ConstantOp)
    assert result.rhs.owner.value.value.data == 5
    assert all(op.parent is None for op in (add_zero, mul_one, folded, comparison, selection))
    assert _control_identity(module) == control
    once = dump(module)
    simplify_scalars(module)
    assert dump(module) == once


def test_actual_addresses_notifications_native_packets_and_pipeline_keep_identity_and_order():
    structure = lower_native_structure(
        N.from_source("""
        def repeated():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(3):
                    with N.Stage(1, name="write"):
                        N.fill(x, 1.0)
                        N.fill(x, 1.0)
""")
    )
    module = structure.module
    addresses = tuple(op for op in module.walk() if isinstance(op, AddressOp))
    natives = tuple(op for op in module.walk() if isinstance(op, TaskOp))
    assert len(addresses) == len(natives) == 2
    original_offsets = tuple(address.offset.owner.value.value.data for address in addresses)
    for address in addresses:
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        offset = arith.AddiOp(address.offset, zero)
        address.parent.insert_ops_before([zero, offset], address)
        address.operands[1] = offset.result

    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    three = arith.ConstantOp.from_int_and_width(3, IndexType())
    event = arith.AddiOp(two, three)
    notifications = tuple(
        NotificationOp.build(
            operands=[event.result],
            attributes={
                "kind": StringAttr(kind),
                "participants": ArrayAttr([StringAttr("aiv0")]),
                "source": StringAttr("V"),
                "destination": StringAttr("MTE3"),
                "protocol": StringAttr("scalar-test"),
            },
        )
        for kind in ("local_set", "local_set", "local_wait", "local_wait")
    )
    natives[0].parent.insert_ops_before([two, three, event, *notifications], natives[0])
    control = _control_identity(module)
    loops = tuple(op for op in module.walk() if isinstance(op, PipelineLoopOp))
    module.verify()

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    assert tuple(op for op in module.walk() if isinstance(op, TaskOp)) == natives
    assert tuple(op for op in module.walk() if isinstance(op, PipelineLoopOp)) == loops
    assert tuple(address.offset.owner.value.value.data for address in addresses) == original_offsets
    assert all(isinstance(op.event.owner, arith.ConstantOp) for op in notifications)
    assert all(op.event.owner.value.value.data == 5 for op in notifications)


def test_division_and_remainder_stay_inside_their_original_short_circuit_regions():
    block = Block(arg_types=[IndexType(), IndexType()])
    numerator, denominator = block.args
    zero, one = (_constant(block, value) for value in (0, 1))
    nonempty = arith.CmpiOp(denominator, zero, "sgt")
    block.add_op(nonempty)
    body = Block(arg_types=[IndexType()])
    body.add_op(scf.YieldOp())
    empty_loop = scf.ForOp(zero, denominator, one, [], body)
    block.add_op(empty_loop)

    guarded = Block()
    division = arith.DivUIOp(numerator, denominator)
    remainder = arith.RemUIOp(numerator, denominator)
    ceiling = arith.CeilDivUIOp(numerator, denominator)
    unused_division = arith.DivUIOp(numerator, denominator)
    pair = arith.AddiOp(division, remainder)
    result = arith.AddiOp(pair, ceiling)
    identity = arith.AddiOp(result, zero)
    guarded.add_ops(
        [
            division,
            remainder,
            ceiling,
            unused_division,
            pair,
            result,
            identity,
            scf.YieldOp(identity),
        ]
    )
    true_block = Block()
    equal = arith.CmpiOp(denominator, denominator, "eq")
    inner = scf.IfOp(
        equal,
        [IndexType()],
        Region(guarded),
        Region(Block([scf.YieldOp(zero)])),
    )
    true_block.add_ops([equal, inner, scf.YieldOp(inner)])
    outer = scf.IfOp(
        nonempty,
        [IndexType()],
        Region(true_block),
        Region(Block([scf.YieldOp(zero)])),
    )
    block.add_op(outer)
    module = _function(block, outer)
    control = _control_identity(module)
    module.verify()

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    assert tuple(guarded.ops) == (
        division,
        remainder,
        ceiling,
        pair,
        result,
        guarded.last_op,
    )
    assert all(op.parent is guarded for op in (division, remainder, ceiling))
    assert unused_division.parent is None
    assert isinstance(inner.cond.owner, arith.ConstantOp)
    assert inner.cond.owner.value.value.data != 0
    assert identity.parent is None


def test_moving_a_definition_invalidates_its_original_guarded_range():
    from ascendflow.passes.simplify_scalars import ScalarFacts

    block = Block()
    zero, one, two, four = (_constant(block, value) for value in (0, 1, 2, 4))
    body = Block(arg_types=[IndexType()])
    iv = body.args[0]
    condition = arith.CmpiOp(iv, one, "sge")
    guarded = Block()
    difference = arith.SubiOp(iv, one)
    remainder = arith.RemUIOp(difference, two)
    use = UnrealizedConversionCastOp.get([remainder], [IndexType()])
    guarded.add_ops([difference, remainder, use, scf.YieldOp()])
    body.add_ops([condition, scf.IfOp(condition, [], Region(guarded), Region()), scf.YieldOp()])
    loop = scf.ForOp(zero, four, one, [], body)
    block.add_op(loop)
    module = _function(block)
    facts = ScalarFacts(ranges={difference.result: (0, 2)})
    simplify_scalars(module, facts=facts)
    module.verify()
    assert difference.parent is body
    assert facts.ranges[difference.result] == (-1, 2)
    # The shifted value is negative on the newly exposed first iteration.
    assert remainder.parent is guarded


def _execute_i64(block, argument):
    values = {block.args[0]: argument}
    for operation in block.ops:
        if isinstance(operation, func.ReturnOp):
            return tuple(values[value] for value in operation.arguments)
        if isinstance(operation, arith.ConstantOp):
            value = operation.value.value.data
        elif isinstance(operation, arith.AddiOp):
            value = values[operation.lhs] + values[operation.rhs]
        elif isinstance(operation, (arith.DivUIOp, arith.CeilDivUIOp, arith.RemUIOp)):
            left, right = (values[operand] % (1 << 64) for operand in operation.operands)
            quotient, remainder = divmod(left, right)
            value = (
                remainder
                if isinstance(operation, arith.RemUIOp)
                else quotient + int(isinstance(operation, arith.CeilDivUIOp) and remainder != 0)
            )
            if value > _MAX_I64:
                value -= 1 << 64
        elif isinstance(operation, arith.CmpiOp) and operation.predicate.value.data == 2:
            value = -int(values[operation.lhs] < values[operation.rhs])
        else:
            raise TypeError(f"unexpected scalar instruction {operation.name}")
        assert -(1 << 63) <= value <= _MAX_I64, "new signed intermediate overflowed"
        values[operation.result] = value


def test_integer_chains_do_not_reassociate_into_overflowing_constant_intermediates():
    block = Block(arg_types=[i64])
    maximum = _constant(block, _MAX_I64, i64)
    one = _constant(block, 1, i64)
    negative_one = _constant(block, -1, i64)
    first = arith.AddiOp(block.args[0], maximum)
    result = arith.AddiOp(first, one)
    safe_fold = arith.AddiOp(maximum, negative_one)
    block.add_ops([first, result, safe_fold])
    module = _function(block, result, safe_fold)
    inputs = (-_MAX_I64, -_MAX_I64 + 1, -1)
    expected = tuple(_execute_i64(block, value) for value in inputs)

    simplify_scalars(module)
    module.verify()

    assert first.parent is result.parent is block
    assert result.lhs is first.result and first.rhs is maximum.result
    assert tuple(_execute_i64(block, value) for value in inputs) == expected
    constant = block.last_op.arguments[1].owner
    assert isinstance(constant, arith.ConstantOp)
    assert constant.value.value.data == _MAX_I64 - 1


def test_integer_cse_reuses_dominators_but_keeps_sibling_divisions_in_their_branches():
    block = Block(arg_types=[IndexType(), IndexType(), i1])
    numerator, denominator, condition = block.args
    one = _constant(block, 1)
    common = arith.AddiOp(numerator, one)
    block.add_op(common)
    regions, divisions = [], []
    for _ in range(2):
        duplicate_sum = arith.AddiOp(numerator, one)
        first = arith.DivUIOp(duplicate_sum, denominator)
        second = arith.DivUIOp(duplicate_sum, denominator)
        branch = Block([duplicate_sum, first, second, scf.YieldOp(first, second)])
        regions.append(Region(branch))
        divisions.append((first, second))
    choice = scf.IfOp(condition, [IndexType(), IndexType()], *regions)
    block.add_op(choice)
    module = _function(block, common, *choice.results)
    control = _control_identity(module)

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    for region, (first, second) in zip(regions, divisions, strict=True):
        assert first.parent is region.block and first.lhs is common.result
        assert second.parent is None
        assert tuple(region.block.last_op.arguments) == (first.result, first.result)
    assert divisions[0][0].result is not divisions[1][0].result
    once = dump(module)
    simplify_scalars(module)
    assert dump(module) == once


def test_runtime_contract_and_actual_loop_bounds_simplify_ranges_without_changing_control():
    from xdsl.dialects.builtin import IntAttr

    block = Block()
    size = RuntimeSizeOp.build(
        result_types=[IndexType()],
        attributes={"sym_name": StringAttr("n"), "minimum": IntAttr(2), "multiple": IntAttr(8)},
    )
    block.add_op(size)
    zero, one, eight, sixteen = (_constant(block, value) for value in (0, 1, 8, 16))
    nonnegative = arith.MaxSIOp(size.value, zero)
    division = arith.DivUIOp(nonnegative, one)
    minimum = arith.CmpiOp(size.value, eight, "sge")
    runtime_uses = UnrealizedConversionCastOp.get(
        [division.result, minimum.result], [IndexType(), i1]
    )
    block.add_ops([nonnegative, division, minimum, runtime_uses])
    body = Block(arg_types=[IndexType()])
    iv = body.args[0]
    maximum = arith.MaxSIOp(iv, zero)
    remainder = arith.RemUIOp(iv, sixteen)
    quotient = arith.DivUIOp(iv, sixteen)
    lower = arith.CmpiOp(iv, zero, "sge")
    upper = arith.CmpiOp(iv, sixteen, "slt")
    loop_uses = UnrealizedConversionCastOp.get(
        [maximum.result, remainder.result, quotient.result, lower.result, upper.result],
        [IndexType(), IndexType(), IndexType(), i1, i1],
    )
    body.add_ops([maximum, remainder, quotient, lower, upper, loop_uses, scf.YieldOp()])
    block.add_op(scf.ForOp(zero, sixteen, one, [], body))
    module = _function(block, size.value)
    control = _control_identity(module)

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    assert runtime_uses.inputs[0] is size.value
    assert runtime_uses.inputs[1].owner.value.value.data != 0
    assert tuple(loop_uses.inputs[:3]) == (iv, iv, zero.result)
    assert all(value.owner.value.value.data != 0 for value in loop_uses.inputs[3:])
    once = dump(module)
    simplify_scalars(module)
    assert dump(module) == once


def test_potentially_wrapping_range_does_not_establish_nonnegative_results():
    block = Block()
    zero, one, four, maximum = (_constant(block, value, i64) for value in (0, 1, 4, _MAX_I64))
    body = Block(arg_types=[i64])
    wrapped = arith.AddiOp(body.args[0], maximum)
    clamp = arith.MaxSIOp(wrapped, zero)
    comparison = arith.CmpiOp(wrapped, zero, "sge")
    uses = UnrealizedConversionCastOp.get([clamp.result, comparison.result], [i64, i1])
    body.add_ops([wrapped, clamp, comparison, uses, scf.YieldOp()])
    block.add_op(scf.ForOp(zero, four, one, [], body))
    module = _function(block)

    simplify_scalars(module)
    module.verify()

    assert wrapped.parent is clamp.parent is comparison.parent is body
    assert tuple(uses.inputs) == (clamp.result, comparison.result)


def test_unsigned_guard_cannot_be_used_as_a_signed_nonnegative_premise():
    block = Block(arg_types=[i64, i64])
    zero = _constant(block, 0, i64)
    condition = arith.CmpiOp(*block.args, "ult")
    block.add_op(condition)
    clamp = arith.MaxSIOp(block.args[0], zero)
    guarded = Block([clamp, scf.YieldOp(clamp)])
    choice = scf.IfOp(condition, [i64], Region(guarded), Region(Block([scf.YieldOp(zero)])))
    block.add_op(choice)
    module = _function(block, choice)

    simplify_scalars(module)
    module.verify()

    assert clamp.parent is block
    assert guarded.last_op.arguments[0] is clamp.result


def test_shared_condition_is_folded_at_each_dominated_control_use():
    block = Block(arg_types=[IndexType()])
    zero, one = (_constant(block, number) for number in (0, 1))
    comparison = arith.CmpiOp(block.args[0], zero, "sgt")
    block.add_op(comparison)
    branches, inner = [], []
    for _ in range(2):
        choice = scf.IfOp(
            comparison,
            [IndexType()],
            Region(Block([scf.YieldOp(one)])),
            Region(Block([scf.YieldOp(zero)])),
        )
        branches.append(Region(Block([choice, scf.YieldOp(choice)])))
        inner.append(choice)
    outer = scf.IfOp(comparison, [IndexType()], *branches)
    block.add_op(outer)
    module = _function(block, outer)
    control = _control_identity(module)

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    assert outer.cond is comparison.result
    assert isinstance(inner[0].cond.owner, arith.ConstantOp)
    assert inner[0].cond.owner.value.value.data != 0
    assert inner[1].cond.owner.value.value.data == 0


def test_unsigned_unit_identities_and_remainder_bounds_hold_for_unknown_sign_bits():
    block = Block(arg_types=[i64])
    value = block.args[0]
    one, eight = (_constant(block, number, i64) for number in (1, 8))
    division = arith.DivUIOp(value, one)
    ceiling = arith.CeilDivUIOp(value, one)
    unit_remainder = arith.RemUIOp(value, one)
    remainder = arith.RemUIOp(value, eight)
    bounded = arith.CmpiOp(remainder, eight, "slt")
    block.add_ops([division, ceiling, unit_remainder, remainder, bounded])
    module = _function(block, division, ceiling, unit_remainder, remainder, bounded)
    inputs = (-(1 << 63), -1, 0, 7, _MAX_I64)
    expected = tuple(_execute_i64(block, number) for number in inputs)
    assert expected == tuple((number, number, 0, number % 8, -1) for number in inputs)

    simplify_scalars(module)
    module.verify()

    returned = block.last_op.arguments
    assert tuple(returned[:2]) == (value, value)
    assert returned[2].owner.value.value.data == 0
    assert returned[3] is remainder.result
    assert returned[4].owner.value.value.data != 0
    assert tuple(_execute_i64(block, number) for number in inputs) == expected


def test_public_admission_rejects_original_unsigned_expression_before_unit_folding(monkeypatch):
    from ascendflow.passes import structured_sync

    def forbidden(module):
        pytest.fail("an invalid original expression reached final scalar simplification")

    monkeypatch.setattr(structured_sync, "simplify_scalars", forbidden)
    source = N.from_source("""
        def invalid():
            with N.Kernel():
                x=N.alloc_ub((2,64))
                for i in N.Pipelined(2):
                    bad=(-1)//(i+1)
                    N.fill(x[bad-bad,:],1.0)
""")
    with pytest.raises(CompileError, match="nonnegative dividend|nonnegative numerator"):
        compile(source)


def test_nested_invariants_keep_overflow_prone_extents_in_the_original_domain():
    from xdsl.dialects.builtin import IntAttr

    block = Block()
    size = RuntimeSizeOp.build(
        result_types=[IndexType()],
        attributes={"sym_name": StringAttr("n"), "minimum": IntAttr(0), "multiple": IntAttr(1)},
    )
    block.add_op(size)
    zero, one, two, four, scale = (_constant(block, number) for number in (0, 1, 2, 4, 256))
    outer = Block(arg_types=[IndexType()])
    inner = Block(arg_types=[IndexType()])
    coordinate = arith.MuliOp(outer.args[0], two)
    extent = arith.MuliOp(size.value, scale)
    uses = UnrealizedConversionCastOp.get([coordinate.result, extent.result], [IndexType()] * 2)
    inner.add_ops([coordinate, extent, uses, scf.YieldOp()])
    child = scf.ForOp(zero, size.value, one, [], inner)
    outer.add_ops([child, scf.YieldOp()])
    block.add_op(scf.ForOp(zero, four, one, [], outer))
    module = _function(block)
    control = _control_identity(module)

    simplify_scalars(module)
    module.verify()

    assert _control_identity(module) == control
    assert coordinate.parent is outer and coordinate.next_op is child
    assert extent.parent is inner
    assert tuple(uses.inputs) == (coordinate.result, extent.result)
    once = dump(module)
    simplify_scalars(module)
    assert dump(module) == once


def test_floating_fastmath_and_vector_arithmetic_remain_outside_scalar_rewrites():
    block = Block(arg_types=[f32])
    left = arith.ConstantOp(FloatAttr(1.0, f32))
    right = arith.ConstantOp(FloatAttr(-1.0, f32))
    flags = arith.FastMathFlagsAttr([arith.FastMathFlag.REASSOC])
    first = arith.AddfOp(block.args[0], left, flags)
    result = arith.AddfOp(first, right, flags)
    vector_type = VectorType(i32, [2])
    vector = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(vector_type, [1, 2]))
    unused_vector_add = arith.AddiOp(vector, vector)
    block.add_ops([left, right, first, result, vector, unused_vector_add])
    module = _function(block, result)
    before = dump(module)

    simplify_scalars(module)
    module.verify()

    assert dump(module) == before
    assert first.parent is result.parent is unused_vector_add.parent is block


def test_public_compile_simplifies_final_ssa_once_after_notification_materialization(monkeypatch):
    from ascendflow.passes import structured_sync

    source = N.from_source("""
        def wired(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(n):
                    N.fill(x, 1.0)
""")
    calls = []
    condition_calls = []
    original_conditions = structured_sync.simplify_conditions

    def observed_conditions(module, *, facts=None):
        assert dump(module) == calls[0]
        result = original_conditions(module, facts=facts)
        condition_calls.append(dump(module))
        return result

    def observed(module, *, facts=None, finalize=True):
        control = _control_identity(module)
        loop = next(op for op in module.walk() if isinstance(op, scf.ForOp))
        original_upper = loop.ub
        if isinstance(original_upper.owner, arith.MaxSIOp):
            original_upper = original_upper.owner.lhs
        assert isinstance(original_upper.owner, RuntimeSizeOp)
        # Inject equivalent arithmetic only after synchronization has established
        # its facts, so this exercises the final owned-module pass connection.
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        one = arith.ConstantOp.from_int_and_width(1, IndexType())
        addition = arith.AddiOp(original_upper, zero)
        multiplication = arith.MuliOp(addition, one)
        loop.parent.insert_ops_before([zero, one, addition, multiplication], loop)
        loop.operands[1] = multiplication.result
        notifications = tuple(op for op in module.walk() if isinstance(op, NotificationOp))
        assert notifications
        simplify_scalars(module, facts=facts, finalize=finalize)
        assert _control_identity(module) == control
        assert loop.ub is original_upper
        assert addition.parent is multiplication.parent is None
        assert tuple(op for op in module.walk() if isinstance(op, NotificationOp)) == notifications
        calls.append(dump(module))

    monkeypatch.setattr(structured_sync, "simplify_scalars", observed)
    monkeypatch.setattr(structured_sync, "simplify_conditions", observed_conditions)
    result = compile(source)
    assert len(calls) == 1
    assert len(condition_calls) == 1 and result.plan_ir == condition_calls[0]
    assert "arith.addi" not in result.plan_ir
    assert "arith.muli" not in result.plan_ir


def test_scoped_range_query_visits_a_shared_dag_once_and_observes_later_guards(monkeypatch):
    from ascendflow.passes import simplify_scalars as integers

    block = Block(arg_types=[IndexType(), IndexType()])
    value, unrelated = block.args
    zero, two = (_constant(block, number) for number in (0, 2))
    result = value
    depth = 20
    for _ in range(depth):
        operation = arith.AddiOp(result, result)
        block.add_op(operation)
        result = operation.result
    _function(block, result).verify()
    parent = integers._Scope(ranges={value: (0, 3), unrelated: (0, 3)})
    child = parent.child()
    child.relation(unrelated, 5, zero.result)
    visits = 0
    original = integers._bounded_interval

    def observed(*args, **kwargs):
        nonlocal visits
        visits += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(integers, "_bounded_interval", observed)
    assert child.interval(result) == (0, 3 << depth)
    child.relation(value, 5, two.result)
    assert child.interval(result) == (2 << depth, 3 << depth)
    sibling = parent.child()
    sibling.relation(unrelated, 5, zero.result)
    assert sibling.interval(result) == (0, 3 << depth)
    assert visits <= 4 * depth


def test_range_reuse_observes_parent_changes_and_broadened_source_facts():
    from ascendflow.passes import simplify_scalars as integers

    block = Block(arg_types=[IndexType(), IndexType()])
    value, unrelated = block.args
    zero, two = (_constant(block, n) for n in (0, 2))
    doubled = arith.AddiOp(value, value)
    block.add_op(doubled)
    _function(block, doubled.result).verify()
    facts = integers.ScalarFacts(ranges={doubled.result: (0, 4)})
    parent = integers._Scope(ranges={value: (0, 3), unrelated: (0, 1)}, facts=facts)
    child = parent.child()
    child.relation(unrelated, 5, zero.result)
    assert child.interval(doubled.result) == (0, 4)
    # Moving the definition out of its old guard broadens the source fact.
    facts.bind_range(doubled.result, (0, 20))
    assert child.interval(doubled.result) == (0, 6)
    parent.bind_range(value, (0, 5))
    assert child.interval(doubled.result) == (0, 10)
    sibling = parent.child()
    sibling.relation(unrelated, 5, zero.result)
    assert sibling.interval(doubled.result) == (0, 10)
    child.relation(value, 5, two.result)
    assert child.interval(doubled.result) == (4, 10)
    assert sibling.interval(doubled.result) == (0, 10)


def test_deep_scope_reuse_does_not_rewalk_unchanged_ancestor_inputs(monkeypatch):
    from ascendflow.passes import simplify_scalars as integers

    value = Block(arg_types=[IndexType()]).args[0]
    parent = integers._Scope(ranges={value: (0, 3)})
    child = parent
    depth = 64
    for _ in range(depth):
        child = child.child()
    assert child.interval(value) == (0, 3)
    visits = 0
    refresh = integers._Scope.refresh

    def observed(scope):
        nonlocal visits
        visits += 1
        return refresh(scope)

    monkeypatch.setattr(integers._Scope, "refresh", observed)
    queries = 128
    for _ in range(queries):
        assert child.interval(value) == (0, 3)
    assert visits <= queries + depth
    # A possible-change stamp is not evidence that an old result is valid.
    # Broaden the real ancestor input after all descendants cached its range.
    parent.bind_range(value, (0, 7))
    assert child.interval(value) == (0, 7)

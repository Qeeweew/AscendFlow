"""Scalar rewrites preserve the physical control and asynchronous program."""

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
    i32,
    i64,
)
from xdsl.ir import Block, Region, SSAValue

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import RuntimeSizeOp, TaskOp
from xdsl_npu.dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.passes.simplify_scalars import simplify_scalars

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
                for i in N.Pipelined(3, stages=2):
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
        unused_division,
        pair,
        result,
        guarded.last_op,
    )
    assert all(op.parent is guarded for op in (division, remainder, ceiling, unused_division))
    assert isinstance(inner.cond.owner, arith.ConstantOp)
    assert inner.cond.owner.value.value.data != 0
    assert identity.parent is None


def _execute_i64(block, argument):
    values = {block.args[0]: argument}
    for operation in block.ops:
        if isinstance(operation, func.ReturnOp):
            return tuple(values[value] for value in operation.arguments)
        if isinstance(operation, arith.ConstantOp):
            value = operation.value.value.data
        elif isinstance(operation, arith.AddiOp):
            value = values[operation.lhs] + values[operation.rhs]
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
    from xdsl_npu.passes import structured_sync

    source = N.from_source("""
        def wired(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(n, stages=1):
                    N.fill(x, 1.0)
    """)
    calls = []

    def observed(module):
        control = _control_identity(module)
        loop = next(op for op in module.walk() if isinstance(op, PipelineLoopOp))
        original_upper = loop.upper
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
        simplify_scalars(module)
        assert _control_identity(module) == control
        assert loop.upper is original_upper
        assert addition.parent is multiplication.parent is None
        assert tuple(op for op in module.walk() if isinstance(op, NotificationOp)) == notifications
        calls.append(dump(module))

    monkeypatch.setattr(structured_sync, "simplify_scalars", observed)
    result = compile(source)
    assert len(calls) == 1
    assert result.plan_ir == calls[0]
    assert "arith.addi" not in result.plan_ir
    assert "arith.muli" not in result.plan_ir

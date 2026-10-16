"""Actual serial-query IR clocks, input domains, and captured address contracts."""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntAttr
from xdsl.ir import Block
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def

from tests.fixtures.nested import nested_copy
from tests.test_scope_serial_source import SOURCE
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.physical import CoreIdOp, CyclicCountOp, RuntimeSizeOp, StreamOp, ViewOp
from xdsl_npu.dialects.types import AsyncEffects
from xdsl_npu.indexing import arguments
from xdsl_npu.lowering import lower_source
from xdsl_npu.scalar import Expr, facts_from_sizes, uniform
from xdsl_npu.scope_model import extract_scopes


@pytest.fixture(scope="module")
def serial_template():
    return lower_source(N.from_source(SOURCE, disjoint_external=True))


@pytest.fixture(scope="module")
def uniform_serial_template():
    text = (
        SOURCE.replace("N.ceildiv(N.max(0,total_queries-group),GROUPS)", "total_queries")
        .replace("((total_queries,2,16,16)", "((GROUPS,total_queries,2,16,16)")
        .replace("((total_queries,4,64)", "((GROUPS,total_queries,4,64)")
        .replace("Q[flat,part,:,:]", "Q[group,query,part,:,:]")
        .replace("O[flat,lane*2+part,:]", "O[group,query,lane*2+part,:]")
    )
    return lower_source(N.from_source(text, disjoint_external=True))


@pytest.fixture
def serial(serial_template):
    return serial_template.clone()


def actor_query(program, actor="aiv0"):
    return program.invocations[actor].op


def first_leaf(program, actor="aiv0"):
    return next(leaf for leaf in program.leaves.values() if leaf.actor == actor)


def replace_query_operand(query, index, value):
    operands = list(query.operands)
    operands[index] = value
    query.operands = operands


def parameter(program, name):
    return next(op for op in program.parameters if op.sym_name.data == name)


def test_extraction_retains_actual_serial_operations_roots_and_three_clocks(serial_template):
    before = dump(serial_template)
    program = extract_scopes(serial_template)
    assert dump(serial_template) == before
    assert program.bound == Expr("argument", ("kv_tiles",))
    facts = facts_from_sizes(program.parameters)
    total = parameter(program, "total_queries")
    for actor, invocation in program.invocations.items():
        query = invocation.op
        parent = program.parents[actor]
        assert tuple(type(op) for op in query.body.block.ops) == (StreamOp, scf.YieldOp)
        assert query.parent_op() is program.kernels[actor]
        assert parent.parent_op() is query
        assert query.ub.owner.total is total.value
        assert isinstance(query.ub.owner, CyclicCountOp)
        assert invocation.bound == invocation.cyclic.local_bound
        assert invocation.cyclic.total == Expr("argument", ("total_queries",))
        assert invocation.cyclic.groups == program.groups == 4
        assert not uniform(invocation.bound, facts)
        leaf = first_leaf(program, actor)
        assert leaf.call.generation is parent.body.block.args[0]
        assert leaf.call.generation is not query.body.block.args[0]
        assert any("$parent1" in arguments(cap.offset) for cap in leaf.captures)
        for captured in leaf.captures:
            assert captured.op.source is captured.root.ref
            if captured.root.external.data:
                assert captured.root.parent_op() is program.root


def test_ordinary_nested_program_has_no_serial_invocation():
    program = extract_scopes(lower_source(nested_copy))
    assert program.invocations == {}


def test_uniform_query_bound_retains_its_actual_shared_ssa(uniform_serial_template):
    program = extract_scopes(uniform_serial_template)
    total = parameter(program, "total_queries")
    assert len(program.invocations) == 3
    for invocation in program.invocations.values():
        assert invocation.op.ub is total.value
        assert invocation.bound == Expr("argument", ("total_queries",))
        assert invocation.cyclic is None


def test_one_actor_cannot_change_its_uniform_query_bound(uniform_serial_template):
    module = uniform_serial_template.clone()
    program = extract_scopes(module)
    replace_query_operand(actor_query(program), 1, parameter(program, "kv_tiles").value)
    with pytest.raises(CompileError, match="query domain or cyclic ownership"):
        extract_scopes(module)


@pytest.mark.parametrize("operand,value", ((0, 1), (2, 2)))
def test_query_control_requires_actual_zero_start_and_unit_step(serial, operand, value):
    program = extract_scopes(serial)
    constant = arith.ConstantOp.from_int_and_width(value, IndexType())
    program.root.body.block.insert_op_before(constant, program.kernels["aic"])
    replace_query_operand(actor_query(program), operand, constant.result)
    with pytest.raises(CompileError, match="zero/unit bounds"):
        extract_scopes(serial)


def test_actual_serial_scalar_carry_is_rejected_before_child_normalization(serial):
    program = extract_scopes(serial)
    query = actor_query(program)
    replacement = scf.ForOp(
        query.lb,
        query.ub,
        query.step,
        [query.lb],
        Block(arg_types=[IndexType(), IndexType()]),
    )
    parent = program.parents["aiv0"]
    query.body.block.args[0].replace_all_uses_with(replacement.body.block.args[0])
    query.body.block.detach_op(parent)
    replacement.body.block.add_ops([parent, scf.YieldOp(replacement.body.block.args[1])])
    query.parent.insert_op_before(replacement, query)
    query.parent.erase_op(query)
    with pytest.raises(CompileError, match="one clock and no carries"):
        extract_scopes(serial)


def test_call_cannot_use_the_outer_query_as_its_kv_clock(serial):
    program = extract_scopes(serial)
    first_leaf(program).call.operands = (actor_query(program).body.block.args[0],)
    with pytest.raises(CompileError, match="call uses a different parent clock"):
        extract_scopes(serial)


def test_native_child_cannot_use_the_outer_query_as_its_fixed_child_clock(serial):
    program = extract_scopes(serial)
    leaf = first_leaf(program)
    view = next(op for op in leaf.stream.walk() if isinstance(op, ViewOp))
    operands = list(view.operands)
    operands[1] = actor_query(program).body.block.args[0]
    view.operands = operands
    with pytest.raises(CompileError, match="different (child|inner) clock"):
        extract_scopes(serial)


def test_one_actor_cannot_change_its_actual_query_total(serial):
    program = extract_scopes(serial)
    actor_query(program).ub.owner.operands = (parameter(program, "kv_tiles").value,)
    with pytest.raises(CompileError, match="query domain or cyclic ownership"):
        extract_scopes(serial)


def test_equivalent_actual_root_total_keeps_the_same_query_domain(serial):
    program = extract_scopes(serial)
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    equivalent = arith.AddiOp(parameter(program, "total_queries").value, zero.result)
    program.root.body.block.insert_ops_before([zero, equivalent], program.kernels["aic"])
    query = actor_query(program)
    query.ub.owner.operands = (equivalent.result,)
    before = dump(serial)
    rebuilt = extract_scopes(serial)
    assert dump(serial) == before
    assert actor_query(rebuilt).ub.owner.total is equivalent.result
    assert rebuilt.invocations["aiv0"].cyclic == rebuilt.invocations["aic"].cyclic


def test_all_mixed_actors_must_retain_the_same_actual_serial_scope(serial):
    program = extract_scopes(serial)
    query = actor_query(program)
    kernel = program.kernels["aiv0"]
    count = query.ub.owner
    query.body.block.args[0].replace_all_uses_with(query.lb)
    parent = program.parents["aiv0"]
    query.body.block.detach_op(parent)
    kernel.body.block.insert_op_before(parent, query)
    kernel.body.block.erase_op(query)
    kernel.body.block.erase_op(count)
    with pytest.raises(CompileError, match="disagree on the actual serial query scope"):
        extract_scopes(serial)


def test_cyclic_count_is_not_a_uniform_kv_limit(serial):
    program = extract_scopes(serial)
    program.parents["aiv0"].operands = (actor_query(program).ub,)
    with pytest.raises(CompileError, match="parent count must be an actual uniform SSA"):
        extract_scopes(serial)


@pytest.mark.parametrize("total", ("core", "actor_parameter", "actor_arithmetic"))
def test_cyclic_total_must_be_an_actual_root_shared_uniform_input(serial, total):
    program = extract_scopes(serial)
    kernel = program.kernels["aiv0"]
    count = actor_query(program).ub.owner
    if total == "core":
        value = next(op.value for op in kernel.body.block.ops if isinstance(op, CoreIdOp))
    elif total == "actor_parameter":
        local = parameter(program, "total_queries").clone()
        kernel.body.block.insert_op_before(local, count)
        value = local.value
    else:
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        local = arith.AddiOp(parameter(program, "total_queries").value, zero.result)
        kernel.body.block.insert_ops_before([zero, local], count)
        value = local.result
    count.operands = (value,)
    with pytest.raises(CompileError, match="actual root-shared uniform total"):
        extract_scopes(serial)


def test_cyclic_query_uses_the_actors_actual_group_count(serial):
    program = extract_scopes(serial)
    program.kernels["aiv0"].attributes["blocks"] = IntAttr(3)
    with pytest.raises(CompileError, match="nested actors disagree on groups"):
        extract_scopes(serial)


def test_cyclic_count_must_be_the_actual_serial_upper_bound(serial):
    program = extract_scopes(serial)
    replace_query_operand(actor_query(program), 1, parameter(program, "total_queries").value)
    with pytest.raises(CompileError, match="actual uniform or cyclic query count"):
        extract_scopes(serial)


@irdl_op_definition
class SerialBodyEffect(IRDLOperation):
    name = "test.serial_body_effect"
    traits = traits_def(AsyncEffects())


def test_serial_body_effect_cannot_be_erased_by_query_extraction(serial):
    program = extract_scopes(serial)
    query = actor_query(program)
    query.body.block.insert_op_before(SerialBodyEffect.create(), program.parents["aiv0"])
    with pytest.raises(CompileError, match="exactly its parent stream and empty yield"):
        extract_scopes(serial)


def test_actual_query_capture_cannot_extend_one_query_past_the_total(serial):
    program = extract_scopes(serial)
    captured = next(
        cap for cap in first_leaf(program, "aic").captures if cap.root.sym_name.data == "Q"
    )
    scale = arith.ConstantOp.from_int_and_width(captured.root.count_scale.data, IndexType())
    shifted = arith.AddiOp(captured.op.offset, scale.result)
    captured.op.parent.insert_ops_before([scale, shifted], captured.op)
    captured.op.operands = (captured.op.source, shifted.result)
    with pytest.raises(CompileError, match="bounded by its tensor extent"):
        extract_scopes(serial)


def test_query_coordinate_still_requires_nonnegative_unsigned_dividends(serial):
    program = extract_scopes(serial)
    leaf = first_leaf(program)
    captured = leaf.captures[0].op
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    negative = arith.SubiOp(actor_query(program).body.block.args[0], one.result)
    remainder = arith.RemUIOp(negative.result, two.result)
    captured.parent.insert_ops_before([one, two, negative, remainder], captured)
    with pytest.raises(
        CompileError, match="unsigned division has an unproven nonnegative dividend"
    ):
        extract_scopes(serial)


def test_query_extraction_does_not_enumerate_large_runtime_domains(serial, monkeypatch):
    from xdsl_npu import scope_model

    program = extract_scopes(serial)
    parameter(program, "total_queries").attributes["minimum"] = IntAttr(1 << 50)
    parameter(program, "kv_tiles").attributes["minimum"] = IntAttr(1 << 40)
    calls = []

    def fixed_child_range(limit):
        calls.append(limit)
        assert limit <= 2
        return range(limit)

    monkeypatch.setattr(scope_model, "range", fixed_child_range, raising=False)
    rebuilt = extract_scopes(serial)
    assert rebuilt.invocations.keys() == program.invocations.keys()
    assert calls and max(calls) <= 2
    assert all(isinstance(op, RuntimeSizeOp) for op in rebuilt.parameters)

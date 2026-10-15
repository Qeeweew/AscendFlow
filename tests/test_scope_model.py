"""Reject mutations of actual nested physical IR before leaf normalization."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def

from tests.fixtures.nested import nested_copy, nested_cv
from tests.test_scope_codegen import VECTOR
from xdsl_npu import language as N
from xdsl_npu.ir import (
    AsyncEffects,
    BufferOp,
    CoreIdOp,
    FixConfigOp,
    InvocationSyncOp,
    LoopSyncOp,
    RegionSyncOp,
    ScopeSyncOp,
    ViewOp,
    dump,
)
from xdsl_npu.model import CompileError
from xdsl_npu.scalar import INT64_MAX
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_sync import autosync_scopes


@pytest.fixture(scope="module")
def vector_template():
    return lower_scopes(nested_copy.to_ir().body.block.first_op)


@pytest.fixture(scope="module")
def mixed_template():
    return lower_scopes(nested_cv.to_ir().body.block.first_op)


@pytest.fixture(scope="module")
def planned_template():
    source = N.from_source(VECTOR, disjoint_external=True).to_ir()
    return autosync_scopes(lower_scopes(source.body.block.first_op))


@pytest.fixture
def vector(vector_template):
    return vector_template.clone()


@pytest.fixture
def mixed(mixed_template):
    return mixed_template.clone()


def first_leaf(module):
    return next(iter(extract_scopes(module).leaves.values()))


def offset(captured, value):
    constant = arith.ConstantOp.from_int_and_width(value, IndexType())
    captured.op.parent.insert_op_before(constant, captured.op)
    captured.op.operands = (captured.op.source, constant.result)


@pytest.mark.parametrize("fixture", ["vector_template", "mixed_template", "planned_template"])
def test_actual_roots_clocks_and_spans_survive_read_only_extraction(request, fixture):
    module = request.getfixturevalue(fixture)
    before = dump(module)
    scope = extract_scopes(module)
    assert dump(module) == before
    for leaf in scope.leaves.values():
        assert leaf.call.generation is scope.parents[leaf.actor].body.block.args[0]
        assert leaf.loop.constant_count == leaf.stream.count.data
        for captured in leaf.captures:
            assert captured.op.source is captured.root.ref
            assert captured.op.nbytes.data > 0
            if len(scope.kernels) > 1 and captured.root.external.data:
                assert captured.root.parent_op() is scope.root


@irdl_op_definition
class HiddenEffect(IRDLOperation):
    name = "test.hidden_effect"
    traits = traits_def(AsyncEffects())


@pytest.mark.parametrize("location", ["mixed_root", "actor", "parent", "call", "child"])
def test_unknown_effect_cannot_disappear_during_scope_extraction(mixed, location):
    scope = extract_scopes(mixed)
    leaf = next(iter(scope.leaves.values()))
    if location == "mixed_root":
        block, anchor = scope.root.body.block, next(iter(scope.kernels.values()))
        diagnostic = "unsupported nested mixed root effect test.hidden_effect"
    elif location == "actor":
        block, anchor = scope.kernels[leaf.actor].body.block, scope.parents[leaf.actor]
        diagnostic = "unsupported nested kernel effect test.hidden_effect"
    elif location == "parent":
        block = scope.parents[leaf.actor].body.block
        anchor, diagnostic = block.last_op, "parent stream must retain real child calls"
    elif location == "call":
        block, anchor = leaf.call.body.block, leaf.stream
        diagnostic = "unknown callsite effect test.hidden_effect"
    else:
        block = leaf.stream.body.block
        anchor, diagnostic = block.last_op, "unsupported nested child effect test.hidden_effect"
    block.insert_op_before(HiddenEffect.create(), anchor)
    with pytest.raises(CompileError, match=diagnostic):
        extract_scopes(mixed)


def test_duplicate_mixed_actor_cannot_be_overwritten_by_dictionary_extraction(mixed):
    scope = extract_scopes(mixed)
    scope.root.body.block.insert_op_before(
        scope.kernels["aic"].clone(), scope.root.body.block.last_op
    )
    with pytest.raises(CompileError, match="unique AIC/two-AIV participants"):
        extract_scopes(mixed)


@pytest.mark.parametrize("used", [False, True])
def test_mixed_actor_cannot_implicitly_add_a_private_gm_argument(mixed, used):
    scope = extract_scopes(mixed)
    leaf = next(leaf for leaf in scope.leaves.values() if leaf.actor == "aiv0")
    captured = next(c for c in leaf.captures if c.root.external.data)
    private = captured.root.clone()
    private.attributes["sym_name"] = StringAttr("private_gm")
    scope.kernels[leaf.actor].body.block.insert_op_before(private, scope.parents[leaf.actor])
    if used:
        captured.op.operands = (private.ref, captured.op.offset)
    with pytest.raises(CompileError, match="cannot add private GM declarations"):
        extract_scopes(mixed)


def test_mixed_root_cannot_hide_a_local_bank_declaration(mixed):
    scope = extract_scopes(mixed)
    local = next(
        op
        for op in scope.kernels["aiv0"].body.block.ops
        if isinstance(op, BufferOp) and not op.external.data
    ).clone()
    scope.root.body.block.insert_op_before(local, scope.kernels["aic"])
    with pytest.raises(CompileError, match="actual shared GM allocations"):
        extract_scopes(mixed)


def test_shared_gm_roots_need_unique_real_allocation_names(mixed):
    scope = extract_scopes(mixed)
    root = next(op for op in scope.root.body.block.ops if isinstance(op, BufferOp))
    scope.root.body.block.insert_op_before(root.clone(), scope.kernels["aic"])
    with pytest.raises(CompileError, match="duplicate nested shared GM root name"):
        extract_scopes(mixed)


def test_actor_cannot_hide_an_unshared_runtime_parameter(mixed):
    scope = extract_scopes(mixed)
    scope.kernels["aiv0"].body.block.insert_op_before(
        scope.parameters[0].clone(), scope.parents["aiv0"]
    )
    with pytest.raises(
        CompileError, match="parameters/channels must be actual shared root declarations"
    ):
        extract_scopes(mixed)


def test_duplicate_runtime_parameters_are_not_one_implicit_launch_argument(vector):
    scope = extract_scopes(vector)
    scope.root.body.block.insert_op_before(scope.parameters[0].clone(), scope.parents["aiv0"])
    with pytest.raises(CompileError, match="duplicate nested runtime scalar name"):
        extract_scopes(vector)


def test_actual_fix_configuration_cannot_move_after_the_parent_pipeline(mixed):
    scope = extract_scopes(mixed)
    block = scope.kernels["aic"].body.block
    config = next(op for op in block.ops if isinstance(op, FixConfigOp))
    block.detach_op(config)
    block.insert_op_before(config, block.last_op)
    with pytest.raises(CompileError, match="configuration must precede the parent stream"):
        extract_scopes(mixed)


@pytest.mark.parametrize(
    "attribute,value", [("target", "other"), ("disjoint_external", 2), ("epoch_divisor", 1)]
)
def test_actor_attributes_cannot_be_overridden_by_leaf_normalization(mixed, attribute, value):
    scope = extract_scopes(mixed)
    scope.kernels["aic"].attributes[attribute] = (
        StringAttr(value) if isinstance(value, str) else IntAttr(value)
    )
    with pytest.raises(CompileError, match="valid target/alias contract without invocation epochs"):
        extract_scopes(mixed)


@pytest.mark.parametrize("count", [-1, INT64_MAX + 1])
def test_fixed_child_count_must_fit_int64_before_native_access_analysis(vector, count):
    leaf = first_leaf(vector)
    leaf.stream.attributes["count"] = IntAttr(count)
    with pytest.raises(CompileError, match="nested child fixed count must fit nonnegative int64"):
        extract_scopes(vector)


def test_fixed_child_count_plus_actual_native_stage_cannot_overflow(vector):
    leaf = first_leaf(vector)
    assert max(t.stage for data in leaf.loop.actors.values() for t in data.program.tasks) == 1
    leaf.stream.attributes["count"] = IntAttr(INT64_MAX)
    with pytest.raises(CompileError, match="fixed child count plus native stage overflows int64"):
        extract_scopes(vector)


def test_fixed_parent_count_plus_actual_call_stage_cannot_overflow(vector):
    scope = extract_scopes(vector)
    parent = scope.parents["aiv0"]
    parent.operands = ()
    parent.attributes["count"] = IntAttr(INT64_MAX)
    with pytest.raises(CompileError, match="fixed parent count plus call stage overflows int64"):
        extract_scopes(vector)


@pytest.mark.parametrize("child", [False, True])
def test_stream_rejects_an_invalid_symbolic_marker(vector, child):
    scope = extract_scopes(vector)
    stream = next(iter(scope.leaves.values())).stream if child else scope.parents["aiv0"]
    stream.attributes["symbolic"] = IntAttr(2)
    with pytest.raises(CompileError, match="invalid symbolic marker"):
        extract_scopes(vector)


def test_fixed_child_cannot_also_claim_a_runtime_limit(vector):
    scope = extract_scopes(vector)
    next(iter(scope.leaves.values())).stream.operands = (scope.parameters[0].value,)
    with pytest.raises(CompileError, match="exactly one fixed count or uniform limit"):
        extract_scopes(vector)


def test_call_must_use_its_actual_parent_clock(vector):
    scope = extract_scopes(vector)
    core = next(op for op in scope.root.body.block.ops if isinstance(op, CoreIdOp))
    next(iter(scope.leaves.values())).call.operands = (core.value,)
    with pytest.raises(CompileError, match="call uses a different parent clock"):
        extract_scopes(vector)


@pytest.mark.parametrize("kind", [ViewOp, LoopSyncOp])
def test_native_view_and_local_sync_must_use_the_actual_child_clock(planned_template, kind):
    module = planned_template.clone()
    leaf, operation = next(
        (leaf, op)
        for leaf in extract_scopes(module).leaves.values()
        for op in leaf.stream.walk()
        if isinstance(op, kind)
    )
    operands = list(operation.operands)
    operands[1 if kind is ViewOp else 0] = leaf.call.generation
    operation.operands = operands
    with pytest.raises(CompileError, match="different child clock"):
        extract_scopes(module)


@pytest.mark.parametrize("parent", [False, True])
def test_scope_sync_clocks_are_checked_before_the_op_is_removed_from_the_leaf(
    planned_template, parent
):
    module = planned_template.clone()
    leaf = first_leaf(module)
    sync = next(op for op in leaf.stream.walk() if isinstance(op, ScopeSyncOp))
    sync.operands = (
        leaf.stream.body.block.args[0] if parent else leaf.call.generation,
        leaf.stream.body.block.args[0] if parent else leaf.call.generation,
    )
    with pytest.raises(CompileError, match="different parent or child clock"):
        extract_scopes(module)


@pytest.mark.parametrize("kind", [InvocationSyncOp, RegionSyncOp])
def test_other_region_protocols_cannot_be_ignored_in_a_nested_child(vector, kind):
    leaf = first_leaf(vector)
    operation = kind.create(
        operands=[leaf.stream.body.block.args[0]],
        attributes={"action": StringAttr("wait"), "channel": StringAttr("hidden")},
    )
    leaf.stream.body.block.insert_op_before(operation, leaf.stream.body.block.last_op)
    with pytest.raises(CompileError, match="unsupported nested child effect"):
        extract_scopes(vector)


def test_capture_source_must_be_an_actual_root_of_this_actor(mixed):
    scope = extract_scopes(mixed)
    source = next(leaf for leaf in scope.leaves.values() if leaf.actor == "aiv1")
    target = next(leaf for leaf in scope.leaves.values() if leaf.actor == "aiv0")
    other = next(c.root.ref for c in source.captures if not c.root.external.data)
    captured = next(c for c in target.captures if not c.root.external.data)
    captured.op.operands = (other, captured.op.offset)
    with pytest.raises(CompileError, match="one distinct actual allocation root"):
        extract_scopes(mixed)


def test_capture_offset_cannot_depend_on_a_child_coordinate(vector):
    leaf = first_leaf(vector)
    captured = leaf.captures[0].op
    captured.operands = (captured.source, leaf.stream.body.block.args[0])
    with pytest.raises(CompileError, match="invalid captured address/type/span"):
        extract_scopes(vector)


@pytest.mark.parametrize("span", [-1, 0, INT64_MAX + 1])
def test_capture_span_must_be_a_positive_int64(vector, span):
    first_leaf(vector).captures[0].op.attributes["nbytes"] = IntAttr(span)
    with pytest.raises(CompileError, match="invalid captured address/type/span"):
        extract_scopes(vector)


def test_capture_cannot_claim_more_bytes_than_its_actual_static_root(vector):
    leaf = first_leaf(vector)
    captured = next(c for c in leaf.captures if not c.root.external.data)
    captured.op.attributes["nbytes"] = IntAttr(captured.root.nbytes.data + 1)
    with pytest.raises(CompileError, match="span exceeds its actual allocation root"):
        extract_scopes(vector)


def test_local_descriptor_padding_cannot_authorize_bytes_outside_the_capture(vector):
    leaf = first_leaf(vector)
    captured = next(c for c in leaf.captures if not c.root.external.data)
    captured.op.attributes["nbytes"] = IntAttr(captured.op.nbytes.data - 1)
    with pytest.raises(CompileError, match="native child footprint exceeds captured span"):
        extract_scopes(vector)


def test_capture_offset_and_span_must_fit_the_actual_root(vector):
    leaf = first_leaf(vector)
    captured = next(c for c in leaf.captures if not c.root.external.data)
    offset(captured, captured.root.nbytes.data)
    with pytest.raises(CompileError, match="not bounded by its tensor extent"):
        extract_scopes(vector)


def test_dma_capture_base_alignment_is_not_proved_by_a_zero_relative_view(vector):
    captured = next(c for c in first_leaf(vector).captures if not c.root.external.data)
    offset(captured, 4)
    with pytest.raises(CompileError, match="captured base violates native 32-byte alignment"):
        extract_scopes(vector)


def test_cube_capture_base_requires_the_actual_512_byte_alignment(mixed):
    leaf = next(leaf for leaf in extract_scopes(mixed).leaves.values() if leaf.actor == "aic")
    captured = next(c for c in leaf.captures if c.root.ref.type.bank.data == "L1")
    captured.root.attributes["nbytes"] = IntAttr(captured.root.nbytes.data + 512)
    offset(captured, 32)
    with pytest.raises(CompileError, match="captured base violates native 512-byte alignment"):
        extract_scopes(mixed)


def test_original_capture_division_is_validated_even_when_its_result_cancels(vector):
    leaf = first_leaf(vector)
    captured = next(c for c in leaf.captures if not c.root.external.data)
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    negative = arith.SubiOp(leaf.call.generation, one.result)
    quotient = arith.DivUIOp(negative.result, two.result)
    zero = arith.SubiOp(quotient.result, quotient.result)
    for operation in (one, two, negative, quotient, zero):
        captured.op.parent.insert_op_before(operation, captured.op)
    captured.op.operands = (captured.op.source, zero.result)
    with pytest.raises(
        CompileError, match="scope unsigned division has an unproven nonnegative dividend"
    ):
        extract_scopes(vector)

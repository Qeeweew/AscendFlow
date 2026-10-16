"""Actual cyclic count/physical-address contracts and an independent owner oracle."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr

from tests.report_contract import semantic_report
from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import (
    BufferOp,
    CyclicCountOp,
    KernelOp,
    LoopChannelOp,
    LoopSyncOp,
    RuntimeSizeOp,
    StreamOp,
    ViewOp,
)
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.sync import verify_plan


def cyclic_program(groups=3, *, offset=0, width=64, split_extent=False, invoked=False):
    k = Kernel("cyclic_scale", blocks=groups, disjoint_external=True)
    total = k.runtime_size("total")
    count = CyclicCountOp.create(operands=[total], result_types=[IndexType()])
    k.block.add_op(count)
    extent, scale = total, 64
    if split_extent:
        factor = arith.ConstantOp.from_int_and_width(16, IndexType())
        product = arith.MuliOp(total, factor.result)
        k.block.add_ops([factor, product])
        extent, scale = product.result, 4
    x = k.buffer(
        "X", "GM", 0, external=True, initialized=True, extent_count=extent, count_scale=scale
    )
    y = k.buffer("Y", "GM", 0, external=True, extent_count=extent, count_scale=scale)
    a = k.buffer("a", "UB", 2 * width)
    b = k.buffer("b", "UB", 2 * width)
    if invoked:
        k.invocations(total)
    k.loop(count.count)
    src = k.view(x, width, stride=groups * 64, core_stride=64, offset=offset)
    dst = k.view(y, width, stride=groups * 64, core_stride=64, offset=offset)
    inp = k.view(a, width, stride=width, slots=2)
    out = k.view(b, width, stride=width, slots=2)
    k.copy("load", src, inp)
    k.scale("scale", inp, out, 2.0, stage=1)
    k.copy("store", out, dst, stage=2)
    return k.finish()


@pytest.mark.parametrize("groups", [1, 2, 3, 5, 8])
def test_local_counts_and_gm_bytes_equal_independent_owner_enumeration(groups):
    loop = extract_loop(cyclic_program(groups))
    actor = next(iter(loop.actors.values()))
    store = next(task for task in actor.program.tasks if task.uid == "store")
    gm = store.refs[1]
    assert loop.cyclic.groups == groups and loop.constant_count is None
    for total in range(24):
        assigned, written = [], set()
        for core in range(groups):
            # Literal owner enumeration, independent of ceildiv and production
            # bound/address helpers when constructing the expected answer.
            expected = [flat for flat in range(total) if flat % groups == core]
            values = {"total": total, "$core": core}
            local = loop.bound.evaluate(values)
            assert local == len(expected)
            assert local <= loop.host_bound.evaluate({"total": total})
            for g, flat in enumerate(expected):
                assert gm.segments(g, core) == ((flat * 64, (flat + 1) * 64),)
                footprint = set(range(flat * 64, (flat + 1) * 64))
                assert not written & footprint
                written.update(footprint)
            assigned.extend(expected)
        assert sorted(assigned) == list(range(total))
        assert written == set(range(total * 64))


def test_subranges_and_equal_extent_products_are_proved_without_padding():
    module = cyclic_program(offset=32, width=32, split_extent=True)
    loop = extract_loop(module)
    actor = next(iter(loop.actors.values()))
    gm = next(task for task in actor.program.tasks if task.uid == "store").refs[1]
    assert gm.segments(1, 2) == ((5 * 64 + 32, 6 * 64),)
    assert loop.storage_periods == {"X": 1, "Y": 1, "aiv0/a": 2, "aiv0/b": 2}


@pytest.mark.parametrize("total", [0, 7])
def test_constant_total_keeps_the_same_owner_domain(total):
    module = cyclic_program(5)
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    parameter = next(op for op in module.walk() if isinstance(op, RuntimeSizeOp))
    constant = arith.ConstantOp.from_int_and_width(total, IndexType())
    kernel.body.block.insert_op_before(constant, parameter)
    parameter.value.replace_all_uses_with(constant.result)
    loop = extract_loop(module)
    assert [loop.bound.evaluate({"$core": c}) for c in range(5)] == [
        len([i for i in range(total) if i % 5 == c]) for c in range(5)
    ]
    assert compile(module).report["native_templates"] == 3


def test_dynamic_positive_radix_is_a_uniform_total_not_a_storage_period():
    module = cyclic_program()
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    count = next(op for op in module.walk() if isinstance(op, CyclicCountOp))
    radix = RuntimeSizeOp.create(
        result_types=[IndexType()],
        attributes={"sym_name": StringAttr("radix"), "minimum": IntAttr(1), "multiple": IntAttr(1)},
    )
    quotient = arith.DivUIOp(count.total, radix.value)
    kernel.body.block.insert_ops_before([radix, quotient], count)
    count.operands = [quotient.result]
    for op in kernel.body.block.ops:
        if isinstance(op, BufferOp) and op.external.data:
            op.operands = [quotient.result]
    artifact = compile(module)
    loop = extract_loop(artifact.plan)
    assert loop.storage_periods == {"X": 1, "Y": 1, "aiv0/a": 2, "aiv0/b": 2}
    assert [loop.bound.evaluate({"total": 41, "radix": 4, "$core": c}) for c in range(3)] == [
        4,
        3,
        3,
    ]
    # Re-read the actual contract after mutation; no previous proof is trusted.
    radix.attributes["minimum"] = IntAttr(0)
    with pytest.raises(CompileError, match="provably positive"):
        extract_loop(module)


def test_cyclic_ir_roundtrip_and_codegen_keep_host_uniform_and_zero_owners_empty():
    artifact = compile(cyclic_program(5))
    plan = parse(artifact.plan_ir)
    assert semantic_report(verify_plan(plan)) == semantic_report(artifact.report)
    assert isinstance(
        next(op for op in plan.walk() if isinstance(op, CyclicCountOp)), CyclicCountOp
    )
    assert "npu.cyclic_count" in dump(plan)
    device, host = artifact.asc.split('extern "C" int launch_', 1)
    assert device.index("const int64_t core_id") < device.index("const int64_t n =")
    assert "core_id" not in host and "$core" not in artifact.asc
    assert "g >= 0 && g < n" in device
    assert "extent multiplication overflow" in host and "loop step overflow" in host
    # Idle owners still execute only the final per-actor lifetime boundary;
    # the loop guard admits no task when its derived n is zero.
    loop = extract_loop(plan)
    assert [loop.bound.evaluate({"total": 2, "$core": c}) for c in range(5)] == [1, 1, 0, 0, 0]
    assert artifact.report["native_templates"] == 3


@pytest.mark.parametrize(
    "target,field,value,diagnostic",
    [
        ("X", "offset", 64, "exceeds its task tile"),
        ("Y", "offset", 64, "exceeds its task tile"),
        ("X", "core_stride", 0, "owner translation"),
        ("Y", "core_stride", 0, "owner translation"),
        ("X", "stride", 64, "stride must divide"),
        ("Y", "modulus", 2, "positive stream stride"),
    ],
)
def test_mutated_real_gm_views_cannot_bypass_cyclic_ownership(target, field, value, diagnostic):
    module = cyclic_program()
    view = next(
        op
        for op in module.walk()
        if isinstance(op, ViewOp)
        and isinstance(op.source.owner, BufferOp)
        and op.source.owner.sym_name.data == target
    )
    view.attributes[field] = IntAttr(value)
    # This deliberately false credential must not authorize the broken view.
    next(op for op in module.walk() if isinstance(op, CyclicCountOp)).attributes["proven"] = (
        IntAttr(1)
    )
    with pytest.raises(CompileError, match=diagnostic):
        extract_loop(module)


@pytest.mark.parametrize("root", ["X", "Y"])
def test_actual_allocation_extent_is_checked_for_input_and_output(root):
    module = cyclic_program()
    allocation = next(
        op for op in module.walk() if isinstance(op, BufferOp) and op.sym_name.data == root
    )
    allocation.attributes["count_scale"] = IntAttr(32)
    with pytest.raises(CompileError, match="allocation must equal total"):
        extract_loop(module)


def test_total_ssa_and_actual_kernel_groups_are_reverified():
    module = cyclic_program()
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    count = next(op for op in module.walk() if isinstance(op, CyclicCountOp))
    other = RuntimeSizeOp.create(
        result_types=[IndexType()],
        attributes={"sym_name": StringAttr("other"), "minimum": IntAttr(0), "multiple": IntAttr(1)},
    )
    kernel.body.block.insert_op_before(other, count)
    count.operands = [other.value]
    with pytest.raises(CompileError, match="allocation must equal total"):
        extract_loop(module)
    module = cyclic_program()
    next(op for op in module.walk() if isinstance(op, KernelOp)).attributes["blocks"] = IntAttr(4)
    with pytest.raises(CompileError, match="allocation must equal total"):
        extract_loop(module)


def test_negative_or_unbound_total_is_not_a_count_certificate():
    module = cyclic_program()
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    count = next(op for op in module.walk() if isinstance(op, CyclicCountOp))
    negative = arith.ConstantOp.from_int_and_width(-1, IndexType())
    kernel.body.block.insert_op_before(negative, count)
    count.operands = [negative.result]
    with pytest.raises(CompileError, match="nonnegative uniform"):
        extract_loop(module)
    module = cyclic_program()
    stream = next(op for op in module.walk() if isinstance(op, StreamOp))
    total = next(op for op in module.walk() if isinstance(op, RuntimeSizeOp))
    stream.operands = [total.value]
    with pytest.raises(CompileError, match="cyclic count must bind this stream"):
        extract_loop(module)


def test_cyclic_domain_explicitly_rejects_sequence_and_invocation_composition():
    with pytest.raises(CompileError, match="without sequence/invocation/mixed"):
        extract_loop(cyclic_program(invoked=True))
    module = cyclic_program()
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    stream = next(op for op in module.walk() if isinstance(op, StreamOp))
    kernel.body.block.insert_op_before(stream.clone(), kernel.body.block.last_op)
    with pytest.raises(CompileError, match="without sequence/invocation/mixed"):
        extract_loop(module)


def test_cyclic_domain_explicitly_rejects_mixed_composition():
    from tests.fixtures.pipelines import mixed_pipeline

    module = mixed_pipeline()
    root = module.body.block.first_op
    total = arith.ConstantOp.from_int_and_width(3, IndexType())
    count = CyclicCountOp.create(operands=[total.result], result_types=[IndexType()])
    root.body.block.insert_ops_before([total, count], root.body.block.first_op)
    with pytest.raises(CompileError, match="without sequence/invocation/mixed"):
        extract_loop(module)


def test_cyclic_owner_contract_does_not_authorize_missing_local_release():
    artifact = compile(cyclic_program())
    plan = artifact.plan
    release = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data > 0
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == release.sym_name:
            op.parent.erase_op(op)
    release.parent.erase_op(release)
    with pytest.raises(CompileError, match="missing loop WAR"):
        verify_plan(plan)

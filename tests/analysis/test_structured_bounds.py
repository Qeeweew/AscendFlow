"""Retained native bounds use real roots, exact scopes, and original axes."""

from math import prod

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    ModuleOp,
    StringAttr,
    f32,
)
from xdsl.ir import Block, Region, SSAValue

from ascendflow import language as N
from ascendflow.analysis.structured_addresses import StructuredAddresses
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.physical import BufferOp, EndOp, KernelOp, RuntimeSizeOp, TaskOp
from ascendflow.dialects.structured import AddressOp
from ascendflow.dialects.types import TaskType, ref_type
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import literal
from ascendflow.target import native
from ascendflow.target.ascend910b import Actor
from ascendflow.target.native import CAPACITY, ITEMSIZE
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.control import SHARED
from tests.support.structured import native_memory


def _index(block, value):
    if isinstance(value, SSAValue):
        return value
    operation = arith.ConstantOp.from_int_and_width(value, IndexType())
    block.add_op(operation)
    return operation.result


def _buffer(block, bank, nbytes, *, dtype="f32", layout="linear", count=None, scale=None):
    attributes = {
        "sym_name": StringAttr("same_name"),
        "nbytes": IntAttr(nbytes),
        "external": IntAttr(int(bank == "GM")),
        "initialized": IntAttr(int(bank == "GM")),
    }
    if scale is not None:
        attributes["count_scale"] = IntAttr(scale)
    operation = BufferOp.build(
        operands=[count],
        attributes=attributes,
        result_types=[ref_type(bank, dtype, layout)],
    )
    block.add_op(operation)
    return operation


def _address(
    block,
    root,
    *,
    origins=(0,),
    extents=(64,),
    widths=(64,),
    axes=(0,),
    nbytes=256,
    rows=1,
    row_stride=0,
    displacement=0,
    offset=None,
):
    if offset is None:
        assert all(type(value) is int for value in (*origins, *extents))
        itemsize = ITEMSIZE[root.ref.type.dtype.data]
        offset = displacement + sum(
            origin * prod(extents[axis + 1 :]) * itemsize for axis, origin in enumerate(origins)
        )
    operation = AddressOp.build(
        operands=[
            root.ref,
            _index(block, offset),
            tuple(_index(block, value) for value in origins),
            tuple(_index(block, value) for value in extents),
            None,
        ],
        attributes={
            "nbytes": IntAttr(nbytes),
            "rows": IntAttr(rows),
            "row_stride": IntAttr(row_stride),
            "widths": ArrayAttr(IntAttr(value) for value in widths),
            "axes": ArrayAttr(IntAttr(value) for value in axes),
            "displacement": IntAttr(displacement),
        },
        result_types=[root.ref.type],
    )
    block.add_op(operation)
    return operation


def _task(block, kind, *addresses):
    operation = TaskOp.create(
        operands=[address.ref for address in addresses],
        attributes={
            "uid": StringAttr(kind),
            "native": StringAttr(kind),
            "stage": IntAttr(0),
            "params": ArrayAttr([]),
            "scalar": FloatAttr(0.0, f32),
        },
        result_types=[TaskType(), f32] if kind == "scalar_load" else [TaskType()],
    )
    block.add_op(operation)
    return operation


def _module(block, *, core="aiv"):
    block.add_op(EndOp())
    kernel = KernelOp.build(
        attributes={
            "sym_name": StringAttr("bounds"),
            "target": StringAttr("dav-2201"),
            "core": StringAttr(core),
            "disjoint_external": IntAttr(1),
        },
        regions=[Region(block)],
    )
    module = ModuleOp([kernel])
    module.verify()
    return module


@pytest.mark.parametrize(
    "source,count",
    [
        (scale, 6),
        (batched_matmul, 13),
        (N.from_source(SHARED, disjoint_external=True), 19),
    ],
)
def test_real_scale_cube_and_shared_aiv_reuse_actual_memory_facts_without_native_or_decode(
    source, count, monkeypatch
):
    memory = native_memory(source)

    def forbidden(*args, **kwargs):
        pytest.fail("bounds admission reran native rules or address decoding")

    monkeypatch.setattr(native, "native_contract", forbidden)
    monkeypatch.setattr(StructuredAddresses, "decode", forbidden)
    result = admit_memory_bounds(memory)
    assert result.memory is memory and result.checked_accesses == count
    assert set(result.addresses) == set(memory.addresses)
    for value, proof in result.addresses.items():
        assert proof.address is memory.addresses[value]
        assert proof.frames
        assert all(frame in memory.control.frames.values() for frame in proof.frames)
    assert not hasattr(result, "asc") and not hasattr(result, "ownership")
    with pytest.raises(TypeError):
        result.addresses[next(iter(result.addresses))] = None
    with pytest.raises(TypeError):
        result.allocations[next(iter(result.allocations))] = literal(0)


def test_runtime_capacity_uses_actual_count_and_scale_not_prototype_nbytes():
    memory = native_memory(scale)
    bounds = admit_memory_bounds(memory)
    root = next(value for value in bounds.allocations if value.owner.sym_name.data == "X")
    assert root.owner.nbytes.data == 256
    assert bounds.allocations[root].evaluate({"tiles": 100_000}) == 100_000 * 256


@pytest.mark.parametrize("offset", [32, 4096])
def test_forged_offset_cannot_borrow_axis_metadata_even_when_inside_allocation(offset):
    block = Block()
    gm, ub = _buffer(block, "GM", 512), _buffer(block, "UB", 256)
    forged = _address(block, gm, extents=(128,), offset=offset)
    destination = _address(block, ub)
    _task(block, "copy", forged, destination)
    memory = collect_native_memory(_module(block))
    with pytest.raises(CompileError, match="does not match original-axis linearization") as error:
        admit_memory_bounds(memory)
    assert error.value.kind == ErrorKind.INVALID
    assert error.value.phase == "structured-bounds"


def test_original_axes_do_not_certify_an_undersized_real_allocation():
    block = Block()
    gm, ub = _buffer(block, "GM", 128), _buffer(block, "UB", 256)
    source, destination = _address(block, gm), _address(block, ub)
    _task(block, "copy", source, destination)
    with pytest.raises(CompileError, match="not bounded by its allocation") as error:
        admit_memory_bounds(collect_native_memory(_module(block)))
    assert error.value.kind == ErrorKind.INVALID


def test_a_smaller_allocation_can_still_contain_the_actual_native_slice():
    block = Block()
    gm, ub = _buffer(block, "GM", 128), _buffer(block, "UB", 128)
    source = _address(block, gm, extents=(128,), widths=(32,), nbytes=128)
    destination = _address(block, ub, extents=(32,), widths=(32,), nbytes=128)
    _task(block, "copy", source, destination)
    assert admit_memory_bounds(collect_native_memory(_module(block))).checked_accesses == 2


def test_rotated_runtime_grid_composes_axes_with_real_capacity(monkeypatch):
    from ascendflow.analysis import structured_bounds

    source = N.from_source("""
def rotated(X:N.Tensor((3,n,64),N.f32),Out:N.Tensor((3,n,64),N.f32,access="write"),
            n:N.int64(minimum=64,multiple_of=64)):
    with N.Kernel(7) as owner:
        tile=N.alloc_ub((64,64),N.f32)
        blocks=n//64
        tasks=3*blocks
        for i in N.serial(N.ceildiv(N.max(tasks-owner,0),7)):
            task=(owner+i*7+tasks//2)%tasks
            head=task//blocks
            row=(task%blocks)*64
            N.copy(X[head,row:row+64,:],tile)
            N.copy(tile,Out[head,row:row+64,:])
""")
    memory = native_memory(source)
    nonnegative = structured_bounds.domain_nonnegative

    def capacity_query(value, domains, facts, *, assumptions=()):
        assert not domains, "bounded axes must not be re-proved as one flat nonlinear address"
        assert not assumptions
        return nonnegative(value, domains, facts)

    monkeypatch.setattr(structured_bounds, "domain_nonnegative", capacity_query)
    bounds = admit_memory_bounds(memory)
    assert bounds.checked_accesses == 4
    assert all(bound.frames for bound in bounds.addresses.values())


def test_large_byte_allocation_cannot_hide_an_invalid_original_axis():
    block = Block()
    gm, ub = _buffer(block, "GM", 1024), _buffer(block, "UB", 256)
    source = _address(block, gm, origins=(32,))
    destination = _address(block, ub)
    _task(block, "copy", source, destination)
    with pytest.raises(CompileError, match="not bounded by its tensor extent") as error:
        admit_memory_bounds(collect_native_memory(_module(block)))
    assert "axis=0" in error.value.witness


def test_pitched_byte_bounds_include_row_gaps_in_the_last_endpoint():
    block = Block()
    gm, ub = _buffer(block, "GM", 512), _buffer(block, "UB", 512)
    source = _address(
        block,
        gm,
        origins=(0, 0),
        extents=(2, 128),
        widths=(2, 64),
        axes=(0, 1),
        nbytes=256,
        rows=2,
        row_stride=512,
    )
    destination = _address(
        block,
        ub,
        origins=(0, 0),
        extents=(2, 64),
        widths=(2, 64),
        axes=(0, 1),
        nbytes=256,
        rows=2,
        row_stride=256,
    )
    _task(block, "copy_rows", source, destination)
    memory = collect_native_memory(_module(block))
    assert memory.addresses[source.ref].geometry.total_nbytes == 512
    with pytest.raises(CompileError, match="not bounded by its allocation") as error:
        admit_memory_bounds(memory)
    assert "span=768" in error.value.witness


@pytest.mark.parametrize("origin,accepted", [(1, False), (8, True)])
def test_native_alignment_is_checked_on_the_actual_legitimate_origin(origin, accepted):
    block = Block()
    gm, ub = _buffer(block, "GM", 512), _buffer(block, "UB", 256)
    source = _address(block, gm, origins=(origin,), extents=(128,))
    destination = _address(block, ub)
    _task(block, "copy", source, destination)
    memory = collect_native_memory(_module(block))
    if accepted:
        result = admit_memory_bounds(memory)
        assert result.addresses[source.ref].alignment == 32
    else:
        with pytest.raises(CompileError, match="32-byte alignment") as error:
            admit_memory_bounds(memory)
        assert error.value.kind == ErrorKind.INVALID


def test_scalar_origins_need_four_bytes_and_do_not_claim_local_initialization():
    block = Block()
    ub = _buffer(block, "UB", 256)
    source = _address(block, ub, origins=(1,), widths=(1,), nbytes=4)
    _task(block, "scalar_load", source)
    result = admit_memory_bounds(collect_native_memory(_module(block)))
    assert source.offset.owner.value.value.data == 4
    assert result.addresses[source.ref].alignment == 4
    assert ub.initialized.data == 0


@pytest.mark.parametrize("multiple,accepted", [(8, True), (1, False)])
def test_symbolic_remainder_alignment_uses_declared_multiples_without_enumeration(
    multiple, accepted
):
    block = Block()
    size = RuntimeSizeOp.build(
        attributes={
            "sym_name": StringAttr("n"),
            "minimum": IntAttr(0),
            "multiple": IntAttr(multiple),
        },
        result_types=[IndexType()],
    )
    block.add_op(size)
    divisor, itemsize = _index(block, 64), _index(block, 4)
    origin = arith.RemUIOp(size.value, divisor)
    block.add_op(origin)
    offset = arith.MuliOp(origin.result, itemsize)
    block.add_op(offset)
    gm, ub = _buffer(block, "GM", 512), _buffer(block, "UB", 256)
    source = _address(block, gm, origins=(origin.result,), extents=(128,), offset=offset.result)
    destination = _address(block, ub)
    _task(block, "copy", source, destination)
    memory = collect_native_memory(_module(block))
    if accepted:
        assert admit_memory_bounds(memory).checked_accesses == 2
    else:
        with pytest.raises(CompileError, match="32-byte alignment") as error:
            admit_memory_bounds(memory)
        assert error.value.kind == ErrorKind.UNSUPPORTED


@pytest.mark.parametrize(
    "displacement,accepted,diagnostic",
    [
        (224, False, "row gap"),
        (512, True, ""),
        (1024, False, "original view footprint"),
    ],
)
def test_subview_must_fit_original_rows_and_cannot_use_their_holes(
    displacement, accepted, diagnostic
):
    block = Block()
    gm, ub = _buffer(block, "GM", 2048), _buffer(block, "UB", 64)
    source = _address(
        block,
        gm,
        origins=(0, 0),
        extents=(2, 128),
        widths=(2, 64),
        axes=(0, 1),
        nbytes=64,
        displacement=displacement,
    )
    destination = _address(block, ub, extents=(16,), widths=(16,), nbytes=64)
    _task(block, "copy", source, destination)
    memory = collect_native_memory(_module(block))
    if accepted:
        assert admit_memory_bounds(memory).checked_accesses == 2
    else:
        with pytest.raises(CompileError, match=diagnostic):
            admit_memory_bounds(memory)


@pytest.mark.parametrize("rows,pitch,accepted", [(2, 512, True), (3, 128, False)])
def test_multirow_subview_needs_each_native_row_inside_an_original_row(rows, pitch, accepted):
    block = Block()
    gm, ub = _buffer(block, "GM", 1024), _buffer(block, "UB", rows * 64)
    source = _address(
        block,
        gm,
        origins=(0, 0),
        extents=(2, 128),
        widths=(2, 64),
        axes=(0, 1),
        nbytes=64,
        rows=rows,
        row_stride=pitch,
    )
    destination = _address(
        block,
        ub,
        origins=(0, 0),
        extents=(rows, 16),
        widths=(rows, 16),
        axes=(0, 1),
        nbytes=64,
        rows=rows,
        row_stride=64,
    )
    _task(block, "copy_rows", source, destination)
    memory = collect_native_memory(_module(block))
    if accepted:
        assert admit_memory_bounds(memory).checked_accesses == 2
    else:
        with pytest.raises(CompileError, match="physical row mapping") as error:
            admit_memory_bounds(memory)
        assert error.value.kind == ErrorKind.UNSUPPORTED


def test_same_address_uses_the_maximum_native_alignment_without_losing_actual_scope():
    block = Block()
    gm = _buffer(block, "GM", 512, dtype="f16", layout="NZ")
    l1 = _buffer(block, "L1", 512, dtype="f16", layout="NZ")
    l0 = _buffer(block, "L0A", 512, dtype="f16", layout="ZZ")
    geometry = {
        "origins": (0, 0),
        "extents": (16, 16),
        "widths": (16, 16),
        "axes": (0, 1),
        "nbytes": 512,
    }
    source, shared, destination = (_address(block, root, **geometry) for root in (gm, l1, l0))
    _task(block, "copy", source, shared)
    _task(block, "copy", shared, destination)
    memory = collect_native_memory(_module(block, core="aic"))
    result = admit_memory_bounds(memory)
    assert result.addresses[source.ref].alignment == 32
    assert result.addresses[shared.ref].alignment == 512
    assert result.addresses[destination.ref].alignment == 512
    assert len(result.addresses[shared.ref].frames) == 1
    assert result.checked_accesses == 4


def test_same_address_in_distinct_real_loops_keeps_both_use_domains():
    block = Block()
    ub = _buffer(block, "UB", 256)
    address = _address(block, ub, widths=(1,), nbytes=4)
    zero, one, two = (_index(block, value) for value in (0, 1, 2))
    for upper in (one, two):
        body = Block(arg_types=[IndexType()])
        _task(body, "scalar_load", address)
        body.add_op(scf.YieldOp())
        block.add_op(scf.ForOp(zero, upper, one, [], body))
    memory = collect_native_memory(_module(block))
    result = admit_memory_bounds(memory)
    proof = result.addresses[address.ref]
    assert result.checked_accesses == 2 and len(proof.frames) == 2
    assert proof.frames[0].loop is not proof.frames[1].loop
    assert proof.frames[0].domains() != proof.frames[1].domains()


def test_inactive_stronger_alignment_does_not_change_an_active_scalar_address_contract():
    block = Block()
    ub = _buffer(block, "UB", 256)
    address = _address(block, ub, origins=(1,), widths=(1,), nbytes=4)
    _task(block, "scalar_load", address)
    zero, one = (_index(block, value) for value in (0, 1))
    body = Block(arg_types=[IndexType()])
    _task(body, "fill", address)
    body.add_op(scf.YieldOp())
    block.add_op(scf.ForOp(zero, zero, one, [], body))
    result = admit_memory_bounds(collect_native_memory(_module(block)))
    assert result.checked_accesses == 1
    assert result.addresses[address.ref].alignment == 4
    assert len(result.addresses[address.ref].frames) == 1


def test_statically_empty_access_skips_bounds_but_retains_real_local_allocation():
    source = N.from_source("""
        def empty():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(0):
                    N.fill(x,0.0)
""")
    plan = lower_native_structure(source)
    address = next(op for op in plan.module.walk() if isinstance(op, AddressOp))
    bad_offset = arith.ConstantOp.from_int_and_width(100_000, IndexType())
    address.parent.insert_op_before(bad_offset, address)
    address.operands = (address.source, bad_offset.result, *address.operands[2:])
    memory = collect_native_memory(plan.module)
    result = admit_memory_bounds(memory)
    assert result.checked_accesses == 0
    assert result.addresses[address.ref].frames == ()
    assert result.local_bytes == {(Actor.AIV0, "UB"): 256}


def test_empty_loop_does_not_excuse_invalid_native_geometry():
    source = N.from_source("""
        def empty():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.Pipelined(0):
                    N.fill(x,0.0)
""")
    plan = lower_native_structure(source)
    address = next(op for op in plan.module.walk() if isinstance(op, AddressOp))
    address.attributes["nbytes"] = IntAttr(0)
    with pytest.raises(CompileError, match="positive row width"):
        collect_native_memory(plan.module)


def test_unused_local_capacity_is_not_excused_by_an_empty_loop():
    source = N.from_source("""
        def empty():
            with N.Kernel():
                unused=N.alloc_ub((50000,))
                x=N.alloc_ub((64,))
                for i in N.Pipelined(0):
                    N.fill(x,0.0)
""")
    memory = native_memory(source)
    with pytest.raises(CompileError, match="UB capacity exceeded") as error:
        admit_memory_bounds(memory)
    assert error.value.kind == ErrorKind.RESOURCE


def test_mixed_private_ub_capacity_is_counted_independently_for_each_aiv():
    plan = lower_native_structure(N.from_source(SHARED, disjoint_external=True))
    root = plan.module.body.block.first_op
    buffer = _buffer(Block(), "UB", 100_000)
    buffer.detach()
    root.body.block.insert_op_before(buffer, root.body.block.first_op)
    result = admit_memory_bounds(collect_native_memory(plan.module))
    assert result.local_bytes[Actor.AIV0, "UB"] == 102_048
    assert result.local_bytes[Actor.AIV1, "UB"] == 102_048
    assert 2 * 102_048 > CAPACITY["UB"]
    assert result.local_bytes[Actor.AIC, "L0C"] == 2048


def test_unused_local_bank_still_needs_the_actual_kernel_participant():
    block = Block()
    _buffer(block, "L1", 512)
    with pytest.raises(CompileError, match="another kernel participant"):
        admit_memory_bounds(collect_native_memory(_module(block)))


def test_local_packing_accounts_for_real_root_alignment_and_does_not_merge_names():
    block = Block()
    roots = (_buffer(block, "L1", 32), _buffer(block, "L1", 32))
    assert roots[0].sym_name == roots[1].sym_name
    result = admit_memory_bounds(collect_native_memory(_module(block, core="aic")))
    assert len(result.allocations) == 2
    assert result.local_bytes == {(Actor.AIC, "L1"): 544}


def test_padding_between_unused_real_roots_can_exceed_bank_capacity():
    block = Block()
    _buffer(block, "L1", CAPACITY["L1"] - 32)
    _buffer(block, "L1", 32)
    with pytest.raises(CompileError, match="L1 capacity exceeded") as error:
        admit_memory_bounds(collect_native_memory(_module(block, core="aic")))
    assert error.value.kind == ErrorKind.RESOURCE


def test_dynamic_local_capacity_without_a_finite_bound_is_explicitly_unsupported():
    block = Block()
    size = RuntimeSizeOp.build(
        attributes={"sym_name": StringAttr("n"), "minimum": IntAttr(0), "multiple": IntAttr(1)},
        result_types=[IndexType()],
    )
    block.add_op(size)
    _buffer(block, "UB", 256, count=size.value, scale=256)
    with pytest.raises(CompileError, match="finite byte upper bound") as error:
        admit_memory_bounds(collect_native_memory(_module(block)))
    assert error.value.kind == ErrorKind.UNSUPPORTED


def test_runtime_capacity_cannot_be_replaced_with_the_static_prototype_size():
    block = Block()
    size = RuntimeSizeOp.build(
        attributes={"sym_name": StringAttr("n"), "minimum": IntAttr(1), "multiple": IntAttr(1)},
        result_types=[IndexType()],
    )
    block.add_op(size)
    gm = _buffer(block, "GM", 256, count=size.value, scale=128)
    ub = _buffer(block, "UB", 256)
    source, destination = _address(block, gm), _address(block, ub)
    _task(block, "copy", source, destination)
    with pytest.raises(CompileError, match="not bounded by its allocation") as error:
        admit_memory_bounds(collect_native_memory(_module(block)))
    assert error.value.kind == ErrorKind.UNSUPPORTED

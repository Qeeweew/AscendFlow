"""mask lifetimes contracts."""

from __future__ import annotations

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.fixtures.masks import (
    _data_predicate_program,
    _masked_fill,
    _ordered,
    _reuse_program,
    _task,
)
from tests.models.structured_events import check_structured_events


def test_partial_update_preserves_initialized_lanes_and_rejects_uninitialized():
    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    memory = collect_native_memory(result.plan)
    (masked,) = [t for t in memory.templates if t.spec.native == "mask_fill"]
    assert masked.spec.modes == ("RW",)
    packet = masked.spec.packet
    assert packet is not None and packet.closed_mask and packet.each_mask_call
    assert packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}
    asc = result.asc
    # Full chunks reuse one all-ones bitmap; no zero bitmap is ever issued.
    assert "asc_set_vector_mask(0, UINT64_MAX)" in asc
    assert "asc_set_vector_mask(0, 0)" not in asc
    assert "PIPE_ALL" not in asc
    uninitialized = N.from_source(
        """
        def kernel(Y:N.Tensor((16,256),N.f32,access="write"), q:N.i64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((16,256),N.f32)
                rows=N.arange(16)[:, None]
                cols=N.arange(256)[None, :]
                N.fill(work,-7.0,mask=(cols + q) > rows)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(uninitialized, timeout_s=120)


def test_data_compare_where_and_mask_logic_keep_actual_packed_reads():
    result = compile(_data_predicate_program(), timeout_s=180)
    memory = collect_native_memory(result.plan)
    natives = [template.spec.native for template in memory.templates]
    assert "compare_scalar_gt" in natives
    assert "compare_lt" in natives
    assert "mask_and" in natives
    assert "select_masked" in natives
    compare = next(t for t in memory.templates if t.spec.native == "compare_scalar_gt")
    select = next(t for t in memory.templates if t.spec.native == "mask_and")
    assert compare.spec.modes == ("R", "RW")
    assert select.spec.modes == ("R", "R", "W")
    packet = next(t for t in memory.templates if t.spec.native == "select_masked").spec.packet
    assert packet is not None and packet.closed_mask
    assert "CMPMASK" in packet.clobbers and packet.internal_fences == ("V",)
    where = next(t.spec for t in memory.templates if t.spec.native == "select_masked")
    assert where.modes == ("R", "R", "R", "W")
    # Packed predicate storage is byte-addressed, never retyped as f32.
    for template in memory.templates:
        if template.spec.native in {"compare_scalar_gt", "compare_lt"}:
            assert template.accesses[-1].address.geometry.dtype == "u8"
    original = collect_native_memory(parse(result.before_ir))
    execution = check_structured_events(memory, logical_reference=original)
    assert execution.bytes_read and execution.bytes_written
    assert execution.hazards
    ordered = _ordered(execution)
    compare_gt = _task(memory, "compare_scalar_gt")
    compare_lt = _task(memory, "compare_lt")
    combine = _task(memory, "mask_and")
    select = _task(memory, "select_masked")
    assert ordered(compare_gt, combine) and ordered(compare_lt, combine)
    assert ordered(combine, select)
    assert ordered(compare_gt, select)


def test_source_overwrite_does_not_change_an_earlier_compare():
    result = compile(_data_predicate_program(), timeout_s=180)
    memory = collect_native_memory(result.plan)
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    first = _task(memory, "compare_scalar_gt")
    overwrite = next(
        template.operation for template in memory.templates if template.spec.native == "muls"
    )
    assert ordered(first, overwrite)
    assert ordered(first, _task(memory, "mask_and"))


def test_predicate_storage_reuse_requires_the_old_mask_to_be_dead():
    with pytest.raises(CompileError, match="possible overwrite before its actual native reader"):
        compile(_reuse_program(consume_before_reuse=False), timeout_s=120)
    result = compile(_reuse_program(consume_before_reuse=True), timeout_s=180)
    memory = collect_native_memory(result.plan)
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    compares = [
        template.operation
        for template in memory.templates
        if template.spec.native.startswith("compare_scalar_")
    ]
    selects = [
        template.operation
        for template in memory.templates
        if template.spec.native == "select_masked_update"
    ]
    assert len(compares) == 2 and len(selects) == 2
    assert ordered(compares[0], selects[0])
    assert ordered(selects[0], compares[1])
    assert ordered(compares[1], selects[1])


def test_materialized_coordinate_predicate_combines_coherently():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                cstore=N.alloc_mask((8,128))
                mstore=N.alloc_mask((8,128))
                N.copy(X,work)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                coord=((cols % 5) == 0) & (cols > rows)
                cmask=N.materialize(coord,out=cstore)
                p=N.compare(work,0.0,kind="gt",out=bits)
                both=N.mask_and(p,cmask,out=mstore)
                N.fill(temp,2.0)
                N.fill(work,-2.0,mask=both,scratch=temp)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=180)
    memory = collect_native_memory(result.plan)
    natives = [template.spec.native for template in memory.templates]
    assert "mask_bits" in natives and "mask_and" in natives
    bits = next(t for t in memory.templates if t.spec.native == "mask_bits")
    assert bits.spec.engine.value == "S" and bits.spec.modes == ("W",)
    assert bits.spec.packet is not None and bits.spec.packet.closed_mask
    for template in memory.templates:
        for access in template.accesses:
            if access.address.geometry.dtype == "u8":
                assert access.address.geometry.nbytes % 32 == 0
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    assert ordered(_task(memory, "mask_bits"), _task(memory, "mask_and"))
    assert ordered(_task(memory, "compare_scalar_gt"), _task(memory, "mask_and"))


def test_mask_alternates_with_count_packets_without_incoming_state():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write"), q:N.i64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                N.copy(X,work)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.fill(work,-1.0,mask=(cols + q) > rows)
                N.exp(work,out=work)
                N.fill(work,2.0,mask=(cols < rows))
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=120)
    memory = collect_native_memory(result.plan)
    masked = [t for t in memory.templates if t.spec.native == "mask_fill"]
    exp = [t for t in memory.templates if t.spec.native == "exp"]
    assert len(masked) == 2 and len(exp) == 1
    for template in masked:
        assert template.spec.packet is not None
        assert template.spec.packet.mask_steps[0].value == "set-mode-normal"
        assert template.spec.packet.each_mask_call
    assert exp[0].spec.packet.each_count_call
    # Each packet configures its own state; no packet consumes an incoming mask.
    assert "PIPE_ALL" not in result.asc


def test_masked_scratch_must_not_alias_destination_allocation():
    alias = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.fill(work,-1.0,mask=p,scratch=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="must not alias"):
        compile(alias, timeout_s=60)
    aliasing_view = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.fill(work,-1.0,mask=p,scratch=work[0:8,0:128])
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="must not alias"):
        compile(aliasing_view, timeout_s=60)
